import os
import random

import cv2
import numpy as np
import skimage.draw

from calib3d import Point3D, Point2D, compute_shear_rectification_matrix
from deepsport_utilities.court import Court, BALL_DIAMETER
from deepsport_utilities.transforms import Transform, RandomCropperTransform
from deepsport_utilities.utils import gamma_correction, setdefaultattr
from deepsport_utilities.ds.instants_dataset import View, ViewKey, Ball, BallState

class AddBallAnnotation(Transform):
    def __call__(self, key, view):
        balls = [a for a in view.annotations if a.type == 'ball']
        assert len(balls) == 1, f"Expected one ball. received {len(balls)}."
        view.ball = balls[0]
        return view

class UndistortTransform(Transform):
    def __call__(self, key, view):
        all_images = []
        for image in view.all_images:
            all_images.append(cv2.undistort(image, view.calib.K, view.calib.kc))
        view.calib = view.calib.update(kc=np.array([0,0,0,0,0]))
        return view

class ComputeDiff(Transform):
    def __init__(self, squash=False, inplace=False):
        self.squash = squash
        self.inplace = inplace

    def __call__(self, view_key: ViewKey, view: View):
        diff = np.abs(view.image.astype(np.int32) - view.all_images[1].astype(np.int32)).astype(np.uint8)
        if self.squash:
            diff = np.mean(diff, axis=2).astype(np.uint8)
        if self.inplace:
            view.image = np.dstack((view.image, diff))
        else:
            view.diff = diff
        return view

class GameGammaColorTransform(Transform):
    def __init__(self, transform_dict):
        assert all([isinstance(k, int) for k in transform_dict.keys()])
        #29582 : [1.04, 1.02, 0.93],
        #24651 : [1.05, 1.02, 0.92],
        #30046 : [1.01, 1.01, 1.01]
        self.transform_dict = transform_dict

    def __call__(self, view_key, view):
        if view_key.instant_key.game_id in self.transform_dict.keys():
            gammas = np.array(self.transform_dict[view_key.instant_key.game_id])
            view.image = gamma_correction(view.image, gammas)
        return view

class BayeringTransform(Transform):
    def __init__(self):
        self.R_filter = np.array([[1,0],[0,0]])
        self.G_filter = np.array([[0,1],[1,0]])
        self.B_filter = np.array([[0,0],[0,1]])
    def __call__(self, view_key, view):
        height, width, _ = view.image.shape
        R_mask = np.tile(self.R_filter, [height//2, width//2])
        G_mask = np.tile(self.G_filter, [height//2, width//2])
        B_mask = np.tile(self.B_filter, [height//2, width//2])
        mask = np.stack((R_mask, G_mask, B_mask), axis=2)
        mask = mask[np.newaxis]
        for i, image in enumerate(view.all_images):
            view.all_images[i] = np.sum(image*mask, axis=3)
        view.image = view.all_images[0]

class GameRGBColorTransform(Transform):
    def __init__(self, transform_dict):
        assert all([isinstance(k, int) for k in transform_dict.keys()])
        self.transform_dict = transform_dict

    def __call__(self, view_key: ViewKey, view: View):
        if view_key.instant_key.game_id in self.transform_dict.keys():
            adaptation_vector = np.array(self.transform_dict[view_key.instant_key.game_id])
            view.image = np.clip(view.image.astype(np.float32)*adaptation_vector, 0, 255).astype(np.uint8)
        return view

class ViewRandomCropperTransform(RandomCropperTransform):
    def _apply_transformation(self, view, A):
        if self.debug:
            w, h = self.output_shape
            points = Point2D(np.linalg.inv(A)@Point2D([0,0,w,w],[0,h,h,0]).H)
            cv2.polylines(view.image, [points.T.astype(np.int32)], True, (255,0,0), self.linewidth)
        else:
            view.image = cv2.warpAffine(view.image, A[0:2,:], self.output_shape, flags=cv2.INTER_LINEAR)
            view.calib = view.calib.update(K=A@view.calib.K, width=self.output_shape[0], height=self.output_shape[1])

            if hasattr(view, "all_images"):
                for i in range(1, len(view.all_images)): # skip first image as it was already done
                    view.all_images[i] = cv2.warpAffine(view.all_images[i], A[0:2,:], self.output_shape, flags=cv2.INTER_LINEAR)
            if hasattr(view, "human_masks") and view.human_masks is not None:
                view.human_masks = cv2.warpAffine(view.human_masks, A[0:2,:], self.output_shape, flags=cv2.INTER_NEAREST)
        return view

class NaiveViewRandomCropperTransform(ViewRandomCropperTransform):
    def __init__(self, *args, scale_min=0.5, scale_max=2, **kwargs):
        """ scale is the scale factor that will be applied to images
        """
        super().__init__(*args, size_min=scale_min, size_max=scale_max, **kwargs)

    def _get_current_parameters(self, view_key, view):
        input_shape = view.calib.width, view.calib.height
        return None, 1, input_shape


class BallViewRandomCropperTransform(ViewRandomCropperTransform):
    """ Create random crops with annotated balls visible in them.
        If [min_size, max_size] range is given, scale factor is chosen s.t. ball
        size is in the [min_size, max_size] range.
        If [def_min, def_max] range is given, scale factor is chosen s.t. image
        definition [px/m] is in the [def_min, def_max] range at ball location.
        If [scale_min, scale_max] range is given, scale factor is chosen in
        that range.
        An error is raised if multiple options are given.

        Arguments:
            on_ball: if True, ball is always visible on the random crop.
                If False, ball is visible in the random crop half of the time.
                Else, on_ball is the probability that the ball is kept visible.
            def_min, def_max: min and max definition [px/m] of the random crop.
            size_min, size_max: min and max size [px] of the ball in the random crop.
            scale_min, scale_max: image's min and max scale factor
    """
    def __init__(self, *args, on_ball=True,
                 def_min=None, def_max=None,
                 size_min=None, size_max=None,
                 scale_min=None, scale_max=None,
                 rectify=True, **kwargs):
        msg = "Only one of ('size_min' and 'size_max') or ('def_min' and 'def_max') or ('scale_min' and 'scale_max') should be defined"
        if size_min is not None and size_max is not None:
            assert all([x is None for x in [def_min, def_max, scale_min, scale_max]]), msg
            super().__init__(*args, size_min=size_min, size_max=size_max, **kwargs)
            self.true_size = BALL_DIAMETER
        elif def_min is not None and def_max is not None:
            assert all([x is None for x in [size_min, size_max, scale_min, scale_max]]), msg
            super().__init__(*args, size_min=def_min, size_max=def_max, **kwargs)
            self.true_size = 100
        elif scale_min is not None and scale_max is not None:
            assert all([x is None for x in [size_min, size_max, def_min, def_max]]), msg
            super().__init__(*args, size_min=scale_min, size_max=scale_max, **kwargs)
            self.true_size = None
        else:
            raise ValueError(msg)

        self.rectify = rectify
        self.on_ball = {False: 0.5, True: 1.0, None: 0.0}.get(on_ball, on_ball)
        if self.debug and self.on_ball != 1.0:
            raise NotImplementedError("Random keypoint should be drawn in view.image (as well as it's projection on court for better visualization)")

    def random_ball_position(self, view):
        court = setdefaultattr(view, "court", Court(getattr(view, "rule_type", "FIBA")))
        #court_polygon = view.calib.get_region_visible_corners_2d(court.corners, 1)
        top_edge = list(court.visible_edges(view.calib))[0]
        start = top_edge[0][0][0]
        stop = top_edge[1][0][0]
        x = np.random.beta(2, 2)*(stop-start)+start
        y = np.random.beta(2, 2)*court.h/2+court.h/4
        z = 0
        return Point3D(x,y,z)

    def _apply_transformation(self, view, A):
        if not self.rectify:
            return super()._apply_transformation(view, A)

        # rectify eccentricity caused by projection on a plane
        X = compute_shear_rectification_matrix(view.calib, view.__shear_center)
        return super()._apply_transformation(view, A@X)

    def _get_current_parameters(self, view_key, view):
        ball = getattr(view, "ball", None)

        # If not `on_ball` use the ball anyway half of the samples
        if random.random() < self.on_ball and ball is not None:
            keypoint = ball.center
            view.ball = ball
        else:
            keypoint = self.random_ball_position(view)
            view.ball = Ball({
                'center': keypoint,
                'origin': "random",
                'image': view_key.camera,
                'visible': False,
            })

        # Use ball if any, else use the random ball (it only affects the strategy to scale)
        if self.true_size is None:
            size = 1
        else:
            point3D = ball.center if ball is not None else keypoint
            size = float(view.calib.compute_length2D(point3D, self.true_size)[0])

        keypoint = view.calib.project_3D_to_2D(keypoint)
        input_shape = view.calib.width, view.calib.height
        view.__shear_center = keypoint
        return keypoint, size, input_shape


class CleverViewRandomCropperTransform(BallViewRandomCropperTransform):
    def __init__(self, def_min=60, def_max=160, **kwargs):
        super().__init__(on_ball=0, def_min=def_min, def_max=def_max, **kwargs)



class PlayerViewRandomCropperTransform(ViewRandomCropperTransform):
    def __init__(self, output_shape, def_min=60, def_max=160, margin=100, **kwargs):
        """
            def -  definition in pixels per meters. 60px/m = ball of 14px
            margin - a margin in cm the keypoints
        """
        super().__init__(output_shape=output_shape, size_min=def_min, size_max=def_max, **kwargs)
        self.margin = margin

    def focus_on_player(self, view_key, view):
        players = [a for a in view.annotations if a.type == "player" and a.camera == view_key.camera]
        if not players:
            return None
        player = random.sample(players, 1)[0]
        keypoints = Point3D([player.head, player.hips, player.foot1, player.foot2])
        return keypoints

    def _get_current_parameters(self, view_key, view):
        raise NotImplementedError("This code was not tested. Images should be visualized.")
        keypoints = self.focus_on_player(view_key, view)
        if keypoints is None:
            return None
        margin = float(view.calib.compute_length2D(Point3D(np.mean(keypoints, axis=1)), self.margin))
        size = float(view.calib.compute_length2D(Point3D(np.mean(keypoints, axis=1)), 100))
        keypoints = view.calib.project_3D_to_2D(keypoints)
        input_shape = view.calib.width, view.calib.height
        return keypoints, size, input_shape

class AddBallSizeFactory(Transform):
    def __init__(self, origins=['annotation', 'interpolation']):
        self.origins = origins
    def __call__(self, view_key, view):
        ball = view.ball
        predicate = lambda ball:    ball.origin in self.origins         \
                                and ball.visible is not False           \
                                and view.calib.projects_in(ball.center) \
                                and ball.center.z < -10
        return {"ball_size": view.calib.compute_length2D(ball.center, BALL_DIAMETER)[0] if predicate(ball) else np.nan}

class AddBallHeightFactory(Transform):
    """ Ball hidden are considered
    """
    def __init__(self, origins=['annotation', 'interpolation']):
        self.origins = origins
    def __call__(self, view_key, view):
        ball = view.ball
        predicate = lambda ball:    ball.origin in self.origins         \
                                and view.calib.projects_in(ball.center) \
                                and ball.center.z < -10
        center2D = view.calib.project_3D_to_2D(ball.center)
        ground2D = view.calib.project_3D_to_2D(Point3D(ball.center.x, ball.center.y, 0))
        return {"ball_height": np.linalg.norm(center2D - ground2D) if predicate(ball) else np.nan}

class AddBallPresenceFactory(Transform):
    def __call__(self, view_key: ViewKey, view: View):
        return {'ball_presence': 1 if view.ball.origin in ['annotation', 'interpolation'] else 0}

class AddBallStateFactory(Transform):
    def __init__(self, state_mapping=None):
        self.state_mapping = state_mapping or {state: np.eye(len(BallState))[s] for s, state in enumerate(BallState)}
        self.nan = np.ones_like(list(self.state_mapping.values())[0])*np.nan
    def __call__(self, view_key, view):
        predicate = lambda ball: view.calib.projects_in(ball.center) and ball.visible
        state = self.state_mapping.get(view.ball.state if predicate(view.ball) else BallState.NONE, self.nan)
        return {"ball_state": state}

class AddBallPositionFactory(Transform):
    def __call__(self, view_key, view):
        return {"ball_position": view.calib.project_3D_to_2D(view.ball.center)}

class AddBallFactory(Transform):
    def __call__(self, view_key, view):
        return {"ball": view.ball}

class AddBallVisibilityFactory(Transform):
    def __call__(self, view_key, view):
        return {"ball_visible": view.ball.visible}

class AddDiffFactory(Transform):
    def __call__(self, view_key, view):
        raise NotImplementedError() # code needs to be re-implemented: current implementation only adds next image
        return {"input_image2": view.all_images[1]}

class AddNextImageFactory(Transform):
    def __call__(self, view_key, view):
        return {"input_image2": view.all_images[1]}

class AddCalibFactory(Transform):
    def __init__(self, as_dict=False):
        self.as_dict = as_dict
    @staticmethod
    def to_basic_dict(calib):
        return {
            "K": calib.K,
            "r": cv2.Rodrigues(calib.R)[0].flatten(),
            "T": calib.T,
            "width": np.array([calib.width]),
            "height": np.array([calib.height]),
            "kc": np.array(calib.kc),
        }
    def __call__(self, view_key, view):
        if self.as_dict:
            return self.to_basic_dict(view.calib)
        return {"calib": view.calib}

class AddCourtFactory(Transform):
    def __call__(self, view_key, view):
        if not getattr(view, "court", None):
            view.court = Court()
        return {
            "court_width": np.array([view.court.w]),
            "court_height": np.array([view.court.h])
        }

class AddImageFactory(Transform):
    def __call__(self, view_key, view):
        return {"input_image": view.image}

class AddHumansSegmentationTargetViewFactory(Transform):
    def __call__(self, view_key, view):
        return {"human_masks": view.human_masks}

class AddBallSegmentationTargetViewFactory(Transform):
    def __call__(self, view_key, view):
        calib = view.calib
        target = np.zeros((calib.height, calib.width), dtype=np.uint8)
        for ball in [a for a in view.annotations if a.type == "ball" and calib.projects_in(a.center) and a.visible]:
            diameter = calib.compute_length2D(ball.center, BALL_DIAMETER)
            center = calib.project_3D_to_2D(ball.center)
            #cv2.circle(target, center.to_int_tuple(), radius=int(diameter/2), color=1, thickness=-1)
            target[skimage.draw.disk(center.flatten()[::-1], radius=float(diameter/2), shape=target.shape)] = 1
        return {
            "target": target
        }

try:
    from calib3d.pycuda import CudaCalib
    import pycuda.driver as cuda
    # pylint: disable=unused-import
    import pycuda.autoinit
    # pylint: enable=unused-import
    from pycuda.compiler import SourceModule

    class AddBallDistance(Transform):
        def __init__(self):
            self._calib_struct_ptr = cuda.mem_alloc(CudaCalib.memsize())
            self._ball_ptr = cuda.mem_alloc(3*8)
            cuda_code = open(os.path.join(os.path.dirname(__file__), "mod_source.c"), "r").read()
            mod = SourceModule(str(CudaCalib.struct_str())+cuda_code)
            self._ball_distance = mod.get_function("BallDistance")
            self._bdim = (32,32,1)

        def __repr__(self):
            return "{}()".format(self.__class__.__name__)

        def __call__(self, key, view: View):
            # copy calib to GPU
            calib = CudaCalib.from_calib(view.calib)
            calib.memset(self._calib_struct_ptr, cuda.memcpy_htod)

            # copy ball position to GPU
            cuda.memcpy_htod(self._ball_ptr, memoryview(view.ball.center))

            # create distmap on GPU
            distmap_gpu = cuda.mem_alloc(calib.img_width * calib.img_height * 8)# 8 bytes per double
            cuda.memset_d8(distmap_gpu, 0, calib.img_width * calib.img_height * 8)

            # compute best block and grid dimensions
            dx, mx = divmod(calib.img_width, self._bdim[0])
            dy, my = divmod(calib.img_height, self._bdim[1])
            gdim = ( (dx + (mx>0)) * self._bdim[0], (dy + (my>0)) * self._bdim[1])

            # call gpu function
            self._ball_distance(distmap_gpu, self._calib_struct_ptr, self._ball_ptr, block=self._bdim, grid=gdim)

            # copy result to memory
            view.ball_distance = np.zeros((calib.img_height,calib.img_width))#, np.int8)
            cuda.memcpy_dtoh(view.ball_distance, distmap_gpu)
            # cuda.Context.synchronize()
            return view
except ModuleNotFoundError as e:
    if e.name == "calib3d.pycuda":
        raise e

except ImportError as e:
    if "CudaCalib" not in str(e.msg):
        raise e
