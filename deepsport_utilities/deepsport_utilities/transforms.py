import abc
import copy
import random

import numpy as np
from calib3d.calib import parameters_to_affine_transform

from deepsport_utilities.utils import jpegBlur

class Transform(metaclass=abc.ABCMeta):
    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

    def __gt__(self, other):
        return self.__repr__() > other.__repr__()

    @abc.abstractmethod
    def __call__(self, key, item):
        pass

    def __repr__(self):
        config = getattr(self, "config", {k:v for k, v in self.__dict__.items() if not k.startswith("_")})
        attributes = ",".join("{}={}".format(k, v) for k,v in config.items())
        return "{}({})".format(self.__class__.__name__, attributes)


class DoNothing(Transform):
    def __call__(self, key, item):
        return item


class DeclutterItems(Transform):
    """ Drops attributes from dataset items. Attributes to drop are given by the
        'drop' argument.
    """
    def __init__(self, drop):
        self.drop = drop
    def __call__(self, key, item):
        for name in self.drop:
            delattr(item, name)
        return item


class JPEGCompressionTransform(Transform):
    def __init__(self, key_names, q_range=(30,60)):
        self.key_names = key_names
        self.q_range = q_range
        assert len(q_range) == 2 and q_range[0] < q_range[1] and q_range[0] > 0 and q_range[1] <= 100

    def __call__(self, key, data):
        if data is None:
            return None
        q = random.randint(*self.q_range)
        for k in data:
            if k in self.key_names:
                data[k] = jpegBlur(data[k], q)
        return data


class IncompatibleCropException(ValueError):
    pass

class RandomCropperTransform(Transform, metaclass=abc.ABCMeta):
    linewidth = 4
    random_size = np.random.uniform
    def __init__(self, output_shape, size_min, size_max, max_angle=0, do_flip=False, padding=0, margin=0, debug=False, regenerate=False):
        """ Randomly scale, crop and rotate dataset items. The scale factor is
            randomly selected to keep the given keypoints of interest between
            `size_min` and `size_max` (At each call, the current keypoint size
            is returned by `_get_current_parameters`).

            Arguments:
                output_shape: Tuple(int, int) final shape of image-like data.
                size_min: (int) lower bound of keypoints random size. If `0`
                    `size_min` and `size_max` are ignored and no random scaling
                    is applied.
                size_max: (int) upper bound of keypoints random size. If `0`
                    `size_min` and `size_max` are ignored and no random scaling
                    is applied.
                max_angle: (int) positive and negative bounds for random
                    rotation (in degrees)
                do_flip: (bool) tells if random flip should be applied
                padding: (int) amount of padding (in pixels) in input image
                margin: (int) minimum margin (in pixels) between keypoints and
                    output image border
                debug: (bool) if `True`, doesn't actually crop but display
                    debug information on image instead.
                regenerate: (bool) if `True`, items are (deep)-copied before
                    calling `_apply_transformation`. Else, transformation can
                    occur in-place.
        """
        self.output_shape = output_shape
        self.size_min = size_min
        self.size_max = size_max
        self.max_angle = max_angle
        self.do_flip = do_flip
        assert not self.do_flip, "There seem to be a bug in the flip"
        self.padding = padding
        self.margin = margin
        self.debug = debug
        self.regenerate = regenerate

    def compute(self, input_shape, keypoints, actual_size, size_min=None, size_max=None):
        if size_min is not None and self.size_min*self.size_max == 0:
            # updated size range after failing to create a crop at given scale
            raise IncompatibleCropException("Impossible to crop image without changing the scale")

        size_min = size_min or self.size_min
        size_max = size_max or self.size_max

        if size_min > size_max:
            raise IncompatibleCropException("Impossible to crop image with object in the given size range")
        target_size = self.random_size(size_min, size_max) if size_min and size_max else actual_size
        ratio = target_size/actual_size
        tmp_width, tmp_height = [int(x/ratio) for x in self.output_shape]
        if tmp_width == 0 or tmp_height == 0:
            raise IncompatibleCropException("Impossible to crop image with object in the given size range")
        input_width, input_height = input_shape

        # If target size makes the output image bigger than input image, try with a lower size
        if tmp_width >= input_width + 2*self.padding or tmp_height >= input_height + 2*self.padding:
            if not size_min or not size_max:
                raise IncompatibleCropException("Impossible to crop image with object in the given size range")
            return self.compute(input_shape, keypoints, actual_size, size_min=target_size*1.1, size_max=size_max)

        margin = self.margin / ratio # margin expressed in pixels in the output image
        if keypoints is not None:
            # Restrict keypoints in input image limits
            max_keypoints_x = min(int(np.max(keypoints.x))+margin, input_width)
            min_keypoints_x = max(0, int(np.min(keypoints.x))-margin)
            max_keypoints_y = min(int(np.max(keypoints.y))+margin, input_height)
            min_keypoints_y = max(0, int(np.min(keypoints.y))-margin)

            # Compute offsets to fit input image
            x_offset_min = max(-self.padding, max_keypoints_x - tmp_width)
            x_offset_max = min(min_keypoints_x, input_width + self.padding)
            y_offset_min = max(-self.padding, max_keypoints_y - tmp_height)
            y_offset_max = min(min_keypoints_y, input_height + self.padding)
        else:
            x_offset_min = -self.padding
            x_offset_max = input_width - tmp_width + self.padding
            y_offset_min = -self.padding
            y_offset_max = input_height - tmp_height + self.padding

        x_offset_max = int(np.ceil(x_offset_max))
        x_offset_min = int(np.floor(x_offset_min))
        y_offset_max = int(np.ceil(y_offset_max))
        y_offset_min = int(np.floor(y_offset_min))

        # If target size makes it incompatible with input image shape and keypoints positions, try with a higher size
        if x_offset_max < x_offset_min or y_offset_max < y_offset_min:
            return self.compute(input_shape, keypoints, actual_size, size_min=size_min, size_max=target_size*0.9)

        x_offset = np.random.randint(x_offset_min, x_offset_max) if x_offset_min != x_offset_max else x_offset_max
        y_offset = np.random.randint(y_offset_min, y_offset_max) if y_offset_min != y_offset_max else y_offset_max

        x_slice = slice(x_offset, x_offset+tmp_width, None)
        y_slice = slice(y_offset, y_offset+tmp_height, None)

        angle = self.max_angle*(2*np.random.beta(2, 2)-1)

        return angle, x_slice, y_slice

    @abc.abstractmethod
    def _get_current_parameters(self, key, item):
        raise NotImplementedError(
            "This method should return (keypoints, actual_size, shape) corresponding to the " \
            "current keypoints and size of object of interest in the image, as well as the " \
            "current image shape (width, height)")

    @abc.abstractmethod
    def _apply_transformation(self, item, A):
        raise NotImplementedError(
            "This method should return the final transformed item, based on the original" \
            "item and the affine transformation matrix")

    def __call__(self, key, item):
        if item is None:
            return None
        parameters = self._get_current_parameters(key, item)
        if parameters is None:
            return None
        keypoints, actual_size, input_shape = parameters
        try:
            angle, x_slice, y_slice = self.compute(input_shape, keypoints, actual_size)
            flip = self.do_flip and bool(np.random.randint(0,2))
        except IncompatibleCropException:
            return None

        A = parameters_to_affine_transform(angle, x_slice, y_slice, self.output_shape, flip)
        if self.regenerate:
            item = copy.deepcopy(item)
        return self._apply_transformation(item, A)


class DataExtractorTransform(Transform):
    def __init__(self, *factories):
        self.factories = list(factories)
    def __call__(self, key, item):
        if not item:
            return None
        data = {}
        for factory in self.factories:
            if factory is None:
                continue
            try:
                data.update(**factory(key, item))
            except BaseException as e:
                print(factory, "failed", e)
                raise
        return data
