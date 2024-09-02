from enum import IntFlag, IntEnum
from mlworkflow import lazyproperty as cached_property
import json
import os
from typing import NamedTuple

import cv2
import imageio
import numpy as np

from mlworkflow import Dataset
from deepsport_utilities.calib import Calib, Point3D
from deepsport_utilities.court import Court, court_dim as COURT_DIM
from deepsport_utilities.ds.generic_dataset import GenericItem

class DownloadFlags(IntFlag):
    NONE = 0
    WITH_IMAGE = 1
    WITH_CALIB_FILE = 2
    #WITH_FOREGROUND_MASK_FILE = 4 # obsolete
    WITH_HUMAN_SEGMENTATION_MASKS = 8
    WITH_FOLLOWING_IMAGE = 16
    WITH_ALL_IMAGES = 32
    ALL = -1

class InstantKey(NamedTuple):
    arena_label: str
    game_id: int
    timestamp: int

class Instant(GenericItem):
    def __init__(self, db_item, dataset_folder, download_flags):
        self.dataset_folder = dataset_folder
        self.download_flags = download_flags

        self.arena_label = db_item["arena_label"]
        self.num_cameras = db_item["num_cameras"]

        self.game_id = db_item["game_id"]
        self.league_id = db_item["league_id"]
        self.rule_type = db_item["rule_type"]
        self.sport = db_item["sport"]

        self.timestamp = db_item["timestamp"]
        self.offsets = db_item["offsets"]

        self.annotation_state = db_item.get("annotation_state", None)
        self.annotator_id = db_item.get("annotator_id", None)
        self.annotation_ts = db_item.get("annotation_ts", None)
        self.annotation_duration = db_item.get("annotation_duration", None)
        self.annotation_game_state = db_item.get("annotation_game_state", "standard_game")
        self.annotated_human_masks = db_item.get("annotated_human_masks", False)

        self.format =  db_item["format"]

        annotation_map = {
            "player": Player,
            "ball": Ball
        }
        self.annotations = [annotation_map[a['type']](a)for a in (db_item.get('annotations', []) or [])]

        self.image_source = db_item.get("image_source", "raw")
        self.court_dim = COURT_DIM[self.rule_type]
        self.court = Court(self.rule_type)
        self.timestamps = [self.timestamp for _ in range(self.num_cameras)]

    def __str__(self):
        return "({}[{:5d}]@{})".format(self.arena_label, self.game_id, self.timestamp)

    def get_filekey(self, prefix, suffix):
        return os.path.join(self.arena_label, str(self.game_id), "{}{}{}".format(prefix, self.timestamp, suffix))

    @cached_property
    def calibs(self):
        return [self.__load_calib(c) for c in range(self.num_cameras)]

    @cached_property
    def all_images(self):
        _ = self.calibs # triggers calib loading
        all_images = {}
        for c in range(self.num_cameras):
            for idx, offset in enumerate(self.offsets):
                if     (idx == 0) \
                    or (idx == 1 and self.download_flags & DownloadFlags.WITH_FOLLOWING_IMAGE) \
                    or (self.download_flags & DownloadFlags.WITH_ALL_IMAGES):
                    try:
                        all_images[(c,offset)] = self.__load_image(c, offset)
                    except BaseException as e:
                        raise ValueError((self.offsets, self.key)) from e
        return all_images

    @property
    def images(self):
        return [img for (c, offset), img in self.all_images.items() if offset == 0]

    @property
    def ball(self):
        balls = [a for a in self.annotations if a.type == 'ball']
        if len(balls) > 1:
            raise ValueError("Too many annotated balls")
        elif len(balls) == 1:
            return balls[0]
        return None

    @cached_property
    def human_masks(self):
        assert self.download_flags & DownloadFlags.WITH_HUMAN_SEGMENTATION_MASKS, \
            "Provided flag doesn't contain 'human_masks'. Recreate your dataset with appropriate DownloadFlags"
        try:
            filenames = [os.path.join(self.dataset_folder, self.get_filekey("camcourt{}_".format(cam_idx+1), "_humans.png")) for cam_idx in range(self.num_cameras)]
            return [imageio.imread(filename) for filename in filenames] # imageio handles 16bits images while cv2 doesn't
        except FileNotFoundError:
            # If one human_masks file is missing for one camera, no human_masks will be available.
            # If file is missing because no human appears on that camera, you should upload an empty image to the bucket.
            return []

    def __load_image(self, cam_idx, offset=0):
        filename = os.path.join(self.dataset_folder, self.get_filekey("camcourt{}_".format(cam_idx+1), "_{}.png".format(offset)))
        image = cv2.imread(filename)
        if image is None:
            raise FileNotFoundError(filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __load_calib(self, cam_idx):
        assert self.download_flags & DownloadFlags.WITH_CALIB_FILE, \
            "Provided flag doesn't contain calib files. Recreate your dataset with appropriate DownloadFlags"
        filename = os.path.join(self.dataset_folder, self.get_filekey("camcourt{}_".format(cam_idx+1), ".json"))
        return parse_DeepSport_calib(json.load(open(filename, 'r'))['calibration'])

    def draw(self, i=None, draw_players=True, draw_ball=True, draw_lines=False):
        if i is None:
            w, h = self.court_dim
            image = np.ones((int(h), int(w), 3), np.uint8)*255
            R = np.identity(3)
            C = Point3D(w/2, h/2, -3000)
            m = w/0.01  # pixel_size (pixels/meters) on 1 centimeter sensor
            f = 0.009  # focal (meters)
            R = np.identity(3)
            calib = Calib(width=w, height=h, T=-R@C[:3], R=R, K=np.array([[f*m,  0, w/2], [0, f*m, h/2], [0,  0,  1]]))
        else:
            image = self.images[i].copy()
            calib = self.calibs[i]

        if draw_lines:
            self.court.draw_lines(image, calib)
        # if fg_detections is not None:
        #     calib = self.calibs[i] if i is not None else None
        #     detections = self.fg_detections[i][fg_detections] if i is not None else \
        #                  itertools.chain(*self.fg_detections)
        #     for det in detections:
        #         v = calib.project_3D_to_2D(det.feet).to_int_tuple()
        #         cv2.circle(image, v[0:2].flatten(), 7, [255, 0, 255], -1)

        for annotation in self.annotations:
            if annotation.type == "player" and draw_players:
                head = calib.project_3D_to_2D(annotation.head).to_int_tuple()
                hips = calib.project_3D_to_2D(annotation.hips).to_int_tuple()
                foot1 = calib.project_3D_to_2D(annotation.foot1).to_int_tuple()
                foot2 = calib.project_3D_to_2D(annotation.foot2).to_int_tuple()

                if any([kp[0] < 0 or kp[1] > image.shape[1] or kp[1] < 0 or kp[1] > image.shape[0] for kp in [head, hips]]):
                    continue

                # head tip
                length = 70 # cm
                headT3D = length * Point3D(np.cos(annotation.headAngle), np.sin(annotation.headAngle), 0)
                headT = calib.project_3D_to_2D(annotation.head+headT3D).to_int_tuple()

                color = [0, 0, 0]
                color[annotation.team-1] = 255

                if i is not None:
                    cv2.line(image, hips, foot1, color, 3)
                    cv2.line(image, hips, foot2, color, 3)
                    cv2.line(image, head, hips, color, 3)
                else:
                    cv2.circle(image, head, 5, color, -1)
                cv2.line(image, head, headT, color, 3)

            elif annotation.type == "ball" and draw_ball:
                center = tuple(int(x) for x in calib.project_3D_to_2D(annotation.center).to_list())
                color = [255, 255, 0]
                cv2.circle(image, center, 5, color, -1)
        for detection in getattr(self, "detections", []):
            if detection.type == "ball" and draw_ball:
                center = tuple(int(x) for x in calib.project_3D_to_2D(detection.center).to_list())
                color = [0, 255, 255]
                cv2.circle(image, center, 5, color, -1)
        return image

    @property
    def files(self):
        for i in range(0, int(self.num_cameras)):
            if self.download_flags & DownloadFlags.WITH_IMAGE:
                for idx, offset in enumerate(self.offsets):
                    if     (self.download_flags & DownloadFlags.WITH_ALL_IMAGES) \
                        or (self.download_flags & DownloadFlags.WITH_FOLLOWING_IMAGE and idx == 1) \
                        or (idx == 0):
                        yield self.get_filekey("camcourt{}_".format(i+1), "_{}.png".format(offset))
            if self.download_flags & DownloadFlags.WITH_CALIB_FILE:
                yield self.get_filekey("camcourt{}_".format(i+1), ".json")
            if self.download_flags & DownloadFlags.WITH_HUMAN_SEGMENTATION_MASKS:
                yield self.get_filekey("camcourt{}_".format(i+1), "_humans.png")

    @property
    def key(self):
        return InstantKey(self.arena_label, self.game_id, self.timestamp)

    @property
    def db_item(self):
        db_item = {
            "format": self.format,
            "image_source": self.image_source,

            # arena relative infos
            "arena_label": self.arena_label,
            "num_cameras": self.num_cameras,

            # game relative infos
            "sport": self.sport,
            "game_id": self.game_id,
            "league_id": self.league_id,
            "rule_type": self.rule_type,

            # instant relative infos
            "timestamp": self.timestamp,
            "offsets": self.offsets,

            "annotation_state": self.annotation_state,
            "annotations": [a.to_dict() for a in self.annotations],
            "annotated_human_masks": self.annotated_human_masks
        }

        for attr in ["annotation_ts", "annotator_id", "annotation_duration", "annotation_game_state"]:
            if value := getattr(self, attr, None):
                db_item[attr] = value
        return db_item

    def to_dict(self):
        return {"db_item": self.db_item, "download_flags": self.download_flags, "dataset_folder": self.dataset_folder}

def parse_DeepSport_calib(data):
    return Calib(
        width = data["img_width"],
        height = data["img_height"],
        T = np.array([data["T"]]).T,
        K = np.array(data["KK"]).reshape((3, 3)),
        kc = np.array(data["kc"]),
        R = np.array(data["R"]).reshape((3, 3))
    )

class InstantsDataset(Dataset):
    items_type = Instant


class BallState(IntEnum):
    NONE = 0
    FLYING = 1
    CONSTRAINT = 2
    DRIBBLING = 3

class Ball():
    state = BallState.NONE # default
    visible = None         # default
    value = None           # default
    def __init__(self, data):
        self.type = "ball"
        self.center = Point3D(*data['center'])
        self.origin = data.get('origin', "annotation")
        self.camera = data['image']
        self.visible = data.get('visible', None)
        self.state = BallState(data.get('state', 0))
        self.value = data.get('value', None)

    def to_dict(self):
        return {
            "type": self.type,
            "origin": self.origin,
            "center": self.center.to_list(),
            "image": self.camera,
            "visible": self.visible,
            "state": int(self.state),
            "value": self.value
        }

    def __repr__(self):
        return "Ball(" + ",".join([
            f"origin='{self.origin}', " \
            f"center=({self.center.x:.01f}, {self.center.y:.01f}, {self.center.z:.01f})" \
        ]) + ")"

class BallAnnotation(Ball):
    origin = "annotation"
    pass # retro-compatibility


class Player():
    def __init__(self, data):
        self.type = "player"
        self.origin = data.get('origin', "annotation")
        self.team = data['team']
        self.head = Point3D(*data['head'])
        self.hips = Point3D(*data['hips'])
        self.foot1 = Point3D(*data['foot1'])
        self.foot2 = Point3D(*data['foot2'])
        self.foot1_at_the_ground = str(data["foot1_at_the_ground"]).lower() == "true"
        self.foot2_at_the_ground = str(data["foot2_at_the_ground"]).lower() == "true"
        self.headAngle = data['headOrientation']
        self.camera = data['image']
        self.hipsAngle = data.get('hipsOrientation', self.headAngle)
        self.feet = (self.foot1 + self.foot2) / 2

    def to_dict(self):
        return {
            "type": self.type,
            "origin": self.origin,
            "team": self.team,
            "head": self.head.to_list(),
            "headOrientation": self.headAngle,
            "hips": self.hips.to_list(),
            "foot1": self.foot1.to_list(),
            "foot2": self.foot2.to_list(),
            "foot1_at_the_ground": self.foot1_at_the_ground,
            "foot2_at_the_ground": self.foot2_at_the_ground,
            "image": self.camera
        }

class ForegroundDetection():
    def __init__(self, detection, camera: int) -> None:
        self.origin = "foreground"
        self.feet = Point3D(*[detection["pos_feet"][0], detection["pos_feet"][1], 0])
        self.confidence = detection["level"]
        self.status = detection["status"]
        self.camera = camera
