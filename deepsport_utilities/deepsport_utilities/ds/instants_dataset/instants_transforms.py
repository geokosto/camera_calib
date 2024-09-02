from dataclasses import dataclass
import numpy as np
from skimage import exposure

from deepsport_utilities.transforms import Transform
from deepsport_utilities.utils import gamma_correction
from . import InstantKey, Instant, DownloadFlags


class EnhanceColorTransform(Transform):
    gamma_corrections = {
        "KS-FR-BLOIS": [0.98, 0.98, 1.01],
    }
    intensity_corrections = {
        "KS-FR-BLOIS": (-20, 240),
    }
    def __call__(self, instant_key: InstantKey, instant: Instant):
        # Gamma correction
        if instant_key.arena_label in self.gamma_corrections:
            gammas = np.array(self.gamma_corrections[instant_key.arena_label])
            for k, image in instant.all_images.items():
                instant.all_images[k] = gamma_correction(image, gammas)

        # Intensity correction
        if instant_key.arena_label in self.intensity_corrections:
            bounds = self.intensity_corrections[instant_key.arena_label]
            for k, image in instant.all_images.items():
                instant.all_images[k] = exposure.rescale_intensity(image, bounds)

        return instant

class GammaCorrectionTransform(Transform):
    def __init__(self, transform_dict=None):
        self.transform_dict = {
            29582 : [1.04, 1.02, 0.93], # Gravelines game
            24651 : [1.05, 1.02, 0.92], # Gravelines game
            69244 : [1.035, 1.025, 0.990], # Gravelines game
            59201 : [1.040, 1.030, 0.990], # Gravelines game
            30046 : [0.98, 0.98, 0.98], # Strasbourg game
            # TODO: LAPUA
            # TODO: ESPOO
            **(transform_dict if transform_dict is not None else {})
        }
        # transform_dict is a dict of game_id, gamma correction triplets
        assert all([isinstance(k, int) for k in self.transform_dict.keys()])

    def __call__(self, instant_key: InstantKey, instant: Instant):
        if instant_key.game_id in self.transform_dict.keys():
            gammas = np.array(self.transform_dict[instant_key.game_id])
            for k, image in instant.all_images.items():
                instant.all_images[k] = gamma_correction(image, gammas)
        return instant

@dataclass
class RemoveGroundTruth(Transform):
    keys: list = None
    masks: bool = True
    players: bool = True
    balls: bool = True
    def __call__(self, instant_key: InstantKey, instant: Instant):
        if not self.keys or instant_key in self.keys:
            if self.players:
                instant.annotations = [a for a in instant.annotations if a.type == 'player']
            if self.balls:
                instant.annotations = [a for a in instant.annotations if a.type == 'ball']
            if self.masks and instant.download_flags & DownloadFlags.WITH_HUMAN_SEGMENTATION_MASKS:
                instant.download_flags = instant.download_flags &~ DownloadFlags.WITH_HUMAN_SEGMENTATION_MASKS # removes human segmentation masks flags of that instant
                instant.human_masks = []
        return instant

class ProjectBall(Transform):
    def __call__(self, instant_key: InstantKey, instant: Instant):
        for annotation in instant.annotations:
            if annotation.type == 'ball':
                calib = instant.calibs[annotation.camera]
                annotation.center = calib.project_2D_to_3D(calib.project_3D_to_2D(annotation.center), Z=0)

class RemoveAnnotationMetadata(Transform):
    def __call__(self, instant_key: InstantKey, instant: Instant):
        for attr_name in [
            'annotation_duration',
            'annotation_state',
            'annotation_ts',
            'annotator_id',
        ]:
            setattr(instant, attr_name, None)
        return instant

class CropBlockDividable(Transform):
    def __init__(self, block_size=16):
        self.block_size = block_size

    def __call__(self, instant_key: InstantKey, instant: Instant):
        for k, image in instant.all_images.items():
            h, w = image.shape[:2]
            h = h - h % self.block_size
            w = w - w % self.block_size
            instant.all_images[k] = image[:h, :w]
        return instant
