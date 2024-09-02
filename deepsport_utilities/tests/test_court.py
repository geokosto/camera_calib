import pytest

import numpy as np
import matplotlib.pyplot as plt

from calib3d import Calib

from deepsport_utilities.court import Court

COURT = Court("NBA")

def _get_bird_eye_view_calib():
    """Return the camera calibration of a bird-eye view of the basketball court
​
    Returns:
        Calib: calibration object
    """
    im_width = int(COURT.w)
    im_height = int(COURT.h)
    intrisic_bird_eye = np.array([[1, 0, im_width / 2], [0, 1, im_height / 2], [0, 0, 1]], dtype=float)
    rotation_matrix = np.diag([1, 1, 1])
    optical_center = np.array([COURT.w/2, COURT.h/2, -1.2], dtype=np.float32).reshape((3, 1))
    return Calib(width=im_width, height=im_height, T=-1*rotation_matrix@optical_center, R=rotation_matrix, K=intrisic_bird_eye)

@pytest.mark.skip(reason="Requires visualization")
def test_draw_lines():
    """ Draw a bird-eye view of the court """
    calib = _get_bird_eye_view_calib()
    rgb_court_img = np.full(shape=(calib.height, calib.width, 3), fill_value=255, dtype=np.uint8)
    COURT.draw_lines(rgb_court_img, calib, color=(0,0,0))
    plt.imshow(rgb_court_img)
    plt.show()