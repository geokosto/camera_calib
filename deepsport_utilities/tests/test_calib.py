import os
import cv2
import pytest
import numpy as np
from deepsport_utilities.calib import Calib
from deepsport_utilities.court import Court

# Keemotion calibration
COURT1 = Court("FIBA")
K1 = np.array([[2.06219798e+03, -9.22434310e-07, -9.60000015e+02], [ 0.00000000e+00, -2.06219803e+03, -5.40000019e+02], [0.00000000e+00, 0.00000000e+00, -1.00000001e+00]])
R1 = np.array([[9.74532373e-01, -2.24246858e-01, -1.31676671e-09], [-6.27818634e-02, -2.72837522e-01, -9.60009440e-01], [2.15279100e-01, 9.35560278e-01, -2.79967633e-01]])
T1 = np.array([[-863.54133491, 164.89179947, -2569.21726222]]).T
IM_WIDTH1 = 1920
IM_HEIGHT1 = 1080
CALIB1 = Calib(width=IM_WIDTH1, height=IM_HEIGHT1, T=T1, K=K1, kc=None, R=R1)

def test_parse_DeepSport():
    assert CALIB1

def test_court_constructor():
    assert COURT1

def test_calib_scale():
    assert CALIB1.scale(100, 100)

def test_fill_polygon():
    new_im_height = 100
    new_im_width = 100
    new_calib = CALIB1.scale(new_im_height, new_im_width)
    overlay_mask = np.zeros((new_im_height, new_im_width), dtype=np.uint8)
    COURT1.fill_court(overlay_mask, new_calib)
    groundtruth = cv2.imread(os.path.join(os.path.dirname(__file__), "groundtruths/test_fill_polygon.png"), cv2.IMREAD_GRAYSCALE)
    assert groundtruth is not None
    assert np.all(overlay_mask == groundtruth)

# Data extracted from Keemotion game 94132, first produced frame
INPUT_IMG2 = cv2.imread(os.path.join(os.path.dirname(__file__), "input_imgs", "game_94132_frame_idx_0.png"))
K2 = np.array([[ 1.24710165e+03, -3.09307822e-07, -9.59999971e+02], [ 0.00000000e+00, -1.24710170e+03, -5.40000005e+02], [ 0.00000000e+00, 0.00000000e+00, -9.99999971e-01]])
R2 = np.array([[ 9.98153533e-01, -6.07414639e-02, -4.16965239e-11], [-2.15711992e-02, -3.54475630e-01, -9.34816405e-01], [ 5.67821169e-02, 9.33090297e-01, -3.55131368e-01]])
T2 = np.array([[-1265.43374602, 305.21811614, -2383.12629084]]).T
COURT2 =  Court("FIBA")
CALIB2 = Calib(width=INPUT_IMG2.shape[1], height=INPUT_IMG2.shape[0], T=T2, K=K2, kc=None, R=R2)

@pytest.mark.skip(reason="Requires visual inspection")
def test_get_visible_corners_2d_with_drawing():
    """ Draw the coordinates delimiting the basketball court """
    courts_corners = CALIB2.get_region_visible_corners_2d(COURT2.corners)
    input_img_with_magenta_court = cv2.fillPoly(INPUT_IMG2.copy(), np.array([[tuple(np.int32(cc)) for cc in courts_corners.T]]), color=(255, 0, 255))
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', input_img_with_magenta_court)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_get_visible_corners_2d():
    """ Get the coordinates delimiting the basketball court """
    CALIB2.get_region_visible_corners_2d(COURT2.corners)


@pytest.mark.skip(reason="Requires visual inspection")
def test_get_left_board_visible_corners_2d_with_drawing():
    """ Draw the coordinates delimiting the left basketball board """
    left_board_corners = CALIB2.get_region_visible_corners_2d(COURT2.left_board)
    input_img_with_magenta_left_board = cv2.fillPoly(INPUT_IMG2.copy(), np.array([[tuple(np.int32(bc)) for bc in left_board_corners.T]]), color=(255, 0, 255))
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', input_img_with_magenta_left_board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_get_left_board_visible_corners_2d():
    """ Get the coordinates delimiting the left basketball board """
    left_board_corners = CALIB2.get_region_visible_corners_2d(COURT2.left_board)
    assert left_board_corners.size, "The left basketball board is visible on the image, and the function should return its coordinates"

def test_get_right_board_visible_corners_2d():
    """ Get the coordinates delimiting the right basketball board """
    right_board_corners = CALIB2.get_region_visible_corners_2d(COURT2.right_board)
    assert right_board_corners.size == 0, "The right basketball board is NOT visible on the image, and the function should return an empty array"

@pytest.mark.skip(reason="Requires visual inspection")
def test_get_left_key_area_visible_corners_2d_with_drawing():
    """ Draw the coordinates delimiting the left basketball key area """
    left_key_area_corners = CALIB2.get_region_visible_corners_2d(COURT2.left_key_area)
    input_img_with_magenta_left_key_area = cv2.fillPoly(INPUT_IMG2.copy(), np.array([[tuple(np.int32(kc)) for kc in left_key_area_corners.T]]), color=(255, 0, 255))
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', input_img_with_magenta_left_key_area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_get_key_areas_visible_corners_2d():
    """ Get the coordinates delimiting the basketball key areas """
    left_key_area_corners = CALIB2.get_region_visible_corners_2d(COURT2.left_key_area)
    assert left_key_area_corners.size, "The left basketball key area is visible on the image, and the function should return its coordinates"

@pytest.mark.skip(reason="Requires visual inspection")
def test_get_right_key_area_visible_corners_2d_with_drawing():
    """ Draw the coordinates delimiting the right basketball key area """
    right_key_area_corners = CALIB2.get_region_visible_corners_2d(COURT2.right_key_area)
    input_img_with_magenta_right_key_area = cv2.fillPoly(INPUT_IMG2.copy(), np.array([[tuple(np.int32(kc)) for kc in right_key_area_corners.T]]), color=(255, 0, 255))
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', input_img_with_magenta_right_key_area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_get_right_key_ares_visible_corners_2d():
    """ Get the coordinates delimiting the right basketball key area """
    right_key_area_corners = CALIB2.get_region_visible_corners_2d(COURT2.right_key_area)
    assert right_key_area_corners.size, "The right basketball key area is visible on the image, and the function should return its coordinates"

