# %%
import cv2
import numpy as np
from calib3d.calib3d.calib import Calib
import json


# %%
# Load calibration data from .npz file
def load_calibration_data(npz_file_path):
    with np.load(npz_file_path) as data:
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        rvecs = data["rvecs"]
        tvecs = data["tvecs"]
    return camera_matrix, dist_coeffs, rvecs, tvecs


# %%
# Load calibration data from the new structure
def load_calibration_data_from_dict(calibration_dict):
    camera_matrix = np.array(calibration_dict["KK"]).reshape(3, 3)
    dist_coeffs = np.array(calibration_dict["kc"])
    R = np.array(calibration_dict["R"]).reshape(3, 3)
    T = np.array(calibration_dict["T"]).reshape(3, 1)

    # Convert rotation matrix R to rotation vector rvec
    rvec, _ = cv2.Rodrigues(R)
    rvecs = [rvec]
    tvecs = [T]

    return camera_matrix, dist_coeffs, rvecs, tvecs


# %%
filename = "data/KS-FR-BOURGEB/24642/camcourt2_1512421231644"

image = cv2.imread(f"{filename}_0.png")

calibration_data = json.load(open(f"{filename}.json"))["calibration"]
# calibration_data = "data/camera_calibration.npz"
# %%

# camera_matrix, dist_coeffs, rvecs, tvecs = load_calibration_data(calibration_data)
camera_matrix, dist_coeffs, rvecs, tvecs = load_calibration_data_from_dict(
    calibration_data
)

# %%
print("Camera matrix (K):", camera_matrix)
print("Distortion coefficients (kc):", dist_coeffs)
print("Rotation vectors (rvecs):", rvecs)
print("Translation vectors (tvecs):", tvecs)

# Convert rotation vectors to rotation matrices
R_matrices = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]

# Use the first rotation matrix and translation vector for the Calib object
R = R_matrices[0]
T = tvecs[0]

# %%

image_height, image_width = image.shape[:2]

# %%
# Create a Calib object
calib = Calib(
    width=image_width,
    height=image_height,
    T=T,
    R=R,
    K=camera_matrix,
    # kc=dist_coeffs,
)

# Print the calibration data
print("Intrinsic matrix (K):", calib.K)
print("Distortion coefficients (kc):", calib.kc)
print("Rotation matrix (R):", calib.R)
print("Translation vector (T):", calib.T)

extrinsic_matrix = np.eye(4)
extrinsic_matrix[:3, :3] = calib.R
extrinsic_matrix[:3, 3] = calib.T.flatten()

print("Extrinsic matrix:", extrinsic_matrix)

# %%


from deepsport_utilities.deepsport_utilities.court import Court

court_rule_type = "FIBA"
court = Court(rule_type=court_rule_type)
court.draw_lines(image, calib)
# court.draw_rim(image, calib)
# court.fill_board(image, calib)


# Display the image with the court drawn on it. make the image smaller
cv2.imshow("Court", cv2.resize(image, (0, 0), fx=0.7, fy=0.7))
cv2.waitKey(0)
cv2.destroyAllWindows()

# save the image
cv2.imwrite("data/court_with_lines.png", image)


# %%
