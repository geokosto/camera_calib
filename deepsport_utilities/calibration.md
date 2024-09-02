# Working with calibrated images captured by the Keemotion system.

The calibration data shared by Keemotion allows determining the relation between the image’s pixels (2D coordinates) and points in the real world (3D coordinates). You can read https://ispgroupucl.github.io/calib3d/calib3d/calib.html for more informations about how it works and how to use it.

For Keemotion produced images, the origin of the 3D world is, by a Keemotion convention, located on the furthest left corner of the basketball court relative to the main camera setup ; more precisely in the inner side of the court lines. The unit of length is the centimeter and axis orientation is illustrated here, with z pointing downward:

<img src="assets/keemotion_3D_world_convention.png" alt="Keemotion 3D world convention located in the furthest left corner of the court relative to the main camera setup" width="600"/>


