# DeepSport Utilities toolkit

This toolkit offers a wide range of helpers to download, transform and use datasets following the DeepSport format.

## Installation
Package is released on PyPI for convenient installation:
```bash
pip install deepsport-utilities
```
For developement, install with:
```bash
git clone
pip install -e ".[dev]"
```

## Available datasets

- [Basketball Instants Dataset](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset) is implemented by `deepsport_utilities.InstantsDataset` and provides raw images captured by the Keemotion system at different *instants*.
- [Basketball Ballistic Trajectories Dataset](https://www.kaggle.com/datasets/gabrielvanzandycke/ballistic-raw-sequences), implemented as multiple successive *instants*.
- Private Keemotion datasets.

## Dataset format

The datasets are stored as a json file holding items metadata and multiple data files. The easiest approach to load the data is to use the `import_dataset` function as illustrated in the scripts provided in the `examples` folder.

The resulting datasets are based on `mlworkflow.Dataset`, a dataset implementation of (key, value) pairs where the keys are light and allow efficient querying of the dataset while the values contain the heavy data. For more information, refer to [mlworkflow repository](https://github.com/ispgroupucl/mlworkflow).

## Toolkit

Along with the provided datasets, the library comes with utility functions to process the dataset work with basketball courts.


## Working with calibrated images captured by the Keemotion system.

Calibration data are implemented with [`cailb3d.Calib`](https://ispgroupucl.github.io/calib3d/calib3d/calib.html) objects, allowing determining the relation between the image pixels (2D coordinates) and points in the real world (3D coordinates).

Images shared by Keemotion follow a convention where the **origin of the 3D world** is located on the furthest left corner of the basketball court relative to the main camera setup ; more precisely in the *inner side* of the court lines. The **unit of length** is the centimeter and **axis orientation** is illustrated here, with z pointing *downward*:

<img src="assets/keemotion_3D_world_convention.png" alt="Keemotion 3D world convention located in the furthest left corner of the court relative to the main camera setup" width="600"/>


## Contributing

This library is open-source and contributions are welcome. However, prior to any implementation, a discussion with the main maintainer Gabriel Van Zandycke is required.


## Authors and acknowledgment
While most of the library was developed by Gabriel Van Zandycke, this library benefited from the work of
- Maxime Istasse for the project initial kick-off
- Cedric Verleysen for some functions in `court.py`
