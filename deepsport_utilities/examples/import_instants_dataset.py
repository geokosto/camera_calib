import os
from tqdm.auto import tqdm
from mlworkflow import PickledDataset
from deepsport_utilities import import_dataset, InstantsDataset
from deepsport_utilities.ds.instants_dataset import DownloadFlags


dataset_folder = "/DATA/datasets/basketball-instants-dataset"

# The `dataset_config` is used to create each dataset item
dataset_config = {
    "download_flags": DownloadFlags.WITH_IMAGE | DownloadFlags.WITH_CALIB_FILE, # corresponds to images and calibration data
    "dataset_folder": dataset_folder  # tells where raw files are stored (.png and .json files)
}

# Load dataset
ds = import_dataset(InstantsDataset, os.path.join(dataset_folder, "basketball-instants-dataset.json"), **dataset_config)

# Save as PickledDataset for convenient loading later
PickledDataset.create(ds, os.path.join(dataset_folder, "basketball-instants-dataset.pickle"), yield_keys_wrapper=tqdm)
