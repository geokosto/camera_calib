import os

from tqdm.auto import tqdm
from mlworkflow import PickledDataset, TransformedDataset, FilteredDataset

from deepsport_utilities.ds.instants_dataset import ViewsDataset, BuildCameraViews, AddBallAnnotation

# Load instants dataset from previously pickled dataset
ids = PickledDataset("/DATA/datasets/gva/deepsport-basketball-instants-dataset.pickle")

# Transform the dataset of instants into a dataset of views for each camera
ds = ViewsDataset(ids, view_builder=BuildCameraViews())

# Add the 'ball' attribute to the views, a shortcut to the ball in the annotation list
ds = TransformedDataset(ds, [AddBallAnnotation()])

# Filter only views for which camera index is the one in which the ball was annotated
ds = FilteredDataset(ds, predicate=lambda k,v: k.camera == v.ball.camera)

# Save the working dataset to disk with data contiguously stored for efficient reading during training
PickledDataset.create(ds, "/DATA/datasets/gva/deepsport-camera-views-dataset.pickle", yield_keys_wrapper=tqdm)
