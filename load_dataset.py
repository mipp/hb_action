"""Load a dataset from a directory of video files."""

import os

def class_from_fname(fname):
  """Get the action class from filename"""
  return os.path.basename(fname).split('_')[0]

def load_dataset(dataset_dir, label_map_path):
  """
  Loads a dataset of videos from a directory. Filenames should begin
  with action class and underscore. 

  Parameters:
  - dataset_dir (string): path to a directory with videos
  - label_map_path (string): path to file with class labels

  Returns:
  - A tuple (X, y) where the y values
    are class indices for the X values. Each X value is a string
    pointing to the file path of a video.
  """

  assert os.path.isdir(dataset_dir), 'Invalid directory for dataset'
  assert os.path.isfile(label_map_path), 'Invalid label map path'

  invalid = ['.DS_Store', '.', '..']
  filenames = [f for f in os.listdir(dataset_dir) if f not in invalid]
  X = [os.path.join(dataset_dir, f) for f in filenames]

  valid_classes = [x.strip() for x in open(label_map_path)]
  y = [valid_classes.index(class_from_fname(f)) for f in filenames]

  return (X, y)
