import os

from .. import config, constants

def get_positive_samples_path():
  return os.path.join(config.data_path, constants.positive_samples_folder)

def get_raw_data_path():
  return os.path.join(config.data_path, constants.raw_data_folder)

def get_negative_samples_path():
  return os.path.join(config.data_path, constants.negative_samples_folder)

def get_train_test_path(path: str, trainning: bool):
  return os.path.join(path, "train" if trainning else "test")
