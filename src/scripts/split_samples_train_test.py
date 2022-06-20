import os

from ..preprocessing import dataset
from ..utils import path_utils

if __name__ == "__main__":

  dataset.split_samples_test_train(path_utils.get_positive_samples_path(), path_utils.get_negative_samples_path(), path_utils.get_hard_samples_path(), 
    p_train=0.8,
    p_hard=0.4)
