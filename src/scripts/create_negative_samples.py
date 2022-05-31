import os

from ..preprocessing import dataset
from ..utils import path_utils

if __name__ == "__main__":

  positive_files = [file for file in os.listdir(path_utils.get_positive_samples_path()) if file.endswith(".wav")]
  dataset.create_negative_samples(
    raw_folder=path_utils.get_raw_data_path(),
    output_folder=path_utils.get_negative_samples_path(), 
    negative_sample_duration=2, 
    max_idx=len(positive_files))
