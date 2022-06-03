import os
import argparse

from ..preprocessing import dataset
from ..utils import path_utils

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Create negative samples from raw data.")
  parser.add_argument("--n_samples", type=int, required=False, help="Nomber of samples to create.", default=-1)
  parser.add_argument("--uuid", type=int, required=True, help="Unique identifier for this batch of extracted samples.")
  parser.add_argument("--log", type=bool, required=False, help="Bool to enable/disable logging.", default=True)

  args = parser.parse_args()

  if args.n_samples < 0:
    positive_files = [file for file in os.listdir(path_utils.get_positive_samples_path()) if file.endswith(".wav")]
    n_samples = len(positive_files)
  else:
    n_samples = args.n_samples
  uuid = args.uuid
  log = args.log

  dataset.create_negative_samples(
    raw_folder=path_utils.get_raw_data_path(),
    output_folder=path_utils.get_negative_samples_path(), 
    negative_sample_duration=2, 
    n_samples=n_samples,
    uuid=uuid,
    log=log)
