import os

from ..utils import audio_utils, path_utils

if __name__ == "__main__":

  positive_train_files = [f for f in os.listdir(path_utils.get_train_test_path(path_utils.get_positive_samples_path(), True)) if f.endswith(".wav")]
  raw_files = [f for f in os.listdir(path_utils.get_raw_data_path()) if f.endswith(".wav")]
  
  raw_mean = 0
  for i, raw_file in enumerate(raw_files):
    raw_file_path = os.path.join(path_utils.get_raw_data_path(), raw_file)
    raw_data, fs = audio_utils.load_audio_file(raw_file_path)
    raw_mean += audio_utils.get_mean_at_call_freq(raw_data, fs)
  raw_mean /= len(raw_files)

  positive_mean = 0
  for i, positive_file in enumerate(positive_train_files):
    positive_file_path = os.path.join(path_utils.get_positive_samples_path(), "train", positive_file)
    positive_data, fs = audio_utils.load_audio_file(positive_file_path)
    positive_mean += audio_utils.get_mean_at_call_freq(positive_data, fs)
  positive_mean /= len(positive_train_files)

  print(f"Raw mean: {raw_mean:3e}, positive mean: {positive_mean:3e}")
  print(f"Difference: {positive_mean - raw_mean}")
  print(f"Ratio: {(positive_mean - raw_mean) / raw_mean}")
