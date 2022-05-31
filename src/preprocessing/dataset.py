import os
import numpy as np

from ..utils import audio_utils

def create_negative_samples(raw_folder: str, output_folder: str, negative_sample_duration: float, max_idx: int) -> None:
  # list all files
  files = [file for file in os.listdir(raw_folder) if file.endswith(".wav")]
  n_files = len(files)

  negative_samples_per_file = int(max_idx / n_files + 0.5)

  idx = 0
  for file in files:
    file_path = os.path.join(raw_folder, file)
    data, fs = audio_utils.load_audio_file(file_path)
    length = negative_sample_duration * fs
    n_splits = int(len(data) / length)
    splitted_data = np.split(data, n_splits)
    for i in np.random.permutation(n_splits)[:negative_samples_per_file]:
      if audio_utils.quick_check_for_call(splitted_data[i], fs) == False:
        audio_utils.save_audio_file(os.path.join(output_folder, f"{idx}.wav"), splitted_data[i], fs)
        idx += 1

def split_samples_test_train(positive_samples_folder: str, negative_samples_folder: str, p_train: float) -> None:
  # list all files
  positive_files = [file for file in os.listdir(positive_samples_folder) if file.endswith(".wav")]
  negative_files = [file for file in os.listdir(negative_samples_folder) if file.endswith(".wav")]
  n_positive_files = len(positive_files)

  n_train = int(n_positive_files * p_train)

  # shuffle files
  np.random.shuffle(positive_files)
  np.random.shuffle(negative_files)

  # split files
  train_positive_files = positive_files[:n_train]
  train_negative_files = negative_files[:n_train]
  test_positive_files = positive_files[n_train:]
  test_negative_files = negative_files[n_train:]

  # create train and test folders if not exist
  train_positive_folder = os.path.join(positive_samples_folder, "train")
  train_negative_folder = os.path.join(negative_samples_folder, "train")
  test_positive_folder = os.path.join(positive_samples_folder, "test")
  test_negative_folder = os.path.join(negative_samples_folder, "test")
  if not os.path.exists(train_positive_folder):
    os.makedirs(train_positive_folder)
  if not os.path.exists(train_negative_folder):
    os.makedirs(train_negative_folder)
  if not os.path.exists(test_positive_folder):
    os.makedirs(test_positive_folder)
  if not os.path.exists(test_negative_folder):
    os.makedirs(test_negative_folder)

  # save files
  for file in train_positive_files:
    os.rename(os.path.join(positive_samples_folder, file), os.path.join(train_positive_folder, file))
  for file in train_negative_files:
    os.rename(os.path.join(negative_samples_folder, file), os.path.join(train_negative_folder, file))
  for file in test_positive_files:
    os.rename(os.path.join(positive_samples_folder, file), os.path.join(test_positive_folder, file))
  for file in test_negative_files:
    os.rename(os.path.join(negative_samples_folder, file), os.path.join(test_negative_folder, file))
