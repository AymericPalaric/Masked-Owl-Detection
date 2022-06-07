import os
from scipy import signal
import numpy as np
from tqdm import tqdm

from ..utils import audio_utils

def create_negative_samples(raw_folder: str, output_folder: str, negative_sample_duration: float, n_samples: int, uuid: int, log: bool=True) -> None:
  # list all files
  files = [file for file in os.listdir(raw_folder) if file.endswith(".wav")]
  n_files = len(files)

  negative_samples_per_file = int(n_samples / n_files + 0.5)

  idx = 0
  for file in tqdm(files, disable=not log):
    file_path = os.path.join(raw_folder, file)
    data, fs = audio_utils.load_audio_file(file_path)
    length = negative_sample_duration * fs
    n_splits = int(len(data) / length)
    splitted_data = np.array_split(data, n_splits)
    for i in np.random.permutation(n_splits)[:negative_samples_per_file]:
      audio_utils.save_audio_file(os.path.join(output_folder, f"{uuid}_{idx}.wav"), splitted_data[i], fs)
      idx += 1

def split_samples_test_train(positive_samples_folder: str, negative_samples_folder: str, hard_samples_folder: str, p_train: float, p_hard: float) -> None:
  """
  Split positive samples into train and test set.
  Distribution: trainning data : 50% of positive samples and 50% of negative samples (p_hard proportion of negative samples are hard samples).
  Repartition: train test data : p_train proportion of positive samples in train data
  All remaining samples are in test data.
  """
  # list all files
  positive_files = [file for file in os.listdir(positive_samples_folder) if file.endswith(".wav")]
  negative_files = [file for file in os.listdir(negative_samples_folder) if file.endswith(".wav")]
  hard_files = [file for file in os.listdir(hard_samples_folder) if file.endswith(".wav")]
  n_positive_files = len(positive_files)
  n_negative_files = len(negative_files)
  n_hard_samples = len(hard_files)

  n_train = int(n_positive_files * p_train)
  n_hard = int(n_train * p_hard)
  n_negative = n_train - n_hard

  # set seed
  np.random.seed(0)

  # shuffle files
  np.random.shuffle(positive_files)
  np.random.shuffle(negative_files)
  np.random.shuffle(hard_files)

  # split files
  train_positive_files = positive_files[:n_train]
  train_negative_files = negative_files[:n_negative]
  train_hard_files = hard_files[n_hard:]

  test_positive_files = positive_files[n_train:]
  test_negative_files = negative_files[n_negative:]
  test_hard_files = hard_files[:n_hard]
  
  # create train and test folders if not exist
  train_positive_folder = os.path.join(positive_samples_folder, "train")
  train_negative_folder = os.path.join(negative_samples_folder, "train")
  train_hard_folder = os.path.join(hard_samples_folder, "train")
  test_positive_folder = os.path.join(positive_samples_folder, "test")
  test_negative_folder = os.path.join(negative_samples_folder, "test")
  test_hard_folder = os.path.join(hard_samples_folder, "test")
  if not os.path.exists(train_positive_folder):
    os.makedirs(train_positive_folder)
  if not os.path.exists(train_negative_folder):
    os.makedirs(train_negative_folder)
  if not os.path.exists(train_hard_folder):
    os.makedirs(train_hard_folder)
  if not os.path.exists(test_positive_folder):
    os.makedirs(test_positive_folder)
  if not os.path.exists(test_negative_folder):
    os.makedirs(test_negative_folder)
  if not os.path.exists(test_hard_folder):
    os.makedirs(test_hard_folder)

  # save files
  for file in train_positive_files:
    os.rename(os.path.join(positive_samples_folder, file), os.path.join(train_positive_folder, file))
  for file in train_negative_files:
    os.rename(os.path.join(negative_samples_folder, file), os.path.join(train_negative_folder, file))
  for file in train_hard_files:
    os.rename(os.path.join(hard_samples_folder, file), os.path.join(train_hard_folder, file))
  for file in test_positive_files:
    os.rename(os.path.join(positive_samples_folder, file), os.path.join(test_positive_folder, file))
  for file in test_negative_files:
    os.rename(os.path.join(negative_samples_folder, file), os.path.join(test_negative_folder, file))
  for file in test_hard_files:
    os.rename(os.path.join(hard_samples_folder, file), os.path.join(test_hard_folder, file))

  # save distribution
  with open(os.path.join(positive_samples_folder, "distribution.txt"), "w") as f:
    f.write(f"{n_train} positive samples in train data\n")
    f.write(f"{n_positive_files - n_train} positive samples in test data\n")
  with open(os.path.join(negative_samples_folder, "distribution.txt"), "w") as f:
    f.write(f"{n_train} negative samples in train data\n")
    f.write(f"{n_negative_files - n_train} negative samples in test data\n")
  with open(os.path.join(hard_samples_folder, "distribution.txt"), "w") as f:
    f.write(f"{n_hard} hard samples in train data\n")
    f.write(f"{n_hard_samples - n_hard} hard samples in test data\n")

def create_detection_dataset(output_folder: str, raw_folder: str, positive_folder: str, hard_folder: str, detection_duration: float, n_samples: int, uuid: int, call_proportion: float, hard_call_proportion: float, log: bool=True) -> None:
  # set seed
  np.random.seed(0)

  raw_files = [file for file in os.listdir(raw_folder) if file.endswith(".wav")]
  positive_files = [file for file in os.listdir(positive_folder) if file.endswith(".wav")]
  hard_files = [file for file in os.listdir(hard_folder) if file.endswith(".wav")]

  samples_per_file = int(n_samples / len(raw_files) + 0.5)

  for i, raw_file in enumerate(tqdm(raw_files, disable=not log)):
    raw_file_path = os.path.join(raw_folder, raw_file)
    raw_data, fs = audio_utils.load_audio_file(raw_file_path)

    # split raw data
    n_splits = int(len(raw_data) / (detection_duration * fs))
    splitted_data = np.array_split(raw_data, n_splits)[:samples_per_file]

    for j in range(len(splitted_data)):
      data_split = splitted_data[j]
      call_indices = []

      # determine the number of positive calls
      n_calls = np.random.poisson(call_proportion)
      indices_call = np.random.permutation(len(positive_files))[:n_calls]
      for k in indices_call:
        positive_file_path = os.path.join(positive_folder, positive_files[k])
        positive_data, _ = audio_utils.load_audio_file(positive_file_path, sr=fs)

        positive_data = extract_data(positive_data)
        add_call_raw_data(data_split, call_indices, positive_data, positif=True)

      # determine the number of hard calls
      n_calls = np.random.poisson(hard_call_proportion)
      indices_call = np.random.permutation(len(hard_files))[:n_calls]
      for k in indices_call:
        hard_file_path = os.path.join(hard_folder, hard_files[k])
        hard_data, _ = audio_utils.load_audio_file(hard_file_path, sr=fs)

        hard_data = extract_data(hard_data)
        add_call_raw_data(data_split, call_indices, hard_data, positif=False)

      call_indices = np.array(call_indices)
      #  save audio and positive call indices
      audio_utils.save_audio_file(os.path.join(output_folder, f"{uuid}_{i}_{j}.wav"), data_split, fs)
      np.save(os.path.join(output_folder, f"{uuid}_{i}_{j}.npy"), call_indices)

def add_call_raw_data(data_split: np.ndarray, call_indices: list[list[int]], call_data: np.ndarray, positif: bool) -> None:
    start = np.random.randint(0, len(data_split) - len(call_data))
    end = start + len(call_data)
    data_split[start:end] += call_data
    call_indices.append([1 if positif else 0, start, end])

def extract_data(data: np.ndarray, p_length_min: float=0.8, p_length_max: float=1.0):
    n_samples = int(len(data) * np.random.uniform(p_length_min, p_length_max))
    start_sample = np.random.uniform(0, len(data) - n_samples)
    return data[int(start_sample):int(start_sample + n_samples)] * signal.windows.hann(n_samples)
