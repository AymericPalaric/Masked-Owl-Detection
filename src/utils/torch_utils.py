import torch
import torchvision
import os

from . import audio_utils, path_utils
from .. import constants

class ClassifDataset(torch.utils.data.Dataset):

  def __init__(self, positive_path, negative_path, hard_path, transform_audio, transform_image):
    self.positive_path = positive_path
    self.negative_path = negative_path
    self.hard_path = hard_path
    self.positive_files = [file for file in os.listdir(self.positive_path) if file.endswith(".wav")]
    self.negative_files = [file for file in os.listdir(self.negative_path) if file.endswith(".wav")]
    self.hard_files = [file for file in os.listdir(self.hard_path) if file.endswith(".wav")]

    self.transform_audio = transform_audio
    self.transform_image = transform_image

  def __len__(self):
    return len(self.positive_files) + len(self.negative_files) + len(self.hard_files)

  def __getitem__(self, idx):
    if idx < len(self.positive_files):
      file_path = os.path.join(self.positive_path, self.positive_files[idx])
      data, fs = audio_utils.load_audio_file(file_path)
      x = self.transform_audio(data)
      x = self.transform_image(x)
      return x, constants.positive_label
    elif idx < len(self.positive_files) + len(self.negative_files):
      file_path = os.path.join(self.negative_path, self.negative_files[idx - len(self.positive_files)])
      data, fs = audio_utils.load_audio_file(file_path)
      x = self.transform_audio(data)
      x = self.transform_image(x)
      return x, constants.negative_label
    else:
      file_path = os.path.join(self.hard_path, self.hard_files[idx - len(self.positive_files) - len(self.negative_files)])
      data, fs = audio_utils.load_audio_file(file_path)
      x = self.transform_audio(data)
      x = self.transform_image(x)
      return x, constants.hard_label

def accuracy(y, targets):
  return (y.argmax(dim=1) == targets).float().mean()

def create_classif_dataset(audio_transform, image_transform, train_test: bool):
  dataset = ClassifDataset(
    positive_path=path_utils.get_train_test_path(path_utils.get_positive_samples_path(), train_test),
    negative_path=path_utils.get_train_test_path(path_utils.get_negative_samples_path(), train_test),
    hard_path=path_utils.get_train_test_path(path_utils.get_hard_samples_path(), train_test),
    transform_audio=audio_transform,
    transform_image=image_transform)
  return dataset

def compute_mean_std_classif(audio_transform, image_transform, batch_size: int, num_workers: int):
  mean = 0.
  std = 0.
  dataset = create_classif_dataset(audio_transform, image_transform, train_test=True)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
  for data, target in dataloader:
    mean += data.mean()
    std += data.std() ** 2
  mean = mean / len(dataset)
  std = (std / len(dataset)).sqrt()
  return mean, std
