from src.utils import transform_utils, torch_utils

import torchvision


if __name__ == "__main__":

  audio_transform = transform_utils.baseline_transform

  reshape_size = (129, 129)
  batch_size = 64
  n_workers = 4

  image_transform_no_standardization = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(reshape_size)])
  mean, std = torch_utils.compute_mean_std_classif(audio_transform=audio_transform, image_transform=image_transform_no_standardization, batch_size=batch_size, num_workers=n_workers)
  print(f"Mean dataset: {mean}, std dataset: {std}")
