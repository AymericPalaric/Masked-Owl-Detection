from src.utils import audio_utils
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
from src.pipeline import sliding_window
from src.utils import transform_utils
from src.models.baseline_cnn.model import Baseline
from src.utils import metrics_utils


if __name__ == "__main__":

  classification_model = Baseline()
  classification_model.load_state_dict(torch.load("trained_models/baseline_cnn_model_4.pt"))
  classification_model.eval()

  window_size = 22000 * 2
  window_overlap = window_size // 4
  window_type = "hann"
  reshape_size = (129, 129)
  audio_transform = transform_utils.baseline_transform
  image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(reshape_size), torchvision.transforms.Normalize(mean=-0.32, std=0.571)])

  freq_max = 129
  treshold_nms = 0.5
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  pipeline = sliding_window.SlidingWindowPipeline(window_size, window_overlap, window_type, classification_model, audio_transform, image_transform, freq_max, treshold_nms, device)

  time_scale = 1 / (256 - 256//4)

  files = [file[:-4] for file in os.listdir("data/detection_samples/train") if file.endswith(".wav")][:5]
  audios = list()
  img_target = list()
  bbxs_target = list()
  for k, file in enumerate(files):
    audio_path = os.path.join("data/detection_samples/train", f"{file}.wav")
    bbxs_path = os.path.join("data/detection_samples/train", f"{file}.npy")
    audio, _ = audio_utils.load_audio_file(audio_path)
    bbxs = np.load(bbxs_path)

    audios.append(audio)

    for bbx in bbxs:
      if bbx[0] == 1:
        img_target.append(k)
        bbxs_target.append([int(bbx[1] * time_scale), 0, int(bbx[2] * time_scale), freq_max])

  audios = np.stack(audios)
  img_target = torch.tensor(img_target, dtype=torch.long)
  bbxs_target = torch.tensor(bbxs_target, dtype=torch.long)

  img_pred, bbxs_pred, score_pred = pipeline(audios)

  mAP = metrics_utils.compute_mAP((img_target, bbxs_target), (img_pred, bbxs_pred, score_pred), 0.5)

  print("mAP: ", mAP)
