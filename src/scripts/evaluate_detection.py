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
from src.models.efficientnet import EfficientNet
from src.utils import metrics_utils
from src import constants
import numpy as np


if __name__ == "__main__":

  classification_model = EfficientNet()
  classification_model.load_state_dict(torch.load("trained_models/efficientnet_nosoft_19.pt"))
  classification_model.eval()

  window_size = int(22000 * 1)
  window_overlap = window_size // 50
  window_type = "boxcar"
  reshape_size = (129, 129)
  audio_transform = transform_utils.baseline_transform
  image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(reshape_size), torchvision.transforms.Normalize(mean=constants.mean_test, std=constants.std_test)])

  freq_max = 12000
  treshold_nms = 0.5
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  pipeline = sliding_window.SlidingWindowPipeline(window_size, window_overlap, window_type, classification_model, audio_transform, image_transform, freq_max, treshold_nms, device)

  time_scale = 1 / (256 - 256//4)

  files = [file[:-4] for file in os.listdir("data/detection_samples/samples") if file.endswith(".wav")][:5]
  audios = list()
  img_target = list()
  bbxs_target = list()
  for k, file in enumerate(files):
    audio_path = os.path.join("data/detection_samples/samples", f"{file}.wav")
    bbxs_path = os.path.join("data/detection_samples/targets", f"{file}.npy")
    audio, _ = audio_utils.load_audio_file(audio_path)
    bbxs = np.load(bbxs_path)

    audios.append(audio)

    for bbx in bbxs:
      if bbx[0] == 1:
        img_target.append(k)
        bbxs_target.append([int(bbx[1] * time_scale), 0, int(bbx[2] * time_scale), freq_max])
  cropping_len = min([len(audio) for audio in audios])
  audios = [audio[:cropping_len] for audio in audios]
  audios = np.stack(audios)
  img_target = torch.tensor(img_target, dtype=torch.long)
  bbxs_target = torch.tensor(bbxs_target, dtype=torch.long)

  #img_pred, bbxs_pred, score_pred = pipeline(audios)
  #print(bbxs_pred)
  #print(10*"=")
  #print(score_pred)
  #mAP = metrics_utils.compute_mAP((img_target, bbxs_target), (img_pred, bbxs_pred, score_pred), 0.5)

  #print("mAP: ", mAP)

