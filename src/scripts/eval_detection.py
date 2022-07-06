import torch
import torchvision
import os
import numpy as np
from torch import nn
from src import constants
from src.models.efficientnet import EfficientNet
from src.preprocessing import dataset
from src.pipeline.sliding_window import SlidingWindowPipeline
from src.utils import metrics_utils as metrics
from src.utils import transform_utils, torch_utils, audio_utils, path_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "./trained_models/efficientnet_detek_19.pt"
model = EfficientNet()
model.load_state_dict(torch.load(
    model_path, map_location=torch.device(device)))
model = model.to(device)

reshape_size = (129, 129)
n_workers = 4 if torch.cuda.is_available() else 1
batch_size = 4
train_test = False

audio_transform = transform_utils.baseline_transform

image_transform_no_standardization = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(reshape_size)])
mean, std = constants.mean_test, constants.std_test
print(f"Mean dataset: {mean}, std dataset: {std}")
image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
), torchvision.transforms.Resize(reshape_size), torchvision.transforms.Normalize(mean=mean, std=std)])

SWPipeline = SlidingWindowPipeline(
    window_size=256,
    window_overlap=256//4,
    window_type='hann',
    classification_model=model,
    audio_transform=audio_transform,
    image_transform=image_transform,
    freq_max=12000,
    threshold_nms=0.5,
    device=device
)

data_path = path_utils.get_train_test_path(
    path_utils.get_detection_samples_path(), train_test)
all_audios = [i for i in os.listdir(data_path) if i.endswith(".wav")]
all_lbls = [i for i in os.listdir(data_path) if not i.endswith(".wav")]

input_lbl = np.load(os.path.join(data_path, all_lbls[0]))
input_sample, fs = audio_utils.load_audio_file(
    os.path.join(data_path, all_audios[0]), 24000)

print(input_sample, input_sample.shape)
pred = SWPipeline(input_sample)

print(pred)
