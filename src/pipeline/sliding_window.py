from __future__ import annotations

from typing import TYPE_CHECKING

from scipy import signal
import torch
import torchvision

from .. import constants

if TYPE_CHECKING:
    import numpy as np


class SlidingWindowPipeline():

    def __init__(self, window_size: int, window_overlap: int, window_type: str, classification_model, audio_transform, image_transform, freq_max, treshold_nms, device) -> None:
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.window_type = window_type

        self.classification_model = classification_model
        self.audio_transform = audio_transform
        self.image_transform = image_transform
        self.freq_max = freq_max
        self.treshold_nms = treshold_nms
        self.device = device

        self.window = signal.windows.get_window(
            self.window_type, self.window_size)

    def forward(self, data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        windows = self.get_windows(data).to(torch.float32).to(self.device)
        self.classification_model.eval()

        predictions = torch.nn.functional.softmax(
            self.classification_model(windows), dim=1)
        bbxs, scores = self.convert_predictions_to_bbx(predictions)
        bbxs, scores = self.apply_nms(bbxs, scores, self.treshold_nms)
        return bbxs, scores

    def get_windows(self, data: np.ndarray):
        windows = list()
        for i in range(0, data.shape[0] - self.window_size, self.window_size - self.window_overlap):
            window = data[i:i+self.window_size]
            window = window * self.window
            window = self.audio_transform(window)
            window: torch.Tensor = self.image_transform(window)
            windows.append(window)
        return torch.stack(windows, dim=0)

    def convert_predictions_to_bbx(self, predictions: torch.Tensor):
        bbxs = list()
        scores = list()
        for i in range(predictions.shape[0]):
            if predictions[i, constants.positive_label] > 0.5:
                idx_start = i * (self.window_size - self.window_overlap)
                bbxs.append([idx_start, 0, idx_start +
                            self.window_size, self.freq_max])
                scores.append(predictions[i, constants.positive_label])
        return torch.tensor(bbxs), torch.tensor(scores)

    def apply_nms(self, bbxs: torch.Tensor, scores: torch.Tensor, threshold: float):
        if bbxs.shape[0] == 0:
            return bbxs, scores
        indices = torchvision.ops.nms(bbxs, scores, threshold)
        return bbxs[indices], scores[indices]

    def batched_forward(self, datas: np.ndarray):
        im_idxs = list()
        full_bbxs = list()
        full_scores = list()
        for k, data in enumerate(datas):
            bbxs, scores = self.forward(data)
            im_idxs.extend([k for _ in range(len(scores))])
            full_bbxs.append(bbxs)
            full_scores.append(scores)
        return torch.tensor(im_idxs), torch.stack(full_bbxs, dim=0), torch.stack(full_scores, dim=0)

    def __call__(self, datas: np.ndarray):
        if datas.ndim == 2:
            return self.batched_forward(datas)
        return self.forward(datas)
