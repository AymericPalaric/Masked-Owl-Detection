from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import result

from scipy import signal
import torch
import torchvision
from ..utils.audio_utils import cwt_roi
from .. import constants

if TYPE_CHECKING:
    import numpy as np

class   RoiWindowPipeline():

    def __init__(self, window_size: int, window_overlap: int, window_type: str, classification_model, audio_transform, image_transform, freq_max, treshold_nms, device) -> None:
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.window_type = window_type

        self.classification_model = classification_model.to(device)
        self.audio_transform = audio_transform
        self.image_transform = image_transform
        self.freq_max = freq_max
        self.treshold_nms = treshold_nms
        self.device = device

        self.window = signal.windows.get_window(
            self.window_type, self.window_size)

    def forward(self, data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        
        output = self.get_windows(data)
        if not output:
            return list(), list()
        else:
            windows,idxs = output
            windows=windows.to(torch.float32).to(self.device)
            self.classification_model.eval()
            predictions = torch.nn.functional.softmax(
                self.classification_model(windows), dim=1)
            bbxs, scores = self.convert_predictions_to_bbx(predictions,idxs)
            bbxs, scores = self.apply_nms(bbxs, scores, self.treshold_nms)
            return bbxs, scores

    def roi_window(self, data: np.ndarray):
        rois=cwt_roi(data,24000)
        windows_indexes=[]
        if rois:
            for roi in rois:
                start_index = roi[0]//(self.window_size-self.window_overlap)
                windows_indexes.append(start_index)
                if (roi[1]-roi[0])%(self.window_size-self.window_overlap)==0:
                    pass
                else:
                    for i in range(start_index+1,start_index+((roi[1]-roi[0])//(self.window_size-self.window_overlap))+1):
                        windows_indexes.append(i)
        return windows_indexes

    def get_windows(self, data: np.ndarray):
        windows = list()
        idxs = list()
        rois= self.roi_window(data)
        if not rois:
            return None
        else:
            for i in rois:
                pad=int((self.window_size-self.window_overlap))
                if i*pad+self.window_size > data.shape[0]:
                    break
                window = data[i*pad:i*pad+self.window_size]
                window = window * self.window
                window = self.audio_transform(window)
                window: torch.Tensor = self.image_transform(window)
                idxs.append(i*pad)
                windows.append(window)
            return torch.stack(windows, dim=0),idxs



    def convert_predictions_to_bbx(self, predictions: torch.Tensor, idxs: list):
        bbxs = list()
        scores = list()
        for i in range(predictions.shape[0]):
            if predictions[i, constants.positive_label] > 0.5:
                idx_start = idxs[i]
                bbxs.append([idx_start, 0, idx_start +
                            self.window_size, self.freq_max])
                scores.append(predictions[i, constants.positive_label])
        
        
        return torch.tensor(bbxs, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32)

    def apply_nms(self, bbxs: torch.Tensor, scores: torch.Tensor, threshold: float):
        if bbxs.shape[0] == 0:
            return bbxs, scores
        indices = torchvision.ops.nms(bbxs, scores, threshold)
        return bbxs[indices], scores[indices]

    # def batched_forward(self, datas: np.ndarray):
    #     im_idxs = list()
    #     full_bbxs = list()
    #     full_scores = list()
    #     for k, data in enumerate(datas):
    #         result = self.forward(data)
    #         if not result:
    #             continue
    #         bbxs, scores = self.forward(data)
    #         im_idxs.extend([k for _ in range(len(scores))])
    #         #if len(bbxs) != 0:
    #         #    full_bbxs.append(bbxs)
    #         #    full_scores.append(scores)
    #         full_bbxs.append(bbxs)
    #         full_scores.append(scores)
    #     #print(full_bbxs, full_scores)
    #     if len(full_bbxs) == 0:
    #         return None
    #     return torch.tensor(im_idxs), torch.stack(full_bbxs, dim=0), torch.stack(full_scores, dim=0)

    def __call__(self, datas: np.ndarray):
        # if datas.ndim == 2:
        #     return self.batched_forward(datas)
        return self.forward(datas)

