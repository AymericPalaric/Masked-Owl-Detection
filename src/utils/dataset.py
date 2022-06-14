import torch
import torchvision
from audio_utils import load_audio_file,padding_audio,cwt_roi,compute_mel_spectrogram
import os
import numpy as np

def transform_audio(data):
    _, _, specto = compute_mel_spectrogram(data, 24000, n_fft=1024,hop_length=512,n_mels=128)
    # freq clip
    specto = specto[:120, :]
    return specto


class SlidingDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dir, transform_audio, window_size, stride):

        self.tensor_directory = raw_dir
        self.transform_audio = transform_audio
        self.reshape_size = (129, 229)
        self.transform_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(self.reshape_size)])
        self.files = os.listdir(raw_dir)
        self.window_size = window_size
        self.stride = stride
        self.data_tuples = []

        for f in self.files:
            file = os.path.join(raw_dir, f)
            data, fs = load_audio_file(file)
            # pad with zeros with tensor is not of right length
            data = padding_audio(data, self.window_size)
            idxs = [i for i in range(0, data.size - self.window_size, self.stride)]
            if len(idxs) == 0:
                continue
            for j in idxs:
                data_tuple = (file, j)
                self.data_tuples.append(data_tuple)

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        sample_tuple = self.data_tuples[idx]
        sample, _ = load_audio_file(sample_tuple[0])
        sample = sample[sample_tuple[1]: sample_tuple[1] + self.window_size]
        sample = self.transform_audio(sample)
        sample = self.transform_image(sample)
        return {'sample': sample, 'file': sample_tuple[0], 'index': sample_tuple[1]}

class ROIDataset(torch.utils.data.Dataset):
    def mix_to_mono(self, sample):
        if len(sample.shape) == 2:
            return sample
        else:
            return np.mean(sample, axis=1)

    def __init__(self, raw_dir, transform_audio, filter=(1000,3000), len=2, threshold=1e-6):
        self.tensor_directory = raw_dir
        self.transform_audio = transform_audio
        self.reshape_size = (129, 229)
        self.transform_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.files = os.listdir(raw_dir)
        self.data_tuples = []
        self.filter=filter
        self.len=len
        self.threshold=threshold

        for f in self.files:
            file = os.path.join(raw_dir, f)
            data, fs = load_audio_file(file)
            # pad with zeros with tensor is not of right length
            idxs=cwt_roi(s=data,fs=fs)
            for idx in idxs:
                data_tuple = (file, idx)
                self.data_tuples.append(data_tuple)


    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        sample_tuple = self.data_tuples[idx]
        sample, fs = load_audio_file(sample_tuple[0])
        sample = sample[int(sample_tuple[1][0]*fs):int(sample_tuple[1][1]*fs)]
        sample = self.transform_audio(sample)
        sample = self.transform_image(sample)
        return {'sample': sample, 'file': sample_tuple[0], 'index': sample_tuple[1]}

    