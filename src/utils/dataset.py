from src.utils.audio_utils import load_audio_file, padding_audio, cwt_roi, compute_mel_spectrogram
from venv import create
import tensorflow as tf
import torch
import torchvision
import numpy as np
from librosa.util import fix_length
import src.utils.path_utils as pu
import src.config as config
import src.constants as constants
import os
import src.utils.audio_utils as au
from tqdm import tqdm


class ClassifDataset(torch.utils.data.Dataset):
    """
    Dataset for classification
    """

    def __init__(self, positive_path, negative_path, hard_path, mean, std, transform_audio):
        self.positive_path = positive_path
        self.negative_path = negative_path
        self.hard_path = hard_path
        self.positive_files = [file for file in os.listdir(
            self.positive_path) if file.endswith(".wav")]
        self.negative_files = [file for file in os.listdir(
            self.negative_path) if file.endswith(".wav")]
        self.hard_files = [file for file in os.listdir(
            self.hard_path) if file.endswith(".wav")]
        self.reshape_size = (129, 229)

        self.transform_audio = transform_audio
        self.transform_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
        ), torchvision.transforms.Resize(self.reshape_size), torchvision.transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.positive_files) + len(self.negative_files) + len(self.hard_files)

    def __getitem__(self, idx):
        if idx < len(self.positive_files):
            file_path = os.path.join(
                self.positive_path, self.positive_files[idx])
            data, fs = au.load_audio_file(file_path)
            x = self.transform_audio(data)
            x = self.transform_image(x)
            return x, 1
        elif idx < len(self.positive_files) + len(self.negative_files):
            file_path = os.path.join(
                self.negative_path, self.negative_files[idx - len(self.positive_files)])
            data, fs = au.load_audio_file(file_path)
            x = self.transform_audio(data)
            x = self.transform_image(x)
            return x, 0
        else:
            file_path = os.path.join(
                self.hard_path, self.hard_files[idx - len(self.positive_files) - len(self.negative_files)])
            data, fs = au.load_audio_file(file_path)
            x = self.transform_audio(data)
            x = self.transform_image(x)
            return x, 0


def create_dataset(train_test: bool):

    def transform_audio(data):
        _, _, specto = au.compute_spectrogram(
            data, 24000, nperseg=256, noverlap=256//4, scale="dB")
        # freq clip
        specto = specto[:120, :]
        return np.stack((specto,)*3, axis=-1)

    dataset = ClassifDataset(
        positive_path=pu.get_train_test_path(
            pu.get_positive_samples_path(), train_test),
        negative_path=pu.get_train_test_path(
            pu.get_negative_samples_path(), train_test),
        hard_path=pu.get_train_test_path(
            pu.get_hard_samples_path(), train_test),
        mean=0.,
        std=1.,
        transform_audio=transform_audio)

    return dataset


def transform_audio(data):
    _, _, specto = compute_mel_spectrogram(
        data, 24000, n_fft=1024, hop_length=512, n_mels=128)
    # freq clip
    specto = specto[:120, :]
    return specto


class SlidingDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dir, transform_audio, window_size, stride):

        self.tensor_directory = raw_dir
        self.transform_audio = transform_audio
        self.reshape_size = (129, 229)
        self.transform_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(self.reshape_size)])
        self.files = os.listdir(raw_dir)
        self.window_size = window_size
        self.stride = stride
        self.data_tuples = []

        for f in self.files:
            file = os.path.join(raw_dir, f)
            data, fs = load_audio_file(file)
            # pad with zeros with tensor is not of right length
            data = padding_audio(data, self.window_size)
            idxs = [i for i in range(
                0, data.size - self.window_size, self.stride)]
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

    def __init__(self, raw_dir, transform_audio, filter=(1000, 3000), len=2, threshold=1e-6):
        self.tensor_directory = raw_dir
        self.transform_audio = transform_audio
        self.reshape_size = (129, 229)
        self.transform_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
        self.files = os.listdir(raw_dir)
        self.data_tuples = []
        self.filter = filter
        self.len = len
        self.threshold = threshold

        for f in self.files:
            file = os.path.join(raw_dir, f)
            data, fs = load_audio_file(file)
            # pad with zeros with tensor is not of right length
            idxs = cwt_roi(s=data, fs=fs)
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
