from src.utils import audio_utils
import numpy as np


def baseline_transform(data):
    """
    Baseline transform.

    Args:
    -----
      - data: numpy.ndarray, audio data.

    Returns:
    --------
      - x: numpy.ndarray, transformed audio data.
    """
    _, _, specto = audio_utils.compute_spectrogram(
        data, 24000, nperseg=256, noverlap=256//4, scale="dB")

    return np.stack((specto,)*3, axis=-1)


def mel_transform(data):
    specto = audio_utils.compute_mel_spectrogram(data, 24000, n_mels=128, n_fft=1024,
                                                 hop_length=256, fmin=0, fmax=12000)
    return np.stack((specto,)*3, axis=-1)
