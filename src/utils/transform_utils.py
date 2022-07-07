from . import audio_utils
import numpy as np

def baseline_transform(data):
  _, _, specto = audio_utils.compute_spectrogram(data, 24000, nperseg=256, noverlap=256//4, scale="dB")
  return np.stack((specto,)*3, axis=0)


def compute_mel_spectrogram(data: np.ndarray, fs: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
  mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=fs//2)
  mel_spectrogram = librosa.display.specshow(librosa.power_to_db(mel_spectrogram))
  return np.stack((mel_spectrogram)*3, axis=0)