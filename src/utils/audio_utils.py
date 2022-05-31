import librosa
from scipy import signal
import numpy as np
from scipy.io import wavfile

def normalize(x: np.ndarray) -> np.ndarray:
  return (x - np.min(x)) / (np.max(x) - np.min(x))

def load_audio_file(file_path: str) -> tuple[np.ndarray, int]:
  data, fs = librosa.load(file_path)
  return normalize(data), fs

def compute_spectrogram(data: np.ndarray, fs: int, nperseg: int, noverlap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  freq, time, Sxx = signal.spectrogram(x=data, fs=fs, nperseg=nperseg, noverlap=noverlap)
  return freq, time, Sxx

def quick_check_for_call(data: np.ndarray, fs: int) -> bool:
  freq, time, Sxx = compute_spectrogram(data, fs, nperseg=256, noverlap=256//4)
  mean = Sxx[15:30].mean()
  return mean > 1e-7

def save_audio_file(file_path: str, data: np.ndarray, fs: int) -> None:
  wavfile.write(filename=file_path, rate=fs, data=data)
