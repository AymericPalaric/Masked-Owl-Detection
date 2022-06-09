import librosa
from scipy import signal
import numpy as np
from scipy.io import wavfile

def load_audio_file(file_path: str, sr=None) -> tuple[np.ndarray, int]:
  data, fs = librosa.load(file_path, sr=sr, dtype=np.float32)
  return data, fs

def padding_audio(data: np.ndarray, window_size:int) -> np.ndarray:
  if  data.size % window_size != 0:
    data=librosa.util.fix_length(data, data.size + window_size - data.size % window_size)
  return data

def compute_spectrogram(data: np.ndarray, fs: int, nperseg: int, noverlap: int, scale: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  if scale not in ('linear', 'dB'):
    raise ValueError(f"scale must be either 'linear' or 'dB' got {scale}")

  freq, time, Sxx = signal.spectrogram(
      x=data, 
      fs=fs, 
      nperseg=nperseg, 
      window=signal.windows.hann(nperseg), 
      noverlap=noverlap, 
      scaling='spectrum',
      mode='magnitude')
  if scale == 'dB':
    Sxx = 20 * np.log10((Sxx + 1e-8) / np.max(Sxx + 1e-8))
  return freq[1:], time, Sxx[1:]

def compute_mel_spectrogram(data: np.ndarray, fs: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
  mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=fs//2)
  return mel_spectrogram

def clip_audio(data: np.ndarray, fs: int, duration: float) -> np.ndarray:
  if data.shape[0] < duration * fs:
    return data
  return data[:int(duration * fs)]



def save_audio_file(file_path: str, data: np.ndarray, fs: int) -> None:
  wavfile.write(filename=file_path, rate=fs, data=data)
