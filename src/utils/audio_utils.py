import librosa
from scipy import signal
import numpy as np
from scipy.io import wavfile

def load_audio_file(file_path: str) -> tuple[np.ndarray, int]:
  data, fs = librosa.load(file_path, sr=None, dtype=np.float32)
  return data, fs

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
    Sxx = 20 * np.log10(Sxx)
  return freq[1:], time, Sxx[1:]

def compute_mel_spectrogram(data: np.ndarray, fs: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
  mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=fs//2)
  return mel_spectrogram

def quick_check_for_call(data: np.ndarray, fs: int, tresh: float = 1e-7) -> bool:
  freq, time, Sxx = compute_spectrogram(data, fs, nperseg=256, noverlap=256//4)
  idx_1500 = np.argmin(np.abs(freq - 1500))
  idx_2500 = np.argmin(np.abs(freq - 2500))
  mean = Sxx[idx_1500:idx_2500].mean()
  return mean > tresh

def save_audio_file(file_path: str, data: np.ndarray, fs: int) -> None:
  wavfile.write(filename=file_path, rate=fs, data=data)
