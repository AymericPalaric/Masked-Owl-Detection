import librosa
from scipy import signal
import numpy as np
from scipy.io import wavfile
import maad


def load_audio_file(file_path: str, sr=None) -> tuple[np.ndarray, int]:
  data, fs = librosa.load(file_path, sr=sr, dtype=np.float32,mono=True)
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
    Sxx = 20 * np.log10((Sxx + 1e-8) / np.max(Sxx + 1e-8))
  return freq[1:], time, Sxx[1:]

def clip_audio(data: np.ndarray, fs: int, duration: float) -> np.ndarray:
  if data.shape[0] < duration * fs:
    return data
  return data[:int(duration * fs)]
  
def cwt_roi(s, fs, flims=(1000,3000), tlen=2, th=1e-6):
  df = maad.rois.find_rois_cwt(s, fs, flims, tlen, th)
  return df.iloc[:,np.r_[1,3]].to_numpy()

def save_audio_file(file_path: str, data: np.ndarray, fs: int) -> None:
  wavfile.write(filename=file_path, rate=fs, data=data)
