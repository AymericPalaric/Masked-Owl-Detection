import librosa
from scipy import signal
import numpy as np
from scipy.io import wavfile
import maad


def load_audio_file(file_path: str, sr=None) -> tuple[np.ndarray, int]:
    """
    Loads a wav file as numpy array.

    Args:
    -----
        - file_path (str): path towards the wav file to load;
        - sr (float): sampling rate to use if resampling audio

    Returns:
    --------
        - data (np.ndarray): wav file loaded, shape=`(n_samples,)`
        -
    """
    data, fs = librosa.load(file_path, sr=sr, dtype=np.float32, mono=True)
    return data, fs


def padding_audio(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Pads the array given to make it match a whole number of the same window.

    Args:
    -----
        - data (np.ndarray): data to pad
        - window_size (int): number of samples corresponding to the window length

    Returns:
    --------
        - data (np.ndarray): padded data
    """
    if data.size % window_size != 0:
        data = librosa.util.fix_length(
            data, data.size + window_size - data.size % window_size)
    return data


def compute_spectrogram(data: np.ndarray, fs: int, nperseg: int, noverlap: int, scale: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes classic spectrogram.

    Args:
    -----
        - data (np.ndarray): array of the audio, shape = `(n_samples,)`
        - fs (int): sampling frequency
        - nperseg (int): length of each segment
        - noverlap (int): number of overlapping samples
        - scale (str): 'linear' or 'dB'

    Returns:
    --------
        - (freq, time, Sxx): lists of frequencies and times, and matrix of intensities for each pair of freq and time.
    """

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


def compute_mel_spectrogram(data: np.ndarray, fs: int, n_mels: int, n_fft: int, hop_length: int, fmin: int, fmax: int) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=data, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return mel_spectrogram


def clip_audio(data: np.ndarray, fs: int, duration: float) -> np.ndarray:
    if data.shape[0] < duration * fs:
        return data
    return data[:int(duration * fs)]


def cwt_roi(s, fs, flims=(1500, 4000), tlen=2, th=1e-7):
    df = maad.rois.find_rois_cwt(s, fs, flims, tlen, th)
    rois = df.iloc[:, np.r_[1, 3]].to_numpy()
    rois =[np.int32(roi*fs) for roi in rois]
    return rois


def save_audio_file(file_path: str, data: np.ndarray, fs: int) -> None:
    """
    Saves the given as a wav file.

    Args:
    -----
        - file_path (str): path in which to store the saved wav file
        - data (np.ndarray): audio array
        - fs (int): sampling frequency
    """
    wavfile.write(filename=file_path, rate=fs, data=data)
