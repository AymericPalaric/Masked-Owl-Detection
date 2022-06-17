from . import audio_utils

def baseline_transform(data):
  _, _, specto = audio_utils.compute_spectrogram(data, 24000, nperseg=256, noverlap=256//4, scale="dB")
  return specto
