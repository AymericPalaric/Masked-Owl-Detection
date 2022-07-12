import streamlit as st
from src.utils import audio_utils
import numpy as np
import os
from csv import DictWriter
import argparse


# CONSTANTS
N_SLICES = 2

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()
STORAGE_PATH = args.output_path
INPUT_PATH = args.input_path
FS = 24000

# Utils functions

# Base audio instance


class AudioFile():
    """
    Audio instance

    Parameters
    ----------
    path : str
        Path to the audio file

    Attributes
    ----------
    name : str
        Name of the audio file
    path : str
        Path to the audio file
    audio : np.array
        Audio signal
    fs : int
        Sampling frequency
    duration : float
        Duration of the audio
    audio_length : int
        Length of the audio signal (number of samples)
    """

    def __init__(self, path):
        self.name = path.split("/")[-1][:-4]
        self.path = path
        self.audio, self.fs = audio_utils.load_audio_file(path, FS)
        self.duration = len(self.audio)/self.fs
        self.audio_length = len(self.audio)

# ===================================


def save_pos_samples(audios, lbls, fs, uploaded_audio: AudioFile, bbxs, base_absc):
    # Save boxes containing calls and the corresponding CSV
    base_name = uploaded_audio.name
    csv = []
    for i in range(len(audios)):
        if lbls[i] == "Positive":
            box = bbxs[i]
            x0 = box[0]/fs
            x1 = box[2]/fs
            audio, fs = audio_utils.load_audio_file(
                os.path.join(STORAGE_PATH, audios[i]), fs)
            audio_utils.save_audio_file(
                f"{STORAGE_PATH}{base_name}_{int(base_absc[i]/fs+x0)}_{(x1-x0)*1000}.wav", audio, fs)

            line = {
                "call_files": f"{base_name }_{int(base_absc[i]/fs+x0)}_{(x1-x0)*1000}.wav",
                "time_stamps": int(base_absc[i]/fs+x0),
                "durations": (x1-x0)*1000
            }
            csv.append(line)

    with open(f"{STORAGE_PATH}{base_name}.csv", "w") as c:
        w = DictWriter(c, line.keys())
        w.writeheader()
        w.writerows(csv)

    return csv


# ===================================
# Streamlit App
st.title("Masked Owl Detector")
"""
This app is a tool to detect masked owl calls in audio files.
"""


# uploaded_audios = st.file_uploader(
#     label='Audio file to analyze', type=['ogg', 'wav'], accept_multiple_files=True)
uploaded_audios = [AudioFile(os.path.join(INPUT_PATH, file))
                   for file in os.listdir(INPUT_PATH) if file.endswith(".wav")]

# When a file is uploaded
for u, uploaded_audio in enumerate(uploaded_audios):
    """
    --------------------------------------------------
    """
    f"""
    ## Audio number {u+1} ({uploaded_audio.name}):
    """
    boxes = np.load(
        f"{STORAGE_PATH}{uploaded_audio.name}_boxes.npy", allow_pickle=True)
    scores = np.load(
        f"{STORAGE_PATH}{uploaded_audio.name}_scores.npy", allow_pickle=True)
    base_absc = np.load(
        f"{STORAGE_PATH}{uploaded_audio.name}_base_absc.npy", allow_pickle=True)

    if st.checkbox("Show entire raw spectrogtam", value=True, key=f"raw_spectro_{u+1}"):

        st.image(STORAGE_PATH +
                 f"{uploaded_audio.name}_raw_spec.png", width=1000)

        st.audio(uploaded_audio.path, format='audio/wav')
    calls_spec = [i for i in os.listdir(STORAGE_PATH) if i.startswith(
        f"{uploaded_audio.name}_spec")]
    calls_spec.sort()
    temp_audios = [i for i in os.listdir(STORAGE_PATH) if i.startswith(
        f"temp_sub_audio_{uploaded_audio.name}_")]
    temp_audios.sort()
    # open_all = st.checkbox("Open All", key="open_all", value=True)
    n_calls = len(calls_spec)
    lbls = [0 for i in range(n_calls)]
    for i in range(n_calls):
        """
        **____________________________________________________________________________**
        """
        if st.checkbox(f"Sample {i+1}", key=f"bttn_{u}_{i}", value=True):
            # st.pyplot(figs[i])

            st.text(f"Confidence: {scores[i]}")
            st.image(STORAGE_PATH +
                     f"{uploaded_audio.name}_spec_{i}.png", width=800)
            st.audio(
                STORAGE_PATH+f"temp_sub_audio_{uploaded_audio.name}_{i}.wav")
            radio_lbl = st.radio("Label of the sample", [
                "Negative", "Positive"], index=1, horizontal=True, key=f"radio_subbox_{u}_{i}")
            lbls[i] = radio_lbl

    if st.button(f"Save samples for record number {u+1}"):
        save_pos_samples(temp_audios, lbls, FS,
                         uploaded_audio, boxes, base_absc)

        st.text(f"Saved samples from audio {uploaded_audio.name} !")


if st.button("Quit the app"):
    # Remove temporary audios
    temp_files = [file for file in os.listdir(
        STORAGE_PATH) if file.startswith(f"temp_sub_audio_") or file.endswith(".npy") or file.endswith(".png")]
    for file in temp_files:
        os.remove(os.path.join(STORAGE_PATH, file))
    quit()
