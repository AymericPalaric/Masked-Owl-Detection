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
STORAGE_PATH = args.output_path + \
    "/" if args.output_path[-1] != "/" else args.output_path
INPUT_PATH = args.input_path + \
    "/" if args.input_path[-1] != "/" else args.input_path
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


@st.cache(suppress_st_warning=True)
def render_boxes(audio: AudioFile, i, score):
    text = f"Confidence: {score}"
    image = STORAGE_PATH + f"{audio.name}_spec_{i}.png"
    audio = STORAGE_PATH+f"temp_sub_audio_{audio.name}_{i}.wav"
    return text, image, audio


@st.cache
def load_temp_files(audio: AudioFile):
    boxes = np.load(
        f"{STORAGE_PATH}{audio.name}_boxes.npy", allow_pickle=True)
    scores = np.load(
        f"{STORAGE_PATH}{audio.name}_scores.npy", allow_pickle=True)
    base_absc = np.load(
        f"{STORAGE_PATH}{audio.name}_base_absc.npy", allow_pickle=True)

    calls_spec = [i for i in os.listdir(STORAGE_PATH) if i.startswith(
        f"{audio.name}_spec")]
    calls_spec.sort()
    temp_audios = [i for i in os.listdir(STORAGE_PATH) if i.startswith(
        f"temp_sub_audio_{audio.name}_")]
    temp_audios.sort()

    return boxes, scores, base_absc, calls_spec, temp_audios


def save_pos_samples(audios, lbls, neg_names: list[str], fs, uploaded_audio: AudioFile, bbxs, base_absc):

    base_name = uploaded_audio.name
    csv = []
    for i in range(len(audios)):
        # Save boxes containing positive calls and the corresponding CSV file
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
                "durations": (x1-x0)*1000,
                "label": "masked_owl"
            }
            csv.append(line)

        # Save negative ones
        if lbls[i] == "Negative" and neg_names[i] != "":
            neg_lbl = neg_names[i].replace(" ", "_")
            neg_save_path = STORAGE_PATH + neg_lbl + "/"
            if not os.path.exists(neg_save_path):
                os.mkdir(neg_save_path)
            box = bbxs[i]
            x0 = box[0]/fs
            x1 = box[2]/fs
            audio, fs = audio_utils.load_audio_file(
                os.path.join(STORAGE_PATH, audios[i]), fs)
            audio_utils.save_audio_file(
                f"{neg_save_path}{base_name}_{int(base_absc[i]/fs+x0)}_{(x1-x0)*1000}.wav", audio, fs)

            line = {
                "call_files": f"{base_name }_{int(base_absc[i]/fs+x0)}_{(x1-x0)*1000}.wav",
                "time_stamps": int(base_absc[i]/fs+x0),
                "durations": (x1-x0)*1000,
                "label": neg_lbl
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
    boxes, scores, base_absc, calls_spec, temp_audios = load_temp_files(
        uploaded_audio)

    if st.checkbox("Show entire raw spectrogtam", value=True, key=f"raw_spectro_{u+1}"):

        st.image(STORAGE_PATH +
                 f"{uploaded_audio.name}_raw_spec.png", width=1000)

        st.audio(uploaded_audio.path, format='audio/wav')

    n_calls = len(calls_spec)
    lbls = [0 for i in range(n_calls)]
    neg_names = ["" for i in range(n_calls)]
    for i in range(n_calls):
        """
        **____________________________________________________________________________**
        """
        if st.checkbox(f"Sample {i+1}", key=f"bttn_{u}_{i}", value=True):
            # st.pyplot(figs[i])
            text, image, audio_ = render_boxes(uploaded_audio, i, scores[i])
            st.text(text)
            st.image(image, width=800)
            st.audio(audio_)
            radio_lbl = st.radio("Label of the sample", [
                "Negative", "Positive"], index=1, horizontal=True, key=f"radio_subbox_{u}_{i}")
            lbls[i] = radio_lbl

            if lbls[i] == "Negative":
                neg_names[i] = st.text_input(
                    "Name of the call", key=f"neg_name_{u}_{i}", value="")

    if n_calls > 0:
        if st.button(f"Save samples for record number {u+1}"):
            save_pos_samples(temp_audios, lbls, neg_names, FS,
                             uploaded_audio, boxes, base_absc)

            st.text(f"Saved samples from audio {uploaded_audio.name} !")


if st.button("Quit the app"):
    # Remove temporary audios
    temp_files = [file for file in os.listdir(
        STORAGE_PATH) if file.startswith(f"temp_sub_audio_") or file.endswith(".npy") or file.endswith(".png")]
    for file in temp_files:
        os.remove(os.path.join(STORAGE_PATH, file))
    st.stop()
