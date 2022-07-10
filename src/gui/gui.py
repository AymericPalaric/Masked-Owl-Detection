import streamlit as st
import streamlit.components.v1 as components
from src.models.efficientnet import EfficientNet
from src.utils import audio_utils, metrics_utils, transform_utils, torch_utils
from src.pipeline import sliding_window
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
from torch import nn
import os
from csv import DictWriter
import src.constants as constants


# CONSTANTS
MAX_DURATION = 12  # max duration for pipeline on cpu = 15s audios
CLASSIF_MODEL = EfficientNet()
CLASSIF_MODEL_PATH = "trained_models/efficientnet_nosoft_19.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_SLICES = 2
# N_BOXES = 10

# Utils functions


def transform_audio_gui(data):
    _, _, specto = audio_utils.compute_spectrogram(
        data, 24000, nperseg=256, noverlap=256//4, scale="dB")
    # specto = specto[:120, :]
    return np.stack((specto,)*3, axis=0)


@st.cache
def load_pipeline(window_size=None, window_overlap=None):
    classification_model = CLASSIF_MODEL
    classification_model.load_state_dict(
        torch.load(CLASSIF_MODEL_PATH, map_location=DEVICE))
    classification_model.eval()
    window_size = int(
        22000 * 1) if window_size is None else int(window_size*24000)
    window_overlap = int(
        window_size // (5/4)) if window_overlap is None else int(window_size*window_overlap)
    window_type = "boxcar"
    reshape_size = (129, 129)
    audio_transform = transform_utils.baseline_transform
    image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(
        reshape_size), torchvision.transforms.Normalize(mean=constants.mean_test, std=constants.std_test)])
    freq_max = 12000
    treshold_nms = 0.5
    device = DEVICE
    pipeline = sliding_window.SlidingWindowPipeline(
        window_size, window_overlap, window_type, classification_model, audio_transform, image_transform, freq_max, treshold_nms, device)
    time_scale = 1 / (256 - 256//4)

    return pipeline


@st.cache
def cut_audio(audio: np.ndarray, fs, n_slices=None):
    audio_duration = len(audio)/fs

    n_slices = round(
        audio_duration/MAX_DURATION) if n_slices is None else n_slices
    sub_audios = []
    for i in range(n_slices):
        start = int(i*total_len/n_slices)
        end = min(int((i+1)*total_len/n_slices), len(audio))

        sub_audios.append(audio[start:end])

    return sub_audios


@st.cache(allow_output_mutation=True)
def collate_boxes(bbxs: torch.Tensor, scores: torch.Tensor):
    true_bbxs = []
    true_scores = []
    if len(bbxs) >= 2:
        bbxs, indices = torch.sort(bbxs, 0)
        scores = scores[indices[:, 0]]
        s = scores[0]
        i = 0
        while i < len(bbxs)-1:

            x0, y0, x1, y1 = bbxs[i, 0], bbxs[i, 2], bbxs[i+1, 0], bbxs[i+1, 2]
            while y0 > x1 and i < len(bbxs)-1:
                s = max(s, scores[i])
                i += 1
                y0, x1, y1 = y1, bbxs[i, 0], bbxs[i, 2]
            true_scores.append(s)
            true_bbxs.append([x0.item(), 0., y0.item(), 12000.])
        return true_bbxs, true_scores

    else:
        return bbxs, scores


@st.cache(allow_output_mutation=True)
def pred_bbxs(pipeline, slice_audio):
    bbxs, scores = pipeline(slice_audio)
    return collate_boxes(bbxs, scores)


def pipeline_on_slice(pipeline, slice_audio):
    bbxs, scores = pred_bbxs(pipeline, slice_audio)

    slice_spectro = compute_spectro(slice_audio, fs)

    factor = slice_spectro.shape[-1]/len(slice_audio)
    figs, axs = [], []
    for i, box in enumerate(bbxs):
        fig, ax = plt.subplots()

        x0 = int(box[0]*factor)
        x1 = int(box[2]*factor)
        y0 = 0
        y1 = 128
        ax.matshow(slice_spectro[:, x0:x1])

        figs.append(fig)
        axs.append(ax)

    return figs, axs, bbxs, scores


@st.cache
def compute_spectro(audio, fs, nperseg=256, noverlap=256//4, scale="dB"):
    freq, time, spectro = audio_utils.compute_spectrogram(
        audio, fs, nperseg=nperseg, noverlap=noverlap, scale=scale)
    return spectro


@st.cache
def load_audio_spectro(uploaded_audio):
    audio, fs = audio_utils.load_audio_file(uploaded_audio)
    spectro = compute_spectro(audio, fs, nperseg=256,
                              noverlap=256//4, scale="dB")
    return audio, spectro, fs


def save_pos_samples(audios, lbls, fs, uploaded_audio, bbxs, base_absc):
    # Save boxes containing calls and the corresponding CSV
    base_name = uploaded_audio.name[:-4]
    csv = []
    for i in range(len(audios)):
        if lbls[i] == "Positive":
            box = bbxs[i]
            x0 = box[0]/fs
            x1 = box[2]/fs
            audio_utils.save_audio_file(
                f"./data/gui_storage/{base_name}_{int(base_absc[i]/fs+x0)}_{(x1-x0)*1000}.wav", audios[i], fs)

            line = {
                "call_files": f"{base_name}_{int(base_absc[i]/fs+x0)}_{(x1-x0)*1000}.wav",
                "time_stamps": int(base_absc[i]/fs+x0),
                "durations": (x1-x0)*1000
            }
            csv.append(line)

    with open(f"./data/gui_storage/{base_name}.csv", "w") as c:
        w = DictWriter(c, line.keys())
        w.writeheader()
        w.writerows(csv)

    # Remove temporary audios
    temp_files = [file for file in os.listdir(
        './data/gui_storage/') if file.startswith("temp_sub_audio")]
    for i in temp_files:
        os.remove(os.path.join('./data/gui_storage/', i))


# ===================================


# Streamlit App

st.title("Masked Owl Detector")


window_size = st.number_input(
    "Length of the window used to display samples (in seconds)", 0.75*22000/24000, 3*22000/24000, value=22000/24000, step=0.01)
window_overlap = st.number_input(
    "Overlap of the windows (percent of the window size)", int(100*0.1), int(100*5/6), value=int(100*3/4), step=1)/100
score_thresh = st.number_input(
    "Threshold for selecting false positives", 0.85, 0.9999, 0.95, 0.0001)

uploaded_audio = st.file_uploader(
    label='Audio file to analyze', type=['ogg', 'wav'])

# When a file is uploaded
if uploaded_audio is not None:

    audio_load_state = st.text('Processing audio')

    audio, spectro, fs = load_audio_spectro(uploaded_audio)
    pipeline = load_pipeline(window_size, window_overlap)
    figs, axs, boxes, scores = [], [], [], []
    total_len = len(audio)
    prog = st.progress(0)
    sub_audios = cut_audio(audio, fs)
    base_absc = []
    pos_audios = []
    sub_process_prog = st.text(f"{int(0*100/len(sub_audios))} %")
    file_count = 0
    for i, slice_audio in enumerate(sub_audios):
        fig, ax, bbxs, score = pipeline_on_slice(pipeline, slice_audio)
        if len(bbxs) > 0:
            base_absc += [i*len(slice_audio)
                          for j in range(len(bbxs)) if score[j] > score_thresh]
            for j, box in enumerate(bbxs):
                if score[j] > score_thresh:
                    audio_utils.save_audio_file(
                        f"./data/gui_storage/temp_sub_audio_{file_count}.wav", slice_audio[int(box[0]):int(box[2])], fs)
                    file_count += 1
                    pos_audios.append(slice_audio[int(box[0]):int(box[2])])
            boxes += [bbxs[b]
                      for b in range(len(bbxs)) if score[b] > score_thresh]
            scores += [s for s in score if s > score_thresh]
            figs += [fig[b]
                     for b in range(len(fig)) if score[b] > score_thresh]
            axs += [ax[b] for b in range(len(ax)) if score[b] > score_thresh]

        sub_process_prog.text(f"{int((i+1)*100/len(sub_audios))} %")
        prog.progress(int((i+1)*100/len(sub_audios)))

    audio_load_state.text("Processed !")

    if st.checkbox("Show entire raw spectrogtam", value=True):
        fig, ax = plt.subplots()
        x_max, y_max = spectro.shape
        ax.matshow(spectro, aspect=750)  # , extent=[
        # 0, y_max/fs, 0, x_max])
        ax.axes.get_yaxis().set_visible(False)
        factor = spectro.shape[-1]/len(audio)
        for i, box in enumerate(boxes):
            x0 = int(box[0]*factor) + int(base_absc[i]*factor)
            x1 = int(box[2]*factor) + int(base_absc[i]*factor)
            y0 = 0
            y1 = 128
            ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                         fill=False, edgecolor='red', linewidth=1))
        st.pyplot(fig)

        st.audio(uploaded_audio, format='audio/wav')

    open_all = st.checkbox("Open All", key="open_all", value=True)
    lbls = [0 for i in range(len(boxes))]
    for i in range(len(boxes)):
        """
        **____________________________________________________________________________**
        """
        if st.checkbox(f"Sample {i+1}", key=f"bttn_{i}", value=open_all):
            st.pyplot(figs[i])
            radio_lbl = st.radio("Label of the sample", [
                "Negative", "Positive"], index=1, horizontal=True, key=f"radio_subbox_{i}")
            lbls[i] = radio_lbl
            st.text(f"Confidence: {scores[i]}")
            st.audio(
                f"./data/gui_storage/temp_sub_audio_{i}.wav")

    if st.button("Save samples"):
        save_pos_samples(pos_audios, lbls, fs,
                         uploaded_audio, boxes, base_absc)

        st.text("Saved !")
