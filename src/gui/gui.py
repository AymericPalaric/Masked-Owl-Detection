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
import src.constants as constants

# Testing vars TO CHANGE WHEN DETECTION COMPLETED

N_BOXES = 5
# CLASSIF_MODEL_PATH = "./trained_models/efficientnet_detek_19.pt"
CLASSIF_MODEL_PATH = "./trained_models/efficientnet-b0_29.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Detection part
classification_model = EfficientNet()
classification_model.load_state_dict(torch.load(
    CLASSIF_MODEL_PATH, map_location=torch.device(DEVICE)))
classification_model.eval()

window_size = int(22000 * 0.5)
window_overlap = window_size // 4
window_type = "hann"
reshape_size = (129, 129)
audio_transform = transform_utils.baseline_transform
image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
), torchvision.transforms.Resize(reshape_size), torchvision.transforms.Normalize(mean=-0.73, std=2.383)])
freq_max = np.inf
treshold_nms = 0.5
pipeline = sliding_window.SlidingWindowPipeline(
    window_size, window_overlap, window_type, classification_model, audio_transform, image_transform, freq_max, treshold_nms, DEVICE)
time_scale = 1 / (256 - 256//4)
# files = [file[:-4] for file in os.listdir("data/detection_samples/train") if file.endswith(".wav")][:5]
# audios = list()
# img_target = list()
# bbxs_target = list()
# for k, file in enumerate(files):
#     audio_path = os.path.join("data/detection_samples/train", f"{file}.wav")
#     bbxs_path = os.path.join("data/detection_samples/train", f"{file}.npy")
#     audio, _ = audio_utils.load_audio_file(audio_path)
#     bbxs = np.load(bbxs_path)
#     audios.append(audio)
#     for bbx in bbxs:
#         if bbx[0] == 1:
#             img_target.append(k)
#             bbxs_target.append([int(bbx[1] * time_scale), 0, int(bbx[2] * time_scale), freq_max])
# audios = np.stack(audios)
# img_target = torch.tensor(img_target, dtype=torch.long)
# bbxs_target = torch.tensor(bbxs_target, dtype=torch.long)
# img_pred, bbxs_pred, score_pred = pipeline(audios)


# Utils functions


def transform_audio_gui(data):
    _, _, specto = audio_utils.compute_spectrogram(
        data, 24000, nperseg=256, noverlap=256//4, scale="dB")
    # specto = specto[:120, :]
    return np.stack((specto,)*3, axis=0)


@st.cache
def load_model():
    model = EfficientNet()
    model.load_state_dict(torch.load(
        CLASSIF_MODEL_PATH, map_location=torch.device(DEVICE)))
    reshape_size = (129, 129)
    mean, std = constants.mean_test, constants.std_test
    transform_audio = transform_utils.baseline_transform
    transform_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
    ), torchvision.transforms.Resize(reshape_size), torchvision.transforms.Normalize(mean=mean, std=std)])
    return model, reshape_size, transform_audio, transform_image


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


@st.cache
def cut_audio(audio, fs, duration=1, name="sub_audio_cut"):
    p = duration * fs
    n = len(audio)
    sub_audios = []
    offset = 0
    while offset + p < n:
        cut = audio[offset:offset+p]
        sub_audios.append(cut)
        offset += p
        audio_utils.save_audio_file(
            f"./data/gui_storage/{name}_{len(sub_audios)-1}.wav", cut, fs)
    return sub_audios


@st.cache
def model_pred(audio, model):
    transform_audio = transform_utils.baseline_transform
    x = transform_audio(audio)
    mean, std = np.mean(x), np.std(x)

    transform_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
    ), torchvision.transforms.Resize(reshape_size), torchvision.transforms.Normalize(mean=mean, std=std)])
    with torch.no_grad():
        x = transform_audio(audio)
        x = transform_image(x)
        x = x.numpy()
        x = torch.tensor([x]).to(DEVICE)
        pred = model(x)
        print(pred)
        pred[pred >= thresh] = 1
        pred[pred < thresh] = 0

        pred = torch.argmax(pred, dim=1)
    return pred


def sub_audio_process(audio_i, fs, model):
    spectro_i = compute_spectro(
        audio_i, fs, nperseg=256, noverlap=256//4, scale="dB")
    fig_i, ax_i = plt.subplots()
    ax_i.matshow(spectro_i, aspect=2)
    pred = model_pred(audio, model).numpy()

    return spectro_i, fig_i, ax_i, pred


def save_pos_samples(audios, lbls, fs, save_name):
    for i in range(len(audios)):
        if lbls[i] == "Positive":
            audio_utils.save_audio_file(
                f"./data/gui_storage/{save_name}_{i}.wav", audios[i], fs)


# ===================================
# ===================================

# Streamlit App
model, reshape_size, transform_audio, transform_image = load_model()
transform_audio = transform_utils.baseline_transform
st.title("Masked Owl Detector")

uploaded_audio = st.file_uploader(
    label='Audio file to analyze', type=['ogg', 'wav'])
thresh = st.number_input(
    "Value of the threshold used to look for potential boxes", value=0.5, min_value=0.1, max_value=0.99)
ol = st.number_input("Overlap of the windows")
win_len = st.number_input("Length of the window used to display samples")
save_name = st.text_input(
    "Basename of the samples used to save the found windows")

# When a file is uploaded
if uploaded_audio is not None:
    audio_load_state = st.text('Processing audio')

    audio, spectro, fs = load_audio_spectro(uploaded_audio)

    bbxs_pred, score_pred = pipeline(torch.tensor(audio, dtype=torch.long))
    st.text([bbxs_pred, score_pred])
    sub_audios = cut_audio(audio, fs, duration=5)
    sub_spectros = []
    figs, axs, preds = [], [], []
    progress_bar = st.progress(0.0)
    for i in range(len(sub_audios)):
        audio_i, fig_i, ax_i, pred = sub_audio_process(
            sub_audios[i], fs, model)
        figs.append(fig_i)
        axs.append(ax_i)
        preds.append(pred)
        progress_bar.progress(i/len(sub_audios))
    progress_bar.progress(1.0)
    N_BOXES = len(sub_audios)

    fig, ax = plt.subplots()
    ax.matshow(spectro, aspect=20)

    pred = model_pred(audio, model)
    st.text(pred)
    audio_load_state.text("Processed !")

    if st.checkbox("Show entire raw spectrogtam"):
        st.pyplot(fig)
        st.audio(uploaded_audio, format='audio/wav')
        st.text(pred)

    open_all = st.checkbox("Open All", key="open_all")

    # st.session_state.open_all = open_all

    boxes = []
    lbls = [0 for i in range(N_BOXES)]
    for i in range(N_BOXES):
        background_i = "<style>:root {background-color: #DD3300;}</style>"
        compo_i = components.html(background_i)
        radio_box_i = compo_i.checkbox(
            f"Box number {i+1}", key=f"sample_box_{i}", value=open_all)
        boxes.append(radio_box_i)
        fig_i, ax_i = figs[i], axs[i]

        pred = preds[i]

        if radio_box_i:
            st.text(f"Activated box number {i+1}")

            st.pyplot(fig_i)
            st.audio(
                f"./data/gui_storage/sub_audio_cut_{i}.wav", format='audio/wav')

            radio_lbl = st.radio("Label of the sample", [
                "Negative", "Positive"], index=int(pred[0]), horizontal=True, key=f"radio_subbox_{i}")
            st.text(radio_lbl)
            lbls[i] = radio_lbl

    # save_bttn = st.button("Save samples", on_click=lambda _: save_pos_samples(
    #     sub_audios, lbls, fs, save_name, _))

    if st.button("Save samples"):
        save_pos_samples(sub_audios, lbls, fs, save_name)
        st.text(lbls)
        st.text("Saved !")
