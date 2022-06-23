import streamlit as st
from src.models.efficientnet import EfficientNet
from src.utils import audio_utils
from src.utils import transform_utils
import matplotlib.pyplot as plt
import torchvision
import torch
from torch import nn
import src.constants as constants

# Testing vars TO CHANGE WHEN DETECTION COMPLETED
N_BOXES = 5
CLASSIF_MODEL_PATH = "./trained_models/efficientnet-b0_29.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Utils functions


# Streamlit App

st.title("Masked Owl Detector")

uploaded_audio = st.file_uploader(
    label='Audio file to analyze', type=['ogg', 'wav'])
thresh = st.number_input(
    "Value of the threshold used to look for potential boxes")
fs = st.number_input("Value of the sampling frequency to use")
win_len = st.number_input("Length of the window used to display samples")
if uploaded_audio is not None:
    audio_load_state = st.text('Processing audio')
    fig, ax = plt.subplots()
    audio, fs = audio_utils.load_audio_file(uploaded_audio)
    freq, time, spectro = audio_utils.compute_spectrogram(
        audio, fs, nperseg=256, noverlap=256//4, scale="dB")
    ax.matshow(spectro)
    model = EfficientNet()
    model.load_state_dict(torch.load(
        CLASSIF_MODEL_PATH, map_location=torch.device(DEVICE)))
    reshape_size = (129, 129)
    mean, std = constants.mean_test, constants.std_test
    transform_audio = transform_utils.baseline_transform
    transform_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
    ), torchvision.transforms.Resize(reshape_size), torchvision.transforms.Normalize(mean=mean, std=std)])
    x = transform_audio(audio)
    x = transform_image(x)
    pred = model(x.reshape((1, *x.shape)))
    pred = pred.argmax(dim=1)
    audio_load_state.text("Processed !")

    if st.checkbox("Show entire raw spectrogtam"):
        st.pyplot(fig)

        st.text(pred)

    for i in range(N_BOXES):
        if st.checkbox(f"Box number {i+1}"):
            st.text(f"Activated box number {i+1}")
