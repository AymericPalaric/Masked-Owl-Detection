import streamlit as st
import streamlit.components.v1 as components
from src.models.efficientnet import EfficientNet
from src.utils import audio_utils, metrics_utils, transform_utils, torch_utils
from src.pipeline import sliding_window
from src.pipeline import roi_window
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
STORAGE_PATH = "./data/gui_storage/detection_testing/"
FS = 24000
# N_BOXES = 10

# Utils functions


def save_pos_samples(audios, lbls, fs, uploaded_audio, bbxs, base_absc):
    # Save boxes containing calls and the corresponding CSV
    base_name = uploaded_audio.name[:-4]
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

    # Remove temporary audios
    temp_files = [file for file in os.listdir(
        STORAGE_PATH) if file.startswith(f"temp_sub_audio_{base_name}")]
    for file in temp_files:
        os.remove(os.path.join(STORAGE_PATH, file))

    return csv
    for i in temp_files:
        os.remove(os.path.join(STORAGE_PATH, i))


# ===================================


# Streamlit App

st.title("Masked Owl Detector")


# window_size = st.number_input(
#     "Length of the window used to display samples (in seconds)", 0.75*22000/24000, 3*22000/24000, value=22000/24000, step=0.01)
# window_overlap = st.number_input(
#     "Overlap of the windows (percent of the window size)", int(100*0.1), int(100*5/6), value=int(100*3/4), step=1)/100
# score_thresh = st.number_input(
#     "Threshold for selecting false positives", 0.85, 0.9999, 0.95, 0.0001)

uploaded_audios = st.file_uploader(
    label='Audio file to analyze', type=['ogg', 'wav'], accept_multiple_files=True)


# When a file is uploaded
for u, uploaded_audio in enumerate(uploaded_audios):

    # audio_load_state = st.text('Processing audio')

    # audio, spectro, fs = load_audio_spectro(uploaded_audio)
    # pipeline = load_pipeline(window_size, window_overlap)
    # figs, axs, boxes, scores = [], [], [], []
    # total_len = len(audio)
    # prog = st.progress(0)
    # sub_audios = cut_audio(audio, fs)
    # base_absc = []
    # pos_audios = []
    # sub_process_prog = st.text(f"{int(0*100/len(sub_audios))} %")
    # file_count = 0
    # for i, slice_audio in enumerate(sub_audios):
    #     fig, ax, bbxs, score = pipeline_on_slice(pipeline, slice_audio)
    #     if len(bbxs) > 0:
    #         base_absc += [i*len(slice_audio)
    #                       for j in range(len(bbxs)) if score[j] > score_thresh]
    #         for j, box in enumerate(bbxs):
    #             if score[j] > score_thresh:
    #                 audio_utils.save_audio_file(
    #                     f"./data/gui_storage/temp_sub_audio_{file_count}.wav", slice_audio[int(box[0]):int(box[2])], fs)
    #                 file_count += 1
    #                 pos_audios.append(slice_audio[int(box[0]):int(box[2])])
    #         boxes += [bbxs[b]
    #                   for b in range(len(bbxs)) if score[b] > score_thresh]
    #         scores += [s for s in score if s > score_thresh]
    #         figs += [fig[b]
    #                  for b in range(len(fig)) if score[b] > score_thresh]
    #         axs += [ax[b] for b in range(len(ax)) if score[b] > score_thresh]

    #     sub_process_prog.text(f"{int((i+1)*100/len(sub_audios))} %")
    #     prog.progress(int((i+1)*100/len(sub_audios)))

    # audio_load_state.text("Processed !")
    f"""
    ## Audio number {u+1} ({uploaded_audio.name}):
    """
    boxes = np.load(
        f"{STORAGE_PATH}{uploaded_audio.name[:-4]}_boxes.npy", allow_pickle=True)
    scores = np.load(
        f"{STORAGE_PATH}{uploaded_audio.name[:-4]}_scores.npy", allow_pickle=True)
    base_absc = np.load(
        f"{STORAGE_PATH}{uploaded_audio.name[:-4]}_base_absc.npy", allow_pickle=True)

    if st.checkbox("Show entire raw spectrogtam", value=True, key=f"raw_spectro_{u+1}"):
        # fig, ax = plt.subplots()
        # x_max, y_max = spectro.shape
        # print(x_max, y_max)
        # ax.matshow(spectro, aspect=750)  # , extent=[
        # # 0, y_max/fs, 0, x_max])
        # ax.axes.get_yaxis().set_visible(False)
        # factor = spectro.shape[-1]/len(audio)
        # for i, box in enumerate(boxes):
        #     x0 = int(box[0]*factor) + int(base_absc[i]*factor)
        #     x1 = int(box[2]*factor) + int(base_absc[i]*factor)
        #     y0 = 0
        #     y1 = 128
        #     ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
        #                  fill=False, edgecolor='red', linewidth=1))
        # st.pyplot(fig)

        st.image(STORAGE_PATH +
                 f"{uploaded_audio.name[:-4]}_raw_spec.png", width=800)

        st.audio(uploaded_audio, format='audio/wav')
    calls_spec = [i for i in os.listdir(STORAGE_PATH) if i.startswith(
        f"{uploaded_audio.name[:-4]}_spec")]
    calls_spec.sort()
    temp_audios = [i for i in os.listdir(STORAGE_PATH) if i.startswith(
        f"temp_sub_audio_{uploaded_audio.name[:-4]}")]
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
                     f"{uploaded_audio.name[:-4]}_spec_{i}.png", width=800)
            st.audio(
                STORAGE_PATH+f"temp_sub_audio_{uploaded_audio.name[:-4]}_{i}.wav")
            radio_lbl = st.radio("Label of the sample", [
                "Negative", "Positive"], index=1, horizontal=True, key=f"radio_subbox_{u}_{i}")
            lbls[i] = radio_lbl

    if st.button(f"Save samples for record number {u+1}"):
        save_pos_samples(temp_audios, lbls, FS,
                         uploaded_audio, boxes, base_absc)

        st.text("Saved !")
