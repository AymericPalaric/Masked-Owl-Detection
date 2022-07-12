from src.models.efficientnet import EfficientNet
from src.utils import audio_utils, metrics_utils, transform_utils, torch_utils
from src.pipeline import sliding_window, roi_window
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import numpy as np
import torch
import os
from csv import DictWriter
import argparse
import src.constants as constants


# CONSTANTS
MAX_DURATION = 15  # max duration for pipeline on cpu = 15s audios
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
    pipeline = roi_window.RoiWindowPipeline(
        window_size, window_overlap, window_type, classification_model, audio_transform, image_transform, freq_max, treshold_nms, device)
    time_scale = 1 / (256 - 256//4)

    return pipeline


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


def collate_boxes(bbxs: torch.Tensor, scores: torch.Tensor):
    true_bbxs = []
    true_scores = []
    if len(bbxs) >= 2:
        bbxs, indices = torch.sort(bbxs, 0)
        scores = scores[indices[:, 0]]
        s = scores[0]
        i = 0
        leave_flag = False
        while i < len(bbxs)-1 and not leave_flag:
            prev_i = i
            x0, y0, x1, y1 = bbxs[i, 0], bbxs[i, 2], bbxs[i+1, 0], bbxs[i+1, 2]
            while y0 > x1 and i < len(bbxs)-1:
                s = max(s, scores[i])
                i += 1
                y0, x1, y1 = y1, bbxs[i, 0], bbxs[i, 2]
            true_scores.append(s)
            true_bbxs.append([x0.item(), 0., y0.item(), 12000.])
            if i == prev_i:
                leave_flag = True
        return true_bbxs, true_scores

    else:
        return bbxs.numpy(), scores.numpy()


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


def compute_spectro(audio, fs, nperseg=256, noverlap=256//4, scale="dB"):
    freq, time, spectro = audio_utils.compute_spectrogram(
        audio, fs, nperseg=nperseg, noverlap=noverlap, scale=scale)
    return spectro


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


# =====================================
# ========== MAIN FUNCTION ============
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_folder", type=str, required=False,
                        help="Path to the audio files", default='./data/detection_samples/')
    parser.add_argument("--output_folder", type=str, required=False,
                        help="Path to the output folder", default='./data/gui_storage/test_folder/')
    parser.add_argument("--win_size", type=float, required=False,
                        help="Window size (in sec)", default=22000/24000)
    parser.add_argument("--win_overlap", type=int, required=False,
                        help="Window overlap (percent of the window size)", default=int(100*3/4))
    parser.add_argument("--score_thresh", type=float, required=False,
                        help="Threshold for selecting false positives (between 0 and 1)", default=0.95)
    args = parser.parse_args()

    window_size = args.win_size
    window_overlap = args.win_overlap/100
    score_thresh = args.score_thresh

    uploaded_audio_path = args.audio_folder
    output_folder = args.output_folder

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    uploaded_audios_files = [i for i in os.listdir(
        uploaded_audio_path) if i.endswith(".wav")]
    print(f"Found {len(uploaded_audios_files)} audio files")

    # When a file is uploaded
    for u, uploaded_audio in enumerate(uploaded_audios_files):
        print(f"Analyzing audio file nb {u+1} ({uploaded_audio[:-4]})...")
        audio, spectro, fs = load_audio_spectro(
            os.path.join(uploaded_audio_path, uploaded_audio))
        pipeline = load_pipeline(window_size, window_overlap)

        figs, axs, boxes, scores = [], [], [], []
        total_len = len(audio)
        sub_audios = cut_audio(audio, fs)
        base_absc = []
        pos_audios = []
        file_count = 0
        for i, slice_audio in tqdm(enumerate(sub_audios)):
            fig, ax, bbxs, score = pipeline_on_slice(pipeline, slice_audio)
            if len(bbxs) > 0:
                base_absc += [i*len(slice_audio)
                              for j in range(len(bbxs)) if score[j] > score_thresh]
                for j, box in enumerate(bbxs):
                    if score[j] > score_thresh:
                        audio_utils.save_audio_file(
                            f"{output_folder}/temp_sub_audio_{uploaded_audio[:-4]}_{file_count}.wav", slice_audio[int(box[0]):int(box[2])], fs)
                        file_count += 1
                        pos_audios.append(slice_audio[int(box[0]):int(box[2])])
                boxes += [bbxs[b]
                          for b in range(len(bbxs)) if score[b] > score_thresh]
                scores += [s for s in score if s > score_thresh]
                figs += [fig[b]
                         for b in range(len(fig)) if score[b] > score_thresh]
                axs += [ax[b]
                        for b in range(len(ax)) if score[b] > score_thresh]
                        
        np.save(
            f"{output_folder}/{uploaded_audio[:-4]}_boxes.npy", boxes)
        np.save(
            f"{output_folder}/{uploaded_audio[:-4]}_scores.npy", scores)
        np.save(
            f"{output_folder}/{uploaded_audio[:-4]}_base_absc.npy", base_absc)
        print("Done!")
        print(f"Found {len(boxes)} positive calls")

        fig, ax = plt.subplots()
        x_max, y_max = spectro.shape
        ax.matshow(spectro)  # , extent=[
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
        fig.savefig(
            f"{output_folder}/{uploaded_audio[:-4]}_raw_spec.png")

        for i in range(len(boxes)):
            """
            **____________________________________________________________________________**
            """
            figs[i].savefig(
                f"{output_folder}/{uploaded_audio[:-4]}_spec_{i}.png")
    print(
        f"All audios and figures have been saved in {output_folder}")
