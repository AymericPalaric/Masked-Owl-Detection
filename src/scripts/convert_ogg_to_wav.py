import os
import argparse
from tqdm import tqdm

from src.utils import audio_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory to search for audio files.")
    parser.add_argument("--duration", type=float, required=False,
                        help="Duration of audio to save clip the audio if needed.", default=-1)
    args = parser.parse_args()
    folder = args.dir
    duration = args.duration

    # check for folder existence
    if not os.path.exists(folder):
        raise ValueError(f"Folder {folder} does not exist.")

    # list ogg files
    ogg_files = [file for file in os.listdir(folder) if file.endswith(".ogg")]

    # convert ogg files to wav files
    for ogg_file in tqdm(ogg_files):
        file_path = os.path.join(folder, ogg_file)
        data, fs = audio_utils.load_audio_file(file_path)
        if duration > 0:
            data = audio_utils.clip_audio(data, fs, duration)
        wav_file_path = file_path.replace(".ogg", ".wav")
        audio_utils.save_audio_file(wav_file_path, data, fs)
        # delete ogg file
        os.remove(file_path)

    print("Done!")
