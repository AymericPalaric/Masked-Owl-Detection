import os
import argparse
import shutil
import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Move adversarial files from multiple folders to the root folder")
    parser.add_argument("--root_folder", type=str,
                        default="./data/adversarial_raw", help="Root folder")
    parser.add_argument("--initial_folder", type=str,
                        default="../RMBL-Robin/data", help="Initial data folder")

    args = parser.parse_args()

    root_dir = args.root_folder
    init_dir = args.initial_folder
    # check if the root folder exists
    if not os.path.exists(root_dir):
        print("Root folder does not exist, creating it...")
        os.mkdir(root_dir)
    # list all files
    files = [file for file in os.listdir(init_dir) if file.endswith(".WAV")]

    for file in tqdm.tqdm(files):
        shutil.move(os.path.join(init_dir, file), os.path.join(root_dir, file))

    print("Done!")
