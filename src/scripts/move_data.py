import os
import argparse

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Move data file from multiple folders to the root folder")
  parser.add_argument("--root_folder", type=str, default="./data/raw_data", help="Root folder")

  args = parser.parse_args()
  
  root_dir = args.root_folder
  # check if the root folder exists
  if not os.path.exists(root_dir):
    print("Root folder does not exist")
    exit(1)
  # list all folder
  folders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]

  for folder in folders:  
    # list all files
    files = [file for file in os.listdir(os.path.join(root_dir, folder)) if file.endswith(".wav") or file.endswith(".ogg")]
    # move file to root dir
    for file in files:
      file_path = os.path.join(root_dir, folder, file)
      os.rename(file_path, os.path.join(root_dir, file))
    # remove folder
    os.rmdir(os.path.join(root_dir, folder))
  print("Done!")
