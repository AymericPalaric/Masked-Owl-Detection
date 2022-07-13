#!/bin/bash
echo "What is the ABSOLUTE path to the root of the project?"
read root_path

cd $root_path

conda init bash -q

echo "Installing dependencies... (this might take a while the first time)"
conda create -y -d -n masked-owl-detection python=3.10
conda activate masked-owl-detection

pip install -r requirements.txt --quiet
echo "What is the input folder ? (relative path from the root of the project)"
read input_folder
echo
echo "What is the output folder ? (relative path from the root of the project)"
read output_folder
python -m src.gui.preproc_gui --input_path $input_folder --output_path $output_folder

echo "Starting GUI..."
python -m streamlit run ./src/gui/light_gui.py -- --input_path $input_folder --output_path $output_folder