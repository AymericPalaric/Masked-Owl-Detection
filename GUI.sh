#!/bin/bash
cd /d/Informatique/Python/CS/DTY/P3/masked-owl-detection
echo "What is the input folder ? (relative path from the root of the project)"
read input_folder
echo "What is the output folder ? (relative path from the root of the project)"
read output_folder
python -m src.gui.preproc_gui --input_path $input_folder --output_path $output_folder   # Create all files needed for the GUI

echo "Starting GUI..."
python -m streamlit run ./src/gui/light_gui.py -- --input_path $input_folder --output_path $output_folder # Launch the GUI