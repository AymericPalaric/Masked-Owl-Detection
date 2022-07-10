# Bird Call Detection

This project aims at detecting 2 seconds Masked owl calls in 1 hour audio records using Machine Learning. The current version uses a Convolutional Neural Network (CNN) on spectrograms to detect the calls. As a 1h audio is too long to be processed in a single run, the project splits the audio into smaller chunks using a sliding window and classification is perfomed on each chunk. 

# Installation

To install all the packages required, use the command `pip install -r requirements.twt` at the root of the project. Python 3.9 or later is recomanded.

You will then have to initiallise the dataset by following thoses steps :
- Create the data folder and the different subfolders and precise the location of the data folder in the config file
- Put your positive and raw data in the right folders with all the samples in wav in a single folder (use the move_data and convert_ogg_to_wav scripts if necessary).
- Download the hard samples from [kaggle](https://www.kaggle.com/competitions/birdclef-2022/data) and place them in the folder.
- Use the move_data and convert_ogg_to_wav in order to have all the samples in wav in a single folder
- Use the create_negative_samples script to create the negative samples.
- Use the split_samples_train_test and compute_mean_std scripts and copy the mean and standart deviation found to the constant file

# Architecture
```
.
├── README.md
├── data
│   ├── detection_samples
│   │   ...
│   ├── gui_storage
│   │   ...
│   ├── hard_samples
│   │   ...
│   ├── positive_samples
│   │   ...
│   ├── negative_samples
│   │   ...
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── config_template.py
│   ├── constants.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   └── classif_evaluation.py
│   ├── gui
│   │   ├── __init__.py
│   │   └── gui.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── baseline_cnn.py
│   │   └── efficientnet.py
│   ├── pipeline
│   │   ├── __init__.py
│   │   └── sliding_window.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── scripts
│   │   ├── compute_mean_std.py
│   │   ├── convert_ogg_to_wav.py
│   │   ├── create_detection_dataset.py
│   │   ├── create_negative_samples.py
│   │   ├── evaluate.py
│   │   ├── evaluate_detection.py
│   │   ├── move_adv_data.py
│   │   ├── move_data.py
│   │   ├── split_samples_train_test.py
│   │   └── train_classification.py
│   ├── training
│   │   ├── __init__.py
│   │   └── training.py
│   └── utils
│       ├── __init__.py
│       ├── audio_utils.py
│       ├── metrics_utils.py
│       ├── path_utils.py
│       ├── torch_utils.py
│       ├── transform_utils.py
│       └── visualization.py
└── trained_models
    ...

```

Appart from the data and the trained models (where you can also find the output images of the evaluation script). The whole project is within the /src folder which is composed of :
- Evaluation
- GUI
- Models
- Pipeline
- Preprocessing
- Scripts
- Training
- Utils

Most names are self explainatory and you most probably will only have to interact with the scripts. You will also wind the config_template.py. Indicate the relative path (from the root) of the data folder and change the name to config_template.py

# CLI
Appart from scripts used while initailizing the project, you can use the following commands :
- `python3 src/scripts/train_classification.py --model_name <model_name>`: to train the classification model (the base model used is a pretrained efficient net model)
Additional options are available, but set to default values:
    - `--epochs`: default to 10;
    - `--batch_size`: default to 64;
    - `--n_workers`: default to 4 (for parallelization);
    - `--model_type`: default to 'efficientnet' (additional informations about using an other type of model can be found in a [further section](##add-a-new-model-type))
    - `--lr`: learning rate, default to $10^{-5}$
    - `--model_args`: additional arguments that you want to pass to your custom model, synthax is the following in the CLI: `--model_args '{"key_1": "value_1", "key_2": "value_2"}'`
- `python3 src/scripts/evaluate.py --model_path <path_to_model> --model_name <model_name>`: to evaluate the classification model (get metrics in the terminal and the confusion matrix in a saved png file). Same additional options as the previous script can be added.

# GUI

To launch the graphic interface, run the command ` python3 -m  streamlit run src/gui/gui.py --server.maxUploadSize 500` at the root of the project. The interface will run in your browser.

You will then have multiple parameters to play with, or to keep to default:
- The size of the window that will be sliding over the whole audio ;
- The % of the window size that will correspond to the overlap between 2 consecutive windows ;
- The threshold to apply to the confidence scores that will filter the false positives.

You can now upload a file, corresponding to a ~1h audio. After the process is completed, the raw spectrogram of the whole audio is displayed, with boxes corresponding to calls detected.
Below this spectrogram, you will be able to see each of the spectrograms, scores and audio samples corresponding to all the calls detected. The default label for all these samples is **"Positive"**, but you can switch them manually to **"Negative"** if the sample is actually a false positive.

Finally, you will be able to click on the **"Save samples"** button, that will save each of the previous detected calls as WAV files, having the following filenames: `<name_of_the_uploaded_audio>_<beginning_of_the_sample_in_the_1h_audio (in s)>_<duration_of_the_call (in ms)>.wav`.
It will also create a CSV file (with the name of the uploaded audio) with the following keys (each line corresponds to one positive sample):
- `call_files`: Names of the audio files of the samples ;
- `time_stamps`: beginning of the samples in the 1h audio ;
- `durations`: durations of the samples.

*Don't forget to click on that **"Save samples"** button !*
-
Otherwise you won't have any of the positive samples saved on your device.
# Next Steps and unmerged branches

## Add a new model type
You can add a custom model type by doing so:
In the `models/` folder, create a file with a custom name (for example `my_model.py`) containing a class creating your **torch** model (named for example `MyModel(torch.nn.Module)`).
To use CLI scripts and the GUI, make sure to import and modify the following files:
- `./src/scripts/train_classification.py` and `./src/scripts/evaluate.py`: import your model and make sure to complete the dictionary `CLASS_MODELS` in both files, and don't forget to call the `--model_type` argument in the CLI.
- `./src/gui/gui.py`: import your model and modify the `CLASSIF_MODEL` and `CLASSIF_MODEL_PATH` variables, with `CLASSIF_MODEL_PATH` being the path from the root to the weights of your trained model.