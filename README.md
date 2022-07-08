# Bird Call Detection

Detect Masked owl calls in audio records.

# Installation

To install all the packages required, use the command `pip install -r requirements.twt` at the root of the project. Python 3.9 or later is recomanded.

You will then have to initiallise the dataset by following thoses steps :
- Create the data folder and the different subfolders and precise the location of the data folder in the config file
- Put your positive and raw data in the right folders with all the samples in wav in a single folder (use the move_data and convert_ogg_to_wav scripts if necessary).
- Download the hard samples from [kaggle](https://www.kaggle.com/competitions/birdclef-2022/data) and place them in the folder.
- Use the move_data and convert_ogg_to_wav in order to have all the samples in wav in a single folder
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


# GUI

To launch the graphic interface, run the command ` python3 -m  streamlit run src/gui/gui.py --server.maxUploadSize 500` at the root of the project. The interface will run in your browser.
# Next Steps and unmerged branches
