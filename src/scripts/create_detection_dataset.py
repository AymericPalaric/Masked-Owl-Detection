import argparse
import os
from src.preprocessing import dataset
from src.utils import path_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create detection dataset.")
    parser.add_argument("--n_samples", type=int, required=False,
                        help="Nomber of samples to create.", default=-1)
    parser.add_argument("--uuid", type=int, required=True,
                        help="Unique identifier for this batch of created samples.")
    parser.add_argument("--log", type=bool, required=False,
                        help="Bool to enable/disable logging.", default=True)
    parser.add_argument("--duration", type=float,
                        help="Duration of audio to save clip the audio if needed.", default=60)
    parser.add_argument("--call_prop", type=float,
                        help="Parameters defining the poisson distribution of positive calls.", default=0.5)
    parser.add_argument("--hard_call_prop", type=float,
                        help="Parameters defining the poisson distribution of hard negative calls.", default=0.3)
    parser.add_argument("--train_test", type=str,
                        help="Bool to determine if the created samples are for training or testing.", default="train")
    args = parser.parse_args()

    train_test = True if args.train_test == "train" else False

    if not os.path.exists(path_utils.get_train_test_path(path_utils.get_detection_samples_path(), train_test)):
        print("Root folder does not exist, creating it...")
        if not os.path.exists(path_utils.get_detection_samples_path()):
            os.mkdir(path_utils.get_detection_samples_path())
            os.mkdir(path_utils.get_train_test_path(
                path_utils.get_detection_samples_path(), train_test))
        else:
            os.mkdir(path_utils.get_train_test_path(
                path_utils.get_detection_samples_path(), train_test))

    dataset.create_detection_dataset(output_folder=path_utils.get_train_test_path(path_utils.get_detection_samples_path(), train_test),
                                     raw_folder=path_utils.get_raw_data_path(),
                                     positive_folder=path_utils.get_train_test_path(
                                         path_utils.get_positive_samples_path(), train_test),
                                     hard_folder=path_utils.get_train_test_path(
                                         path_utils.get_hard_samples_path(), train_test),
                                     detection_duration=args.duration,
                                     n_samples=args.n_samples,
                                     uuid=args.uuid,
                                     call_proportion=args.call_prop,
                                     hard_call_proportion=args.hard_call_prop,
                                     log=args.log)
