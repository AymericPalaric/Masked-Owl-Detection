import os
import argparse
from src.preprocessing import dataset
from src.utils import path_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create negative samples from raw data.")
    parser.add_argument("--n_samples", type=int, required=False,
                        help="Number of samples to create.", default=-1)
    parser.add_argument("--uuid", type=int, required=True,
                        help="Unique identifier for this batch of extracted samples.")
    parser.add_argument("--duration", type=float, required=False,
                        help="Duration of the samples.", default=2.0)
    parser.add_argument("--log", type=bool, required=False,
                        help="Bool to enable/disable logging.", default=True)
    parser.add_argument("--neg_path", type=str, required=False,
                        help="Path to the negative samples.", default="")

    args = parser.parse_args()

    if args.n_samples < 0:
        positive_files = [file for file in os.listdir(
            path_utils.get_positive_samples_path()) if file.endswith(".wav")]
        n_samples = len(positive_files)
    else:
        n_samples = args.n_samples
    uuid = args.uuid
    log = args.log
    neg_path = args.neg_path if args.neg_path != "" else path_utils.get_raw_data_path()

    dataset.create_negative_samples(
        raw_folder=neg_path,
        output_folder=path_utils.get_negative_samples_path(),
        negative_sample_duration=args.duration,
        n_samples=n_samples,
        uuid=uuid,
        log=log)
