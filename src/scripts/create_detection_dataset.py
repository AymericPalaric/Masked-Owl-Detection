import argparse

from ..preprocessing import dataset
from ..utils import path_utils

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Create detection dataset.")
  parser.add_argument("--n_samples", type=int, required=False, help="Nomber of samples to create.", default=-1)
  parser.add_argument("--uuid", type=int, required=True, help="Unique identifier for this batch of created samples.")
  parser.add_argument("--log", type=bool, required=False, help="Bool to enable/disable logging.", default=True)
  parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the created samples.")
  parser.add_argument("--duration", type=float, required=True, help="Duration of audio to save clip the audio if needed.", default=60)
  parser.add_argument("--call_prop", type=float, required=True, help="Parameters defining the poisson distribution of positive calls.", default=0.5)
  parser.add_argument("--hard_call_prop", type=float, required=True, help="Parameters defining the poisson distribution of hard negative calls.", default=0.3)
  parser.add_argument("--train_test", type=bool, required=True, help="Bool to determine if the created samples are for training or testing.", default=True)
  args = parser.parse_args()


  dataset.create_detection_dataset(output_folder=args.output_dir, 
    raw_folder=path_utils.get_raw_data_path(), 
    positive_folder=path_utils.get_train_test_path(path_utils.get_positive_samples_path(), args.train_test),
    hard_folder=path_utils.get_train_test_path(path_utils.get_hard_samples_path(), args.train_test),
    detection_duration=args.duration,
    n_samples=args.n_samples,
    uuid=args.uuid,
    call_proportion=args.call_prop,
    hard_call_proportion=args.hard_call_prop,
    log=args.log)
