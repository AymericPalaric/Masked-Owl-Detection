import argparse
from ..evaluation.classif_evaluation import ClassifEvaluator
from ..models.efficientnet import EfficientNet
from ..models.baseline_cnn.model import Baseline
from ..models.baseline_cnn import training
from ..utils import torch_utils, transform_utils
import matplotlib.pyplot as plt
import torchvision
import torch
import json
import argparse
nn = torch.nn


# Models available
CLASS_MODELS = {'efficientnet': EfficientNet, 'baseline_cnn': Baseline}

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="efficientnet")
    parser.add_argument("--model_name", type=str, default='unnamed')
    parser.add_argument("--model_args", type=json.loads,
                        required=False, default='{}')
    parser.add_argument("--model_path", type=str, required=False, default='')
    parser.add_argument("--train_test", type=str, default='test')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--output", type=str, choices=["confusion_matrix", "metrics", "both"], default="both")

    args = parser.parse_args()
    model_type = args.model_type
    model_name = args.model_name
    model_args = args.model_args
    model_path = args.model_path
    train_test = args.train_test
    batch_size = args.batch_size
    num_workers = args.num_workers
    output = args.output

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    evaluator = ClassifEvaluator(model_type=CLASS_MODELS[model_type], device=device, name=model_name,
                                 model_path=model_path, batch_size=batch_size, num_workers=num_workers, train_test=train_test, **model_args)

    if output == "confusion_matrix":
        evaluator.conf_matrix()

    elif output == "metrics":
        print(evaluator.evaluate())

    elif output == "both":
        print(evaluator.evaluate())
        evaluator.conf_matrix()
