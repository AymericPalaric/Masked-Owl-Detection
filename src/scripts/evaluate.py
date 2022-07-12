import argparse
from src.evaluation.classif_evaluation import ClassifEvaluator
from src.models.efficientnet import EfficientNet
from src.models.baseline_cnn import Baseline
import matplotlib.pyplot as plt
import torchvision
import torch
import json
import argparse
from tqdm import tqdm
nn = torch.nn
"""
Script to evaluate quality of prediction of a model previously trained.
Arguments are:
    - model_type;
    - model_name;
    - model_args;
    - model_path;
    - train_test;
    - batch_size;
    - num_workers;
    - output
"""

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
        "--output", type=str, choices=["confusion_matrix", "metrics", "both", "thresh_range"], default="both")

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

    evaluator = ClassifEvaluator(model_class=CLASS_MODELS[model_type], device=device, name=model_name,
                                 model_path=model_path, batch_size=batch_size, num_workers=num_workers, train_test=train_test, **model_args)

    if output == "confusion_matrix":
        print("Evaluating model through confusion matrix")
        evaluator.conf_matrix()

    elif output == "metrics":
        print("Evaluating model through metrics")
        print(evaluator.evaluate())

    elif output == "both":
        print("Evaluating model through both metrics and confusion matrix")
        print(evaluator.evaluate())
        evaluator.conf_matrix()

    elif output == "thresh_range":
        print("Evaluating preds for a range of thresholds...")
        evals = {}
        thresh_range = [i/100 for i in range(30,99,1)]
        for thresh in tqdm(range(30, 99, 1)):
            evals[thresh] = evaluator.evaluate(thresh/100, verbose=False)
            #evaluator.conf_matrix(thresh/100, verbose=False)
        print(evals)
        fig = plt.figure()
        for key in ["accuracy", "recall", "precision"]:
            plt.plot(thresh_range, [evals[i][key] for i in evals])
        plt.legend([key for key in evals[thresh]])
        plt.savefig("./trained_models/thresh_range_"+model_name+".png")
