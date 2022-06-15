import tensorflow as tf
import torch
import os
import argparse
from src.models import birdnet_model as birdnet
import json
import numpy as np
from src.utils import dataset
from src.training import train_classif as train

MODELS = {'birdnet': birdnet.BirdNet_loaded}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification model.")
    parser.add_argument("--model_name", type=str,
                        required=True, help="Name of the model to train.")
    parser.add_argument("--epochs", type=int, required=False,
                        help="Number of epochs to train.", default=10)
    parser.add_argument("--batch_size", type=int,
                        required=False, help="Batch size.", default=32)
    parser.add_argument("--learning_rate", type=float,
                        required=False, help="Learning rate.", default=0.0005)
    parser.add_argument("--model_args", type=json.loads,
                        required=False, help="Model kwargs", default='{}')  # In windows, add: --model_args '{"key_1": "value_1", "key_2": "value_2"}'

    args = parser.parse_args()

    model_name = args.model_name
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    model_args = args.model_args

    # Load model

    if model_name not in MODELS:
        raise(NotImplementedError("Model {} is not implemented.".format(model_name)))

    model = MODELS[model_name](**model_args)

    # Load data
    train_dl = dataset.create_dataset_birdnet(
        batch_size=batch_size, max_samples=10)
    test_dl = dataset.create_dataset_birdnet(
        batch_size=batch_size, train_test='test', max_samples=10)

    # Training parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics_name = ['accuracy']
    metrics = [tf.keras.metrics.Accuracy()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train model
    for epoch in range(1, epochs+1):
        print("-"*20)
        train.train_loop(train_dl, model, loss, batch_size, optimizer,
                         metrics, metrics_name, epoch=epoch)
