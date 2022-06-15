import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import tensorflow as tf
from math import ceil


# Training step for keras model
@tf.function
def train_step(X, y, model, loss_fn, optimizer):
    """
    Args:
    -----
        - X (tf.Tensor): Input data
        - y (tf.Tensor): Target data
        - model (tf.keras.Model): Model to train
        - loss_fn (tf.keras.losses.Loss): Loss to use for training
        - optimizer (tf.keras.optimizers.Optimizer): Optimizer for training
    """
    with tf.GradientTape() as tape:
        preds = model(X, training=True)
        loss = loss_fn(y, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return preds, loss


# Training loop for keras model
def train_loop(dataset, model, loss_fn, batch_size, optimizer, metrics, metrics_name, epoch=0, batch_update=1):
    """ Train model for 1 epoch.
    Args:
    -----
        - dataset (tf.data.Dataset): Train data loader sliced into batches of tuples (X,y)
        - model (tf.keras.Model): Model to train
        - loss_fn (tf.keras.losses.Loss): Loss to use for training
        - batch_size (int): Size of batch, only used for verbose
        - optimizer (tf.keras.optimizers.Optimizer): Optimizer for training
        - metrics (tf.keras.metrics.Metric list)[n_metrics]: List of metrics to compute during training
        - metrics_name (String list)[n_metrics]: List of name for metrics
        - dataset_length (int): Size of dataset, only used for verbose
        - epoch (int): Current epoch
        - batch_update (int): Regulate the verbose. Print metrics every <<batch_uptdate>> batches
    """

    print(f"Epoch {epoch}")
    for batch, (X, y) in enumerate(dataset):

        preds, loss = train_step(X, y, model, loss_fn, optimizer)

        if batch % batch_update == 0:
            metrics_values = [m(y, preds) for m in metrics]
            loss_value, current = loss.numpy(), batch * len(X)
            metrics_printer = sum(
                [f"{metrics_name[i]}: {metrics_values[i]:>7f}" for i in range(len(metrics_name))])
            print("Batch: {}, Loss: {}, {}".format(
                epoch, batch, loss_value, metrics_printer))


# Test step for keras model
@tf.function
def test_step(X, y, model, loss_fn):
    preds = model(X, training=False)
    loss = loss_fn(y, preds)
    return preds, loss


def test_loop(dataset, model, loss_fn, batch_size, metrics, metrics_name, dataset_length=1, epoch=0):
    """ Test model on test dataset.
    Args:
        - dataset (tf.data.Dataset): Test data loader sliced into batches of tuples (X,y)
        - model (tf.keras.Model): Model to test
        - loss_fn (tf.keras.losses.Loss): Loss used for training
        - batch_size (int): Size of batch, only used for verbose
        - metrics (tf.keras.metrics.Metric list)[n_metrics]: List of metrics to compute
        - metrics_name (String list)[n_metrics]: List of name for metrics
        - dataset_length (int): Size of dataset, only used for verbose
        - epoch (int): Current epoch
    """

    metrics_values = [0. for i in range(len(metrics))]
    loss_value = 0.

    for batch, (X, y) in enumerate(dataset):

        preds, loss = test_step(X, y, model, loss_fn)

        for i in range(len(metrics)):
            metrics_values[i] += metrics[i](y, preds)
        loss_value += loss.numpy()

    for i in range(len(metrics)):
        metrics_values[i] /= ceil(dataset_length/batch_size)
    loss_value /= ceil(dataset_length/batch_size)

    metrics_printer = sum(
        [f"{metrics_name[i]}: {metrics_values[i]:>7f}" for i in range(len(metrics_name))])
    print("Test: Loss: {}, {}".format(epoch, loss_value, metrics_printer))
