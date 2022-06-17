import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import tensorflow as tf
from math import ceil

# ==================================================================================
# Training step for keras model


@tf.function
def train_step_keras(X, y, model, loss_fn, optimizer):
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
def train_loop_keras(dataset, model, loss_fn, batch_size, optimizer, metrics, metrics_name, epoch=0, batch_update=1):
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

        preds, loss = train_step_keras(X, y, model, loss_fn, optimizer)

        if batch % batch_update == 0:
            metrics_values = [m(y, preds) for m in metrics]
            loss_value, current = loss.numpy(), batch * len(X)
            metrics_printer = " | ".join(
                [f"{metrics_name[i]}: {metrics_values[i]:>7f}" for i in range(len(metrics_name))])
            print("Batch: {}, Loss: {}, {}".format(
                batch, loss_value, metrics_printer))


# Test step for keras model
@tf.function
def test_step_keras(X, y, model, loss_fn):
    preds = model(X, training=False)
    loss = loss_fn(y, preds)
    return preds, loss


def test_loop_keras(dataset, model, loss_fn, batch_size, metrics, metrics_name, dataset_length=1, epoch=0):
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

    metrics_values = [0. for m in metrics]
    loss_value = 0.

    for batch, (X, y) in enumerate(dataset):

        preds, loss = test_step_keras(X, y, model, loss_fn)

        for i in range(len(metrics)):
            metrics_values[i] += metrics[i](y, preds)
        loss_value += loss.numpy()

    for i in range(len(metrics)):
        metrics_values[i] /= ceil(dataset_length/batch_size)
    loss_value /= ceil(dataset_length/batch_size)

    metrics_printer = " | ".join(
        [f"{metrics_name[i]}: {metrics_values[i]:>7f}" for i in range(len(metrics_name))])
    print("Test: Loss: {}, {}".format(loss_value, metrics_printer))

# ==================================================================================

# ==================================================================================
# Training step for pytorch model


def train_torch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Test step for pytorch model


def test_torch(dataloader, model, loss_fn, device, metrics, metrics_name):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
# ==================================================================================
