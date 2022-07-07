import torch
from sklearn.metrics import precision_recall_fscore_support

from ..utils import visualization, torch_utils
from .. import constants

def train_loop(model, dataloader, optimizer, loss_fn, device):
  size = len(dataloader.dataset)
  printer = visualization.MetricsPrint("Train loop", ["accuracy"], size)
  chrono = visualization.TimePredictor(len(dataloader))
  model.train()
  optimizer.zero_grad()

  mean_loss = 0.0

  # Iterate over the dataset
  chrono.start()
  for batch, (x, targets) in enumerate(dataloader):
    # Work with the GPU if available
    x = x.to(device)
    targets = targets.to(device)

    # Compute prediction error
    y = model(x)
    loss = loss_fn(y, targets)
    mean_loss += loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Print metrics
    chrono.lap()
    if batch % 2 == 0 or batch == len(dataloader) - 1:
      loss_value, current = loss.item(), (batch+1) * len(x)
      printer.print_loss_metrics(loss_value=loss_value, metrics_values=[torch_utils.accuracy(y, targets)], n_samples=current, time_elapsed=chrono.elapsed(), time_remaining=chrono.remaining())

  return mean_loss / len(dataloader)

def test_loop(dataloader, model, loss_fn, device):
  """ Test function No metrics yet implemented """
  # Initialize training
  size = len(dataloader.dataset)
  printer = visualization.MetricsPrint("Test loop", ["precision", "recall", "f1-score"], size)
  chrono = visualization.TimePredictor(len(dataloader))
  model.eval()

  mean_loss = 0.0

  full_targets = list()
  full_predictions = list()

  # Iterate over the dataset
  chrono.start()
  with torch.no_grad():
    for batch, (x, targets) in enumerate(dataloader):

      # Work with the GPU if available
      x = x.to(device)
      targets = targets.to(device)
      
      y = model(x)
      loss = loss_fn(y, targets)
      mean_loss += loss.item()

      full_predictions.extend(torch.argmax(y, dim=1).cpu().numpy().tolist())
      full_targets.extend(targets.cpu().numpy().tolist())

  mean_loss /= len(dataloader)
  precision, recall, f1, _  = precision_recall_fscore_support(full_targets, full_predictions, average='binary', pos_label=constants.positive_label)

  printer.print_loss_metrics(loss_value=mean_loss, metrics_values=[precision, recall, f1], n_samples=1, time_elapsed=chrono.elapsed(), time_remaining=0)

  return mean_loss
