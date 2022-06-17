import time

class MetricsPrint():
  """ Class to visualize metrics during training


  Template display :

  -----name------|------loss-----|------me1------|-------m2------|-time elapsed--|time remaining-|
  ----15/----1000|---3.457e-01---|---3.457e-01---|---3.457e-01---|----123m64s----|----123m64s----|

  """
  def __init__(self, name, metric_names, n_samples_batch):
    self.metrics_names = metric_names
    self.n_samples_batch = n_samples_batch
    self.column_size = 15
    self.initial_print(name)

  def initial_print(self, name):
    pad = self.column_size - len(name)
    text = " " * (pad//2) + name + " " * (pad//2 + pad % 2)
    text += "|      loss     "
    for name in self.metrics_names:
        pad = self.column_size - len(name)
        text += "|" + " " * (pad//2) + name + " " * (pad//2 + pad%2)
    text += "| time elapsed  "
    text += "|time remaining "
    text += "|"
    print(text)

  def print_loss_metrics(self, loss_value, metrics_values, n_samples, time_elapsed, time_remaining):
    text = f"{n_samples:>7}/{self.n_samples_batch:>7}"
    text += f"|   {loss_value:.3e}   "
    for v in metrics_values:
        text += f"|   {v:.3e}   "
    time_elapsed = f"{round(time_elapsed//60)}m{round(time_elapsed%60)}s"
    pad = self.column_size - len(time_elapsed)
    text += "|" + " " * (pad//2) + time_elapsed + " " * (pad//2 + pad%2)
    time_remaining = f"{round(time_remaining//60)}m{round(time_remaining%60)}s"
    pad = self.column_size - len(time_remaining)
    text += "|" + " " * (pad//2) + time_remaining + " " * (pad//2 + pad%2)
    text += "|"
    print(text)

class TimePredictor():

  def __init__(self, n_iterations, beta=0.2):
    self.n_iterations = n_iterations
    self.beta = beta

  def start(self):
    self.starting_time = time.time()
    self.last = time.time()
    self.iter = 0
    self.mean_iter = None

  def lap(self):
    self.iter += 1
    elapsed = time.time() - self.last
    self.last = time.time()
    self.mean_iter = self.beta * elapsed + (1 - self.beta) * self.mean_iter if self.mean_iter != None else elapsed
    return elapsed

  def remaining(self):
    if self.mean_iter == None:
      return 0
    return self.mean_iter * (self.n_iterations - self.iter)

  def elapsed(self):
    return time.time() - self.starting_time
