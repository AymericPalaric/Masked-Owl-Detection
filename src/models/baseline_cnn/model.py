import torch
nn= torch.nn

class Baseline(nn.Module):
  def __init__(self):
    super().__init__()
    self.main = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
    )
    self.linear = nn.Linear(128, 2)
  
  def forward(self, x):
    x = self.main(x)
    # global average pooling
    x = nn.functional.avg_pool2d(x, kernel_size=x.shape[2:]).view(x.shape[0], -1)
    x = self.linear(x)
    return x
