import torch
from torch import nn

# Residual Block


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        if stride > 1:
            self.bn_1 = nn.BatchNorm2d(out_channels)
            self.conv_1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding='same')
            self.maxpool = nn.MaxPool2d(
                kernel_size=(stride, stride), stride=stride)

        self.do = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding='same')
        self.relu = nn.ReLU(inplace=True)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding='same')

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=(stride, stride), stride=stride),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.stride > 1:
            ds = self.conv_1(x)
            ds = self.bn_1(ds)
        else:
            ds = x

        residual = x

        out = self.conv1(ds)
        out = self.bn1(out)
        out = self.relu(out)

        if self.stride > 1:
            out = self.maxpool(out)
        out = self.do(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if out.shape != residual.shape:
            residual = self.shortcut(residual)
        out += residual
        return out
