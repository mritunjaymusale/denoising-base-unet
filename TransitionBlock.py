import torch
from torch import nn


class TransitionBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(TransitionBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(input_channels, output_channels, 1, 1)
        self.conv31 = nn.Conv2d(input_channels, output_channels, 1, 1)
        self.conv32 = nn.Conv2d(
            output_channels, output_channels, 3, 1, padding=1)
        self.conv41 = nn.Conv2d(input_channels, output_channels, 1, 1)
        self.conv42 = nn.Conv2d(output_channels, output_channels, 5, 1, 2)
        self.convf = nn.Conv2d(output_channels+output_channels +
                               output_channels+output_channels, output_channels, 3, 1, 1)
        self.batch_norm = nn.BatchNorm2d(num_features=output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_conv1 = self.relu(x_conv1)
        pooling = self.pool(x)
        x_conv2 = self.conv2(pooling)
        x_conv2 = self.relu(x_conv2)
        x_conv31 = self.conv31(x)
        x_conv31 = self.relu(x_conv31)
        x_conv32 = self.conv32(x_conv31)
        x_conv32 = self.relu(x_conv32)
        x_conv41 = self.conv41(x)
        x_conv41 = self.relu(x_conv41)
        x_conv42 = self.conv42(x_conv41)
        x_conv42 = self.relu(x_conv42)
        x = torch.cat([x_conv1, x_conv2, x_conv32, x_conv42], dim=1)
        x = self.convf(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
