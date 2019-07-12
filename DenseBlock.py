import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 12, 3, 1, 1)
        self.batch_norm = nn.BatchNorm2d(num_features=12)
        self.conv2 = nn.Conv2d(input_channels+12, 12, 3, 1, 1)
        self.conv3 = nn.Conv2d(input_channels+12+12, output_channels, 3, 1, 1)
        self.batch_norm3 = nn.BatchNorm2d(num_features=output_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_batch_norm1 = self.batch_norm(x_conv1)
        x_batch_norm1 = self.relu(x_batch_norm1)
        x_temp = torch.cat([x, x_batch_norm1], dim=1)
        x_conv2 = self.conv2(x_temp)
        x_batch_norm2 = self.batch_norm(x_conv2)
        x_batch_norm2 = self.relu(x_batch_norm2)
        x_temp = torch.cat([x, x_batch_norm1, x_batch_norm2], dim=1)
        x_conv3 = self.conv3(x_temp)
        x_batch_norm3 = self.batch_norm3(x_conv3)
        x_batch_norm3 = self.relu(x_batch_norm3)
        return x_batch_norm3
