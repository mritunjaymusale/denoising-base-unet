import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DenseBlock, self).__init__()
        growth_rate = 12
        self.conv1 = nn.Conv2d(input_channels, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(
            input_channels+growth_rate, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(
            input_channels+growth_rate+growth_rate, output_channels, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_first_conv = self.first_convBlock(x)
        x_temp = torch.cat([x, x_first_conv], dim=1)
        x_second_conv = self.second_convBlock(x_temp)
        x_temp = torch.cat([x, x_first_conv, x_second_conv], dim=1)
        x_third_conv = self.third_convBlock(x_temp)
        return x_third_conv

    def first_convBlock(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

    def second_convBlock(self, x):
        x = self.conv2(x)
        x = self.relu(x)
        return x

    def third_convBlock(self, x):
        x = self.conv3(x)
        x = self.relu(x)
        return x
