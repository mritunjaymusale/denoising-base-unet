import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownBlock, self).__init__()
        
        internal_channels = int((input_channels+output_channels)/2)
        self.conv1 = nn.Conv2d(input_channels, internal_channels, 3, 1, 1, 1)
        self.conv2 = nn.Conv2d(input_channels, internal_channels, 3, 1, 3, 3)
        self.conv3 = nn.Conv2d(input_channels, internal_channels, 3, 1, 5, 5)
        self.conv4 = nn.Conv2d(
            internal_channels+internal_channels+internal_channels, output_channels, 3, 2, 1)
        self.conv5 = nn.Conv2d(output_channels, output_channels, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_conv1 = self.first_conv_block(x)
        x_conv2 = self.second_conv_block(x)
        x_conv3 = self.third_conv_block(x)
        x = torch.cat([x_conv1, x_conv2, x_conv3], dim=1)
        x = self.fourth_conv_block(x)
        x = self.fifth_conv_block(x)
        return x

    def first_conv_block(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

    def second_conv_block(self, x):
        x = self.conv2(x)
        x = self.relu(x)
        return x

    def third_conv_block(self, x):
        x = self.conv3(x)
        x = self.relu(x)
        return x

    def fourth_conv_block(self, x):
        x = self.conv4(x)
        x = self.relu(x)
        return x

    def fifth_conv_block(self, x):
        x = self.conv5(x)
        x = self.relu(x)
        return x
