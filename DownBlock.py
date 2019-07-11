import torch
from torch import nn
import torch


class DownBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, 1, 1, 1)
        self.conv2 = nn.Conv2d(input_channels, 64, 3, 1, 3, 3)
        self.conv3 = nn.Conv2d(input_channels, 64, 3, 1, 5, 5)
        self.conv4 = nn.Conv2d(64+64+64, output_channels,3)
        self.conv5 = nn.Conv2d(output_channels, output_channels, 1, 1)

    def forward(self, x):
        print(x.shape)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x)
        x_conv3= self.conv3(x)
        x=torch.cat([x_conv1,x_conv2,x_conv3],dim=1)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


