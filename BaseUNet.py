import torch
from torch import nn
from TransitionBlock import TransitionBlock
from DownBlock import DownBlock


class BaseUNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(BaseUNet, self).__init__()
        self.trans1 = TransitionBlock(input_channels, 8)
        self.down1 = DownBlock(8, 16)
        self.down2 = DownBlock(16, 32)
        self.down3 = DownBlock(32, 64)
        self.down4 = DownBlock(64, 96)
        self.down5 = DownBlock(96, 128)
        self.trans2 = TransitionBlock(128, 256)

    def forward(self, x):
        x = self.trans1(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.trans2(x)
        return x
