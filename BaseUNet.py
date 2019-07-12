import torch
from torch import nn
from TransitionBlock import TransitionBlock
from DownBlock import DownBlock
from DenseBlock import DenseBlock
from UpBlock import UpBlock


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
        # dense block channels are confusing in the paper don't know how the Upblock expects 256 when dense and down5 combined are 160, so we used 256 
        self.dense = DenseBlock(256, 256)
        self.up1 = UpBlock(256+128, 96)
        self.up2 = UpBlock(96+96, 64)
        self.up3 = UpBlock(64+64, 32)
        self.up4 = UpBlock(32+32, 16)
        self.up5 = UpBlock(16+16, 16)
        self.conv_last = nn.Conv2d(16,output_channels,3)

    def forward(self, x):
        x = self.trans1(x)
        x = self.down1(x)
        down1 = x
        x = self.down2(x)
        down2 = x
        x = self.down3(x)
        down3 = x
        x = self.down4(x)
        down4 = x
        x = self.down5(x)
        down5 = x
        x = self.trans2(x)
        x = self.dense(x)
        x = torch.cat([x, down5], dim=1)
        x = self.up1(x)
        x = torch.cat([x, down4], dim=1)
        x = self.up2(x)
        x = torch.cat([x, down3], dim=1)
        x = self.up3(x)
        x = torch.cat([x, down2], dim=1)
        x = self.up4(x)
        x = torch.cat([x, down1], dim=1)
        x = self.up5(x)
        x= self.conv_last(x)
        return x
