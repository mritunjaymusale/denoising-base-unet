import torch
from torch import nn


class UpBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UpBlock, self).__init__()
        internal_channels = 32
        kernel_size = 7
        self.conv1 = nn.Conv2d(input_channels, internal_channels, 3, 1, 1)
        # ConvTranspose2d replaced with Upsample since noise reduction
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        # Convolution layer to compensate for the change in number of channels 
        self.conv_compensate = nn.Conv2d(
            internal_channels, output_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1)
        self.batch_norm = nn.BatchNorm2d(num_features=output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.first_conv_block(x)
        x = self.upsampling(x)
        x = self.second_conv_block(x)
        return x

    def first_conv_block(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

    def second_conv_block(self, x):
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

    def upsampling(self, x):
        x = self.upsample(x)
        x = self.conv_compensate(x)
        x = self.relu(x)
        return x
