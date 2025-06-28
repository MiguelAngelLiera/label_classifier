from torch import nn
from typing import Union

class SeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: Union[int, str], depth_multiplier: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1)
        self.depthwise_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels*depth_multiplier,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_channels)

    def forward(self, x):
        x = self.pointwise_conv(x)
        x = self.depthwise_conv(x)
        return x
