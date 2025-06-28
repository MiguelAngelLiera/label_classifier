from torch import nn
import torch.nn.functional as F
from src.Xception.separable_conv import SeparableConv
from src.Xception.utils import out_size, same_pad
import keras.applications.xception


class EntryFlow(nn.Module):
    def __init__(self, input_size: int = 229):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride = 2)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.os_conv1 = out_size(dimension=input_size, kernel_size=3, padding=0, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.os_conv2 = out_size(dimension=self.os_conv1, kernel_size=3)

        pad = same_pad(self.os_conv2, 1, 2)
        self.res_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=pad)
        self.bn_res_conv1 = nn.BatchNorm2d(128)
        self.os_res_conv1 = out_size(dimension=self.os_conv2, kernel_size=1, padding=pad, stride=2)

        pad = same_pad(self.os_conv2, 3, 1)
        self.sep_conv1 = SeparableConv(in_channels=64, out_channels=128, kernel_size=3, padding=pad)
        self.bn_sep1 = nn.BatchNorm2d(128)
        self.os_sep1 = out_size(dimension=self.os_conv2, kernel_size=3, padding=pad)

        pad = same_pad(self.os_sep1, 3, 1)
        self.sep_conv2 = SeparableConv(in_channels=128, out_channels=128, kernel_size=3, padding=pad)
        self.bn_sep2 = nn.BatchNorm2d(128)
        self.os_sep2 = out_size(dimension=self.os_sep1, kernel_size=3, padding=pad)

        pad = same_pad(self.os_sep2, 3, 2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.os_mp1 = out_size(dimension=self.os_sep2, kernel_size=3, stride=2, padding=pad)

    def forward(self, x):
        # input 229x229x3
        x = self.conv1(x)
        x = self.bn_conv1(x)
        print(f'out_size conv1: {self.os_conv1}')
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn_conv2(x)
        print(f'out_size conv2: {self.os_conv2}')
        x = F.relu(x)

        res_x = self.res_conv1(x)
        res_x = self.bn_res_conv1(res_x)
        print(f'out_size residual: {self.os_res_conv1}')

        x = self.sep_conv1(x)
        x = self.bn_sep1(x)
        print(f'out_size sep1: {self.os_sep1}')
        x = F.relu(x)

        x = self.sep_conv2(x)
        x = self.bn_sep2(x)
        print(f'out_size sep2: {self.os_sep2}')

        x = F.pad(x, pad=(same_pad(self.os_sep2, 3, 2),)*4, mode='constant', value=None)
        x = self.max_pool(x)
        print(f'out_size mp: {self.os_mp1}')
        print(res_x.size(), x.size())
        x = res_x + x
        return x
