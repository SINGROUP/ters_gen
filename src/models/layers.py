import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, Dataloader

from typing import Tuple, Callable, Any

class ResBlock(nn.Module):
    """Residual Block with two convolutional layers and a skip connection."""

    def __init__(self, input_channels,output_channels, kernel_size=3, stride=1, activation=nn.ReLU):
        super().__init__()

        self.res = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride), 
            nn.BatchNorm2d(output_channels),
            activation(),
            nn.Conv2d(output_channels, output_channels, kernel_size, stride),
            nn.BatchNorm2d(output_channels), 
            activation()
        )

        if input_channels != output_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channels)
            )
            self.flag = True

    def forward(self, x):
        if self.flag:
            return self.res(x) + self.skip(x)
        else:
            return self.res(x) + x



class AttentionBlock3D(nn.Module):
    
    def __init__(self, channels_x, channels_g, attention_channels, kernel_size, conv_activation=nn.ReLU, att_activation = nn.Sigmoid):
        super().__init__()

        padding = kernel_size // 2
        self.W_g = nn.Sequential(
            nn.Conv3d(in_channels=channels_x, out_channels = attention_channels, kernel_size=kernel_size, padding = padding),
            nn.BatchNorm3d(attention_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels = channels_x, out_channels = attention_channels, kernel_size=kernel_size, padding = padding),
            nn.BatchNorm3d(attention_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(in_channels = attention_channels, out_channels = 1, kernel_size=kernel_size, padding = padding), 
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.up = nn.Upsample(scale_factor=(2,2,1), mode='trilinear')
        self.relu = nn.ReLU()


    def forward(self, g,x):

        g = self.up(g)
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi, psi
    


