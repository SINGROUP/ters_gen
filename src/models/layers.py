import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, Dataloader

from typing import Tuple, Callable, Any

class ResBlock(nn.Module):
    """Residual Block with two convolutional layers and a skip connection."""

    def __init__(self, channels, kernel_size=3, stride=1, activation=nn.ReLU):
        super().__init__()

        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride), 
            nn.BatchNorm2d(channels),
            activation(),
            nn.Conv2d(channels, channels, kernel_size, stride),
            nn.BatchNorm2d(channels), 
            activation()
        )

        if x.shape != residual.shape:
            residual = 