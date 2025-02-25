import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return (x - torch.mean(x)) / torch.std(x)


class MinimumToZero(nn.Module):
    def __init__(self):
        super(MinimumToZero, self).__init__()

    def forward(self, x):
        return x - torch.min(x)