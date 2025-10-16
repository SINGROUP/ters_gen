import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        x_mean = x.mean()
        x_std = x.std()
        if x_std == 0:
            # For zero-std images, just subtract the mean (which is 0 anyway)
            return x - x_mean
        else:
            # Normal normalization with a small epsilon for stability
            return (x - x_mean) / x_std 



class MinimumToZero(nn.Module):
    def __init__(self):
        super(MinimumToZero, self).__init__()

    def forward(self, x):
        return x - torch.min(x)