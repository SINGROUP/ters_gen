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


class NormalizeVectorized(nn.Module):
    """
    Vectorized per-channel normalization for (C, H, W) tensors.
    Each channel is normalized independently.
    """
    def __init__(self, eps=1e-8):
        super(NormalizeVectorized, self).__init__()
        self.eps = eps

    def forward(self, x):
        # x shape: (C, H, W)
        # Compute per-channel mean and std
        mean = x.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        std = x.std(dim=(1, 2), keepdim=True)    # (C, 1, 1)
        # Avoid division by zero
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (x - mean) / std


class MinimumToZero(nn.Module):
    def __init__(self):
        super(MinimumToZero, self).__init__()

    def forward(self, x):
        return x - torch.min(x)


class MinimumToZeroVectorized(nn.Module):
    """
    Vectorized per-channel minimum subtraction for (C, H, W) tensors.
    Each channel's minimum is subtracted independently.
    """
    def __init__(self):
        super(MinimumToZeroVectorized, self).__init__()

    def forward(self, x):
        # x shape: (C, H, W)
        channel_min = x.amin(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        return x - channel_min