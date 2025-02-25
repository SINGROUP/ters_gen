import torch
import torch.nn as nn

class AddNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.01*1600, seed=0):
        """
        Adds Gaussian noise to the input data.
        :param mean: Mean of the Gaussian noise to be added.
        :param std: Standard deviation of the Gaussian noise to be added.
        :param seed: Seed for random number generation for reproducibility.
        """
        super(AddNoise, self).__init__()
        self.mean = mean
        self.std = std
        self.seed = seed

    def forward(self, x):
        # Set the random seed if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        # Generate noise with the same shape as x
        noise = torch.randn_like(x) * self.std + self.mean
        # Add noise to the input data
        return x + noise




# print("Data shifted to minimum zero:", x_min_to_zero)

# # Apply noise addition with a random seed
# add_noise = AddNoise(mean=0.0, std=0.1, seed=42)
# x_noisy = add_noise(x_min_to_zero)
# print("Data with added noise:", x_noisy)
