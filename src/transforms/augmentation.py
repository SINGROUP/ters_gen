import torch 
import torchvision.transforms.functional as TF
import random


class AugmentTransform:
    def __init__(self, p_rotate = 0.5, p_hflip = 0.5, p_vflip = 0.5,  p_gauss = 0.5, gauss_std_range = (0.01, 0.1)):
        self.p_rotate = p_rotate
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_gauss = p_gauss
        self.gauss_std_range = gauss_std_range  # tuple: (min_std, max_std)


    def __call__(self, img, mask):
        if random.random() < self.p_rotate:
            k = random.choice([0, 1, 2, 3])  # 0: no rotation, 1: 90 degrees, 2: 180 degrees, 3: 270 degrees
            img = torch.rot90(img, k=k, dims=(1, 2))
            mask = torch.rot90(mask, k=k, dims=(1, 2))

        if random.random() < self.p_hflip:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() < self.p_vflip:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        if random.random() < self.p_gauss:
            gauss_std = random.uniform(*self.gauss_std_range)
            noise = torch.randn_like(img) * gauss_std
            img = img + noise
            img = torch.clamp(img, 0, 1)

        return img, mask