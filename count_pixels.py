import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.datasets.ters_image_to_image_sh import Ters_dataset_filtered_skip
from src.transforms import Normalize, MinimumToZero

# … your config, transforms, etc …
data_transform = transforms.Compose([Normalize(), MinimumToZero()])



# 1) Build your dataset & loader
train_ds = Ters_dataset_filtered_skip(
    filename='/scratch/phys/sin/sethih1/data_files/all_group_plane_fchk_split_images_ters/train',
    frequency_range=[0, 4000],
    num_channels=5,
    std_deviation_multiplier=2, 
    sg_ch=False, 
    circle_radius=5,
    t_image=data_transform, 
    train_aug=False,
)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)

# 2) Prepare a counter for each class
n_classes = 5
pixel_counts = torch.zeros(n_classes, dtype=torch.long)

# 3) Iterate and accumulate
for batch in train_loader:
    # Assume your batch returns a dict with the mask under 'mask'; 
    # or if it's tuple-based: images, masks = batch

    images, frequencies, tgt_image = batch
    masks = tgt_image.argmax(dim=1)  # Assuming tgt_image is [B, C, H, W] and you want the class index
    
    # If masks are one‑hot [B, C, H, W], convert:
    # masks = masks.argmax(dim=1)
    
    # Count pixels per class in this batch
    for cls in range(n_classes):
        pixel_counts[cls] += torch.sum(masks == cls).item()

# 4) Print out results
print("Per-class pixel counts:")
for cls, cnt in enumerate(pixel_counts):
    print(f"  Class {cls}: {cnt} pixels")

# 5) (Optional) Compute median‑frequency weights or log‑inverse weights:
freq = pixel_counts.float() / pixel_counts.sum()
median = freq.median()
alpha_mfb = median / freq               # median frequency balancing
alpha_loginv = 1.0 / torch.log(1.02 + freq)  # ENet-style

print("\nExample α via median-frequency balancing:")
print(alpha_mfb)
print("\nExample α via log-inverse weighting:")
print(alpha_loginv)
