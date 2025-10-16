import os


from src.datasets import Ters_dataset_filtered_skip
from src.metrics import Metrics
from src.transforms import Normalize, MinimumToZero
from notebooks.utils.visualization import molecule_visualization_image
from notebooks.utils.read_files import read_npz
from notebooks.utils.planarity import pca, planarity


import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from tqdm import tqdm

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from scipy.stats import gaussian_kde


import random


# ——— Dataset & model setup ———
suffix = 'val'
data_path = f"/scratch/phys/sin/sethih1/data_files/final_data_used/{suffix}/"

num_channels = 100 #400
sg_ch = True

data_transform = transforms.Compose([Normalize(), MinimumToZero()])

ters_set = Ters_dataset_filtered_skip(
    filename=data_path,
    frequency_range=[0, 4000],
    num_channels=num_channels,
    std_deviation_multiplier=2,
    sg_ch=sg_ch,
    t_image=data_transform,
    t_freq=None, 
    flag=True
)

print(f"Number of samples: {len(ters_set)}")

ters_loader = DataLoader(ters_set, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(
   '/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images_ters/posnet_hyperopt_all_50_epochs_just_aug_val/augmented/config_hypopt_all/best_model.pt',
    map_location=device
)
model.eval()



# ——— Collect IoUs, Dice coefficients, images, masks & preds ———
# Lists to store actual image/mask/prediction data
images_list, masks_list, preds_list = [], [], []

# List to store records for the DataFrame
records = []

device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.no_grad():
    for filename, atom_count, _, images, _, masks in tqdm(ters_loader, desc="Eval Metrics"):
        images = images.to(device)
        masks  = masks.to(device)

        probs = model(images)                             # → (B,1,H,W)
        preds = (probs > 0.5).long().squeeze(1)          # → (B,H,W)


        

        for i in range(masks.size(0)):
            pred_i = preds[i]
            mask_i = masks[i]

            pred_flat = pred_i.view(-1)
            mask_flat = mask_i.view(-1)

            inter = torch.logical_and(pred_flat==1, mask_flat==1).sum().float()
            union = torch.logical_or(pred_flat==1, mask_flat==1).sum().float()
            iou = (inter / (union + 1e-6)).item()

            dice = (2 * inter / (torch.sum(pred_flat) + torch.sum(mask_flat) + 1e-6)).item()

            # Append actual data to lists
            images_list.append(images[i].cpu())
            masks_list.append(mask_i.cpu().squeeze(0).numpy())
            preds_list.append(pred_i.cpu().squeeze(0).numpy())


            # Compute planarity metrics
            npz_file = os.path.join(data_path, f"{filename[i]}.npz")
            coords, atomic_numbers = read_npz(npz_file)
            eigvals, eigvecs, X = pca(coords)
            planarity_pca, planarity_rms, rmsd_val = planarity(eigvals, eigvecs, X)


            
            # Save record
            idx = len(images_list) - 1
            records.append({
                "index": idx,
                "filename": filename[i],
                "atom_count": atom_count[i],
                "iou": iou,
                "dice": dice,
                "planarity_pca": planarity_pca,
                "planarity_rms": planarity_rms,
                "rmsd": rmsd_val
            })


# Convert to DataFrame
df_metrics = pd.DataFrame(records)

# Save metrics only
df_metrics.to_csv("ters_metrics.csv", index=False)

# Example: access image/mask/prediction by index
example_idx = 0
image_example = images_list[example_idx]
mask_example = masks_list[example_idx]
pred_example = preds_list[example_idx]

print(df_metrics.head())


metrics = Metrics(model, ters_loader, config=None)  # config can be None if not used


all_ground_truths = np.concatenate(masks_list, axis=0)
all_predictions = np.concatenate(preds_list, axis=0)


# Initialize Metrics class
metrics = Metrics(model=model, data={"pred": all_predictions, "ground_truth": all_ground_truths}, config={})

# Compute metrics
results = metrics.evaluate()

# Print metrics
print("Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

df_metrics.to_csv('check.csv')