from src.datasets import Ters_dataset_filtered_skip
from src.metrics import Metrics
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

from src.transforms import Normalize, MinimumToZero

import torchvision.transforms as transforms

# ——— Dataset & model setup ———
suffix = 'val'
data_path = f"/scratch/phys/sin/sethih1/data_files/planar_molecules_256/{suffix}/"
#data_path = "/scratch/phys/sin/sethih1/data_files/plane_third_group_images_nr_256_new/"

num_channels = 400
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

ters_loader = DataLoader(ters_set, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(
    '/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images/hyperopt/config2/seg_bs_16_lr_0.00024002900476800525_loss_dice_loss.pt',
    #'/scratch/phys/sin/sethih1/models/planar_256/config7/seg_bs_32_lr_0.0001_loss_dice_loss.pt',
    map_location=device
)
model.eval()

# ——— Collect IoUs, Dice coefficients, images, masks & preds ———
ious = []
dice_coeffs = []
images_list = []
masks_list = []
preds_list = []
filename_list = []

batch = next(iter(ters_loader))
print(len(batch))                    # should be 6
for i, elem in enumerate(batch):
    print(i, type(elem), getattr(elem, 'shape', len(elem)))

with torch.no_grad():
    for filename, _, _, images, _, masks in tqdm(ters_loader, desc="Eval Metrics"):

       
        images = images.to(device)
        masks  = masks.to(device)


        probs = model(images)                            # → (B,1,H,W)
        #probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).long().squeeze(1)          # → (B,H,W)

        for i in range(masks.size(0)):
            pred_i = preds[i]
            mask_i = masks[i]

            pred_flat = pred_i.view(-1)
            mask_flat = mask_i.view(-1)

            # Calculate Intersection and Union for IoU
            inter = torch.logical_and(pred_flat==1, mask_flat==1).sum().float()
            union = torch.logical_or(pred_flat==1, mask_flat==1).sum().float()
            iou   = (inter / (union + 1e-6)).item()
            ious.append(iou)

            # Calculate Dice Coefficient
            dice = (2 * inter / (torch.sum(pred_flat) + torch.sum(mask_flat) + 1e-6)).item()
            dice_coeffs.append(dice)

            images_list.append(images[i].cpu())
            masks_list.append(mask_i.cpu().squeeze(0).numpy())
            preds_list.append(pred_i.cpu().squeeze(0).numpy())
            filename_list.append(filename[i])
            
        


metrics = Metrics(model, ters_loader, config=None)  # config can be None if not used

print(preds_list[0].shape, masks_list[0].shape)
print(len(preds_list), len(masks_list))


all_ground_truths = np.concatenate(masks_list, axis=0)
all_predictions = np.concatenate(preds_list, axis=0)
#print("Ground Truth: ", all_ground_truths)
#print("Predictions: ", all_predictions)
# Initialize Metrics class
metrics = Metrics(model=model, data={"pred": all_predictions, "ground_truth": all_ground_truths}, config={})

# Compute metrics
results = metrics.evaluate()

# Print metrics
print("Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")


# ——— Compute and print mean IoU and Dice Coefficient ———
mean_iou = sum(ious) / len(ious)
mean_dice = sum(dice_coeffs) / len(dice_coeffs)
print(f"Mean IoU over {len(ious)} samples: {mean_iou:.4f}")
print(f"Mean Dice Coefficient over {len(dice_coeffs)} samples: {mean_dice:.4f}")

# ——— Plot & save overall histograms ———
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(ious, bins=30, range=(0, 1), edgecolor='black')
plt.title(f"IoU Distribution (mean={mean_iou:.3f})")
plt.xlabel("IoU")
plt.ylabel("Count")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(dice_coeffs, bins=30, range=(0, 1), edgecolor='black')
plt.title(f"Dice Coefficient Distribution (mean={mean_dice:.3f})")
plt.xlabel("Dice Coefficient")
plt.ylabel("Count")
plt.grid(True)

plt.tight_layout()
plt.savefig(f"iou_dice_histograms_{suffix}.png")
plt.close()

# ——— Bin samples by Dice Coefficient and save up to 5 per bin ———
bin_edges_dice = [i * 0.1 for i in range(11)]  # [0.0, 0.1, ..., 1.0]
bins_dice = defaultdict(list)

for idx, val in enumerate(dice_coeffs):
    bin_idx = min(int(val * 10), 9)   # clamp Dice=1.0 into last bin
    bins_dice[bin_idx].append(idx)

os.makedirs("dice_bins", exist_ok=True)

for bin_idx in range(10):
    sample_idxs = bins_dice.get(bin_idx, [])
    if not sample_idxs:
        continue

    # sort by Dice Coefficient ascending within this bin
    sample_idxs.sort(key=lambda i: dice_coeffs[i])

    low, high = bin_edges_dice[bin_idx], bin_edges_dice[bin_idx+1]
    for rank, idx in enumerate(sample_idxs[:5], start=1):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Input = mean over spectral channels
        avg_chan = torch.mean(images_list[idx], dim=0)
        axs[0].imshow(avg_chan.squeeze(), cmap="gray")
        axs[0].set_title("Average TERS intensity observed")

        # Ground truth mask
        axs[1].imshow(masks_list[idx].squeeze(), cmap="gray")
        axs[1].set_title("Ground Truth")

        # Predicted mask
        axs[2].imshow(preds_list[idx].squeeze(), cmap="gray")
        axs[2].set_title(f"Pred (IoU={ious[idx]:.3f}, Dice={dice_coeffs[idx]:.3f})")

        for ax in axs:
            ax.axis("off")

        fig.suptitle(f'Molecule id: {filename_list[idx]}')
        plt.tight_layout()
        os.makedirs(f"dice_bins_{suffix}", exist_ok=True)
        fname = f"dice_bins_{suffix}/{low:.1f}-{high:.1f}_rank{rank}_iou{ious[idx]:.3f}_dice{dice_coeffs[idx]:.3f}.png"
        plt.savefig(fname)
        plt.close()

print("Saved up to 5 images per Dice Coefficient bin in 'dice_bins/'")