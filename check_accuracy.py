from src.datasets import Ters_dataset_filtered_skip
from src.metrics import Metrics

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from collections import defaultdict



# ——— Dataset & model setup ———
data_path = "/scratch/phys/sin/sethih1/data_files/planar_molecules_256"
num_channels = 400
sg_ch = True

ters_set = Ters_dataset_filtered_skip(
    filename=data_path,
    frequency_range=[0, 4000],
    num_channels=num_channels,
    std_deviation_multiplier=2,
    sg_ch=sg_ch,
    t_image=None,
    t_freq=None
)

ters_loader = DataLoader(ters_set, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(
    '/scratch/phys/sin/sethih1/models/planar_256/config7/seg_bs_32_lr_0.0001_loss_dice_loss.pt',
    map_location=device
)
model.eval()


# ——— Collect IoUs, images, masks & preds ———
ious = []
images_list = []
masks_list = []
preds_list = []

with torch.no_grad():
    for images, _, masks in tqdm(ters_loader, desc="Eval IoU"):
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)                            # → (B,1,H,W)
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).long().squeeze(1)          # → (B,H,W)

        for i in range(masks.size(0)):
            pred_i = preds[i]
            mask_i = masks[i]

            pred_flat = pred_i.view(-1)
            mask_flat = mask_i.view(-1)

            inter = torch.logical_and(pred_flat==1, mask_flat==1).sum().float()
            union = torch.logical_or(pred_flat==1, mask_flat==1).sum().float()
            iou   = (inter / (union + 1e-6)).item()

            ious.append(iou)
            images_list.append(images[i].cpu())
            masks_list.append(mask_i.cpu())
            preds_list.append(pred_i.cpu())




metrics = Metrics(model, ters_loader, config=None)  # config can be None if not used

data={"pred": preds_list, "ground_truth": masks_list}

# Initialize Metrics class
metrics = Metrics(model=model, data=data, config={})
# Compute metrics
results = metrics.evaluate()

# Print metrics
print("Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")


# ——— Compute and print mean IoU ———
mean_iou = sum(ious) / len(ious)
print(f"Mean IoU over {len(ious)} samples: {mean_iou:.4f}")

# ——— Plot & save overall histogram ———
plt.figure(figsize=(8,6))
plt.hist(ious, bins=30, range=(0,1), edgecolor='black')
plt.title(f"IoU Distribution (mean={mean_iou:.3f})")
plt.xlabel("IoU")
plt.ylabel("Count")
plt.grid(True)
plt.savefig("iou_histogram.png")
plt.close()

# ——— Bin samples by IoU and save up to 5 per bin ———
bin_edges = [i * 0.1 for i in range(11)]  # [0.0, 0.1, ..., 1.0]
bins = defaultdict(list)

for idx, val in enumerate(ious):
    bin_idx = min(int(val * 10), 9)   # clamp IoU=1.0 into last bin
    bins[bin_idx].append(idx)

os.makedirs("iou_bins", exist_ok=True)

for bin_idx in range(10):
    sample_idxs = bins.get(bin_idx, [])
    if not sample_idxs:
        continue

    # sort by IoU ascending within this bin
    sample_idxs.sort(key=lambda i: ious[i])

    low, high = bin_edges[bin_idx], bin_edges[bin_idx+1]
    for rank, idx in enumerate(sample_idxs[:5], start=1):
        fig, axs = plt.subplots(1, 3, figsize=(12,4))

        # Input = mean over spectral channels
        avg_chan = torch.mean(images_list[idx], dim=0)
        axs[0].imshow(avg_chan.squeeze(), cmap="gray")
        axs[0].set_title("Input")

        # Ground truth mask
        axs[1].imshow(masks_list[idx].squeeze(), cmap="gray")
        axs[1].set_title("Ground Truth")

        # Predicted mask
        axs[2].imshow(preds_list[idx].squeeze(), cmap="gray")
        axs[2].set_title(f"Pred (IoU={ious[idx]:.3f})")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        fname = f"iou_bins/{low:.1f}-{high:.1f}_rank{rank}_iou{ious[idx]:.3f}.png"
        plt.savefig(fname)
        plt.close()

print("Saved up to 5 images per IoU bin in 'iou_bins/'")
