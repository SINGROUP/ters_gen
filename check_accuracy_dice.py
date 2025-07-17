from src.datasets import Ters_dataset_filtered_skip
from src.metrics import Metrics
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

import os
from collections import defaultdict

from src.transforms import Normalize, MinimumToZero

import torchvision.transforms as transforms

# ——— Dataset & model setup ———
suffix = 'val'
data_path = f"/scratch/phys/sin/sethih1/data_files/all_group_plane_fchk_split_images_ters/{suffix}/"

suffix = f'config7_{suffix}'
#data_path = '/home/sethih1/masque_new/masque/check/'
#data_path = "/scratch/phys/sin/sethih1/data_files/plane_third_group_images_nr_256_new/"

num_channels = 100
sg_ch = False

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
    #'/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images_ters/composnet/hyperopt/config3/seg_bs_16_lr_0.0005246261149425009_loss_dice_loss.pt',
    '/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images_ters/composnet/hyperopt/augmented/config7/seg_bs_16_lr_0.0008078921183289167_loss_dice_loss.pt',
    #'/scratch/phys/sin/sethih1/models/composnet/all_group_plane_fchk_split_images_ters/hyperopt/config3/seg_bs_16_lr_0.00015629566600182425_loss_dice_loss.pt',
    #'/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images/hyperopt/config2/seg_bs_16_lr_0.00024002900476800525_loss_dice_loss.pt',
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


        probs = model(images)                           
        probs  = torch.sigmoid(probs)  # Apply sigmoid to get probabilities
        preds  = (probs > 0.5).long()
        print('probs shape:', probs.shape)
        print('preds shape:', preds.shape) 


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
            
        


#metrics = Metrics(model, ters_loader, config=None)  # config can be None if not used

print(preds_list[0].shape, masks_list[0].shape)
print(len(preds_list), len(masks_list))


all_ground_truths = np.concatenate(masks_list, axis=0)
all_predictions = np.concatenate(preds_list, axis=0)


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



    # Custom colormap: [Background, Hydrogen, Carbon, Oxygen, Nitrogen]
    label_colors = [
    (0, 0, 0),           # 0 - Background (black)
    (1.0, 1.0, 1.0),     # 1 - Hydrogen (white)
    (0.5, 0.5, 0.5),     # 2 - Carbon (grey)
    (0.0, 0.0, 1.0),      # 3 - Nitrogen (blue)
    (1.0, 0.0, 0.0),     # 4 - Oxygen (red)
    ]
    label_cmap = ListedColormap(label_colors)

    class_labels = ["Background", "H", "C","N", "O" ]
    #colors = [plt.cm.tab20(i) for i in range(1, 5)]  # Skip background (index 0)
    colors = label_colors
    # Create legend handles as dots only
    legend_handles = [
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label=label,
        markerfacecolor=col,
        markersize=8
    )
    for label, col in zip(class_labels, colors)]


    def add_bg(one_hot):
        # one_hot: (C, H, W), C=4 (H, C, N, O)
        bg = (one_hot.sum(axis=0) == 0)        # True where no class
        label = np.argmax(one_hot, axis=0) + 1 # temporarily 1–4
        label[bg] = 0                          # set empty → 0
        return label                           # now: 0=BG, 1=H,2=C,3=N,4=O
    

    
    for rank, idx in enumerate(sample_idxs[:5], start=1):
        fig, axs = plt.subplots(6, 3, figsize=(25, 15))

        # Input = mean over spectral channels
        avg_chan = torch.mean(images_list[idx], dim=0)
        axs[0, 0].imshow(avg_chan.squeeze(), cmap="viridis")
        axs[0, 0].set_title("Average TERS intensity observed")


        # Ground truth mask

        mask = masks_list[idx]
        mask_n = np.zeros((mask.shape[0]+1, mask.shape[1], mask.shape[2]), dtype=int)
        mask_n[1:, :, :] = mask.astype(int)
        gt_map = add_bg(masks_list[idx])
        axs[0, 1].imshow(gt_map, cmap=label_cmap)
        axs[0, 1].set_title("Ground Truth")

        # axs[1].legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=4)

        # Predicted mask
        mask = preds_list[idx]
        mask_n = np.zeros((mask.shape[0]+1, mask.shape[1], mask.shape[2]), dtype = int)
        mask_n[1:, :, :] = mask.astype(int)
        pred_map = add_bg(preds_list[idx])
        axs[0,2].imshow(pred_map, cmap=label_cmap)
        axs[0,2].set_title(f"Pred (IoU={ious[idx]:.3f}, Dice={dice_coeffs[idx]:.3f})")
        print("Unique values in GT and Pred maps:")

        print(np.unique(gt_map), np.unique(pred_map))

        #axs[2].legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=4)

        for i in range(preds_list[idx].shape[0]):

            gt_channel = (masks_list[idx][i, :, :] * (i+1)).astype(int)
            pred_channel = (preds_list[idx][i, :, :] * (i+1)).astype(int)

            axs[i+1,1 ].imshow(gt_channel, cmap=label_cmap, vmin = 0, vmax = 4)
            axs[i+1,1].set_title(f"Ground Truth Channel {class_labels[i+1]}")
            axs[i+1,2].imshow(pred_channel, cmap=label_cmap, vmin = 0, vmax = 4)
            axs[i+1,2].set_title(f"Predicted Channel {class_labels[i+1]}")

            

        for ax in axs.flatten():
            ax.axis("off")

        fig.suptitle(f'Molecule id: {filename_list[idx]}')
        plt.tight_layout()
        os.makedirs(f"dice_bins_{suffix}", exist_ok=True)
        fname = f"dice_bins_{suffix}/{low:.1f}-{high:.1f}_rank{rank}_iou{ious[idx]:.3f}_dice{dice_coeffs[idx]:.3f}.png"
        plt.savefig(fname)
        plt.close()



# ——— Per-class accuracy, IoU, Dice ———
n_classes = masks_list[0].shape[0]
tp = np.zeros(n_classes, dtype=np.int64)
tn = np.zeros(n_classes, dtype=np.int64)
fp = np.zeros(n_classes, dtype=np.int64)
fn = np.zeros(n_classes, dtype=np.int64)

for gt, pr in zip(masks_list, preds_list):
    gt_f = gt.reshape(n_classes, -1)
    pr_f = pr.reshape(n_classes, -1)
    for c in range(n_classes):
        g, p = gt_f[c], pr_f[c]
        tp[c] += np.logical_and(p==1, g==1).sum()
        tn[c] += np.logical_and(p==0, g==0).sum()
        fp[c] += np.logical_and(p==1, g==0).sum()
        fn[c] += np.logical_and(p==0, g==1).sum()

acc = (tp+tn) / (tp+tn+fp+fn + 1e-12)
iou_c = tp / (tp+fp+fn + 1e-12)
dice_c = 2*tp / (2*tp + fp+fn + 1e-12)


labels = class_labels[1:]   # ['H','C','N','O']
values = dice_c             # e.g. [0.80, 0.75, 0.60, 0.45]
x = np.arange(len(values))  # array([0,1,2,3])

plt.figure(figsize=(6,4))
plt.bar(x, values, tick_label=labels)
plt.ylim(0,1)
plt.xlabel('Class')
plt.ylabel('Dice Coefficient')
plt.title('Dice Coefficient per Class')

# Optionally annotate each bar with its value
for xi, v in zip(x, values):
    plt.text(xi, v + 0.02, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.savefig(f"dice_per_class_{suffix}.png")
plt.show()

print("Saved up to 5 images per Dice Coefficient bin in 'dice_bins/'")