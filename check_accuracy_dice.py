from src.datasets import Ters_dataset_filtered_skip
from src.metrics import Metrics
from notebooks.utils.visualization import molecule_visualization_image


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import os
from collections import defaultdict
from scipy.stats import gaussian_kde

from src.transforms import Normalize, MinimumToZero

import torchvision.transforms as transforms

import random


# ——— Dataset & model setup ———
suffix = 'test'

rms = 0.05
# data_path = f"/scratch/phys/sin/sethih1/data_files/final_data_used/{suffix}/"
# model_path = '/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images_ters/posnet_hyperopt_all_50_epochs_just_aug_val/augmented/config_hypopt_all/best_model.pt'
# dir_viz = '/scratch/phys/sin/sethih1/runs_ters/all_group_plane_fchk_split_images_ters/posnet_hyperopt_all_50_epochs_just_aug_val/augmented/config_hypopt_all/'

data_path = f"/scratch/phys/sin/sethih1/data_files/combined_npz_images_32x32/{suffix}/"
model_path = '/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images_ters/32x32/posnet_hyperopt_all_50_epochs_just_aug_val/augmented/config_hypopt_all/best_model.pt'
dir_viz = '/scratch/phys/sin/sethih1/runs_ters/all_group_plane_fchk_split_images_ters/32x32/posnet_hyperopt_all_50_epochs_just_aug_val/augmented/config_hypopt_all/'


data_path = f"/scratch/phys/sin/sethih1/Extended_TERS_data/planar_oct_2025/planar_again/planar_npz_{rms}/{suffix}/"
model_path = '/scratch/phys/sin/sethih1/Extended_TERS_data/run_planar_again/run_planar_npz_0.05/models/seg_trial8_bs16_lr7e-04_lossdice_loss.pt'
dir_viz = '/scratch/phys/sin/sethih1/Extended_TERS_data/run_planar_again/planar_comparison_viz/dice/'
           

#data_path = f"/scratch/phys/sin/sethih1/data_files/all_group_plane_fchk_split_images_ters/{suffix}/"

#data_path = '/home/sethih1/masque_new/masque/check/'

#data_path = f"/scratch/phys/sin/sethih1/data_files/planar_molecules_256/{suffix}/"
#data_path = "/scratch/phys/sin/sethih1/data_files/plane_third_group_images_nr_256_new/"

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
    #'/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images_ters/hyperopt_old/config2/seg_bs_16_lr_0.0005116967471012849_loss_dice_loss.pt',
    #'/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images_ters/hyperopt/config2/seg_bs_16_lr_0.00040199284987490726_loss_dice_loss.pt',
    #'/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images_ters/posnet_hyperopt_all_50_epochs/augmented/config_hypopt_all/best_model.pt',
    #'/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images_ters/posnet_hyperopt_all_50_epochs_just_aug/augmented/config_hypopt_all/best_model.pt',
    #'/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images_ters/posnet_hyperopt_all_50_epochs_just_aug_val/augmented/config_hypopt_all/best_model.pt',
    model_path, 
    map_location=device
)

    #'/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images_ters/hyperopt/config2/seg_bs_32_lr_0.00045407163452873707_loss_dice_loss.pt',
    #'/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images/hyperopt/config2/seg_bs_16_lr_0.00024002900476800525_loss_dice_loss.pt',
    #'/scratch/phys/sin/sethih1/models/all_group_plane_fchk_split_images/hyperopt/config2/seg_bs_16_lr_0.00037197401223588886_loss_dice_loss.pt',
    #'/scratch/phys/sin/sethih1/models/planar_256/config7/seg_bs_32_lr_0.0001_loss_dice_loss.pt',
model.eval()

# ——— Collect IoUs, Dice coefficients, images, masks & preds ———
ious = []
dice_coeffs = []
images_list = []
masks_list = []
preds_list = []
filename_list = []


atom_counts = []

batch = next(iter(ters_loader))
print(len(batch))                    # should be 6
for i, elem in enumerate(batch):
    print(i, type(elem), getattr(elem, 'shape', len(elem)))

with torch.no_grad():
    for filename, atom_count, _, images, _, masks in tqdm(ters_loader, desc="Eval Metrics"):

       
        images = images.to(device)
        masks  = masks.to(device)


        probs = model(images)                            # → (B,1,H,W)
        probs  = torch.sigmoid(probs)
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

            # Save number of atoms
            #atom_count = atom_pos[i].shape[0]  # assuming shape = (N_atoms, 3)
            atom_counts.append(atom_count[i])
            
        


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

        #path = '/scratch/phys/sin/sethih1/runs_ters/all_group_plane_fchk_split_images_ters/posnet_hyperopt_all_50_epochs/augmented/config_hypopt_all/' + f"dice_bins_{suffix}"
        #path = '/scratch/phys/sin/sethih1/runs_ters/all_group_plane_fchk_split_images_ters/posnet_hyperopt_all_50_epochs_just_aug_val/augmented/config_hypopt_all/' + f"dice_bins_{suffix}"
        path = dir_viz + f"dice_bins_{suffix}"

        os.makedirs(path, exist_ok=True)
        fname = f"{path}/{low:.1f}-{high:.1f}_rank{rank}_iou{ious[idx]:.3f}_dice{dice_coeffs[idx]:.3f}.png"
        plt.savefig(fname)
        plt.close()

print("Saved up to 5 images per Dice Coefficient bin in 'dice_bins/'")



# ——— Save 4 random samples from the best Dice Coefficient ones, as separate images ———
TOP_N = 10
NUM_SAMPLES = 4
random.seed(42)
sorted_indices = sorted(range(len(dice_coeffs)), key=lambda i: dice_coeffs[i], reverse=True)
best_indices = sorted_indices[:TOP_N]
chosen_indices = random.sample(best_indices, min(NUM_SAMPLES, len(best_indices)))
best_samples_path = os.path.join(path, f"best_samples_{suffix}")
os.makedirs(best_samples_path, exist_ok=True)

for idx in chosen_indices:
    avg_intensity = torch.mean(images_list[idx], dim=0).squeeze().cpu().numpy()
    gt_mask = masks_list[idx].squeeze()
    pred_mask = preds_list[idx].squeeze()
    mol_id = filename_list[idx]

    img_tensor = images_list[idx]


    img_min = img_tensor.view(img_tensor.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
    img_max = img_tensor.view(img_tensor.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
    img_tensor_norm = (img_tensor - img_min) / (img_max - img_min + 1e-8)

    # Spectral sum across normalized channels
    spectral_sum = torch.sum(img_tensor_norm, dim=0).squeeze()

    # Normalize the spectral sum map for visualization
    spectral_sum_norm = (spectral_sum - spectral_sum.min()) / (spectral_sum.max() - spectral_sum.min() + 1e-8)
    spectral_sum_np = spectral_sum_norm.cpu().numpy()

    # For averaged TERS intensity map – visually intuitive and colourblind-friendly
    plt.imsave(f"{best_samples_path}/sample_{mol_id}_avg.png", avg_intensity, cmap="jet", origin='lower')


    plt.imsave(f"{best_samples_path}/sample_{mol_id}_spectralsum.png", spectral_sum_np, cmap="jet", origin='lower')
    
    # For ground truth – red tones for clear separation
    plt.imsave(f"{best_samples_path}/sample_{mol_id}_ref.png", gt_mask, cmap="Reds", origin='lower')
    
    # For prediction – blue tones for contrast with GT
    plt.imsave(f"{best_samples_path}/sample_{mol_id}_pred.png", pred_mask, cmap="Blues", origin='lower')



print(f"Saved {len(chosen_indices)} best samples in '{best_samples_path}' (avg, ref, pred images separately)")



# --- 0.8–0.9 Dice range ---
range_08_09 = [i for i, d in enumerate(dice_coeffs) if 0.8 <= d < 0.9]
chosen_indices_08_09 = random.sample(range_08_09, min(NUM_SAMPLES, len(range_08_09)))
range_path_08_09 = os.path.join(path, f"samples_dice_08_09")
os.makedirs(range_path_08_09, exist_ok=True)

for idx in chosen_indices_08_09:
    avg_intensity = torch.mean(images_list[idx], dim=0).squeeze().cpu().numpy()
    gt_mask = masks_list[idx].squeeze()
    pred_mask = preds_list[idx].squeeze()
    mol_id = filename_list[idx]

    img_tensor = images_list[idx]
    img_min = img_tensor.view(img_tensor.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
    img_max = img_tensor.view(img_tensor.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
    img_tensor_norm = (img_tensor - img_min) / (img_max - img_min + 1e-8)
    spectral_sum = torch.sum(img_tensor_norm, dim=0).squeeze()
    spectral_sum_norm = (spectral_sum - spectral_sum.min()) / (spectral_sum.max() - spectral_sum.min() + 1e-8)
    spectral_sum_np = spectral_sum_norm.cpu().numpy()

    plt.imsave(f"{range_path_08_09}/sample_{mol_id}_avg.png", avg_intensity, cmap="jet", origin='lower')
    plt.imsave(f"{range_path_08_09}/sample_{mol_id}_spectralsum.png", spectral_sum_np, cmap="jet", origin='lower')
    plt.imsave(f"{range_path_08_09}/sample_{mol_id}_ref.png", gt_mask, cmap="Reds", origin='lower')
    plt.imsave(f"{range_path_08_09}/sample_{mol_id}_pred.png", pred_mask, cmap="Blues", origin='lower')

print(f"Saved {len(chosen_indices_08_09)} samples from 0.8–0.9 range in '{range_path_08_09}'")


# --- 0.7–0.8 Dice range ---
range_07_08 = [i for i, d in enumerate(dice_coeffs) if 0.7 <= d < 0.8]
chosen_indices_07_08 = random.sample(range_07_08, min(NUM_SAMPLES, len(range_07_08)))
range_path_07_08 = os.path.join(path, f"samples_dice_07_08")
os.makedirs(range_path_07_08, exist_ok=True)

for idx in chosen_indices_07_08:
    avg_intensity = torch.mean(images_list[idx], dim=0).squeeze().cpu().numpy()
    gt_mask = masks_list[idx].squeeze()
    pred_mask = preds_list[idx].squeeze()
    mol_id = filename_list[idx]

    img_tensor = images_list[idx]
    img_min = img_tensor.view(img_tensor.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
    img_max = img_tensor.view(img_tensor.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
    img_tensor_norm = (img_tensor - img_min) / (img_max - img_min + 1e-8)
    spectral_sum = torch.sum(img_tensor_norm, dim=0).squeeze()
    spectral_sum_norm = (spectral_sum - spectral_sum.min()) / (spectral_sum.max() - spectral_sum.min() + 1e-8)
    spectral_sum_np = spectral_sum_norm.cpu().numpy()

    plt.imsave(f"{range_path_07_08}/sample_{mol_id}_avg.png", avg_intensity, cmap="jet", origin='lower')
    plt.imsave(f"{range_path_07_08}/sample_{mol_id}_spectralsum.png", spectral_sum_np, cmap="jet", origin='lower')
    plt.imsave(f"{range_path_07_08}/sample_{mol_id}_ref.png", gt_mask, cmap="Reds", origin='lower')
    plt.imsave(f"{range_path_07_08}/sample_{mol_id}_pred.png", pred_mask, cmap="Blues", origin='lower')

print(f"Saved {len(chosen_indices_07_08)} samples from 0.7–0.8 range in '{range_path_07_08}'")



# --- 0.7–0.8 Dice range ---
range_07_08 = [i for i, d in enumerate(dice_coeffs) if 0.6 <= d < 0.7]
chosen_indices_07_08 = random.sample(range_07_08, min(NUM_SAMPLES, len(range_07_08)))
range_path_07_08 = os.path.join(path, f"samples_dice_06_07")
os.makedirs(range_path_07_08, exist_ok=True)

for idx in chosen_indices_07_08:
    avg_intensity = torch.mean(images_list[idx], dim=0).squeeze().cpu().numpy()
    gt_mask = masks_list[idx].squeeze()
    pred_mask = preds_list[idx].squeeze()
    mol_id = filename_list[idx]

    img_tensor = images_list[idx]
    img_min = img_tensor.view(img_tensor.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
    img_max = img_tensor.view(img_tensor.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
    img_tensor_norm = (img_tensor - img_min) / (img_max - img_min + 1e-8)
    spectral_sum = torch.sum(img_tensor_norm, dim=0).squeeze()
    spectral_sum_norm = (spectral_sum - spectral_sum.min()) / (spectral_sum.max() - spectral_sum.min() + 1e-8)
    spectral_sum_np = spectral_sum_norm.cpu().numpy()

    plt.imsave(f"{range_path_07_08}/sample_{mol_id}_avg.png", avg_intensity, cmap="jet", origin='lower')
    plt.imsave(f"{range_path_07_08}/sample_{mol_id}_spectralsum.png", spectral_sum_np, cmap="jet", origin='lower')
    plt.imsave(f"{range_path_07_08}/sample_{mol_id}_ref.png", gt_mask, cmap="Reds", origin='lower')
    plt.imsave(f"{range_path_07_08}/sample_{mol_id}_pred.png", pred_mask, cmap="Blues", origin='lower')

print(f"Saved {len(chosen_indices_07_08)} samples from 0.7–0.8 range in '{range_path_07_08}'")



import os
import random
import torch
import matplotlib.pyplot as plt

# Define ranges
dice_ranges = [
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3), 
    (0.5, 0.6)
]

for lower, upper in dice_ranges:
    # Filter indices for current Dice range
    range_indices = [i for i, d in enumerate(dice_coeffs) if lower <= d < upper]
    chosen_indices = random.sample(range_indices, min(NUM_SAMPLES, len(range_indices)))

    # Create output path
    range_path = os.path.join(path, f"samples_dice_{str(lower).replace('.', '')}_{str(upper).replace('.', '')}")
    os.makedirs(range_path, exist_ok=True)

    for idx in chosen_indices:
        avg_intensity = torch.mean(images_list[idx], dim=0).squeeze().cpu().numpy()
        gt_mask = masks_list[idx].squeeze()
        pred_mask = preds_list[idx].squeeze()
        mol_id = filename_list[idx]

        img_tensor = images_list[idx]
        img_min = img_tensor.view(img_tensor.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
        img_max = img_tensor.view(img_tensor.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
        img_tensor_norm = (img_tensor - img_min) / (img_max - img_min + 1e-8)
        spectral_sum = torch.sum(img_tensor_norm, dim=0).squeeze()
        spectral_sum_norm = (spectral_sum - spectral_sum.min()) / (spectral_sum.max() - spectral_sum.min() + 1e-8)
        spectral_sum_np = spectral_sum_norm.cpu().numpy()

        plt.imsave(f"{range_path}/sample_{mol_id}_avg.png", avg_intensity, cmap="jet", origin='lower')
        plt.imsave(f"{range_path}/sample_{mol_id}_spectralsum.png", spectral_sum_np, cmap="jet", origin='lower')
        plt.imsave(f"{range_path}/sample_{mol_id}_ref.png", gt_mask, cmap="Reds", origin='lower')
        plt.imsave(f"{range_path}/sample_{mol_id}_pred.png", pred_mask, cmap="Blues", origin='lower')

    print(f"Saved {len(chosen_indices)} samples from {lower}–{upper} range in '{range_path}'")





# Define Dice ranges to evaluate
dice_ranges = [
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.4, 0.5), 
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),  
    (0.8, 0.9), 
    (0.9, 1.0)
]

for lower, upper in dice_ranges:
    # Get indices in current Dice range
    range_indices = [i for i, d in enumerate(dice_coeffs) if lower <= d < upper]
    chosen_indices = random.sample(range_indices, min(NUM_SAMPLES, len(range_indices)))

    # Output directory
    range_path = os.path.join(path, f"group/samples_dice_{str(lower).replace('.', '')}_{str(upper).replace('.', '')}")
    os.makedirs(range_path, exist_ok=True)

    for idx in chosen_indices:
        mol_id = filename_list[idx]
        img_tensor = images_list[idx]
        gt_mask = masks_list[idx].squeeze()
        pred_mask = preds_list[idx].squeeze()

        # Normalize avg intensity
        avg_intensity = torch.mean(img_tensor, dim=0).squeeze().cpu().numpy()

        # Normalize spectral sum
        img_min = img_tensor.view(img_tensor.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
        img_max = img_tensor.view(img_tensor.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
        img_tensor_norm = (img_tensor - img_min) / (img_max - img_min + 1e-8)
        spectral_sum = torch.sum(img_tensor_norm, dim=0).squeeze()
        spectral_sum_norm = (spectral_sum - spectral_sum.min()) / (spectral_sum.max() - spectral_sum.min() + 1e-8)
        spectral_sum_np = spectral_sum_norm.cpu().numpy()

        npz_file = os.path.join(data_path, f'{filename_list[idx]}.npz')
        with np.load(npz_file) as data:
            atom_pos = data['atom_pos']
            atomic_numbers = data['atomic_numbers']

        # Convert masks to numpy for plotting
        gt_np = gt_mask
        pred_np = pred_mask

        # Compute metrics
        dice = dice_coeffs[idx]
        iou = ious[idx]

        # Plot
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"Sample: {mol_id}", fontsize=14)


        molecule_visualization_image(np.array(atom_pos), np.array(atomic_numbers), axes=axs[0])
        axs[0].set_title("Original Molecule")
        axs[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

        axs[0].set_frame_on(True)


        axs[1].imshow(avg_intensity, cmap="jet", origin='lower')
        axs[1].set_title("TERS Spectral Average")
        axs[1].axis('off')

        axs[2].imshow(gt_np, cmap="Reds", origin='lower')
        axs[2].set_title("Ground Truth")
        axs[2].axis('off')

        axs[3].imshow(pred_np, cmap="Blues", origin='lower')
        axs[3].set_title(f"Prediction\n(Dice: {dice:.3f}, IoU: {iou:.3f})")
        axs[3].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.subplots_adjust(wspace=0.3, top=0.85)
        out_file = os.path.join(range_path, f"sample_{mol_id}.png")
        plt.savefig(out_file, dpi=200)
        plt.close()

    print(f"Saved {len(chosen_indices)} samples from {lower:.1f}–{upper:.1f} range in '{range_path}'")


print("Dice coeffs: ", dice_coeffs)

print("Flattened: ")
#flat_atom_counts = [t.item() for sublist in atom_counts for t in sublist]

# Convert to tensor


print(len(atom_counts))


print(len(dice_coeffs))


# ——— Boxplot Dice vs. #atoms ———
# 1. build from scalars
cleaned_data = [
    {'num_atoms': n, 'dice': float(d)}
    for n, d in zip(atom_counts, dice_coeffs)
]
df = pd.DataFrame(cleaned_data)

# 2. (optional) enforce types
df['num_atoms'] = df['num_atoms'].astype(int)
df['dice']      = df['dice'].astype(float)



# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(dice_coeffs, bins=30, alpha=0.8, color='mediumorchid', edgecolor='black')
ax.set_xlabel('Dice Coefficient')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Dice Scores')
# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(path, "histogram.png"), dpi=300, bbox_inches='tight', transparent=True)
plt.close()


# 3. boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='num_atoms', y='dice')
plt.title("Dice Coefficient Distribution vs Number of Atoms")
plt.xlabel("Number of Atoms")
plt.ylabel("Dice Coefficient")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(path, f"dice_vs_atoms_boxplot_{suffix}.png"))
plt.close()



plt.figure(figsize=(10, 6))

plt.hist2d(
    atom_counts, dice_coeffs,
    bins=[range(min(atom_counts), max(atom_counts) + 2), 30],
    cmap='Purples'
)

plt.colorbar(label='Count')
plt.xlabel("Number of Atoms")
plt.ylabel("Dice Coefficient")
plt.title("2D Histogram: Dice Coefficient vs Number of Atoms")
plt.tight_layout()
plt.savefig(os.path.join(path, f"dice_vs_atoms_hist2d_{suffix}.png"))
plt.close()


plt.figure(figsize=(12, 6))

grouped = df.groupby('num_atoms')['dice']
for atom_count, group in grouped:
    plt.hist(
        group, bins=20, alpha=0.5,
        color='purple', label=f"{atom_count} atoms"
    )

plt.xlabel("Dice Coefficient")
plt.ylabel("Count")
plt.title("Dice Coefficient Histograms by Atom Count")
plt.legend(title="Number of Atoms", fontsize='small', title_fontsize='medium', loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(path, f"dice_hist_by_atom_count_{suffix}.png"))
plt.close()


# --- 1) Define atom-count bins and labels ---
bins = [0, 5, 10, 15, 20, 25, 30] 
labels = [f"{bins[i]+1}-{bins[i+1]}" for i in range(len(bins)-1)]

# Add a new column 'atom_range' to df
df['atom_range'] = pd.cut(
    df['num_atoms'], 
    bins=bins, 
    labels=labels, 
    include_lowest=True
)

# Also bin the dice scores into 10 equal-width intervals
dice_bins = np.linspace(0, 1, 11)
dice_labels = [f"{dice_bins[i]:.1f}-{dice_bins[i+1]:.1f}" for i in range(len(dice_bins)-1)]
df['dice_range'] = pd.cut(df['dice'], bins=dice_bins, labels=dice_labels, include_lowest=True)


# --- 2) Boxplot: Dice vs. Atom‑Count Range ---
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df, 
    x='atom_range', 
    y='dice',
    order=labels
)
plt.title("Dice Coefficient Distribution by Atom‑Count Range")
plt.xlabel("Number of Atoms (Binned)")
plt.ylabel("Dice Coefficient")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(path, f"dice_vs_atom_range_boxplot_{suffix}.png"))
plt.close()


# --- 3) Heatmap: Frequency by Atom‑Count Range × Dice‑Score Range ---
pivot = df.pivot_table(
    index='atom_range', 
    columns='dice_range', 
    aggfunc='size', 
    fill_value=0
)

plt.figure(figsize=(12, 6))
sns.heatmap(
    pivot, 
    cmap='Purples', 
    cbar_kws={'label': 'Sample Count'},
    linewidths=0.5,
    linecolor='white'
)
plt.title("Heatmap: Dice‑Score Frequency by Atom‑Count Range")
plt.xlabel("Dice Coefficient Range")
plt.ylabel("Atom‑Count Range")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(path, f"dice_vs_atom_range_heatmap_{suffix}.png"))
plt.close()


# --- 4) Overlaid Histograms: Dice Distribution per Atom‑Count Range ---
plt.figure(figsize=(12, 6))
for rng in labels:
    subset = df[df['atom_range'] == rng]['dice']
    if len(subset) == 0:
        continue
    plt.hist(
        subset, 
        bins=20, 
        alpha=0.5, 
        edgecolor='black', 
        label=rng
    )

plt.title("Dice Coefficient Distribution per Atom‑Count Range")
plt.xlabel("Dice Coefficient")
plt.ylabel("Count")
plt.legend(title="Atom‑Count Range", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(path, f"dice_hist_by_atom_range_{suffix}.png"))
plt.close()



# Optional: make output directory
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)


# --------------------------------------------------------
# 1) BUBBLE‑SCATTER with Density Contours
# --------------------------------------------------------
# 1.1 Compute pointwise density
xy = np.vstack([df['num_atoms'], df['dice']])
kde = gaussian_kde(xy)
densities = kde(xy)

plt.figure(figsize=(10, 6))
plt.scatter(
    df['num_atoms'], df['dice'],
    s=densities * densities.max() * 200,  # scale bubbles
    alpha=0.6,
    c=df['dice'],
    cmap='viridis',
    edgecolors='none'
)
# 1.2 Overlay KDE contours
sns.kdeplot(
    x=df['num_atoms'], y=df['dice'],
    levels=5, color='gray', linewidths=1.2, alpha=0.7
)

plt.colorbar(label='Dice Score')
plt.xlabel("Number of Atoms")
plt.ylabel("Dice Coefficient")
plt.title("Bubble‑Scatter: Dice vs Atoms (bubble size ∝ local density)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "bubble_scatter_density.png"))
plt.close()


# --------------------------------------------------------
# 2) RIDGELINE (“JOY”) PLOT of Dice by Atom‑Count Ranges
# --------------------------------------------------------
# 2.1 Bin your atom counts into ranges
bins = [0, 5, 10, 15, 20, 30, 50, 100]
labels = [f"{bins[i]+1}-{bins[i+1]}" for i in range(len(bins)-1)]
df['atom_range'] = pd.cut(
    df['num_atoms'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# 2.2 Build the FacetGrid
pal = sns.cubehelix_palette(len(labels), start=2, rot=0, dark=0.2, light=0.8)
g = sns.FacetGrid(
    df, row="atom_range", hue="atom_range",
    aspect=6, height=0.8, palette=pal
)
# 2.3 Map the KDEs
g.map(sns.kdeplot, "dice",
      clip_on=False,
      shade=True,
      alpha=0.7,
      linewidth=1.5)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# 2.4 Tidy up
g.fig.subplots_adjust(hspace=-0.3)
g.set_titles("")    # remove "atom_range = ..." headers
g.set(yticks=[])
g.set_xlabels("Dice Coefficient")
for ax, label in zip(g.axes.flatten(), labels):
    ax.text(-0.05, 0.2, label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold")

plt.suptitle("Ridgeline Plot: Dice Distributions by Molecule‑Size Range", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ridgeline_dice_by_atom_range.png"))
plt.close()


# --------------------------------------------------------
# 3) HEXBIN + MARGINAL HISTOGRAMS
# --------------------------------------------------------
fig = plt.figure(figsize=(8, 8))
grid = plt.GridSpec(4, 4, hspace=0.4, wspace=0.4)

main_ax = fig.add_subplot(grid[1:4, 0:3])
x_hist = fig.add_subplot(grid[0, 0:3], sharex=main_ax)
y_hist = fig.add_subplot(grid[1:4, 3], sharey=main_ax)

# 3.1 Main hexbin plot
hb = main_ax.hexbin(
    df['num_atoms'], df['dice'],
    gridsize=(20, 20),
    cmap='Purples',
    mincnt=1
)
cb = fig.colorbar(hb, ax=main_ax, label='Sample Count')
main_ax.set_xlabel('Number of Atoms')
main_ax.set_ylabel('Dice Coefficient')

# 3.2 Marginal histograms
x_hist.hist(df['num_atoms'], bins=20, edgecolor='black')
x_hist.axis('off')
y_hist.hist(df['dice'], bins=20, orientation='horizontal', edgecolor='black')
y_hist.axis('off')

plt.suptitle("Hexbin + Marginal Histograms: Dice vs #Atoms")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "hexbin_marginals.png"))
plt.close()



# Add this code to the end of your existing script

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
from collections import Counter
import seaborn as sns

def create_3d_surface_plot(atom_counts, dice_coeffs, save_path=None):
    """
    Create a 3D surface plot showing the relationship between:
    - Number of atoms (X-axis)
    - Dice coefficient (Y-axis) 
    - Frequency of occurrence (Z-axis/height)
    """
    
    # Convert to numpy arrays
    atom_counts = np.array(atom_counts)
    dice_coeffs = np.array(dice_coeffs)
    
    print(f"Data range - Atoms: {atom_counts.min()}-{atom_counts.max()}, "
          f"Dice: {dice_coeffs.min():.3f}-{dice_coeffs.max():.3f}")
    
    # Create 2D histogram to get frequencies
    atom_bins = 20  # Number of bins for atom counts
    dice_bins = 20  # Number of bins for dice scores
    
    # Define bin edges
    atom_edges = np.linspace(atom_counts.min(), atom_counts.max(), atom_bins + 1)
    dice_edges = np.linspace(0, 1, dice_bins + 1)
    
    # Calculate 2D histogram
    hist, atom_bin_edges, dice_bin_edges = np.histogram2d(
        atom_counts, dice_coeffs, bins=[atom_edges, dice_edges]
    )
    
    # Create meshgrids for plotting
    atom_centers = (atom_bin_edges[:-1] + atom_bin_edges[1:]) / 2
    dice_centers = (dice_bin_edges[:-1] + dice_bin_edges[1:]) / 2
    X, Y = np.meshgrid(atom_centers, dice_centers)
    Z = hist.T  # Transpose to match meshgrid orientation
    
    # Create the 3D plot
    fig = plt.figure(figsize=(15, 10))
    
    # Main 3D surface plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Surface plot with colormap
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    # Customize the plot
    ax1.set_xlabel('Number of Atoms', fontsize=12)
    ax1.set_ylabel('Dice Coefficient', fontsize=12)
    ax1.set_zlabel('Frequency', fontsize=12)
    ax1.set_title('3D Surface: Molecular Dice Coefficient Analysis', fontsize=14)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Alternative view: Wireframe
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot_wireframe(X, Y, Z, alpha=0.7, color='blue')
    ax2.set_xlabel('Number of Atoms')
    ax2.set_ylabel('Dice Coefficient')
    ax2.set_zlabel('Frequency')
    ax2.set_title('Wireframe View')
    
    # 2D heatmap view from above
    ax3 = fig.add_subplot(2, 2, 3)
    im = ax3.imshow(Z, extent=[atom_centers.min(), atom_centers.max(), 
                               dice_centers.min(), dice_centers.max()],
                    aspect='auto', origin='lower', cmap='viridis')
    ax3.set_xlabel('Number of Atoms')
    ax3.set_ylabel('Dice Coefficient')
    ax3.set_title('Top-down Heatmap')
    plt.colorbar(im, ax=ax3)
    
    # Contour plot
    ax4 = fig.add_subplot(2, 2, 4)
    contour = ax4.contour(X, Y, Z, levels=10)
    ax4.clabel(contour, inline=True, fontsize=8)
    ax4.set_xlabel('Number of Atoms')
    ax4.set_ylabel('Dice Coefficient')
    ax4.set_title('Contour Plot')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/molecular_dice_3d_surface.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, (X, Y, Z)

def create_detailed_analysis_plots(atom_counts, dice_coeffs, save_path=None):
    """
    Create additional analysis plots to understand the data better
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Scatter plot with density
    ax = axes[0, 0]
    scatter = ax.scatter(atom_counts, dice_coeffs, alpha=0.6, c=dice_coeffs, 
                        cmap='RdYlBu_r', s=20)
    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Dice Coefficient')
    ax.set_title('Scatter Plot: Atoms vs Dice Score')
    plt.colorbar(scatter, ax=ax)
    
    # 2. Hexbin plot (2D histogram)
    ax = axes[0, 1]
    hb = ax.hexbin(atom_counts, dice_coeffs, gridsize=20, cmap='Blues')
    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Dice Coefficient')
    ax.set_title('Hexbin Plot (Frequency Density)')
    plt.colorbar(hb, ax=ax)
    
    # 3. Box plot: Dice scores grouped by atom count ranges
    ax = axes[0, 2]
    atom_ranges = pd.cut(atom_counts, bins=8, precision=0)
    df_temp = pd.DataFrame({'atom_range': atom_ranges, 'dice': dice_coeffs})
    df_temp.boxplot(column='dice', by='atom_range', ax=ax)
    ax.set_xlabel('Atom Count Range')
    ax.set_ylabel('Dice Coefficient')
    ax.set_title('Dice Score Distribution by Atom Count')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 4. Marginal distributions
    ax = axes[1, 0]
    ax.hist(atom_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Atom Counts')
    
    ax = axes[1, 1]
    ax.hist(dice_coeffs, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_xlabel('Dice Coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Dice Scores')
    
    # 5. Correlation analysis
    ax = axes[1, 2]
    correlation = np.corrcoef(atom_counts, dice_coeffs)[0, 1]
    
    # Add trend line
    z = np.polyfit(atom_counts, dice_coeffs, 1)
    p = np.poly1d(z)
    ax.scatter(atom_counts, dice_coeffs, alpha=0.6, color='navy', s=20)
    ax.plot(atom_counts, p(atom_counts), "r--", alpha=0.8, linewidth=2)
    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Dice Coefficient')
    ax.set_title(f'Correlation Analysis (r={correlation:.3f})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/molecular_dice_analysis.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig

def print_data_statistics(atom_counts, dice_coeffs):
    """
    Print detailed statistics about the data
    """
    print("\n" + "="*50)
    print("MOLECULAR DICE COEFFICIENT ANALYSIS STATISTICS")
    print("="*50)
    
    print(f"\nDataset size: {len(atom_counts)} molecules")
    
    print(f"\nAtom counts:")
    print(f"  Range: {min(atom_counts)} - {max(atom_counts)}")
    print(f"  Mean: {np.mean(atom_counts):.2f}")
    print(f"  Median: {np.median(atom_counts):.2f}")
    print(f"  Std: {np.std(atom_counts):.2f}")
    
    print(f"\nDice coefficients:")
    print(f"  Range: {min(dice_coeffs):.4f} - {max(dice_coeffs):.4f}")
    print(f"  Mean: {np.mean(dice_coeffs):.4f}")
    print(f"  Median: {np.median(dice_coeffs):.4f}")
    print(f"  Std: {np.std(dice_coeffs):.4f}")
    
    # Correlation
    correlation = np.corrcoef(atom_counts, dice_coeffs)[0, 1]
    print(f"\nCorrelation between atom count and dice score: {correlation:.4f}")
    
    # Performance by molecule size
    small_molecules = np.array(dice_coeffs)[np.array(atom_counts) <= np.percentile(atom_counts, 33)]
    medium_molecules = np.array(dice_coeffs)[(np.array(atom_counts) > np.percentile(atom_counts, 33)) & 
                                           (np.array(atom_counts) <= np.percentile(atom_counts, 66))]
    large_molecules = np.array(dice_coeffs)[np.array(atom_counts) > np.percentile(atom_counts, 66)]
    
    print(f"\nPerformance by molecule size:")
    print(f"  Small molecules (≤{np.percentile(atom_counts, 33):.0f} atoms): {np.mean(small_molecules):.4f} ± {np.std(small_molecules):.4f}")
    print(f"  Medium molecules: {np.mean(medium_molecules):.4f} ± {np.std(medium_molecules):.4f}")
    print(f"  Large molecules (≥{np.percentile(atom_counts, 66):.0f} atoms): {np.mean(large_molecules):.4f} ± {np.std(large_molecules):.4f}")
    





# ——— Add this to your existing script after computing dice_coeffs and atom_counts ———

# Print statistics
print_data_statistics(atom_counts, dice_coeffs)

# Create the 3D surface plot
save_directory = '/scratch/phys/sin/sethih1/runs_ters/all_group_plane_fchk_split_images_ters/posnet_hyperopt_all_50_epochs/augmented/config_hypopt_all/'
fig, surface_data = create_3d_surface_plot(atom_counts, dice_coeffs, save_path=save_directory)

# Create additional analysis plots
fig_analysis = create_detailed_analysis_plots(atom_counts, dice_coeffs, save_path=save_directory)

# Optional: Interactive 3D plot using plotly (if available)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
    def create_interactive_3d_plot(atom_counts, dice_coeffs):
        """Create an interactive 3D plot using plotly"""
        
        # Create 2D histogram for surface
        atom_bins = np.linspace(min(atom_counts), max(atom_counts), 25)
        dice_bins = np.linspace(0, 1, 25)
        hist, atom_edges, dice_edges = np.histogram2d(atom_counts, dice_coeffs, bins=[atom_bins, dice_bins])
        
        # Create meshgrid
        atom_centers = (atom_edges[:-1] + atom_edges[1:]) / 2
        dice_centers = (dice_edges[:-1] + dice_edges[1:]) / 2
        
        # Create the surface plot
        fig = go.Figure(data=[go.Surface(
            x=atom_centers,
            y=dice_centers, 
            z=hist.T,
            colorscale='Viridis',
            colorbar=dict(title="Frequency")
        )])
        
        fig.update_layout(
            title='Interactive 3D Surface: Molecular Dice Coefficient Analysis',
            scene=dict(
                xaxis_title='Number of Atoms',
                yaxis_title='Dice Coefficient',
                zaxis_title='Frequency'
            ),
            width=900,
            height=700
        )
        
        fig.show()
        
        # Save as HTML
        fig.write_html(f"{save_directory}/interactive_3d_surface.html")
        print(f"Interactive plot saved as HTML in {save_directory}")
    
    create_interactive_3d_plot(atom_counts, dice_coeffs)
    
except ImportError:
    print("Plotly not available. Skipping interactive plot.")

print(f"\nAll plots saved to: {save_directory}")
print("Generated files:")
print("- molecular_dice_3d_surface.png (3D surface plot)")
print("- molecular_dice_analysis.png (detailed analysis)")
print("- interactive_3d_surface.html (if plotly available)")



# ——— Compute and print mean IoU and Dice Coefficient ———
mean_iou = sum(ious) / len(ious)
mean_dice = sum(dice_coeffs) / len(dice_coeffs)
print(f"Mean IoU over {len(ious)} samples: {mean_iou:.4f}")
print(f"Mean Dice Coefficient over {len(dice_coeffs)} samples: {mean_dice:.4f}")

# ——— Plot & save overall histograms ———
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# IoU histogram
axes[0].hist(ious, bins=10, range=(0, 1), alpha=0.8, color='mediumorchid', edgecolor='black')
axes[0].set_title(f'Distribution of IoU Scores\n(mean={mean_iou:.3f})')
axes[0].set_xlabel("IoU")
axes[0].set_ylabel("Frequency")
axes[0].grid(True)

# Dice histogram
axes[1].hist(dice_coeffs, bins=10, range=(0, 1), alpha=0.8, color='mediumorchid', edgecolor='black')
axes[1].set_title(f'Distribution of Dice Scores\n(mean={mean_dice:.3f})')
axes[1].set_xlabel("Dice Coefficient")
axes[1].set_ylabel("Frequency")
axes[1].grid(True)

# Final layout and save
plt.tight_layout()
plt.savefig(os.path.join(path, f"iou_dice_histograms_{suffix}.png"), dpi=300, bbox_inches='tight', transparent=True)
plt.close()




#################### Violin graph #################


# -----------------------------
# APPEND THIS BLOCK AT THE END OF YOUR SCRIPT
# Computes RMS-based planarity from .npz atom_pos and plots violin (dice vs planarity_rms)
# -----------------------------
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# --- helpers: safe filename conversion & readers ---
def _to_str(x):
    # turn torch tensor / bytes / numpy scalar into str
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
    except Exception:
        pass
    if isinstance(x, (np.bytes_, bytes)):
        try:
            return x.decode('utf-8')
        except Exception:
            return str(x)
    if isinstance(x, np.ndarray) and x.shape == ():
        return str(x.item())
    return str(x)

def read_npz(filepath):
    """
    Read coords and atomic_numbers from a .npz. Try common keys, fallback to first arrays.
    Returns coords (Nx3) and atomic_numbers (array or empty array).
    """
    with np.load(filepath, allow_pickle=True) as data:
        keys = list(data.files)
        coords = None
        atomic_numbers = np.array([], dtype=int)

        # try common names
        for cand in ['atom_pos', 'coords', 'positions', 'pos', 'xyz']:
            if cand in data:
                coords = data[cand]
                break
        if coords is None and len(keys) >= 1:
            coords = data[keys[0]]

        # atomic numbers
        for cand in ['atomic_numbers', 'Z', 'atomic_num', 'atomic_nums']:
            if cand in data:
                atomic_numbers = data[cand]
                break
        # fallback second array if it looks integer
        if atomic_numbers.size == 0 and len(keys) > 1:
            maybe = data[keys[1]]
            if np.issubdtype(maybe.dtype, np.integer):
                atomic_numbers = maybe

    # ensure coords shaped (N,3) if possible
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1 and coords.size % 3 == 0:
        coords = coords.reshape(-1, 3)
    return coords, atomic_numbers

def pca(X):
    """
    Compute covariance PCA. Returns eigvals (ascending), eigvecs (columns),
    and centered X (X - mean).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1 and X.size % 3 == 0:
        X = X.reshape(-1, 3)
    Xc = X - X.mean(axis=0)
    # handle degenerate case (fewer than 2 points)
    if Xc.shape[0] < 2:
        cov = np.eye(Xc.shape[1]) * 1e-8
    else:
        cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    return eigvals, eigvecs, Xc

# --- your planarity() implementation (uses PCA eigenvalues/eigenvectors and centered X) ---
def planarity(eigvals, eigvecs, X):
    # sort descending 
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # PCA-based planarity (percent)
    planarity_pca = 100.0 * (eigvals[0] + eigvals[1]) / (eigvals.sum() + 1e-12)

    # RMSD based planarity
    normal = eigvecs[:, 2]  # eigenvector for smallest eigenvalue (after descending sort)
    # project centered coordinates onto normal
    d = (X @ normal) / (np.linalg.norm(normal) + 1e-12)
    rmsd = np.sqrt(np.mean(d**2))
    L = np.sqrt(max(eigvals[0] + eigvals[1], 1e-12))
    planarity_rms = 100.0 * (1.0 - (rmsd / L))
    # clamp to sensible range
    planarity_rms = float(np.clip(planarity_rms, -100.0, 100.0))
    return planarity_pca, planarity_rms, float(rmsd)

# --- compute planarity_rms for each sample (aligned with filename_list, dice_coeffs, atom_counts) ---
planarity_rms_list = []
missing = 0

N_expected = min(len(filename_list), len(dice_coeffs), len(atom_counts))
for i in range(N_expected):
    fn_raw = filename_list[i]
    fn = _to_str(fn_raw)
    npz_file = os.path.join(data_path, f"{fn}.npz")
    if not os.path.exists(npz_file):
        warnings.warn(f"Missing .npz for {fn} (expected {npz_file}), appending NaN")
        planarity_rms_list.append(np.nan)
        missing += 1
        continue
    try:
        coords, atomic_numbers = read_npz(npz_file)
        if coords is None or coords.size == 0:
            warnings.warn(f"No coords in {npz_file}; appending NaN")
            planarity_rms_list.append(np.nan)
            missing += 1
            continue
        eigvals, eigvecs, Xc = pca(coords)
        plan_pca, plan_rms, rmsd = planarity(eigvals, eigvecs, Xc)
        # plan_rms is percent (0..100-ish). Normalize to 0..1 for binning/plotting.
        plan_rms_norm = np.clip(plan_rms / 100.0, 0.0, 1.0)
        planarity_rms_list.append(plan_rms_norm)
    except Exception as e:
        warnings.warn(f"Failed planarity for {npz_file}: {e}; appending NaN")
        planarity_rms_list.append(np.nan)
        missing += 1

if missing:
    print(f"Note: {missing} samples missing/failed planarity computation (they will be dropped).")

# --- build DataFrame and clean ---
df_plan = pd.DataFrame({
    'filename': [ _to_str(x) for x in filename_list[:N_expected] ],
    'dice': [ float(x) for x in dice_coeffs[:N_expected] ],
    'num_atoms': [ int(x) if (not isinstance(x, (list, tuple, np.ndarray))) else int(np.asarray(x).item()) for x in atom_counts[:N_expected] ],
    'planarity_rms': [ float(x) if (not np.isnan(x)) else np.nan for x in planarity_rms_list[:N_expected] ]
})

df_clean = df_plan.dropna(subset=['planarity_rms', 'dice']).copy()
print(f"Using {len(df_clean)} samples for planarity vs dice plotting (out of {N_expected}).")
if len(df_clean) == 0:
    print("No valid samples to plot after planarity computation. Exiting this analysis block.")
else:
    # create planarity bins (equal-width 10 bins 0..1)
    start, end, step = 0.90, 1.00, 0.01
    bin_edges = np.arange(start, end + step, step)  # Include last edge
    num_bins = len(bin_edges) - 1
    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(num_bins)]
    df_clean['planarity_bin'] = pd.cut(
    df_clean['planarity_rms'],
    bins=bin_edges,
    labels=bin_labels,
    include_lowest=True,
    ordered=True
)
    
    print(df_clean['planarity_bin'].value_counts())

    # --- violin plot (dice vs planarity_rms bins) ---
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_clean, x='planarity_bin', y='dice', inner='quartile', palette="Set2")
    sns.pointplot(data=df_clean, x='planarity_bin', y='dice', color='black', markers='o', linestyles='-', ci=None)
    plt.title("Dice Coefficient Distribution vs Planarity (RMS-based)")
    plt.xlabel("Planarity (RMS-based bins, normalized 0..1)")
    plt.ylabel("Dice Coefficient")
    plt.xticks(rotation=45)
    plt.xlim()
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    # choose save directory (create a subdir under a consistent root)
    save_root = os.path.join('/scratch/phys/sin/sethih1/runs_ters', 'analysis_outputs')
    os.makedirs(save_root, exist_ok=True)
    out_violin = os.path.join(save_root, f"dice_vs_planarity_rms_violin_{suffix}.png")
    plt.savefig(out_violin, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved violin plot: {out_violin}")

    # --- additional summary / correlation ---
    corr = np.corrcoef(df_clean['planarity_rms'].values, df_clean['dice'].values)[0,1]
    mean_by_bin = df_clean.groupby('planarity_bin')['dice'].agg(['mean','median','count'])
    print("\nPlanarity (RMS) vs Dice summary:")
    print(mean_by_bin)
    print(f"\nPearson r (planarity_rms, dice) = {corr:.4f}")

    # Save CSV
    out_csv = os.path.join(save_root, f"dice_planarity_rms_table_{suffix}.csv")
    df_clean.to_csv(out_csv, index=False)
    print(f"Saved planarity/dice table: {out_csv}")

# -----------------------------
# End of appended block
# -----------------------------





# --- Save Top 10% and Bottom 10% Dice score samples ---

save_root = os.path.join(path, "best_worst_dice_samples")
os.makedirs(save_root, exist_ok=True)

dice_array = np.array(dice_coeffs)
sorted_indices = np.argsort(dice_array)

# bottom 10% and top 10%
num_samples = len(dice_array)
k = max(1, num_samples // 10)  # 10% of dataset, at least 1 sample

bottom_indices = sorted_indices[:k]
top_indices = sorted_indices[-k:]

def save_samples(indices, save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)
    for idx in indices:
        avg_intensity = torch.mean(images_list[idx], dim=0).squeeze().cpu().numpy()
        gt_mask = masks_list[idx].squeeze()
        pred_mask = preds_list[idx].squeeze()
        mol_id = filename_list[idx]

        img_tensor = images_list[idx]
        img_min = img_tensor.view(img_tensor.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
        img_max = img_tensor.view(img_tensor.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
        img_tensor_norm = (img_tensor - img_min) / (img_max - img_min + 1e-8)
        spectral_sum = torch.sum(img_tensor_norm, dim=0).squeeze()
        spectral_sum_norm = (spectral_sum - spectral_sum.min()) / (spectral_sum.max() - spectral_sum.min() + 1e-8)
        spectral_sum_np = spectral_sum_norm.cpu().numpy()

        plt.imsave(f"{save_dir}/sample_{mol_id}_avg.png", avg_intensity, cmap="jet", origin='lower')
        plt.imsave(f"{save_dir}/sample_{mol_id}_spectralsum.png", spectral_sum_np, cmap="jet", origin='lower')
        plt.imsave(f"{save_dir}/sample_{mol_id}_ref.png", gt_mask, cmap="Reds", origin='lower')
        plt.imsave(f"{save_dir}/sample_{mol_id}_pred.png", pred_mask, cmap="Blues", origin='lower')

    print(f"Saved {len(indices)} {tag} samples in '{save_dir}'")

# save bottom 10%
save_samples(bottom_indices, os.path.join(save_root, "bottom_10_percent"), "bottom 10%")

# save top 10%
save_samples(top_indices, os.path.join(save_root, "top_10_percent"), "top 10%")
