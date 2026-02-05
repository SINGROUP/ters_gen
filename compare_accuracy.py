import os
from pathlib import Path
from typing import List, Any, Optional, Dict
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project-specific imports (keep your module paths)
from src.datasets import Ters_dataset_filtered_skip
from notebooks.utils.read_files import read_npz
from notebooks.utils.planarity import pca, planarity
from src.transforms import Normalize, MinimumToZero, NormalizeVectorized, MinimumToZeroVectorized
from src.models import AttentionUNet


suffixes = ['train', 'val', 'test']
rmss = [0.05, 0.1, 0.5, 1.0]

dir_viz = Path('/scratch/phys/sin/sethih1/Extended_TERS_data/run_planar_again/planar_comparison_viz')
num_channels = 100
sg_ch = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_transform = torch.nn.Sequential()  # placeholder; we'll use transforms.Compose below

# Use your Normalize, MinimumToZero
data_transform = torch.nn.Sequential()  # placeholder to avoid mismatched imports if torch.nn.Sequential used
from torchvision import transforms as _transforms
data_transform = _transforms.Compose([Normalize(), MinimumToZero()])
data_transform = _transforms.Compose([NormalizeVectorized(), MinimumToZeroVectorized()])


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _to_str(x: Any) -> str:
    """Turn torch tensor / bytes / numpy scalar into str (same helper as before)."""
    try:
        import torch as _torch
        if isinstance(x, _torch.Tensor):
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

def try_load_model(model_path: Path, device: str):
    
    model = torch.load(str(model_path), map_location=device)
    return model

# ----------------------
# Metrics
# ----------------------
def compute_sample_metrics(pred: torch.Tensor, mask: torch.Tensor) -> (float, float):
    """pred and mask are torch tensors (H,W) binary (0/1). Returns (iou, dice)."""
    pred_flat = pred.view(-1).bool()
    mask_flat = mask.view(-1).bool()
    inter = torch.logical_and(pred_flat, mask_flat).sum().float()
    union = torch.logical_or(pred_flat, mask_flat).sum().float()
    iou   = (inter / (union + 1e-6)).item()
    dice  = (2 * inter / (pred_flat.sum().float() + mask_flat.sum().float() + 1e-6)).item()
    return iou, dice

# ----------------------
# Plot / save helpers
# ----------------------
def save_combined_histogram(ious: List[float], dices: List[float], out_path: Path):
    safe_mkdir(out_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(dices, bins=30, range=(0,1), edgecolor='black')
    axes[0].set_title(f"Dice Distribution (mean={np.nanmean(dices):.3f})")
    axes[0].set_xlabel("Dice")
    axes[1].hist(ious, bins=30, range=(0,1), edgecolor='black')
    axes[1].set_title(f"IoU Distribution (mean={np.nanmean(ious):.3f})")
    axes[1].set_xlabel("IoU")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_dice_bin_examples(images_list, masks_list, preds_list, filename_list, dices, ious, out_dir: Path, max_per_bin=5):
    """Bins by dice (0.0..1.0, 10 bins) and saves up to max_per_bin images per bin.
       Uses average spectral intensity as input visualization (mean over channels)."""
    bin_edges = [i * 0.1 for i in range(11)]
    bins = {i: [] for i in range(10)}
    for idx, d in enumerate(dices):
        bin_idx = min(int(d * 10), 9)
        bins[bin_idx].append(idx)

    safe_mkdir(out_dir)
    for bin_idx, idxs in bins.items():
        if not idxs:
            continue
        idxs.sort(key=lambda i: dices[i])
        low, high = bin_edges[bin_idx], bin_edges[bin_idx+1]
        bin_dir = out_dir / f"{low:.1f}-{high:.1f}"
        safe_mkdir(bin_dir)
        for rank, i in enumerate(idxs[:max_per_bin], start=1):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            # images_list[i] is tensor (C, H, W)
            avg_chan = torch.mean(images_list[i], dim=0)
            axs[0].imshow(avg_chan.squeeze(), cmap='gray', origin='lower')
            axs[0].set_title("Average TERS intensity")
            axs[1].imshow(masks_list[i].squeeze(), cmap='gray', origin='lower')
            axs[1].set_title("Ground Truth")
            axs[2].imshow(preds_list[i].squeeze(), cmap='gray', origin='lower')
            axs[2].set_title(f"Pred (IoU={ious[i]:.3f}, Dice={dices[i]:.3f})")
            for ax in axs:
                ax.axis('off')
            fig.suptitle(f'Molecule id: {filename_list[i]}')
            fig.tight_layout()
            out_file = bin_dir / f"{low:.1f}-{high:.1f}_rank{rank}_iou{ious[i]:.3f}_dice{dices[i]:.3f}.png"
            fig.savefig(out_file, dpi=200)
            plt.close(fig)

def save_best_worst_samples(images_list, masks_list, preds_list, filename_list, dices, ious, out_dir: Path, frac=0.1):
    safe_mkdir(out_dir)
    dice_array = np.array(dices)
    n = len(dice_array)
    k = max(1, int(math.ceil(n * frac)))
    sorted_indices = np.argsort(dice_array)
    bottom_indices = sorted_indices[:k]
    top_indices = sorted_indices[-k:]

    def _save(indices, subdir):
        d = out_dir / subdir
        safe_mkdir(d)
        for idx in indices:
            img_tensor = images_list[idx]  # (C,H,W)
            # normalize each spectral channel then sum
            img_min = img_tensor.view(img_tensor.size(0), -1).min(dim=1)[0].view(-1,1,1)
            img_max = img_tensor.view(img_tensor.size(0), -1).max(dim=1)[0].view(-1,1,1)
            img_norm = (img_tensor - img_min) / (img_max - img_min + 1e-8)
            spectral_sum = torch.sum(img_norm, dim=0).squeeze().cpu().numpy()
            avg_intensity = torch.mean(images_list[idx], dim=0).squeeze().cpu().numpy()
            gt_mask = masks_list[idx].squeeze()
            pred_mask = preds_list[idx].squeeze()
            mol_id = _to_str(filename_list[idx])
            plt.imsave(str(d / f"sample_{mol_id}_avg.png"), avg_intensity, cmap='jet', origin='lower')
            plt.imsave(str(d / f"sample_{mol_id}_spectralsum.png"), spectral_sum, cmap='jet', origin='lower')
            plt.imsave(str(d / f"sample_{mol_id}_ref.png"), gt_mask, cmap='Reds', origin='lower')
            plt.imsave(str(d / f"sample_{mol_id}_pred.png"), pred_mask, cmap='Blues', origin='lower')
        print(f"Saved {len(indices)} {subdir} samples in '{d}'")

    _save(bottom_indices, "bottom_10_percent")
    _save(top_indices, "top_10_percent")

# ----------------------
# Experiment evaluation
# ----------------------

# ----------------------
# Experiment evaluation
# ----------------------
def evaluate_experiment(data_path: str, model_path: str, out_root: str, suffix: str, rms: float,
                        batch_size: int = 32, num_channels: int = 100, device: Optional[str] = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = Path(data_path)
    model_path = Path(model_path)
    out_root = Path(out_root) / f"{suffix}" / f"rms_{rms}"
    for p in ['hist', 'dice_bins', 'best_worst', 'tables', 'violin']:
        safe_mkdir(out_root / p)

    # load model
    model = try_load_model(model_path, device)
    model.to(device)
    model.eval()


    num_channels = model.conv.weight.shape[1]

    # dataset + loader (use your transforms and class)
    dataset = Ters_dataset_filtered_skip(
        filename=str(data_path),
        frequency_range=[0, 4000],
        num_channels=num_channels,
        std_deviation_multiplier=2,
        sg_ch=sg_ch,
        t_image=data_transform,
        t_freq=None,
        flag=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 1, pin_memory=False, persistent_workers=True)

    # accumulators
    ious = []
    dice_coeffs = []
    images_list = []
    masks_list = []
    preds_list = []
    filename_list = []
    atom_counts = []

    # inference + metrics
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {suffix} rms={rms}"):
            # Unpack as in your original batch: filename, atom_count, _, images, _, masks
            try:
                filename_b, atom_count_b, _, images_b, _, masks_b = batch
            except Exception:
                # fallback defensive unpack
                filename_b = batch[0]
                atom_count_b = batch[1]
                images_b = batch[3]
                masks_b = batch[5]

            images_b = images_b.to(device)
            masks_b  = masks_b.to(device)

            probs = model(images_b)              # (B,1,H,W) or (B,1,H,W)like

            probs = torch.sigmoid(probs)
            preds_b = (probs > 0.5).long().squeeze(1)   # (B,H,W)

            B = masks_b.size(0)
            for i in range(B):
                pred_i = preds_b[i]
                mask_i = masks_b[i]

                pred_flat = pred_i.view(-1)
                mask_flat = mask_i.view(-1)

                inter = torch.logical_and(pred_flat==1, mask_flat==1).sum().float()
                union = torch.logical_or(pred_flat==1, mask_flat==1).sum().float()
                #iou   = (inter / (union + 1e-6)).item() 
                iou   = (inter / (union)).item() if union.item() != 0 else 0
                #dice  = (2 * inter / (torch.sum(pred_flat) + torch.sum(mask_flat) + 1e-6)).item()

                dice  = (2 * inter / (torch.sum(pred_flat) + torch.sum(mask_flat))).item() if (torch.sum(pred_flat) + torch.sum(mask_flat)).item() != 0 else 0

                ious.append(iou)
                dice_coeffs.append(dice)

                images_list.append(images_b[i].cpu())
                masks_list.append(mask_i.cpu().squeeze(0).numpy())
                preds_list.append(pred_i.cpu().squeeze(0).numpy())
                filename_list.append(_to_str(filename_b[i]))

                # atom_counts
                try:
                    atom_counts.append(int(_to_str(atom_count_b[i])))
                except Exception:
                    try:
                        atom_counts.append(int(np.asarray(atom_count_b[i]).item()))
                    except Exception:
                        atom_counts.append(-1)

    mean_iou = float(np.nanmean(ious)) if len(ious) > 0 else float('nan')
    mean_dice = float(np.nanmean(dice_coeffs)) if len(dice_coeffs) > 0 else float('nan')

    # save combined hist
    hist_path = out_root / 'hist' / f"hist_combined_{suffix}_rms{rms}.png"
    save_combined_histogram(ious, dice_coeffs, hist_path)

    # save dice bin examples
    save_dice_bin_examples(images_list, masks_list, preds_list, filename_list, dice_coeffs, ious, out_root / 'dice_bins', max_per_bin=5)

    # save top/bottom
    save_best_worst_samples(images_list, masks_list, preds_list, filename_list, dice_coeffs, ious, out_root / 'best_worst', frac=0.1)

    # compute planarity per sample by reading the .npz files (using filename_list)
    planarity_rms_list = []
    rmsd_list = [] # <--- MODIFIED: Added list for rmsd
    missing = 0
    for fn in filename_list:
        npz_file = data_path / f"{fn}.npz"
        if not npz_file.exists():
            planarity_rms_list.append(np.nan)
            rmsd_list.append(np.nan) # <--- MODIFIED
            missing += 1
            continue
        try:
            coords, atomic_numbers = read_npz(str(npz_file))
            if coords is None or coords.size == 0:
                planarity_rms_list.append(np.nan)
                rmsd_list.append(np.nan) # <--- MODIFIED
                missing += 1
                continue
            eigvals, eigvecs, Xc = pca(coords)
            plan_pca, plan_rms, rmsd = planarity(eigvals, eigvecs, Xc)
            plan_rms_norm = float(np.clip(plan_rms / 100.0, 0.0, 1.0))
            planarity_rms_list.append(plan_rms_norm)
            rmsd_list.append(float(rmsd)) # <--- MODIFIED: Store the rmsd value
        except Exception as e:
            planarity_rms_list.append(np.nan)
            rmsd_list.append(np.nan) # <--- MODIFIED
            missing += 1

    if missing:
        print(f"Note: {missing} samples missing/failed planarity computation (will be NaN).")

    # Build DataFrame
    df = pd.DataFrame({
        'filename': filename_list,
        'dice': [float(x) for x in dice_coeffs],
        'iou': [float(x) for x in ious],
        'num_atoms': [int(x) if (not isinstance(x, (list, tuple, np.ndarray))) else int(np.asarray(x).item()) for x in atom_counts],
        'planarity_rms': [float(x) if (not (x is None or (isinstance(x, float) and np.isnan(x)))) else np.nan for x in planarity_rms_list],
        'rmsd': [float(x) if (not (x is None or (isinstance(x, float) and np.isnan(x)))) else np.nan for x in rmsd_list] # <--- MODIFIED: Add rmsd to DataFrame
    })

    # save table
    out_table = out_root / 'tables' / f"metrics_{suffix}_rms{rms}.csv"
    df.to_csv(out_table, index=False)
    print(f"Saved metrics table to {out_table}")

    # <--- ENTIRE PLOTTING BLOCK MODIFIED TO USE 'rmsd' --->
    # make violin vs rmsd (if enough data)
    df_clean = df.dropna(subset=['rmsd', 'dice']).copy()
    if len(df_clean) >= 5:
        # use quantile bins to avoid empty bins
        n_bins = 10
        qedges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = df_clean['rmsd'].quantile(qedges).values
        bin_edges = np.unique(bin_edges) # Handle cases where quantiles are identical
        
        # Fallback if quantiles fail (e.g., all values are the same)
        if len(bin_edges) - 1 < 2:
            min_r, max_r = df_clean['rmsd'].min(), df_clean['rmsd'].max()
            if max_r - min_r < 1e-6: # Avoid zero-width bins
                 max_r = min_r + 1.0
            bin_edges = np.linspace(min_r, max_r, 6) # Use 5 bins as a fallback
        
        # Create labels and ensure uniqueness
        labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
        labels_unique = len(labels) == len(set(labels))
        
        df_clean['rmsd_bin'] = pd.cut(df_clean['rmsd'], bins=bin_edges, labels=labels, include_lowest=True, ordered=labels_unique)

        sns.set(style="whitegrid")
        plt.figure(figsize=(10,6))
        sns.violinplot(data=df_clean, x='rmsd_bin', y='dice', inner='quartile')
        sns.pointplot(data=df_clean, x='rmsd_bin', y='dice', color='black', markers='o', linestyles='-', ci=None)
        plt.title(f"Dice vs RMSD (suffix={suffix}, rms={rms})")
        plt.xlabel("RMSD Bins")
        plt.xticks(rotation=45)
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        violin_path = out_root / 'violin' / f"dice_vs_rmsd_{suffix}_rms{rms}.png"
        plt.savefig(violin_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved violin: {violin_path}")
    else:
        print("Not enough rmsd-available samples for violin plot in this experiment.")
    # <--- END OF MODIFIED BLOCK --- >

    summary = {
        'n_samples': len(df),
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'paths': {k: str(out_root / k) for k in ['hist', 'dice_bins', 'best_worst', 'tables', 'violin']},
        'metrics_table': str(out_table)
    }

    return summary, df


# ----------------------
# Aggregation & comparison plots
# ----------------------
def save_comparison_plots(all_df: pd.DataFrame, out_root: Path):
    cmp_dir = out_root / 'comparison'
    safe_mkdir(cmp_dir)
    df = all_df.copy()
    df['rms_numeric'] = pd.to_numeric(df['rms'], errors='coerce')
    df['rms_str'] = df['rms'].astype(str)

    # Option A: per-suffix comparisons across RMS
    for suffix in df['suffix'].unique():
        df_s = df[df['suffix'] == suffix]
        if df_s.empty:
            continue
        order = df_s.groupby('rms_str')['rms_numeric'].first().sort_values().index.tolist()

        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(data=df_s, x='rms_str', y='dice', order=order, ax=ax)
        ax.set_title(f"Boxplot Dice across RMS (suffix={suffix})")
        ax.set_xlabel("RMS")
        ax.set_ylabel("Dice")
        fig.savefig(cmp_dir / f"boxplot_suffix_{suffix}.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10,6))
        sns.violinplot(data=df_s, x='rms_str', y='dice', order=order, inner='quartile', ax=ax)
        sns.pointplot(data=df_s, x='rms_str', y='dice', order=order, color='black', markers='o', linestyles='-', ci=None, ax=ax)
        ax.set_title(f"Violin Dice across RMS (suffix={suffix})")
        ax.set_xlabel("RMS")
        ax.set_ylabel("Dice")
        fig.savefig(cmp_dir / f"violin_suffix_{suffix}.png", dpi=300)
        plt.close(fig)

    # Option B: combined suffix:rms
    df['suffix_rms'] = df['suffix'].astype(str) + ":" + df['rms'].astype(str)
    order_comb = df.groupby(['suffix','rms_numeric']).size().reset_index().sort_values(['suffix','rms_numeric'])
    order_list = (order_comb['suffix'].astype(str) + ":" + order_comb['rms_numeric'].astype(str)).tolist()

    fig, ax = plt.subplots(figsize=(14,6))
    sns.boxplot(data=df, x='suffix_rms', y='dice', order=order_list, ax=ax)
    ax.set_title('Boxplot Dice by suffix:rms (combined)')
    ax.set_xlabel('suffix:rms')
    ax.set_ylabel('Dice')
    plt.xticks(rotation=45, ha='right')
    fig.savefig(cmp_dir / f"boxplot_combined_suffix_rms.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14,6))
    sns.violinplot(data=df, x='suffix_rms', y='dice', order=order_list, inner='quartile', ax=ax)
    sns.pointplot(data=df, x='suffix_rms', y='dice', order=order_list, color='black', markers='o', linestyles='-', ci=None, ax=ax)
    ax.set_title('Violin Dice by suffix:rms (combined)')
    ax.set_xlabel('suffix:rms')
    ax.set_ylabel('Dice')
    plt.xticks(rotation=45, ha='right')
    fig.savefig(cmp_dir / f"violin_combined_suffix_rms.png", dpi=300)
    plt.close(fig)

    print(f"Saved comparison plots to {cmp_dir}")

# ----------------------
# Runner
# ----------------------
def run_all_experiments(suffixes_list = suffixes, rmss_list = rmss, out_root: Optional[str] = None, device: Optional[str] = None):
    out_root = Path(out_root) if out_root else dir_viz
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}
    per_exp_tables = []

    for suffix in suffixes_list:
        for rms in rmss_list:
            data_path = f"/scratch/phys/sin/sethih1/Extended_TERS_data/planar_oct_2025/planar_again/planar_npz_{rms}/{suffix}/"
            model_path = f"/scratch/phys/sin/sethih1/Extended_TERS_data/run_planar_again/run_planar_npz_{rms}/models/best_model.pt"
            summary, df = evaluate_experiment(data_path=data_path, model_path=model_path, out_root=str(out_root), suffix=suffix, rms=rms, device=device, num_channels=num_channels)
            results[f"{suffix}_{rms}"] = summary
            # attach suffix/rms for aggregation
            df['suffix'] = suffix
            df['rms'] = rms
            per_exp_tables.append(df)

    # aggregate
    if per_exp_tables:
        all_df = pd.concat(per_exp_tables, ignore_index=True)
        safe_mkdir(out_root)
        out_csv = out_root / 'all_scores.csv'
        all_df.to_csv(out_csv, index=False)
        print(f"Saved aggregated scores to {out_csv}")
        save_comparison_plots(all_df, out_root)
        return results, all_df
    else:
        print("No experiments produced tables.")
        return results, pd.DataFrame()

# ----------------------
# CLI style main
# ----------------------
if __name__ == "__main__":
    results, all_df = run_all_experiments()
    print("Done. Results saved under:", dir_viz)
