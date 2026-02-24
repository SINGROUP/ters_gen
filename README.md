# TERS-to-Structure Pipeline (`ters_gen`)

Tools for:
- Simulating TERS images from Gaussian `.fchk` files
- Training an image-to-image segmentation model (Attention U-Net) to recover molecular structure masks
- Evaluating and comparing model performance

## Repository Layout

```text
.
├── ters_img_simulator/          # TERS simulation pipeline (.fchk -> .npz)
├── src/                         # Models, datasets, trainer, metrics, transforms
├── configs/                     # Hyperparameter search YAML configs
├── notebooks/                   # Analysis and visualization notebooks/scripts
├── hyperopt.py                  # Main training + Optuna hyperparameter search
├── evaluate_model.py            # Evaluate one trained model on one dataset folder
├── check_accuracy_dice.py       # Detailed dice/IoU analysis and plots
├── compare_accuracy.py          # Cross-condition comparison plots/tables
├── model_checkpoints/           # Pretrained checkpoints
├── train_parameter_search.sh    # SLURM wrapper for hyperopt.py
└── run_evaluate.sh              # SLURM wrapper for evaluate_model.py
```

## Environment Setup

Dependencies are pinned in `requirements.txt`.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Format

The model training/evaluation dataset expects directories containing `.npz` files.

Each file should contain at least:
- `atom_pos`
- `atomic_numbers`
- `frequencies`
- `spectrums` (TERS intensity tensor on an `(x, y, mode)` grid)

The training dataset class is implemented in:
- `src/datasets/ters_image_to_image_sh.py`

Typical split layout:

```text
<data_root>/
├── train/*.npz
├── val/*.npz
└── test/*.npz
```

## 1) Generate Simulated TERS Data

From repo root:

```bash
python ters_img_simulator/scripts/point_spectrum_generation.py \
  <fchk_dir> \
  <output_npz_dir> \
  <log_file_or_log_dir>
```

Optional args:

```bash
--molecule_rotation PHI THETA PSI
--plot_spectrum W1 W2 W3 ...
```

Notes:
- If `--molecule_rotation` is omitted, PCA-based auto-rotation is used.
- Output is one `.npz` per input `.fchk`.
- Log parsing helper:

```bash
python ters_img_simulator/scripts/log_reading.py 0 <log_file>   # unfinished
python ters_img_simulator/scripts/log_reading.py 1 <log_file>   # errors
```

See also: [`ters_img_simulator/README.md`](/scratch/work/sethih1/ters_gen/ters_img_simulator/README.md)

## 2) Hyperparameter Search + Training

Main entry point:

```bash
python hyperopt.py --config configs/config_hypopt_all_val.yaml
```

With Weights & Biases logging:

```bash
python hyperopt.py --config configs/config_hypopt_all_val.yaml --use_wandb
```

What it does:
- Loads YAML config via `src/configs/base.py`
- Runs Optuna trials with `AttentionUNet`
- Trains per trial with `src/trainer/trainer_image_to_image.py`
- Saves trial models to `save_path`
- Writes trial table and Optuna HTML plots to `log_path`
- Copies best model to `<save_path>/best_model.pt`

## 3) Evaluate a Single Model

```bash
python evaluate_model.py \
  --model <path/to/model.pt> \
  --data <path/to/npz_dir> \
  --batch_size 32
```

This computes the same global metrics pipeline used during training (Dice/IoU/precision/recall/F1 from `src/metrics/metrics.py`).

## Pretrained Checkpoints

Pretrained models are in `model_checkpoints/`.

Naming convention:
- `best_model_0.05.pt` means the model was trained on dataset variant with `0.05` RMSD.
- `best_model_0.1.pt` means the model was trained on dataset variant with `0.1` RMSD.
- `best_model_0.5.pt` means the model was trained on dataset variant with `0.5` RMSD.

Example evaluation:

```bash
python evaluate_model.py \
  --model model_checkpoints/by_rmsd/best_model_0.05.pt \
  --data <path/to/npz_dir> \
  --batch_size 32
```

## 4) Analysis Scripts

- `check_accuracy_dice.py`: per-sample IoU/Dice analysis and visualization outputs.
- `compare_accuracy.py`: compares performance across different dataset settings (e.g., RMS/noise conditions).

These scripts currently include hard-coded paths; update paths near the top of each file before running.

## Config File Reference

Example config: `configs/config_hypopt_all_val.yaml`

Main sections:
- `model`: architecture search space (`in_channels`, `filters_options`, `att_channels_options`, etc.)
- `training`: epochs, trial count, batch sizes, LR range, loss options
- `data`: train/val directories, augmentation, label generation options
- `save_path`: where model checkpoints are written
- `log_path`: where TensorBoard and Optuna logs are written

## SLURM Wrappers

Provided job scripts:
- `train_parameter_search.sh`
- `run_evaluate.sh`
- `call_accuracy.sh`
- `compare_accuracy.sh`

They are cluster-specific (partition names, env paths, log directories). Update these for your cluster before running.

## Common Issues

- `ModuleNotFoundError`: run commands from repository root so local imports resolve.
- Empty dataset: verify `<dir>/*.npz` exists and keys are present.
- Slow runs: simulation uses dense 256x256 grids by default; reduce grid constants in `ters_img_simulator/scripts/point_spectrum_generation.py` for faster experiments.
- W&B auth errors: use `wandb login` or disable `--use_wandb`.
