# TERS-to-Structure Pipeline (`ters_gen`)

This repository provides a reproducible pipeline for:
- Simulating TERS images from Gaussian `.fchk` files
- Training an Attention U-Net model for molecular-structure mask prediction
- Evaluating trained checkpoints on held-out `.npz` datasets

## Repository Contents

```text
.
├── configs/                     # YAML configs for hyperparameter search/training
├── model_checkpoints/           # Provided pretrained checkpoints
├── notebooks/                   # Inference notebook and notebook utilities
├── src/                         # Models, datasets, trainer, metrics, transforms
├── ters_img_simulator/          # TERS simulation pipeline (.fchk -> .npz)
├── evaluate_model.py            # Single-model evaluation entrypoint
├── hyperopt.py                  # Training + Optuna hyperparameter search entrypoint
├── requirements.txt             # Python dependency pins
├── run_evaluate.sh              # SLURM wrapper for evaluate_model.py
└── train_parameter_search.sh    # SLURM wrapper for hyperopt.py
```

## Environment Setup

Install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: `requirements.txt` currently contains two `PyYAML` pins. Depending on the resolver, this may cause an install conflict and should be harmonized in your local environment.

## Quick Inference (Notebook)

For most users, the fastest way to inspect model behavior is:
- `notebooks/inference.ipynb`

Run:

```bash
jupyter lab notebooks/inference.ipynb
```

The notebook is the primary interactive entrypoint for single-molecule inference and visualization (input spectrum, ground truth, prediction, and IoU/Dice summary).

## Data Format

Training and evaluation expect directories of `.npz` files.

Required keys per `.npz`:
- `atom_pos`
- `atomic_numbers`
- `frequencies`
- `spectrums`


Typical split layout:

```text
<data_root>/
├── train/*.npz
├── val/*.npz
└── test/*.npz
```

Dataset implementation:
- `src/datasets/ters_image_to_image_sh.py`

## Training

Canonical training command:

```bash
python hyperopt.py --config configs/config_hypopt_all_val.yaml
```

With Weights & Biases logging:

```bash
python hyperopt.py --config configs/config_hypopt_all_val.yaml --use_wandb
```

SLURM wrapper usage:

```bash
export WANDB_API_KEY=<your_wandb_key>
sbatch train_parameter_search.sh configs/config_hypopt_all_val.yaml
```

Training outputs are controlled by YAML keys such as `save_path` and `log_path`.

## Evaluation

Canonical evaluation command:

```bash
python evaluate_model.py \
  --model <path/to/model.pt> \
  --data <path/to/npz_dir> \
  --batch_size 32
```

SLURM wrapper usage:

```bash
sbatch run_evaluate.sh <model_path> <data_path> [batch_size]
```

Example:

```bash
sbatch run_evaluate.sh model_checkpoints/best_model_0.05.pt /path/to/val 32
```

`evaluate_model.py` computes global Accuracy, Precision, Recall, F1, IoU, and Dice via `src/metrics/metrics.py`.

## Checkpoints

Provided checkpoints are in `model_checkpoints/`:
- `best_model_0.05.pt`
- `best_model_0.1.pt`
- `best_model_0.5.pt`

Suffix convention:
- `0.05` means trained on the dataset variant with `0.05` RMSD
- `0.1` means trained on the dataset variant with `0.1` RMSD
- `0.5` means trained on the dataset variant with `0.5` RMSD

Example:

```bash
python evaluate_model.py \
  --model model_checkpoints/best_model_0.05.pt \
  --data <path/to/npz_dir> \
  --batch_size 32
```

## Simulation Pipeline

The simulator is documented in [`ters_img_simulator/README.md`](ters_img_simulator/README.md).  
Use the same Python environment defined in this top-level README (`pip install -r requirements.txt`), then follow simulator-specific usage and options in the simulator README.

## Notebooks

- `notebooks/inference.ipynb`: single-molecule inference and visualization workflow
- `notebooks/utils/`: notebook utility modules (data reading, planarity, visualization)

