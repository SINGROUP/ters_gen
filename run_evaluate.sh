#!/bin/bash
# SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --partition=gpu-debug
#SBATCH -o /home/sethih1/masque_new/ters_gen/log_file/slurm_%j.out

# ============================================
# Evaluate a single model on a single dataset
# Usage: sbatch run_evaluate.sh
# ============================================

# --- EDIT THESE PATHS ---
MODEL_PATH="/scratch/phys/sin/sethih1/Extended_TERS_data/run_planar_again/run_planar_npz_0.05/models/seg_trial8_bs16_lr7e-04_lossdice_loss.pt"
DATA_PATH="/scratch/phys/sin/sethih1/Extended_TERS_data/planar_oct_2025/planar_again/planar_npz_0.05/val/"
# ------------------------

# Load environment
source /scratch/phys/sin/sethih1/venv/masque_env/bin/activate

echo "============================================"
echo "Model: $MODEL_PATH"
echo "Data:  $DATA_PATH"
echo "============================================"

# Run evaluation
python evaluate_model.py --model "$MODEL_PATH" --data "$DATA_PATH" --batch_size 32
