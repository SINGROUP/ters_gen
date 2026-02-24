#!/bin/bash
set -euo pipefail

# SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -c 8

# ============================================
# Evaluate a single model on a single dataset
# Usage:
#   sbatch run_evaluate.sh <model_path> <data_path> [batch_size]
# Example:
#   sbatch run_evaluate.sh /path/to/model.pt /path/to/val_dir 32
# ============================================

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: sbatch run_evaluate.sh <model_path> <data_path> [batch_size]"
  exit 0
fi

if [[ $# -lt 2 ]]; then
  echo "Error: missing required arguments."
  echo "Usage: sbatch run_evaluate.sh <model_path> <data_path> [batch_size]"
  exit 1
fi

MODEL_PATH="$1"
DATA_PATH="$2"
BATCH_SIZE="${3:-32}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Error: model file not found: $MODEL_PATH"
  exit 1
fi

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Error: data directory not found: $DATA_PATH"
  exit 1
fi

# Load environment
source /scratch/phys/sin/sethih1/venv/masque_env/bin/activate

echo "============================================"
echo "Model: $MODEL_PATH"
echo "Data:  $DATA_PATH"
echo "Batch: $BATCH_SIZE"
echo "============================================"

# Run evaluation
python evaluate_model.py --model "$MODEL_PATH" --data "$DATA_PATH" --batch_size "$BATCH_SIZE"
