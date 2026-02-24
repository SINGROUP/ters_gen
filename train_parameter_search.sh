#!/bin/bash
set -euo pipefail

#SBATCH --time=120:00:00
#SBATCH --mem=1400G
#SBATCH --gpus=2
#SBATCH -c 64
# SBATCH --mem-per-cpu=10G

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch train_parameter_search.sh <config_path>"
  exit 1
fi

arg1="$1"


echo $CUDA_VISIBLE_DEVICES


# Load Environments
source /scratch/phys/sin/sethih1/venv/masque_env/bin/activate

# Start GPU monitoring: Log GPU usage every 2 seconds into gpu_usage.log
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,temperature.gpu \
         --format=csv -l 2 > /scratch/work/sethih1/slurm_logs_0.1/gpu_usage.log &
GPU_MONITOR_PID=$!

# Optionally, start system resource monitoring: Log CPU, memory, etc. every 1 second
vmstat -n 1 > /scratch/work/sethih1/slurm_logs_0.1/resource_usage.log &
RESOURCE_MONITOR_PID=$!

# Run your main Python job
# python parameter_search.py --config "$arg1"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Error: WANDB_API_KEY is not set."
  echo "Set your own key before submitting:"
  echo "  export WANDB_API_KEY=<your_wandb_key>"
  echo "  sbatch train_parameter_search.sh <config_path>"
  exit 1
fi

python ./hyperopt.py --config "$arg1" --use_wandb 
# python check_train.py --config $arg1

# After your job finishes, stop the monitoring processes
kill $GPU_MONITOR_PID
kill $RESOURCE_MONITOR_PID
