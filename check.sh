#!/bin/bash
# SBATCH --time=61:00:00
#SBATCH --mem=256G
#SBATCH --gpus=1
#SBATCH -c 12
# SBATCH --partition=gpu-h100-80g-short
# SBATCH --partition=gpu-a100-80g
# SBATCH --partition=gpu-debug
#SBATCH -o /home/sethih1/masque_new/composnet/ters_gen/log_file/slurm_%i.out



# Load Environments
source /scratch/phys/sin/sethih1/venv/masque_env/bin/activate


python /home/sethih1/masque_new/composnet/ters_gen/count_pixels.py
