#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH -c 12
#SBATCH -o /home/sethih1/masque_new/masque_gan/log_file/slurm_%j.out

arg1=$1

# Load Environments
source /scratch/phys/sin/sethih1/venv/masque_env/bin/activate
python parameter_search.py  /scratch/phys/sin/sethih1/data_files/balanced_group --save_path /scratch/phys/sin/sethih1/data_files/gans/ --epoch 10 
