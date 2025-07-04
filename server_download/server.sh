#!/bin/bash
#
# SLURM submission script for the planar‐molecule downloader
#
#SBATCH --job-name=planar_filter
#SBATCH --output=planar_filter_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=72:00:00

## load any required modules or activate your environment
# module load python/3.11.6    # or whichever module is appropriate
# source /path/to/venv/bin/activate

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo "Using $SLURM_CPUS_PER_TASK CPUs"

## move to submission directory
cd $SLURM_SUBMIT_DIR

source /scratch/phys/sin/sethih1/venv/masque_env/bin/activate
## run your script
python /home/sethih1/masque_new/ters_gen/server_download/copy_server.py

echo "Job finished at $(date)"
