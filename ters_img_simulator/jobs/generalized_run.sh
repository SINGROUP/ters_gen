#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH -c 32

# Load Environments 
source /scratch/phys/sin/sethih1/venv/masque_env/bin/activate

# Arguments from wrapper: input_dir, output_dir, log_dir
INPUT_DIR=$1
OUTPUT_DIR=$2
LOG_DIR=$3

python /home/sethih1/masque_new/masque/masque/point_spectrum_generation.py "$INPUT_DIR" "$OUTPUT_DIR" "$LOG_DIR"
