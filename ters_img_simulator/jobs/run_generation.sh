#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --mem=64G
#SBATCH -c 16

# Load Environments 
source /scratch/phys/sin/sethih1/venv/masque_env/bin/activate
# python point_spectrum_generation.py /scratch/phys/sin/sethih1/data_files/first_group_plane_95_99_fchk/ /scratch/phys/sin/sethih1/data_files/first_group_plane_95_99_npz/ /scratch/phys/sin/sethih1/data_files/first_group_log/

python point_spectrum_generation.py /home/sethih1/masque_new/masque/check /home/sethih1/masque_new/masque/check /home/sethih1/masque_new/masque/check/log
