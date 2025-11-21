#!/bin/bash
#SBATCH --job-name=myjob          # Name of the job
#SBATCH --output=output.log       # Output file
#SBATCH --time=04:00:00           # Max runtime (hh:mm:ss)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=8         # Number of CPUs
#SBATCH --mem=64G                  # Memory

source /scratch/phys/sin/sethih1/venv/masque_env/bin/activate

# Run your Python script
python planar_molecule_finder.py