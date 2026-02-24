#!/bin/bash
# submit_all.sh

# Path to your virtual‐env activation
VENV_ACTIVATE="/scratch/phys/sin/sethih1/venv/masque_env/bin/activate"

for i in {1..10}; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ptspec_part${i}
#SBATCH --time=100:00:00
#SBATCH --mem=64G
#SBATCH -c 12

# Load environment
source ${VENV_ACTIVATE}

# Run point_spectrum_generation for part $i
python point_spectrum_generation.py \
  /scratch/phys/sin/sethih1/data_files/all_group_plane_fchk_split/part${i}/ \
  /scratch/phys/sin/sethih1/data_files/all_group_plane_fchk_split_images_ters/part${i}/ \
  /scratch/phys/sin/sethih1/data_files/all_group_plane_fchk_split_log/
EOF

  echo "Submitted job for part${i}"
done
