#!/bin/bash

# Parent directory containing subdirectories
PARENT_DIR="/scratch/phys/sin/sethih1/data_files/all_group_plane_fchk_split"
PARENT_DIR="/scratch/work/sethih1/planar_oct_2025/planar_FCHK_1.0"
# PARENT_DIR="/scratch/work/sethih1/planar_oct_2025/planar_FCHK_0.1_split"

# Output folder (everything collected here)
OUTPUT_DIR="/scratch/phys/sin/sethih1/data_files/combined_npz_images_32x32"
OUTPUT_DIR="/scratch/work/sethih1/planar_oct_2025/planar_npz_1.0"
OUTPUT_DIR="/scratch/phys/sin/sethih1/Extended_TERS_data/planar_oct_2025/planar_npz_1.0_again"
#OUTPUT_DIR="/scratch/phys/sin/sethih1/Extended_TERS_data/planar_oct_2025/planar_npz_0.1"

# Path to your sbatch template
SBATCH_SCRIPT="/home/sethih1/masque_new/masque/masque/generalized_run.sh"

# Make sure output exists
mkdir -p "$OUTPUT_DIR"

# Loop over each subdirectory
for SUBDIR in "$PARENT_DIR"/*; do
    if [ -d "$SUBDIR" ]; then
        BASENAME=$(basename "$SUBDIR")

        # Submit job with paths
        sbatch "$SBATCH_SCRIPT" "$SUBDIR" "$OUTPUT_DIR" "$OUTPUT_DIR/log_${BASENAME}"
    fi
done
