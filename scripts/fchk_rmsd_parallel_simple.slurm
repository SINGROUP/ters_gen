#!/bin/bash
#SBATCH --job-name=fchk-rmsd-par
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G

# Simple SLURM wrapper to run scripts/fchk_rmsd_parallel_simple.py
# Usage:
#   sbatch scripts/fchk_rmsd_parallel_simple.slurm /path/to/FCHK /path/to/out.csv [workers] [bohr]
# Example:
#   sbatch scripts/fchk_rmsd_parallel_simple.slurm /scratch/phys/sin/sethih1/check_temp_count/FCHK /scratch/phys/sin/sethih1/check_temp_count/FCHK/rmsd_parallel.csv 12 bohr

set -euo pipefail

FCHK_DIR="$1"
OUT_CSV="${2:-$FCHK_DIR/rmsd_planarity_parallel.csv}"
WORKERS="${3:-12}"
BOHR_FLAG="${4:-}"

if [ -z "$FCHK_DIR" ]; then
  echo "Usage: $0 /path/to/FCHK /path/to/out.csv [workers] [bohr]"; exit 2
fi

if [ "$BOHR_FLAG" = "bohr" ] || [ "$BOHR_FLAG" = "--bohr" ]; then
  BOHR_ARG="--bohr"
else
  BOHR_ARG=""
fi

echo "Running on: $FCHK_DIR"
echo "Output CSV: $OUT_CSV"
echo "Workers: $WORKERS  (SBATCH cpus-per-task=${SLURM_CPUS_PER_TASK:-12})"

echo "Command: python3 scripts/fchk_rmsd_parallel_simple.py $FCHK_DIR -o $OUT_CSV --workers $WORKERS $BOHR_ARG"
python3 scripts/fchk_rmsd_parallel_simple.py "$FCHK_DIR" -o "$OUT_CSV" --workers "$WORKERS" $BOHR_ARG

exit $?
