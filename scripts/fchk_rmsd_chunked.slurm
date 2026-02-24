#!/bin/bash
#SBATCH --job-name=fchk-rmsd-chunk
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=04:00:00
# SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G

# Usage (example): sbatch --array=1-10 scripts/fchk_rmsd_chunked.slurm fchk_list.txt /out/dir 10 32
FILE_LIST="$1"      # text file with one .fchk path per line
OUTDIR="$2"         # directory where per-task CSVs are written
NUM_CHUNKS="$3"     # total number of chunks (array size)
WORKERS="$4"        # workers per task (defaults to SBATCH --cpus-per-task)

if [ -z "$FILE_LIST" ] || [ -z "$OUTDIR" ] || [ -z "$NUM_CHUNKS" ]; then
  echo "Usage: sbatch --array=1-N $0 fchk_list.txt /out/dir <num_chunks> [workers]"
  exit 2
fi

WORKERS=${WORKERS:-32}

TASK_ID=${SLURM_ARRAY_TASK_ID}
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/task_${TASK_ID}.csv"

echo "[${TASK_ID}] chunk ${TASK_ID}/${NUM_CHUNKS} -> workers=${WORKERS} -> $OUTFILE"
python3 scripts/fchk_rmsd_to_csv.py --file-list "$FILE_LIST" --num-chunks "$NUM_CHUNKS" --chunk-index "$TASK_ID" --workers "$WORKERS" -o "$OUTFILE"
