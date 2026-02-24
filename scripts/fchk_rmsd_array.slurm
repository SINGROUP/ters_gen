#!/bin/bash
#SBATCH --job-name=fchk-rmsd
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=02:00:00
# SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# NOTE: array range should be supplied at sbatch time (example: --array=1-100)

# Usage inside sbatch: sbatch --array=1-N scripts/fchk_rmsd_array.slurm fchk_list.txt /path/to/outdir [path/to/python_script]
FCHK_LIST="$1"          # path to file that has one .fchk path per line
OUTDIR="$2"             # output directory for per-task CSVs
PY_SCRIPT="${3:-scripts/fchk_rmsd_to_csv.py}"

if [ -z "$FCHK_LIST" ] || [ -z "$OUTDIR" ]; then
  echo "Usage: sbatch --array=1-N $0 fchk_list.txt /out/dir [python_script]"
  exit 2
fi

FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$FCHK_LIST")
if [ -z "$FILE" ]; then
  echo "No file for index ${SLURM_ARRAY_TASK_ID}"; exit 1
fi

mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/task_${SLURM_ARRAY_TASK_ID}.csv"

echo "[${SLURM_ARRAY_TASK_ID}] processing: $FILE -> $OUTFILE"
python3 "$PY_SCRIPT" --file "$FILE" -o "$OUTFILE"
exit $?
