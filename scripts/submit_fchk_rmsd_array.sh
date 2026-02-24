#!/bin/bash
# Helper to submit the SLURM array for `fchk_rmsd_to_csv.py` and then merge results.
# Usage: ./scripts/submit_fchk_rmsd_array.sh /scratch/.../FCHK /scratch/.../FCHK/rmsd_summary.csv
set -euo pipefail

FCHK_DIR="$1"
FINAL_CSV="${2:-$FCHK_DIR/rmsd_planarity_summary.csv}"
PY_SCRIPT="${3:-scripts/fchk_rmsd_to_csv.py}"
TMP_LIST="$(mktemp -u ${FCHK_DIR}/fchk_list.XXXXXX)"

# build file list
find "$FCHK_DIR" -type f -name '*.fchk' | sort > "$TMP_LIST"
N=$(wc -l < "$TMP_LIST" | tr -d ' ')
if [ "$N" -eq 0 ]; then
  echo "No .fchk files found in $FCHK_DIR"; exit 1
fi

OUTDIR="$(dirname "$FINAL_CSV")/fchk_rmsd_tasks"
mkdir -p "$OUTDIR"

# submit array (limit concurrency with %200 — change if you want more or less parallelism)
ARRAY_DEF="1-${N}%200"
JOBID=$(sbatch --parsable --array=${ARRAY_DEF} scripts/fchk_rmsd_array.slurm "$TMP_LIST" "$OUTDIR" "$PY_SCRIPT")
if [ -z "$JOBID" ]; then
  echo "sbatch failed"; exit 1
fi

echo "Submitted SLURM array job ${JOBID} (1..${N}) — per-task CSVs -> ${OUTDIR}/task_*.csv"

# submit merge job dependent on array completion
MERGE_JOBID=$(sbatch --parsable --dependency=afterok:${JOBID} scripts/merge_fchk_rmsd_results.sh "$OUTDIR" "$FINAL_CSV")
if [ -z "$MERGE_JOBID" ]; then
  echo "Failed to submit merge job"; exit 1
fi

echo "Merge job ${MERGE_JOBID} will create final CSV: ${FINAL_CSV}"

echo "Cleaning up temporary list $TMP_LIST"
rm -f "$TMP_LIST"
