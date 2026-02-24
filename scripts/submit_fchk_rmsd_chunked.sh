#!/bin/bash
# Submit chunked SLURM array where each array task uses multiple workers (e.g. 32 cores)
# Usage: ./scripts/submit_fchk_rmsd_chunked.sh /path/to/FCHK /path/to/final.csv [num_chunks] [workers_per_task] [concurrency]
set -euo pipefail

FCHK_DIR="$1"
FINAL_CSV="${2:-$FCHK_DIR/rmsd_planarity_summary.csv}"
REQUESTED_CHUNKS="${3:-}"   # optional
WORKERS_PER_TASK="${4:-32}"
CONCURRENCY_LIMIT="${5:-20}"

if [ -z "$FCHK_DIR" ]; then
  echo "Usage: $0 /path/to/FCHK /path/to/final.csv [num_chunks] [workers_per_task] [concurrency]"; exit 2
fi

FILE_LIST="$(mktemp -u ${FCHK_DIR}/fchk_list.XXXXXX)"
find "$FCHK_DIR" -type f -name '*.fchk' | sort > "$FILE_LIST"
N=$(wc -l < "$FILE_LIST" | tr -d ' ')
if [ "$N" -eq 0 ]; then
  echo "No .fchk files found in $FCHK_DIR"; rm -f "$FILE_LIST"; exit 1
fi

# default heuristic: ~128 files per chunk (so each 32-worker task has ~4 files/worker)
if [ -n "$REQUESTED_CHUNKS" ]; then
  NUM_CHUNKS="$REQUESTED_CHUNKS"
else
  CHUNK_SIZE=128
  NUM_CHUNKS=$(( (N + CHUNK_SIZE - 1) / CHUNK_SIZE ))
fi

OUTDIR="$(dirname "$FINAL_CSV")/fchk_rmsd_tasks_chunked"
mkdir -p "$OUTDIR"

ARRAY_DEF="1-${NUM_CHUNKS}%${CONCURRENCY_LIMIT}"
JOBID=$(sbatch --parsable --array=${ARRAY_DEF} scripts/fchk_rmsd_chunked.slurm "$FILE_LIST" "$OUTDIR" "$NUM_CHUNKS" "$WORKERS_PER_TASK")
if [ -z "$JOBID" ]; then
  echo "sbatch failed"; rm -f "$FILE_LIST"; exit 1
fi

echo "Submitted chunked SLURM array job ${JOBID} (1..${NUM_CHUNKS}) — per-task CSVs -> ${OUTDIR}/task_*.csv"

# merge job dependent on the array
MERGE_JOBID=$(sbatch --parsable --dependency=afterok:${JOBID} scripts/merge_fchk_rmsd_results.sh "$OUTDIR" "$FINAL_CSV")
if [ -z "$MERGE_JOBID" ]; then
  echo "Failed to submit merge job"; rm -f "$FILE_LIST"; exit 1
fi

echo "Merge job ${MERGE_JOBID} will create final CSV: ${FINAL_CSV}"
rm -f "$FILE_LIST"
