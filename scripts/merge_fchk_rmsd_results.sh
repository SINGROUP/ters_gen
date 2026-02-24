#!/bin/bash
# Merge per-task CSVs created by the array job into a single CSV (keeps header once).
set -euo pipefail

TASK_DIR="$1"   # directory containing task_*.csv produced by array
FINAL_CSV="$2"

if [ ! -d "$TASK_DIR" ]; then
  echo "Task dir not found: $TASK_DIR"; exit 1
fi

CSV_FILES=("$TASK_DIR"/task_*.csv)
if [ ${#CSV_FILES[@]} -eq 0 ]; then
  echo "No task CSV files found in $TASK_DIR"; exit 1
fi

# Use header from the first file, then append all data lines (skip headers)
head -n1 "${CSV_FILES[0]}" > "$FINAL_CSV"
for f in "${CSV_FILES[@]}"; do
  tail -n +2 "$f" >> "$FINAL_CSV"
done

echo "Merged ${#CSV_FILES[@]} files into $FINAL_CSV"
# optional: remove per-task CSVs after merge
# rm -f "${CSV_FILES[@]}"
