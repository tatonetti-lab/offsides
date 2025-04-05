#!/bin/bash

# Parameters
START_YEAR=2004
END_YEAR=2004

# Paths
INPUT_DIR="results/${START_YEAR}-${END_YEAR}/twopsm"
OUTPUT_FILE="results/${START_YEAR}-${END_YEAR}/two_hdpsm_nrep10_mratio5_maxsamp10000.csv.gz"

# Find all csv.gz files in the input directory
FILES=("$INPUT_DIR"/*.csv.gz)

# Check if there are files
if [ ${#FILES[@]} -eq 0 ]; then
  echo "No .csv.gz files found in $INPUT_DIR"
  exit 1
fi

# Concatenate
{
  # Print header from the first file
  gunzip -c "${FILES[0]}" | head -n 1

  # Print all data rows (skip header) from all files
  for f in "${FILES[@]}"; do
    gunzip -c "$f" | tail -n +2
  done
} | gzip > "$OUTPUT_FILE"

echo "Concatenation complete. Output: $OUTPUT_FILE"
