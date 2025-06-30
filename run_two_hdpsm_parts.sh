#!/bin/bash

# Define start and end years
start_year=2004
end_year=2009

mkdir -p logs/${start_year}-${end_year}

for part in {2..9}; do
    LOGFILE="logs/${start_year}-${end_year}/two_hdpsm_${start_year}-${end_year}_part_${part}.log"
    CMD="python3 src/two_hdpsm.py --start_year $start_year --end_year $end_year --part $part --total_parts 10"
    
    echo "Launching part $part with '$CMD'..."
    echo "  Logs will be saved at $LOGFILE"
    nohup $CMD > "$LOGFILE" 2>&1 &
done
