#!/bin/bash
# Batch-run the C++ optimizer on all cases in /mnt/c/A_TangJian/opt/.
# Each case gets <case>/our_out/ with: input copies, optim_log.txt,
# optimized_quaternions.csv. Rendering is a separate step (batch_render.sh).
set -u
OPT_DIR=/mnt/c/A_TangJian/opt
BIN=/home/tangjian5/quartet/build/quartet
SUMMARY=/tmp/batch_opt_summary.txt

: > "$SUMMARY"
for C in "$OPT_DIR"/*/; do
    NAME=$(basename "$C")
    mkdir -p "$C/our_out"
    cp "$C/conf.yaml" \
       "$C/DumpHIS/DbgFile_Frame_OriginalCamOrient.csv" \
       "$C/DumpHIS/DbgFile_WarpGrids_RSCFRMMid.csv" \
       "$C/our_out/" || { echo "$NAME: input copy FAILED" | tee -a "$SUMMARY"; continue; }

    START=$(date +%s)
    if "$BIN" \
        --quat   "$C/our_out/DbgFile_Frame_OriginalCamOrient.csv" \
        --yaml   "$C/our_out/conf.yaml" \
        --rcs    "$C/our_out/DbgFile_WarpGrids_RSCFRMMid.csv" \
        --output "$C/our_out/optimized_quaternions.csv" \
        > "$C/our_out/optim_log.txt" 2>&1; then
        N=$(wc -l < "$C/our_out/optimized_quaternions.csv")
        END=$(date +%s)
        echo "$NAME: OK frames=$N time=$((END - START))s" | tee -a "$SUMMARY"
    else
        echo "$NAME: FAILED (see $C/our_out/optim_log.txt)" | tee -a "$SUMMARY"
    fi
done
echo "=== done. summary: $SUMMARY ==="
cat "$SUMMARY"
