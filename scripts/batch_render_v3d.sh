#!/bin/bash
# Batch-render stabilized videos for all cases in /mnt/c/A_TangJian/opt/.
# Uses the v3d optimizer output; adds a 0-based frame number (top-left).
# Output: <case>/our_out_v3d/final_stabilized_render2.mp4
set -u
OPT_DIR=/mnt/c/A_TangJian/opt
SCRIPT=/home/tangjian5/quartet/utils/render_videos.py
SUMMARY=/tmp/batch_render_v3d_summary.txt

: > "$SUMMARY"
for C in "$OPT_DIR"/*/; do
    NAME=$(basename "$C")
    VID=$(ls "$C"/*.mp4 2>/dev/null | head -1)
    [ -n "$VID" ] || { echo "$NAME: no video" | tee -a "$SUMMARY"; continue; }
    [ -f "$C/our_out_v3d/optimized_quaternions.csv" ] || { echo "$NAME: no v3d output" | tee -a "$SUMMARY"; continue; }
    START=$(date +%s)
    if python3 "$SCRIPT" \
        --video   "$VID" \
        --grid    "$C/DumpHIS/DbgFile_WarpGrids_RSCFRMMid.csv" \
        --quat-raw "$C/DumpHIS/DbgFile_Frame_OriginalCamOrient.csv" \
        --quat-opt "$C/our_out_v3d/optimized_quaternions.csv" \
        --yaml    "$C/conf.yaml" \
        --out     "$C/our_out_v3d" \
        > "$C/our_out_v3d/render_log.txt" 2>&1; then
        END=$(date +%s)
        echo "$NAME: OK time=$((END - START))s" | tee -a "$SUMMARY"
    else
        echo "$NAME: FAILED (see $C/our_out_v3d/render_log.txt)" | tee -a "$SUMMARY"
    fi
done
echo "=== done ===" | tee -a "$SUMMARY"
