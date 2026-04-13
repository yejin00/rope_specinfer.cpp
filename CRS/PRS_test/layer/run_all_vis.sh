#!/bin/bash

# Paths (Absolute paths for safety)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
COMPARE_SCRIPT="$ROOT_DIR/scripts/compare_absmax.py"

# Files provided by user
DATA_FILE="$ROOT_DIR/CRS/PRS_test/rope_longbench.bin"
SCALES_FILE="$ROOT_DIR/CRS/PRS_test/scales_sqrt_cap_k4.bin"
OUTPUT_DIR="$SCRIPT_DIR/vis_longbench"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "Generating Visualizations for Rope LongBench Data"
echo "Script: $COMPARE_SCRIPT"
echo "Data:   $DATA_FILE"
echo "Scales: $SCALES_FILE"
echo "Output: $OUTPUT_DIR"
echo "========================================================"

# Loop through Layers 0-31 and Heads 0-7
for layer in {0..31}; do
    for head in {0..7}; do
        echo "[Processing] Layer $layer, Head $head..."
        
        python3 "$COMPARE_SCRIPT" \
            "$DATA_FILE" \
            "$DATA_FILE" \
            "$SCALES_FILE" \
            --layer "$layer" \
            --head "$head" \
            --output "$OUTPUT_DIR/vis_l${layer}_h${head}.png" \
            > /dev/null 2>&1  # Suppress stdout to reduce noise, errors will still show
            
        # Optional: Print progress dot
        echo -n "."
    done
    echo "" # Newline after each layer
done

echo "========================================================"
echo "Done! All visualizations saved to: $OUTPUT_DIR"
echo "========================================================"
