#!/bin/bash
# Batch process multiple JSON files with Athene reward scoring
# Usage: bash batch_score_multiple.sh <input_directory> <output_directory>

MODEL_PATH="${MODEL_PATH:-/checkpoint/ram/tianjian/reward_models/Athene-RM-8B}"
BATCH_SIZE="${BATCH_SIZE:-16}"

if [ $# -lt 2 ]; then
    echo "Usage: bash batch_score_multiple.sh <input_directory> <output_directory>"
    echo ""
    echo "Example:"
    echo "  bash batch_score_multiple.sh ./data/ ./results/"
    echo ""
    echo "This will process all .json files in input_directory and save results to output_directory"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_PATH - Path to Athene model (default: /checkpoint/ram/tianjian/reward_models/Athene-RM-8B)"
    echo "  BATCH_SIZE - Batch size for processing (default: 16)"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' not found!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Count JSON files
json_files=("$INPUT_DIR"/*.json)
num_files=${#json_files[@]}

if [ $num_files -eq 0 ]; then
    echo "Error: No .json files found in '$INPUT_DIR'"
    exit 1
fi

echo "==================================="
echo "Batch Athene Reward Scoring"
echo "==================================="
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Model Path: $MODEL_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "Files to process: $num_files"
echo "==================================="
echo ""

# Process each JSON file
processed=0
failed=0

for input_file in "$INPUT_DIR"/*.json; do
    # Get filename without path
    filename=$(basename "$input_file")
    output_file="$OUTPUT_DIR/${filename%.json}_rewards.json"
    log_file="$OUTPUT_DIR/${filename%.json}_rewards.log"

    echo "[$((processed + failed + 1))/$num_files] Processing: $filename"

    # Run the scoring script
    python score_athene_rewards.py \
        --model-path "$MODEL_PATH" \
        --input "$input_file" \
        --output "$output_file" \
        --batch-size "$BATCH_SIZE" \
        --device auto \
        --log-file "$log_file" 2>&1 | tee -a "$log_file"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "  ✓ Success: $output_file"
        ((processed++))
    else
        echo "  ✗ Failed: $filename (see $log_file)"
        ((failed++))
    fi
    echo ""
done

# Print summary
echo "==================================="
echo "Batch Processing Complete"
echo "==================================="
echo "Total files: $num_files"
echo "Successful: $processed"
echo "Failed: $failed"
echo "Results saved to: $OUTPUT_DIR"
echo "==================================="

# Exit with error if any failed
if [ $failed -gt 0 ]; then
    exit 1
fi