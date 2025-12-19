#!/bin/bash
# Simple shell script to run Athene reward scoring
# Usage: bash run_athene_scoring.sh <input_file> <output_file>

# Configuration
MODEL_PATH="${MODEL_PATH:-Nexusflow/Athene-RM-8B}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: bash run_athene_scoring.sh <input_file> [output_file] [batch_size]"
    echo ""
    echo "Example:"
    echo "  bash run_athene_scoring.sh data.json"
    echo "  bash run_athene_scoring.sh data.json results.json"
    echo "  bash run_athene_scoring.sh data.json results.json 32"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_PATH - Path to Athene model (default: Nexusflow/Athene-RM-8B)"
    echo "  BATCH_SIZE - Batch size for processing (default: 16)"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-${INPUT_FILE%.json}_rewards.json}"
BATCH_SIZE="${3:-$BATCH_SIZE}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Print configuration
echo "==================================="
echo "Athene Reward Scoring"
echo "==================================="
echo "Model Path: $MODEL_PATH"
echo "Input File: $INPUT_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Batch Size: $BATCH_SIZE"
echo "==================================="
echo ""

# Run the scoring script
python score_athene_rewards.py \
    --model-path "$MODEL_PATH" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --batch-size "$BATCH_SIZE" \
    --device auto \
    --log-file "${OUTPUT_FILE%.json}.log"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "✓ Scoring completed successfully"
    echo "Results: $OUTPUT_FILE"
    echo "Logs: ${OUTPUT_FILE%.json}.log"
    echo "==================================="
else
    echo ""
    echo "==================================="
    echo "✗ Scoring failed with exit code $exit_code"
    echo "Check logs for details: ${OUTPUT_FILE%.json}.log"
    echo "==================================="
fi

exit $exit_code