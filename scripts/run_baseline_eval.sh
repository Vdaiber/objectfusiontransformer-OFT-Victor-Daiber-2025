#!/bin/bash

# Baseline Virtual Sensor Evaluation Script
# 
# This script runs baseline evaluation for all virtual sensors
# without any transformer processing.
#
# Usage:
#   ./scripts/run_baseline_eval.sh [config_file] [output_dir]
#
# Examples:
#   ./scripts/run_baseline_eval.sh
#   ./scripts/run_baseline_eval.sh config/pipeline_staged.yaml
#   ./scripts/run_baseline_eval.sh config/pipeline_staged.yaml results/baseline_eval

set -e  # Exit on any error

# Default values
CONFIG_FILE=${1:-"config/pipeline_staged.yaml"}
OUTPUT_DIR=${2:-"evaluation_results/baseline_$(date +%Y%m%d_%H%M%S)"}

echo "=========================================="
echo "VIRTUAL SENSOR BASELINE EVALUATION"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run baseline evaluation
echo "🚀 Starting baseline evaluation..."
python src/oft/transformer/scripts/evaluate_virtual_sensors_baseline.py \
    --config "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Baseline evaluation completed successfully!"
    echo "📊 Results saved to: $OUTPUT_DIR"
    echo ""
    echo "📁 Generated files:"
    ls -la "$OUTPUT_DIR"
    echo ""
    echo "🔍 To view results:"
    echo "   ls $OUTPUT_DIR/*_metrics.json"
    echo "   cat $OUTPUT_DIR/virtual_lidar_metrics.json"
else
    echo "❌ Baseline evaluation failed!"
    exit 1
fi