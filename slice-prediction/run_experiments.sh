#!/bin/bash
# Quick start script for running all experiments

set -e  # Exit on error

DATA_DIR="/data/BraTS/training"  # Update this path
OUTPUT_DIR="./checkpoints"
RESULTS_DIR="./results"

echo "====================================="
echo "Middleslice Reconstruction Experiments"
echo "====================================="

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $RESULTS_DIR

# 1. Train Swin-UNETR baseline
echo ""
echo ">>> Training Swin-UNETR baseline..."
python train.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_type swin \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-4 \
  --img_size 256

# 2. Train Haar wavelet model
echo ""
echo ">>> Training Fast-cWDM-Haar..."
python train.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_type wavelet_haar \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-4 \
  --img_size 256

# 3. Train db2 wavelet model
echo ""
echo ">>> Training Fast-cWDM-db2..."
python train.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_type wavelet_db2 \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-4 \
  --img_size 256

# 4. Evaluate all models
echo ""
echo ">>> Evaluating Swin-UNETR..."
python evaluate.py \
  --checkpoint $OUTPUT_DIR/swin_best.pth \
  --model_type swin \
  --data_dir $DATA_DIR \
  --output $RESULTS_DIR/swin \
  --batch_size 16

echo ""
echo ">>> Evaluating Haar..."
python evaluate.py \
  --checkpoint $OUTPUT_DIR/wavelet_haar_best.pth \
  --model_type wavelet_haar \
  --data_dir $DATA_DIR \
  --output $RESULTS_DIR/haar \
  --batch_size 16

echo ""
echo ">>> Evaluating db2..."
python evaluate.py \
  --checkpoint $OUTPUT_DIR/wavelet_db2_best.pth \
  --model_type wavelet_db2 \
  --data_dir $DATA_DIR \
  --output $RESULTS_DIR/db2 \
  --batch_size 16

# 5. Generate comparison figure
echo ""
echo ">>> Generating comparison figure..."
python generate_figure2.py \
  --data_dir $DATA_DIR \
  --swin_checkpoint $OUTPUT_DIR/swin_best.pth \
  --haar_checkpoint $OUTPUT_DIR/wavelet_haar_best.pth \
  --db2_checkpoint $OUTPUT_DIR/wavelet_db2_best.pth \
  --output ./figures/figure2_comparison.pdf

echo ""
echo "====================================="
echo "All experiments completed!"
echo "====================================="
echo ""
echo "Results saved in: $RESULTS_DIR"
echo "Checkpoints saved in: $OUTPUT_DIR"
echo "Figure saved in: ./figures/"
echo ""
echo "Check metrics:"
echo "  - Swin:  $RESULTS_DIR/swin/metrics_summary.csv"
echo "  - Haar:  $RESULTS_DIR/haar/metrics_summary.csv"
echo "  - db2:   $RESULTS_DIR/db2/metrics_summary.csv"