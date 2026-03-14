#!/bin/bash
# Combined training and prediction script for Graph-TFT
# Usage: ./train_and_predict.sh <exp_name> [max_epochs] [calibration_steps]
#
# Example: ./train_and_predict.sh graph_tft_v4 60 100

set -e

# Parameters
EXP_NAME=${1:-graph_tft_v3}
MAX_EPOCHS=${2:-60}
CALIBRATION_STEPS=${3:-100}
BATCH_SIZE=${4:-4}

echo "============================================================"
echo "Training Configuration:"
echo "  Experiment: $EXP_NAME"
echo "  Max Epochs: $MAX_EPOCHS"
echo "  Calibration Steps: $CALIBRATION_STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "============================================================"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate santa2025
cd ~/urbanfloodbench

# Train Model 1
echo ""
echo "[$(date)] Starting Model 1 training..."
python train_graph_tft.py \
    --model_id 1 \
    --exp_name $EXP_NAME \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --accelerator cuda \
    --data_dir ./data

echo "[$(date)] Model 1 training complete!"

# Train Model 2
echo ""
echo "[$(date)] Starting Model 2 training..."
python train_graph_tft.py \
    --model_id 2 \
    --exp_name $EXP_NAME \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --accelerator cuda \
    --data_dir ./data

echo "[$(date)] Model 2 training complete!"

# Find best checkpoints
echo ""
echo "[$(date)] Finding best checkpoints..."
M1_BEST=$(find checkpoints/model_1/$EXP_NAME/ -name '*.ckpt' ! -name 'last.ckpt' -type f 2>/dev/null | head -1)
M2_BEST=$(find checkpoints/model_2/$EXP_NAME/ -name '*.ckpt' ! -name 'last.ckpt' -type f 2>/dev/null | head -1)

if [ -z "$M1_BEST" ]; then
    M1_BEST="checkpoints/model_1/$EXP_NAME/last.ckpt"
fi
if [ -z "$M2_BEST" ]; then
    M2_BEST="checkpoints/model_2/$EXP_NAME/last.ckpt"
fi

echo "Model 1 checkpoint: $M1_BEST"
echo "Model 2 checkpoint: $M2_BEST"

# Generate predictions
echo ""
echo "[$(date)] Generating Model 1 predictions (calibration_steps=$CALIBRATION_STEPS)..."
python predict_graph_tft.py \
    --model_id 1 \
    --checkpoint "$M1_BEST" \
    --calibrate_latent \
    --calibration_steps $CALIBRATION_STEPS

echo ""
echo "[$(date)] Generating Model 2 predictions (calibration_steps=$CALIBRATION_STEPS)..."
python predict_graph_tft.py \
    --model_id 2 \
    --checkpoint "$M2_BEST" \
    --calibrate_latent \
    --calibration_steps $CALIBRATION_STEPS

# Format submission
echo ""
echo "[$(date)] Formatting submission..."
python format_submission.py

echo ""
echo "============================================================"
echo "[$(date)] All done!"
echo "Submission file: submission_final.parquet"
ls -la submission_*.parquet
echo "============================================================"
