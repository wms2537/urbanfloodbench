#!/bin/bash
# All-in-one training and prediction script for Graph-TFT
# Runs everything sequentially - no background processes
#
# Usage: ./train_and_predict_all.sh <exp_name> [options]
# Example: ./train_and_predict_all.sh graph_tft_v5 --hidden_dim 128 --loss_type huber --patience 20

set -e

# Default parameters
EXP_NAME=${1:-graph_tft}
shift || true

# Parse remaining args (will be passed to training)
EXTRA_ARGS="$@"

# Fixed parameters
MAX_EPOCHS=60
BATCH_SIZE=4
CALIBRATION_STEPS=100

echo "============================================================"
echo "Graph-TFT Full Pipeline"
echo "============================================================"
echo "Experiment: $EXP_NAME"
echo "Extra args: $EXTRA_ARGS"
echo "============================================================"

# Setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate santa2025
cd ~/urbanfloodbench

# Train Model 1
echo ""
echo "[$(date)] ========== TRAINING MODEL 1 =========="
python train_graph_tft.py \
    --model_id 1 \
    --exp_name $EXP_NAME \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --accelerator cuda \
    --data_dir ./data \
    $EXTRA_ARGS

echo "[$(date)] Model 1 training complete!"

# Train Model 2
echo ""
echo "[$(date)] ========== TRAINING MODEL 2 =========="
python train_graph_tft.py \
    --model_id 2 \
    --exp_name $EXP_NAME \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --accelerator cuda \
    --data_dir ./data \
    $EXTRA_ARGS

echo "[$(date)] Model 2 training complete!"

# Find best checkpoints
echo ""
echo "[$(date)] ========== FINDING CHECKPOINTS =========="
M1_BEST=$(find checkpoints/model_1/$EXP_NAME/ -name '*.ckpt' ! -name 'last.ckpt' -type f 2>/dev/null | head -1)
M2_BEST=$(find checkpoints/model_2/$EXP_NAME/ -name '*.ckpt' ! -name 'last.ckpt' -type f 2>/dev/null | head -1)

# Fallback to last.ckpt if no best found
[ -z "$M1_BEST" ] && M1_BEST="checkpoints/model_1/$EXP_NAME/last.ckpt"
[ -z "$M2_BEST" ] && M2_BEST="checkpoints/model_2/$EXP_NAME/last.ckpt"

echo "Model 1 checkpoint: $M1_BEST"
echo "Model 2 checkpoint: $M2_BEST"

# Generate predictions
echo ""
echo "[$(date)] ========== GENERATING PREDICTIONS =========="
echo "Using calibration_steps=$CALIBRATION_STEPS"

python predict_graph_tft.py \
    --model_id 1 \
    --checkpoint "$M1_BEST" \
    --calibrate_latent \
    --calibration_steps $CALIBRATION_STEPS

echo "[$(date)] Model 1 predictions complete!"

python predict_graph_tft.py \
    --model_id 2 \
    --checkpoint "$M2_BEST" \
    --calibrate_latent \
    --calibration_steps $CALIBRATION_STEPS

echo "[$(date)] Model 2 predictions complete!"

# Format submission
echo ""
echo "[$(date)] ========== FORMATTING SUBMISSION =========="
python format_submission_v2.py

echo ""
echo "============================================================"
echo "[$(date)] PIPELINE COMPLETE!"
echo "============================================================"
echo "Submission file:"
ls -lah submission_final.parquet
echo ""
echo "Water level stats:"
python -c "import pandas as pd; df=pd.read_parquet('submission_final.parquet'); print(df['water_level'].describe())"
echo "============================================================"
