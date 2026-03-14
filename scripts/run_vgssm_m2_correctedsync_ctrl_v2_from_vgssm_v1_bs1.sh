#!/usr/bin/env bash
set -euo pipefail

cd ~/urbanfloodbench
source ~/miniconda3/etc/profile.d/conda.sh
conda activate santa2025

python train_vgssm_standalone.py \
  --model_id 2 \
  --exp_name vgssm_m2_correctedsync_ctrl_v2_from_vgssm_v1_bs1 \
  --max_epochs 12 \
  --batch_size 1 \
  --accumulate_grad_batches 4 \
  --accelerator cuda \
  --devices 1 \
  --data_dir ./data \
  --prediction_horizon 90 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --precision 32 \
  --future_inlet_mode_train missing \
  --recon_balance_mode equal \
  --horizon_weight_by_valid_count \
  --init_ckpt checkpoints/model_2/vgssm_v1/best.ckpt \
  > training_vgssm_m2_correctedsync_ctrl_v2_from_vgssm_v1_bs1.log 2>&1
