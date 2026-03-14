#!/usr/bin/env bash
set -euo pipefail

cd ~/urbanfloodbench
source ~/miniconda3/etc/profile.d/conda.sh
conda activate santa2025

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

python train_vgssm_standalone.py \
  --model_id 2 \
  --exp_name vgssm_m2_correctedsync_full399_tail_exp_allocfix_from_epoch15 \
  --max_epochs 12 \
  --batch_size 1 \
  --accumulate_grad_batches 4 \
  --accelerator cuda \
  --devices 1 \
  --data_dir ./data \
  --prediction_horizon 399 \
  --strict_horizon_check \
  --lr 2e-5 \
  --weight_decay 1e-4 \
  --precision 32 \
  --future_inlet_mode_train missing \
  --recon_balance_mode equal \
  --horizon_weight_by_valid_count \
  --horizon_weighting exp \
  --patience 8 \
  --init_ckpt checkpoints/model_2/vgssm_m2_correctedsync_full399_from_ctrl_v2/epoch=15-val_std_rmse=0.0999.ckpt \
  > training_vgssm_m2_correctedsync_full399_tail_exp_allocfix_from_epoch15.log 2>&1
