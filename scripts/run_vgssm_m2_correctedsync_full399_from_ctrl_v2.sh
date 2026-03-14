#!/usr/bin/env bash
set -euo pipefail

cd ~/urbanfloodbench
source ~/miniconda3/etc/profile.d/conda.sh
conda activate santa2025

python train_vgssm_standalone.py \
  --model_id 2 \
  --exp_name vgssm_m2_correctedsync_full399_from_ctrl_v2 \
  --max_epochs 20 \
  --batch_size 1 \
  --accumulate_grad_batches 4 \
  --accelerator cuda \
  --devices 1 \
  --data_dir ./data \
  --prediction_horizon 399 \
  --strict_horizon_check \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --precision 32 \
  --future_inlet_mode_train missing \
  --recon_balance_mode equal \
  --horizon_weight_by_valid_count \
  --use_curriculum \
  --curriculum_stages 90,128,195,256,399 \
  --epochs_per_stage 2 \
  --patience_per_stage 2 \
  --patience 20 \
  --init_ckpt checkpoints/model_2/vgssm_m2_correctedsync_ctrl_v2_from_vgssm_v1_bs1/best.ckpt \
  > training_vgssm_m2_correctedsync_full399_from_ctrl_v2.log 2>&1
