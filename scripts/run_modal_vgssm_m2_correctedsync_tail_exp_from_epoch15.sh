#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_LOCAL="${ROOT_DIR}/checkpoints/model_2/epoch=15-val_std_rmse=0.0999.ckpt"
CKPT_REMOTE="/model_2/vgssm_m2_correctedsync_full399_from_ctrl_v2/epoch=15-val_std_rmse=0.0999.ckpt"

cd "$ROOT_DIR"

if ! command -v modal >/dev/null 2>&1; then
  echo "modal CLI not found in PATH" >&2
  exit 1
fi

if [[ ! -f "$CKPT_LOCAL" ]]; then
  echo "Missing local checkpoint: $CKPT_LOCAL" >&2
  exit 1
fi

echo "Uploading init checkpoint to Modal checkpoint volume..."
modal volume put -f urbanfloodbench-checkpoints "$CKPT_LOCAL" "$CKPT_REMOTE"

echo "Launching corrected-data Model 2 tail-weighted continuation on Modal..."
modal run modal_vgssm.py --train-args "\
--model_id 2 \
--exp_name vgssm_m2_correctedsync_full399_tail_exp_modal_v1 \
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
--check_val_every_n_epoch 1 \
--patience 8 \
--init_ckpt checkpoints/model_2/vgssm_m2_correctedsync_full399_from_ctrl_v2/epoch=15-val_std_rmse=0.0999.ckpt"
