# Modal Training Quickstart

## Status Note (2026-03-15)

- The original instructions below are historical and focused on `modal_dualflood.py`.
- The latest corrected-data VGSSM workflow uses:
  - `modal_vgssm.py`
  - `scripts/sync_corrected_1d_dynamic_to_modal.sh`
  - `scripts/run_modal_vgssm_m2_correctedsync_tail_exp_from_epoch15.sh`
- Important volume-layout note:
  - the live dataset root inside the Modal volume is the subtree under `/data/data`
  - inside the container this resolves through the mounted volume path and is detected automatically by `modal_vgssm.py`
- Before launching corrected rerelease training, overwrite the affected `196` `1d_nodes_dynamic_all.csv` files into the live root using the sync script.

This project now includes a Modal launcher at:

- `modal_dualflood.py`

It runs `train_dual_flood.py` on a high-VRAM Modal GPU and persists outputs to Modal Volumes.

## 1) One-time setup

Create volumes:

```bash
modal volume create urbanfloodbench-data
modal volume create urbanfloodbench-checkpoints
modal volume create urbanfloodbench-logs
```

Upload dataset (local `./data` -> volume root):

```bash
cd /Users/sohweimeng/Documents/projects/urbanfloodbench
modal volume put urbanfloodbench-data ./data /
```

## 2) Launch training on Modal

Run with your normal `train_dual_flood.py` args:

```bash
cd /Users/sohweimeng/Documents/projects/urbanfloodbench
modal run modal_dualflood.py --train-args "--model_id 2 --exp_name dual_flood_modal_test --max_epochs 1 --batch_size 1 --pred_len 399 --accumulate_grad_batches 8 --accelerator cuda --data_dir ./data --precision 16-mixed --lr 1e-5 --transition_scale 0.05 --coupling_scale 0.03 --lambda_edge 0.03 --lambda_physics 0.002 --physics_mode hybrid --horizon_weight_power 1.5 --future_inlet_mode_train mixed --future_inlet_mode_val missing --future_inlet_dropout_prob_train 0.3 --future_inlet_seq_dropout_prob_train 0.7 --val_start_only --init_ckpt checkpoints/model_2/dual_flood_v11_push_maskstaticedge_h399/best.ckpt --use_moe_transition --moe_num_experts 4"
```

## 3) Inspect artifacts

Check outputs in volumes:

```bash
modal volume ls urbanfloodbench-checkpoints /
modal volume ls urbanfloodbench-logs /
```

Download artifacts if needed:

```bash
modal volume get urbanfloodbench-checkpoints / ./modal_checkpoints
modal volume get urbanfloodbench-logs / ./modal_logs
```
