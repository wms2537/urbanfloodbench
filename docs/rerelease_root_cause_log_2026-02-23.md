# UrbanFloodBench Rerelease Root-Cause Log (2026-02-23)

## Scope
- Goal: understand why recent VGSSM runs plateau and fix root causes scientifically.
- Dataset context: competition rerelease fixed 1D dynamic node mapping; retraining is required.

## Current Status Snapshot
- Best known Kaggle submissions in this branch of work (user-reported screenshot):
1. `submission_dualflood_v11_missing_last_blend50_on_vgssm_base.parquet`: `0.0891`
2. `submission_dualflood_v11_last_on_vgssm_base.parquet`: `0.0892`
3. `submission_dualflood_v11_missing_on_vgssm_base.parquet`: `0.0893`
- Reported failure cases:
1. One run scored around `0.2472`.
2. One malformed/buggy submission path produced around `1.44`.

## Evidence Collected

### A) Model checkpoints and val metrics
- Model 1:
1. `/vol/checkpoints/model_1/vgssm_rerelease_m1_fullrollout_fp32_v2_curr_nobound_finetune/best.ckpt`
2. Best `val_std_rmse` in checkpoint: `0.0296`
- Model 2:
1. `/vol/checkpoints/model_2/vgssm_rerelease_m2_fullrollout_fp32_v1/best.ckpt`
2. Best `val/std_rmse` in logs: `0.0512`
- Recent poor Model 2 finetune:
1. `/vol/checkpoints/model_2/vgssm_rerelease_m2_truefull399_fullwin_physfinetune_v3_s8_lr5e5/best.ckpt`
2. Best `val_std_rmse`: `0.1375`

### B) Physics loss contribution check
- In the poor finetune run (`...physfinetune_v3_s8_lr5e5`):
1. `train/physics_contribution` remained tiny (`~1e-7` to `1e-4`).
2. Plateau happened despite near-zero physics contribution.
- Conclusion: this plateau was not caused by physics loss dominating optimization.

### C) Loss/metric mismatch diagnosis (mathematical)
- Competition objective averages 1D and 2D errors, effectively equalizing subsystem importance.
- Legacy reconstruction objective in current script had a masking normalization bug:
1. Mask `[batch]` was expanded for multiplication but denominator was only `mask.sum()` (batch count), not element count.
2. This scales loss roughly with node count and overweights 2D nodes.
- Node-count imbalance:
1. Model 1: 1D `17`, 2D `3716` -> legacy weighting approx `0.455%` (1D) vs `99.545%` (2D).
2. Model 2: 1D `198`, 2D `4299` -> legacy weighting approx `4.403%` (1D) vs `95.597%` (2D).
- This is a direct objective mismatch with leaderboard scoring and explains unstable/biased optimization behavior.

## Root-Cause Decision Matrix
- Architecture problem:
1. Not primary root cause for the current plateau evidence.
2. Strong checkpoints already exist with same architecture family (M1 `0.0296`, M2 `0.0512` val).
- Loss problem:
1. Yes, confirmed root cause in training objective normalization and subsystem weighting.
- Physics modeling problem:
1. Not primary for this plateau (physics term near-zero in failed finetune).
2. Physics can still be explored later, but only after core objective alignment.
- Latent modeling problem:
1. Possible secondary factor, but not the first blocker shown by current evidence.
2. Fixing optimization objective is prerequisite before deeper latent redesign.

## Code Fixes Implemented

### 1) Reconstruction loss normalization bug fix
- File: `/Users/sohweimeng/Documents/projects/urbanfloodbench/train_vgssm_standalone.py`
- Change:
1. `_compute_loss` now expands masks to element-wise shape and normalizes by total valid elements.
2. This removes artificial scaling by node count.

### 2) Metric-aligned 1D/2D reconstruction balancing
- File: `/Users/sohweimeng/Documents/projects/urbanfloodbench/train_vgssm_standalone.py`
- New args:
1. `--recon_balance_mode {equal,sum}` (default `equal`)
2. `--recon_weight_1d` (default `0.5`)
3. `--recon_weight_2d` (default `0.5`)
- Trainer behavior:
1. `equal`: `loss_recon = w1 * loss_1d + w2 * loss_2d` (metric-aligned default).
2. `sum`: legacy behavior retained for ablations.

### 3) Additional logging for scientific debugging
- File: `/Users/sohweimeng/Documents/projects/urbanfloodbench/train_vgssm_standalone.py`
- Added logs:
1. `train/loss_recon_1d`
2. `train/loss_recon_2d`
3. `train/recon_weight_1d`
4. `train/recon_weight_2d`
5. `train/recon_balance_sum_mode`

### 4) Faster controlled probes without changing default training
- File: `/Users/sohweimeng/Documents/projects/urbanfloodbench/train_vgssm_standalone.py`
- New arg:
1. `--limit_train_batches` (default `1.0`)
- Purpose:
1. run short, reproducible probe epochs for fast val feedback while keeping full training defaults unchanged.

### 5) Horizon weighting fix for variable-rollout windows
- File: `/Users/sohweimeng/Documents/projects/urbanfloodbench/train_vgssm_standalone.py`
- New arg:
1. `--horizon_weight_by_valid_count` (off by default for backward-compatible ablations)
- Change:
1. Training now optionally scales each horizon weight by the number of valid samples at that step.
2. This avoids over-weighting late rollout steps with very low support when using variable-length window masks.
3. Added log: `train/horizon_weight_by_valid_count`.

## Training Operations Notes (Modal)
- Preemption:
1. Existing launcher already supports preemptible runs and restart-from-last behavior.
- Runtime issues encountered during root-fix rollout:
1. OOM with resumed training at `batch_size=4` and horizon `399`.
2. BF16 failed due fused GRU kernel incompatibility (`_thnn_fused_gru_cell_cuda` for BFloat16).
3. FP16 mixed avoided OOM but produced early NaN in this setup.
4. Active stable path is FP32 with `batch_size=1` + accumulation.

## Modal Log Aggregation Update (2026-02-23 late)
- Parsed TensorBoard event logs from Modal volume for historical runs.
- Model 1 (best observed runs):
1. `vgssm_rerelease_m1_aligned_currfixed_v2`: best `0.0065` (not full-rollout-safe; curriculum-heavy)
2. `vgssm_rerelease_m1_fullrollout_fp32_v2_curr_nobound_finetune`: best `0.0296` (full-rollout capable)
3. `vgssm_rerelease_m1_fullrollout_fp32_v1_corrected`: best `0.0854`
- Model 2 (best observed runs):
1. `vgssm_rerelease_m2_fullrollout_fp32_v1`: best `0.0512` (strongest corrected-data baseline found)
2. `vgssm_rerelease_m2_fullrollout_fp32_v2_initreset_bs8`: best `0.0686`
3. `vgssm_rerelease_sigmoid_curriculumforce_m2_probe_v1`: best `0.0610` but later unstable
4. `vgssm_rerelease_m2_lossbalance_rootfix_v1d`: best `0.1337`
5. `vgssm_rerelease_m2_scratch_corrected_v2_h128`: best `0.4587`

## Operational Diagnosis Update
- The worst recent probes (`~1.0` val) were short probe runs with aggressive constraints (`limit_train_batches=0.1`, high LR variants), not representative of full training convergence.
- Best reproducible corrected-data baseline remains the historical full-rollout Model 2 checkpoint at `0.0512`.
- GPU utilization on current L40S path remains moderate (`~25-30%` during steady train steps) with `batch_size=1`, indicating a data/sequence-length bound regime rather than pure compute saturation.

## What Worked vs Did Not Work
- Worked:
1. Full-horizon VGSSM checkpoints can reach strong validation (M1 `0.0296`, M2 `0.0512`).
2. FP32 with small batch is stable.
3. Loss normalization + metric alignment patch compiles and passes numeric sanity checks.
- Did not work:
1. Recent M2 physics finetune (`...v3_s8_lr5e5`) plateaued badly (`~0.1375`).
2. BF16 path with current GRU stack is not usable.
3. FP16 mixed in this exact finetune path produced NaN quickly.

## Immediate Next Scientific Steps
1. Continue from strongest M2 baseline checkpoint (`m2_fullrollout_fp32_v1`, best `0.0512`) with low-LR continuation and full rollout.
2. Use per-epoch validation with `val_start_only=True` for faster convergence signal, then confirm with full-window validation once trend is positive.
3. Rebuild submission from the strongest full-rollout pair:
4. Model 1: `vgssm_rerelease_m1_fullrollout_fp32_v2_curr_nobound_finetune`
5. Model 2: `vgssm_rerelease_m2_fullrollout_fp32_v1` (or improved continuation checkpoint)
6. Validate exact submission schema and key alignment against `sample_submission.parquet` before any Kaggle upload.

## Current Active Run (latest)
1. App: `ap-zjALt775f1nNM8lvDF3V4P`
2. Experiment: `vgssm_rerelease_m2_fullrollout_fp32_v1_cont_lr1e4_e1val_v1`
3. Init checkpoint: `/vol/checkpoints/model_2/vgssm_rerelease_m2_fullrollout_fp32_v1/best.ckpt`
4. Key settings:
5. `--batch_size 1 --accumulate_grad_batches 8 --precision 32`
6. `--prediction_horizon 399 --future_inlet_mode_train missing`
7. `--check_val_every_n_epoch 1 --val_start_only --max_epochs 12`

## Architecture + Physics Upgrade (2026-02-26)

### A) Physics redesign: residual conservation in physical units
- File: `/Users/sohweimeng/Documents/projects/urbanfloodbench/train_vgssm_standalone.py`
- Added:
1. `PhysicsResidualLoss` with normalized local and global mass-balance residuals:
   - `r_i = A_i * (h_i(t)-h_i(t-1))/dt - (Q_in-Q_out) - S_i`
2. Denormalization helpers so physics is computed in physical units (not normalized target units).
3. Learnable source scaling for:
   - 1D inlet forcing (`exp(physics_source_log_scale_1d)`)
   - 2D rainfall forcing (`exp(physics_source_log_scale_2d)`)
4. Conductance prior regularization on `PhysicsBasedFlow` (`K_pipe`, `K_surface`, `K_coupling`) to avoid degenerate scaling.
5. Optimizer now includes trainer-level learnable parameters (physics source scales), not only model weights.

- New CLI args:
1. `--physics_loss_mode {residual,light,legacy}` (default `residual`)
2. `--physics_residual_huber_delta` (default `0.1`)

- Operational behavior:
1. Residual mode no longer uses legacy `EdgeFlowHead` rollout flow tensors.
2. Physics contribution uses fixed normalized scaling in residual mode (no ratio shrinkage), improving optimizer signal stability.

### B) Timer successor integration into transition dynamics
- File: `/Users/sohweimeng/Documents/projects/urbanfloodbench/train_vgssm_standalone.py`
- Added:
1. `TimerTemporalPriorV5` (derivative-aware, bidirectional history encoding, adaptive last-vs-mean fusion).
2. `TimerEnhancedTransition` now supports:
   - `timer_variant='v3'|'v4'|'v5'` (v4/v5 route to the new prior)
   - gated Timer contribution on 1D nodes
   - optional low-cost 2D temporal context path (`timer_enable_2d_context`)
3. `VGSSM` transition now activates Timer transition when either `--use_timer` or `--use_timer_v4` is enabled.
4. Added safe routing flag `transition_uses_timer_history` to avoid history argument mismatch when using non-Timer transitions.

- New CLI args:
1. `--timer_transition_variant {auto,v3,v5}` (default `auto`)
2. `--timer_enable_2d_context`

### C) Validation and smoke checks
1. `python -m py_compile train_vgssm_standalone.py` passed.
2. Smoke-tested:
   - `TimerEnhancedTransition` with v5 + 2D context
   - `PhysicsResidualLoss` forward pass

## Early-Stopping Root Fix + New Run (2026-03-04)

### A) New root cause confirmed
- Failure pattern in `vgssm_m2_rerelease_v6_h399_strict_v2`:
1. Curriculum advanced only to stage 3/11 (`rollout_len=8`), never reached long-horizon stages.
2. Validation was unstable and poor (best around `0.6990`, then mostly `~0.76-1.55`).
- Mathematical implication:
1. Training objective stayed short-horizon while competition metric is long-horizon rollout.
2. This creates objective mismatch and high compounding rollout error.

### B) Bug fix implemented
- File: `/Users/sohweimeng/Documents/projects/urbanfloodbench/train_vgssm_standalone.py`
- Class: `CurriculumAwareEarlyStopping`
- Fix:
1. Added shared guard `_should_skip(...)`.
2. Applied guard to both `on_validation_end` and `on_train_epoch_end`.
3. Prevents early-stopping decisions before curriculum reaches final stage.

### C) Relaunched detached training (warm start, full curriculum)
- Host run:
1. `exp_name=vgssm_m2_rerelease_v7_rootfix_curr399`
2. init checkpoint: `checkpoints/model_2/vgssm_timer_v4_best/best.ckpt`
3. full stages: `1,4,8,16,32,64,90,128,195,256,399`
4. precision/batching: `FP32`, `batch_size=1`, `accumulate_grad_batches=8`
5. horizon: `399` with strict check enabled
- Log path:
1. `/home/soh/urbanfloodbench/training_vgssm_m2_rerelease_v7_rootfix_curr399.log`

### D) Submission gate (strict)
Only generate Kaggle submission if all are true:
1. Curriculum reaches final stage (`rollout_len=399`).
2. No NaN/inf in training or validation.
3. Validation on corrected data is competitive vs our stable baseline (`<=0.0512` target region).

## Remote Dataset Mismatch Root Cause (2026-03-11)

### A) Training server was not using the corrected 1D dynamic dataset
- Verified by full-manifest hash over all affected files:
1. Local manifest over all `196` `Model_[12]/*/event_*/1d_nodes_dynamic_all.csv` files:
   - `8f2db3024489dbdf4238dc4b76e243da5fc1bb1c7daf271850303ad54df3bd3f`
2. Remote manifest before sync:
   - `09bde00006dca13d4e972ce0a1e25d18c9d6b63abd8f34b52b939a047ffbc1d9`
- Concrete file evidence:
1. Local `/Users/sohweimeng/Documents/projects/urbanfloodbench/data/Model_1/test/event_18/1d_nodes_dynamic_all.csv`
   had `node_idx=1 -> water_level=296.98056`
2. Remote `/home/soh/urbanfloodbench/data/Model_1/test/event_18/1d_nodes_dynamic_all.csv`
   had `node_idx=1 -> water_level=287.83286`
- This exactly matches the competition announcement scope: 1D dynamic node mapping was the affected component.

### B) Failed March 11 M2 replacements mainly broke 1D, not 2D
- Compared against the best currently-scored public base submission:
1. Base: `submission_vgssm_rerelease_m1v2e07_on_m2fullrolloutv1_base_20260221.parquet` (`0.2260`)
2. Full399 replacement score: `0.2279`
3. Control replacement score: `0.2400`

- Model 2 row-wise deviation vs base:
1. `full399_from_ctrl_v1`
   - mean abs delta on 1D rows: `3.54`
   - mean abs delta on 2D rows: `0.44`
2. `ctrl_v1_from_vgssm_v1_bs1`
   - mean abs delta on 1D rows: `3.34`
   - mean abs delta on 2D rows: `0.40`

- Worst-drift events were concentrated in 1D rows for events:
1. `17, 88, 99, 52, 60, 84, 65, 82, 51, 77, 73, 8`

- Scientific implication:
1. The dominant failure mode in the recent bad M2 submissions was 1D dynamic behavior.
2. That is exactly the subsystem corrupted by the stale remote dataset.
3. Therefore the stale remote 1D dataset is not a minor operational issue; it is a direct causal explanation for the recent remote-train failure pattern.

### C) Corrective action executed
- Synced all corrected affected files from local to remote:
1. `196` `1d_nodes_dynamic_all.csv` files under Model 1 and Model 2 train/test
2. Transfer summary:
   - sent `118,854,370` bytes
   - received `609,809` bytes

- Post-sync verification:
1. Remote manifest now matches local exactly:
   - `8f2db3024489dbdf4238dc4b76e243da5fc1bb1c7daf271850303ad54df3bd3f`

### D) New corrected-data control run
- Relaunched detached control training on the now-corrected remote dataset:
1. Experiment: `vgssm_m2_correctedsync_ctrl_v2_from_vgssm_v1_bs1`
2. Init checkpoint: `checkpoints/model_2/vgssm_v1/best.ckpt`
3. Key settings:
   - `prediction_horizon=90`
   - `batch_size=1`
   - `accumulate_grad_batches=4`
   - `future_inlet_mode_train=missing`
   - `recon_balance_mode=equal`
   - `horizon_weight_by_valid_count`

### E) Corrected-data full399 submission result
- Submission built from corrected-data full399 checkpoint:
1. M2 checkpoint: `checkpoints/model_2/vgssm_m2_correctedsync_full399_from_ctrl_v2/epoch=15-val_std_rmse=0.0999.ckpt`
2. M1 base: `submission_vgssm_rerelease_m1v2e07_on_m2fullrolloutv1_base_20260221.parquet`
3. Final file: `submissions/submission_20260314_m1v2e07_base_m2_correctedsync_full399_epoch15.parquet`
4. Kaggle public score: `0.0748`
- Scientific implication:
1. The corrected remote dataset sync materially improved real leaderboard performance.
2. The corrected full399 Model 2 branch is now the real baseline to beat.

### F) Holdout evaluator fixes
- `evaluate_vgssm_checkpoint_holdout.py` had three runtime bugs that blocked scientific comparison:
1. Missing architecture defaults (`hidden_dim`, `latent_dim`, `num_gnn_layers`, `num_transition_gnn_layers`, `dropout`)
2. Missing output-bound defaults (`output_bounds_*`)
3. Incorrect fallback that allowed `prediction_horizon=0` from checkpoint/runtime merge
- All three were fixed locally and synced to remote before evaluation.

### G) Competition-aligned holdout findings on corrected Model 2
- Exact split: seed-42 validation events
- Evaluation mode: autoregressive rollout, competition-aligned `max_timesteps=399`

1. `vgssm_m2_correctedsync_ctrl_v2_from_vgssm_v1_bs1` with reset-per-chunk rollout:
   - `std_rmse = 0.6579`
   - catastrophic jumps at chunk boundaries:
     - step `90`: `0.1401`
     - step `91`: `0.9080`
     - step `181`: `1.1194`
     - step `271`: `1.2391`
2. Same checkpoint with `--predict_rollout_stateful`:
   - `std_rmse = 0.2216`
   - chunk-boundary discontinuities disappear:
     - step `90`: `0.1401`
     - step `91`: `0.1404`
     - step `181`: `0.1673`
     - step `271`: `0.2400`
     - step `399`: `0.4743`
3. `vgssm_m2_correctedsync_full399_from_ctrl_v2/epoch=15-val_std_rmse=0.0999.ckpt` forced to true `399` horizon:
   - `std_rmse = 0.1459`
   - smoother late-horizon behavior than the 90-step seed:
     - step `90`: `0.1372`
     - step `181`: `0.1304`
     - step `257`: `0.1364`
     - step `361`: `0.1935`
     - step `399`: `0.2282`
4. `epoch=11` from the same full399 family is worse than `epoch=15`:
   - `epoch 11 std_rmse = 0.1550`
   - `epoch 15 std_rmse = 0.1459`
   - tail at step `399`: `0.2948` vs `0.2282`

- Scientific conclusions:
1. The corrected full399 family is stronger than the corrected 90-step family, even when the 90-step family uses stateful chunk carry.
2. Stateful rollout is still a real improvement for short-horizon checkpoints; reset-per-chunk inference is mathematically discontinuous at chunk boundaries and should not be trusted for long rollouts.
3. The remaining weakness in the best corrected branch is late-horizon drift after roughly step `360`, not early-horizon fit.

### H) Next controlled experiment launched
- New continuation run started from the best corrected full399 checkpoint:
1. Experiment: `vgssm_m2_correctedsync_full399_tail_exp_from_epoch15`
2. Init checkpoint: `checkpoints/model_2/vgssm_m2_correctedsync_full399_from_ctrl_v2/epoch=15-val_std_rmse=0.0999.ckpt`
3. Changes vs baseline:
   - fixed full horizon `399`
   - no curriculum
   - lower LR `2e-5`
   - `horizon_weighting=exp`
4. Rationale:
   - only change the optimization pressure toward later timesteps where the current corrected full399 branch still drifts

### I) Full399 continuation attempts after `0.0748`
- Goal:
1. improve the corrected full399 branch specifically at late horizons where holdout drift remains visible
- Attempts from `epoch=15-val_std_rmse=0.0999.ckpt`:
1. `vgssm_m2_correctedsync_full399_tail_exp_from_epoch15`
   - change: `horizon_weighting=exp`, fixed horizon `399`, FP32
   - result: immediate CUDA OOM at epoch `0`, step `5`
2. `vgssm_m2_correctedsync_full399_tail_exp_allocfix_from_epoch15`
   - same as above + `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`
   - result: still CUDA OOM at epoch `0`, step `5`
3. `vgssm_m2_correctedsync_full399_tail_exp_fp16_from_epoch15`
   - same as above + AMP / FP16
   - result: memory fit succeeded, but training loss was `NaN` from the first batches
- Scientific implication:
1. On the GTX 1080 Ti, full399 continuation from this checkpoint is hardware-limited in FP32.
2. AMP solves memory fit but is numerically unstable for this continuation path.
3. So immediate further gain should come from inference-side correction or different hardware, not brute-force retraining on this GPU.

### J) Late-horizon bias diagnosis on best corrected full399 checkpoint
- Checkpoint: `vgssm_m2_correctedsync_full399_from_ctrl_v2/epoch=15-val_std_rmse=0.0999.ckpt`
- Competition-aligned holdout (`399` steps) shows systematic negative bias that grows with horizon:
1. Mean bias 1D (normalized): `-0.0606`
2. Mean bias 2D (normalized): `-0.0350`
3. 1D per-step bias examples:
   - step `1`: `-0.0122`
   - step `180`: `-0.0738`
   - step `300`: `-0.1198`
   - step `399`: `-0.1646`
4. 2D per-step bias examples:
   - step `1`: `+0.0197`
   - step `180`: `-0.0420`
   - step `300`: `-0.0675`
   - step `399`: `-0.0830`
- Interpretation:
1. The best corrected full399 checkpoint is not exploding randomly.
2. It is systematically underpredicting water level at later horizons.
3. That makes a small horizon-dependent additive correction scientifically defensible.

### K) Conservative inference calibration and submission
- Fitted low-capacity linear bias ramps from holdout mean bias (normalized units):
1. 1D: `bias_norm(x) ≈ -0.135705 * x - 0.004942`
2. 2D: `bias_norm(x) ≈ -0.092670 * x - 0.000395`
   - where `x = step_idx / 398`
- Chosen conservative shrink factor:
1. `alpha = 0.75`
- Converted to physical additive correction for inference:
1. 1D add: `0.01173 + 0.32201 * x`
2. 2D add: `0.00080 + 0.18881 * x`
- Rationale:
1. keeps correction small and monotonic
2. targets only the measured late-horizon underprediction
3. avoids high-capacity per-event or per-node overfitting
- Submitted file:
1. `submissions/submission_20260314_m1v2e07_base_m2_correctedsync_full399_epoch15_calib_lin075.parquet`
2. Kaggle status at launch: `PENDING`
