# UrbanFloodBench Final Wrap-Up (2026-03-15)

## Scope
- Repository wrap-up for the UrbanFloodBench competition work up to `2026-03-15`.
- Focus is the corrected-dataset rerelease phase, especially Model 2.
- This document is the final high-level summary. Detailed chronology remains in:
  - `docs/rerelease_root_cause_log_2026-02-23.md`
  - `docs/experiments.md`
  - `docs/modal_training.md`

## Final Outcome
- Best public score achieved in this phase: `0.0701`
- Best scored submission:
  - `submission_20260314_m1v2e07_base_m2_correctedsync_full399_epoch15_calib_poly3.parquet`
- Improvement path on corrected data:
  1. Raw corrected full399 submission: `0.0748`
  2. Linear calibration `lin075`: `0.0705`
  3. Linear calibration `lin100`: `0.0702`
  4. Polynomial calibration `poly3`: `0.0701`
- User-reported leaderboard leader as of `2026-03-15`: `0.0124`
- Bottom line:
  - corrected-data recovery was real and substantial
  - it was not enough to contend for first place

## Key Scientific Findings

### 1) Remote stale dataset was a major real bug
- The competition rerelease fixed 1D dynamic node mapping.
- The local corrected files and remote training server files did not match.
- Evidence:
  - local affected-file manifest SHA-256:
    - `8f2db3024489dbdf4238dc4b76e243da5fc1bb1c7daf271850303ad54df3bd3f`
  - remote stale manifest before sync:
    - `09bde00006dca13d4e972ce0a1e25d18c9d6b63abd8f34b52b939a047ffbc1d9`
- Consequence:
  - multiple remote training runs and submissions before the sync were not scientifically valid for the corrected dataset

### 2) Model 2 was the true bottleneck
- Model 1 stayed relatively stable and was reused from a trusted base.
- Public score swings were driven mainly by Model 2 changes.
- Failed March 11 replacements diverged much more on Model 2 1D than on 2D.
- After syncing corrected 1D dynamic files, Model 2 performance improved materially.

### 3) Long-horizon generalization remained the hard problem
- Corrected-data control run at horizon `90` was stable:
  - best validation around `0.0915`
- Full rollout `399` continuation from that seed was stable but weaker in validation:
  - best validation around `0.0999`
- Even so, the full-rollout checkpoint scored better on Kaggle than the shorter-horizon control family.
- Interpretation:
  - validation did not fully capture the public test behavior
  - competition performance depended on long-horizon behavior more than short-horizon validation quality alone

### 4) Remaining corrected-data error was systematic late-horizon underprediction
- Holdout diagnostics on the best corrected full399 checkpoint showed monotonic negative bias with horizon.
- Representative normalized bias examples:
  - 1D:
    - step `1`: `-0.0122`
    - step `180`: `-0.0738`
    - step `300`: `-0.1198`
    - step `399`: `-0.1646`
  - 2D:
    - step `1`: `+0.0197`
    - step `180`: `-0.0420`
    - step `300`: `-0.0675`
    - step `399`: `-0.0830`
- This justified low-capacity horizon-dependent additive corrections at inference time.

### 5) Inference calibration worked, but only marginally
- Low-capacity horizon calibrations improved score from `0.0748` to `0.0701`.
- Prefix-only event-latent calibration did not generalize on full holdout.
- Conclusion:
  - post-hoc correction can recover small systematic drift
  - it is not enough to bridge the gap from `0.0701` to `0.0124`

### 6) Hardware constraints blocked the next real training step on the GTX 1080 Ti
- Full399 tail-weighted continuation from the corrected full399 checkpoint:
  - FP32: OOM
  - FP32 with allocator tuning: still OOM
  - FP16 mixed: memory fit succeeded, but loss went NaN almost immediately
- Consequence:
  - the next real scientific step required larger VRAM hardware
  - this motivated the shift to Modal

## What Worked
1. Correcting the remote dataset mismatch.
2. Re-establishing a stable corrected-data Model 2 baseline at horizon `90`.
3. Continuing that branch to full rollout `399`.
4. Diagnosing late-horizon bias on holdout rather than guessing.
5. Conservative inference-time calibration based on measured holdout drift.
6. Preparing a valid high-VRAM Modal continuation path with corrected data and warm-start checkpoint.

## What Did Not Work
1. Any result obtained from the stale remote dataset.
2. Short-horizon control branches as final submission candidates.
3. Early full399 continuation attempts before the dataset sync.
4. Prefix-only event-latent calibration on full holdout.
5. FP16 continuation on the corrected full399 tail run.
6. Blindly treating lower validation RMSE as sufficient for Kaggle ranking.

## Important Code and Pipeline Changes

### Training and evaluation
- `train_vgssm_standalone.py`
  - fixed reconstruction loss normalization
  - added metric-aligned 1D/2D balancing
  - added horizon-valid-count weighting
  - fixed curriculum-aware early stopping so it cannot stop before full rollout stages complete
  - added latent override support for scientific rollout evaluation
- `evaluate_vgssm_checkpoint_holdout.py`
  - added competition-aligned holdout rollout evaluation
  - added prefix-only latent calibration evaluation

### Modal execution
- `modal_vgssm.py`
  - hardened dataset-root detection
  - supports corrected-data VGSSM continuation on high-VRAM Modal GPUs
- `scripts/sync_corrected_1d_dynamic_to_modal.sh`
  - syncs all `196` affected rerelease files into the live Modal dataset root
- `scripts/run_modal_vgssm_m2_correctedsync_tail_exp_from_epoch15.sh`
  - launches the corrected-data tail-weighted continuation from the best local corrected full399 checkpoint

### Stable local/remote launchers
- `scripts/run_vgssm_m2_correctedsync_ctrl_v2_from_vgssm_v1_bs1.sh`
- `scripts/run_vgssm_m2_correctedsync_full399_from_ctrl_v2.sh`

## Submission Lineage Worth Keeping
1. Base trusted submission:
   - `submission_vgssm_rerelease_m1v2e07_on_m2fullrolloutv1_base_20260221.parquet`
2. Best raw corrected-data M2 replacement:
   - `submission_20260314_m1v2e07_base_m2_correctedsync_full399_epoch15.parquet`
   - public score `0.0748`
3. Best calibrated corrected-data submission:
   - `submission_20260314_m1v2e07_base_m2_correctedsync_full399_epoch15_calib_poly3.parquet`
   - public score `0.0701`

## Modal End State At Wrap-Up
- Active Modal app during the final push:
  - `ap-pjC03EdmZl9Rq2NTbo8CrW`
- Purpose:
  - corrected-data Model 2 continuation on larger VRAM
- State at wrap-up:
  - confirmed corrected-data sync to Modal volume: `196/196`
  - confirmed warm start from corrected full399 checkpoint
  - training had started cleanly
  - app was explicitly stopped during repository wrap-up to avoid additional cost

## Repository Release Notes
- This public repository intentionally excludes:
  - competition dataset
  - checkpoints
  - predictions and submissions
  - runtime logs
  - local operational notes in `CLAUDE.md`
- What remains:
  - model code
  - training and evaluation code
  - launch scripts
  - documentation of findings and experimental trajectory

## Recommended Next Work After Deadline
1. Move corrected full399 continuation entirely onto high-VRAM hardware and judge by long-horizon holdout plus Kaggle score.
2. Revisit Model 2 objective design specifically for late-horizon drift, rather than adding more post-hoc calibration.
3. Build a cleaner competition-aligned validation harness that correlates better with public LB for Model 2.
4. Separate public-score optimization from scientific debugging:
   - inference correction for small recoverable drift
   - architecture/loss redesign for real ranking jumps

## Final Assessment
- This project recovered from a serious but hidden data mismatch bug.
- The main technical win was re-establishing a corrected-data Model 2 path and pushing it from `0.2260`-range failure back to `0.0701`.
- The main technical limitation was not code correctness at the end; it was the remaining modeling gap on corrected Model 2 long-horizon behavior.
