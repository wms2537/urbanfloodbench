# Urban Flood Modelling - Experiment Log

Historical log. For the corrected-dataset final status and repository wrap-up, start with `docs/final_wrapup_2026-03-15.md`.

## Competition
- **Name**: Urban Flood Modelling
- **Platform**: Kaggle
- **URL**: https://www.kaggle.com/competitions/urban-flood-modelling
- **Objective**: Predict water levels in urban flood scenarios using coupled 1D-2D hydraulic models

---

## Dataset Overview

### Data Structure
- **Total Size**: ~18GB
- **Models**: 2 separate hydraulic models with different network configurations

### Model 1 (Smaller Network)
| Component | Count |
|-----------|-------|
| 1D Nodes (Sewer) | 17 |
| 2D Cells (Surface) | 3,716 |
| Coupling Connections | 16 |
| Training Events | 130 |
| Test Events | 29 |

### Model 2 (Larger Network)
| Component | Count |
|-----------|-------|
| 1D Nodes (Sewer) | 198 |
| 2D Cells (Surface) | 4,299 |
| Coupling Connections | 197 |
| Training Events | 130 |
| Test Events | 29 |

### Feature Dimensions
- **Static 1D Features**: 6 (invert_level, max_depth, length, roughness, geometry_type, system_type)
- **Static 2D Features**: 9 (elevation, min_elevation, bed_resistance, area, cell_type, and derived features)
- **Dynamic 1D Features**: 2 (water_level, discharge)
- **Dynamic 2D Features**: 3 (water_level, total_rainfall, infiltration)

---

## Architecture: Coupled Latent Digital Twin Surrogate (CL-DTS)

### Design Philosophy
Treat urban flooding as a **partial-observation digital twin problem** where:
- 1D sewer network and 2D surface mesh are coupled subsystems
- Latent states capture unobserved physics
- Event-specific calibration handles varying rainfall patterns

### Model Components

#### 1. Heterogeneous Graph Neural Network
- **Node Types**: `node_1d` (sewer), `node_2d` (surface)
- **Edge Types**:
  - `1d_to_1d`: Sewer network connectivity
  - `2d_to_2d`: Surface mesh adjacency
  - `1d_to_2d` / `2d_to_1d`: Coupling connections (manholes, inlets)
- **Convolution**: SAGEConv with heterogeneous message passing
- **Layers**: 3 GNN layers with residual connections

#### 2. Temporal Encoder
- **Architecture**: Bidirectional GRU
- **Layers**: 2 temporal layers
- **Purpose**: Encode time series context for latent inference

#### 3. Event Latent Encoder (c_e)
- **Type**: Variational encoder (VAE-style)
- **Latent Dimension**: 16
- **Purpose**: Capture event-specific characteristics (rainfall intensity, antecedent conditions)
- **Training**: ELBO with KL annealing

#### 4. Prediction Heads
- Separate MLPs for 1D and 2D water level prediction
- Output: Next timestep water levels

### Hyperparameters
```yaml
# Model Architecture
hidden_dim: 64
latent_dim: 32
event_latent_dim: 16
num_gnn_layers: 3
num_temporal_layers: 2
use_attention: true
use_event_latent: true
use_dynamic_latent: false

# Sequence Configuration
seq_len: 16
prefix_len: 8
pred_len: 1
rollout_steps: 8

# Training
learning_rate: 0.001
weight_decay: 1e-5
dropout: 0.1
beta: 0.1  # KL weight
```

---

## Experiment 1: Baseline CL-DTS

### Configuration
- **Date**: January 2025
- **Hardware**: Apple Silicon (MPS acceleration)
- **Epochs**: 20 per model
- **Batch Size**: 16
- **Optimizer**: AdamW

### Training Progress

#### Model 1
| Epoch | Train Loss | Val Loss | Val RMSE |
|-------|------------|----------|----------|
| 0 | 0.0217 | 0.0175 | 0.0135 |
| 1 | 0.0168 | 0.0173 | **0.0133** |
| 2 | 0.0161 | 0.0176 | 0.0135 |
| ... | ... | ... | ... |

**Best Checkpoint**: Epoch 1, Val RMSE = 0.0133

#### Model 2
| Epoch | Train Loss | Val Loss | Val RMSE |
|-------|------------|----------|----------|
| 0 | 0.0180 | 0.0187 | **0.0138** |
| 1 | 0.0172 | 0.0190 | 0.0140 |
| ... | ... | ... | ... |

**Best Checkpoint**: Epoch 0, Val RMSE = 0.0138

### Prediction Statistics
| Model | Rows | Water Level Range | Mean |
|-------|------|-------------------|------|
| Model 1 | 18.6M | [0, 368] | 322 |
| Model 2 | 32.3M | [22, 56] | 43 |
| **Combined** | **50.9M** | - | - |

### Submission
- **File**: `submission_combined.parquet`
- **Size**: 450 MB
- **NaN Values**: 0

---

## Technical Challenges & Solutions

### 1. NaN in Static Features
**Problem**: 12 NaN values in `min_elevation` column for 2D nodes.

**Solution**: Fill NaN with corresponding `elevation` value.
```python
df_2d['min_elevation'] = df_2d['min_elevation'].fillna(df_2d['elevation'])
```

### 2. NaN in Test Data Dynamic Features
**Problem**: Test data only has water_level for first 10 timesteps (warmup), rest is NaN (target to predict).

**Solution**: Forward-fill along time axis in dataset preprocessing.
```python
# Forward fill along time axis for each node
for n in range(data.shape[1]):
    node_vals = data[:, n, f]
    valid_idx = np.where(~np.isnan(node_vals))[0]
    if len(valid_idx) > 0:
        last_valid_idx = valid_idx[-1]
        if last_valid_idx < len(node_vals) - 1:
            data[last_valid_idx + 1:, n, f] = node_vals[last_valid_idx]
```

### 3. Parameter Name Mismatch
**Problem**: PyTorch Geometric's HeteroConv expects `hidden_channels`, not `hidden_dim`.

**Solution**: Updated parameter passing in model initialization.

### 4. Target Shape Mismatch
**Problem**: Target tensor had extra dimension causing broadcast errors.

**Solution**: Added `.squeeze(-1)` to align shapes.

### 5. Protobuf Corruption
**Problem**: Corrupted protobuf egg file causing import errors.

**Solution**: Removed from easy-install.pth.

---

## Key Findings

### 1. Early Convergence
Both models achieved best validation RMSE very early (epoch 0-1), suggesting:
- Model capacity is appropriate for the task
- Potential for overfitting with longer training
- Consider early stopping with patience=3-5

### 2. Different Scale Ranges
- Model 1 water levels: 0-368 (larger range, possibly deeper sewer system)
- Model 2 water levels: 22-56 (smaller range, shallower system)

### 3. Event Latent Effectiveness
Using `use_event_latent=true` allows the model to adapt to different rainfall events without explicit event features.

### 4. Coupling is Critical
The heterogeneous graph structure with explicit 1D-2D coupling edges is essential for capturing the hydraulic interactions between sewer overflow and surface flooding.

---

## Critical Bug Found (January 2025)

**Issue**: During autoregressive prediction, we keep dynamic features (rainfall, infiltration) **fixed** from the last warmup timestep. But test data provides **full rainfall sequences**!

**Impact**: Without correct rainfall input, predictions are essentially guessing. This likely explains the large gap between our score (0.43) and top solutions (0.085).

**Fix Required in `predict.py`**:
```python
# WRONG: Using stale rainfall from warmup
next_input_2d = curr_input_2d.clone()
next_input_2d[:, :, :, 1] = pred_2d.squeeze(-1)  # Only updating water_level

# CORRECT: Use actual rainfall from test data
next_input_2d[:, :, :, 0] = actual_rainfall[:, t]  # Use real rainfall
next_input_2d[:, :, :, 1] = pred_2d.squeeze(-1)    # Predicted water_level
next_input_2d[:, :, :, 2] = actual_infiltration[:, t]  # Use real infiltration if available
```

---

## Future Improvements

### PRIORITY: Fix Prediction Code
- Use actual rainfall values during autoregressive rollout
- Use actual infiltration values if available in test data
- For 1D nodes: use actual inlet_flow if available

### Phase C: Dynamic Latent State (z_t)
- Add per-timestep latent state for full Variational State Space Model
- Enable filtering/smoothing for better temporal consistency

### Phase D: Test-Time Optimization
- Optimize event latent c_e during prediction using first 10 timesteps
- Could improve adaptation to unseen rainfall patterns

### Phase E: Advanced Techniques
- Ensemble multiple checkpoints
- Multi-scale GNN with different receptive fields
- Physics-informed loss (conservation laws)
- Attention-based temporal modeling (Transformer)

### Hyperparameter Tuning
- Grid search over hidden_dim: [32, 64, 128]
- Learning rate scheduling
- Longer training with early stopping
- Different GNN architectures (GAT, GIN)

---

## Competition Metric Update (January 2025)

**Important**: The competition changed from node-wise NSE to **standardized RMSE with equal weighting for 1D and 2D nodes**.

### New Metric Formula
```
Standardized RMSE = RMSE / std_dev

Final Score = (std_rmse_1d + std_rmse_2d) / 2
```

### Standard Deviation Values (computed from training data)
| Model | Node Type | Std Dev |
|-------|-----------|---------|
| 1 | 1D | 16.877747 |
| 1 | 2D | 14.378797 |
| 2 | 1D | 3.191784 |
| 2 | 2D | 2.727131 |

### Implications
- Equal weighting means 1D errors are now as important as 2D (despite 200x fewer nodes)
- Standardization by std ensures fair comparison across different water level scales
- Our loss function and checkpointing now optimized for `val/std_rmse`

---

## Official Autoregressive Guidance (from Kaggle Discussion)

**Key insight from competition host (Jia Yu Lim):**

### Data Availability During Prediction:
- **t = 1 to 10**: Spin-up period - ground-truth water levels and all dynamic variables available
- **t = 11 to 100**: Prediction period - must predict autoregressively
- **Rainfall is available for ALL timesteps (t=1 to 100)** - this is critical!
- Other dynamic variables (volume, flow, velocity, inlet_flow) are **NOT available** after t=10

### Correct Input Setup for Predicting at timestep t:
1. Water levels: t=1-10 (ground truth) + t=11 to t-1 (model predictions)
2. **Rainfall: t=1 to t (actual values from test data)** ← KEY INSIGHT
3. Other dynamics: t=1-10 (ground truth) + use predictions or keep from warmup

### Leaderboard Reference (as of Jan 2025):
| Rank | Team | Score |
|------|------|-------|
| 1 | Timothee Henry | 0.0858 |
| 2 | Gegerout | 0.0895 |
| 3 | Matt Motoki | 0.1636 |
| 4 | The Laplacian | 0.2152 |

---

## Submission History

| Date | Model | Public Score | Notes |
|------|-------|--------------|-------|
| 2025-01-18 | CL-DTS v1 | **0.4232** | **Best** - kept features from warmup (stale rainfall) |
| 2025-01-18 | CL-DTS v2 | 0.9684 | Buggy - introduced NaN inlet_flow |
| 2025-01-18 | CL-DTS v3 | 1.5016 | Buggy - introduced NaN water_volume |
| 2025-01-18 | CL-DTS v4 | 0.6722 | Used actual rainfall - worse than v1! |

### Key Finding: Using Actual Rainfall Hurts Performance!
- **v4 used actual rainfall** from test data (as recommended by competition host)
- **v4 scored 0.6722** - worse than v1's 0.4232
- **Root cause**: Model was trained with stale features propagated from previous timestep
- **Training/inference mismatch**: Changing input pattern at inference causes distribution shift

### Implications:
1. The model learned to rely on stale rainfall patterns during training
2. To properly use actual rainfall, the model needs to be **retrained** with:
   - Teacher forcing using actual rainfall during training
   - Or curriculum learning to adapt to rainfall changes
3. For now, **v1 approach (stale features) gives best results**

### Gap Analysis (vs Top Teams):
| Team | Score | Gap from v1 |
|------|-------|-------------|
| 1st (Timothee Henry) | 0.0858 | 5x better |
| 2nd (Gegerout) | 0.0895 | 4.7x better |
| Our best (v1) | 0.4232 | baseline |

**To close the gap, likely need:**
- Different model architecture (not just prediction code fixes)
- Better temporal encoding of rainfall sequence
- Ensemble approaches
- Test-time optimization of event latent

### Submission v1 Details
- **Competition**: UrbanFloodBench: Flood Modelling
- **Deadline**: 2026-03-01
- **Teams**: 45
- **Prize Pool**: $7,000 USD
- **File Submitted**: `submission_combined.parquet` (429 MB)

---

## Experiment 2: Multi-Step Rollout Training (v3)

### Root Cause Analysis: Why Overfitting After 1 Epoch?

The model achieved best validation RMSE at epoch 0-1, then performance degraded. Analysis revealed:

1. **Training uses 1-step prediction**: Model only learns to predict next timestep
2. **1-step prediction is TOO EASY**: Due to time series autocorrelation, water_level(t+1) ≈ water_level(t)
3. **Model learns shortcut**: Just copy last timestep → low training/validation loss
4. **But 90-step rollout fails**: Errors compound exponentially over long horizons
5. **Training/inference mismatch**: Model never learned to use rainfall properly

### Solution: Multi-Step Rollout Training

Key changes to `src/training/trainer.py`:

1. **Multi-step rollout in training**: Train on 8 consecutive steps, not just 1
   - Forces model to predict further into future
   - Errors accumulate during training, teaching robustness

2. **Teacher forcing with ACTUAL rainfall**:
   - Always use real rainfall from data (it's an external forcing)
   - This matches inference where rainfall IS available for all timesteps

3. **Scheduled sampling for water_level**:
   - Start with 100% teacher forcing (use ground truth)
   - Linearly decay to 0% over 30 epochs
   - Model gradually learns to use its own predictions

4. **Validation without teacher forcing**:
   - Evaluate with NO teacher forcing (matches inference)
   - Properly measures autoregressive performance

### Training Configuration Changes

```yaml
# Regularization (increased to prevent overfitting)
dropout: 0.2  # was 0.1
weight_decay: 1e-4  # was 1e-5

# Multi-step rollout
rollout_steps: 8  # predict 8 steps during training

# Scheduled sampling
initial_teacher_forcing: 1.0
min_teacher_forcing: 0.0
teacher_forcing_decay_epochs: 30
```

### Feature Indices (Important!)
- **1D features** `[water_level, inlet_flow]`: index 0 = water_level
- **2D features** `[rainfall, water_level, water_volume]`: index 0 = rainfall, index 1 = water_level

### Expected Improvements
- Training loss will be HIGHER initially (multi-step is harder)
- But model will learn to use rainfall properly
- Better generalization to 90-step inference rollout
- Validation metric will be more predictive of test performance

---

## Code Structure

```
urbanfloodbench/
├── src/
│   ├── data/
│   │   ├── graph_builder.py   # Heterogeneous graph construction
│   │   └── dataset.py         # FloodEventDataset, FloodDataModule
│   ├── models/
│   │   ├── coupled_gnn.py     # CoupledHeteroGNN
│   │   ├── temporal.py        # TemporalEncoder (GRU)
│   │   └── cldts.py           # CLDTSModel (main)
│   └── training/
│       ├── losses.py          # ELBO, physics regularizers
│       └── trainer.py         # CLDTSLightningModule
├── train.py                   # Training script
├── predict.py                 # Prediction & submission generation
├── checkpoints/               # Saved model weights
├── logs/                      # TensorBoard logs
└── docs/
    └── experiments.md         # This file
```

---

## Experiment 3: v8_bugfix - Critical Bug Fixes (January 2025)

### Investigation Summary

After scoring 0.6648 with v7, a thorough code review identified **3 critical bugs**:

### Bug 1: Double Encoder Pass

**Files Affected**: `predict.py`, `src/training/trainer.py`

**Problem**: During autoregressive rollout, the temporal encoder was called TWICE per timestep:
```python
# BUGGY CODE - encoder called twice!
# First call (getting current encoding)
curr_z_1d, h_1d = model.model.st_encoder_1d(spatial_1d, next_input_1d, h_1d)
curr_z_2d, h_2d = model.model.st_encoder_2d(spatial_2d, next_input_2d, h_2d)

# Second call (updating hidden state) - THIS CORRUPTS HIDDEN STATE!
_, h_1d = model.model.st_encoder_1d(spatial_1d, next_input_1d, h_1d)
_, h_2d = model.model.st_encoder_2d(spatial_2d, next_input_2d, h_2d)
```

**Impact**: Hidden states were processed twice with the same input, corrupting temporal memory.

**Fix**: Single encoder pass:
```python
# FIXED - single encoder pass
curr_z_1d, h_1d = model.model.st_encoder_1d(spatial_1d, next_input_1d, h_1d)
curr_z_2d, h_2d = model.model.st_encoder_2d(spatial_2d, next_input_2d, h_2d)
curr_z_1d = curr_z_1d[:, -1:]
curr_z_2d = curr_z_2d[:, -1:]
```

### Bug 2: Normalization Using Only 10 Training Events

**File Affected**: `src/data/dataset.py` (line 394)

**Problem**: Normalization statistics were computed from only the first 10 training events:
```python
# BUGGY CODE
for event_id in event_ids[:10]:  # Only first 10 events!
```

**Impact**: Normalization stats were not representative of the full training distribution.

**Fix**: Use ALL training events:
```python
# FIXED
for event_id in event_ids:  # Use ALL training events
```

### Bug 3: Normalization Mismatch Between Training and Inference

**File Affected**: `predict.py`

**Problem**: During inference, `predict.py` was computing normalization stats from each TEST event separately:
```python
# BUGGY CODE
event_dataset = FloodEventDataset(
    args.data_dir, model_id, event_id, "test",
    graph, seq_len=16, pred_len=1, stride=1,
    normalize=True  # This computed stats from TEST event!
)
```

**Impact**: Model received inputs normalized differently than during training = distribution shift.

**Fix**: Added `compute_global_norm_stats()` function to compute normalization from TRAINING data and pass it to test datasets:
```python
# FIXED - compute stats from training data
global_norm_stats = compute_global_norm_stats(args.data_dir, model_id, graph)

event_dataset = FloodEventDataset(
    args.data_dir, model_id, event_id, "test",
    graph, seq_len=16, pred_len=1, stride=1,
    normalize=True, normalization_stats=global_norm_stats  # Use training stats!
)
```

### Training Configuration

```bash
python train.py \
    --model_id X \
    --exp_name v8_bugfix \
    --max_epochs 15 \
    --batch_size 4 \
    --seq_len 24 \
    --prefix_len 10 \
    --rollout_steps 8 \
    --hidden_dim 64 \
    --num_gnn_layers 3 \
    --dropout 0.2 \
    --lr 1e-3 \
    --stride 4 \
    --patience 7 \
    --accelerator mps \
    --num_workers 0
```

### Training Results

| Model | Best Checkpoint | val/std_rmse | val/std_rmse_1d | val/std_rmse_2d |
|-------|-----------------|--------------|-----------------|-----------------|
| Model 1 | epoch=09-val/std_rmse=0.0007.ckpt | 0.00115 | 0.00148 | 0.00081 |
| Model 2 | epoch=09-val/std_rmse=0.0177.ckpt | 0.04161 | 0.07452 | 0.00870 |

### Prediction Statistics

```
Shape: (50910192, 6)
water_level stats:
  count    5.091019e+07
  mean     1.447018e+02
  std      1.348827e+02
  min      2.320784e+01
  25%      4.189041e+01
  50%      4.490445e+01
  75%      3.166895e+02
  max      3.607200e+02
```

### Kaggle Submission Result

| Date | Version | Public Score | Notes |
|------|---------|--------------|-------|
| 2025-01-21 | v8_bugfix | **0.6972** | WORSE than v7 (0.6648) |

### Analysis: Why Did "Bug Fixes" Make It Worse?

**Hypothesis 1: Training/Inference Consistency**
- The "bugs" may have been consistent between training and inference
- By fixing inference without retraining with the same fixes, we created a mismatch
- The model learned to work with the "buggy" behavior

**Hypothesis 2: Normalization Change Impact**
- Using all training events vs 10 events changes the normalization statistics
- Model 1: target_1d mean=308.09, std=16.88
- Model 2: target_1d mean=39.89, std=3.19
- These stats may differ from what the model learned during v7 training

**Hypothesis 3: Double Encoder Was Not Actually a Bug**
- The "double encoder pass" may have been intentional or beneficial
- First pass: encode current state
- Second pass: update hidden state for next step
- Removing this changed the model's temporal dynamics

### Key Lesson Learned

**Training and inference code MUST be consistent.** Fixing "bugs" in inference code without retraining can break the learned patterns. The model adapts to whatever behavior exists during training.

---

## Updated Submission History

| Date | Model | Public Score | Notes |
|------|-------|--------------|-------|
| 2025-01-18 | CL-DTS v1 | **0.4232** | **Best** - kept features from warmup |
| 2025-01-18 | CL-DTS v2 | 0.9684 | Buggy - introduced NaN inlet_flow |
| 2025-01-18 | CL-DTS v3 | 1.5016 | Buggy - introduced NaN water_volume |
| 2025-01-18 | CL-DTS v4 | 0.6722 | Used actual rainfall - worse! |
| 2025-01-XX | CL-DTS v7 | 0.6648 | Unknown changes |
| 2025-01-21 | CL-DTS v8_bugfix | 0.6972 | Bug fixes made it worse |

### Score Trend Analysis

```
Best:  v1 (0.4232) - baseline approach
       ↓
       v4 (0.6722) - attempted rainfall fix
       ↓
       v7 (0.6648) - slight improvement
       ↓
Worst: v8 (0.6972) - "bug fixes" hurt performance
```

**Conclusion**: v1 remains the best approach. All subsequent "improvements" have degraded performance.

---

## Recommendations for Future Work

### Option A: Revert to v1 Architecture
- Use the original v1 code as baseline
- Make small, incremental changes
- Test each change individually

### Option B: Full System Retrain with Bug Fixes
- Apply all bug fixes to training code
- Retrain from scratch
- Ensure training and inference are consistent

### Option C: Architecture Investigation
- Study what v1 was doing differently
- The "bugs" may have been accidental features
- Understand why stale features work better than actual rainfall

### Option D: Ensemble Approaches
- Combine v1 predictions with other versions
- May capture different aspects of the problem

---

## Experiment 4: Graph-TFT Architecture (January 2025)

### Motivation: Why a Fundamental Architecture Change?

All previous attempts (v1-v8) used the same core architecture:
- **GRU-based temporal encoder**: Sequential, autoregressive prediction
- **1-step prediction at training**: Model learns to predict only next timestep
- **90-step autoregressive rollout at inference**: Errors compound exponentially

The fundamental problem is **training/inference mismatch**:
- Training: Predict 1 step with teacher forcing
- Inference: Predict 90 steps autoregressively
- Model never learned to recover from errors

### Solution: Graph-TFT with Multi-Horizon Prediction

Inspired by Temporal Fusion Transformer (TFT) architecture:

**Key Design Changes:**

1. **Multi-Horizon Prediction** (predict ALL K steps at once)
   - Training: Predict timesteps 11-100 simultaneously
   - Inference: Same - predict timesteps 11-100 simultaneously
   - No autoregressive rollout = no error accumulation

2. **TFT Temporal Encoder** (replaces GRU)
   - Gated Residual Networks (GRN) for nonlinear processing
   - Multi-head self-attention for long-range temporal patterns
   - LSTM for local patterns (combined with attention)
   - Variable selection for feature importance

3. **Known Future Inputs** (rainfall)
   - TFT natively handles "known future" covariates
   - Rainfall is available for t=1-100 in test data
   - Model can directly condition on future rainfall

4. **Event Latent Calibration** (test-time optimization)
   - Optimize event latent c_e using warmup period (t=1-10)
   - Calibrates hidden parameters (roughness, blockages) per event
   - "Digital twin tuning" without weight updates

### Architecture Overview

```
Input Data (1D & 2D sequences, t=1-10)
    ↓
Spatial Encoding (HeteroGNN - unchanged)
    ↓
Event Latent Encoding (c_e from prefix)
    ↓
TFT Temporal Encoding (LSTM + Self-Attention)
    ↓
Multi-Horizon Decoder (90 parallel output heads)
    ↓
Predictions for t=11-100 (all at once)
```

### New Files Created

- `src/models/tft.py`: TFT components
  - `GatedResidualNetwork`: Core building block
  - `VariableSelectionNetwork`: Feature importance
  - `InterpretableMultiHeadAttention`: Temporal attention
  - `TemporalFusionEncoder`: Main TFT encoder
  - `SpatioTemporalTFT`: TFT for node sequences
  - `MultiHorizonDecoder`: Parallel output heads

- `src/models/graph_tft.py`: Complete Graph-TFT model
  - `EventLatentEncoderTFT`: Attention-based event encoding
  - `GraphTFT`: Main model class
  - Combines HeteroGNN spatial + TFT temporal

- `src/training/graph_tft_trainer.py`: Training utilities
  - `GraphTFTTrainer`: PyTorch Lightning module
  - Horizon-weighted loss (linear/exp weighting)
  - KL annealing for event latent

- `train_graph_tft.py`: Training script
- `predict_graph_tft.py`: Prediction script

### Training Configuration

```bash
python train_graph_tft.py \
    --model_id 1 \
    --exp_name graph_tft_v1 \
    --hidden_dim 64 \
    --num_gnn_layers 3 \
    --num_tft_layers 2 \
    --num_heads 4 \
    --prediction_horizon 90 \
    --prefix_len 10 \
    --batch_size 4 \
    --max_epochs 30 \
    --lr 1e-3 \
    --horizon_weighting linear \
    --accelerator cuda
```

### Expected Improvements

1. **No error accumulation**: Direct multi-horizon prediction
2. **Better temporal modeling**: Attention captures long-range patterns
3. **Proper rainfall usage**: Known future handled natively by TFT
4. **Event calibration**: Test-time c_e optimization adapts to each storm

### Why This Should Work

- TFT was designed specifically for multi-horizon forecasting
- Successfully used in energy demand, weather, and financial forecasting
- Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (https://arxiv.org/abs/1912.09363)

### Training Results (January 22, 2025)

Trained on GPU server (NVIDIA GTX 1080 Ti, 11.7 GB VRAM)

| Model | Epochs | Best val/std_rmse | Training Time |
|-------|--------|-------------------|---------------|
| Model 1 | 30 | **0.00141** | ~75 min |
| Model 2 | ~9 (early stopped) | **0.0953** | ~30 min |

**Training Progress (Model 1):**
- Epoch 0: val_rmse = 0.00348
- Epoch 6: val_rmse = 0.00177 (new best)
- Epoch 28: val_rmse = **0.00141** (final best)
- train_loss decreased: 2.02 → 0.012

**Model 2 Note:** Higher validation error (0.0953) suggests Model 2's dynamics are harder to capture. Early stopped due to no improvement after epoch 8.

### Kaggle Submission Result

| Date | Version | Public Score | Notes |
|------|---------|--------------|-------|
| 2025-01-22 | Graph-TFT v1 | **0.2595** | Multi-horizon prediction, no autoregressive rollout |

### Analysis

**Improvement over baseline:**
- Previous best (CL-DTS v1): 0.4232
- Graph-TFT v1: 0.2595
- **Improvement: 38.7%** (0.4232 → 0.2595)

**Gap to top teams:**
- 1st place: 0.0858 (we need ~3x improvement)
- Our score: 0.2595

**What worked:**
1. Multi-horizon prediction eliminated autoregressive error accumulation
2. TFT temporal encoder captures long-range dependencies
3. Consistent training/inference (no mismatch)

**What needs investigation:**
1. Model 2 has much higher error (0.0953 vs 0.00141) - may need different architecture or hyperparameters
2. Test-time calibration was not enabled (could improve per-event adaptation)
3. Known future inputs (rainfall) were disabled due to dataset shape issues

### Issues Encountered & Fixed

1. **Dimension mismatch (known_future_dim)**: Dataset didn't provide future_rainfall in expected format. Fixed by setting `known_future_dim=0` in decoder.

2. **Target tensor shape**: Targets had extra dimension [B, H, N, 1] vs predictions [B, H, N]. Fixed by squeezing targets in trainer.

3. **Normalization stats**: Had to use FloodDataModule instead of non-existent `compute_global_norm_stats` function.

4. **Path issues**: Data directories use `Model_1` (capitalized) vs `model1` (lowercase).

5. **Warmup timesteps**: Predictions included warmup period (10 steps) that should be excluded from submission.

---

## Experiment 5: Next Steps for Graph-TFT

### Priority 1: Fix Model 2 Performance

Model 2's validation error (0.0953) is ~70x worse than Model 1 (0.00141). Possible causes:
- Larger network (198 vs 17 1D nodes)
- Different dynamics/scaling
- May need longer training or different hyperparameters

**Actions:**
- [ ] Train Model 2 with more epochs (50+)
- [ ] Try larger hidden_dim (128)
- [ ] Check if normalization stats are appropriate
- [ ] Analyze which nodes have highest errors

### Priority 2: Enable Test-Time Calibration

Current submission does NOT use test-time calibration. This is a potential quick win:

```python
# predict_graph_tft.py
python predict_graph_tft.py \
    --model_id 1 \
    --checkpoint checkpoints/model_1/graph_tft_v1/last.ckpt \
    --calibrate_latent \
    --calibration_steps 100
```

Expected improvement: 10-30% based on similar approaches in weather forecasting.

### Priority 3: Enable Known Future Inputs (Rainfall)

TFT was designed to handle known future covariates. Current implementation disabled this due to dimension issues:

```python
# Current (disabled):
self.decoder_2d = MultiHorizonDecoder(..., known_future_dim=0)
pred_2d = self.decoder_2d(z_2d, spatial_2d, c_e, None)

# Target (enabled):
self.decoder_2d = MultiHorizonDecoder(..., known_future_dim=1)  # rainfall
pred_2d = self.decoder_2d(z_2d, spatial_2d, c_e, future_rainfall)
```

**Actions:**
- [ ] Fix dataset to provide future_rainfall in correct shape
- [ ] Retrain with known_future_dim=1
- [ ] This should significantly help since rainfall IS available for all timesteps

### Priority 4: Architecture Improvements

1. **Deeper TFT**: num_tft_layers=4 (currently 2)
2. **More attention heads**: num_heads=8 (currently 4)
3. **Larger hidden_dim**: 128 (currently 64)
4. **Interpretable attention**: Analyze which timesteps matter most

### Priority 5: Ensemble

Combine multiple models:
- Different random seeds
- Different horizon weightings (linear, exp, uniform)
- With/without test-time calibration

---

## Updated Submission History

| Date | Model | Public Score | Notes |
|------|-------|--------------|-------|
| 2025-01-18 | CL-DTS v1 | 0.4232 | GRU baseline, autoregressive |
| 2025-01-18 | CL-DTS v2 | 0.9684 | Buggy |
| 2025-01-18 | CL-DTS v3 | 1.5016 | Buggy |
| 2025-01-18 | CL-DTS v4 | 0.6722 | Used actual rainfall |
| 2025-01-XX | CL-DTS v7 | 0.6648 | Unknown |
| 2025-01-21 | CL-DTS v8 | 0.6972 | Bug fixes |
| 2025-01-22 | **Graph-TFT v1** | **0.2595** | **Best** - Multi-horizon, no AR |

### Score Progress

```
1.5016 (v3) - worst
   ↓
0.9684 (v2)
   ↓
0.6972 (v8)
   ↓
0.6722 (v4)
   ↓
0.6648 (v7)
   ↓
0.4232 (v1) - previous best
   ↓
0.2595 (Graph-TFT) - NEW BEST! 🎉
   ↓
0.0858 (1st place) - target
```

**Key insight**: Fundamental architecture change (multi-horizon TFT) gave bigger improvement than all previous bug fixes and tweaks combined.

---

## Experiment 6: Graph-TFT v2 - Three Key Improvements (January 2025)

### Implementation Summary

Based on the TFT design principles discussed, implemented three key improvements:

#### 1. Test-Time Event Latent Calibration (c_e Optimization)

**The "killer move" for digital twin calibration:**
- Use warmup period (t=1-10) to optimize c_e for each test event
- Keep model weights frozen, only optimize the event latent
- This adapts the model to unseen rainfall patterns without retraining

**Code Changes:**
- `predict_graph_tft.py`: Moved calibration code outside `no_grad()` to enable gradients
- `src/models/graph_tft.py`: Added `c_e_override` parameter to forward() for using calibrated c_e

**Key insight**: During test-time, we have ground truth for first 10 timesteps. By optimizing c_e to minimize prediction error on warmup, we "teach" the model about this specific event's characteristics.

#### 2. Known Future Inputs (Rainfall)

**TFT's advantage: Proper handling of known future covariates:**
- Rainfall is available for ALL timesteps (t=1-100) per competition rules
- Previous model ignored this - `known_future_dim=0` in decoder
- Now properly passing rainfall as known future covariate

**Code Changes:**
- `src/models/graph_tft.py`: Changed `known_future_dim=0` to `known_future_dim=1` for 2D decoder
- `src/data/dataset.py`: Added `future_rainfall` to dataset __getitem__() output
- Model now uses rainfall at timestep t when predicting water level at t

**Key insight**: This is THE differentiating feature of TFT - it can use known future inputs properly, unlike pure autoregressive models.

#### 3. Model 2 Capacity Increase

**Investigation findings:**
- Model 1: 17 1D nodes, 3716 2D nodes
- Model 2: 198 1D nodes (~12x more), 4299 2D nodes
- Model 2 validation error was 70x worse (0.0953 vs 0.00141)

**Root cause**: Same hyperparameters for vastly different graph sizes

**Solution - Model-specific hyperparameters:**
```yaml
# Model 1 (default):
hidden_dim: 64
num_gnn_layers: 3
num_tft_layers: 2
dropout: 0.2

# Model 2 (larger):
hidden_dim: 96
num_gnn_layers: 4
num_tft_layers: 3
dropout: 0.15
```

**Code Changes:**
- `train_graph_tft.py`: Added `get_model_specific_config()` function
- `predict_graph_tft.py`: Added automatic config selection based on model_id

### Training Commands

```bash
# Model 1 - default settings
python train_graph_tft.py \
    --model_id 1 \
    --exp_name graph_tft_v2 \
    --max_epochs 30 \
    --batch_size 4 \
    --accelerator cuda

# Model 2 - larger capacity automatically applied
python train_graph_tft.py \
    --model_id 2 \
    --exp_name graph_tft_v2 \
    --max_epochs 30 \
    --batch_size 4 \
    --accelerator cuda
```

### Prediction Commands

```bash
# With test-time calibration enabled
python predict_graph_tft.py \
    --model_id 1 \
    --checkpoint checkpoints/model_1/graph_tft_v2/best.ckpt \
    --calibrate_latent \
    --calibration_steps 50

python predict_graph_tft.py \
    --model_id 2 \
    --checkpoint checkpoints/model_2/graph_tft_v2/best.ckpt \
    --calibrate_latent \
    --calibration_steps 50
```

### Expected Improvements

| Improvement | Expected Impact |
|-------------|-----------------|
| Test-time c_e calibration | Major - adapts to unseen events |
| Known future rainfall | Major - proper TFT input handling |
| Model 2 capacity | Moderate - better for larger graph |

### Status: Pending Retraining

Files modified:
- `src/models/graph_tft.py` - c_e_override, known_future_dim=1, future_rainfall pass-through
- `src/data/dataset.py` - future_rainfall in batch output
- `predict_graph_tft.py` - calibration outside no_grad, c_e_override usage
- `train_graph_tft.py` - model-specific hyperparameters

---

## Experiment 7: Graph-TFT v2 - Best Result (January 2025)

### Training Results

Trained with the v2 improvements (test-time calibration, known future rainfall, model-specific capacity).

| Model | Epochs | Best val/std_rmse | Notes |
|-------|--------|-------------------|-------|
| Model 1 | 30 | **0.00135** | Default hidden_dim=64 |
| Model 2 | 30 | **0.0812** | Auto-scaled hidden_dim=96 |

### Prediction Configuration

```bash
python predict_graph_tft.py \
    --model_id 1 \
    --checkpoint checkpoints/model_1/graph_tft_v2/best.ckpt \
    --calibrate_latent \
    --calibration_steps 100

python predict_graph_tft.py \
    --model_id 2 \
    --checkpoint checkpoints/model_2/graph_tft_v2/best.ckpt \
    --calibrate_latent \
    --calibration_steps 100
```

### Kaggle Submission Result

| Date | Version | Public Score | Notes |
|------|---------|--------------|-------|
| 2025-01-22 | Graph-TFT v2 | **0.2281** | **BEST RESULT** - Test-time calibration enabled |

### Analysis

**Improvement over v1:**
- Graph-TFT v1: 0.2595
- Graph-TFT v2: 0.2281
- **Improvement: 12.1%**

**Key factors:**
1. Test-time c_e calibration with 100 steps
2. Known future rainfall properly passed to decoder
3. Model 2 auto-scaled to hidden_dim=96

---

## Experiment 8: Graph-TFT v3 - Early Stopping Issues (January 2025)

### Objective

Test if longer training and higher patience improves results.

### Configuration

| Parameter | v2 | v3 |
|-----------|-----|-----|
| hidden_dim | 64 | 64 |
| loss_type | MSE | MSE |
| patience | 7 | 7 |
| max_epochs | 30 | 60 |

### Training Results

Both models stopped early due to patience=7:

| Model | Stopped at Epoch | Best val/std_rmse | Notes |
|-------|------------------|-------------------|-------|
| Model 1 | 15 | 0.00193 | Worse than v2 (0.00135) |
| Model 2 | 11 | 0.0888 | Worse than v2 (0.0812) |

### Kaggle Submission Result

| Date | Version | Public Score | Notes |
|------|---------|--------------|-------|
| 2025-01-23 | Graph-TFT v3 | **0.2793** | WORSE - Early stopping too aggressive |

### Analysis

**v3 performed worse than v2:**
- v2: 0.2281
- v3: 0.2793
- **Degradation: 22.4%**

**Root cause**: Early stopping triggered before model converged properly. The validation metrics may have fluctuated, causing premature stopping.

**Lesson learned**: Higher patience might be needed for this task.

---

## Experiment 9: Graph-TFT v4 - Larger Model + Huber Loss (January 2025)

### Objective

Test if larger model capacity and Huber loss (more robust to outliers) improve results.

### Configuration Changes

| Parameter | v2 (baseline) | v4 |
|-----------|---------------|-----|
| hidden_dim | 64 | **128** |
| loss_type | MSE | **Huber** |
| patience | 7 | **20** |
| max_epochs | 30 | 60 |

### Model 2 Memory Issues

Initial attempt with hidden_dim=128 and batch_size=4 caused OOM on GTX 1080 Ti (11.7GB).

**Solution**: Reduced batch_size to 2, added gradient accumulation to maintain effective batch size:

| Model | batch_size | accumulate_grad_batches | Effective batch |
|-------|------------|------------------------|-----------------|
| Model 1 | 4 | 1 | 4 |
| Model 2 | 2 | 2 | 4 |

### Training Results

| Model | Epochs | Best val/std_rmse | Notes |
|-------|--------|-------------------|-------|
| Model 1 | 41 | 0.00169 | Trained to convergence |
| Model 2 | 60 | 0.0824 | Full 60 epochs |

### Training Commands

```bash
# All-in-one pipeline script created to avoid monitoring issues
python run_full_pipeline.py \
    --exp_name graph_tft_v4 \
    --hidden_dim 128 \
    --loss_type huber \
    --patience 20 \
    --batch_size 4 \
    --batch_size_m2 2 \
    --accumulate_grad_batches_m2 2 \
    --calibration_steps 100
```

### Kaggle Submission Result

| Date | Version | Public Score | Notes |
|------|---------|--------------|-------|
| 2025-01-24 | Graph-TFT v4 | **0.2291** | Slightly worse than v2 |

### Analysis

**v4 performed slightly worse than v2:**
- v2: 0.2281 (hidden_dim=64, MSE)
- v4: 0.2291 (hidden_dim=128, Huber)
- **Degradation: 0.4%**

**Conclusions:**
1. Larger hidden_dim (128 vs 64) did not improve generalization
2. Huber loss did not outperform MSE for this task
3. Model capacity is likely not the bottleneck
4. The architecture may need different changes (not just scaling up)

### New Files Created

- `run_full_pipeline.py`: All-in-one training and prediction script
  - Runs Model 1 training → Model 2 training → Model 1 prediction → Model 2 prediction → Format submission
  - Supports skip flags for already-trained models
  - Separate batch_size and accumulate_grad_batches for each model
  - Avoids background process monitoring issues

- `format_submission_v2.py`: Memory-efficient submission formatter
  - Uses gc.collect() to manage memory
  - Loads predictions separately to avoid OOM

---

## Updated Submission History

| Date | Model | Public Score | Notes |
|------|-------|--------------|-------|
| 2025-01-18 | CL-DTS v1 | 0.4232 | GRU baseline |
| 2025-01-18 | CL-DTS v2 | 0.9684 | Buggy |
| 2025-01-18 | CL-DTS v3 | 1.5016 | Buggy |
| 2025-01-18 | CL-DTS v4 | 0.6722 | Used actual rainfall |
| 2025-01-XX | CL-DTS v7 | 0.6648 | Unknown |
| 2025-01-21 | CL-DTS v8 | 0.6972 | Bug fixes |
| 2025-01-22 | Graph-TFT v1 | 0.2595 | Multi-horizon, no AR |
| 2025-01-22 | **Graph-TFT v2** | **0.2281** | **BEST** - Test-time calibration |
| 2025-01-23 | Graph-TFT v3 | 0.2793 | Early stopping issues |
| 2025-01-24 | Graph-TFT v4 | 0.2291 | Larger model, Huber loss |

### Score Progress (Graph-TFT only)

```
0.2595 (v1) - baseline multi-horizon
   ↓
0.2281 (v2) - BEST, test-time calibration 🎉
   ↑
0.2793 (v3) - early stopping hurt
   ↑
0.2291 (v4) - larger model didn't help
```

### Key Findings

1. **Test-time calibration is crucial**: v2's main improvement came from c_e optimization
2. **Model capacity is not the bottleneck**: hidden_dim=128 performed worse than 64
3. **Loss function choice**: Huber vs MSE made minimal difference
4. **Patience matters**: Too aggressive early stopping (v3) hurts performance
5. **Architecture > Hyperparameters**: The jump from CL-DTS to Graph-TFT (0.4232 → 0.2281) was bigger than all hyperparameter tuning combined

---

## Recommendations for Future Work

### Option 1: Ensemble v2 + v4
- Average predictions from best models
- May capture different aspects of the dynamics

### Option 2: Isolate Variables
- Test one change at a time:
  - v5a: hidden_dim=64, Huber loss (just change loss)
  - v5b: hidden_dim=64, MSE, patience=20 (just change patience)

### Option 3: Architecture Changes
- More/fewer GNN layers
- Different attention mechanisms
- Residual connections in decoder

### Option 4: Data Augmentation
- Synthetic rainfall patterns
- Noise injection during training

### Option 5: Different Calibration Strategy
- More calibration steps (200+)
- Different learning rate for c_e
- Calibrate on different warmup windows

---

---

## Experiment 10: VGSSM - Variational Graph State-Space Model (January 2025)

### Motivation

Graph-TFT achieved 0.2281 with multi-horizon prediction, but it predicts all 90 steps from a single latent state (the final TFT hidden state). This may miss temporal dynamics that evolve over the prediction horizon.

**Key insight**: In real hydraulic systems, the hidden state (flow potential, water momentum, etc.) evolves continuously. A state-space model explicitly captures this evolution.

### Architecture: VGSSM

VGSSM extends Graph-TFT with per-timestep latent dynamics `z_t`:

```
Current Graph-TFT:
  Prefix → TFT Encoder → z_final → Multi-Horizon Decoder → All predictions

VGSSM:
  Prefix → Inference Net → z_0 ~ q(z_0 | prefix, c_e)
  For t = 1 to horizon:
      z_t = z_{t-1} + f_transition(z_{t-1}, graph, u_t, c_e)
      y_t = g_decoder(z_t, spatial_emb)
```

### Key Components

#### 1. Latent Transition Model
Models how the hidden hydraulic state evolves:
```python
z_t+1 = z_t + GNN(z_t, graph) + MLP(z_t, u_t, c_e)
```
- **GNN**: Propagates information spatially (water flows through pipes/surfaces)
- **MLP**: Incorporates external forcing (rainfall, inlet flow) and event context (c_e)
- **Residual**: Ensures stable gradients and smooth transitions

#### 2. Latent Inference Network
Infers initial latent state from prefix:
```python
z_0 ~ q(z_0 | prefix, c_e) = N(μ(prefix, c_e), σ(prefix, c_e))
```
- Separate inference nets for 1D and 2D nodes
- Uses GRU to encode prefix temporal patterns
- Conditioned on event latent c_e

#### 3. Dual KL Regularization
Two KL divergence terms with free-bits:
```python
L = L_recon + β_ce·KL(c_e || prior) + β_z·KL(z_0 || prior)
```
- β_ce = 0.01 (event latent)
- β_z = 0.001 (initial state - smaller since more dimensions)
- Free bits prevent posterior collapse

#### 4. Test-Time Calibration
Optimizes both c_e AND z_0 during inference:
```python
c_e, z0_1d, z0_2d = optimize_latents(
    model, graph, prefix,
    warmup_targets, warmup_rainfall,
    num_steps=50, lr=0.01
)
```

### New Files

- `src/models/vgssm.py`:
  - `HeteroGNNBlock`: Lightweight GNN for transition
  - `LatentTransition`: `z_{t+1} = z_t + f(z_t, graph, u_t, c_e)`
  - `LatentInferenceNet`: `q(z_0 | prefix, c_e)`
  - `LatentDecoder`: `y_t = g(z_t)`
  - `VGSSM`: Main model class

- `src/training/vgssm_trainer.py`:
  - `VGSSMTrainer`: Lightning module with dual KL losses
  - Horizon-weighted loss
  - KL annealing
  - Free-bits implementation

- `train_vgssm.py`: Training script
- `predict_vgssm.py`: Prediction with latent calibration

### Training Configuration

```bash
# Model 1
python train_vgssm.py \
    --model_id 1 \
    --exp_name vgssm_v1 \
    --hidden_dim 64 \
    --latent_dim 32 \
    --event_latent_dim 16 \
    --num_gnn_layers 3 \
    --num_transition_gnn_layers 2 \
    --beta_ce 0.01 \
    --beta_z 0.001 \
    --max_epochs 30 \
    --batch_size 4 \
    --accelerator cuda

# Model 2 (auto-scales to larger capacity)
python train_vgssm.py \
    --model_id 2 \
    --exp_name vgssm_v1 \
    --max_epochs 30 \
    --batch_size 4 \
    --accelerator cuda
```

### Model-Specific Hyperparameters

| Parameter | Model 1 | Model 2 |
|-----------|---------|---------|
| hidden_dim | 64 | 96 |
| latent_dim | 32 | 48 |
| num_gnn_layers | 3 | 4 |
| num_transition_gnn_layers | 2 | 3 |
| dropout | 0.2 | 0.15 |

### Prediction Commands

```bash
# With test-time calibration of c_e and z_0
python predict_vgssm.py \
    --model_id 1 \
    --checkpoint checkpoints/model_1/vgssm_v1/best.ckpt \
    --calibrate_latent \
    --calibration_steps 50

python predict_vgssm.py \
    --model_id 2 \
    --checkpoint checkpoints/model_2/vgssm_v1/best.ckpt \
    --calibrate_latent \
    --calibration_steps 50
```

### Expected Improvements

1. **Temporal coherence**: z_t evolves smoothly, avoiding discontinuities
2. **Physics alignment**: State-space matches hydraulic dynamics
3. **Better calibration**: Optimizing z_0 (not just c_e) captures per-event initial conditions
4. **Interpretability**: z_t can be analyzed to understand hidden dynamics

### Potential Risks

1. **Posterior collapse**: Addressed with free-bits and low β_z
2. **Slow convergence**: May need more epochs than Graph-TFT
3. **Memory**: Sequential rollout during training is heavier than parallel

### Status

**Implemented** - Ready for training and evaluation.

---

## Architecture Comparison

| Model | Temporal | Latent | Prediction | Calibration |
|-------|----------|--------|------------|-------------|
| CL-DTS | GRU (AR) | c_e only | 1-step, rollout | c_e |
| Graph-TFT | TFT | c_e only | Multi-horizon | c_e |
| **VGSSM** | State-space | c_e + z_t | Rollout | c_e + z_0 |

---

## References

1. Urban Flood Modelling Kaggle Competition
2. PyTorch Geometric - Heterogeneous Graph Learning
3. Variational Autoencoders for Sequential Data
4. Graph Neural Networks for Physical Simulations
5. Temporal Fusion Transformers (https://arxiv.org/abs/1912.09363)
6. State Space Models for Machine Learning (Gu et al., 2022)
