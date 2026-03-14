# Urban Flood Modelling - CL-DTS

**Coupled Latent Digital Twin Surrogate** for the [Kaggle UrbanFloodBench Competition](https://www.kaggle.com/competitions/urban-flood-modelling).

## Status Update (2026-03-15)

- Latest corrected-dataset best public score in this repository lineage: `0.0701`
- Best scored submission:
  - `submission_20260314_m1v2e07_base_m2_correctedsync_full399_epoch15_calib_poly3.parquet`
- Canonical wrap-up document:
  - `docs/final_wrapup_2026-03-15.md`
- Detailed corrected-dataset debugging log:
  - `docs/rerelease_root_cause_log_2026-02-23.md`
- Important:
  - large competition assets are intentionally not tracked in this public repo
  - older sections below are retained for historical context and are not the latest status

## Overview

This solution treats the urban flood modelling problem as a **partial-observation digital twin** problem. We learn a **coupled 1D-2D autoregressive surrogate** with:

- **Heterogeneous Graph Neural Networks** for spatial message passing across 1D (pipe network) and 2D (surface mesh) domains
- **Temporal GRU/TCN** for sequential dynamics
- **Event Latent (c_e)**: Captures event-specific unknowns (roughness, blockages, inlet efficiency)
- **Dynamic Latent (z_t)**: Hidden physical state (flow potential, velocity field) - *Phase D*
- **Test-time event calibration**: Optimize c_e on observation prefix for each test event

## Architecture

```
                    ┌─────────────────┐
                    │  Static Graph   │
                    │  (1D-2D coupled)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Spatial GNN    │
                    │  (Hetero Conv)  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│  Dynamic 1D   │   │  Event Latent   │   │  Dynamic 2D   │
│  Features     │   │    Encoder      │   │  Features     │
└───────┬───────┘   └────────┬────────┘   └───────┬───────┘
        │                    │                    │
┌───────▼───────┐            │            ┌───────▼───────┐
│  Temporal GRU │            │            │  Temporal GRU │
│  (1D nodes)   │◄───────────┼───────────►│  (2D nodes)   │
└───────┬───────┘            │            └───────┬───────┘
        │                    │                    │
        └────────────┬───────┴───────┬────────────┘
                     │               │
              ┌──────▼─────┐   ┌─────▼──────┐
              │  Decoder   │   │  Decoder   │
              │    (1D)    │   │    (2D)    │
              └──────┬─────┘   └─────┬──────┘
                     │               │
                     ▼               ▼
              water_level_1d   water_level_2d
```

## Project Structure

```
urbanfloodbench/
├── src/
│   ├── data/
│   │   ├── graph_builder.py        # Heterogeneous graph construction
│   │   └── dataset.py              # PyTorch datasets for events
│   ├── models/
│   │   ├── coupled_gnn.py          # Coupled 1D-2D GNN
│   │   ├── temporal.py             # GRU/TCN temporal blocks
│   │   ├── cldts.py                # Original CL-DTS model
│   │   ├── tft.py                  # TFT components (GRN, attention)
│   │   ├── graph_tft.py            # Graph-TFT model
│   │   └── vgssm.py                # Variational Graph State-Space Model
│   ├── training/
│   │   ├── losses.py               # ELBO, rollout, physics losses
│   │   ├── trainer.py              # PyTorch Lightning trainer
│   │   ├── graph_tft_trainer.py    # Graph-TFT trainer
│   │   └── vgssm_trainer.py        # VGSSM trainer
│   └── utils/
│       ├── metrics.py              # Evaluation metrics
│       └── normalization.py        # Data normalization
├── configs/
│   └── baseline.yaml               # Configuration file
├── train.py                        # Training script (CL-DTS)
├── predict.py                      # Prediction script (CL-DTS)
├── train_graph_tft.py              # Training script (Graph-TFT)
├── predict_graph_tft.py            # Prediction script (Graph-TFT)
├── train_vgssm.py                  # Training script (VGSSM)
├── predict_vgssm.py                # Prediction script (VGSSM)
└── test_setup.py                   # Verify installation
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch pytorch-lightning torch-geometric geopandas pandas numpy scikit-learn
```

### 2. Verify Setup

```bash
python test_setup.py
```

### 3. Train Model

```bash
# Train on Model 1
python train.py --model_id 1 --max_epochs 50

# Train on Model 2
python train.py --model_id 2 --max_epochs 50
```

### 4. Generate Predictions

```bash
python predict.py \
    --checkpoint checkpoints/cldts_v1/model_1/last.ckpt \
    --output submission.csv
```

### 5. Submit to Kaggle

```bash
kaggle competitions submit -c urban-flood-modelling -f submission.csv -m "CL-DTS baseline"
```

## Best Results (Graph-TFT v2)

| Version | Architecture | Public Score | Notes |
|---------|--------------|--------------|-------|
| CL-DTS v1 | GRU autoregressive | 0.4232 | Original baseline |
| **Graph-TFT v2** | **TFT multi-horizon** | **0.2281** | **Best result** |
| Graph-TFT v4 | TFT + larger model | 0.2291 | Larger hidden_dim didn't help |

### Key Improvements (v1 → v2)
- **Architecture change**: GRU → TFT with multi-horizon prediction (46% improvement)
- **Test-time calibration**: Optimize event latent c_e on warmup period
- **Known future inputs**: Properly pass rainfall to decoder

### Submission Statistics
- **Total Predictions**: 50.9 million rows
- **Model 1**: 19.7M rows, water level range [287, 360]
- **Model 2**: 33.6M rows, water level range [23, 55]
- **File Size**: 330 MB (parquet format)

For detailed experiment logs, see [docs/experiments.md](docs/experiments.md).

## Experiment Phases

### Phase A: Graph Topology ✅
- Heterogeneous graph with 1D, 2D nodes
- Edge types: pipe, surface, coupling
- Bidirectional message passing

### Phase B: Deterministic Baseline ✅
- Coupled GNN + GRU temporal encoder
- Teacher forcing training
- Multi-step rollout loss

### Phase C: Event Latent c_e ✅
- CVAE-style event encoding
- KL regularization with annealing (beta=0.1)
- 16-dimensional event latent space

### Phase D: Dynamic Latent z_t ✅ (VGSSM)
- **Variational Graph State-Space Model (VGSSM)** implemented
- Per-timestep latent dynamics: `z_{t+1} = z_t + f(z_t, graph, u_t, c_e)`
- Inference network for z_0: `q(z_0 | prefix, c_e)`
- Dual KL losses with free-bits to prevent posterior collapse
- Graph-conditioned transition model for spatial propagation

### Phase E: Final Polish (Planned)
- Test-time event latent optimization
- Ensemble of models
- Quantile clipping
- EMA/SWA weights

## Model Architectures

### 1. CL-DTS (Original)
GRU-based autoregressive model with event latent.

### 2. Graph-TFT (Best Score: 0.2281)
TFT with multi-horizon prediction - predicts all 90 steps at once.

### 3. VGSSM (Latest)
Variational Graph State-Space Model with per-timestep latent dynamics.

```
┌─────────────────────────────────────────────────────────────────┐
│                    VGSSM Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│  Prefix (t=1-10) → Event Encoder → c_e (event-level latent)    │
│                         ↓                                        │
│  Prefix → Inference Net → z_0 ~ q(z_0 | prefix, c_e)           │
│                         ↓                                        │
│  For t = 1 to horizon:                                          │
│      z_t = z_{t-1} + GNN(z_{t-1}, graph) + MLP(z_{t-1}, u_t, c_e)│
│      y_t = Decoder(z_t, spatial_emb)                            │
│                         ↓                                        │
│  Loss = L_recon + β_ce·KL(c_e) + β_z·KL(z_0)                   │
└─────────────────────────────────────────────────────────────────┘
```

#### VGSSM Training
```bash
python train_vgssm.py --model_id 1 --exp_name vgssm_v1 --max_epochs 30
```

#### VGSSM Prediction with Calibration
```bash
python predict_vgssm.py \
    --model_id 1 \
    --checkpoint checkpoints/model_1/vgssm_v1/best.ckpt \
    --calibrate_latent \
    --calibration_steps 50
```

## Key Hyperparameters

### Graph-TFT / CL-DTS
| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 64 | Hidden dimension |
| num_gnn_layers | 3 | GNN message passing layers |
| num_temporal_layers | 2 | Temporal GRU/TFT layers |
| seq_len | 16 | Input sequence length |
| rollout_steps | 8 | Training rollout horizon |
| beta | 0.1 | KL weight for ELBO |
| event_latent_dim | 16 | Event latent dimension |

### VGSSM-Specific
| Parameter | Default | Description |
|-----------|---------|-------------|
| latent_dim | 32 | Per-timestep latent dimension (z_t) |
| beta_ce | 0.01 | KL weight for event latent c_e |
| beta_z | 0.001 | KL weight for initial state z_0 |
| num_transition_gnn_layers | 2 | GNN layers in latent transition |
| free_bits_ce | 0.1 | Free bits for c_e (prevents collapse) |
| free_bits_z | 0.05 | Free bits for z_0 (prevents collapse) |

## Data

The dataset contains two urban drainage models:
- **Model 1**: 17 1D nodes, 3,716 2D cells, 16 coupling connections
- **Model 2**: 198 1D nodes, 4,299 2D cells, 197 coupling connections

Each model has ~70 training events and ~30 test events.

### Data Structure

```
data/
├── Model_1/
│   ├── shapefiles/         # Static geometry
│   ├── train/
│   │   ├── event_1/        # Dynamic time series
│   │   └── ...
│   └── test/
│       ├── event_5/
│       └── ...
└── Model_2/
    └── ...
```

## License

MIT

## Citation

If you use this code, please cite:

```
@software{cldts2024,
  title={CL-DTS: Coupled Latent Digital Twin Surrogate for Urban Flood Modelling},
  author={Your Name},
  year={2024},
  url={https://github.com/yourname/urbanfloodbench}
}
```
