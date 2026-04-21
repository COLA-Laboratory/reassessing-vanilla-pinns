# Reassessing Vanilla PINNs

Code repository for the paper:

**Reassessing Vanilla PINNs: Hyperparameters, Optimization Robustness, and Stronger Baselines**

## Overview

This repository provides 14 self-contained demos that reproduce the best vanilla PINN (MLP) results from a systematic 48-configuration grid search. Each demo trains a PINN on a specific PDE using the optimal hyperparameters identified in the paper, following a two-phase training pipeline: Adam (beta2=0.9, 80k epochs) + L-BFGS (200 iterations).

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Requires PyTorch >= 2.0 with CUDA support (CPU also works but is slower).

### Run a Demo

```bash
cd reassessing_vanilla_pinns
python demos/demo01_convection.py
```

Each demo:
1. Trains the PINN with the optimal configuration from the 48-grid search
2. Saves a `.pkl` file to `demos/data/` (full training history + model weights)
3. Saves a `.png` field plot to `demos/data/` (True / Predicted / Error)

### Best Configurations (48-Grid Search)

| Demo | PDE | Depth | Width | Activation | LR |
|------|-----|-------|-------|------------|-----|
| 01 | Convection (beta=30) | 2 | 10 | sin | 5e-2 |
| 02 | Reaction (rho=5) | 5 | 50 | swish | 5e-3 |
| 03 | Diffusion | 5 | 50 | swish | 5e-5 |
| 04 | Wave (alpha=4, beta=3) | 2 | 10 | sin | 5e-2 |
| 05 | Burgers | 5 | 10 | tanh | 5e-3 |
| 06 | Klein-Gordon | 2 | 10 | sin | 5e-2 |
| 07 | Allen-Cahn | 5 | 50 | tanh | 5e-3 |
| 08 | Helmholtz (n=2) | 5 | 50 | swish | 5e-3 |
| 09 | Convection (beta=50) | 2 | 10 | sin | 5e-2 |
| 10 | Convection (beta=100) | 2 | 50 | sin | 5e-2 |
| 11 | Reaction (rho=6) | 5 | 10 | tanh | 5e-2 |
| 12 | Wave (alpha=4, beta=4) | 5 | 50 | sin | 5e-4 |
| 13 | Helmholtz (n=3) | 5 | 50 | sin | 5e-4 |
| 14 | NS2D (Re=100) | 5 | 50 | swish | 5e-3 |

All demos use: seed=42, grid=100x100, Adam(beta2=0.9, 80k epochs) + L-BFGS(200 iters).

### Training Pipeline

```
Phase 1: Adam optimizer (beta1=0.9, beta2=0.9), 80,000 epochs
Phase 2: L-BFGS optimizer (strong Wolfe line search), 200 iterations
```

### Data

#### `Data/LHS/` — Section 4 Large-Scale Search

2,000 Latin Hypercube Samples for each of the 8 benchmark PDEs (16,000 runs total) from the hyperparameter importance analysis. Each CSV includes:

| Column | Description |
|--------|-------------|
| `rRMSE` | Relative root mean square error |
| `pde_loss` | PDE residual loss |
| `bc_loss` | Boundary condition loss |
| `ic_loss` | Initial condition loss |
| `total_loss` | Weighted total loss |
| `num_layers` | Network depth |
| `num_nodes` | Hidden layer width |
| `activation_func` | Activation function |
| `epochs` | Training epochs |
| `grid_size` | Collocation grid size |
| `learning_rate` | Learning rate |

#### `Data/HPO/` — Section 6 Compact 48-Grid Results

Full 48-configuration search on all 14 PDEs (8 benchmark + 6 harder) for three methods (672 runs per file, 2,016 total).

| File | Method | Pipeline |
|------|--------|----------|
| `proposed.csv` | Vanilla PINN, 48-grid | Adam ($\beta_2 = 0.9$) + L-BFGS (200 iters) |
| `default.csv`  | Vanilla PINN, 48-grid | Adam ($\beta_2 = 0.999$) + L-BFGS (200 iters) |
| `spformer.csv` | S-Pformer, 48 random configs | Source pipeline (L-BFGS 1000 iters with NTK reweighting) |

##### Vanilla PINN CSVs (`proposed.csv`, `default.csv`)

| Column | Description |
|--------|-------------|
| `pde` | PDE key (`pde1`–`pde8`, `ch6_conv50/100`, `ch6_react6`, `ch6_wave44`, `ch6_helm3_la500`, `ns2d`) |
| `depth`, `width`, `activation`, `learning_rate` | Searched hyperparameters |
| `adam_beta2`, `adam_epochs`, `lbfgs_iters` | Training pipeline settings |
| `grid_size`, `la_bc` | Fixed hyperparameters |
| `adam_final_rrmse` | rRMSE after Adam, before L-BFGS |
| `final_rrmse` | rRMSE after the full pipeline |
| `adam_time`, `lbfgs_time`, `training_time` | Wall-clock time (seconds) |
| `seed` | Random seed (42 for all runs) |

##### S-Pformer CSV (`spformer.csv`)

| Column | Description |
|--------|-------------|
| `pde` | PDE key (same 14 PDEs as above) |
| `trial` | Random-search trial index (0–47) |
| `d_hidden`, `d_model`, `mapping_size` | Searched architectural hyperparameters |
| `n_params` | Number of trainable parameters |
| `lbfgs_iters` | L-BFGS iterations (1000 for all trials) |
| `final_rrmse`, `final_rmae` | rRMSE and rMAE on the evaluation grid |
| `train_time` | Wall-clock time (seconds) |
| `seed` | Random seed (42 for all trials) |

## Citation

```bibtex
@article{reassessing_vanilla_pinns,
  title={Reassessing Vanilla PINNs: Hyperparameters, Optimization Robustness, and Stronger Baselines},
  author={},
  year={2026}
}
```
