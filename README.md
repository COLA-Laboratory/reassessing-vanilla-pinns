# Reassessing Vanilla PINNs

We show that a well-tuned vanilla PINN (MLP) combined with our proposed training pipeline is a substantially stronger baseline on benchmark PDEs than commonly reported. This repository provides 14 reproducible demos and the full hyperparameter search data behind the paper.

![Training animation for Convection (beta=100)](assets/conv100_training.gif)

Live training of a vanilla PINN on Convection (β=100) under the proposed pipeline. Adam (β₂=0.9) escapes the partial-loss attractor near step 60k; L-BFGS then drives rRMSE from about 0.7 down to about 8×10⁻⁴.

## Installation

```bash
pip install -r requirements.txt
```

Requires PyTorch >= 2.0 with CUDA support (CPU also works but is slower).

## Demos

The `demos/` directory contains 14 self-contained scripts (`demo01_*.py` through `demo14_*.py`), each reproducing the best vanilla PINN (MLP) result for one PDE under the proposed two-phase training pipeline: Adam ($\beta_2 = 0.9$, 80,000 epochs) followed by L-BFGS (200 iterations). Each demo uses a $100 \times 100$ collocation grid, trains with the best hyperparameters identified by the 48-configuration search in the paper, and saves a `.pkl` (training history + model weights) and a `.png` field plot under `demos/Results/`.

Demos 01-08 cover the eight benchmark PDEs (Convection, Reaction, Diffusion, Wave, Burgers, Klein-Gordon, Allen-Cahn, Helmholtz). Demos 09-14 cover the six harder PDEs (Convection $\beta=50, 100$; Reaction $\rho=6$; Wave $m=4$; Helmholtz $n=3$; 2D Navier-Stokes).

```bash
python demos/demo10_conv100.py
```

## Data

The `Data/` directory provides the full hyperparameter search results from the paper.

- **`Data/LHS/`** — 2,000 Latin Hypercube Samples per PDE across 8 benchmarks (16,000 runs total), used for the Section 4 hyperparameter importance analysis.
- **`Data/HPO/`** — 48-configuration search results on all 14 PDEs for three methods (672 runs per file): `proposed.csv` (vanilla PINN with Adam $\beta_2=0.9$ + L-BFGS), `default.csv` (vanilla PINN with Adam $\beta_2=0.999$ + L-BFGS), and `spformer.csv` (S-Pformer with its source L-BFGS 1000 pipeline). Used for the Section 6 HPO results.

Each CSV includes per-run hyperparameters, final rRMSE, and training time. See column headers for field details.
