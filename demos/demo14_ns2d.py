"""NS2D: 2D Navier-Stokes Cylinder Wake (Re=100)
Best 48-grid config: d=5, w=50, swish, lr=5e-3

NOTE: This demo takes ~60 min on GPU due to expensive NS residual computation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model import create_model
from src.trainer import set_seed, save_results
from src.ns2d import train_ns2d
from src.utils import plot_ns2d_field

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(42)

# NS2D: 3 inputs (x, y, t) -> 2 outputs (psi, p)
model = create_model(depth=5, width=50, activation='swish',
                     in_dim=3, out_dim=2, device=device)
model, results = train_ns2d(model, lr=5e-3, device=device)
results.update(pde='ns2d', depth=5, width=50, activation='swish',
               model_state_dict={k: v.cpu() for k, v in model.state_dict().items()})
save_results(results, 'ns2d')
plot_ns2d_field(model, 'ns2d', rrmse=results['final_rrmse'], device=device)
print(f"\nFinal rRMSE: {results['final_rrmse']:.4e}")
