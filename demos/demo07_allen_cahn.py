"""PDE7: Allen-Cahn — u_t - 0.001*u_xx - 5*(u - u^3) = 0
Best 48-grid config: d=5, w=50, tanh, lr=5e-3
NOTE: Uses output transform for hard BC/IC enforcement.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model import create_model
from src.pdes import AllenCahn
from src.trainer import set_seed, train, save_results
from src.utils import plot_field

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(42)

pde = AllenCahn(device=device)
model = create_model(depth=5, width=50, activation='tanh', device=device)
pde.apply_output_transform(model)  # hard BC/IC
model, results = train(model, pde, lr=5e-3, device=device)
# Allen-Cahn uses reference data; plot predicted field only
results.update(pde='pde7_allen_cahn', depth=5, width=50, activation='tanh')
save_results(results, 'pde7_allen_cahn')
plot_field(pde, model, 'pde7_allen_cahn', device=device)
