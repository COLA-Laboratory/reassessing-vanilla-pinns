"""PDE4: Wave (alpha=4, beta=3) — u_tt - 4*u_xx = 0
Best 48-grid config: d=2, w=10, sin, lr=0.05
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model import create_model
from src.pdes import Wave
from src.trainer import set_seed, train, save_results
from src.utils import plot_field

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(42)

pde = Wave(beta=3, device=device)
model = create_model(depth=2, width=10, activation='sin', device=device)
model, results = train(model, pde, lr=0.05, device=device)
results.update(pde='pde4_wave', depth=2, width=10, activation='sin')
save_results(results, 'pde4_wave')
plot_field(pde, model, 'pde4_wave', device=device)
