"""PDE5: Burgers (nu=0.01/pi) — u_t + u*u_x - nu*u_xx = 0
Best 48-grid config: d=5, w=10, tanh, lr=5e-3
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model import create_model
from src.pdes import Burgers
from src.trainer import set_seed, train, save_results
from src.utils import plot_field

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(42)

pde = Burgers(device=device)
model = create_model(depth=5, width=10, activation='tanh', device=device)
model, results = train(model, pde, lr=5e-3, device=device)
# Burgers uses reference data; plot_field handles non-analytical PDEs
results.update(pde='pde5_burgers', depth=5, width=10, activation='tanh')
save_results(results, 'pde5_burgers')
plot_field(pde, model, 'pde5_burgers', device=device)
