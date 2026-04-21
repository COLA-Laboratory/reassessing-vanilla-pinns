"""Convection (beta=100) — u_t + 100*u_x = 0
Best 48-grid config: d=2, w=50, sin, lr=0.05
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model import create_model
from src.pdes import Convection
from src.trainer import set_seed, train, save_results
from src.utils import plot_field

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(42)

pde = Convection(beta=100, device=device)
model = create_model(depth=2, width=50, activation='sin', device=device)
model, results = train(model, pde, lr=0.05, device=device)
results.update(pde='conv100', depth=2, width=50, activation='sin')
save_results(results, 'conv100')
plot_field(pde, model, 'conv100', device=device)
