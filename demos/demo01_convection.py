"""PDE1: Convection (beta=30) — u_t + 30*u_x = 0
Best 48-grid config: d=2, w=10, sin, lr=0.05
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

pde = Convection(beta=30, device=device)
model = create_model(depth=2, width=10, activation='sin', device=device)
model, results = train(model, pde, lr=0.05, device=device)
results.update(pde='pde1_convection', depth=2, width=10, activation='sin')
save_results(results, 'pde1_convection')
plot_field(pde, model, 'pde1_convection', device=device)
