"""Reaction (rho=6) — u_t = 6*u*(1-u)
Best 48-grid config: d=5, w=10, tanh, lr=0.05
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model import create_model
from src.pdes import Reaction
from src.trainer import set_seed, train, save_results
from src.utils import plot_field

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(42)

pde = Reaction(rho=6, device=device)
model = create_model(depth=5, width=10, activation='tanh', device=device)
model, results = train(model, pde, lr=0.05, device=device)
results.update(pde='react6', depth=5, width=10, activation='tanh')
save_results(results, 'react6')
plot_field(pde, model, 'react6', device=device)
