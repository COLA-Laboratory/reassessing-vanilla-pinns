"""Wave (alpha=4, beta=4) — u_tt - 4*u_xx = 0
Best 48-grid config: d=5, w=50, sin, lr=5e-4
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

pde = Wave(beta=4, device=device)
model = create_model(depth=5, width=50, activation='sin', device=device)
model, results = train(model, pde, lr=5e-4, device=device)
results.update(pde='wave44', depth=5, width=50, activation='sin')
save_results(results, 'wave44')
plot_field(pde, model, 'wave44', device=device)
