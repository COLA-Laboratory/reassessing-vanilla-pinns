"""PDE8: Helmholtz (n=2, k0=4*pi) — -u_xx - u_yy - k0^2*u = f(x,y)
Best 48-grid config: d=5, w=50, swish, lr=5e-3, la_bc=500
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model import create_model
from src.pdes import Helmholtz
from src.trainer import set_seed, train, save_results
from src.utils import plot_field

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(42)

pde = Helmholtz(n=2, device=device)
model = create_model(depth=5, width=50, activation='swish', device=device)
model, results = train(model, pde, lr=5e-3, la_bc=500.0, device=device)
results.update(pde='pde8_helmholtz', depth=5, width=50, activation='swish')
save_results(results, 'pde8_helmholtz')
plot_field(pde, model, 'pde8_helmholtz', device=device)
