"""PDE2: Reaction (rho=5) — u_t = 5*u*(1-u)
Best 48-grid config: d=5, w=50, swish, lr=5e-3
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

pde = Reaction(rho=5, device=device)
model = create_model(depth=5, width=50, activation='swish', device=device)
model, results = train(model, pde, lr=5e-3, device=device)
results.update(pde='pde2_reaction', depth=5, width=50, activation='swish')
save_results(results, 'pde2_reaction')
plot_field(pde, model, 'pde2_reaction', device=device)
