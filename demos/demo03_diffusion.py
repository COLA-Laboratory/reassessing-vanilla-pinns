"""PDE3: Diffusion — u_t - u_xx + source = 0
Best 48-grid config: d=5, w=50, swish, lr=5e-5
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model import create_model
from src.pdes import Diffusion
from src.trainer import set_seed, train, save_results
from src.utils import plot_field

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(42)

pde = Diffusion(device=device)
model = create_model(depth=5, width=50, activation='swish', device=device)
model, results = train(model, pde, lr=5e-5, device=device)
results.update(pde='pde3_diffusion', depth=5, width=50, activation='swish')
save_results(results, 'pde3_diffusion')
plot_field(pde, model, 'pde3_diffusion', device=device)
