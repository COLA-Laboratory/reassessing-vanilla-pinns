"""
Unified training pipeline for scalar PDEs: Adam (beta2=0.9) + L-BFGS.
"""
import os, time, random, pickle
import numpy as np
import torch
from .pdes import get_data_2d

ADAM_EPOCHS = 80000
ADAM_BETA2 = 0.9
LBFGS_ITERS = 200
GRID_SIZE = 100
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demos', 'Results')


def set_seed(seed=SEED):
    """Set all random seeds for reproducibility (must be called before model creation)."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(model, pde, lr, la_bc=1.0, device='cpu', log_every=10000):
    """Train a scalar PDE model with Adam + L-BFGS.

    NOTE: Call set_seed() BEFORE create_model() to ensure reproducibility.

    Returns:
        model:   Trained model.
        results: Dict with full training history, timing, and model_state_dict.
    """
    # Setup data
    points, b1, b2, b3, b4 = get_data_2d(GRID_SIZE, GRID_SIZE, pde.domain, device)
    pts_test, _ = pde.get_test_data(GRID_SIZE)

    train_history = []

    # Phase 1: Adam
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, ADAM_BETA2))
    t0 = time.time()
    for ep in range(ADAM_EPOCHS):
        def closure():
            opt.zero_grad()
            loss, metrics = pde.compute_losses(
                model, points, b1, b2, b3, b4, pts_test, la_bc=la_bc)
            loss.backward()
            closure.loss = loss.item()
            closure.metrics = metrics
            return loss
        opt.step(closure)
        train_history.append(closure.metrics.copy())
        if (ep + 1) % log_every == 0:
            m = closure.metrics
            print(f"  Adam [{ep+1}/{ADAM_EPOCHS}]  loss={closure.loss:.2e}  "
                  f"rRMSE={m['rrmse']:.2e}  "
                  f"(pde={m['pde']:.1e} bc={m['bc']:.1e} ic={m['ic']:.1e})")
    adam_time = time.time() - t0
    adam_rrmse = closure.metrics['rrmse']
    print(f"  Adam done ({adam_time:.0f}s), rRMSE={adam_rrmse:.2e}")

    # Phase 2: L-BFGS
    opt_l = torch.optim.LBFGS(model.parameters(), lr=1.0,
                               line_search_fn='strong_wolfe')
    t1 = time.time()
    for it in range(LBFGS_ITERS):
        def closure_l():
            opt_l.zero_grad()
            loss, metrics = pde.compute_losses(
                model, points, b1, b2, b3, b4, pts_test, la_bc=la_bc)
            loss.backward()
            closure_l.metrics = metrics
            return loss
        opt_l.step(closure_l)
        train_history.append(closure_l.metrics.copy())
    lbfgs_time = time.time() - t1

    final_rrmse = closure_l.metrics['rrmse']
    print(f"  L-BFGS done ({lbfgs_time:.0f}s), rRMSE={final_rrmse:.2e}")
    print(f"  Total: {adam_time + lbfgs_time:.0f}s")

    results = dict(
        lr=lr,
        la_bc=la_bc,
        grid_size=GRID_SIZE,
        adam_epochs=ADAM_EPOCHS,
        adam_beta2=ADAM_BETA2,
        lbfgs_iters=LBFGS_ITERS,
        seed=SEED,
        train_history=train_history,
        training_time=adam_time + lbfgs_time,
        adam_time=adam_time,
        lbfgs_time=lbfgs_time,
        adam_final_rrmse=adam_rrmse,
        final_rrmse=final_rrmse,
        model_state_dict={k: v.cpu() for k, v in model.state_dict().items()},
    )
    return model, results


def save_results(results, name):
    """Save results dict as pkl to results/ directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Saved: {path}")
    return path
