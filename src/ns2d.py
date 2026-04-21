"""
2D Navier-Stokes (Cylinder Wake) — Raissi et al. 2019.

Separate module because NS2D has a fundamentally different setup:
  - 3 inputs (x, y, t) -> 2 outputs (stream function psi, pressure p)
  - Velocity derived: u = psi_y, v = -psi_x
  - Loss = data (velocity matching) + physics (NS residual)
  - Reference data from cylinder_nektar_wake.mat
"""
import os, time
import numpy as np
import scipy.io
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demos')

ADAM_EPOCHS = 80000
ADAM_BETA2 = 0.9
LBFGS_ITERS = 200
N_TRAIN = 2500
SEED = 42
NU = 0.01  # kinematic viscosity (Re=100)


def load_ns2d_data(device='cpu'):
    """Load cylinder wake reference data.

    Returns dict with training tensors and test snapshot.
    """
    data = scipy.io.loadmat(os.path.join(DATA_DIR, 'cylinder_nektar_wake.mat'))
    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']       # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(t_star, (1, N)).T

    x_all = XX.flatten()[:, None]
    y_all = YY.flatten()[:, None]
    t_all = TT.flatten()[:, None]
    u_all = U_star[:, 0, :].flatten()[:, None]
    v_all = U_star[:, 1, :].flatten()[:, None]

    idx = np.random.choice(N * T, N_TRAIN, replace=False)

    train = dict(
        x=torch.tensor(x_all[idx], dtype=torch.float32, requires_grad=True).to(device),
        y=torch.tensor(y_all[idx], dtype=torch.float32, requires_grad=True).to(device),
        t=torch.tensor(t_all[idx], dtype=torch.float32, requires_grad=True).to(device),
        u=torch.tensor(u_all[idx], dtype=torch.float32).to(device),
        v=torch.tensor(v_all[idx], dtype=torch.float32).to(device),
    )

    snap = 100
    test = dict(
        x=torch.tensor(X_star[:, 0:1], dtype=torch.float32, requires_grad=True).to(device),
        y=torch.tensor(X_star[:, 1:2], dtype=torch.float32, requires_grad=True).to(device),
        t=torch.tensor(TT[:, snap:snap+1], dtype=torch.float32, requires_grad=True).to(device),
        p_ref=P_star[:, snap:snap+1],
    )
    return train, test


def compute_ns2d_loss(model, train):
    """Compute combined data + physics loss for NS2D."""
    src = torch.cat((train['x'], train['y'], train['t']), dim=-1)
    out = model(src)
    psi, p = out[:, 0:1], out[:, 1:2]

    # Velocity from stream function
    u = torch.autograd.grad(psi, train['y'], torch.ones_like(psi),
                            retain_graph=True, create_graph=True)[0]
    v = -torch.autograd.grad(psi, train['x'], torch.ones_like(psi),
                             retain_graph=True, create_graph=True)[0]

    # NS residuals
    u_t = torch.autograd.grad(u, train['t'], torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, train['x'], torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, train['y'], torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, train['x'], torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, train['y'], torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

    v_t = torch.autograd.grad(v, train['t'], torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_x = torch.autograd.grad(v, train['x'], torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, train['y'], torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, train['x'], torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, train['y'], torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

    p_x = torch.autograd.grad(p, train['x'], torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, train['y'], torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    f_u = u_t + (u * u_x + v * u_y) + p_x - NU * (u_xx + u_yy)
    f_v = v_t + (u * v_x + v * v_y) + p_y - NU * (v_xx + v_yy)

    loss = (torch.mean((u - train['u'])**2) + torch.mean((v - train['v'])**2)
            + torch.mean(f_u**2) + torch.mean(f_v**2))
    return loss


def eval_pressure(model, test):
    """Evaluate pressure rRMSE at test snapshot (with optimal constant correction)."""
    with torch.no_grad():
        src = torch.cat((test['x'], test['y'], test['t']), dim=-1)
        out = model(src)
        p_pred = out[:, 1:2].cpu().numpy()
    p_ref = test['p_ref']
    C_opt = np.mean(p_ref - p_pred)
    p_corr = p_pred + C_opt
    rrmse = np.linalg.norm(p_ref - p_corr) / np.linalg.norm(p_ref)
    return rrmse


def train_ns2d(model, lr, device='cpu', log_every=10000):
    """Train NS2D model with Adam + L-BFGS pipeline.

    Returns:
        model: trained model
        history: list of (epoch, loss, rrmse) tuples
    """
    train, test = load_ns2d_data(device)
    history = []

    # Phase 1: Adam
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, ADAM_BETA2))
    t0 = time.time()
    for ep in range(ADAM_EPOCHS):
        def closure():
            opt.zero_grad()
            loss = compute_ns2d_loss(model, train)
            loss.backward()
            closure.val = loss.item()
            return loss
        opt.step(closure)
        if (ep + 1) % log_every == 0:
            rr = eval_pressure(model, test)
            history.append((ep + 1, closure.val, rr))
            print(f"  Adam [{ep+1}/{ADAM_EPOCHS}]  loss={closure.val:.2e}  rRMSE={rr:.2e}")
    adam_time = time.time() - t0
    adam_rrmse = eval_pressure(model, test)
    print(f"  Adam done ({adam_time:.0f}s), rRMSE={adam_rrmse:.2e}")

    # Phase 2: L-BFGS
    opt_l = torch.optim.LBFGS(model.parameters(), lr=1.0,
                               line_search_fn='strong_wolfe')
    t1 = time.time()
    for it in range(LBFGS_ITERS):
        def closure_l():
            opt_l.zero_grad()
            loss = compute_ns2d_loss(model, train)
            loss.backward()
            closure_l.val = loss.item()
            return loss
        opt_l.step(closure_l)
    lbfgs_time = time.time() - t1

    final_rrmse = eval_pressure(model, test)
    print(f"  L-BFGS done ({lbfgs_time:.0f}s), rRMSE={final_rrmse:.2e}")
    print(f"  Total: {adam_time + lbfgs_time:.0f}s")

    history.append((ADAM_EPOCHS + LBFGS_ITERS, closure_l.val, final_rrmse))

    results = dict(
        lr=lr,
        n_train=N_TRAIN,
        adam_epochs=ADAM_EPOCHS,
        adam_beta2=ADAM_BETA2,
        lbfgs_iters=LBFGS_ITERS,
        seed=SEED,
        history=history,
        training_time=adam_time + lbfgs_time,
        adam_time=adam_time,
        lbfgs_time=lbfgs_time,
        adam_final_rrmse=adam_rrmse,
        final_rrmse=final_rrmse,
    )
    return model, results
