"""
Plotting utilities for PINN prediction fields.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
})

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demos', 'Results')


def _fmt_rrmse(val):
    exp = int(np.floor(np.log10(val)))
    coeff = val / 10**exp
    return f'$\\mathrm{{rRMSE}}={coeff:.1f}\\times 10^{{{exp}}}$'


def plot_field(pde, model, save_name, rrmse=None, nx=300, nt=200, device='cpu'):
    """Plot True | Predicted | Absolute Error for a scalar PDE.

    Args:
        pde:       PDE instance with domain, true_solution, display_name.
        model:     Trained model.
        save_name: Filename stem (saved to figures/).
        rrmse:     If None, computed from the prediction.
        nx, nt:    Grid resolution for visualization.
        device:    Torch device (model should already be on this device).
    """
    model.eval()
    domain = pde.domain
    x = np.linspace(domain[0][0], domain[0][1], nx)
    t = np.linspace(domain[1][0], domain[1][1], nt)
    X, T = np.meshgrid(x, t, indexing='ij')

    pts = torch.tensor(np.stack([X.ravel(), T.ravel()], axis=1),
                       dtype=torch.float32, device=device)
    with torch.no_grad():
        u_pred = model(pts).cpu().numpy().reshape(nx, nt)

    # True solution
    if pde.has_analytical:
        u_true = pde.true_solution(pts).cpu().numpy().reshape(nx, nt)
    else:
        # For reference-based PDEs, evaluate on this grid
        u_true = pde.true_solution(pts).cpu().numpy().reshape(nx, nt) \
            if hasattr(pde, '_interp_true') else None

    if u_true is None:
        # Fall back: plot only predicted field
        _plot_pred_only(X, T, u_pred, pde, save_name, rrmse)
        return

    u_err = np.abs(u_pred - u_true)
    if rrmse is None:
        rrmse = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    vmin = min(u_true.min(), u_pred.min())
    vmax = max(u_true.max(), u_pred.max())
    levels = np.linspace(vmin, vmax, 101)

    def _cb_fmt(x, _):
        if x == 0:
            return '0'
        return f'{x:.2g}'

    # True
    cf0 = axes[0].contourf(X, T, u_true, levels=levels, cmap='RdBu_r')
    axes[0].set_title('True Solution')
    cb0 = plt.colorbar(cf0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.tick_params(labelsize=12)
    cb0.set_ticks([vmin, vmax])
    cb0.ax.yaxis.set_major_formatter(plt.FuncFormatter(_cb_fmt))

    # Predicted
    cf1 = axes[1].contourf(X, T, u_pred, levels=levels, cmap='RdBu_r')
    axes[1].set_title('Predicted')
    cb1 = plt.colorbar(cf1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=12)
    cb1.set_ticks([vmin, vmax])
    cb1.ax.yaxis.set_major_formatter(plt.FuncFormatter(_cb_fmt))

    # Error
    err_max = u_err.max()
    levels_err = np.linspace(0, err_max, 101)
    cf2 = axes[2].contourf(X, T, u_err, levels=levels_err, cmap='hot_r')
    axes[2].set_title('Absolute Error')
    cb2 = plt.colorbar(cf2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=12)
    cb2.set_ticks([0, err_max])
    cb2.ax.yaxis.set_major_formatter(plt.FuncFormatter(_cb_fmt))

    xl, yl = pde.axis_labels
    for ax in axes:
        ax.set_xlabel(xl, fontsize=14)
        ax.set_ylabel(yl, fontsize=14)

    # Pi-ticks for [0, 2*pi] domains
    import math
    if abs(domain[0][1] - 2 * math.pi) < 0.01:
        for ax in axes:
            ax.set_xticks([0, math.pi, 2 * math.pi])
            ax.set_xticklabels(['$0$', '$\\pi$', '$2\\pi$'], fontsize=12)

    fig.tight_layout(pad=0.8)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f'{save_name}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_pred_only(X, T, u_pred, pde, save_name, rrmse):
    """Fallback: plot just the predicted field (for reference-data PDEs)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    levels = np.linspace(u_pred.min(), u_pred.max(), 101)
    cf = ax.contourf(X, T, u_pred, levels=levels, cmap='RdBu_r')
    title = 'Predicted'
    if rrmse is not None:
        title += f' ({_fmt_rrmse(rrmse)})'
    ax.set_title(title)
    xl, yl = pde.axis_labels
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2g}' if x != 0 else '0'))
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f'{save_name}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_ns2d_field(model, save_name, rrmse=None, device='cpu', snap=100):
    """Plot True | Predicted | Error for NS2D pressure at a time snapshot.

    Uses tricontourf for irregular mesh data.
    """
    import scipy.io
    import matplotlib.tri as tri

    data = scipy.io.loadmat(os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'demos', 'cylinder_nektar_wake.mat'))
    X_star = data['X_star']   # (N, 2)
    P_star = data['p_star']   # (N, T)
    t_star = data['t']        # (T, 1)

    x_pts = X_star[:, 0]
    y_pts = X_star[:, 1]
    p_true = P_star[:, snap]

    # Predict
    model.eval()
    t_val = t_star[snap, 0]
    N = X_star.shape[0]
    src = torch.tensor(
        np.column_stack([x_pts, y_pts, np.full(N, t_val)]),
        dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(src).cpu().numpy()
    p_pred = out[:, 1]

    # Optimal constant correction
    C_opt = np.mean(p_true - p_pred)
    p_pred_corr = p_pred + C_opt
    p_err = np.abs(p_true - p_pred_corr)

    if rrmse is None:
        rrmse = np.linalg.norm(p_true - p_pred_corr) / np.linalg.norm(p_true)

    # Triangulation
    triang = tri.Triangulation(x_pts, y_pts)

    def _cb_fmt(x, _):
        if x == 0:
            return '0'
        return f'{x:.2g}'

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    vmin = min(p_true.min(), p_pred_corr.min())
    vmax = max(p_true.max(), p_pred_corr.max())
    levels = np.linspace(vmin, vmax, 101)

    # True
    cf0 = axes[0].tricontourf(triang, p_true, levels=levels, cmap='RdBu_r')
    axes[0].set_title('True Solution')
    cb0 = plt.colorbar(cf0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.tick_params(labelsize=12)
    cb0.set_ticks([vmin, vmax])
    cb0.ax.yaxis.set_major_formatter(plt.FuncFormatter(_cb_fmt))

    # Predicted
    cf1 = axes[1].tricontourf(triang, p_pred_corr, levels=levels, cmap='RdBu_r')
    axes[1].set_title('Predicted')
    cb1 = plt.colorbar(cf1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=12)
    cb1.set_ticks([vmin, vmax])
    cb1.ax.yaxis.set_major_formatter(plt.FuncFormatter(_cb_fmt))

    # Error
    err_max = p_err.max()
    levels_err = np.linspace(0, err_max, 101)
    cf2 = axes[2].tricontourf(triang, p_err, levels=levels_err, cmap='hot_r')
    axes[2].set_title('Absolute Error')
    cb2 = plt.colorbar(cf2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=12)
    cb2.set_ticks([0, err_max])
    cb2.ax.yaxis.set_major_formatter(plt.FuncFormatter(_cb_fmt))

    for ax in axes:
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_aspect('equal')

    fig.tight_layout(pad=0.8)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f'{save_name}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {out}")
