"""
PDE definitions for 13 scalar benchmark problems.

Each PDE class provides:
  - domain:          [[x_min, x_max], [t_min, t_max]]
  - compute_losses:  (model, points, b1-b4, pts_test, la_bc) -> (loss, metrics)
  - true_solution:   analytical solution (or reference-based evaluation)
  - display_name:    human-readable name for plots
"""
import torch
import numpy as np
import math
import os
import scipy.io

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demos')


# ══════════════════════════════════════════════════════════════════════
# Data Generation
# ══════════════════════════════════════════════════════════════════════
def get_data_2d(num_x, num_y, domain, device):
    """Generate uniform grid of collocation + boundary points.

    Returns:
        points: (num_x*num_y, 2)  interior collocation points
        b1: (num_y, 2)  left   boundary (x = x_min)
        b2: (num_y, 2)  right  boundary (x = x_max)
        b3: (num_x, 2)  bottom boundary (t = t_min)
        b4: (num_x, 2)  top    boundary (t = t_max)
    """
    x = torch.linspace(domain[0][0], domain[0][1], num_x, device=device)
    y = torch.linspace(domain[1][0], domain[1][1], num_y, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)

    b1 = torch.stack([torch.full((num_y,), domain[0][0], device=device), y], dim=1)
    b2 = torch.stack([torch.full((num_y,), domain[0][1], device=device), y], dim=1)
    b3 = torch.stack([x, torch.full((num_x,), domain[1][0], device=device)], dim=1)
    b4 = torch.stack([x, torch.full((num_x,), domain[1][1], device=device)], dim=1)
    return points, b1, b2, b3, b4


# ══════════════════════════════════════════════════════════════════════
# Base Class
# ══════════════════════════════════════════════════════════════════════
class PDEBase:
    domain = [[0, 1], [0, 1]]
    display_name = ''
    has_analytical = True   # whether true_solution is implemented
    axis_labels = ('$x$', '$t$')
    field_ticks = None      # colorbar ticks for true/pred fields (None = auto)

    def __init__(self, device='cpu'):
        self.device = device
        self._test_cache = None

    def true_solution(self, x):
        raise NotImplementedError

    def compute_losses(self, model, points, b1, b2, b3, b4, pts_test, la_bc=1.0):
        raise NotImplementedError

    def get_test_data(self, grid_size):
        if self._test_cache is not None:
            return self._test_cache
        pts, _, _, _, _ = get_data_2d(grid_size, grid_size, self.domain, self.device)
        true = self.true_solution(pts)
        self._test_cache = (pts, true)
        return pts, true

    def compute_rrmse(self, model, pts_test, true_test):
        pred = model(pts_test)
        return (torch.norm(pred - true_test) / torch.norm(true_test)).item()


# ══════════════════════════════════════════════════════════════════════
# Convection:  u_t + beta * u_x = 0
# ══════════════════════════════════════════════════════════════════════
class Convection(PDEBase):
    """Periodic BC, IC: u(x,0) = sin(x). Exact: u = sin(x - beta*t)."""
    domain = [[0, 2*math.pi], [0, 1]]
    field_ticks = [-1, 0, 1]  # sin range

    def __init__(self, beta=30, device='cpu'):
        super().__init__(device)
        self.beta = beta
        self.display_name = f'Convection ($\\beta={beta}$)'

    def true_solution(self, x):
        return torch.sin(x[:, 0:1] - self.beta * x[:, 1:2])

    def compute_losses(self, model, points, b1, b2, b3, b4, pts_test, la_bc=1.0):
        pts = points.clone().requires_grad_(True)
        pred = model(pts)
        dy = torch.autograd.grad(pred, pts, torch.ones_like(pred), create_graph=True)[0]
        res = dy[:, 1:2] + self.beta * dy[:, 0:1]
        pde_loss = torch.mean(res ** 2)

        bc_loss = torch.mean((model(b1) - model(b2)) ** 2)
        ic_loss = torch.mean((model(b3) - self.true_solution(b3)) ** 2)
        rrmse = self.compute_rrmse(model, pts_test, self.true_solution(pts_test))

        total = pde_loss + la_bc * bc_loss + ic_loss
        return total, dict(pde=pde_loss.item(), bc=bc_loss.item(),
                           ic=ic_loss.item(), rrmse=rrmse)


# ══════════════════════════════════════════════════════════════════════
# Reaction:  u_t = rho * u * (1 - u)
# ══════════════════════════════════════════════════════════════════════
class Reaction(PDEBase):
    """Periodic BC, Gaussian IC. Exact: logistic growth."""
    domain = [[0, 2*math.pi], [0, 1]]

    def __init__(self, rho=5, device='cpu'):
        super().__init__(device)
        self.rho = rho
        self.display_name = f'Reaction ($\\rho={rho}$)'

    def _gaussian(self, x):
        return torch.exp(-(x[:, 0:1] - math.pi) ** 2 / (2 * (math.pi / 4) ** 2))

    def true_solution(self, x):
        h = self._gaussian(x)
        exp_rt = torch.exp(self.rho * x[:, 1:2])
        return h * exp_rt / (h * exp_rt + 1 - h)

    def compute_losses(self, model, points, b1, b2, b3, b4, pts_test, la_bc=1.0):
        pts = points.clone().requires_grad_(True)
        pred = model(pts)
        dy = torch.autograd.grad(pred, pts, torch.ones_like(pred), create_graph=True)[0]
        res = dy[:, 1:2] - self.rho * pred * (1 - pred)
        pde_loss = torch.mean(res ** 2)

        bc_loss = torch.mean((model(b1) - model(b2)) ** 2)
        ic_loss = torch.mean((model(b3) - self._gaussian(b3)) ** 2)
        rrmse = self.compute_rrmse(model, pts_test, self.true_solution(pts_test))

        total = pde_loss + la_bc * bc_loss + ic_loss
        return total, dict(pde=pde_loss.item(), bc=bc_loss.item(),
                           ic=ic_loss.item(), rrmse=rrmse)


# ══════════════════════════════════════════════════════════════════════
# Diffusion:  u_t - u_xx + source = 0
# ══════════════════════════════════════════════════════════════════════
class Diffusion(PDEBase):
    """Dirichlet BC (zero), analytical IC. Exact: sin(pi*x)*exp(-t)."""
    domain = [[-1, 1], [0, 1]]
    display_name = 'Diffusion'

    def true_solution(self, x):
        return torch.sin(math.pi * x[:, 0:1]) * torch.exp(-x[:, 1:2])

    def compute_losses(self, model, points, b1, b2, b3, b4, pts_test, la_bc=1.0):
        pts = points.clone().requires_grad_(True)
        pred = model(pts)
        dy = torch.autograd.grad(pred, pts, torch.ones_like(pred), create_graph=True)[0]
        dy_t = dy[:, 1:2]
        dy_x = dy[:, 0:1]
        dy_xx = torch.autograd.grad(dy_x, pts, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
        source = torch.exp(-pts[:, 1:2]) * (
            torch.sin(math.pi * pts[:, 0:1]) -
            math.pi ** 2 * torch.sin(math.pi * pts[:, 0:1]))
        res = dy_t - dy_xx + source
        pde_loss = torch.mean(res ** 2)

        bc_loss = torch.mean(model(b1) ** 2) + torch.mean(model(b2) ** 2)
        ic_loss = torch.mean((model(b3) - self.true_solution(b3)) ** 2)
        rrmse = self.compute_rrmse(model, pts_test, self.true_solution(pts_test))

        total = pde_loss + la_bc * bc_loss + ic_loss
        return total, dict(pde=pde_loss.item(), bc=bc_loss.item(),
                           ic=ic_loss.item(), rrmse=rrmse)


# ══════════════════════════════════════════════════════════════════════
# Wave:  u_tt - 4 * u_xx = 0
# ══════════════════════════════════════════════════════════════════════
class Wave(PDEBase):
    """Dirichlet BC (zero), IC: u(x,0) + u_t(x,0) = 0. Two-mode solution."""
    domain = [[0, 1], [0, 1]]

    def __init__(self, beta=3, device='cpu'):
        super().__init__(device)
        self.beta = beta
        self.display_name = f'Wave ($\\alpha=4, \\beta={beta}$)'

    def true_solution(self, x):
        return (torch.sin(math.pi * x[:, 0:1]) *
                torch.cos(2 * math.pi * x[:, 1:2]) +
                0.5 * torch.sin(self.beta * math.pi * x[:, 0:1]) *
                torch.cos(self.beta * 2 * math.pi * x[:, 1:2]))

    def compute_losses(self, model, points, b1, b2, b3, b4, pts_test, la_bc=1.0):
        pts = points.clone().requires_grad_(True)
        pred = model(pts)
        dy = torch.autograd.grad(pred, pts, torch.ones_like(pred), create_graph=True)[0]
        dy_x, dy_t = dy[:, 0:1], dy[:, 1:2]
        dy_xx = torch.autograd.grad(dy_x, pts, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
        dy_tt = torch.autograd.grad(dy_t, pts, torch.ones_like(dy_t), create_graph=True)[0][:, 1:2]
        res = dy_tt - 4 * dy_xx
        pde_loss = torch.mean(res ** 2)

        bc_loss = torch.mean(model(b1) ** 2) + torch.mean(model(b2) ** 2)

        # IC: u(x,0) and u_t(x,0) = 0
        b3c = b3.clone().requires_grad_(True)
        pred_ic = model(b3c)
        dy_ic = torch.autograd.grad(pred_ic, b3c, torch.ones_like(pred_ic), create_graph=True)[0]
        true_ic = (torch.sin(math.pi * b3c[:, 0:1]) +
                   0.5 * torch.sin(self.beta * math.pi * b3c[:, 0:1]))
        ic_loss = torch.mean(dy_ic[:, 1:2] ** 2) + torch.mean((pred_ic - true_ic) ** 2)

        rrmse = self.compute_rrmse(model, pts_test, self.true_solution(pts_test))

        total = pde_loss + la_bc * bc_loss + ic_loss
        return total, dict(pde=pde_loss.item(), bc=bc_loss.item(),
                           ic=ic_loss.item(), rrmse=rrmse)


# ══════════════════════════════════════════════════════════════════════
# Burgers:  u_t + u * u_x - (0.01/pi) * u_xx = 0
# ══════════════════════════════════════════════════════════════════════
class Burgers(PDEBase):
    """Dirichlet BC (zero), IC: -sin(pi*x). Reference solution from data file."""
    domain = [[-1, 1], [0, 0.99]]
    display_name = 'Burgers'
    has_analytical = False

    def __init__(self, device='cpu'):
        super().__init__(device)
        data = np.load(os.path.join(DATA_DIR, 'Burgers.npz'))
        self.x_ref = data['x']
        self.t_ref = data['t']
        self.u_ref = data['usol']

    def get_test_data(self, grid_size):
        if self._test_cache is not None:
            return self._test_cache
        x_star = self.x_ref.flatten()
        t_star = self.t_ref.flatten()
        X, T = np.meshgrid(x_star, t_star, indexing='ij')
        pts = torch.tensor(np.stack([X.flatten(), T.flatten()], axis=1),
                           dtype=torch.float32, device=self.device)
        true = torch.tensor(self.u_ref.flatten()[:, None],
                            dtype=torch.float32, device=self.device)
        self._test_cache = (pts, true)
        return pts, true

    def compute_losses(self, model, points, b1, b2, b3, b4, pts_test, la_bc=1.0):
        pts = points.clone().requires_grad_(True)
        pred = model(pts)
        dy = torch.autograd.grad(pred, pts, torch.ones_like(pred), create_graph=True)[0]
        dy_x, dy_t = dy[:, 0:1], dy[:, 1:2]
        dy_xx = torch.autograd.grad(dy_x, pts, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
        res = dy_t + pred * dy_x - 0.01 / math.pi * dy_xx
        pde_loss = torch.mean(res ** 2)

        bc_loss = torch.mean(model(b1) ** 2) + torch.mean(model(b2) ** 2)
        ic_val = -torch.sin(math.pi * b3[:, 0:1])
        ic_loss = torch.mean((model(b3) - ic_val) ** 2)

        _, true_test = self.get_test_data(0)
        pred_test = model(pts_test)
        rrmse = (torch.norm(pred_test - true_test) / torch.norm(true_test)).item()

        total = pde_loss + la_bc * bc_loss + ic_loss
        return total, dict(pde=pde_loss.item(), bc=bc_loss.item(),
                           ic=ic_loss.item(), rrmse=rrmse)


# ══════════════════════════════════════════════════════════════════════
# Klein-Gordon:  u_tt - u_xx + u^2 + source = 0
# ══════════════════════════════════════════════════════════════════════
class KleinGordon(PDEBase):
    """Dirichlet BC, IC: u(x,0) = x, u_t(x,0) = 0. Exact: x*cos(t)."""
    domain = [[-1, 1], [0, 10]]
    display_name = 'Klein-Gordon'

    def true_solution(self, x):
        return x[:, 0:1] * torch.cos(x[:, 1:2])

    def compute_losses(self, model, points, b1, b2, b3, b4, pts_test, la_bc=1.0):
        pts = points.clone().requires_grad_(True)
        pred = model(pts)
        dy = torch.autograd.grad(pred, pts, torch.ones_like(pred), create_graph=True)[0]
        dy_x, dy_t = dy[:, 0:1], dy[:, 1:2]
        dy_xx = torch.autograd.grad(dy_x, pts, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
        dy_tt = torch.autograd.grad(dy_t, pts, torch.ones_like(dy_t), create_graph=True)[0][:, 1:2]
        source = pts[:, 0:1] * torch.cos(pts[:, 1:2]) - \
                 pts[:, 0:1] ** 2 * torch.cos(pts[:, 1:2]) ** 2
        res = dy_tt - dy_xx + pred ** 2 + source
        pde_loss = torch.mean(res ** 2)

        bc_loss = (torch.mean((model(b1) + torch.cos(b1[:, 1:2])) ** 2) +
                   torch.mean((model(b2) - torch.cos(b2[:, 1:2])) ** 2))

        # IC: u(x,0) = x, u_t(x,0) = 0
        b3c = b3.clone().requires_grad_(True)
        pred_ic = model(b3c)
        dy_ic = torch.autograd.grad(pred_ic, b3c, torch.ones_like(pred_ic), create_graph=True)[0]
        ic_loss = torch.mean((pred_ic - b3c[:, 0:1]) ** 2) + torch.mean(dy_ic[:, 1:2] ** 2)

        rrmse = self.compute_rrmse(model, pts_test, self.true_solution(pts_test))

        total = pde_loss + la_bc * bc_loss + ic_loss
        return total, dict(pde=pde_loss.item(), bc=bc_loss.item(),
                           ic=ic_loss.item(), rrmse=rrmse)


# ══════════════════════════════════════════════════════════════════════
# Allen-Cahn:  u_t - 0.001*u_xx - 5*(u - u^3) = 0  (hard BC via transform)
# ══════════════════════════════════════════════════════════════════════
class AllenCahn(PDEBase):
    """Output transform enforces IC/BC analytically. Reference data for test."""
    domain = [[-1, 1], [0, 1]]
    display_name = 'Allen-Cahn'
    has_analytical = False

    def __init__(self, device='cpu'):
        super().__init__(device)
        mat = scipy.io.loadmat(os.path.join(DATA_DIR, 'Allen_Cahn.mat'))
        t = mat['t'].flatten()
        x = mat['x'].flatten()
        u = mat['u']
        if u.shape[0] == len(t) and u.shape[1] == len(x):
            u = u.T
        X, T = np.meshgrid(x, t, indexing='ij')
        self._ref_pts = torch.tensor(
            np.stack([X.flatten(), T.flatten()], axis=1),
            dtype=torch.float32, device=device)
        self._ref_u = torch.tensor(
            u.flatten()[:, None], dtype=torch.float32, device=device)

    @staticmethod
    def output_transform(x, y):
        return (x[:, 0:1] ** 2 * torch.cos(math.pi * x[:, 0:1]) +
                x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y)

    def apply_output_transform(self, model):
        model.apply_output_transform(self.output_transform)

    def get_test_data(self, grid_size):
        if self._test_cache is not None:
            return self._test_cache
        self._test_cache = (self._ref_pts, self._ref_u)
        return self._ref_pts, self._ref_u

    def compute_losses(self, model, points, b1, b2, b3, b4, pts_test, la_bc=1.0):
        pts = points.clone().requires_grad_(True)
        pred = model(pts)
        dy = torch.autograd.grad(pred, pts, torch.ones_like(pred), create_graph=True)[0]
        dy_t = dy[:, 1:2]
        dy_x = dy[:, 0:1]
        dy_xx = torch.autograd.grad(dy_x, pts, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
        res = dy_t - 0.001 * dy_xx - 5 * (pred - pred ** 3)
        pde_loss = torch.mean(res ** 2)

        _, true_test = self.get_test_data(0)
        pred_test = model(pts_test)
        rrmse = (torch.norm(pred_test - true_test) / torch.norm(true_test)).item()

        total = pde_loss
        return total, dict(pde=pde_loss.item(), bc=0.0, ic=0.0, rrmse=rrmse)


# ══════════════════════════════════════════════════════════════════════
# Helmholtz:  -u_xx - u_yy - k0^2 * u = f(x,y)  (elliptic, 2D spatial)
# ══════════════════════════════════════════════════════════════════════
class Helmholtz(PDEBase):
    """Dirichlet BC (zero on all 4 boundaries). Exact: sin(k0*x)*sin(k0*y)."""
    domain = [[0, 1], [0, 1]]
    axis_labels = ('$x$', '$y$')

    def __init__(self, n=2, device='cpu'):
        super().__init__(device)
        self.n = n
        self.k0 = 2 * math.pi * n
        self.display_name = f'Helmholtz ($n={n}$)'

    def true_solution(self, x):
        return torch.sin(self.k0 * x[:, 0:1]) * torch.sin(self.k0 * x[:, 1:2])

    def compute_losses(self, model, points, b1, b2, b3, b4, pts_test, la_bc=1.0):
        pts = points.clone().requires_grad_(True)
        pred = model(pts)
        dy = torch.autograd.grad(pred, pts, torch.ones_like(pred), create_graph=True)[0]
        dy_x, dy_y = dy[:, 0:1], dy[:, 1:2]
        dy_xx = torch.autograd.grad(dy_x, pts, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
        dy_yy = torch.autograd.grad(dy_y, pts, torch.ones_like(dy_y), create_graph=True)[0][:, 1:2]
        source = self.k0 ** 2 * torch.sin(self.k0 * pts[:, 0:1]) * \
                 torch.sin(self.k0 * pts[:, 1:2])
        res = -dy_xx - dy_yy - self.k0 ** 2 * pred - source
        pde_loss = torch.mean(res ** 2)

        bc_loss = (torch.mean(model(b1) ** 2) + torch.mean(model(b2) ** 2) +
                   torch.mean(model(b3) ** 2) + torch.mean(model(b4) ** 2))
        rrmse = self.compute_rrmse(model, pts_test, self.true_solution(pts_test))

        total = pde_loss + la_bc * bc_loss
        return total, dict(pde=pde_loss.item(), bc=bc_loss.item(),
                           ic=0.0, rrmse=rrmse)


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════
PDE_REGISTRY = {
    'pde1': lambda dev: Convection(beta=30, device=dev),
    'pde2': lambda dev: Reaction(rho=5, device=dev),
    'pde3': lambda dev: Diffusion(device=dev),
    'pde4': lambda dev: Wave(beta=3, device=dev),
    'pde5': lambda dev: Burgers(device=dev),
    'pde6': lambda dev: KleinGordon(device=dev),
    'pde7': lambda dev: AllenCahn(device=dev),
    'pde8': lambda dev: Helmholtz(n=2, device=dev),
    'conv50': lambda dev: Convection(beta=50, device=dev),
    'conv100': lambda dev: Convection(beta=100, device=dev),
    'react6': lambda dev: Reaction(rho=6, device=dev),
    'wave44': lambda dev: Wave(beta=4, device=dev),
    'helm3': lambda dev: Helmholtz(n=3, device=dev),
}

def create_pde(name, device='cpu'):
    return PDE_REGISTRY[name](device)
