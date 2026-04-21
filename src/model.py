"""
Vanilla PINN model: MLP with configurable depth, width, and activation.
"""
import torch
import torch.nn as nn
import copy


# ── Activation functions ─────────────────────────────────────────────
class SinAct(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

ACTIVATIONS = {
    'sin': SinAct,
    'tanh': nn.Tanh,
    'swish': Swish,
}


# ── Weight initialization ───────────────────────────────────────────
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


# ── PINNs (vanilla MLP) ─────────────────────────────────────────────
class PINNs(nn.Module):
    """
    Physics-Informed Neural Network (vanilla MLP).

    Args:
        in_dim:    Input dimension (2 for scalar PDEs, 3 for NS2D).
        hidden_dim: Width of hidden layers.
        out_dim:   Output dimension (1 for scalar PDEs, 2 for NS2D).
        num_layer: Total number of layers (including input and output).
                   num_layer=2 means input->output, num_layer=5 means 3 hidden layers.
        activation: One of 'sin', 'tanh', 'swish'.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, activation='tanh'):
        super().__init__()
        act = ACTIVATIONS[activation]()
        layers = [nn.Linear(in_dim, hidden_dim), copy.deepcopy(act)]
        for _ in range(num_layer - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), copy.deepcopy(act)]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self._output_transform = None

    def apply_output_transform(self, transform):
        self._output_transform = transform

    def forward(self, x):
        raw = self.net(x)
        if self._output_transform is not None:
            return self._output_transform(x, raw)
        return raw


def create_model(depth, width, activation, in_dim=2, out_dim=1, device='cpu'):
    """Create and initialize a PINNs model."""
    model = PINNs(in_dim=in_dim, hidden_dim=width, out_dim=out_dim,
                  num_layer=depth, activation=activation)
    model.apply(init_weights)
    return model.to(device)
