import torch
import torch.nn as nn
import numpy as np

class ContinuousGraph(nn.Module):
    """
    Non-Layered Continuous-Time Recurrent Graph (CTRNN / Coupled Reservoir).
    All neurons exist in a homogeneous topological manifold:
        \\dot{h}(t) = -h(t) + W_{rec} \\tanh(h(t)) + W_{in} x
    Hessian spectrum is bounded by the spectral radius \\rho(W_{rec}) ~ 1,
    eliminating the depth-induced CFL stiffness bottleneck.
    """
    def __init__(self, in_features=2, n_nodes=48, out_features=1, dt=0.4, steps=6):
        super().__init__()
        self.n_nodes = n_nodes
        self.dt = dt
        self.steps = steps

        self.W_in = nn.Parameter(torch.randn(in_features, n_nodes) * 0.3)
        self.W_rec = nn.Parameter(torch.randn(n_nodes, n_nodes) * (0.8 / np.sqrt(n_nodes)))
        self.bias = nn.Parameter(torch.zeros(n_nodes))
        self.W_out = nn.Parameter(torch.randn(n_nodes, out_features) * 0.3)
        self.b_out = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        h = torch.zeros(x.shape[0], self.n_nodes, device=x.device, dtype=x.dtype)
        for _ in range(self.steps):
            act = torch.tanh(x @ self.W_in + h @ self.W_rec + self.bias)
            h = (1.0 - self.dt) * h + self.dt * act
        return torch.sigmoid(h @ self.W_out + self.b_out)
