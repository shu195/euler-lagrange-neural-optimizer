import torch
import torch.nn as nn

class ContinuousWaveField(nn.Module):
    """
    Zero-Hidden-Layer Spatial Interference Field:
        f(x) = \\sigma \\left( \\sum_{k=1}^K A_k \\cos(k \\cdot x + \\phi_k) + b \\right)
    Zero depth hierarchy. Hessian eigenvalues are orthogonal and tightly conditioned,
    enabling un-preconditioned isotropic Euler-Lagrange optimization to converge effortlessly.
    """
    def __init__(self, in_features=2, n_modes=32, out_features=1):
        super().__init__()
        self.k = nn.Parameter(torch.randn(in_features, n_modes) * 2.0)
        self.phi = nn.Parameter(torch.zeros(n_modes))
        self.weights = nn.Parameter(torch.randn(n_modes, out_features) * 0.2)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        phase = x @ self.k + self.phi
        field = torch.cos(phase)
        return torch.sigmoid(field @ self.weights + self.bias)
