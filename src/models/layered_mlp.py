import torch
import torch.nn as nn

class LayeredMLP(nn.Module):
    """
    Standard hierarchical feedforward MLP with depth L.
    Demonstrates exponential block-hierarchical Hessian ill-conditioning:
        kappa = \\lambda_{max} / \\lambda_{min} ~ O(e^{2L})
    """
    def __init__(self, in_features=2, hidden_dim=32, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
