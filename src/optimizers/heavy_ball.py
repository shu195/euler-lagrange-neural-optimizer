import torch
from .base import LagrangianOptimizer

class HeavyBallLagrangian(LagrangianOptimizer):
    """
    Classical Euler-Lagrange with Constant Rayleigh Dissipation:
        m \\ddot{\\theta} + \\gamma \\dot{\\theta} + \\nabla L(\\theta) = 0
    Discretized via area-preserving Semi-Implicit Symplectic Euler:
        v_{k+1} = (1 - \\gamma dt) v_k - (dt / m) \\nabla L(\\theta_k)
        \\theta_{k+1} = \\theta_k + dt v_{k+1}
    """
    def __init__(self, params, lr=1e-3, dt=0.02, mass=1.0, gamma=0.8):
        defaults = dict(lr=lr, dt=dt, mass=mass, gamma=gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            dt = group['dt']
            mass = group['mass']
            gamma = group['gamma']
            decay = max(0.0, 1.0 - gamma * dt)

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(p)

                v = state['velocity']
                grad = p.grad
                # Symplectic velocity update
                v.mul_(decay).add_(grad, alpha=-(dt / mass))
                # Coordinate update
                p.add_(v, alpha=dt)

        return loss
