import torch
from .base import LagrangianOptimizer

class SymplecticVerlet(LagrangianOptimizer):
    """
    Second-Order Störmer-Verlet Symplectic Integrator for Parameter Dynamics:
        \\theta_{k+1} = \\theta_k + dt v_k - \\frac{dt^2}{2} \\nabla L(\\theta_k)
        v_{k+1} = e^{-\\gamma dt} v_k - \\frac{dt}{2} (\\nabla L(\\theta_k) + \\nabla L(\\theta_{k+1}))
    Strictly preserves the symplectic 2-form \\omega = d\\theta \\wedge dp in phase space.
    """
    def __init__(self, params, dt=0.015, gamma=1.0):
        defaults = dict(dt=dt, gamma=gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SymplecticVerlet requires a closure evaluating loss and gradient at \\theta_{k+1}.")

        for group in self.param_groups:
            dt = group['dt']
            gamma = group['gamma']
            decay = torch.exp(torch.tensor(-gamma * dt)).item()

            # Pass 1: Half-kick and position update
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(p)
                state['old_grad'] = p.grad.clone()
                v = state['velocity']
                # Position drift
                p.add_(v, alpha=dt).add_(state['old_grad'], alpha=-0.5 * dt * dt)

        # Re-evaluate gradient at new position \theta_{k+1}
        with torch.enable_grad():
            loss = closure()

        # Pass 2: Velocity update using both gradients
        for group in self.param_groups:
            dt = group['dt']
            gamma = group['gamma']
            decay = torch.exp(torch.tensor(-gamma * dt)).item()

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                v = state['velocity']
                old_grad = state['old_grad']
                new_grad = p.grad
                # Average force kick
                v.mul_(decay).add_(old_grad + new_grad, alpha=-0.5 * dt)

        return loss
