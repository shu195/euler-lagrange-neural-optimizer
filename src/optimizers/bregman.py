import torch
from .base import LagrangianOptimizer

class BregmanLagrangian(LagrangianOptimizer):
    """
    Bregman Lagrangian Euler-Lagrange with Critical Asymptotic Damping:
        \\ddot{\\theta} + \\frac{\\alpha}{t} \\dot{\\theta} + \\nabla L(\\theta) = 0
    Recovers Su-Boyd-Candes (2014) continuous Nesterov acceleration with rate O(1/t^2).
    """
    def __init__(self, params, dt=0.02, alpha=3.0):
        defaults = dict(dt=dt, alpha=alpha, step_count=0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            dt = group['dt']
            alpha = group['alpha']
            group['step_count'] += 1
            k = group['step_count']
            gamma_t = alpha / (k * dt + 1.0)
            decay = max(0.0, 1.0 - gamma_t * dt)

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(p)

                v = state['velocity']
                v.mul_(decay).add_(p.grad, alpha=-dt)
                p.add_(v, alpha=dt)

        return loss
