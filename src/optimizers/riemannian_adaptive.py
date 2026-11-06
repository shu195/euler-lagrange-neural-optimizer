import torch
from .base import LagrangianOptimizer

class RiemannianAdaptiveLagrangian(LagrangianOptimizer):
    """
    Riemannian Symplectic Euler-Lagrange with Adaptive Diagonal Metric Tensor G(\\theta):
        T(\\dot{\\theta}) = \\frac{1}{2} \\dot{\\theta}^T G(\\theta) \\dot{\\theta}
    Solves depth-induced Hessian ill-conditioning on layered neural networks by matching
    the diagonal Riemannian curvature tensor G_{ii} = \\sqrt{E[g_i^2]} + \\epsilon.
    """
    def __init__(self, params, lr=1e-2, dt=0.1, gamma=1.8, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, dt=dt, gamma=gamma, beta2=beta2, eps=eps, step_count=0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            dt = group['dt']
            gamma = group['gamma']
            beta2 = group['beta2']
            eps = group['eps']
            group['step_count'] += 1
            step = group['step_count']
            bias_corr = 1.0 - beta2 ** step
            decay = max(0.0, 1.0 - gamma * dt)

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(p)
                    state['metric_diag'] = torch.zeros_like(p)

                v = state['velocity']
                G = state['metric_diag']
                grad = p.grad

                # Update Riemannian metric tensor
                G.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                G_hat = G / bias_corr
                inv_mass_scale = 1.0 / (torch.sqrt(G_hat) + eps)

                # Preconditioned geodesic force
                force = - grad * inv_mass_scale

                # Semi-implicit symplectic step
                v.mul_(decay).add_(force, alpha=dt)
                p.add_(v, alpha=lr)

        return loss
