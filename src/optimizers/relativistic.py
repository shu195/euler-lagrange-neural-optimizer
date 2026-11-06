import torch
from .base import LagrangianOptimizer

class RelativisticLagrangian(LagrangianOptimizer):
    """
    Relativistic Euler-Lagrange Dynamics with Lorentz-Invariant Velocity Clamping:
        L_{rel} = -m c^2 \\sqrt{1 - ||\\dot{\\theta}||^2 / c^2} - V(\\theta)
    Canonical momentum:
        p = m v / \\sqrt{1 - ||v||^2 / c^2}  =>  v = p / \\sqrt{m^2 + ||p||^2 / c^2}
    Guarantees ||\\dot{\\theta}|| < c strictly, preventing gradient blowup on steep cliffs.
    """
    def __init__(self, params, dt=0.02, c=5.0, mass=1.0, gamma=1.0):
        defaults = dict(dt=dt, c=c, mass=mass, gamma=gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            dt = group['dt']
            c = group['c']
            mass = group['mass']
            gamma = group['gamma']
            c_sq = c * c
            m_sq = mass * mass

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)

                p_mom = state['momentum']
                grad = p.grad

                # Physical velocity from canonical momentum
                p_norm_sq = torch.sum(p_mom * p_mom)
                lorentz_inv = 1.0 / torch.sqrt(m_sq + p_norm_sq / c_sq)
                v = p_mom * lorentz_inv

                # Update canonical momentum with friction: dp/dt = -grad - gamma * v
                p_mom.add_(grad, alpha=-dt).add_(v, alpha=-gamma * dt)

                # Update velocity with post-step momentum
                p_norm_sq_new = torch.sum(p_mom * p_mom)
                lorentz_inv_new = 1.0 / torch.sqrt(m_sq + p_norm_sq_new / c_sq)
                v_new = p_mom * lorentz_inv_new

                # Generalized coordinate update
                p.add_(v_new, alpha=dt)

        return loss
