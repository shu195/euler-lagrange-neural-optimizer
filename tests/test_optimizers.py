import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimizers import (
    HeavyBallLagrangian,
    BregmanLagrangian,
    RelativisticLagrangian,
    SymplecticVerlet,
    RiemannianAdaptiveLagrangian,
)
from src.models import LayeredMLP, ContinuousGraph, ContinuousWaveField

class TestLagrangianOptimizers(unittest.TestCase):
    def test_heavy_ball_step(self):
        p = nn.Parameter(torch.tensor([2.0, -3.0], requires_grad=True))
        opt = HeavyBallLagrangian([p], dt=0.05, mass=1.0, gamma=0.5)
        loss = torch.sum(p ** 2)
        loss.backward()
        opt.step()
        self.assertTrue(torch.all(p.abs() < 3.0))

    def test_bregman_step(self):
        p = nn.Parameter(torch.tensor([2.0, -3.0], requires_grad=True))
        opt = BregmanLagrangian([p], dt=0.05, alpha=3.0)
        loss = torch.sum(p ** 2)
        loss.backward()
        opt.step()
        self.assertTrue(torch.all(p.abs() < 3.0))

    def test_relativistic_velocity_bounded(self):
        # Huge gradient to test relativistic speed of light cap
        p = nn.Parameter(torch.tensor([10.0, 10.0], requires_grad=True))
        c_speed = 3.0
        dt = 0.05
        opt = RelativisticLagrangian([p], dt=dt, c=c_speed, mass=1.0, gamma=1.0)
        p.grad = torch.tensor([1e6, 1e6]) # Massive gradient cliff
        p_prev = p.clone()
        opt.step()
        delta_p = torch.norm(p - p_prev).item()
        velocity = delta_p / dt
        # Velocity must never exceed speed of light barrier
        self.assertLessEqual(velocity, c_speed * 1.05)

    def test_symplectic_verlet_step(self):
        p = nn.Parameter(torch.tensor([2.0, -1.0], requires_grad=True))
        opt = SymplecticVerlet([p], dt=0.02, gamma=1.0)
        def closure():
            opt.zero_grad()
            l = torch.sum(p ** 2)
            l.backward()
            return l
        loss = closure()
        opt.step(closure=closure)
        self.assertTrue(torch.all(p.abs() < 3.0))

    def test_riemannian_adaptive_step(self):
        p = nn.Parameter(torch.tensor([1.5, -2.5], requires_grad=True))
        opt = RiemannianAdaptiveLagrangian([p], lr=0.01, dt=0.1)
        loss = torch.sum(p ** 2)
        loss.backward()
        opt.step()
        self.assertTrue(torch.all(p.abs() < 3.0))

    def test_models_forward(self):
        x = torch.randn(10, 2)
        mlp = LayeredMLP(2, 16, 1)
        graph = ContinuousGraph(2, 20, 1)
        wave = ContinuousWaveField(2, 16, 1)

        self.assertEqual(mlp(x).shape, (10, 1))
        self.assertEqual(graph(x).shape, (10, 1))
        self.assertEqual(wave(x).shape, (10, 1))

if __name__ == "__main__":
    unittest.main()
