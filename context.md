# Project Context & Research Ledger: Euler-Lagrange Neural Optimizer

> **Persistent Epistemic State & Session Ledger**  
> *Maintained autonomously after every verified research milestone. Single source of mathematical, architectural, and empirical truth.*

---

## 1. Executive Publication & Portfolio State

- **Preprint Title:** *Variational Parameter Dynamics: The Courant-Friedrichs-Lewy Depth Catastrophe and Relativistic Momentum in Neural Networks*
- **Author:** **Shubham Barmase** (`barmase32@gmail.com`)
- **Official CERN Zenodo Version DOI:** [`10.5281/zenodo.24011081`](https://doi.org/10.5281/zenodo.24011081)
- **CERN Zenodo Concept DOI:** [`10.5281/zenodo.24011080`](https://doi.org/10.5281/zenodo.24011080)
- **Repository:** [`https://github.com/shu195/euler-lagrange-neural-optimizer`](https://github.com/shu195/euler-lagrange-neural-optimizer)
- **Interactive 60 FPS Web Laboratory:** [`https://shu195.github.io/euler-lagrange-neural-optimizer/`](https://shu195.github.io/euler-lagrange-neural-optimizer/)
- **Master GitHub Profile:** [`https://github.com/shu195`](https://github.com/shu195)
- **Zenodo Publication Trilogy:**
  1. Paper 1 (Sep 29, 2026): *Deterministic Spatial Planning on Hypercomplex Manifolds* (DOI: [`10.5281/zenodo.23032915`](https://doi.org/10.5281/zenodo.23032915))
  2. Paper 2 (Oct 16, 2026): *Deterministic 4D Spacetime Geodesic Planning on Hypercomplex Toroidal Manifolds* (DOI: [`10.5281/zenodo.23423238`](https://doi.org/10.5281/zenodo.23423238))
  3. Paper 3 (Nov 06, 2026): *Variational Parameter Dynamics & CFL Depth Catastrophe* (DOI: [`10.5281/zenodo.24011081`](https://doi.org/10.5281/zenodo.24011081))

---

## 2. Core Mathematical Discoveries & Exact Invariants

### 2.1 The Extended Euler-Lagrange Principle
Parameter trajectories $\theta(t) \in \mathbb{R}^D$ are governed by Hamilton's stationary action $S[\theta] = \int (T - V) \, dt$ with Rayleigh dissipation $\mathcal{R}(\dot{\theta}) = \frac{1}{2}\gamma(t)\|\dot{\theta}\|^2$:
$$\frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{\theta}}\right) - \frac{\partial \mathcal{L}}{\partial \theta} = -\gamma(t)\dot{\theta}$$
Where potential energy $V(\theta) \equiv L_{\text{loss}}(\theta)$.

### 2.2 Kinetic Geometries
1. **Euclidean (Polyak Heavy Ball):** $T = \frac{1}{2}m\|\dot{\theta}\|^2 \implies m\ddot{\theta} + \gamma\dot{\theta} + \nabla L = 0$.
2. **Bregman (Nesterov Acceleration):** Critical asymptotic damping $\gamma(t) = \frac{3}{t} \implies \mathcal{O}(1/t^2)$ convergence.
3. **Relativistic (Lorentz Clamping):**
   $$T_{\text{rel}} = -mc^2\sqrt{1 - \frac{\|\dot{\theta}\|^2}{c^2}} \implies p = \frac{m\dot{\theta}}{\sqrt{1 - \|\dot{\theta}\|^2/c^2}}$$
   $$\dot{\theta} = \frac{p}{\sqrt{m^2 + \|p\|^2/c^2}} \implies \|\dot{\theta}\| < c \quad \forall t \ge 0$$
   Eliminates gradient explosion and wall reflection without manual thresholding.
4. **Riemannian (Natural Gradient):** $T = \frac{1}{2}\dot{\theta}^T G(\theta)\dot{\theta} \implies G\ddot{\theta} + \Gamma(\dot{\theta}, \dot{\theta}) + \gamma\dot{\theta} + \nabla L = 0$.

### 2.3 The Parameter-Space CFL Depth Catastrophe
- **Hessian Block Hierarchy:** Hierarchical feedforward depth acts as an exponential eigenvalue multiplier:
  $$\frac{\lambda_{\max}(H_{LL})}{\lambda_{\min}(H_{11})} \sim \mathcal{O}(e^{2L})$$
- **Modal CFL Ceiling:** The stiffest mode $\lambda_{\max}$ dictates the maximum stable integration time-step:
  $$\Delta t < \frac{2\sqrt{m}}{\sqrt{\lambda_{\max}}}$$
- **Mode Freezing:** The slowest mode requires $N_{\text{cycle}} \approx \pi\sqrt{\kappa} \approx 1,000$ steps per single oscillation when $\kappa \sim 10^5$, freezing early layers under isotropic physics.
- **The Epistemic Role of Adam:** Adaptive optimizers (Adam) are crude diagonal approximations to a Riemannian metric tensor designed specifically to counteract this artificial feedforward depth stiffness.

### 2.4 The Non-Layered Spectral Remedy
- **Continuous Wave Interference Fields (0 Hidden Layers):**
  $$f(x) = \sigma\left( \sum_{k=1}^K A_k \cos(k \cdot x + \phi_k) + b \right)$$
- Compresses the Hessian condition number by **$6,140\times$** ($\kappa \approx 23.1$ vs $1.42 \times 10^5$), allowing pure un-preconditioned isotropic symplectic mechanics to converge directly.

---

## 3. Empirical Ground Truth Baselines

### 3.1 Rosenbrock Ravine Optimization (1,500 Steps, $(-1.5, 2.0) \to (1.0, 1.0)$)
| Optimizer | Final Loss | Target Distance | Physical Inductive Bias |
| :--- | :--- | :--- | :--- |
| **Relativistic Euler-Lagrange** | $\mathbf{1.669 \times 10^{-20}}$ | $\mathbf{4.08 \times 10^{-10}}$ | **Exact Lorentz Speed Ceiling ($v < c$)** |
| Polyak Heavy Ball | $9.638 \times 10^{-14}$ | $3.10 \times 10^{-7}$ | Newtonian Inertia (Wall Oscillations) |
| Störmer-Verlet Symplectic | $1.186 \times 10^{-11}$ | $2.23 \times 10^{-6}$ | Strict Area Preserving ($\det(J) \equiv 1.0$) |
| Bregman E-L ($3/t$) | $3.242 \times 10^{-4}$ | $0.0385$ | Critical Time-Varying Dissipation |
| Standard SGD | $1.954 \times 10^{-1}$ | $0.8124$ | Aristotelian Zero-Inertia (Stuck in Ravine) |

### 3.2 Non-Linear Concentric Rings Architecture Benchmark (400 Epochs, Isotropic $m=1.0$)
| Architecture | Isotropic E-L Loss | Adam Loss | Hessian Condition Number $\kappa$ |
| :--- | :--- | :--- | :--- |
| **Wave Field (0-Layer)** | $\mathbf{0.076311}$ | $0.033675$ | $\mathbf{2.31 \times 10^1}$ (Compressed by $6,140\times$) |
| CTRNN Graph | $0.373615$ | $0.004488$ | $4.83 \times 10^2$ |
| Layered MLP (Depth 3) | $0.628461$ | $0.016529$ | $1.42 \times 10^5$ (CFL Catastrophe) |

### 3.3 Symplectic Phase-Space Volume Preservation
- **Semi-Implicit / Verlet:** $\det(J) \equiv 1.0$, bounded shadow Hamiltonian energy drift ($\Delta E < 10^{-2}$).
- **Explicit Forward Euler:** $\det(J) > 1.0$, exponential energy blowup ($\Delta E \to \infty$) within 200 steps.

---

## 4. Repository Codebase Map

```
euler-lagrange-neural-optimizer/
├── assets/
│   └── index.html                # Standalone backup visualizer
├── benchmarks/
│   ├── benchmark_architectures.py# Concentric rings & Hessian eigenspectrum
│   ├── benchmark_rosenbrock.py   # Rosenbrock trajectory benchmark
│   └── benchmark_symplectic_energy.py # Phase space & shadow Hamiltonian bounds
├── data/
│   └── rosenbrock_trajectories.npz # Exact trajectory coordinate arrays
├── docs/
│   └── index.html                # GitHub Pages live 60 FPS simulator (HiDPI, sliders, charts)
├── figures/
│   ├── figure1_rosenbrock_trajectories_and_loss.png
│   ├── figure2_layered_vs_nonlayered_dynamics.png
│   └── figure3_symplectic_vs_explicit_hamiltonian.png
├── paper/
│   ├── paper.tex                 # LaTeX preprint source (Shubham Barmase only)
│   ├── paper.pdf                 # Compiled 5-page preprint (2,406,110 bytes)
│   └── references.bib            # BibTeX references
├── src/
│   ├── models/
│   │   ├── continuous_graph.py   # Continuous-time recurrent neural network
│   │   ├── continuous_wave_field.py # Spatial wave interference field
│   │   └── layered_mlp.py        # Baseline deep feedforward network
│   └── optimizers/
│       ├── base.py               # Abstract physical optimizer interface
│       ├── bregman.py            # Bregman Lagrangian with gamma(t)=3/t
│       ├── heavy_ball.py         # Newtonian Euclidean momentum
│       ├── relativistic.py       # Lorentz velocity-clamped Euler-Lagrange
│       ├── riemannian_adaptive.py# Diagonal metric tensor adaptive flow
│       └── symplectic_verlet.py  # Störmer-Verlet symplectic integrator
├── tests/
│   └── test_optimizers.py        # PyTest suite verifying all physical integrators
├── context.md                    # This persistent ledger
├── pyproject.toml                # Build configuration
└── README.md                     # Master GitHub repo documentation with Zenodo DOI badge
```

---

## 5. Milestone & Session Changelog

- **2026-11-06 (Session 1):** Conceived and derived Variational Parameter Dynamics. Formulated Lorentz relativistic velocity clamping and parameter-space CFL depth catastrophe. Proved $6,140\times$ condition compression with continuous wave fields.
- **2026-11-06 (Session 2):** Implemented complete PyTorch package and test suite. Generated 300 DPI publication figures. Built interactive 60 FPS HTML5 canvas laboratory.
- **2026-11-06 (Session 3):** Wrote and compiled 5-page academic preprint in LaTeX (`paper.pdf`). Visually audited all 5 rendered pages at 150 DPI, fixing table widths, column breaks, float locations, and heading hyphenations. Removed AI co-authorship; set author strictly to **Shubham Barmase**.
- **2026-11-06 (Session 4):** Published preprint to **CERN Zenodo** (`10.5281/zenodo.24011081`). Added official DOI badges to `README.md` and `docs/index.html`. Created master GitHub profile repo (`shu195/shu195`). Latched all 3 trilogy papers into Geodesic Memory Graph (`total nodes: 359`, weight $w = 4.208$).
- **2026-11-07 (Session 5):** Initialized `context.md` persistent state ledger for cross-session deterministic recall.
