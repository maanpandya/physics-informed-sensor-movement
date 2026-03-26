# Gaussian Processes for Sensor Repositioning in PDE-Driven Systems

![ICASSP 2026](https://img.shields.io/badge/ICASSP-2026-blue?style=flat-square)
![IEEE](https://img.shields.io/badge/IEEE-Signal%20Processing-lightgrey?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Maan Pandya · Bianca Giovanardi · Raj Thilak Rajan**  
*Faculty of Aerospace Engineering, Delft University of Technology*  
*Raj Thilak Rajan also: Faculty of EEMCS, Delft University of Technology*

[maanp@stanford.edu](mailto:maanp@stanford.edu) · [b.giovanardi@tudelft.nl](mailto:b.giovanardi@tudelft.nl) · [r.t.rajan@tudelft.nl](mailto:r.t.rajan@tudelft.nl)

---

> **Paper:** *Gaussian Processes for Sensor Repositioning in PDE-Driven Systems*  
> IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026

<details>
<summary>BibTeX Citation</summary>

```bibtex
@inproceedings{pandya2026gpsensors,
  title     = {Gaussian Processes for Sensor Repositioning in {PDE}-Driven Systems},
  author    = {Pandya, Maan and Giovanardi, Bianca and Rajan, Raj Thilak},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}
```

</details>

---

## Abstract

Field estimation in spatio-temporally evolving environments remains challenging, particularly when limited sensor resources must capture dynamic features while contending with modeling errors and measurement noise e.g., in environmental monitoring using aerial vehicles, where system dynamics interact with practical sensing limitations.  In this work, we consider a scenario where a network of mobile sensor nodes measure an advection-diffusion field, where the sensor locations can be dynamically optimized based on PDE residuals e.g., sensors on-board drones. Our novel two-stage framework strategically integrates Gaussian Process regression with PDE constraints. An initial inference stage estimates key parameters (e.g., advection velocity, diffusion coefficient) through stationary sensor measurements and finite-difference derivative approximations, while a subsequent mobility stage employs forward-Euler time-stepping to dynamically relocate the sensors toward regions of high PDE residual. Simulations based on a 2D advection-diffusion field experiment reveals upto an order magnitude improvement in field reconstruction error, as compared to information theoretic deployments. We conclude with future directions of extending our work and suggest applications.

---

## Overview

This repository implements a novel two-stage, physics-informed framework for adaptive sensor repositioning in spatio-temporally evolving fields. The method targets scenarios where mobile sensors (e.g., drones) must reconstruct a field governed by a known PDE with *unknown* parameters — such as an advection-diffusion equation.

**Stage I — Calibration (Inference):** Sensors remain stationary while a Gaussian Process (GP) is fitted to measurements at each time step. Spatial derivatives are computed analytically from the GP posterior mean, and the unknown PDE parameters (advection velocity **v**, diffusion coefficient *D*) are recovered via least-squares regression over the accumulated PDE residuals.

**Stage II — Repositioning:** Using the estimated parameters, sensors are dynamically relocated at each step toward anchor points with the highest Physics-Informed Loss (PIL) — the point-wise squared PDE residual of the GP field estimate. A forward-Euler prediction step enables *proactive* movement by anticipating the next-step residual field before moving.

Three repositioning policies are provided:

| Method | Description |
|--------|-------------|
| **Baseline** | Each sensor moves greedily to the highest-PIL anchor within its movement radius. |
| **No Overlap** | Same as Baseline, but each anchor point may only be claimed by one sensor per step. |
| **Box-Constrained** | Same as Baseline, but each sensor is confined to a rectangular box around its initial position, preventing long-term clustering (recommended; see paper). |

The Box-Constrained method achieves at least a **2× improvement** in reconstruction MSE over a Maximum-Variance (information-theoretic) baseline after 100 repositioning steps.

---

## Results

| Method | MSE @ 20 Steps | IF @ 20 | MSE @ 100 Steps | IF @ 100 |
|--------|---------------|---------|----------------|---------|
| Baseline | 1.89 × 10⁻⁵ | 0.75 | 1.28 × 10⁻⁷ | 0.11 |
| No Overlap | 1.89 × 10⁻⁵ | 0.75 | 1.55 × 10⁻⁷ | 0.09 |
| **Box-Constrained** | **1.88 × 10⁻⁵** | **0.76** | **4.93 × 10⁻⁹** | **2.95** |

*IF = Improvement Factor = MSE(MaxVar) / MSE(PIL). IF > 1 means our method outperforms the information-theoretic baseline.*

---

## Repository Structure

```
├── calibrationphase.py        # Stage I: sensor initialization, field observation, GP fitting, calibration loop
├── coefficientestimation.py   # Least-squares PDE coefficient estimation (v_x, v_y, D)
├── gpmodel.py                 # GP model (ExactGP + RBF kernel), PIL computation, finite-difference derivatives
├── completesimulation.py      # End-to-end two-stage simulation for a single repositioning policy
├── comparison_simulation.py   # Side-by-side comparison: Box-Constrained PIL vs. Max-Variance baseline
├── requirements.txt           # Python dependencies
└── outputs/                   # Generated plots and animations (created automatically at runtime)
```

---

## Core Components

### GP Model (`gpmodel.py`)

The `GPModel` class wraps GPyTorch's `ExactGP` with a Scaled RBF (Squared Exponential) kernel. Hyperparameters are trained by maximising the exact marginal log-likelihood via Adam.

Spatial derivatives of the GP posterior mean are computed via **central finite differences** at perturbed query points. This yields the gradient vector [∂u/∂x, ∂u/∂y] and a Laplacian estimate ∇²u used directly in the PDE residual.

The module-level `calculate_physics_informed_loss` function computes the PIL at any set of evaluation points given estimated PDE parameters:

```
PIL(ξ) = ( ∂u/∂t + v̂·∇u − D̂∇²u )²
```

---

### Calibration Phase (`calibrationphase.py`)

Implements Stage I of Algorithm 1 in the paper:

1. Initializes *M* sensors on a uniform grid over the domain Ω.
2. At each of the *N′* inference time steps, observes the field at the (stationary) sensor locations and fits a GP.
3. Computes GP posterior derivatives (gradient, Laplacian) at the anchor-point grid.
4. After all *N′* steps, calls `estimate_coefficients` to recover **v̂** and **D̂**.

Also exposes `field_function` — the ground-truth analytical advection-diffusion field used throughout the simulation:

```
u(x, y, t) = sin(x − v_x t) · cos(y − v_y t) · exp(−D t)
```

with true parameters v_x = 2.0, v_y = 1.0, D = 0.5.

---

### Coefficient Estimation (`coefficientestimation.py`)

Given GP-estimated field derivatives at anchor points across *N′* time steps, recovers the unknown PDE parameters by solving a least-squares problem (Eq. 9 in the paper):

```
(v̂_x, v̂_y, D̂) = argmin  Σ_n Σ_j  R_n(ξ_j)²
```

where the PDE residual is:

```
R_n(ξ_j) = ∂u_n/∂t (ξ_j) + v·∇u_n(ξ_j) − D∇²u_n(ξ_j)
```

Temporal derivatives are approximated via a **central difference scheme** (Eq. 7). The bounds-constrained SciPy `least_squares` solver is used to ensure physically meaningful (positive) estimates.

---

### Complete Simulation (`completesimulation.py`)

End-to-end driver for the two-stage framework. Key user-facing parameters are grouped at the top of the file:

```python
REPOSITIONING_METHOD = "Constrained"   # "Baseline" | "No Overlap" | "Constrained"
NUM_REPOSITIONING_STEPS = 100
MAX_MOVE_DIST = 0.1                     # Maximum per-step movement radius r_max
BOX_HALF_WIDTH  = 0.5                   # Box constraint half-width (Constrained only)
BOX_HALF_HEIGHT = 0.5                   # Box constraint half-height (Constrained only)
COMPARE_STEPWISE = True                 # Plot step-by-step IF vs. static sensors
```

Generates:
- Field snapshot + absolute error plots at step 100.
- Sensor trajectory plot over all repositioning steps.
- Stepwise Improvement Factor plot (IF vs. static sensors).
- Animated GIF of the predicted field and sensor movement.

---

### Comparison Simulation (`comparison_simulation.py`)

Runs **Box-Constrained PIL** and **Maximum-Variance** side-by-side from a shared Stage I calibration, then compares them directly. Generates:
- Improvement Factor over time (replicates Fig. 2 in the paper).
- MSE-over-time comparison plot.
- Final-state field snapshots for both methods.
- Side-by-side sensor trajectory plots.

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Spatial domain Ω | [0, 10] × [0, 5] |
| Time step Δt | 0.1 |
| Inference steps N′ | 10 |
| Number of sensors M | 56 |
| Anchor points | 200 × 100 grid (20 000 total) |
| Max movement r_max (Box-Constrained) | 0.25 |
| Max movement r_max (Baseline / No Overlap) | 0.10 |
| True advection velocity **v** | [2.0, 1.0] |
| True diffusion coefficient D | 0.5 |
| Estimated **v** | [1.944, 0.982] (error ≈ [2.8%, 1.8%]) |
| Estimated D | 0.523 (error ≈ 4.7%) |

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch, GPyTorch, NumPy, SciPy, Matplotlib (Python 3.10+)

### Running the Full Simulation

```bash
# Two-stage simulation for a single repositioning policy
python completesimulation.py
```

Select the policy by setting `REPOSITIONING_METHOD` near the top of the file to `"Baseline"`, `"No Overlap"`, or `"Constrained"`.

### Running the Comparison Against Max-Variance

```bash
# Box-Constrained PIL vs. information-theoretic (MaxVar) baseline
python comparison_simulation.py
```

### Output Location

All figures and animations are saved to `outputs/` (created automatically).

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/<method>_snapshot_100_step.png` | Predicted field + absolute error at step 100 |
| `outputs/<method>_sensor_trajectories.png` | Sensor movement trajectories over all steps |
| `outputs/<method>_improvement_factor.png` | Stepwise IF vs. static sensors |
| `outputs/<method>_animation.gif` | Animated predicted vs. exact field |
| `outputs/Comparison_ImprovementFactor.pdf` | PIL vs. MaxVar IF plot (Fig. 2 in paper) |
| `outputs/Comparison_MSE_PIL_vs_Variance.png` | MSE-over-time comparison |
| `outputs/Comparison_FinalState_PIL_vs_Variance.png` | Final-state field snapshots |
| `outputs/Comparison_Trajectories.pdf` | Side-by-side trajectory comparison |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

This work is partially funded by the EU-HORIZON-KDT-JU-2023-2-RIA under grant agreement No. 101139996, the ShapeFuture project.
