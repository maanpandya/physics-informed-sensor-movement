"""
PDE coefficient estimation for the advection-diffusion equation.

Given GP-estimated field derivatives at the anchor points across N' time steps,
the unknown parameters (v_x, v_y, D) are recovered by solving a least-squares
problem over the accumulated PDE residuals (Eq. 9 in the paper).
"""

import torch
import numpy as np
from scipy.optimize import least_squares

def define_anchor_points(domain_limits, num_points_x, num_points_y):
    """
    Defines a 2D grid of anchor points in the spatial domain.
    
    Args:
        domain_limits: Tuple of ((x_min, x_max), (y_min, y_max)) defining 2D spatial domain boundaries.
        num_points_x: Number of grid points along the x dimension.
        num_points_y: Number of grid points along the y dimension.
        
    Returns:
        anchor_points: Tensor of shape (num_points_x * num_points_y, 2),
                       representing a meshgrid flattened into a list of 2D points.
    """
    (x_min, x_max), (y_min, y_max) = domain_limits
    x_coords = torch.linspace(x_min, x_max, num_points_x)
    y_coords = torch.linspace(y_min, y_max, num_points_y)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    anchor_points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    return anchor_points

def finite_difference(u_next, u_prev, time_step_size):
    """Central-difference approximation of ∂u/∂t (Eq. 7 in the paper)."""
    return (u_next - u_prev) / (2.0 * time_step_size)

def least_squares_fit(temporal_derivatives, spatial_gradients, laplacian_values):
    """
    Performs least squares estimation to solve for 2D advection (v_x, v_y) and diffusion (D).
    
    Args:
        temporal_derivatives: Tensor of shape (N,) for PDE time derivative.
        spatial_gradients: Tensor of shape (N,2) for 2D gradients [du/dx, du/dy].
        laplacian_values: Tensor of shape (N,) for Laplacian (d^2u/dx^2 + d^2u/dy^2).
        
    Returns:
        v_x, v_y: Estimated advection velocity components.
        D: Estimated  diffusion coefficient.
    """
    grad_u = spatial_gradients.detach().numpy()   # shape (N, 2)
    laplacian_u = laplacian_values.detach().numpy()  # shape (N,)
    du_dt = temporal_derivatives.detach().numpy()    # shape (N,)
    
    # Define the residual function for least squares: R = du/dt + (v_x, v_y)⋅∇u - D⋅laplacian(u)
    def residuals(params):
        v_x, v_y, D = params
        return du_dt + v_x * grad_u[:, 0] + v_y * grad_u[:, 1] - D * laplacian_u

    # Initial guess for [v_x, v_y, D]
    initial_guess = [0.1, 0.1, 0.1]
    result = least_squares(residuals, initial_guess, bounds=(0, np.inf))

    v_x, v_y, D = result.x
    return v_x, v_y, D

def estimate_coefficients(field_values, spatial_gradients_all, laplacian_values_all, time_step_size, num_time_steps):
    """
    Estimates a single set of coefficients (v_x, v_y, D) using accumulated data across time steps.
    
    Args:
        field_values: Dict with mean predictions for each time step: field_values[k].
        spatial_gradients_all: Dict with 2D gradients (du/dx, du/dy) for each time step.
        laplacian_values_all: Dict with Laplacian values for each time step.
        time_step_size: Time interval between consecutive steps.
        num_time_steps: Total number of time steps in the simulation.
        
    Returns:
        coefficients: Dictionary with estimated 'v_x', 'v_y', and 'D'.
    """
    temporal_derivatives = torch.tensor([])
    spatial_gradients = torch.tensor([])
    laplacian_values = torch.tensor([])

    for time_step in range(1, num_time_steps-1):
        print(f"\n--- Time Step {time_step+1}/{num_time_steps} ---")
        u_next = field_values[time_step+1]
        u_prev = field_values[time_step-1]
        grad_u = spatial_gradients_all[time_step]       # shape (N,2)
        laplacian_u = laplacian_values_all[time_step]   # shape (N,)

        # Compute temporal derivative
        temporal_derivative = finite_difference(u_next, u_prev, time_step_size)

        temporal_derivatives = torch.cat((temporal_derivatives, temporal_derivative), dim=0)
        spatial_gradients = torch.cat((spatial_gradients, grad_u), dim=0)
        laplacian_values = torch.cat((laplacian_values, laplacian_u), dim=0)
    
    v_x, v_y, D = least_squares_fit(temporal_derivatives, spatial_gradients, laplacian_values)
    coefficients = {'v_x': v_x, 'v_y': v_y, 'D': D}
    print(f"\nFinal estimated coefficients - v_x: {v_x}, v_y: {v_y}, D: {D}")

    residuals = torch.square(
        temporal_derivatives
        + v_x * spatial_gradients[:, 0]
        + v_y * spatial_gradients[:, 1]
        - D * laplacian_values
    )
    print(f"Mean squared residual of least-squares fit: {torch.mean(residuals):.6e}")
    return coefficients