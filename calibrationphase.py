"""
Stage I — Calibration phase.

Sensors remain at their initial positions for N' time steps. At each step, a
GP is fitted to the sensor measurements and derivatives of the posterior mean
are computed at the anchor points. After all N' steps, PDE coefficients
(v_x, v_y, D) are estimated via least squares (see coefficientestimation.py).

Key public interface:
    field_function(xy, t)  — analytical advection-diffusion field (ground truth).
    observe_field(...)     — noise-free field measurement at sensor locations.
    main_loop(...)         — runs the full calibration phase and returns results.
"""

import torch
import gpytorch
from gpmodel import GPModel, initialize_gp_model
from coefficientestimation import estimate_coefficients, define_anchor_points

import matplotlib.pyplot as plt

def visualize_gp_fit(gp_model, sensor_locations, sensor_data, domain_limits, time_step, time_step_size, field_func):
    """
    Visualizes GP fit for 2D data by plotting predicted slices along a uniform grid.
    For demonstration, we show a slice if needed or a scatter plot of predicted vs. exact.
    """
    ((x_min, x_max), (y_min, y_max)) = domain_limits
    # Generate a small grid for visualization
    num_plot = 30
    x_coords = torch.linspace(x_min, x_max, num_plot)
    y_coords = torch.linspace(y_min, y_max, num_plot)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    test_xy = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    gp_model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = gp_model.likelihood(gp_model(test_xy))
        mean = pred.mean
        lower, upper = pred.confidence_region()

    u_exact = field_func(test_xy, time_step*time_step_size)

    plt.figure(figsize=(8,6))
    sc = plt.scatter(test_xy[:,0].numpy(), test_xy[:,1].numpy(), c=mean.numpy(), cmap='viridis', s=20, marker='s')
    plt.colorbar(sc, label="Predicted Field")
    plt.scatter(sensor_locations[:,0].numpy(), sensor_locations[:,1].numpy(), c='red', edgecolors='k', s=50, label='Sensors')
    plt.title(f"GP Prediction (Mean) at Time Step {time_step+1}")
    plt.legend()
    plt.show()

def field_function(xy, t, vx=2.0, vy=1.0, D=0.5):
    """
    Analytical advection-diffusion field (ground truth):
        u(x, y, t) = sin(x - vx*t) * cos(y - vy*t) * exp(-D*t)

    True parameters used in the paper: vx=2.0, vy=1.0, D=0.5.
    """
    x = xy[:,0]
    y = xy[:,1]
    return torch.sin(x - vx*t) * torch.cos(y - vy*t) * torch.exp(torch.tensor(-D*t))

def initialize_sensors(num_sensors, domain_limits, random=False):
    """
    Initializes sensor positions in 2D within the spatial domain.
    
    Args:
        num_sensors: Number of sensors to initialize.
        domain_limits: Tuple ((x_min, x_max),(y_min, y_max)) for 2D boundaries.
    
    Returns:
        sensor_locations: Tensor of shape (num_sensors, 2).
    """
    (x_min, x_max), (y_min, y_max) = domain_limits
    if random:
        sensor_x = x_min + (x_max - x_min) * torch.rand(num_sensors)
        sensor_y = y_min + (y_max - y_min) * torch.rand(num_sensors)
    else:
        # For simplicity, place them in a gridlike pattern
        nx = int(num_sensors**0.5)
        ny = nx
        if nx*ny < num_sensors:
            ny += 1
        xs = torch.linspace(x_min, x_max, nx)
        ys = torch.linspace(y_min, y_max, ny)
        X, Y = torch.meshgrid(xs, ys, indexing='xy')
        XY = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        sensor_locations = XY[:num_sensors]
        return sensor_locations
    sensor_locations = torch.stack([sensor_x, sensor_y], dim=-1)
    return sensor_locations

def observe_field(sensor_locations, time, field_func):
    """
    Observes the 2D field at sensor locations for a given time step.
    """
    sensor_data = field_func(sensor_locations, time)
    return sensor_data.view(-1)

def main_loop(num_time_steps, time_step_size, domain_limits,
              num_sensors, num_anchor_points_x, num_anchor_points_y,
              visualize=False):
    """
    Main loop for calibration phase in 2D.
    """
    sensor_locations = initialize_sensors(num_sensors, domain_limits, random=False)
    field_func_2d = field_function

    # Anchor points for PDE coefficient estimation
    anchor_points = define_anchor_points(domain_limits, num_anchor_points_x, num_anchor_points_y)

    field_values = {}
    spatial_gradients_all = {}
    laplacian_values_all = {}

    for time_step in range(num_time_steps):
        print(f"\n--- Time Step {time_step+1}/{num_time_steps} ---")
        time = time_step * time_step_size

        # 1. Observe field
        sensor_data = observe_field(sensor_locations, time, field_func_2d)
        print("Observed sensor data.")

        # 2. Fit GP
        gp_model, likelihood = initialize_gp_model(sensor_locations, sensor_data)
        gp_model.fit(sensor_locations, sensor_data, num_iter=100, learning_rate=0.01)
        print("GP model fitted.")

        # 3. Predict at anchor points
        mean_prediction = gp_model.predict(anchor_points)[0]  # shape (N,)
        grad_u, lapla_u = gp_model.compute_finite_difference_derivatives(anchor_points)

        # Store
        field_values[time_step] = mean_prediction
        spatial_gradients_all[time_step] = grad_u
        laplacian_values_all[time_step] = lapla_u

        # Visualization
        if visualize:
            visualize_gp_fit(gp_model, sensor_locations, sensor_data,
                             domain_limits, time_step, time_step_size, field_func_2d)

    # 4. Estimate coefficients
    coefficients = estimate_coefficients(field_values, spatial_gradients_all,
                                         laplacian_values_all, time_step_size, num_time_steps)
    print("\nSimulation complete.")
    print(coefficients)
    return coefficients, field_values, spatial_gradients_all, laplacian_values_all, anchor_points, sensor_locations