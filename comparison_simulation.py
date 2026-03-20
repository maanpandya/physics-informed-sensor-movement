"""
Comparison simulation: Physics-Informed (PIL, Box-Constrained) vs.
Information-Theoretic (Maximum Variance) sensor repositioning.

Both methods share the same Stage I calibration and initial sensor
placement, then run independently for Stage II. Results are compared
via Improvement Factor (IF), MSE over time, final field snapshots, and
sensor trajectory plots. All outputs are saved to the `outputs/` directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import calibrationphase
import gpmodel

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Stage I: Calibration (shared between both methods)
# ---------------------------------------------------------------------------

print("=" * 60)
print("Stage I: Calibration (shared)")
print("=" * 60)

NUM_CALIBRATION_STEPS = 10
TIME_STEP_SIZE = 0.1
DOMAIN_LIMITS = ((0, 10), (0, 5))
NUM_SENSORS = 56
NUM_ANCHOR_X = 200
NUM_ANCHOR_Y = 100

(coefficients,
 _,
 _,
 _,
 anchor_points,
 sensor_locations_cal) = calibrationphase.main_loop(
    NUM_CALIBRATION_STEPS,
    TIME_STEP_SIZE,
    DOMAIN_LIMITS,
    NUM_SENSORS,
    NUM_ANCHOR_X,
    NUM_ANCHOR_Y,
    visualize=False,
)

v_x = coefficients["v_x"]
v_y = coefficients["v_y"]
D = coefficients["D"]

print(f"\nCalibration complete. Estimated v_x={v_x:.4f}, v_y={v_y:.4f}, D={D:.4f}")

# ---------------------------------------------------------------------------
# Shared Stage II parameters
# ---------------------------------------------------------------------------

NUM_REPOSITIONING_STEPS = 100
MAX_MOVE_DIST = 0.25          # r_max used for both methods
BOX_HALF_WIDTH = 0.5
BOX_HALF_HEIGHT = 0.5

initial_sensor_locations = sensor_locations_cal.clone()

final_time = (NUM_CALIBRATION_STEPS - 1) * TIME_STEP_SIZE + NUM_REPOSITIONING_STEPS * TIME_STEP_SIZE

# ---------------------------------------------------------------------------
# Method 1: Physics-Informed (Box-Constrained PIL)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Method 1: Physics-Informed Repositioning (Box-Constrained)")
print("=" * 60)

sensor_locations = initial_sensor_locations.clone()
pil_mse_history = []
pil_sensor_history = []

for step in range(NUM_REPOSITIONING_STEPS):
    if (step + 1) % 10 == 0:
        print(f"  PIL step {step + 1}/{NUM_REPOSITIONING_STEPS}")

    t_current = step * TIME_STEP_SIZE + (NUM_CALIBRATION_STEPS - 1) * TIME_STEP_SIZE

    sensor_data = calibrationphase.observe_field(
        sensor_locations, t_current, calibrationphase.field_function
    )
    gp_current, _ = gpmodel.initialize_gp_model(sensor_locations, sensor_data)
    gp_current.fit(sensor_locations, sensor_data, num_iter=50, learning_rate=0.01)

    mean_sensors = gp_current.predict(sensor_locations)[0]
    grad_sensors, lap_sensors = gp_current.compute_finite_difference_derivatives(sensor_locations)
    mean_anchors = gp_current.predict(anchor_points)[0]

    exact_anchors = calibrationphase.field_function(anchor_points, t_current)
    mse = torch.mean((mean_anchors - exact_anchors) ** 2).item()
    pil_mse_history.append(mse)
    pil_sensor_history.append(sensor_locations.clone())

    du_dt_sensors = -(v_x * grad_sensors[:, 0] + v_y * grad_sensors[:, 1]) + D * lap_sensors
    predicted_next = mean_sensors + TIME_STEP_SIZE * du_dt_sensors

    gp_next, _ = gpmodel.initialize_gp_model(sensor_locations, predicted_next)
    gp_next.fit(sensor_locations, predicted_next, num_iter=50, learning_rate=0.01)

    mean_next_anchors = gp_next.predict(anchor_points)[0]
    du_dt_anchors = (mean_next_anchors - mean_anchors) / TIME_STEP_SIZE
    grad_next_anchors, lap_next_anchors = gp_next.compute_finite_difference_derivatives(anchor_points)
    _, pil_per_point = gpmodel.calculate_physics_informed_loss(
        du_dt_anchors, grad_next_anchors, lap_next_anchors, v_x, v_y, D
    )

    new_locations = []
    for i, sensor_loc in enumerate(sensor_locations):
        distances = torch.norm(anchor_points - sensor_loc, dim=1)
        feasible = torch.where(distances <= MAX_MOVE_DIST)[0]
        candidate = sensor_loc if len(feasible) == 0 else \
            anchor_points[feasible[torch.argmax(pil_per_point[feasible])]]

        init_pos = initial_sensor_locations[i]
        lb = torch.tensor([init_pos[0] - BOX_HALF_WIDTH, init_pos[1] - BOX_HALF_HEIGHT])
        ub = torch.tensor([init_pos[0] + BOX_HALF_WIDTH, init_pos[1] + BOX_HALF_HEIGHT])
        new_locations.append(torch.max(torch.min(candidate, ub), lb))

    sensor_locations = torch.stack(new_locations)

# Final step
sensor_data_final = calibrationphase.observe_field(sensor_locations, final_time, calibrationphase.field_function)
gp_final, _ = gpmodel.initialize_gp_model(sensor_locations, sensor_data_final)
gp_final.fit(sensor_locations, sensor_data_final, num_iter=50, learning_rate=0.01)
mean_final_pil = gp_final.predict(anchor_points)[0]
exact_final = calibrationphase.field_function(anchor_points, final_time)
mse_final_pil = torch.mean((mean_final_pil - exact_final) ** 2).item()
pil_mse_history.append(mse_final_pil)
pil_sensor_history.append(sensor_locations.clone())

print(f"Method 1 complete. Final MSE: {mse_final_pil:.6e}")

# ---------------------------------------------------------------------------
# Method 2: Information-Theoretic (Maximum Variance)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Method 2: Information-Theoretic Repositioning (Max Variance)")
print("=" * 60)

sensor_locations = initial_sensor_locations.clone()
var_mse_history = []
var_sensor_history = []

for step in range(NUM_REPOSITIONING_STEPS):
    if (step + 1) % 10 == 0:
        print(f"  MaxVar step {step + 1}/{NUM_REPOSITIONING_STEPS}")

    t_current = step * TIME_STEP_SIZE + (NUM_CALIBRATION_STEPS - 1) * TIME_STEP_SIZE

    sensor_data = calibrationphase.observe_field(
        sensor_locations, t_current, calibrationphase.field_function
    )
    gp_current, _ = gpmodel.initialize_gp_model(sensor_locations, sensor_data)
    gp_current.fit(sensor_locations, sensor_data, num_iter=50, learning_rate=0.01)

    mean_anchors, variance_anchors = gp_current.predict(anchor_points)

    exact_anchors = calibrationphase.field_function(anchor_points, t_current)
    mse = torch.mean((mean_anchors - exact_anchors) ** 2).item()
    var_mse_history.append(mse)
    var_sensor_history.append(sensor_locations.clone())

    new_locations = []
    for sensor_loc in sensor_locations:
        distances = torch.norm(anchor_points - sensor_loc, dim=1)
        feasible = torch.where(distances <= MAX_MOVE_DIST)[0]
        if len(feasible) == 0:
            new_locations.append(sensor_loc)
        else:
            best = feasible[torch.argmax(variance_anchors[feasible])]
            new_locations.append(anchor_points[best])

    sensor_locations = torch.stack(new_locations)

# Final step
sensor_data_final = calibrationphase.observe_field(sensor_locations, final_time, calibrationphase.field_function)
gp_final, _ = gpmodel.initialize_gp_model(sensor_locations, sensor_data_final)
gp_final.fit(sensor_locations, sensor_data_final, num_iter=50, learning_rate=0.01)
mean_final_var = gp_final.predict(anchor_points)[0]
mse_final_var = torch.mean((mean_final_var - exact_final) ** 2).item()
var_mse_history.append(mse_final_var)
var_sensor_history.append(sensor_locations.clone())

print(f"Method 2 complete. Final MSE: {mse_final_var:.6e}")

# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)
print(f"  PIL    — MSE @ step 20: {pil_mse_history[20]:.6e}  |  Final MSE: {mse_final_pil:.6e}")
print(f"  MaxVar — MSE @ step 20: {var_mse_history[20]:.6e}  |  Final MSE: {mse_final_var:.6e}")
if len(pil_mse_history) > 0 and pil_mse_history[20] > 0:
    if_20 = var_mse_history[20] / pil_mse_history[20]
    if_100 = mse_final_var / mse_final_pil
    print(f"  IF @ step 20: {if_20:.2f}  |  IF @ step 100: {if_100:.2f}")

# ---------------------------------------------------------------------------
# Plot 1: Improvement Factor over time
# ---------------------------------------------------------------------------

mse_ratios = np.array(var_mse_history) / np.array(pil_mse_history)
total_steps = len(mse_ratios)

plt.figure(figsize=(10, 4))
plt.plot(range(total_steps), mse_ratios,
         color="blue", marker="o", linestyle="-", linewidth=2, markersize=4)
plt.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.7)
plt.fill_between(range(total_steps), 1, mse_ratios,
                 where=mse_ratios >= 1, color="green", alpha=0.3, interpolate=True,
                 label="PIL Better")
plt.fill_between(range(total_steps), 1, mse_ratios,
                 where=mse_ratios < 1, color="red", alpha=0.3, interpolate=True,
                 label="MaxVar Better")
plt.xlabel("Repositioning Step", fontsize=36)
plt.ylabel("Improvement Factor (IF)", fontsize=36)
plt.title("Physics Informed vs Information Theoretic", fontsize=36)
plt.ylim(bottom=0)
plt.xlim(0, total_steps - 1)
plt.grid(True, alpha=0.3)
plt.tick_params(axis="both", which="major", labelsize=36)
plt.tight_layout()
if_path = os.path.join(OUTPUT_DIR, "Comparison_ImprovementFactor.pdf")
plt.savefig(if_path, dpi=300)
print(f"\nSaved: {if_path}")
plt.show()

# ---------------------------------------------------------------------------
# Plot 2: MSE over time
# ---------------------------------------------------------------------------

plt.figure(figsize=(10, 6))
time_axis = np.arange(NUM_REPOSITIONING_STEPS + 1)
plt.plot(time_axis, pil_mse_history, label="Physics-Informed (PIL, Box-Constrained)", linewidth=2)
plt.plot(time_axis, var_mse_history, label="Info-Theoretic (Max Variance)", linewidth=2, linestyle="--")
plt.xlabel("Repositioning Step", fontsize=14)
plt.ylabel("MSE", fontsize=14)
plt.title("Performance Comparison: MSE over Time", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
mse_path = os.path.join(OUTPUT_DIR, "Comparison_MSE_PIL_vs_Variance.png")
plt.savefig(mse_path, dpi=300)
print(f"Saved: {mse_path}")
plt.show()

# ---------------------------------------------------------------------------
# Plot 3: Final-state field snapshots
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=2, figsize=(14, 6))

sensors_pil_final = pil_sensor_history[-1]
sensor_data_pil = calibrationphase.observe_field(sensors_pil_final, final_time, calibrationphase.field_function)
gp_pil, _ = gpmodel.initialize_gp_model(sensors_pil_final, sensor_data_pil)
gp_pil.fit(sensors_pil_final, sensor_data_pil, num_iter=50, learning_rate=0.01)
mean_pil = gp_pil.predict(anchor_points)[0]

ax1 = axes[0]
sc1 = ax1.scatter(anchor_points[:, 0].numpy(), anchor_points[:, 1].numpy(),
                  c=mean_pil.detach().numpy(), cmap="viridis", s=20)
plt.colorbar(sc1, ax=ax1, label="Predicted Field")
ax1.scatter(sensors_pil_final[:, 0], sensors_pil_final[:, 1], c="red", s=40, label="Sensors")
ax1.set_title(f"Physics-Informed (MSE: {mse_final_pil:.2e})", fontsize=14)
ax1.legend()

sensors_var_final = var_sensor_history[-1]
sensor_data_var = calibrationphase.observe_field(sensors_var_final, final_time, calibrationphase.field_function)
gp_var, _ = gpmodel.initialize_gp_model(sensors_var_final, sensor_data_var)
gp_var.fit(sensors_var_final, sensor_data_var, num_iter=50, learning_rate=0.01)
mean_var = gp_var.predict(anchor_points)[0]

ax2 = axes[1]
sc2 = ax2.scatter(anchor_points[:, 0].numpy(), anchor_points[:, 1].numpy(),
                  c=mean_var.detach().numpy(), cmap="viridis", s=20)
plt.colorbar(sc2, ax=ax2, label="Predicted Field")
ax2.scatter(sensors_var_final[:, 0], sensors_var_final[:, 1], c="red", s=40, label="Sensors")
ax2.set_title(f"Info-Theoretic (MSE: {mse_final_var:.2e})", fontsize=14)
ax2.legend()

plt.tight_layout()
snap_path = os.path.join(OUTPUT_DIR, "Comparison_FinalState_PIL_vs_Variance.png")
plt.savefig(snap_path, dpi=300)
print(f"Saved: {snap_path}")
plt.show()

# ---------------------------------------------------------------------------
# Plot 4: Sensor trajectory comparison
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=2, figsize=(14, 6))

for ax, sensor_history, title in [
    (axes[0], pil_sensor_history, "Physics-Informed Trajectories"),
    (axes[1], var_sensor_history, "Info-Theoretic Trajectories"),
]:
    positions = torch.stack(sensor_history).detach().cpu().numpy()
    for i in range(NUM_SENSORS):
        ax.plot(positions[:, i, 0], positions[:, i, 1], alpha=0.5, linewidth=1)
    ax.scatter(positions[0, :, 0], positions[0, :, 1], c="green", s=20, label="Start")
    ax.scatter(positions[-1, :, 0], positions[-1, :, 1], c="red", s=20, label="End")
    ax.set_title(title, fontsize=14)
    ax.set_xlim(DOMAIN_LIMITS[0])
    ax.set_ylim(DOMAIN_LIMITS[1])

plt.tight_layout()
traj_path = os.path.join(OUTPUT_DIR, "Comparison_Trajectories.pdf")
plt.savefig(traj_path, dpi=300)
print(f"Saved: {traj_path}")
plt.show()
