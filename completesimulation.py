"""
Two-stage physics-informed sensor repositioning simulation.

Stage I (Calibration): Sensors remain stationary while a GP is fitted at each
time step. PDE parameters (v_x, v_y, D) are estimated via least squares.

Stage II (Repositioning): Sensors are moved toward anchor points with the
highest Physics-Informed Loss (PIL) at each step. Three policies are available:
  - "Baseline"    : each sensor targets the highest-PIL anchor in its radius.
  - "No Overlap"  : same as Baseline, but each anchor may only be claimed once.
  - "Constrained" : same as Baseline, but each sensor is confined to a box
                    around its initial position (recommended; see paper).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

import calibrationphase
import gpmodel

# ---------------------------------------------------------------------------
# User-facing parameters
# ---------------------------------------------------------------------------

# Stage I
NUM_CALIBRATION_STEPS = 10
TIME_STEP_SIZE = 0.1
DOMAIN_LIMITS = ((0, 10), (0, 5))
NUM_SENSORS = 56
NUM_ANCHOR_X = 200
NUM_ANCHOR_Y = 100

# Stage II
NUM_REPOSITIONING_STEPS = 100
MAX_MOVE_DIST = 0.1          # Maximum per-step movement radius r_max
REPOSITIONING_METHOD = "Constrained"   # "Baseline" | "No Overlap" | "Constrained"

# Box half-widths for the "Constrained" method
BOX_HALF_WIDTH = 0.5
BOX_HALF_HEIGHT = 0.5

# If True, a stepwise MSE comparison against static sensors is computed and plotted
COMPARE_STEPWISE = True

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Stage I: Calibration
# ---------------------------------------------------------------------------

print("=" * 60)
print("Stage I: Calibration")
print("=" * 60)

(coefficients,
 _field_values_cal,
 _gradients_cal,
 _laplacians_cal,
 anchor_points,
 sensor_locations) = calibrationphase.main_loop(
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
# Stage II: Repositioning
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"Stage II: Repositioning  (method='{REPOSITIONING_METHOD}')")
print("=" * 60)

initial_sensor_locations = sensor_locations.clone()

# Accumulators for visualisation and post-processing
all_sensor_locations = []
all_mean_anchors = []
all_exact_anchors = []
all_times = []

for step in range(NUM_REPOSITIONING_STEPS):
    print(f"\n--- Repositioning Step {step + 1}/{NUM_REPOSITIONING_STEPS} ---")
    t_current = step * TIME_STEP_SIZE + (NUM_CALIBRATION_STEPS - 1) * TIME_STEP_SIZE

    # 1. Observe field at current sensor locations
    sensor_data = calibrationphase.observe_field(
        sensor_locations, t_current, calibrationphase.field_function
    )

    # 2. Fit GP to current observations
    gp_current, _ = gpmodel.initialize_gp_model(sensor_locations, sensor_data)
    gp_current.fit(sensor_locations, sensor_data, num_iter=50, learning_rate=0.01)

    # 3. Evaluate GP posterior at sensors and anchor points
    mean_sensors = gp_current.predict(sensor_locations)[0]
    grad_sensors, lap_sensors = gp_current.compute_finite_difference_derivatives(sensor_locations)
    mean_anchors = gp_current.predict(anchor_points)[0]

    # 4. Forward-Euler prediction of next-step field values at sensor locations
    du_dt_sensors = -(v_x * grad_sensors[:, 0] + v_y * grad_sensors[:, 1]) + D * lap_sensors
    predicted_next = mean_sensors + TIME_STEP_SIZE * du_dt_sensors

    # 5. Fit a predictive GP using the forward-Euler estimate
    gp_next, _ = gpmodel.initialize_gp_model(sensor_locations, predicted_next)
    gp_next.fit(sensor_locations, predicted_next, num_iter=50, learning_rate=0.01)

    # 6. Compute PIL at all anchor points using the predictive GP
    mean_next_anchors = gp_next.predict(anchor_points)[0]
    du_dt_anchors = (mean_next_anchors - mean_anchors) / TIME_STEP_SIZE
    grad_next_anchors, lap_next_anchors = gp_next.compute_finite_difference_derivatives(anchor_points)
    _, pil_per_point = gpmodel.calculate_physics_informed_loss(
        du_dt_anchors, grad_next_anchors, lap_next_anchors, v_x, v_y, D
    )

    # 7. Move each sensor to the highest-PIL anchor within its movement radius
    new_locations = []

    if REPOSITIONING_METHOD == "Baseline":
        for sensor_loc in sensor_locations:
            distances = torch.norm(anchor_points - sensor_loc, dim=1)
            feasible = torch.where(distances <= MAX_MOVE_DIST)[0]
            if len(feasible) == 0:
                new_locations.append(sensor_loc)
            else:
                best = feasible[torch.argmax(pil_per_point[feasible])]
                new_locations.append(anchor_points[best])

    elif REPOSITIONING_METHOD == "No Overlap":
        claimed = set()
        for sensor_loc in sensor_locations:
            distances = torch.norm(anchor_points - sensor_loc, dim=1)
            feasible = torch.where(distances <= MAX_MOVE_DIST)[0]
            if len(feasible) == 0:
                new_locations.append(sensor_loc)
                continue
            sorted_idxs = feasible[torch.argsort(pil_per_point[feasible], descending=True)]
            chosen = None
            for idx in sorted_idxs:
                if idx.item() not in claimed:
                    claimed.add(idx.item())
                    chosen = anchor_points[idx]
                    break
            new_locations.append(chosen if chosen is not None else sensor_loc)

    elif REPOSITIONING_METHOD == "Constrained":
        for i, sensor_loc in enumerate(sensor_locations):
            distances = torch.norm(anchor_points - sensor_loc, dim=1)
            feasible = torch.where(distances <= MAX_MOVE_DIST)[0]
            if len(feasible) == 0:
                candidate = sensor_loc
            else:
                best = feasible[torch.argmax(pil_per_point[feasible])]
                candidate = anchor_points[best]

            init_pos = initial_sensor_locations[i]
            lb = torch.tensor([init_pos[0] - BOX_HALF_WIDTH, init_pos[1] - BOX_HALF_HEIGHT])
            ub = torch.tensor([init_pos[0] + BOX_HALF_WIDTH, init_pos[1] + BOX_HALF_HEIGHT])
            new_locations.append(torch.max(torch.min(candidate, ub), lb))

    else:
        raise ValueError(f"Unknown repositioning method: '{REPOSITIONING_METHOD}'. "
                         "Choose from 'Baseline', 'No Overlap', or 'Constrained'.")

    # Record state for post-processing
    exact_anchors = calibrationphase.field_function(anchor_points, t_current)
    all_sensor_locations.append(sensor_locations.clone())
    all_mean_anchors.append(mean_anchors.clone())
    all_exact_anchors.append(exact_anchors.clone())
    all_times.append(t_current)

    if (step + 1) == 20:
        mse_20 = torch.mean((mean_anchors - exact_anchors) ** 2).item()
        print(f"\n[Checkpoint] MSE at step 20: {mse_20:.6e}")

    sensor_locations = torch.stack(new_locations)

print("\nRepositioning phase complete.")

# ---------------------------------------------------------------------------
# Final-step GP evaluation
# ---------------------------------------------------------------------------

final_time = (NUM_CALIBRATION_STEPS - 1) * TIME_STEP_SIZE + NUM_REPOSITIONING_STEPS * TIME_STEP_SIZE
final_data = calibrationphase.observe_field(sensor_locations, final_time, calibrationphase.field_function)
gp_final, _ = gpmodel.initialize_gp_model(sensor_locations, final_data)
gp_final.fit(sensor_locations, final_data, num_iter=50, learning_rate=0.01)
final_mean = gp_final.predict(anchor_points)[0]
final_exact = calibrationphase.field_function(anchor_points, final_time)

all_sensor_locations.append(sensor_locations.clone())
all_mean_anchors.append(final_mean.clone())
all_exact_anchors.append(final_exact.clone())
all_times.append(final_time)

mse_final = torch.mean((final_mean - final_exact) ** 2).item()
print(f"Final MSE after repositioning: {mse_final:.6e}")

# ---------------------------------------------------------------------------
# Plot: field snapshot at step 100
# ---------------------------------------------------------------------------

for label, idx in {"100_step": 100}.items():
    t = all_times[idx]
    sensors = all_sensor_locations[idx].numpy()
    pred_field = all_mean_anchors[idx].detach().numpy()
    exact_field = all_exact_anchors[idx].detach().numpy()
    abs_diff = np.abs(exact_field - pred_field)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

    ax1 = axes[0]
    sc1 = ax1.scatter(anchor_points[:, 0].numpy(), anchor_points[:, 1].numpy(),
                      c=pred_field, cmap="viridis", s=20)
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.ax.tick_params(labelsize=24)
    ax1.scatter(sensors[:, 0], sensors[:, 1], color="orangered", marker="o", s=40)
    ax1.set_title("Predicted Field & Sensors", fontsize=24)
    ax1.tick_params(axis="both", which="major", labelsize=24)
    ax1.set_xticks([0, 5, 10])
    ax1.set_yticks([0, 2.5, 5])
    ax1.set_xlabel("$x$", fontsize=24)
    ax1.set_ylabel("$y$", fontsize=24)

    ax2 = axes[1]
    sc2 = ax2.scatter(anchor_points[:, 0].numpy(), anchor_points[:, 1].numpy(),
                      c=abs_diff, cmap="viridis", s=20, vmin=0, vmax=0.0025)
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.ax.tick_params(labelsize=24)
    ax2.set_title("Absolute Error", fontsize=24)
    ax2.tick_params(axis="both", which="major", labelsize=24)
    ax2.set_xticks([0, 5, 10])
    ax2.set_yticks([0, 2.5, 5])
    ax2.set_xlabel("$x$", fontsize=24)
    ax2.set_ylabel("$y$", fontsize=24)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{label}_{REPOSITIONING_METHOD}_snapshot.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved snapshot: {out_path}")
    plt.show()

# ---------------------------------------------------------------------------
# Plot: sensor trajectories
# ---------------------------------------------------------------------------

positions = torch.stack(all_sensor_locations).detach().cpu().numpy()
num_steps_plot, num_sensors_plot, _ = positions.shape

plt.figure(figsize=(10, 8))
cmap = plt.cm.gnuplot2
for s in range(num_sensors_plot):
    color = cmap(s / num_sensors_plot)
    plt.plot(positions[:, s, 0], positions[:, s, 1], linestyle="-", color=color, zorder=1)
plt.scatter(positions[0, :, 0], positions[0, :, 1],
            color="green", marker="s", s=80, label="Initial Position", zorder=10)
plt.scatter(positions[-1, :, 0], positions[-1, :, 1],
            color="red", marker="X", s=80, label="Final Position", zorder=10)
plt.xlabel("$x$", fontsize=24)
plt.ylabel("$y$", fontsize=24)
plt.title("Sensor Trajectories", fontsize=24)
plt.tick_params(axis="both", which="major", labelsize=24)
plt.tight_layout()
traj_path = os.path.join(OUTPUT_DIR, f"{REPOSITIONING_METHOD}_sensor_trajectories.png")
plt.savefig(traj_path, dpi=300)
print(f"Saved trajectories: {traj_path}")
plt.show()

# ---------------------------------------------------------------------------
# Stepwise comparison against static sensors
# ---------------------------------------------------------------------------

if COMPARE_STEPWISE:
    mse_ratios = []
    better_count = 0
    total_steps = len(all_times)

    for i in range(total_steps):
        moved_mse = torch.mean((all_mean_anchors[i] - all_exact_anchors[i]) ** 2).item()

        static_data = calibrationphase.observe_field(
            initial_sensor_locations, all_times[i], calibrationphase.field_function
        )
        gp_static, _ = gpmodel.initialize_gp_model(initial_sensor_locations, static_data)
        gp_static.fit(initial_sensor_locations, static_data, num_iter=50, learning_rate=0.01)
        static_pred = gp_static.predict(anchor_points)[0]
        static_mse = torch.mean((static_pred - all_exact_anchors[i]) ** 2).item()

        ratio = static_mse / moved_mse
        mse_ratios.append(ratio)
        if moved_mse <= static_mse:
            better_count += 1

    pct = 100.0 * better_count / total_steps
    print(f"\nRepositioned sensors outperformed static on {better_count}/{total_steps} steps ({pct:.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(total_steps), mse_ratios,
            color="blue", marker="o", linestyle="-", linewidth=2, markersize=4)
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.fill_between(range(total_steps), 1, mse_ratios,
                    where=np.array(mse_ratios) >= 1,
                    color="green", alpha=0.3, interpolate=True, label="Repositioning Better")
    ax.fill_between(range(total_steps), 1, mse_ratios,
                    where=np.array(mse_ratios) < 1,
                    color="red", alpha=0.3, interpolate=True, label="Static Better")
    ax.set_xlabel("Repositioning Step", fontsize=24)
    ax.set_ylabel("Improvement Factor (IF)", fontsize=24)
    ax.set_title(f"Physics-Informed vs Static — {REPOSITIONING_METHOD}", fontsize=24)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, total_steps - 1)
    ax.set_xticks(range(0, total_steps, 10))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=24)
    fig.tight_layout()
    if_path = os.path.join(OUTPUT_DIR, f"{REPOSITIONING_METHOD}_improvement_factor.png")
    plt.savefig(if_path, dpi=300)
    print(f"Saved improvement factor plot: {if_path}")
    plt.show()

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

fig_anim, axes_anim = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax_pred, ax_exact = axes_anim

sc_pred = ax_pred.scatter([], [], c=[], cmap="viridis", s=20)
sc_sensors = ax_pred.scatter([], [], c="red", s=40, marker="o")
sc_gt = ax_exact.scatter([], [], c=[], cmap="viridis", s=20)
ax_pred.set_title("Predicted Field + Sensors")
ax_exact.set_title("Exact Field")

(x_min, x_max), (y_min, y_max) = DOMAIN_LIMITS
for ax in (ax_pred, ax_exact):
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def _init():
    sc_pred.set_offsets(np.empty((0, 2)))
    sc_pred.set_array([])
    sc_sensors.set_offsets(np.empty((0, 2)))
    sc_gt.set_offsets(np.empty((0, 2)))
    sc_gt.set_array([])
    return sc_pred, sc_sensors, sc_gt


def _update(frame):
    sc_pred.set_offsets(anchor_points.numpy())
    sc_pred.set_array(all_mean_anchors[frame].detach().numpy())
    sc_sensors.set_offsets(all_sensor_locations[frame].numpy())
    sc_gt.set_offsets(anchor_points.numpy())
    sc_gt.set_array(all_exact_anchors[frame].detach().numpy())
    ax_pred.set_title(f"Predicted Field + Sensors (t={all_times[frame]:.2f}) — {REPOSITIONING_METHOD}")
    ax_exact.set_title(f"Exact Field (t={all_times[frame]:.2f})")
    return sc_pred, sc_sensors, sc_gt


ani = FuncAnimation(fig_anim, _update, frames=len(all_sensor_locations),
                    init_func=_init, blit=False)
anim_path = os.path.join(OUTPUT_DIR, f"{REPOSITIONING_METHOD}_animation.gif")
ani.save(anim_path, writer="Pillow")
print(f"Saved animation: {anim_path}")
plt.tight_layout()
plt.show()
