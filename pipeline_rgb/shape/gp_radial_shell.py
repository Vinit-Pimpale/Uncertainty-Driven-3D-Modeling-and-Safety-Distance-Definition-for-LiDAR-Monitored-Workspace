#!/usr/bin/env python3
"""
Stage-3.1 — Gaussian Process Radial Shape Modeling
==================================================

This script takes the *aggregated* radial statistics from Stage-2.3
(radial_stats.npz) and fits a Gaussian Process to model the radial
extent r(u) as a function of direction u on the unit sphere.

Inputs
------
MODEL_DIR = pipeline_rgb/shape/model_stage2

  shape_params.json
    - master_rotation: 3x3 rotation matrix (GLOBAL orientation)
    - master_axes, master_dims (not strictly required here)

  radial_stats.npz  (from Stage-2.3)
    - theta_centers: [n_theta]
    - phi_centers:   [n_phi]
    - mean_r:        [n_theta, n_phi]
    - std_r:         [n_theta, n_phi]
    - counts:        [n_theta, n_phi]

Outputs
-------
In MODEL_DIR:

  gp_radial_stats.npz
      - theta_pred: [n_theta_pred]
      - phi_pred:   [n_phi_pred]
      - mean_r_gp:  [n_theta_pred, n_phi_pred]
      - std_r_gp:   [n_theta_pred, n_phi_pred]

  gp_mean_shell.ply   (yellow)
  gp_k2_shell.ply     (red, mean + 2σ)
  gp_k3_shell.ply     (blue, mean + 3σ)

All shells are expressed in the GLOBAL frame (same as ellipsoid_mean).
"""

import os
import json
import numpy as np
from plyfile import PlyData, PlyElement

# You need scikit-learn installed in your venv:
#   pip install scikit-learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

MODEL_DIR = "pipeline_rgb/shape/model_stage2"


# ----------------------------------------------------------------------
# Utility: write PLY
# ----------------------------------------------------------------------
def write_ply(path, xyz, rgb):
    N = xyz.shape[0]
    arr = np.empty(
        N,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)


# ----------------------------------------------------------------------
# Build spherical grid for prediction
# ----------------------------------------------------------------------
def create_prediction_grid(n_theta=96, n_phi=48):
    """
    Create a finer prediction grid for θ, φ.

    theta ∈ [-π, π]
    phi   ∈ [0, π]
    """
    theta_pred = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    phi_pred = np.linspace(0.0, np.pi, n_phi)
    return theta_pred, phi_pred


# ----------------------------------------------------------------------
# Turn (θ, φ) into unit vector u = (ux, uy, uz)
# ----------------------------------------------------------------------
def spherical_to_unit(theta, phi):
    """
    theta, phi are arrays broadcastable to same shape.

    theta: azimuth in [-π, π]
    phi:   polar in [0, π], 0 at +Z

    Returns:
      ux, uy, uz of same shape
    """
    ux = np.cos(theta) * np.sin(phi)
    uy = np.sin(theta) * np.sin(phi)
    uz = np.cos(phi)
    return ux, uy, uz


# ----------------------------------------------------------------------
# Main Stage-3.1
# ----------------------------------------------------------------------
def main():
    # ---------------------------------------------------------------
    # 1. Load shape parameters & radial stats
    # ---------------------------------------------------------------
    params_path = os.path.join(MODEL_DIR, "shape_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Cannot find shape_params.json in {MODEL_DIR}")

    with open(params_path, "r") as f:
        params = json.load(f)

    master_R = np.array(params["master_rotation"], dtype=np.float32)  # 3x3

    stats_path = os.path.join(MODEL_DIR, "radial_stats.npz")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Cannot find radial_stats.npz in {MODEL_DIR}. "
            "Run Stage-2.3 first."
        )

    data = np.load(stats_path)
    theta_centers = data["theta_centers"]  # [n_theta]
    phi_centers = data["phi_centers"]      # [n_phi]
    mean_r = data["mean_r"]                # [n_theta, n_phi]
    std_r = data["std_r"]                  # [n_theta, n_phi]
    counts = data["counts"]                # [n_theta, n_phi]

    n_theta = theta_centers.shape[0]
    n_phi = phi_centers.shape[0]
    print(f"[Stage3.1] Loaded radial stats: {n_theta} x {n_phi} bins.")

    # ---------------------------------------------------------------
    # 2. Build training data for GP
    # ---------------------------------------------------------------
    # We'll train the GP on bin centers where we have >= min_count samples.
    min_count = 3
    valid = counts >= min_count

    if not np.any(valid):
        print("[Stage3.1] WARNING: No bins with enough samples; lowering min_count to 1.")
        valid = counts >= 1
        if not np.any(valid):
            raise RuntimeError("No valid bins at all; radial stats are empty.")

    # Build 2D grids of theta, phi
    theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers, indexing="ij")

    theta_valid = theta_grid[valid]
    phi_valid = phi_grid[valid]
    r_valid = mean_r[valid]
    std_valid = std_r[valid]

    # Convert direction to 3D unit vector for GP input
    ux, uy, uz = spherical_to_unit(theta_valid, phi_valid)
    X_train = np.stack((ux, uy, uz), axis=1)   # [N, 3]
    y_train = r_valid.astype(np.float64)       # [N]

    # Optional: we can use std_valid as noise hint, but to keep it simple we
    # just let WhiteKernel learn noise level.
    print(f"[Stage3.1] Training points: {X_train.shape[0]}")

    # ---------------------------------------------------------------
    # 3. Define and fit Gaussian Process
    # ---------------------------------------------------------------
    # Kernel: Constant * RBF + White
    kernel = (
        ConstantKernel(1.0, (0.1, 10.0))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,           # we use WhiteKernel instead of alpha
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=0,
    )

    print("[Stage3.1] Fitting Gaussian Process (this may take a moment)...")
    gp.fit(X_train, y_train)
    print("[Stage3.1] GP fit complete.")
    print("[Stage3.1] Learned kernel:", gp.kernel_)

    # ---------------------------------------------------------------
    # 4. Predict radial mean & std on a finer spherical grid
    # ---------------------------------------------------------------
    theta_pred, phi_pred = create_prediction_grid(
        n_theta=96,
        n_phi=48,
    )
    n_theta_pred = theta_pred.shape[0]
    n_phi_pred = phi_pred.shape[0]

    theta_mesh, phi_mesh = np.meshgrid(theta_pred, phi_pred, indexing="ij")
    ux_p, uy_p, uz_p = spherical_to_unit(theta_mesh, phi_mesh)

    X_pred = np.stack((ux_p, uy_p, uz_p), axis=-1).reshape(-1, 3)  # [N_pred, 3]

    print("[Stage3.1] Predicting GP posterior on fine grid...")
    mean_pred, std_pred = gp.predict(X_pred, return_std=True)

    mean_r_gp = mean_pred.reshape(n_theta_pred, n_phi_pred)
    std_r_gp = std_pred.reshape(n_theta_pred, n_phi_pred)

    # ---------------------------------------------------------------
    # 5. Save GP radial stats
    # ---------------------------------------------------------------
    gp_stats_path = os.path.join(MODEL_DIR, "gp_radial_stats.npz")
    np.savez_compressed(
        gp_stats_path,
        theta_pred=theta_pred,
        phi_pred=phi_pred,
        mean_r_gp=mean_r_gp,
        std_r_gp=std_r_gp,
    )
    print(f"[Stage3.1] Saved GP radial stats to {gp_stats_path}")

    # ---------------------------------------------------------------
    # 6. Build shells in GLOBAL frame (using master_R)
    # ---------------------------------------------------------------
    def build_shell(r_field, color):
        """
        r_field: [n_theta_pred, n_phi_pred] radii in local frame.
        color: (R,G,B)
        Returns:
          xyz_world: [N, 3]
          rgb:       [N, 3] uint8
        """
        # theta_mesh, phi_mesh already created above
        x = r_field * np.cos(theta_mesh) * np.sin(phi_mesh)
        y = r_field * np.sin(theta_mesh) * np.sin(phi_mesh)
        z = r_field * np.cos(phi_mesh)

        pts_local = np.stack((x, y, z), axis=-1)          # [nθ, nφ, 3]
        pts_local_flat = pts_local.reshape(-1, 3)         # [N, 3]

        # Transform to world frame via master_R
        pts_world = pts_local_flat @ master_R             # [N, 3]

        rgb = np.tile(np.array(color, dtype=np.uint8), (pts_world.shape[0], 1))
        return pts_world, rgb

    # Mean shell
    mean_xyz, mean_rgb = build_shell(mean_r_gp, (255, 255, 0))   # yellow
    mean_shell_path = os.path.join(MODEL_DIR, "gp_mean_shell.ply")
    write_ply(mean_shell_path, mean_xyz, mean_rgb)
    print(f"[Stage3.1] Saved GP mean shell: {mean_shell_path}")

    # k = 2 (approx 95% CI)
    k2_r = mean_r_gp + 2.0 * std_r_gp
    k2_xyz, k2_rgb = build_shell(k2_r, (255, 0, 0))       # red
    k2_shell_path = os.path.join(MODEL_DIR, "gp_k2_shell.ply")
    write_ply(k2_shell_path, k2_xyz, k2_rgb)
    print(f"[Stage3.1] Saved GP k=2 shell: {k2_shell_path}")

    # k = 3 (approx 99.7% CI)
    k3_r = mean_r_gp + 3.0 * std_r_gp
    k3_xyz, k3_rgb = build_shell(k3_r, (0, 0, 255))       # blue
    k3_shell_path = os.path.join(MODEL_DIR, "gp_k3_shell.ply")
    write_ply(k3_shell_path, k3_xyz, k3_rgb)
    print(f"[Stage3.1] Saved GP k=3 shell: {k3_shell_path}")

    print("\n[Stage3.1] GP-based radial shape modeling completed.")


if __name__ == "__main__":
    main()
