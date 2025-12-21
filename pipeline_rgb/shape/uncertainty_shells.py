#!/usr/bin/env python3
"""
Stage-2.3 — Uncertainty-based Shape Expansion (Radial Shells)
=============================================================

Goal
----
Use all robot-only frames to estimate:
  - radial mean r(θ, φ)
  - radial std σ(θ, φ)
and build uncertainty-inflated shells around the global ellipsoid.

Inputs
------
1) Global shape parameters (from Stage-2.2):
   pipeline_rgb/shape/model_stage2/shape_params.json
   with keys:
     - master_axes      (ellipsoid radii [a, b, c])
     - master_dims      (OBB dims, not directly used here)
     - master_rotation  (3x3 rotation matrix, global orientation)

2) Robot-only overlays (from highlight_robot / Stage-2.1):
   pipeline_rgb/shape/highlighted_uncertainty/
     frame_XXXX_overlay_go1_uncertainty_clean.ply
   where robot points are those with green == 255.

Outputs
-------
In: pipeline_rgb/shape/model_stage2/

  radial_stats.npz              # θ, φ grids, mean_r, std_r, counts
  radial_mean_shell.ply         # mean shell
  radial_k2_shell.ply           # mean + 2σ (~95% CI)
  radial_k3_shell.ply           # mean + 3σ (~99.7% CI)

These shells are expressed in the same global frame as ellipsoid_mean.ply.
"""

import os
import json
import numpy as np
from plyfile import PlyData, PlyElement

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
MODEL_DIR = "pipeline_rgb/shape/model_stage2"
ROBOT_DIR = "pipeline_rgb/shape/highlighted_uncertainty"

OUT_DIR = MODEL_DIR  # keep outputs together


# -------------------------------------------------------------------
# I/O utilities
# -------------------------------------------------------------------
def load_xyz_and_colors(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32)
    rgb = np.vstack((v["red"], v["green"], v["blue"])).T.astype(np.uint8)
    return xyz, rgb


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


# -------------------------------------------------------------------
# Spherical binning helpers
# -------------------------------------------------------------------
def spherical_coordinates(points):
    """
    Convert 3D points to spherical (r, theta, phi) in local frame.

    r     = ||p||
    theta = atan2(y, x) in [-pi, pi]
    phi   = arccos(z / r) in [0, pi]  (0 = +Z pole)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)

    # avoid division by zero
    eps = 1e-8
    r_safe = np.maximum(r, eps)

    theta = np.arctan2(y, x)
    cos_phi = np.clip(z / r_safe, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    return r, theta, phi


def create_direction_grid(n_theta=48, n_phi=24):
    """
    Create grid centers for θ, φ:

      theta in [-π, π]
      phi   in [0, π]

    Returns:
      theta_centers, phi_centers, theta_edges, phi_edges
    """
    theta_edges = np.linspace(-np.pi, np.pi, n_theta + 1)
    phi_edges = np.linspace(0.0, np.pi, n_phi + 1)

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    return theta_centers, phi_centers, theta_edges, phi_edges


# -------------------------------------------------------------------
# Main Stage-2.3 logic
# -------------------------------------------------------------------
def main():
    # ---------------------------------------------------------------
    # 1. Load global shape parameters
    # ---------------------------------------------------------------
    params_path = os.path.join(MODEL_DIR, "shape_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Cannot find shape_params.json at {params_path}")

    with open(params_path, "r") as f:
        params = json.load(f)

    master_axes = np.array(params["master_axes"], dtype=np.float32)      # [a, b, c]
    master_R = np.array(params["master_rotation"], dtype=np.float32)     # 3x3
    # master_dims = np.array(params["master_dims"], dtype=np.float32)    # not used here

    # We will work in the global frame, but for radial statistics
    # we define a "local" frame aligned with master_R.
    R_T = master_R.T  # to go world -> local

    # ---------------------------------------------------------------
    # 2. Build direction grid
    # ---------------------------------------------------------------
    n_theta = 48
    n_phi = 24
    theta_centers, phi_centers, theta_edges, phi_edges = create_direction_grid(
        n_theta=n_theta, n_phi=n_phi
    )

    # Accumulators
    # shape: [n_theta, n_phi]
    sum_r = np.zeros((n_theta, n_phi), dtype=np.float64)
    sum_r2 = np.zeros((n_theta, n_phi), dtype=np.float64)
    counts = np.zeros((n_theta, n_phi), dtype=np.int64)

    # ---------------------------------------------------------------
    # 3. Loop over robot frames and accumulate radial stats
    # ---------------------------------------------------------------
    files = sorted(
        f
        for f in os.listdir(ROBOT_DIR)
        if f.startswith("frame_") and f.endswith("_overlay_go1_uncertainty_clean.ply")
    )

    if not files:
        raise RuntimeError(f"No robot overlay files found in {ROBOT_DIR}")

    print(f"[Stage2.3] Found {len(files)} robot overlay frames.")

    for fi, fname in enumerate(files):
        path = os.path.join(ROBOT_DIR, fname)
        xyz, rgb = load_xyz_and_colors(path)

        # Select robot points (green == 255)
        mask_robot = (rgb[:, 1] == 255)
        pts_robot = xyz[mask_robot]

        if pts_robot.shape[0] < 20:
            # too few points, skip
            continue

        # Compute per-frame robot centroid in world frame
        centroid = pts_robot.mean(axis=0, keepdims=True)

        # Express points in local shape frame:
        #   1) subtract centroid (body center of that frame)
        #   2) rotate by R_T so that "robot body axes" align with master_R
        pts_local = (pts_robot - centroid) @ R_T

        # Convert to spherical (r, theta, phi)
        r, theta, phi = spherical_coordinates(pts_local)

        # Filter out very small radii (almost-centroid) to avoid numerical noise
        valid = r > 1e-3
        r = r[valid]
        theta = theta[valid]
        phi = phi[valid]

        if r.size == 0:
            continue

        # Determine bin indices
        # theta ∈ [-π, π]
        theta_idx = np.searchsorted(theta_edges, theta, side="right") - 1
        phi_idx = np.searchsorted(phi_edges, phi, side="right") - 1

        # Clamp to valid range
        theta_idx = np.clip(theta_idx, 0, n_theta - 1)
        phi_idx = np.clip(phi_idx, 0, n_phi - 1)

        # Accumulate sums per bin
        for t_i, p_i, ri in zip(theta_idx, phi_idx, r):
            sum_r[t_i, p_i] += ri
            sum_r2[t_i, p_i] += ri * ri
            counts[t_i, p_i] += 1

        if (fi + 1) % 50 == 0 or fi == len(files) - 1:
            print(f"[Stage2.3] Processed {fi + 1}/{len(files)} frames.")

    # ---------------------------------------------------------------
    # 4. Compute mean and std radius per direction
    # ---------------------------------------------------------------
    mean_r = np.zeros_like(sum_r, dtype=np.float64)
    std_r = np.zeros_like(sum_r, dtype=np.float64)

    valid_bins = counts > 0
    mean_r[valid_bins] = sum_r[valid_bins] / counts[valid_bins]

    var_r = np.zeros_like(sum_r)
    var_r[valid_bins] = sum_r2[valid_bins] / counts[valid_bins] - mean_r[valid_bins] ** 2
    var_r[var_r < 0.0] = 0.0  # numerical safety
    std_r[valid_bins] = np.sqrt(var_r[valid_bins])

    # For bins with no data, fall back to ellipsoid radius formula
    # radius of ellipsoid with axes (a,b,c) along unit direction u:
    #   1/r^2 = (u_x/a)^2 + (u_y/b)^2 + (u_z/c)^2
    # => r = 1 / sqrt(...)
    theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers, indexing="ij")

    u_x = np.cos(theta_grid) * np.sin(phi_grid)
    u_y = np.sin(theta_grid) * np.sin(phi_grid)
    u_z = np.cos(phi_grid)

    a, b, c = master_axes
    denom = (u_x / a) ** 2 + (u_y / b) ** 2 + (u_z / c) ** 2
    denom = np.maximum(denom, 1e-8)
    ellipsoid_r = 1.0 / np.sqrt(denom)

    # Fill empty bins with ellipsoid radius and a small baseline std
    empty_bins = ~valid_bins
    mean_r[empty_bins] = ellipsoid_r[empty_bins]
    # baseline std = 5% of ellipsoid radius
    std_r[empty_bins] = 0.05 * ellipsoid_r[empty_bins]

    # ---------------------------------------------------------------
    # 5. Save radial statistics
    # ---------------------------------------------------------------
    stats_path = os.path.join(OUT_DIR, "radial_stats.npz")
    np.savez_compressed(
        stats_path,
        theta_centers=theta_centers,
        phi_centers=phi_centers,
        mean_r=mean_r,
        std_r=std_r,
        counts=counts,
    )
    print(f"[Stage2.3] Saved radial stats to {stats_path}")

    # ---------------------------------------------------------------
    # 6. Build shells: mean, k=2 (95%), k=3 (~99.7%)
    # ---------------------------------------------------------------
    def build_shell(r_field, color):
        """
        r_field: [n_theta, n_phi] radii in local frame.
        color: (R,G,B)
        Returns xyz, rgb in GLOBAL frame (using master_R).
        """
        # Use the same theta_grid, phi_grid defined above
        x = r_field * np.cos(theta_grid) * np.sin(phi_grid)
        y = r_field * np.sin(theta_grid) * np.sin(phi_grid)
        z = r_field * np.cos(phi_grid)

        pts_local = np.stack((x, y, z), axis=-1)     # [n_theta, n_phi, 3]
        pts_local_flat = pts_local.reshape(-1, 3)    # [N, 3]

        # Transform to world frame via master_R
        pts_world = pts_local_flat @ master_R       # [N, 3]

        rgb = np.tile(np.array(color, dtype=np.uint8), (pts_world.shape[0], 1))
        return pts_world, rgb

    # Mean shell
    mean_xyz, mean_rgb = build_shell(mean_r, color=(255, 255, 0))  # yellow
    mean_shell_path = os.path.join(OUT_DIR, "radial_mean_shell.ply")
    write_ply(mean_shell_path, mean_xyz, mean_rgb)
    print(f"[Stage2.3] Saved mean shell: {mean_shell_path}")

    # k = 2 (approx 95% CI if Gaussian)
    k2_r = mean_r + 2.0 * std_r
    k2_xyz, k2_rgb = build_shell(k2_r, color=(255, 0, 0))  # red
    k2_shell_path = os.path.join(OUT_DIR, "radial_k2_shell.ply")
    write_ply(k2_shell_path, k2_xyz, k2_rgb)
    print(f"[Stage2.3] Saved k=2 shell: {k2_shell_path}")

    # k = 3 (approx 99.7% CI)
    k3_r = mean_r + 3.0 * std_r
    k3_xyz, k3_rgb = build_shell(k3_r, color=(0, 0, 255))  # blue
    k3_shell_path = os.path.join(OUT_DIR, "radial_k3_shell.ply")
    write_ply(k3_shell_path, k3_xyz, k3_rgb)
    print(f"[Stage2.3] Saved k=3 shell: {k3_shell_path}")

    print("\n[Stage2.3] Uncertainty shells generated successfully.")


if __name__ == "__main__":
    main()
