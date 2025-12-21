#!/usr/bin/env python3
"""
Stage-3.3 — Probabilistic Safety Distance Shells
================================================

Goal
----
Build FINAL static safety shells around the robot using:
  - GP radial model (Stage-3.1)
  - Uncertainty (std over directions)
  - Extra fixed safety margin (d_margin)

Formula:
  r_safety(θ, φ) = μ_r(θ, φ) + k * σ_r(θ, φ) + d_margin

With:
  - caps on std
  - caps on overall inflation vs base ellipsoid

Inputs
------
MODEL_DIR = pipeline_rgb/shape/model_stage2

  shape_params.json
    - master_rotation: 3x3
    - master_axes: [a, b, c]

  gp_radial_stats.npz  (from Stage-3.1)
    - theta_pred:   [n_theta]
    - phi_pred:     [n_phi]
    - mean_r_gp:    [n_theta, n_phi]
    - std_r_gp:     [n_theta, n_phi]

Outputs
-------
In MODEL_DIR:
  safety_shell_95_static.ply
  safety_shell_99_static.ply

These are GLOBAL-frame shells (centered at origin, oriented with master_R).
"""

import os
import json
import numpy as np
from plyfile import PlyData, PlyElement

MODEL_DIR = "pipeline_rgb/shape/model_stage2"


# ------------------------------------------------------------
# Write PLY utility
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Spherical to unit vector
# ------------------------------------------------------------
def spherical_to_unit(theta, phi):
    """
    theta: azimuth in [-π, π]
    phi:   polar in [0, π], 0 at +Z
    """
    ux = np.cos(theta) * np.sin(phi)
    uy = np.sin(theta) * np.sin(phi)
    uz = np.cos(phi)
    return ux, uy, uz


# ------------------------------------------------------------
# Main Stage-3.3 logic
# ------------------------------------------------------------
def main():
    # -------------------------------
    # 1. Load shape parameters
    # -------------------------------
    params_path = os.path.join(MODEL_DIR, "shape_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Cannot find shape_params.json at {params_path}")

    with open(params_path, "r") as f:
        params = json.load(f)

    master_R = np.array(params["master_rotation"], dtype=np.float32)
    master_axes = np.array(params["master_axes"], dtype=np.float32)  # [a,b,c]

    # -------------------------------
    # 2. Load GP radial stats
    # -------------------------------
    gp_stats_path = os.path.join(MODEL_DIR, "gp_radial_stats.npz")
    if not os.path.exists(gp_stats_path):
        raise FileNotFoundError(
            f"Cannot find gp_radial_stats.npz at {gp_stats_path}. "
            "Run Stage-3.1 first."
        )

    data = np.load(gp_stats_path)
    theta_pred = data["theta_pred"]      # [n_theta]
    phi_pred = data["phi_pred"]          # [n_phi]
    mean_r_gp = data["mean_r_gp"]        # [n_theta, n_phi]
    std_r_gp = data["std_r_gp"]          # [n_theta, n_phi]

    n_theta = theta_pred.shape[0]
    n_phi = phi_pred.shape[0]

    print(f"[Stage3.3] GP radial grid: {n_theta} x {n_phi}")

    # Create meshgrid
    theta_grid, phi_grid = np.meshgrid(theta_pred, phi_pred, indexing="ij")

    # -------------------------------
    # 3. Base ellipsoid radius per direction
    # -------------------------------
    ux, uy, uz = spherical_to_unit(theta_grid, phi_grid)

    a, b, c = master_axes
    denom = (ux / a) ** 2 + (uy / b) ** 2 + (uz / c) ** 2
    denom = np.maximum(denom, 1e-8)
    ellipsoid_r = 1.0 / np.sqrt(denom)  # [n_theta, n_phi]

    # -------------------------------
    # 4. Clean & cap std
    # -------------------------------
    mean_r = mean_r_gp.copy()
    std_r = std_r_gp.copy()

    # Negative or NaN std: fix
    std_r = np.nan_to_num(std_r, nan=0.0, posinf=0.0, neginf=0.0)
    std_r[std_r < 0] = 0.0

    # Optional: baseline minimum mean radius – avoid division by 0
    mean_floor = np.maximum(mean_r, 1e-3)

    # Cap std so it cannot exceed a fraction of mean
    std_cap_factor = 0.3  # 30% of mean radius
    std_r = np.minimum(std_r, std_cap_factor * mean_floor)

    # For directions with extremely few data, we might want a small baseline std
    # (already implicit in GP regularization; here we keep it as-is)

    # -------------------------------
    # 5. Safety margin parameters
    # -------------------------------
    # Extra constant safety margin [m] – adjust for your scenario:
    d_margin = 0.05  # 5 cm, can be tuned up for stricter safety

    # Inflation cap vs base ellipsoid:
    max_scale_95 = 1.3  # 30% larger than ellipsoid for 95%
    max_scale_99 = 1.5  # 50% larger for 99%

    # Confidence levels:
    CONFIGS = [
        {
            "name": "95",
            "k": 2.0,
            "color": (255, 0, 0),
            "max_scale": max_scale_95,
            "outfile": os.path.join(MODEL_DIR, "safety_shell_95_static.ply"),
        },
        {
            "name": "99",
            "k": 3.0,
            "color": (0, 0, 255),
            "max_scale": max_scale_99,
            "outfile": os.path.join(MODEL_DIR, "safety_shell_99_static.ply"),
        },
    ]

    # -------------------------------
    # 6. Build safety shells
    # -------------------------------
    for cfg in CONFIGS:
        name = cfg["name"]
        k = cfg["k"]
        color = cfg["color"]
        max_scale = cfg["max_scale"]
        outfile = cfg["outfile"]

        # Raw safety radius (before cap):
        raw_r = mean_r + k * std_r + d_margin

        # Cap vs ellipsoid
        r_safe = np.minimum(raw_r, max_scale * ellipsoid_r)

        # Convert to xyz in local frame
        x = r_safe * np.cos(theta_grid) * np.sin(phi_grid)
        y = r_safe * np.sin(theta_grid) * np.sin(phi_grid)
        z = r_safe * np.cos(phi_grid)

        pts_local = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # Transform to global frame using master_R
        pts_world = pts_local @ master_R

        rgb = np.tile(np.array(color, dtype=np.uint8), (pts_world.shape[0], 1))

        write_ply(outfile, pts_world, rgb)
        print(f"[Stage3.3] Saved {name}% safety shell → {outfile}")

    print("\n[Stage3.3] Probabilistic safety shells generated successfully.")


if __name__ == "__main__":
    main()
