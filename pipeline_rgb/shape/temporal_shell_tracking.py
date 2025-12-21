#!/usr/bin/env python3
"""
Stage-3.2 — Temporal Shape Tracking with SAFE, BOUNDED Shells
==============================================================

This is the FIXED version with:
  - capped std
  - capped shell inflation
  - reduced default uncertainty
  - sliding-window temporal smoothing

It produces per-frame safety shells that do NOT blow up in size.
"""

import os
import json
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement

MODEL_DIR = "pipeline_rgb/shape/model_stage2"
ROBOT_DIR = "pipeline_rgb/shape/highlighted_uncertainty"
OUT_SHELL_DIR = "pipeline_rgb/shape/per_frame_gp_shells"
os.makedirs(OUT_SHELL_DIR, exist_ok=True)


# ---------------------------------------------------------------
# PLY utilities
# ---------------------------------------------------------------
def load_xyz_rgb(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    if "red" in v.dtype.names:
        rgb = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.uint8)
    else:
        rgb = np.tile(np.array([150, 150, 150], np.uint8), (xyz.shape[0], 1))

    return xyz, rgb


def write_ply(path, xyz, rgb):
    arr = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1")
        ]
    )
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)


# ---------------------------------------------------------------
# Spherical conversion
# ---------------------------------------------------------------
def spherical_coordinates(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)

    eps = 1e-8
    r_safe = np.maximum(r, eps)

    theta = np.arctan2(y, x)        # [-pi, pi]
    phi = np.arccos(np.clip(z / r_safe, -1, 1))  # [0, pi]

    return r, theta, phi


def create_bins(n_theta=48, n_phi=24):
    theta_edges = np.linspace(-np.pi, np.pi, n_theta + 1)
    phi_edges = np.linspace(0, np.pi, n_phi + 1)
    theta_cent = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_cent = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    return theta_cent, phi_cent, theta_edges, phi_edges


# ---------------------------------------------------------------
# Build a shell for frame t
# ---------------------------------------------------------------
def build_shell(r_field, theta_cent, phi_cent, master_R, centroid, color):
    th, ph = np.meshgrid(theta_cent, phi_cent, indexing="ij")

    x = r_field * np.cos(th) * np.sin(ph)
    y = r_field * np.sin(th) * np.sin(ph)
    z = r_field * np.cos(ph)

    pts_local = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    pts_world = pts_local @ master_R + centroid

    rgb = np.tile(np.array(color, np.uint8), (pts_world.shape[0], 1))
    return pts_world, rgb


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():

    # -------------------------------
    # Load shape params
    # -------------------------------
    with open(f"{MODEL_DIR}/shape_params.json") as f:
        params = json.load(f)

    master_R = np.array(params["master_rotation"], float)
    master_axes = np.array(params["master_axes"], float)

    R_T = master_R.T

    # -------------------------------
    # Binning grid
    # -------------------------------
    n_theta, n_phi = 48, 24
    theta_cent, phi_cent, theta_edges, phi_edges = create_bins(n_theta, n_phi)

    # -------------------------------
    # Ellipsoid fallback radius
    # -------------------------------
    th, ph = np.meshgrid(theta_cent, phi_cent, indexing="ij")
    ux = np.cos(th) * np.sin(ph)
    uy = np.sin(th) * np.sin(ph)
    uz = np.cos(ph)

    a, b, c = master_axes
    denom = (ux / a) ** 2 + (uy / b) ** 2 + (uz / c) ** 2
    denom = np.maximum(denom, 1e-8)
    ellipsoid_r = 1 / np.sqrt(denom)

    # -------------------------------
    # Sliding window storage
    # -------------------------------
    WINDOW = 15  # smaller → tighter shell
    sum_r_H = []
    sum_r2_H = []
    count_H = []
    centroids = []

    files = sorted([f for f in os.listdir(ROBOT_DIR)
                    if f.endswith("_overlay_go1_uncertainty_clean.ply")])

    os.makedirs(OUT_SHELL_DIR, exist_ok=True)

    log_rows = []

    # -----------------------------------------------------------
    # Loop through time
    # -----------------------------------------------------------
    for idx, fname in enumerate(files):
        fid = fname.split("_")[1]  # frame_XXXX_...

        xyz, rgb = load_xyz_rgb(f"{ROBOT_DIR}/{fname}")
        robot_mask = (rgb[:, 1] == 255)
        pts = xyz[robot_mask]

        if pts.shape[0] < 20:
            # store dummy
            sum_r_H.append(np.zeros((n_theta, n_phi)))
            sum_r2_H.append(np.zeros((n_theta, n_phi)))
            count_H.append(np.zeros((n_theta, n_phi), int))
            centroids.append(None)
            continue

        centroid = pts.mean(axis=0)
        centroids.append(centroid)

        # Transform into local frame aligned with master_R
        pts_local = (pts - centroid) @ R_T

        r, theta, phi = spherical_coordinates(pts_local)
        valid = r > 1e-3
        r, theta, phi = r[valid], theta[valid], phi[valid]

        # Per-frame radial accumulators
        sum_r = np.zeros((n_theta, n_phi))
        sum_r2 = np.zeros((n_theta, n_phi))
        count = np.zeros((n_theta, n_phi), int)

        if r.size > 0:
            ti = np.searchsorted(theta_edges, theta, side="right") - 1
            pi = np.searchsorted(phi_edges, phi, side="right") - 1

            ti = np.clip(ti, 0, n_theta - 1)
            pi = np.clip(pi, 0, n_phi - 1)

            for t_i, p_i, ri in zip(ti, pi, r):
                sum_r[t_i, p_i] += ri
                sum_r2[t_i, p_i] += ri * ri
                count[t_i, p_i] += 1

        sum_r_H.append(sum_r)
        sum_r2_H.append(sum_r2)
        count_H.append(count)

        # --------------------------
        # Sliding window aggregation
        # --------------------------
        start = max(0, idx - WINDOW + 1)
        end = idx + 1

        w_sum = np.zeros((n_theta, n_phi))
        w_sum2 = np.zeros((n_theta, n_phi))
        w_cnt = np.zeros((n_theta, n_phi), int)

        for j in range(start, end):
            w_sum += sum_r_H[j]
            w_sum2 += sum_r2_H[j]
            w_cnt += count_H[j]

        valid_bins = w_cnt > 0
        mean_r = np.zeros_like(w_sum)
        std_r = np.zeros_like(w_sum)

        mean_r[valid_bins] = w_sum[valid_bins] / w_cnt[valid_bins]
        var_r = w_sum2[valid_bins] / w_cnt[valid_bins] - mean_r[valid_bins] ** 2
        var_r[var_r < 0] = 0
        std_r[valid_bins] = np.sqrt(var_r)

        # --------------------------
        # FIX #1: Small baseline std
        # --------------------------
        empty_bins = ~valid_bins
        std_r[empty_bins] = 0.01 * ellipsoid_r[empty_bins]  # was 0.05

        # --------------------------
        # FIX #2: Cap std (no explosion)
        # --------------------------
        std_cap_factor = 0.25  # max 25% of mean radius
        std_r = np.minimum(std_r, std_cap_factor * np.maximum(mean_r, 1e-3))

        # --------------------------
        # FIX #3: Cap safe shell vs ellipsoid
        # --------------------------
        k = 1.5  # slightly lower inflation
        raw_safe = mean_r + k * std_r

        max_scale = 1.25  # max 25% above ellipsoid
        r_safe = np.minimum(raw_safe, max_scale * ellipsoid_r)

        # --------------------------
        # Build shell for this frame
        # --------------------------
        if centroids[-1] is None:
            continue

        C = centroids[-1]

        shell_xyz, shell_rgb = build_shell(
            r_safe, theta_cent, phi_cent, master_R, C, (255, 0, 0)
        )

        out_path = f"{OUT_SHELL_DIR}/gp_shell_k2_frame_{fid}.ply"
        write_ply(out_path, shell_xyz, shell_rgb)

        print(f"[OK] Frame {fid} → shell saved")

        log_rows.append({
            "frame": fid,
            "centroid_x": C[0],
            "centroid_y": C[1],
            "centroid_z": C[2],
            "robot_points": pts.shape[0],
            "window_start": start,
            "window_end": end - 1,
        })

    # Save log
    df = pd.DataFrame(log_rows)
    df.to_csv(f"{MODEL_DIR}/temporal_shell_log.csv", index=False)

    print("\n[Stage 3.2] DONE: Tight, bounded GP shells created.")


if __name__ == "__main__":
    main()
