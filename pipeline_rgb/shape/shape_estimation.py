#!/usr/bin/env python3
"""
Stage-2.2 — Global Shape Estimation from Good Frames Only
=========================================================

This script:
  1. Reads robot-only frames from:
         pipeline_rgb/shape/highlighted_uncertainty/
  2. For each frame, computes:
         - centroid
         - PCA axes (axis1, axis2, axis3)
         - PCA eigenvectors
         - OBB dimensions (1.7 * axes)
  3. Automatically selects 'good' frames where the robot shape
     is small and plausible (no walls/columns).
  4. Learns a single GLOBAL robot shape:
         - master_axes (ellipsoid radii)
         - master_dims (OBB side lengths)
         - master_R (orientation)
  5. Saves:
         pipeline_rgb/shape/model_stage2/
             - all_frames_raw_stats.csv
             - shape_params.json
             - ellipsoid_mean.ply
             - obb_mean.ply
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement

ROOT_DIR = "pipeline_rgb/shape/highlighted_uncertainty"
OUT_DIR  = "pipeline_rgb/shape/model_stage2"
os.makedirs(OUT_DIR, exist_ok=True)


def load_robot_frames(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".ply")])
    frames = []

    for fname in files:
        path = os.path.join(folder, fname)
        ply = PlyData.read(path)
        v = ply["vertex"].data

        xyz = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32)
        mask = (v["green"] == 255)  # robot only
        robot_pts = xyz[mask]

        if robot_pts.shape[0] > 25:
            frames.append((fname, robot_pts))

    return frames


def compute_pca_shape(pts):
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    pca = PCA(n_components=3)
    pca.fit(centered)

    eigvecs = pca.components_
    eigvals = pca.explained_variance_
    axes = 2.0 * np.sqrt(eigvals)  # radii-like

    return centroid, axes, eigvecs


def save_ellipsoid(center, axes, R, out_path):
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)

    x = axes[0] * np.outer(np.cos(u), np.sin(v))
    y = axes[1] * np.outer(np.sin(u), np.cos(v))
    z = axes[2] * np.outer(np.ones_like(u), np.sin(v))

    pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    pts = pts @ R + center

    vertex = np.empty(
        pts.shape[0],
        dtype=[("x","f4"),("y","f4"),("z","f4"),
               ("red","u1"),("green","u1"),("blue","u1")]
    )
    vertex["x"], vertex["y"], vertex["z"] = pts[:,0], pts[:,1], pts[:,2]
    vertex["red"], vertex["green"], vertex["blue"] = 200, 200, 0

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(out_path)


def save_obb(center, dims, R, out_path):
    dx, dy, dz = dims / 2.0
    corners = np.array([
        [ dx, dy, dz], [ dx, dy,-dz], [ dx,-dy, dz], [ dx,-dy,-dz],
        [-dx, dy, dz], [-dx, dy,-dz], [-dx,-dy, dz], [-dx,-dy,-dz],
    ])

    pts = corners @ R + center

    vertex = np.empty(
        pts.shape[0],
        dtype=[("x","f4"),("y","f4"),("z","f4"),
               ("red","u1"),("green","u1"),("blue","u1")]
    )
    vertex["x"], vertex["y"], vertex["z"] = pts[:,0], pts[:,1], pts[:,2]
    vertex["red"], vertex["green"], vertex["blue"] = 255, 0, 0

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(out_path)


def stage2_shape_estimation():
    frames = load_robot_frames(ROOT_DIR)
    if not frames:
        print("No robot frames found.")
        return

    print(f"[ShapeEst] Found {len(frames)} frames with robot points.")

    # Collect per-frame stats
    names = []
    centroids = []
    axes_list = []
    eig_list = []
    dims_list = []
    rows = []

    for fname, pts in frames:
        c, axes, R = compute_pca_shape(pts)
        dims = axes * 1.7
        vol  = float(dims[0] * dims[1] * dims[2])

        names.append(fname)
        centroids.append(c)
        axes_list.append(axes)
        eig_list.append(R)
        dims_list.append(dims)

        rows.append({
            "frame": fname,
            "num_points": pts.shape[0],
            "centroid_x": c[0],
            "centroid_y": c[1],
            "centroid_z": c[2],
            "axis1": axes[0],
            "axis2": axes[1],
            "axis3": axes[2],
            "obb_dx": dims[0],
            "obb_dy": dims[1],
            "obb_dz": dims[2],
            "obb_volume": vol
        })

    df = pd.DataFrame(rows)
    raw_csv_path = os.path.join(OUT_DIR, "all_frames_raw_stats.csv")
    df.to_csv(raw_csv_path, index=False)
    print(f"[ShapeEst] Saved raw stats to {raw_csv_path}")

    centroids = np.vstack(centroids)
    axes_arr  = np.vstack(axes_list)
    dims_arr  = np.vstack(dims_list)
    eig_arr   = np.stack(eig_list)

    # -------- Choose 'good' frames by small axis1 (robot-scale) --------
    axis1 = axes_arr[:, 0]
    num_pts = df["num_points"].values

    small_mask = axis1 < 1.0  # small-ish in data units
    small_axis1 = axis1[small_mask]

    if small_axis1.size > 10:
        a1_thr = np.median(small_axis1) * 1.5
    else:
        a1_thr = np.median(axis1) * 1.5

    good_mask = (axis1 <= a1_thr) & (num_pts >= 40)
    good_idx = np.where(good_mask)[0]

    print(f"[ShapeEst] Good frames (shape prior): {len(good_idx)} / {len(frames)}")
    if len(good_idx) == 0:
        print("[ShapeEst] No good frames found, using all frames as fallback.")
        good_idx = np.arange(len(frames))

    C_good = centroids[good_idx]
    A_good = axes_arr[good_idx]
    D_good = dims_arr[good_idx]
    R_good = eig_arr[good_idx]

    master_centroid = C_good.mean(axis=0)
    master_axes     = np.median(A_good, axis=0)
    master_dims     = np.median(D_good, axis=0)

    # average rotation then orthonormalize
    R_mean = R_good.mean(axis=0)
    U, _, _ = np.linalg.svd(R_mean)
    master_R = U

    # Save shape parameters
    shape_params = {
        "master_centroid": master_centroid.tolist(),
        "master_axes": master_axes.tolist(),
        "master_dims": master_dims.tolist(),
        "master_rotation": master_R.tolist(),
        "axis1_threshold": float(a1_thr),
        "num_good_frames": int(len(good_idx)),
    }

    params_path = os.path.join(OUT_DIR, "shape_params.json")
    with open(params_path, "w") as f:
        json.dump(shape_params, f, indent=4)
    print(f"[ShapeEst] Saved shape_params.json to {params_path}")

    # Save global mean ellipsoid & OBB (centered at master_centroid)
    ell_path = os.path.join(OUT_DIR, "ellipsoid_mean.ply")
    obb_path = os.path.join(OUT_DIR, "obb_mean.ply")
    save_ellipsoid(master_centroid, master_axes, master_R, ell_path)
    save_obb(master_centroid, master_dims, master_R, obb_path)

    print("[ShapeEst] Global shape estimation finished.")


if __name__ == "__main__":
    stage2_shape_estimation()
