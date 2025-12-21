#!/usr/bin/env python3
"""
Uncertainty-aware RoboDog Highlight (Enhanced Filtering)
========================================================

Updates for "Mini Factory" Removal:
  1. AGGRESSIVE STATIC FILTER: min_frame_ratio lowered to 0.25.
     Any "robot" voxels appearing in >25% of frames are considered static noise.
  2. STRICTER SIZE FILTER: Min dimensions increased to [0.3, 0.2, 0.2].
     Removes small debris/items misclassified as robot.

Inputs:
  - pred_dir: pipeline_rgb/inference/predictions
  - unc_dir:  pipeline_rgb/inference/uncertainty_mc
Outputs:
  - out_dir:  pipeline_rgb/shape/highlighted_uncertainty
"""

import os
from typing import Optional, Dict, Tuple, Set
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN


# ==============================================================
# Basic I/O
# ==============================================================

def read_pred_ply(path: str):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    pts = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32)
    labels = np.asarray(v["pred"], dtype=np.int32)
    return pts, labels


def read_uncertainty_ply(path: str):
    if not os.path.exists(path):
        return None
    ply = PlyData.read(path)
    v = ply["vertex"].data
    mean_prob = np.asarray(v["mean_prob"], dtype=np.float32)
    entropy   = np.asarray(v["entropy"], dtype=np.float32)
    mc_var    = np.asarray(v["mc_var"], dtype=np.float32)
    tta_var   = np.asarray(v["tta_var"], dtype=np.float32)
    aleatoric = np.asarray(v["aleatoric"], dtype=np.float32)
    return mean_prob, entropy, mc_var, tta_var, aleatoric


def write_overlay(path: str, pts: np.ndarray, robot_idx: np.ndarray):
    N = pts.shape[0]
    vertex = np.empty(
        N,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = pts[:, 0], pts[:, 1], pts[:, 2]

    # default grey
    colors = np.tile(np.array([150, 150, 150], dtype=np.uint8), (N, 1))

    # robot green
    if robot_idx.size > 0:
        colors[robot_idx] = np.array([0, 255, 0], dtype=np.uint8)

    vertex["red"], vertex["green"], vertex["blue"] = (
        colors[:, 0], colors[:, 1], colors[:, 2]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


# ==============================================================
# Geometry Helpers
# ==============================================================

def estimate_floor_height(pts: np.ndarray) -> float:
    return float(np.percentile(pts[:, 2], 5))


def is_column_like(dims: np.ndarray) -> bool:
    dx, dy, dz = dims
    base = max(dx, dy)
    return dz > 1.2 and base < 0.5


def is_wall_like(dims: np.ndarray) -> bool:
    dx, dy, dz = dims
    longest = max(dx, dy)
    thinnest = min(dx, dy)
    return longest > 2.0 and thinnest < 0.2 and dz > 1.0


# ==============================================================
# Pass 1: Build Static Misclassification Mask
# ==============================================================

def build_static_robot_voxels(
    pred_dir: str,
    files,
    robo_label: int,
    voxel_size: float = 0.10,
    min_frame_ratio: float = 0.25, # <--- UPDATED: More aggressive (was 0.7)
) -> Set[Tuple[int, int]]:
    """
    Identifies voxels where 'robot' points appear persistently.
    If a voxel contains robot points in >25% of frames, it's assumed to be 
    a static object (like the mini factory) and is masked out.
    """
    counts: Dict[Tuple[int, int], int] = {}
    num_frames = len(files)

    print(f"[Pass1] Learning static background noise (Threshold: {int(min_frame_ratio*100)}%)...")

    for i, fname in enumerate(files):
        pts, labels = read_pred_ply(os.path.join(pred_dir, fname))
        robo_idx = np.where(labels == robo_label)[0]
        if robo_idx.size == 0:
            continue

        xy = pts[robo_idx, :2]
        vox = np.floor(xy / voxel_size).astype(np.int32)
        uniq_vox = np.unique(vox, axis=0)

        for ix, iy in uniq_vox:
            key = (int(ix), int(iy))
            counts[key] = counts.get(key, 0) + 1
        
        if (i+1) % 50 == 0:
             print(f"  Scanned {i+1}/{num_frames} frames", end="\r")

    print(f"\n[Pass1] Scan complete.")

    threshold = int(min_frame_ratio * num_frames)
    static_voxels = {k for k, c in counts.items() if c >= threshold}

    print(f"[Pass1] Masked {len(static_voxels)} static voxels (Mini Factory/Debris).")
    return static_voxels


# ==============================================================
# Uncertainty Filter
# ==============================================================

def apply_uncertainty_filters_soft(
    robo_idx: np.ndarray,
    unc: Optional[tuple],
    min_prob: float = 0.55,
    max_entropy: float = 1.3,
    max_mc: float = 0.05,
    max_tta: float = 0.05,
) -> np.ndarray:
    
    if unc is None or robo_idx.size == 0:
        return robo_idx

    mean_prob, entropy, mc_var, tta_var, aleatoric = unc

    keep_mask = (
        (mean_prob[robo_idx] > min_prob) &
        (entropy[robo_idx]   < max_entropy) &
        (mc_var[robo_idx]    < max_mc) &
        (tta_var[robo_idx]   < max_tta)
    )

    filtered = robo_idx[keep_mask]

    # Fallback if we removed too much (safety)
    if filtered.size < 30 or filtered.size < 0.4 * robo_idx.size:
        return robo_idx

    return filtered


# ==============================================================
# Pass 2: Per-frame Extraction
# ==============================================================

def extract_robot_indices(
    pts: np.ndarray,
    labels: np.ndarray,
    last_centroid: Optional[np.ndarray],
    static_robot_voxels: Set[Tuple[int, int]],
    robo_label: int = 3,
    voxel_size: float = 0.10,
    unc: Optional[tuple] = None,
):
    # 1. Select robot points
    robo_idx = np.where(labels == robo_label)[0]
    if robo_idx.size == 0:
        return np.array([], dtype=np.int64), last_centroid

    # 2. Remove static voxels (The Mini Factory Fix)
    if static_robot_voxels:
        xy = pts[robo_idx, :2]
        vox = np.floor(xy / voxel_size).astype(np.int32)
        # Keep only if voxel NOT in static set
        keep_mask = [(int(ix), int(iy)) not in static_robot_voxels for ix, iy in vox]
        robo_idx = robo_idx[np.array(keep_mask, dtype=bool)]

    # 3. Fallback check
    if robo_idx.size == 0:
        # If we cleaned everything, try reverting (unless it's purely static noise)
        # But for mini factory, we prefer empty over noise.
        return np.array([], dtype=np.int64), last_centroid

    # 4. Uncertainty
    robo_idx = apply_uncertainty_filters_soft(robo_idx, unc)
    robo_pts = pts[robo_idx]

    # 5. Floor Filter
    floor_z = estimate_floor_height(pts)
    near_floor_mask = robo_pts[:, 2] < floor_z + 0.7
    robo_idx_nf = robo_idx[near_floor_mask]
    robo_pts_nf = robo_pts[near_floor_mask]

    if robo_idx_nf.size < 20:
        # Keep old centroid if lost
        return robo_idx, last_centroid

    # 6. DBSCAN
    try:
        cl = DBSCAN(eps=0.18, min_samples=20).fit_predict(robo_pts_nf)
    except Exception:
        cl = np.full(robo_pts_nf.shape[0], -1)

    valid = cl != -1
    if not np.any(valid):
        return robo_idx, last_centroid

    uniq = np.unique(cl[valid])

    # 7. Choose Best Cluster
    best_cluster = None
    best_score = -1e9

    for cid in uniq:
        mask = (cl == cid)
        cid_idx = robo_idx_nf[mask]
        if cid_idx.size < 20: continue

        cpts = pts[cid_idx]
        dims = cpts.max(0) - cpts.min(0)

        # --- UPDATED SIZE FILTER ---
        # Unitree Go1 is ~0.6m long. 
        # We reject anything tiny (<30cm long, <20cm wide).
        min_dim = np.array([0.30, 0.20, 0.20]) # Was [0.2, 0.1, 0.1]
        max_dim = np.array([1.20, 0.90, 0.90])
        
        if np.any(dims < min_dim) or np.any(dims > max_dim):
            continue

        if is_column_like(dims) or is_wall_like(dims):
            continue

        centroid = cpts.mean(0)

        # Scoring: Points + Tracking Consistency
        size_score = cid_idx.size
        track_score = 0.0
        if last_centroid is not None:
            dist = np.linalg.norm(centroid - last_centroid)
            track_score = -500.0 * dist # Strongly penalize jumps > 1m

        score = size_score + track_score
        if score > best_score:
            best_score = score
            best_cluster = cid_idx

    if best_cluster is None:
        # If no cluster fits criteria, better to return nothing than noise
        return np.array([], dtype=np.int64), last_centroid

    # 8. Grow Cluster & Update Centroid
    cpts = pts[best_cluster]
    centroid = cpts.mean(0)
    
    # Smooth centroid
    new_centroid = centroid if last_centroid is None else 0.7 * last_centroid + 0.3 * centroid
    if last_centroid is not None:
        # Reset if huge jump (teleportation error)
        if np.linalg.norm(centroid - last_centroid) > 1.0:
            new_centroid = centroid

    return best_cluster, new_centroid


# ==============================================================
# Main Pipeline
# ==============================================================

def highlight_robot_frames(pred_dir: str, unc_dir: str, out_dir: str, robo_label: int = 3):

    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(pred_dir) if f.endswith("_pred.ply")])

    if not files:
        print("[Highlight] No *_pred.ply files found.")
        return

    # Pass 1: Learn Background Noise (Mini Factory)
    static_robot_voxels = build_static_robot_voxels(
        pred_dir, files, robo_label=robo_label,
        voxel_size=0.10, min_frame_ratio=0.25 
    )

    # Pass 2: Extract Moving Robot
    print(f"[Highlight][Pass2] Processing {len(files)} frames...")
    last_centroid: Optional[np.ndarray] = None

    for i, fname in enumerate(files):
        pred_path = os.path.join(pred_dir, fname)
        pts, labels = read_pred_ply(pred_path)

        unc_path = os.path.join(unc_dir, fname.replace("_pred.ply", "_uncertainty_mc.ply"))
        unc = read_uncertainty_ply(unc_path)

        # Periodically reset tracking to prevent latching onto bad objects
        if i % 100 == 0:
            last_centroid = None

        robot_idx, last_centroid = extract_robot_indices(
            pts, labels, last_centroid,
            static_robot_voxels=static_robot_voxels,
            robo_label=robo_label,
            voxel_size=0.10,
            unc=unc,
        )

        outname = fname.replace("_pred.ply", "_overlay_go1_uncertainty_clean.ply")
        write_overlay(os.path.join(out_dir, outname), pts, robot_idx)

        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(files)}", end="\r")

    print("\n[Highlight] Done. Static noise removed.")


if __name__ == "__main__":
    highlight_robot_frames(
        pred_dir="pipeline_rgb/inference/predictions",
        unc_dir="pipeline_rgb/inference/uncertainty_mc",
        out_dir="pipeline_rgb/shape/highlighted_uncertainty",
        robo_label=3,
    )