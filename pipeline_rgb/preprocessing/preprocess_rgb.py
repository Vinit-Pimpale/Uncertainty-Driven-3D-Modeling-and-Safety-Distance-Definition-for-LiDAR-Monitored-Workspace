#!/usr/bin/env python3
"""
Scenario #1 RGB Preprocessing for KPConv Pipeline
-------------------------------------------------
This file is a pipeline-safe adaptation of generate_rgb_input_frames.py.
It produces MODEL INPUT frames: x y z r g b (NO labels).

Classes (for coloring only):
 0 Floor        → Red
 1 Wall         → Green
 2 Column       → Blue
 3 Robo Dog     → Yellow
 4 Screen+Stand → Magenta
"""

import os
import re
import math
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from collections import deque


# ------------------------------------------------------------
# Progress bar helper
# ------------------------------------------------------------

def progress_bar(current, total, bar_length=40):
    frac = current / total
    filled = int(frac * bar_length)
    bar = "█" * filled + "-" * (bar_length - filled)
    print(f"\r[Scenario #1] Processing: |{bar}| {current}/{total}", end="", flush=True)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def natural_key(s):
    """Helper for numeric sorting."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_pcd(path, voxel):
    """Load PLY and optionally downsample."""
    pcd = o3d.io.read_point_cloud(path)
    return pcd.voxel_down_sample(voxel) if voxel > 0 else pcd


def plane_normal(coeffs):
    n = np.array(coeffs[:3])
    n /= (np.linalg.norm(n) + 1e-12)
    return n


def point_to_plane_abs(pts, coeffs):
    a, b, c, d = coeffs
    n = math.sqrt(a*a + b*b + c*c) + 1e-12
    return np.abs(a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / n


def point_to_plane_signed(pts, coeffs):
    a, b, c, d = coeffs
    n = math.sqrt(a*a + b*b + c*c + 1e-12)
    return (a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / n


def color_map(labels):
    """Scenario #1 color palette."""
    cmap = {
        -1: [0.6, 0.6, 0.6],
        0:  [1.0, 0.2, 0.2],    # Floor: red
        1:  [0.2, 1.0, 0.2],    # Wall: green
        2:  [0.2, 0.4, 1.0],    # Column: blue
        3:  [1.0, 1.0, 0.0],    # Robo Dog: yellow
        4:  [0.8, 0.0, 1.0],    # Screen+Stand: magenta
    }
    return np.vstack([cmap.get(int(l), [0.6, 0.6, 0.6]) for l in labels])


def write_rgb_ply(path, pts, colors):
    """Save RGB PLY (x,y,z,r,g,b)."""
    colors = (colors * 255).astype(np.uint8)
    v = np.empty(
        len(pts),
        dtype=[("x","f4"),("y","f4"),("z","f4"),
               ("red","u1"),("green","u1"),("blue","u1")]
    )
    v["x"], v["y"], v["z"] = pts[:,0], pts[:,1], pts[:,2]
    v["red"], v["green"], v["blue"] = colors[:,0], colors[:,1], colors[:,2]
    PlyData([PlyElement.describe(v,"vertex")], text=False).write(path)


# ------------------------------------------------------------
# MAIN SCENARIO #1 GENERATOR
# ------------------------------------------------------------

def main(
    input_dir="pipeline_rgb/input/ply_frames",
    output_dir="pipeline_rgb/preprocessing/ready",
    frames=None,
    voxel=0.03,
    motion=0.04,
    lookback=5
):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".ply")]
    files.sort(key=natural_key)
    if frames is not None:
        files = files[:frames]

    print(f"[Scenario #1] Preprocessing {len(files)} RGB frames...")

    # --------------------------------------------------------
    # 1. Detect floor + wall from FIRST frame (original logic)
    # --------------------------------------------------------
    base = load_pcd(os.path.join(input_dir, files[0]), voxel)
    pts = np.asarray(base.points)
    up  = np.array([0,0,1])

    planes, res = [], base

    # Multi-plane extraction
    for _ in range(10):
        if len(res.points) < 1000:
            break
        model, idx = res.segment_plane(0.02, 3, 3000)
        idx = np.asarray(idx)
        if len(idx) < 800:
            break
        planes.append((model, idx))
        res = res.select_by_index(idx, invert=True)

    horizontals = []
    verticals   = []

    for model, idx in planes:
        n = plane_normal(model)
        cz = abs(np.dot(n, up))
        if cz > 0.8:
            horizontals.append((model, idx, n))
        elif cz < 0.2:
            verticals.append((model, idx, n))

    if not horizontals or not verticals:
        raise RuntimeError("Could not find both floor and wall in first frame.")

    floor_model = min(horizontals, key=lambda p: np.mean(pts[p[1],2]))[0]
    wall_model, wall_idx, wall_n = max(verticals, key=lambda p: len(p[1]))
    print("[✓] Floor and wall detected.")

    # For Robo Dog motion detection
    hist = deque(maxlen=lookback)
    last_robot_centroid = None
    CENTROID_MOVE_THRESHOLD = 0.03

    # --------------------------------------------------------
    # 2. Process each frame (REAL Scenario #1 logic)
    # --------------------------------------------------------
    total_frames = len(files)

    for i, f in enumerate(files):
        path = os.path.join(input_dir, f)
        pcd = load_pcd(path, voxel)
        pts = np.asarray(pcd.points)

        labels = np.full(len(pts), -1, np.int32)

        # ---------------- FLOOR / WALL ----------------
        d_floor = point_to_plane_abs(pts, floor_model)
        d_wall  = point_to_plane_abs(pts, wall_model)

        labels[d_floor < 0.03] = 0
        labels[(labels < 0) & (d_wall < 0.05)] = 1

        floor_z = np.percentile(pts[:,2], 5)

        # ---------------- ROBO DOG (motion-based) ----------------
        robot_near_floor_mask = d_floor < 0.7
        robot_found = False

        if not robot_found and len(hist) == lookback:
            ref = hist[0]
            d = np.asarray(pcd.compute_point_cloud_distance(ref))
            moving = d > motion
            mask = moving & robot_near_floor_mask & (labels < 0)
            if np.sum(mask) > 50:
                labels[mask] = 3
                robot_found = True
                last_robot_centroid = pts[mask].mean(0)

        # ---------------- SCREEN + STAND ----------------
        signed = point_to_plane_signed(pts, wall_model)
        front_mask = (signed < 0) & (signed > -2.0)
        mid_mask   = (pts[:,2] > floor_z+0.5) & (pts[:,2] < floor_z+2.2)

        cand = np.where(front_mask & mid_mask & (labels < 0))[0]
        if len(cand) > 500:
            cand_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[cand]))
            cl = np.array(cand_pcd.cluster_dbscan(eps=0.2, min_points=80))
            for cid in np.unique(cl):
                if cid == -1:
                    continue
                sub = pts[cand[cl == cid]]
                height = sub[:,2].max() - sub[:,2].min()
                if height > 0.3:
                    labels[cand[cl == cid]] = 4
                    break

        # ---------------- COLUMN DETECTION ----------------
        rem = np.where((labels < 0) & (d_floor > 0.05))[0]
        if len(rem) > 0:
            rem_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[rem]))
            cl = np.array(rem_pcd.cluster_dbscan(eps=0.15, min_points=80))
            for cid in np.unique(cl):
                if cid == -1:
                    continue
                sub = pts[rem[cl == cid]]
                ext = sub.max(0) - sub.min(0)
                if 0.6 < ext[2] < 3.0 and max(ext[0], ext[1]) < 1.0:
                    labels[rem[cl == cid]] = 2

        # ---------------- CLEANUP ----------------
        labels[labels < 0] = 1
        colors = color_map(labels)

        out = os.path.join(output_dir, f.replace(".ply","_rgb.ply"))
        write_rgb_ply(out, pts, colors)

        # PROGRESS BAR UPDATE
        progress_bar(i + 1, total_frames)

        hist.append(pcd)

    print("\n\n[Scenario #1] RGB preprocessing complete.")
    print("Output = x y z r g b (model input)\n")


if __name__ == "__main__":
    main()
