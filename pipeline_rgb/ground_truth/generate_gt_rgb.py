#!/usr/bin/env python3
"""
Scenario #1 Ground Truth Generator
==================================

This is the Ground_Truth_Generator logic adapted for the pipeline on the GPU server.

- Same plane detection (floor + wall)
- Same Robo Dog tracking (tracker-first + motion fallback)
- Same Screen+Stand and Column detection
- Same color map and labels:
    0 = Floor (red)
    1 = Wall (green)
    2 = Column (blue)
    3 = Robo Dog (yellow)
    4 = Screen+Stand (magenta)

Differences from the original script:
- No argparse; uses a function-style main(...) for pipeline integration.
- Default paths point to the pipeline folders, not ~/LiDAR_Data.
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
    print(f"\r[GT] Processing: |{bar}| {current}/{total}", end="", flush=True)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def natural_key(s):
    """Numeric-aware sort key."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_pcd(path, voxel):
    """Load a PLY point cloud and optionally voxel downsample."""
    pcd = o3d.io.read_point_cloud(path)
    return pcd.voxel_down_sample(voxel) if voxel > 0 else pcd


def plane_normal(coeffs):
    n = np.array(coeffs[:3])
    n /= np.linalg.norm(n) + 1e-12
    return n


def point_to_plane_abs(pts, coeffs):
    a, b, c, d = coeffs
    n = math.sqrt(a*a + b*b + c*c) + 1e-12
    return np.abs(a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / n


def point_to_plane_signed(pts, coeffs):
    a, b, c, d = coeffs
    n = math.sqrt(a*a + b*b + c*c) + 1e-12
    return (a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / n


def color_map(labels):
    cmap = {
        -1: [0.6, 0.6, 0.6],
         0: [1.0, 0.2, 0.2],
         1: [0.2, 1.0, 0.2],
         2: [0.2, 0.4, 1.0],
         3: [1.0, 1.0, 0.0],
         4: [0.8, 0.0, 1.0],
    }
    return np.vstack([cmap.get(int(l), [0.6, 0.6, 0.6]) for l in labels])


def write_ply(path, pts, labels, colors):
    """Write GT PLY with xyz, rgb, and integer label."""
    colors = (colors * 255).astype(np.uint8)
    v = np.empty(
        len(pts),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "i4"),
        ],
    )
    v["x"], v["y"], v["z"] = pts[:,0], pts[:,1], pts[:,2]
    v["red"], v["green"], v["blue"] = colors[:,0], colors[:,1], colors[:,2]
    v["label"] = labels
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(path)


# ------------------------------------------------------------
# MAIN (Ground Truth generation)
# ------------------------------------------------------------

def main(
    input_dir="pipeline_rgb/input/ply_frames",
    output_dir="pipeline_rgb/ground_truth/gt_ply",
    frames=None,
    voxel=0.03,
    motion=0.04,
    lookback=5,
):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".ply")]
    files.sort(key=natural_key)

    if frames is not None:
        files = files[:frames]

    print(f"[GT] Found {len(files)} frames.")

    # ---------------- FLOOR & WALL DETECTION ----------------
    base = load_pcd(os.path.join(input_dir, files[0]), voxel)
    pts = np.asarray(base.points)
    up = np.array([0, 0, 1])

    planes = []
    res = base

    for _ in range(10):
        if len(res.points) < 1000:
            break
        model, idx = res.segment_plane(0.02, 3, 3000)
        idx = np.asarray(idx)
        if len(idx) < 800:
            break
        planes.append((model, idx))
        res = res.select_by_index(idx, invert=True)

    horizontals, verticals = [], []

    for model, idx in planes:
        n = plane_normal(model)
        cz = abs(np.dot(n, up))
        if cz > 0.8:
            horizontals.append((model, idx, n))
        elif cz < 0.2:
            verticals.append((model, idx, n))

    if not horizontals or not verticals:
        print("[GT] ERROR: Could not detect floor & wall.")
        return

    floor_model = min(horizontals, key=lambda p: np.mean(pts[p[1],2]))[0]
    wall_model, wall_idx, wall_n = max(verticals, key=lambda p: len(p[1]))
    print("[GT] Floor & wall detected.")

    # Robot history
    hist = deque(maxlen=lookback)
    last_robot_centroid = None
    stationary_frames = 0
    centroid_history = deque(maxlen=5)
    CENTROID_MOVE_THRESHOLD = 0.03

    total_frames = len(files)

    # ---------------- PROCESS EACH FRAME ----------------
    for i, f in enumerate(files):
        path = os.path.join(input_dir, f)
        pcd = load_pcd(path, voxel)
        pts = np.asarray(pcd.points)

        labels = np.full(len(pts), -1, np.int32)

        # ----- FLOOR / WALL -----
        d_floor = point_to_plane_abs(pts, floor_model)
        d_wall = point_to_plane_abs(pts, wall_model)

        labels[d_floor < 0.03] = 0
        labels[(labels < 0) & (d_wall < 0.05)] = 1

        floor_z = np.percentile(pts[:,2], 5)

        # ----- ROBO DOG (TRACKER + MOTION) -----
        robot_found_this_frame = False
        robot_near_floor_mask = d_floor < 0.7

        # (Tracker logic preserved; prints kept)
        if last_robot_centroid is not None:
            candidate_mask = (labels <= 1) & robot_near_floor_mask
            candidate_indices = np.where(candidate_mask)[0]

            if len(candidate_indices) > 50:
                candidate_pts = pts[candidate_indices]
                dists_to_centroid = np.linalg.norm(
                    candidate_pts - last_robot_centroid, axis=1
                )
                SEARCH_RADIUS = 1.0
                near_centroid_mask = dists_to_centroid < SEARCH_RADIUS

                sphere_idx = candidate_indices[near_centroid_mask]

                if len(sphere_idx) > 50:
                    sphere_pts = pts[sphere_idx]
                    pcd_sphere = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(sphere_pts)
                    )
                    cl = np.array(pcd_sphere.cluster_dbscan(eps=0.12, min_points=40))

                    unique_labels, counts = np.unique(
                        cl[cl != -1], return_counts=True
                    )

                    if len(counts) > 0:
                        largest_id = unique_labels[np.argmax(counts)]
                        robot_idx = sphere_idx[cl == largest_id]

                        if len(robot_idx) > 40:
                            robot_found_this_frame = True
                            labels[robot_idx] = 3

                            new_centroid = pts[robot_idx].mean(0)
                            centroid_history.append(new_centroid)

                            last_robot_centroid = new_centroid

            if not robot_found_this_frame:
                last_robot_centroid = None
                centroid_history.clear()
                stationary_frames = 0

        # Motion fallback
        if not robot_found_this_frame:
            ref = hist[0] if len(hist) == lookback else None
            if ref is not None:
                d = np.asarray(pcd.compute_point_cloud_distance(ref))
                moving = d > motion
                idx = np.where(moving & robot_near_floor_mask & (labels < 0))[0]

                if len(idx) > 50:
                    labels[idx] = 3
                    last_robot_centroid = pts[idx].mean(0)
                    centroid_history.clear()
                    centroid_history.append(last_robot_centroid)

        # ----- SCREEN + STAND -----
        signed_d = point_to_plane_signed(pts, wall_model)
        front_mask = (signed_d < 0) & (signed_d > -2.0)
        mid_mask = (pts[:,2] > floor_z+0.5) & (pts[:,2] < floor_z+2.2)

        cand = np.where(front_mask & mid_mask & (labels < 0))[0]

        if len(cand) > 500:
            cpcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[cand]))
            cl = np.array(cpcd.cluster_dbscan(eps=0.2, min_points=80))

            best = None
            best_height = 0.0
            for cid in np.unique(cl):
                if cid == -1:
                    continue
                sub = pts[cand[cl == cid]]
                height = sub[:,2].max() - sub[:,2].min()
                if height > 0.3 and height > best_height:
                    best = cid
                    best_height = height

            if best is not None:
                labels[cand[cl == best]] = 4

        # ----- COLUMN DETECTION -----
        rem = np.where((labels < 0) & (d_floor > 0.05))[0]
        if len(rem) > 0:
            rpcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[rem]))
            cl = np.array(rpcd.cluster_dbscan(eps=0.15, min_points=80))

            for cid in np.unique(cl):
                if cid == -1:
                    continue
                sub = pts[rem[cl == cid]]
                ext = sub.max(0) - sub.min(0)

                if 0.6 < ext[2] < 3.0 and max(ext[0], ext[1]) < 1.0:
                    labels[rem[cl == cid]] = 2

        labels[labels < 0] = 1
        colors = color_map(labels)

        out = os.path.join(
            output_dir, os.path.splitext(f)[0] + "_labeled.ply"
        )
        write_ply(out, pts, labels, colors)

        # PROGRESS BAR UPDATE
        progress_bar(i + 1, total_frames)

        hist.append(pcd)

    print("\n\n[GT] Ground truth generation complete.")
    print("Legend: 0=Floor 1=Wall 2=Column 3=Robo Dog 4=Screen+Stand")


if __name__ == "__main__":
    main()
