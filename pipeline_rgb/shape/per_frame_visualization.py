#!/usr/bin/env python3
"""
Per-frame Visualization with Global Shape
=========================================

Uses the global shape (from shape_params.json) and, for each frame:
  - computes the robot centroid
  - smooths centroid over time
  - draws a fixed-size ellipsoid + OBB at that centroid
  - overlays on scene + robot points

Outputs:
  pipeline_rgb/shape/per_frame_visualization_final/overlay_frame_xxxx.ply
"""

import os
import json
import numpy as np
from plyfile import PlyData, PlyElement

SCENE_DIR = "pipeline_rgb/input/ply_frames"
ROBOT_DIR = "pipeline_rgb/shape/highlighted_uncertainty"
MODEL_DIR = "pipeline_rgb/shape/model_stage2"
OUT_DIR   = "pipeline_rgb/shape/per_frame_visualization_final"

os.makedirs(OUT_DIR, exist_ok=True)


def load_xyz(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32)
    return xyz, v


def write_cloud(path, xyz, rgb):
    N = xyz.shape[0]
    out = np.empty(
        N,
        dtype=[("x","f4"),("y","f4"),("z","f4"),
               ("red","u1"),("green","u1"),("blue","u1")]
    )
    out["x"], out["y"], out["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    out["red"], out["green"], out["blue"] = rgb[:,0], rgb[:,1], rgb[:,2]
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(path)


def make_ellipsoid(center, axes, R, num_u=40, num_v=20):
    u = np.linspace(0, 2*np.pi, num_u)
    v = np.linspace(0, np.pi, num_v)

    x = axes[0] * np.outer(np.cos(u), np.sin(v))
    y = axes[1] * np.outer(np.sin(u), np.cos(v))
    z = axes[2] * np.outer(np.ones_like(u), np.sin(v))

    pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    pts = pts @ R + center
    rgb = np.tile(np.array([255,255,0], dtype=np.uint8), (pts.shape[0],1))
    return pts, rgb


def make_obb_edges(center, dims, R, edge_res=40):
    dx, dy, dz = dims / 2.0
    corners = np.array([
        [ dx, dy, dz], [ dx, dy,-dz], [ dx,-dy, dz], [ dx,-dy,-dz],
        [-dx, dy, dz], [-dx, dy,-dz], [-dx,-dy, dz], [-dx,-dy,-dz],
    ])
    corners = corners @ R + center

    edges = [
        (0,1),(0,2),(0,4),
        (1,3),(1,5),(2,3),(2,6),
        (3,7),(4,5),(4,6),(5,7),(6,7)
    ]

    segments = []
    for i, j in edges:
        p1, p2 = corners[i], corners[j]
        t = np.linspace(0, 1, edge_res)
        seg = (1 - t)[:,None] * p1[None,:] + t[:,None] * p2[None,:]
        segments.append(seg)

    pts = np.vstack(segments)
    rgb = np.tile(np.array([255,0,0], dtype=np.uint8), (pts.shape[0],1))
    return pts, rgb


def main():
    # Load global shape params
    params_path = os.path.join(MODEL_DIR, "shape_params.json")
    with open(params_path, "r") as f:
        params = json.load(f)

    master_axes = np.array(params["master_axes"], dtype=np.float32)
    master_dims = np.array(params["master_dims"], dtype=np.float32)
    master_R    = np.array(params["master_rotation"], dtype=np.float32)

    scene_files = sorted([f for f in os.listdir(SCENE_DIR) if f.startswith("frame_")])
    if not scene_files:
        print("No scene frames found.")
        return

    last_centroid = None
    max_step = 0.4  # limit centroid jump per frame

    for f in scene_files:
        fid = f.replace("frame_","").replace(".ply","")

        # Scene
        scene_xyz, _ = load_xyz(os.path.join(SCENE_DIR, f))
        scene_rgb = np.tile(np.array([150,150,150], dtype=np.uint8),
                            (scene_xyz.shape[0],1))

        # Robot-only
        rob_path = os.path.join(
            ROBOT_DIR, f"frame_{fid}_overlay_go1_uncertainty_clean.ply"
        )
        if not os.path.exists(rob_path):
            print(f"[Vis] Missing robot overlay for frame {fid}, skipping.")
            continue

        rob_xyz_all, v = load_xyz(rob_path)
        mask_robot = (v["green"] == 255)
        robot_xyz = rob_xyz_all[mask_robot]

        if robot_xyz.shape[0] < 10:
            print(f"[Vis] Too few robot points in frame {fid}, skipping.")
            continue

        raw_centroid = robot_xyz.mean(axis=0)

        # smooth centroid over time to avoid jumps in bad frames
        if last_centroid is None:
            centroid = raw_centroid
        else:
            diff = raw_centroid - last_centroid
            dist = np.linalg.norm(diff)
            if dist > max_step:
                diff = diff / (dist + 1e-8) * max_step
            centroid = last_centroid + diff

        last_centroid = centroid

        robot_rgb = np.tile(np.array([0,255,0], dtype=np.uint8),
                            (robot_xyz.shape[0],1))

        # Global ellipsoid & OBB at this centroid
        ell_xyz, ell_rgb = make_ellipsoid(centroid, master_axes, master_R)
        obb_xyz, obb_rgb = make_obb_edges(centroid, master_dims, master_R)

        xyz = np.vstack((scene_xyz, robot_xyz, ell_xyz, obb_xyz))
        rgb = np.vstack((scene_rgb, robot_rgb, ell_rgb, obb_rgb))

        out_path = os.path.join(OUT_DIR, f"overlay_frame_{fid}.ply")
        write_cloud(out_path, xyz, rgb)
        print(f"[Vis] Saved {out_path}")

    print("\n[Vis] Per-frame visualization with global shape finished.")


if __name__ == "__main__":
    main()
