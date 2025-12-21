#!/usr/bin/env python3
"""
Overlay Visualizer for Stage 2
==============================

This script overlays:
  - Original scene points (gray)
  - Highlighted robot points (green)
  - Mean ellipsoid (yellow)
  - Mean OBB (red)

Output:
  pipeline_rgb/shape/model_stage2/visualization/
      overlay_ellipsoid_frame_xxxx.ply
      overlay_obb_frame_xxxx.ply
      overlay_both_frame_xxxx.ply
"""

import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
import json


ROOT_SCENE  = "pipeline_rgb/input/ply_frames"
ROOT_HL     = "pipeline_rgb/shape/highlighted_uncertainty"
ROOT_MODEL  = "pipeline_rgb/shape/model_stage2"
OUT_DIR     = os.path.join(ROOT_MODEL, "visualization")
os.makedirs(OUT_DIR, exist_ok=True)


def load_ply_xyz(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32)
    return xyz, v


def load_robot_points(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32)
    mask = v["green"] == 255  # robot = green
    return xyz[mask]


def load_ellipsoid():
    ply = PlyData.read(os.path.join(ROOT_MODEL, "ellipsoid_mean.ply"))
    v = ply["vertex"].data
    xyz = np.vstack((v["x"], v["y"], v["z"])).T
    return xyz


def load_obb():
    ply = PlyData.read(os.path.join(ROOT_MODEL, "obb_mean.ply"))
    v = ply["vertex"].data
    xyz = np.vstack((v["x"], v["y"], v["z"])).T
    return xyz


def write_overlay(path, xyz, rgb):
    N = xyz.shape[0]
    out = np.empty(
        N,
        dtype=[("x","f4"),("y","f4"),("z","f4"),
               ("red","u1"),("green","u1"),("blue","u1")]
    )
    out["x"], out["y"], out["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    out["red"], out["green"], out["blue"] = rgb[:,0], rgb[:,1], rgb[:,2]
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(path)


def build_overlay(frame_id):
    # ---------------------------
    # Load scene
    # ---------------------------
    scene_path = os.path.join(ROOT_SCENE, f"frame_{frame_id}.ply")
    scene_xyz, _ = load_ply_xyz(scene_path)
    scene_rgb = np.tile(np.array([150,150,150], dtype=np.uint8), (scene_xyz.shape[0],1))

    # ---------------------------
    # Load robot-only points
    # ---------------------------
    hl_path = os.path.join(ROOT_HL, f"frame_{frame_id}_overlay_go1_uncertainty_clean.ply")
    robot_xyz, _v = load_ply_xyz(hl_path)
    robot_mask = (_v["green"] == 255)
    robot_xyz = robot_xyz[robot_mask]
    robot_rgb = np.tile(np.array([0,255,0], dtype=np.uint8), (robot_xyz.shape[0],1))

    # ---------------------------
    # Load ellipsoid & OBB
    # ---------------------------
    ell_xyz = load_ellipsoid()
    ell_rgb = np.tile(np.array([255,255,0], dtype=np.uint8), (ell_xyz.shape[0],1))

    obb_xyz = load_obb()
    obb_rgb = np.tile(np.array([255,0,0], dtype=np.uint8), (obb_xyz.shape[0],1))

    # build overlay variants
    # A) Scene + robot + ellipsoid
    xyz_A = np.vstack((scene_xyz, robot_xyz, ell_xyz))
    rgb_A = np.vstack((scene_rgb, robot_rgb, ell_rgb))
    write_overlay(os.path.join(OUT_DIR, f"overlay_ellipsoid_frame_{frame_id}.ply"), xyz_A, rgb_A)

    # B) Scene + robot + OBB
    xyz_B = np.vstack((scene_xyz, robot_xyz, obb_xyz))
    rgb_B = np.vstack((scene_rgb, robot_rgb, obb_rgb))
    write_overlay(os.path.join(OUT_DIR, f"overlay_obb_frame_{frame_id}.ply"), xyz_B, rgb_B)

    # C) Scene + robot + ellipsoid + OBB
    xyz_C = np.vstack((scene_xyz, robot_xyz, ell_xyz, obb_xyz))
    rgb_C = np.vstack((scene_rgb, robot_rgb, ell_rgb, obb_rgb))
    write_overlay(os.path.join(OUT_DIR, f"overlay_both_frame_{frame_id}.ply"), xyz_C, rgb_C)

    print(f"\nGenerated overlays in:\n  {OUT_DIR}\n")
    print("Files created:")
    print(f"  - overlay_ellipsoid_frame_{frame_id}.ply")
    print(f"  - overlay_obb_frame_{frame_id}.ply")
    print(f"  - overlay_both_frame_{frame_id}.ply\n")


def auto_select_middle_frame():
    files = sorted(os.listdir(ROOT_SCENE))
    files = [f for f in files if f.endswith(".ply")]
    if not files:
        raise RuntimeError("No PLY files in scene folder.")
    mid = files[len(files)//2]
    return mid.replace("frame_","").replace(".ply","")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, default=None, help="Frame ID (e.g., 0123)")
    args = parser.parse_args()

    if args.frame is None:
        frame_id = auto_select_middle_frame()
        print(f"No frame specified. Using middle frame: {frame_id}")
    else:
        frame_id = args.frame.zfill(4)

    build_overlay(frame_id)
