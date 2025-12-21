#!/usr/bin/env python3
"""
Stage 3.3 Visualization:
Overlay scene + robot + FINAL probabilistic safety shell per frame.

Inputs:
  pipeline_rgb/input/ply_frames/frame_XXXX.ply
  pipeline_rgb/shape/highlighted_uncertainty/frame_XXXX_overlay_go1_uncertainty_clean.ply
  pipeline_rgb/shape/model_stage2/safety_shell_95_static.ply  (or 99)

Output:
  pipeline_rgb/shape/per_frame_visualization_safety_shell/
      overlay_frame_XXXX.ply
"""

import os
import numpy as np
from plyfile import PlyData, PlyElement

SCENE_DIR = "pipeline_rgb/input/ply_frames"
ROBOT_DIR = "pipeline_rgb/shape/highlighted_uncertainty"
SHELL_FILE = "pipeline_rgb/shape/model_stage2/safety_shell_95_static.ply"

OUT_DIR = "pipeline_rgb/shape/per_frame_visualization_safety_shell"
os.makedirs(OUT_DIR, exist_ok=True)


def load_xyz_rgb(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32)

    if "red" in v.dtype.names:
        rgb = np.vstack((v["red"], v["green"], v["blue"])).T.astype(np.uint8)
    else:
        rgb = np.tile(np.array([150,150,150],dtype=np.uint8),(xyz.shape[0],1))

    return xyz, rgb


def write_ply(path, xyz, rgb):
    arr = np.empty(
        xyz.shape[0],
        dtype=[
            ("x","f4"),("y","f4"),("z","f4"),
            ("red","u1"),("green","u1"),("blue","u1")
        ]
    )
    arr["x"], arr["y"], arr["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    arr["red"], arr["green"], arr["blue"] = rgb[:,0], rgb[:,1], rgb[:,2]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)



def main():

    safety_xyz, safety_rgb = load_xyz_rgb(SHELL_FILE)

    scene_frames = sorted([f for f in os.listdir(SCENE_DIR) if f.endswith(".ply")])

    for sf in scene_frames:
        fid = sf.replace("frame_","").replace(".ply","")

        scene_xyz, scene_rgb = load_xyz_rgb(f"{SCENE_DIR}/{sf}")
        scene_rgb[:] = [150,150,150]

        rob_path = f"{ROBOT_DIR}/frame_{fid}_overlay_go1_uncertainty_clean.ply"
        if not os.path.exists(rob_path):
            print(f"[Skip] No robot overlay {fid}")
            continue

        rob_xyz, rob_rgb = load_xyz_rgb(rob_path)
        mask = (rob_rgb[:,1] == 255)
        robot_xyz = rob_xyz[mask]
        robot_rgb = rob_rgb[mask]

        if robot_xyz.shape[0] < 1:
            print(f"[Skip] Empty robot {fid}")
            continue

        centroid = robot_xyz.mean(axis=0)

        safety_shift = safety_xyz + centroid

        xyz = np.vstack((scene_xyz, robot_xyz, safety_shift))
        rgb = np.vstack((scene_rgb, robot_rgb, safety_rgb))

        out = f"{OUT_DIR}/overlay_frame_{fid}.ply"
        write_ply(out, xyz, rgb)
        print("[OK]", out)


if __name__ == "__main__":
    main()
