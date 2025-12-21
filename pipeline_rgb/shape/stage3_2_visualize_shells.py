#!/usr/bin/env python3
"""
Stage 3.2 Visualization:
Overlay scene + robot + GP temporal shell per frame.

Uses:
  - pipeline_rgb/input/ply_frames/
  - pipeline_rgb/shape/highlighted_uncertainty/
  - pipeline_rgb/shape/per_frame_gp_shells/

Outputs:
  pipeline_rgb/shape/per_frame_visualization_gp_shells/
    overlay_frame_XXXX.ply
"""

import os
import numpy as np
from plyfile import PlyData, PlyElement

SCENE_DIR = "pipeline_rgb/input/ply_frames"
ROBOT_DIR = "pipeline_rgb/shape/highlighted_uncertainty"
SHELL_DIR = "pipeline_rgb/shape/per_frame_gp_shells"

OUT_DIR = "pipeline_rgb/shape/per_frame_visualization_gp_shells"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------------
# PLY load/save helpers
# -------------------------------
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
    arr["x"],arr["y"],arr["z"] = xyz[:,0],xyz[:,1],xyz[:,2]
    arr["red"],arr["green"],arr["blue"] = rgb[:,0],rgb[:,1],rgb[:,2]
    PlyData([PlyElement.describe(arr,"vertex")],text=False).write(path)


# -------------------------------
# Main visualization loop
# -------------------------------
def main():

    scene_files = sorted([f for f in os.listdir(SCENE_DIR) if f.endswith(".ply")])

    for sf in scene_files:
        frame_id = sf.replace("frame_","").replace(".ply","")

        # Scene
        scene_xyz, scene_rgb = load_xyz_rgb(f"{SCENE_DIR}/{sf}")
        scene_rgb[:] = [150,150,150]  # grey

        # Robot clean overlay
        rob_path = f"{ROBOT_DIR}/frame_{frame_id}_overlay_go1_uncertainty_clean.ply"
        if not os.path.exists(rob_path):
            print(f"[Skip] No robot overlay for {frame_id}")
            continue

        rob_xyz, rob_rgb = load_xyz_rgb(rob_path)
        robot_mask = (rob_rgb[:,1] == 255)
        robot_xyz = rob_xyz[robot_mask]
        robot_rgb = rob_rgb[robot_mask]

        if robot_xyz.shape[0] == 0:
            print(f"[Skip] Frame {frame_id}: no robot points.")
            continue

        # GP Shell
        shell_path = f"{SHELL_DIR}/gp_shell_k2_frame_{frame_id}.ply"
        if not os.path.exists(shell_path):
            print(f"[Skip] No GP shell for {frame_id}")
            continue

        shell_xyz, shell_rgb = load_xyz_rgb(shell_path)

        # Combine
        xyz = np.vstack((scene_xyz, robot_xyz, shell_xyz))
        rgb = np.vstack((scene_rgb, robot_rgb, shell_rgb))

        out_path = f"{OUT_DIR}/overlay_frame_{frame_id}.ply"
        write_ply(out_path, xyz, rgb)

        print(f"[OK] Saved overlay for frame {frame_id}")

    print("\n[Stage3.2 Visualization] Done.")


if __name__ == "__main__":
    main()
