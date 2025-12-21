#!/usr/bin/env python3
"""
Stage 3.4 — Robot–Wall Safety Check (Ghost Wall Visualization)
==============================================================

Updates:
  1. GHOST WALL VISUALIZATION:
     - Adds a secondary 'Magenta' point cloud floating in front of the wall.
     - The float distance = 0.10m (Base) + (Uncertainty * 10.0).
     - Allows you to SEE the uncertainty clearly.
  2. Safety Logic:
     - Checks collision between Robot Shell and Inflated Wall logic.

Inputs
------
  - Scene, Preds, Robot Overlay, Safety Shell
  - Uncertainty PLY

Outputs
-------
  - CSV Log, Overlays, Video
"""

import os
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
import open3d as o3d
import cv2
from sklearn.cluster import DBSCAN

# --------------- PATHS ---------------
SCENE_DIR = "pipeline_rgb/input/ply_frames"
PRED_DIR  = "pipeline_rgb/inference/predictions"
UNC_DIR   = "pipeline_rgb/inference/uncertainty_mc"
ROBOT_DIR = "pipeline_rgb/shape/highlighted_uncertainty"
ROBOT_SHELL_FILE = "pipeline_rgb/shape/model_stage2/safety_shell_95_static.ply"

OVERLAY_DIR = "pipeline_rgb/shape/per_frame_robot_wall_overlap"
os.makedirs(OVERLAY_DIR, exist_ok=True)

CSV_LOG   = "pipeline_rgb/shape/model_stage2/robot_wall_overlap_log.csv"
VIDEO_OUT = "pipeline_rgb/shape/robot_wall_overlap_video.mp4"

# --------------- CONFIG ---------------
FLOOR_TOLERANCE = 0.20
ROBOT_EXCLUSION_RADIUS = 1.0
MIN_WALL_HEIGHT = 0.60
ISO_SAFE_DIST = 0.60

# Safety Math
UNC_SCALE_FACTOR = 1.0    # Used for actual safety calculation

# Visualization Config
GHOST_BASE_OFFSET = 0.30  # Floating 10cm off the wall (so it's always visible)
GHOST_UNC_SCALE   = 10.0  # Scale uncertainty x10 visually so it "pops"


# --------------- HELPERS ---------------
def load_xyz_rgb(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32)
    if "red" in v.dtype.names:
        rgb = np.vstack((v["red"], v["green"], v["blue"])).T.astype(np.uint8)
    else:
        rgb = np.tile(np.array([150,150,150], np.uint8), (xyz.shape[0], 1))
    return xyz, rgb

def load_pred_labels(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32)
    labels = np.array(v["pred"], dtype=np.int32)
    return xyz, labels

def load_uncertainty_values(path):
    if not os.path.exists(path): return None
    ply = PlyData.read(path)
    v = ply["vertex"].data
    if "mc_var" in v.dtype.names:
        return np.array(v["mc_var"], dtype=np.float32)
    return np.zeros(v.count, dtype=np.float32)

def write_ply(path, xyz, rgb):
    arr = np.empty(xyz.shape[0], dtype=[("x","f4"),("y","f4"),("z","f4"),
                                       ("red","u1"),("green","u1"),("blue","u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    arr["red"], arr["green"], arr["blue"] = rgb[:,0], rgb[:,1], rgb[:,2]
    PlyData([PlyElement.describe(arr,"vertex")], text=False).write(path)


# =======================================================
# 1) COMPUTE SAFETY & GENERATE GHOST OVERLAYS
# =======================================================
def compute_robot_wall_safety_and_overlays():
    if not os.path.exists(ROBOT_SHELL_FILE):
        print(f"[Error] Safety shell not found.")
        return

    shell_xyz_base, _ = load_xyz_rgb(ROBOT_SHELL_FILE)
    scene_frames = sorted([f for f in os.listdir(SCENE_DIR) if f.endswith(".ply")])
    rows = []

    print(f"[Stage 3.4] Processing {len(scene_frames)} frames (Ghost Wall Vis)...")

    for sf in scene_frames:
        frame_id = sf.replace("frame_", "").replace(".ply", "")

        # --- Load Data ---
        scene_xyz, scene_rgb = load_xyz_rgb(os.path.join(SCENE_DIR, sf))
        scene_rgb[:] = [150,150,150]

        # Robot
        rob_path = os.path.join(ROBOT_DIR, f"frame_{frame_id}_overlay_go1_uncertainty_clean.ply")
        has_robot = False
        centroid = None
        
        if os.path.exists(rob_path):
            rob_xyz_all, rob_rgb_all = load_xyz_rgb(rob_path)
            robot_mask = (rob_rgb_all[:,1] == 255)
            if np.sum(robot_mask) > 10:
                robot_xyz = rob_xyz_all[robot_mask]
                robot_rgb = rob_rgb_all[robot_mask]
                centroid = robot_xyz.mean(axis=0)
                has_robot = True
        
        if not has_robot:
            # Just write scene
            out_ply = os.path.join(OVERLAY_DIR, f"overlay_frame_{frame_id}.ply")
            write_ply(out_ply, scene_xyz, scene_rgb)
            continue

        # Predictions & Uncertainty
        pred_path = os.path.join(PRED_DIR, f"frame_{frame_id}_pred.ply")
        unc_path  = os.path.join(UNC_DIR, f"frame_{frame_id}_uncertainty_mc.ply")
        pred_xyz, pred_labels = load_pred_labels(pred_path)
        wall_unc = load_uncertainty_values(unc_path)
        if wall_unc is None: wall_unc = np.zeros(len(pred_xyz))

        # --- Filter Wall Points ---
        indices = np.where(pred_labels == 1)[0]
        
        # Floor Height Filter
        floor_z = np.percentile(pred_xyz[:, 2], 2)
        h_mask = pred_xyz[indices, 2] > (floor_z + FLOOR_TOLERANCE)
        indices = indices[h_mask]

        # Robot Exclusion
        if len(indices) > 0:
            dists = np.linalg.norm(pred_xyz[indices] - centroid, axis=1)
            e_mask = dists > ROBOT_EXCLUSION_RADIUS
            indices = indices[e_mask]

        # Object Height Filter (DBSCAN)
        final_wall_pts = np.empty((0,3))
        final_wall_unc = np.empty((0,))

        if len(indices) > 0:
            cand_pts = pred_xyz[indices]
            cand_unc = wall_unc[indices]
            
            clustering = DBSCAN(eps=0.20, min_samples=10).fit(cand_pts)
            labels = clustering.labels_
            valid_mask = np.zeros(len(cand_pts), dtype=bool)

            for lbl in set(labels):
                if lbl == -1: continue
                c_mask = (labels == lbl)
                c_pts = cand_pts[c_mask]
                if (np.max(c_pts[:,2]) - np.min(c_pts[:,2])) >= MIN_WALL_HEIGHT:
                    valid_mask |= c_mask
            
            final_wall_pts = cand_pts[valid_mask]
            final_wall_unc = cand_unc[valid_mask]

        if len(final_wall_pts) == 0:
            # Safe (No Walls)
            shell_xyz = shell_xyz_base + centroid
            shell_rgb = np.tile(np.array([0,0,255], np.uint8), (len(shell_xyz), 1))
            XYZ = np.vstack((scene_xyz, robot_xyz, shell_xyz))
            RGB = np.vstack((scene_rgb, robot_rgb, shell_rgb))
            write_ply(os.path.join(OVERLAY_DIR, f"overlay_frame_{frame_id}.ply"), XYZ, RGB)
            print(f"[SAFE  ] Frame {frame_id} (No Walls)")
            continue

        # --- GENERATE GHOST WALL (VISUALIZATION) ---
        # 1. Calculate direction vector from Wall Point -> Robot Centroid
        #    This is the direction "in front" of the wall relative to the robot.
        vecs = centroid - final_wall_pts
        dists = np.linalg.norm(vecs, axis=1, keepdims=True)
        # Avoid div by zero
        dists[dists < 1e-3] = 1.0 
        dirs = vecs / dists

        # 2. Shift amount: Base (10cm) + Uncertainty * Scale
        #    This ensures even low uncertainty creates a visible layer.
        shifts = GHOST_BASE_OFFSET + (final_wall_unc[:, None] * GHOST_UNC_SCALE)
        
        ghost_pts = final_wall_pts + (dirs * shifts)
        
        # 3. Ghost Color: Magenta [255, 0, 255]
        ghost_rgb = np.tile(np.array([255, 0, 255], dtype=np.uint8), (len(ghost_pts), 1))


        # --- COMPUTE SAFETY ---
        shell_xyz = shell_xyz_base + centroid
        wall_pcd = o3d.geometry.PointCloud()
        wall_pcd.points = o3d.utility.Vector3dVector(final_wall_pts)
        kdtree = o3d.geometry.KDTreeFlann(wall_pcd)

        unsafe = False
        min_margin = float("inf")
        check_step = max(1, len(shell_xyz) // 2000)
        
        for i in range(0, len(shell_xyz), check_step):
            k, idx_list, d2 = kdtree.search_knn_vector_3d(shell_xyz[i], 1)
            if k > 0:
                dist = np.sqrt(d2[0])
                u_wall = final_wall_unc[idx_list[0]]
                
                # Math Safety Check
                req_dist = ISO_SAFE_DIST + (u_wall * UNC_SCALE_FACTOR)
                margin = dist - req_dist
                
                if margin < min_margin: min_margin = margin
                if dist < req_dist:
                    unsafe = True

        # --- ASSEMBLE PLY ---
        # Wall = Yellow
        wall_rgb = np.tile(np.array([255,255,0], np.uint8), (len(final_wall_pts), 1))
        
        # Shell = Red/Blue
        s_color = [255, 0, 0] if unsafe else [0, 0, 255]
        shell_rgb = np.tile(np.array(s_color, dtype=np.uint8), (len(shell_xyz), 1))

        # Stack: Scene + Wall + Ghost + Robot + Shell
        XYZ = np.vstack((scene_xyz, final_wall_pts, ghost_pts, robot_xyz, shell_xyz))
        RGB = np.vstack((scene_rgb, wall_rgb,      ghost_rgb, robot_rgb, shell_rgb))

        write_ply(os.path.join(OVERLAY_DIR, f"overlay_frame_{frame_id}.ply"), XYZ, RGB)

        status = "UNSAFE" if unsafe else "SAFE  "
        print(f"[{status}] Frame {frame_id} | Margin: {min_margin:.3f} | GhostPts: {len(ghost_pts)}")
        rows.append({"frame": frame_id, "safe": not unsafe, "margin": min_margin})

    pd.DataFrame(rows).to_csv(CSV_LOG, index=False)


# =======================================================
# 2) VIDEO GENERATION
# =======================================================
def generate_video_from_overlays():
    files = sorted([f for f in os.listdir(OVERLAY_DIR) if f.endswith(".ply")])
    if not files:
        print("[Video] No overlay PLYs found.")
        return

    print(f"[Video] Rendering {len(files)} frames to {VIDEO_OUT}...")

    WIDTH, HEIGHT = 1920, 1080
    renderer = o3d.visualization.rendering.OffscreenRenderer(WIDTH, HEIGHT)
    scene = renderer.scene
    scene.set_background([0,0,0,1])
    scene.camera.set_projection(60.0, WIDTH/HEIGHT, 0.1, 100.0, o3d.visualization.rendering.Camera.FovType.Vertical)

    writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), 20, (WIDTH, HEIGHT))

    for i, f in enumerate(files):
        scene.clear_geometry()
        cloud = o3d.io.read_point_cloud(os.path.join(OVERLAY_DIR, f))
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        scene.add_geometry("cloud", cloud, mat)

        bbox = cloud.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_max_extent()
        
        # Left Side View
        eye = center + np.array([-extent, 0, 0.6*extent])
        scene.camera.look_at(center, eye, np.array([0,0,1]))

        img = np.asarray(renderer.render_to_image())
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        if (i+1)%50==0: print(f"  Rendered {i+1} frames...")

    writer.release()
    print("[Video] Saved.")


if __name__ == "__main__":
    compute_robot_wall_safety_and_overlays()
    generate_video_from_overlays()