#!/usr/bin/env python3
"""
Generate A/B Comparison Sequence + Video (4K Resolution)
========================================================

Updates:
  1. Resolution: Increased to 4K (3840 x 2160).
  2. Scaling: Font size and line width doubled for visibility.

Outputs:
  - Images: pipeline_rgb/comparison/visualizations/ab_sequence/frame_XXXX.jpg
  - Video:  pipeline_rgb/comparison/visualizations/ab_comparison_video_4k.mp4
"""

import os
import numpy as np
import open3d as o3d
import cv2

# --- PATHS ---
SCENE_DIR   = "pipeline_rgb/input/ply_frames"
ROBOT_DIR   = "pipeline_rgb/shape/highlighted_uncertainty"
OURS_DIR    = "pipeline_rgb/shape/per_frame_robot_wall_overlap" 
OUT_IMG_DIR = "pipeline_rgb/comparison/visualizations/ab_sequence"
OUT_VIDEO   = "pipeline_rgb/comparison/visualizations/ab_comparison_video_4k.mp4"

os.makedirs(OUT_IMG_DIR, exist_ok=True)

# --- CONFIG (4K) ---
WIDTH, HEIGHT = 3840, 2160  # 4K UHD
SPLIT_W = WIDTH // 2        # 1920 width per side

def load_ply_as_pcd(path, paint_color=None):
    if not os.path.exists(path): return None
    pcd = o3d.io.read_point_cloud(path)
    if paint_color is not None:
        pcd.paint_uniform_color(paint_color)
    return pcd

def add_text(img, text, pos):
    """Adds text with a black outline for visibility (Scaled for 4K)."""
    # Font scale 1.2 -> 2.5
    # Thickness 6 -> 10 (Outline), 2 -> 4 (Text)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,0), 10) 
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 4)

def main():
    files = sorted([f for f in os.listdir(OURS_DIR) if f.endswith(".ply")])
    if not files:
        print(f"[Error] No frames found in {OURS_DIR}.")
        return

    print(f"[A/B Gen] Processing {len(files)} frames at 4K Resolution...")

    # --- SETUP RENDERER ---
    # Maximize quality
    renderer = o3d.visualization.rendering.OffscreenRenderer(SPLIT_W, HEIGHT)
    scene = renderer.scene
    scene.set_background([0.05, 0.05, 0.05, 1])

    def render_view(geometries, center, look_eye):
        scene.clear_geometry()
        for i, geom in enumerate(geometries):
            if geom is None: continue
            
            mat = o3d.visualization.rendering.MaterialRecord()
            if isinstance(geom, o3d.geometry.LineSet):
                mat.shader = "unlitLine"
                mat.line_width = 8.0  # Thicker lines for 4K
            else:
                mat.shader = "defaultUnlit"
                # Optional: Increase point size if points look too small
                mat.point_size = 6.0 
            
            scene.add_geometry(f"geom_{i}", geom, mat)

        scene.camera.look_at(center, look_eye, np.array([0, 0, 1]))
        scene.camera.set_projection(60.0, SPLIT_W/HEIGHT, 0.1, 100.0, o3d.visualization.rendering.Camera.FovType.Vertical)
        
        img = np.asarray(renderer.render_to_image())
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # --- 1. GENERATE IMAGES ---
    generated_images = []

    for i, fname in enumerate(files):
        fid = fname.replace("overlay_frame_", "").replace(".ply", "")
        
        path_scene = os.path.join(SCENE_DIR, f"frame_{fid}.ply")
        path_robot = os.path.join(ROBOT_DIR, f"frame_{fid}_overlay_go1_uncertainty_clean.ply")
        path_ours  = os.path.join(OURS_DIR, fname)

        if not os.path.exists(path_scene) or not os.path.exists(path_robot):
            continue

        pcd_scene = load_ply_as_pcd(path_scene, [0.5, 0.5, 0.5])
        pcd_robot = load_ply_as_pcd(path_robot)

        bbox = pcd_robot.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_max_extent()
        eye = center + np.array([-extent, 0, 0.6*extent])

        # --- LEFT: BASELINE ---
        aabb = pcd_robot.get_axis_aligned_bounding_box()
        aabb.color = (0, 1, 0)
        ls_aabb = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
        
        img_left = render_view([pcd_scene, pcd_robot, ls_aabb], center, eye)
        add_text(img_left, "Baseline: Fixed BBox", (100, 150)) # Adjusted pos

        # --- RIGHT: OURS ---
        pcd_ours = load_ply_as_pcd(path_ours)
        
        img_right = render_view([pcd_ours], center, eye)
        add_text(img_right, "Ours: Probabilistic Shell", (100, 150)) # Adjusted pos

        # --- STITCH ---
        combined = np.hstack((img_left, img_right))
        
        out_path = os.path.join(OUT_IMG_DIR, f"frame_{fid}.jpg")
        cv2.imwrite(out_path, combined)
        generated_images.append(out_path)
        
        if (i+1) % 20 == 0:
            print(f"  Rendered {i+1}/{len(files)} frames...", end="\r")

    print(f"\n[A/B Gen] Image generation complete. Creating video...")

    # --- 2. CREATE VIDEO ---
    if not generated_images:
        print("No images generated.")
        return

    writer = cv2.VideoWriter(
        OUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (WIDTH, HEIGHT)
    )

    for img_path in generated_images:
        frame = cv2.imread(img_path)
        writer.write(frame)
    
    writer.release()
    print(f"[Done] 4K Video saved to: {OUT_VIDEO}")

if __name__ == "__main__":
    main()