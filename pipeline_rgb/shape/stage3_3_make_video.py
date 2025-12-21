#!/usr/bin/env python3
"""
Create HD video from Stage-3.3 FINAL safety shell overlays.
"""

import os
import numpy as np
import cv2
import open3d as o3d

IN_DIR = "pipeline_rgb/shape/per_frame_visualization_safety_shell"
OUT_FILE = f"{IN_DIR}/safety_shell_video.mp4"


def main():
    files = sorted([f for f in os.listdir(IN_DIR) if f.endswith(".ply")])
    if not files:
        print("No overlay files found.")
        return

    print(f"[Video] Found {len(files)} frames.")

    WIDTH, HEIGHT = 1920, 1080

    renderer = o3d.visualization.rendering.OffscreenRenderer(WIDTH, HEIGHT)
    scene = renderer.scene
    scene.set_background([0,0,0,1])

    fov = 60
    aspect = WIDTH / HEIGHT
    scene.camera.set_projection(
        fov, aspect, 0.1, 100.0,
        o3d.visualization.rendering.Camera.FovType.Vertical
    )

    writer = cv2.VideoWriter(
        OUT_FILE,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (WIDTH, HEIGHT)
    )

    def add_cloud(path):
        cloud = o3d.io.read_point_cloud(path)
        m = o3d.visualization.rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        scene.add_geometry("cloud", cloud, m)
        return cloud

    for i, f in enumerate(files):
        print(f"[Frame] {i+1}/{len(files)} {f}")

        scene.clear_geometry()

        cloud = add_cloud(f"{IN_DIR}/{f}")
        bbox = cloud.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_max_extent()

        eye = center + np.array([0, -extent, 0.6*extent])
        scene.camera.look_at(center, eye, np.array([0,0,1]))

        img_o3d = renderer.render_to_image()
        img = np.asarray(img_o3d)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        writer.write(img_bgr)

    writer.release()
    print("[DONE] Video saved:", OUT_FILE)


if __name__ == "__main__":
    main()
