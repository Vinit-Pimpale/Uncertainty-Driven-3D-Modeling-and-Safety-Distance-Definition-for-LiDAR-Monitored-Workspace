#!/usr/bin/env python3
"""
Make video from overlay_frame_XXXX.ply (HEADLESS, HIGHER RESOLUTION)
===================================================================

Reads:
  pipeline_rgb/shape/per_frame_visualization_final/overlay_frame_XXXX.ply

Writes:
  pipeline_rgb/shape/per_frame_visualization_final/shape_video.mp4
"""

import os
import numpy as np
import cv2
import open3d as o3d

INPUT_DIR = "pipeline_rgb/shape/per_frame_visualization_final"
OUTPUT_FILE = os.path.join(INPUT_DIR, "shape_video.mp4")


def main():
    files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.startswith("overlay_frame_") and f.endswith(".ply")
    ])
    if not files:
        print("No overlay_frame_XXXX.ply found.")
        return

    print(f"[Video] Found {len(files)} frames.")

    # Higher resolution for sharper video
    WIDTH, HEIGHT = 1920, 1080

    renderer = o3d.visualization.rendering.OffscreenRenderer(WIDTH, HEIGHT)
    scene = renderer.scene
    scene.set_background([0, 0, 0, 1])

    fov = 60.0
    aspect = WIDTH / HEIGHT
    near, far = 0.1, 100.0
    scene.camera.set_projection(
        fov,
        aspect,
        near,
        far,
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )

    fps = 20
    writer = cv2.VideoWriter(
        OUTPUT_FILE,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (WIDTH, HEIGHT),
    )

    def load_cloud(path, name="cloud"):
        cloud = o3d.io.read_point_cloud(path)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        scene.add_geometry(name, cloud, mat)
        return cloud

    for i, fname in enumerate(files):
        print(f"[Video] {i + 1}/{len(files)}: {fname}")
        scene.clear_geometry()

        path = os.path.join(INPUT_DIR, fname)
        cloud = load_cloud(path, "frame")

        bbox = cloud.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_max_extent()

        eye = center + np.array([0.0, -extent, 0.6 * extent])
        up = np.array([0, 0, 1])

        scene.camera.look_at(center, eye, up)

        img_o3d = renderer.render_to_image()
        img = np.asarray(img_o3d)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)

    writer.release()
    print(f"\n[Video] Saved video to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
