import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from plyfile import PlyData

# Paths
MODEL_DIR = "pipeline_rgb/shape/model_stage2"
SHELL_95 = os.path.join(MODEL_DIR, "safety_shell_95_static.ply")
SHELL_99 = os.path.join(MODEL_DIR, "safety_shell_99_static.ply")
OUTPUT_IMG = "safety_zones_nested_plot.png"

def load_ply_points(path):
    if not os.path.exists(path):
        return None
    ply = PlyData.read(path)
    v = ply["vertex"].data
    # Downsample for faster plotting (plot every 10th point)
    step = 5 
    x = v["x"][::step]
    y = v["y"][::step]
    z = v["z"][::step]
    return x, y, z

def main():
    print("Reading PLY files...")
    p95 = load_ply_points(SHELL_95)
    p99 = load_ply_points(SHELL_99)

    if p95 is None and p99 is None:
        print("Error: No shell files found.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Blue Shell (Warning) - Outer Layer (alpha = 0.4 for better visibility)
    if p99 is not None:
        ax.scatter(p99[0], p99[1], p99[2], c='blue', s=1, alpha=0.4, label='Warning Zone (99%)')

    # Plot Red Shell (Stop) - Inner Layer
    if p95 is not None:
        ax.scatter(p95[0], p95[1], p95[2], c='red', s=1, alpha=0.8, label='Stop Zone (95%)')

    # Styling
    # ax.set_title("Hierarchical Safety Zones", fontsize=15) # This line is removed
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    
    # Force equal aspect ratio to make it look like a shell, not a pancake
    max_range = np.array([p99[0].max()-p99[0].min(), p99[1].max()-p99[1].min(), p99[2].max()-p99[2].min()]).max() / 2.0
    mid_x = (p99[0].max()+p99[0].min()) * 0.5
    mid_y = (p99[1].max()+p99[1].min()) * 0.5
    mid_z = (p99[2].max()+p99[2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    
    # Save
    plt.savefig(OUTPUT_IMG, dpi=200)
    print(f"Success! Image saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()