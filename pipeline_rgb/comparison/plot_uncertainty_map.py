#!/usr/bin/env python3
"""
Generate Uncertainty Trajectory Map
===================================
Visualizes the robot's path colored by the model's predictive uncertainty.
Identifies 'Blind Spots' or 'High Confusion Areas' in the workspace.

Output: pipeline_rgb/comparison/visualizations/uncertainty_heatmap.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

# Paths
UNC_DIR = "pipeline_rgb/inference/uncertainty_mc"
OUT_IMG = "pipeline_rgb/comparison/visualizations/uncertainty_heatmap.png"

def main():
    files = sorted([f for f in os.listdir(UNC_DIR) if f.endswith(".ply")])
    if not files:
        print("No uncertainty files found.")
        return

    print(f"Generating Uncertainty Map from {len(files)} frames...")
    
    data = []
    
    for i, f in enumerate(files):
        # Load PLY
        ply = PlyData.read(os.path.join(UNC_DIR, f))
        v = ply["vertex"].data
        
        # Filter for Robot Points (Pred = 3)
        # We want to know how uncertain the model is ABOUT THE ROBOT
        mask = (v['pred'] == 3)
        if np.sum(mask) < 10: continue
        
        pts_x = v['x'][mask]
        pts_y = v['y'][mask]
        
        # Metric: Entropy (Total Uncertainty) or MC_Var (Model Uncertainty)
        # Using Entropy is often better for general "confusion"
        unc = v['entropy'][mask]
        
        # Centroid of the robot for this frame
        c_x = np.mean(pts_x)
        c_y = np.mean(pts_y)
        avg_unc = np.mean(unc)
        
        data.append([c_x, c_y, avg_unc])

        if (i+1) % 50 == 0: print(f"  Processed {i+1} frames...", end="\r")

    data = np.array(data)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with color mapping
    sc = plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap='plasma', 
                     s=30, alpha=0.8, edgecolor='none')
    
    cbar = plt.colorbar(sc)
    cbar.set_label("Mean Entropy (Uncertainty)", fontsize=12)
    
    plt.title("Spatial Uncertainty Map\n(Where is the Model Confused?)", fontsize=15)
    plt.xlabel("Room X (m)")
    plt.ylabel("Room Y (m)")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate high uncertainty zones
    max_idx = np.argmax(data[:,2])
    plt.annotate('Max Uncertainty', 
                 xy=(data[max_idx,0], data[max_idx,1]), 
                 xytext=(data[max_idx,0]+1, data[max_idx,1]+1),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    plt.savefig(OUT_IMG, dpi=150)
    print(f"\n[Done] Heatmap saved to: {OUT_IMG}")

if __name__ == "__main__":
    main()