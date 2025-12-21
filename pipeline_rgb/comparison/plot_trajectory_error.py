#!/usr/bin/env python3
"""
Trajectory Error Plot (Smoothed Visualization)
==============================================
Demonstrates the effect of temporal smoothing on the robot's path.

Outputs:
  - pipeline_rgb/comparison/visualizations/trajectory_smoothing.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
LOG_CSV = "pipeline_rgb/shape/model_stage2/temporal_shell_log.csv"
OUT_IMG = "pipeline_rgb/comparison/visualizations/trajectory_smoothing.png"

def main():
    if not os.path.exists(LOG_CSV):
        print("Error: Log CSV not found. Run temporal_shell_tracking.py first!")
        return

    print("Generating Trajectory Plot...")
    
    # 1. Load Data
    df = pd.read_csv(LOG_CSV)
    
    # 2. Create "Raw" vs "Smooth" series
    # The CSV contains frame-by-frame centroids. We treat this as "Raw Detection".
    raw_x = df['centroid_x'].values
    raw_y = df['centroid_y'].values
    
    # Apply Rolling Window Smoothing (Window=10 frames) to simulate stable trajectory
    # This aligns with the "Temporal Shell" logic of using history.
    smooth_x = df['centroid_x'].rolling(window=10, center=True, min_periods=1).mean().values
    smooth_y = df['centroid_y'].rolling(window=10, center=True, min_periods=1).mean().values
    
    # 3. Calculate Stats (Jitter Reduction)
    # Jitter = sum of distances between consecutive points
    dist_raw = np.sum(np.sqrt(np.diff(raw_x)**2 + np.diff(raw_y)**2))
    dist_smooth = np.sum(np.sqrt(np.diff(smooth_x)**2 + np.diff(smooth_y)**2))
    
    reduction = (1 - dist_smooth/dist_raw) * 100

    # 4. Plot
    plt.figure(figsize=(10, 6))
    
    # Plot faint raw path
    plt.plot(raw_x, raw_y, 'r.', markersize=4, alpha=0.25, label='Instantaneous Detection')
    plt.plot(raw_x, raw_y, 'r-', linewidth=0.5, alpha=0.15)
    
    # Plot strong smooth path
    plt.plot(smooth_x, smooth_y, 'b-', linewidth=2.5, label='Temporally Smoothed')
    
    plt.title(f"Trajectory Stabilization\nJitter Reduction: {reduction:.1f}%", fontsize=14)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    plt.savefig(OUT_IMG, dpi=150)
    print(f"[Done] Plot saved to: {OUT_IMG}")
    print(f"Stats: Raw Path Length: {dist_raw:.2f}m | Smoothed Path: {dist_smooth:.2f}m")

if __name__ == "__main__":
    main()