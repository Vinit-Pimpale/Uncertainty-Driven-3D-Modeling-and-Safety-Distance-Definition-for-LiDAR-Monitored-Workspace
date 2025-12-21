#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_CSV = "pipeline_rgb/shape/model_stage2/robot_wall_overlap_log.csv"
OUT_IMG = "pipeline_rgb/comparison/visualizations/safety_margin_histogram.png"

def main():
    if not os.path.exists(LOG_CSV):
        print("Run overlap_video.py first!")
        return

    df = pd.read_csv(LOG_CSV)
    
    # 'margin' column is (Actual_Dist - Required_Dist). 
    # Positive = Safe, Negative = Unsafe.
    margins = df['margin'].values

    plt.figure(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = plt.hist(margins, bins=30, alpha=0.7, color='blue', edgecolor='black')
    
    # Color bars based on safety
    for bin_val, patch in zip(bins, patches):
        if bin_val < 0:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('green')

    plt.axvline(0, color='k', linestyle='dashed', linewidth=2, label='Safety Threshold')
    
    plt.title(f"Safety Margin Distribution (N={len(df)} Frames)", fontsize=14)
    plt.xlabel("Safety Margin (meters)\n< 0: Unsafe | > 0: Safe", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)

    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    plt.savefig(OUT_IMG, dpi=150)
    print(f"[Done] Saved histogram to {OUT_IMG}")

if __name__ == "__main__":
    main()