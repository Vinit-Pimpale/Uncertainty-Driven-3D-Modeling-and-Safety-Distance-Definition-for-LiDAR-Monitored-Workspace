#!/usr/bin/env python3
"""
Safety Sensitivity Analysis
===========================
Simulates the safety check with varying Uncertainty Scale Factors.
Answers: "How does changing the safety strictness affect availability?"

Output: pipeline_rgb/comparison/visualizations/sensitivity_analysis.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We re-use the log file to get base distances
LOG_CSV = "pipeline_rgb/shape/model_stage2/robot_wall_overlap_log.csv"
OUT_IMG = "pipeline_rgb/comparison/visualizations/sensitivity_analysis.png"

# ISO Standard Distance
ISO_DIST = 0.60

def main():
    if not os.path.exists(LOG_CSV):
        print("Run overlap_video.py first!")
        return
        
    df = pd.read_csv(LOG_CSV)
    
    # We need to reverse-engineer the "Base Margin" and "Uncertainty Cost"
    # Current Margin = Dist - (ISO + Unc_Cost)
    # So, Dist - ISO = Margin + Unc_Cost
    # This is an approximation since we don't have the raw per-frame uncertainty cost stored in CSV.
    # Instead, we will simulate a simpler sensitivity:
    # "Impact of Safety Buffer Size on Violation Count"
    
    margins = df['margin'].values
    # Reconstruct the 'Actual Distance' approximation (assuming Unc=0 for base case)
    # This is a synthetic test: "If we increase the required buffer, how many frames fail?"
    
    buffer_increases = np.linspace(0.0, 0.5, 20) # Add 0cm to 50cm extra buffer
    violation_rates = []
    
    for buff in buffer_increases:
        # New Margin = Old Margin - Extra Buffer
        new_margins = margins - buff
        num_unsafe = np.sum(new_margins < 0)
        rate = (num_unsafe / len(margins)) * 100
        violation_rates.append(rate)
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(buffer_increases * 100, violation_rates, 'o-', linewidth=2, color='darkorange')
    
    plt.title("Sensitivity Analysis: Safety Buffer vs. Availability", fontsize=14)
    plt.xlabel("Additional Safety Buffer (cm)", fontsize=12)
    plt.ylabel("% of Frames Marked 'Unsafe'", fontsize=12)
    plt.grid(True, alpha=0.6)
    
    # Threshold line
    plt.axhline(y=violation_rates[0], color='grey', linestyle='--', label='Current Baseline')
    plt.legend()
    
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    plt.savefig(OUT_IMG, dpi=150)
    print(f"[Done] Sensitivity plot saved to: {OUT_IMG}")

if __name__ == "__main__":
    main()