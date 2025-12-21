#!/usr/bin/env python3
"""
Generate Uncertainty Calibration Plots (Reliability Diagrams)
===========================================================

Goal:
  Validate that the model's 'mean_prob' (confidence) actually corresponds
  to its true accuracy.

Inputs:
  - Ground Truth: pipeline_rgb/ground_truth/gt_ply/frame_XXXX_labeled.ply
  - Uncertainty:  pipeline_rgb/inference/uncertainty_mc/frame_XXXX_uncertainty_mc.ply

Outputs:
  - pipeline_rgb/comparison/visualizations/calibration_plot.png
  - Console output of ECE (Expected Calibration Error)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from sklearn.neighbors import KDTree

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
GT_DIR  = "pipeline_rgb/ground_truth/gt_ply"
UNC_DIR = "pipeline_rgb/inference/uncertainty_mc"
OUT_IMG = "pipeline_rgb/comparison/visualizations/calibration_plot.png"

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def read_ply(path):
    ply = PlyData.read(path)
    data = ply['vertex'].data
    return data

def get_matching_files(gt_dir, unc_dir):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith("_labeled.ply")])
    pairs = []
    for gt in gt_files:
        fid = gt.replace("_labeled.ply", "")
        unc_name = f"{fid}_uncertainty_mc.ply"
        if os.path.exists(os.path.join(unc_dir, unc_name)):
            pairs.append((fid, os.path.join(gt_dir, gt), os.path.join(unc_dir, unc_name)))
    return pairs

# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
def main():
    pairs = get_matching_files(GT_DIR, UNC_DIR)
    
    if not pairs:
        print("[Error] No matching GT and Uncertainty frames found.")
        return

    print(f"[Calibration] Found {len(pairs)} frames. Accumulating data...")

    all_confidences = []
    all_accuracies = []

    # Process every 5th frame to save time/memory (or remove slice for full run)
    for i, (fid, gt_path, unc_path) in enumerate(pairs[::5]):
        print(f"  Processing {fid}...", end="\r")

        # Load Data
        gt_data = read_ply(gt_path)
        unc_data = read_ply(unc_path)

        # 1. Align Points (KDTree)
        # GT and Inference clouds might have slightly different point orders
        # or counts if preprocessing changed. We match nearest neighbors.
        gt_pts = np.vstack((gt_data["x"], gt_data["y"], gt_data["z"])).T
        unc_pts = np.vstack((unc_data["x"], unc_data["y"], unc_data["z"])).T
        
        # Build tree on GT
        tree = KDTree(gt_pts)
        dists, indices = tree.query(unc_pts, k=1)
        
        # Valid matches (< 2cm distance)
        valid_mask = dists[:,0] < 0.02
        indices = indices[:,0][valid_mask]
        
        # 2. Extract Metrics
        # GT Label
        y_true = gt_data["label"][indices].astype(np.int32)
        
        # Pred Label & Confidence (mean_prob)
        # Note: uncertainty_mc.ply has 'pred' and 'mean_prob'
        y_pred = unc_data["pred"][valid_mask].astype(np.int32)
        conf   = unc_data["mean_prob"][valid_mask].astype(np.float32)
        
        # Correctness (1 if Pred == GT, else 0)
        is_correct = (y_pred == y_true).astype(np.int32)

        # 3. Store
        all_confidences.append(conf)
        all_accuracies.append(is_correct)

    print("\n[Calibration] Computing bins...")
    
    # Flatten arrays
    confidences = np.concatenate(all_confidences)
    accuracies  = np.concatenate(all_accuracies)

    # ---------------------------------------------------------
    # Compute Calibration Curve (Binning)
    # ---------------------------------------------------------
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        # Find samples in this bin
        bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if i == n_bins - 1: # Include 1.0 in last bin
            bin_mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i+1])
            
        count = np.sum(bin_mask)
        if count > 0:
            avg_conf = np.mean(confidences[bin_mask])
            avg_acc  = np.mean(accuracies[bin_mask])
            
            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
            bin_counts.append(count)
            
            # ECE Calculation
            # | Accuracy - Confidence | * (samples_in_bin / total_samples)
            ece += np.abs(avg_acc - avg_conf) * (count / total_samples)
        else:
            # Empty bin handling
            bin_accs.append(0.0)
            bin_confs.append((bin_edges[i] + bin_edges[i+1])/2)
            bin_counts.append(0)

    print(f"------------------------------------------------")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"------------------------------------------------")

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(6, 6))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    
    # Our model
    plt.plot(bin_confs, bin_accs, "s-", label=f"KPConv (ECE={ece:.3f})")

    # Formatting
    plt.xlabel("Confidence (Predicted Probability)")
    plt.ylabel("Accuracy (Fraction Correct)")
    plt.title("Reliability Diagram (Uncertainty Calibration)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Save
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    plt.savefig(OUT_IMG)
    print(f"[Calibration] Plot saved to: {OUT_IMG}")

if __name__ == "__main__":
    main()