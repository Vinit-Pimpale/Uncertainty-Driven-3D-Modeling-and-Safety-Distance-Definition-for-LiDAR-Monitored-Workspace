#!/usr/bin/env python3
"""
Compare per-frame prediction PLYs with per-frame ground-truth PLYs.
Computes:
  - Overall Accuracy
  - Per-class Precision, Recall, F1
  - Per-class IoU
  - Macro F1-score
  - Confusion Matrix
  - Confidence stats
  - Entropy stats

Inputs:
    pipeline_rgb/inference/predictions/frame_XXXX_pred.ply
    pipeline_rgb/ground_truth/gt_ply/frame_XXXX_labeled.ply

Outputs:
    pipeline_rgb/comparison/metrics/metrics.txt
    pipeline_rgb/comparison/visualizations/confusion_matrix.png
"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from plyfile import PlyData


# ---------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------
def progress_bar(current, total, bar_length=40):
    frac = current / total
    filled = int(frac * bar_length)
    bar = "█" * filled + "-" * (bar_length - filled)
    print(f"\r[Compare] Matching frames: |{bar}| {current}/{total}", end="", flush=True)


# ---------------------------------------------------------
# Utility: read PLY into dict
# ---------------------------------------------------------
def read_ply(path):
    ply = PlyData.read(path)
    data = ply['vertex'].data
    arr = {}
    for k in data.dtype.names:
        arr[k] = data[k]
    return arr


# ---------------------------------------------------------
# Load GT and Prediction pairs
# ---------------------------------------------------------
def load_frame_pairs(gt_dir, pred_dir):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith("_labeled.ply")])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith("_pred.ply")])

    pairs = []
    for gt in gt_files:
        frame_id = gt.replace("_labeled.ply", "")
        pred_name = frame_id + "_pred.ply"
        if pred_name in pred_files:
            pairs.append((os.path.join(gt_dir, gt),
                          os.path.join(pred_dir, pred_name),
                          frame_id))
    return pairs


# ---------------------------------------------------------
# KD-tree nearest neighbor
# ---------------------------------------------------------
def match_points(gt_pts, pred_pts):
    from sklearn.neighbors import KDTree
    tree = KDTree(gt_pts)
    dist, idx = tree.query(pred_pts, k=1)

    dist = dist[:, 0]
    idx = idx[:, 0]

    valid_mask = dist < 0.02   # 2 cm
    return idx, valid_mask


# ---------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------
def evaluate(gt_dir, pred_dir, out_metrics, out_fig):
    pairs = load_frame_pairs(gt_dir, pred_dir)

    if len(pairs) == 0:
        print("No matching GT–Prediction frames found.")
        return

    print(f"[Compare] Found {len(pairs)} matching frame pairs.\n")

    all_gt_labels = []
    all_pred_labels = []
    all_conf = []
    all_entropy = []

    num_classes = 5
    class_names = ["Floor", "Wall", "Column", "RoboDog", "Screen+Stand"]

    total_frames = len(pairs)

    # ---------------------------------------------------------
    # Frame-by-frame comparison
    # ---------------------------------------------------------
    for i, (gt_path, pred_path, frame_id) in enumerate(pairs):

        # Update progress bar
        progress_bar(i + 1, total_frames)

        gt_data = read_ply(gt_path)
        pred_data = read_ply(pred_path)

        gt_pts = np.vstack((gt_data["x"], gt_data["y"], gt_data["z"])).T
        pred_pts = np.vstack((pred_data["x"], pred_data["y"], pred_data["z"])).T

        gt_labels = gt_data["label"]
        pred_labels = pred_data["pred"]

        # Confidence handling
        if "conf" in pred_data:
            conf = pred_data["conf"]
        else:
            conf = np.ones(len(pred_labels))
        all_conf.append(conf)

        # Entropy (placeholder)
        entropy = -conf * np.log(conf + 1e-9)
        all_entropy.append(entropy)

        idx, valid_mask = match_points(gt_pts, pred_pts)

        matched_gt = gt_labels[idx][valid_mask]
        matched_pred = pred_labels[valid_mask]

        all_gt_labels.append(matched_gt)
        all_pred_labels.append(matched_pred)

    print("\n")  # Finish progress bar cleanly

    # ---------------------------------------------------------
    # Merge all frames
    # ---------------------------------------------------------
    all_gt = np.concatenate(all_gt_labels)
    all_pred = np.concatenate(all_pred_labels)
    all_conf = np.concatenate(all_conf)
    all_entropy = np.concatenate(all_entropy)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    oa = accuracy_score(all_gt, all_pred)

    prec = precision_score(all_gt, all_pred, average=None, labels=range(num_classes), zero_division=0)
    rec  = recall_score(all_gt, all_pred, average=None, labels=range(num_classes), zero_division=0)
    f1   = f1_score(all_gt, all_pred, average=None, labels=range(num_classes), zero_division=0)
    macro_f1 = f1_score(all_gt, all_pred, average="macro")

    cm = confusion_matrix(all_gt, all_pred, labels=range(num_classes))

    # IoU per class
    iou = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        denom = tp + fp + fn
        iou.append(tp / denom if denom > 0 else 0.0)

    conf_stats = {
        "min": float(np.min(all_conf)),
        "max": float(np.max(all_conf)),
        "mean": float(np.mean(all_conf)),
        "std": float(np.std(all_conf)),
    }

    entropy_stats = {
        "min": float(np.min(all_entropy)),
        "max": float(np.max(all_entropy)),
        "mean": float(np.mean(all_entropy)),
        "std": float(np.std(all_entropy)),
    }

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------
    with open(out_metrics, "w") as f:
        f.write("=== RGB Evaluation Metrics ===\n\n")
        f.write(f"Overall Accuracy: {oa:.2%}\n")
        f.write(f"Macro F1-score:  {macro_f1:.4f}\n\n")

        for i, name in enumerate(class_names):
            f.write(f"{name:<15} | IoU: {iou[i]*100:.2f}% "
                    f"| Precision: {prec[i]*100:.2f}% "
                    f"| Recall: {rec[i]*100:.2f}% "
                    f"| F1: {f1[i]*100:.2f}%\n")

        f.write("\nConfidence Statistics:\n")
        for k, v in conf_stats.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nEntropy Statistics:\n")
        for k, v in entropy_stats.items():
            f.write(f"  {k}: {v}\n")

    print(f"[Compare] Saved metrics → {out_metrics}")

    # ---------------------------------------------------------
    # Confusion matrix figure
    # ---------------------------------------------------------
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(out_fig)
    print(f"[Compare] Saved confusion matrix → {out_fig}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    GT_DIR = "pipeline_rgb/ground_truth/gt_ply"
    PRED_DIR = "pipeline_rgb/inference/predictions"
    OUT_METRICS = "pipeline_rgb/comparison/metrics/metrics.txt"
    OUT_FIG = "pipeline_rgb/comparison/visualizations/confusion_matrix.png"

    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

    evaluate(GT_DIR, PRED_DIR, OUT_METRICS, OUT_FIG)

    print("\n✓ Evaluation complete.\n")
