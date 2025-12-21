#!/usr/bin/env python3

"""
Full RGB Pipeline Runner (Uncertainty-Aware Version)
===================================================

Pipeline order:

  1. Check PLY frames
  1.5. Rename frames
  2. RGB preprocessing
  3. Ground truth generation
  4. KPConv inference (predictions)
  5. Evaluation
  6. Uncertainty inference (MC Dropout + TTA)
  7. Uncertainty-aware RoboDog extraction (highlight with filtering)

Everything else remains unchanged.
"""

import os
import sys
import argparse
from datetime import datetime

# ---------------------------------------------------------
# Setup paths
# ---------------------------------------------------------
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PIPELINE = os.path.join(ROOT, "pipeline_rgb")
KPCONV = os.path.join(ROOT, "KPConv-PyTorch")

sys.path.insert(0, PIPELINE)
sys.path.insert(0, KPCONV)

# ---------------------------------------------------------
# Import pipeline stage functions
# ---------------------------------------------------------
from pipeline_rgb.preprocessing.preprocess_rgb import main as preprocess_rgb_main
from pipeline_rgb.ground_truth.generate_gt_rgb import main as generate_gt_main
from pipeline_rgb.inference.run_inference_rgb import run_inference
from pipeline_rgb.inference.run_inference_rgb_uncertainty_mc import run_uncertainty_inference
from pipeline_rgb.comparison.compare_inference_vs_gt import evaluate
from pipeline_rgb.utils.rename_frames import rename_files

# NEW: import the uncertainty-aware highlight script
from pipeline_rgb.shape.highlight_robot import highlight_robot_frames


# ---------------------------------------------------------
# Folders
# ---------------------------------------------------------
PLY_FRAMES = os.path.join("pipeline_rgb", "input", "ply_frames")
READY_DIR = os.path.join("pipeline_rgb", "preprocessing", "ready")
GT_DIR = os.path.join("pipeline_rgb", "ground_truth", "gt_ply")
PRED_DIR = os.path.join("pipeline_rgb", "inference", "predictions")
UNC_DIR = os.path.join("pipeline_rgb", "inference", "uncertainty_mc")
METRICS_TXT = os.path.join("pipeline_rgb", "comparison", "metrics", "metrics.txt")
CONF_MATRIX = os.path.join("pipeline_rgb", "comparison", "visualizations", "confusion_matrix.png")


def folder_has_ply(folder):
    return os.path.exists(folder) and any(f.endswith(".ply") for f in os.listdir(folder))


# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def run_pipeline(bag_path: str):
    print("\n==============================")
    print("     FULL RGB PIPELINE RUN    ")
    print("==============================\n")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Start Time: {timestamp}")
    print(f"Bag argument (logged only): {bag_path}\n")

    # -----------------------------------------------------
    # 1. Check PLY input frames
    # -----------------------------------------------------
    print("[1] Checking PLY frames...")

    if folder_has_ply(PLY_FRAMES):
        n = len([f for f in os.listdir(PLY_FRAMES) if f.endswith(".ply")])
        print(f"  Found {n} PLY frame(s).")
    else:
        print(f"  No PLY frames found in '{PLY_FRAMES}'.")
        print("  Please place your .ply frames in that folder.")
        return

    print("\n[1.5] Renaming PLY frames...")
    rename_files(PLY_FRAMES)

    # -----------------------------------------------------
    # 2. RGB Preprocessing
    # -----------------------------------------------------
    print("\n[2] RGB Preprocessing")

    if folder_has_ply(READY_DIR):
        n = len([f for f in os.listdir(READY_DIR) if f.endswith(".ply")])
        print(f"  Found {n} preprocessed frames → Skipping.\n")
    else:
        print("  Running RGB preprocessing...")
        preprocess_rgb_main()
        print("  Preprocessing completed.\n")

    # -----------------------------------------------------
    # 3. Ground Truth Generation
    # -----------------------------------------------------
    print("[3] Ground Truth Generation")

    if folder_has_ply(GT_DIR):
        n = len([f for f in os.listdir(GT_DIR) if f.endswith(".ply")])
        print(f"  Found {n} GT frames → Skipping.\n")
    else:
        print("  Generating ground truth...")
        generate_gt_main()
        print("  GT generation completed.\n")

    # -----------------------------------------------------
    # 4. Standard KPConv Inference
    # -----------------------------------------------------
    print("[4] KPConv Inference")
    run_inference()
    print("  Inference completed.\n")

    # -----------------------------------------------------
    # 5. Evaluation
    # -----------------------------------------------------
    print("[5] Evaluation")

    os.makedirs(os.path.dirname(METRICS_TXT), exist_ok=True)
    os.makedirs(os.path.dirname(CONF_MATRIX), exist_ok=True)

    evaluate(GT_DIR, PRED_DIR, METRICS_TXT, CONF_MATRIX)

    print("  Evaluation completed.")
    print(f"  Metrics saved to: {METRICS_TXT}")
    print(f"  Confusion matrix: {CONF_MATRIX}\n")

    # -----------------------------------------------------
    # 6. Uncertainty Inference (MC Dropout + TTA)
    # -----------------------------------------------------
    print("[6] Uncertainty Inference (MC Dropout + TTA)")
    run_uncertainty_inference()
    print("  Uncertainty inference completed.\n")

    # -----------------------------------------------------
    # 7. Uncertainty-Aware RoboDog Highlighting
    # -----------------------------------------------------
    print("[7] Highlight RoboDog (with uncertainty filtering)")

    highlight_dir = os.path.join("pipeline_rgb", "shape", "highlighted_uncertainty")

    highlight_robot_frames(
        pred_dir=PRED_DIR,
        unc_dir=UNC_DIR,
        out_dir=highlight_dir,
        robo_label=3,
    )

    print("  Highlighting completed.\n")

    print("================================")
    print("       PIPELINE COMPLETED       ")
    print("================================\n")


# ---------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to ROS2 bag folder (logged only).",
    )
    args = parser.parse_args()

    run_pipeline(args.bag)
