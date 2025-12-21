#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm   # <-- Added for progress bar

# ---------------------------------------------------------
# Add KPConv-PyTorch root directory to Python path
# ---------------------------------------------------------
HERE = os.path.dirname(__file__)  # .../thesis_lidar_seg/pipeline_rgb/inference
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # .../thesis_lidar_seg
KPCONV = os.path.join(ROOT, "KPConv-PyTorch")

sys.path.insert(0, KPCONV)

# Now imports work correctly
from models.architectures import KPFCNN
from utils.ply import read_ply, write_ply
from utils.config import Config
from datasets.COVERED import COVEREDCollate

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
INPUT_DIR = os.path.join("pipeline_rgb", "preprocessing", "ready")
OUTPUT_DIR = os.path.join("pipeline_rgb", "inference", "predictions")

# Use an ABSOLUTE checkpoint path built from KPCONV
CHECKPOINT_PATH = os.path.join(
    KPCONV,
    "training_logs_SEYOND_KPFCNN",
    "Log_2025-11-04_02-22-11",
    "checkpoints",
    "best_chkp_adaptive.tar",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class colors for colored predictions
CLASS_COLORS = np.array([
    [200, 200, 200],  # 0 Floor (overwritten below if you want)
    [255,   0,   0],  # 1 Wall
    [139,  69,  19],  # 2 Column
    [255, 255,   0],  # 3 Robo Dog
    [128,   0, 128],  # 4 Screen+Stand
], dtype=np.uint8)

# ---------------------------------------------------------
# Dataset for frame-by-frame inference
# ---------------------------------------------------------
class SingleFrameRGBDataset(Dataset):

    def __init__(self, folder):
        self.folder = folder
        self.files = sorted(
            [f for f in os.listdir(folder) if f.endswith("_rgb.ply")]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder, self.files[idx])
        data = read_ply(file_path)

        pts = np.vstack((data["x"], data["y"], data["z"])).T.astype(np.float32)
        colors = np.vstack(
            (data["red"], data["green"], data["blue"])
        ).T / 255.0

        # KPConv expects F features; here: RGB + 2 dummy dims = 5
        dummy = np.zeros((pts.shape[0], 2), dtype=np.float32)
        feats = np.concatenate([colors, dummy], axis=1)

        labels = np.zeros(pts.shape[0], dtype=np.int32)  # dummy

        return pts, feats, labels, idx, self.files[idx]


# ---------------------------------------------------------
# Helper: load Config from parameters.txt
# ---------------------------------------------------------
def load_config_from_log(ckpt_path):
    """
    Given a checkpoint path, find the log directory and load
    KPConv's Config from parameters.txt in that directory.
    """
    log_dir = os.path.dirname(os.path.dirname(ckpt_path))
    print(f"[Inference] Loading Config from log directory: {log_dir}")

    cfg = Config()
    cfg.load(log_dir)
    cfg.saving = False  # disable saving in inference

    # Ensure critical fields exist
    if not hasattr(cfg, "num_classes"):
        cfg.num_classes = 5
    if not hasattr(cfg, "num_inputs"):
        cfg.num_inputs = 3  # RGB → 3 inputs

    return cfg


# ---------------------------------------------------------
# Main inference function
# ---------------------------------------------------------
def run_inference():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Inference] Using device: {device}")

    # ---- Load model config from parameters.txt ----
    config = load_config_from_log(CHECKPOINT_PATH)

    # ---- Load network ----
    print("[Inference] Loading KPConv model...")
    label_values = [0, 1, 2, 3, 4]
    ignored_labels = np.array([], dtype=np.int32)

    net = KPFCNN(config, label_values, ignored_labels)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device)
    net.eval()

    softmax = torch.nn.Softmax(dim=1)

    dataset = SingleFrameRGBDataset(INPUT_DIR)

    print(f"[Inference] Found {len(dataset)} frames.\n")

    # -------------------------------
    # Progress bar replaces print loop
    # -------------------------------
    for idx in tqdm(range(len(dataset)), desc="Inferring", unit="frame"):

        pts, feats, _, _, filename = dataset[idx]

        loader = DataLoader(
            [(pts, feats, np.zeros(len(pts)), 0)],
            batch_size=1,
            collate_fn=COVEREDCollate,
        )

        for batch in loader:
            batch.to(device)
            out = net(batch, config)

            prob = softmax(out).detach().cpu().numpy()
            pred = np.argmax(prob, axis=1).astype(np.int32)
            conf = prob.max(axis=1).astype(np.float32)

        base = filename.replace("_rgb.ply", "")
        pred_path = os.path.join(OUTPUT_DIR, base + "_pred.ply")
        pred_col_path = os.path.join(OUTPUT_DIR, base + "_pred_colored.ply")
        conf_path = os.path.join(OUTPUT_DIR, base + "_conf.ply")

        # ---- Save raw predictions ----
        write_ply(pred_path, [pts, pred], ["x", "y", "z", "pred"])

        # ---- Save colored predictions ----
        colors = CLASS_COLORS[np.clip(pred, 0, len(CLASS_COLORS) - 1)]
        write_ply(
            pred_col_path,
            [pts, colors[:, 0], colors[:, 1], colors[:, 2], pred],
            ["x", "y", "z", "red", "green", "blue", "pred"],
        )

        # ---- Save confidence ----
        write_ply(
            conf_path,
            [pts, conf],
            ["x", "y", "z", "conf"],
        )

    print("\n✓ Frame-by-frame inference complete.\n")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    run_inference()
