#!/usr/bin/env python3
"""
MC Dropout + TTA (with JITTER) Uncertainty Inference
----------------------------------------------------

Updated: Added Gaussian Jitter to TTA.
Now performs:
  1. Identity
  2. Flip X
  3. Flip Y
  4. Gaussian Jitter (sigma=0.01m) <--- NEW
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# KPConv Path Setup
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KPCONV = os.path.join(ROOT, "KPConv-PyTorch")
sys.path.insert(0, KPCONV)

from models.architectures import KPFCNN
from utils.ply import read_ply, write_ply
from utils.config import Config
from datasets.COVERED import COVEREDCollate

# Paths
INPUT_DIR = os.path.join("pipeline_rgb", "preprocessing", "ready")
OUTPUT_DIR = os.path.join("pipeline_rgb", "inference", "uncertainty_mc")
CHECKPOINT_PATH = os.path.join(
    KPCONV, "training_logs_SEYOND_KPFCNN", "Log_2025-11-04_02-22-11", "checkpoints", "best_chkp_adaptive.tar"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

class SingleFrameRGBDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith("_rgb.ply")])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        file_path = os.path.join(self.folder, self.files[idx])
        data = read_ply(file_path)
        pts = np.vstack((data["x"], data["y"], data["z"])).T.astype(np.float32)
        colors = np.vstack((data["red"], data["green"], data["blue"])).T / 255.0
        dummy = np.zeros((pts.shape[0], 2), dtype=np.float32)
        feats = np.concatenate([colors, dummy], axis=1)
        labels = np.zeros(pts.shape[0], dtype=np.int32)
        return pts, feats, labels, idx, self.files[idx]

def load_config_from_log(ckpt_path):
    log_dir = os.path.dirname(os.path.dirname(ckpt_path))
    cfg = Config()
    cfg.load(log_dir)
    cfg.saving = False
    if not hasattr(cfg, "num_classes"): cfg.num_classes = 5
    if not hasattr(cfg, "num_inputs"): cfg.num_inputs = 3
    return cfg

def mc_forward(net, batch, config, softmax, runs=10):
    probs = []
    for _ in range(runs):
        out = net(batch, config)
        probs.append(softmax(out).detach().cpu().numpy())
    probs = np.stack(probs, axis=0)
    return probs.mean(axis=0), probs.var(axis=0).sum(axis=1)

# --- UPDATED TTA FUNCTION WITH JITTER ---
def tta_forward(net, pts, feats, labels, config, softmax, device):
    """
    Apply TTA: Flips AND Gaussian Jitter.
    """
    def flip_x(p): 
        f = p.copy(); f[:,0] *= -1; return f
    def flip_y(p): 
        f = p.copy(); f[:,1] *= -1; return f
    
    # NEW: Jitter Function (Simulates Sensor Noise)
    def add_jitter(p):
        # 1cm noise (0.01m) standard deviation
        noise = np.random.normal(0, 0.01, p.shape).astype(np.float32)
        return p + noise

    TTAs = [
        lambda p: p,    # Identity
        flip_x,         # Flip X
        flip_y,         # Flip Y
        add_jitter      # Gaussian Jitter (NEW)
    ]

    outputs = []
    for tta in TTAs:
        pts_aug = tta(pts)
        loader = DataLoader([(pts_aug, feats, labels, 0)], batch_size=1, collate_fn=COVEREDCollate)
        for batch in loader:
            batch.to(device)
            out = net(batch, config)
            outputs.append(softmax(out).detach().cpu().numpy())

    outputs = np.stack(outputs, axis=0)
    return outputs.mean(axis=0), outputs.var(axis=0).sum(axis=1)

def compute_aleatoric(prob):
    return -(prob * np.log(prob + 1e-9)).sum(axis=1)

def run_uncertainty_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Uncertainty] Device: {device} | Mode: MC Dropout + TTA (w/ Jitter)")

    config = load_config_from_log(CHECKPOINT_PATH)
    label_values = [0, 1, 2, 3, 4]
    net = KPFCNN(config, label_values, np.array([], dtype=np.int32))
    
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device)
    net.train() # Dropout ON

    softmax = torch.nn.Softmax(dim=1)
    dataset = SingleFrameRGBDataset(INPUT_DIR)

    for idx in tqdm(range(len(dataset)), desc="Inferring", unit="frame"):
        pts, feats, labels, _, filename = dataset[idx]

        # 1. TTA (Now includes Jitter)
        mean_tta, tta_var = tta_forward(net, pts, feats, labels, config, softmax, device)

        # 2. MC Dropout
        loader = DataLoader([(pts, feats, labels, 0)], batch_size=1, collate_fn=COVEREDCollate)
        for batch in loader:
            batch.to(device)
            mean_mc, mc_var = mc_forward(net, batch, config, softmax, runs=10)

        # Combine
        final_prob = 0.5 * mean_tta + 0.5 * mean_mc
        pred = np.argmax(final_prob, axis=1).astype(np.int32)
        mean_prob = final_prob.max(axis=1).astype(np.float32)
        entropy = -(final_prob * np.log(final_prob + 1e-9)).sum(axis=1)
        aleatoric = compute_aleatoric(final_prob)

        # Save
        base = filename.replace("_rgb.ply", "")
        out_path = os.path.join(OUTPUT_DIR, base + "_uncertainty_mc.ply")
        write_ply(out_path, [pts, pred, mean_prob, entropy, tta_var.astype(np.float32), 
                             mc_var.astype(np.float32), aleatoric.astype(np.float32)],
                  ["x","y","z","pred","mean_prob","entropy","tta_var","mc_var","aleatoric"])

    print("\n✓ Uncertainty Inference (MC + TTA-Jitter) Complete.\n")

if __name__ == "__main__":
    run_uncertainty_inference()