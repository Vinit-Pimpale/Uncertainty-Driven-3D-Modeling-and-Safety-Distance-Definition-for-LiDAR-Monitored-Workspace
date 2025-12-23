#!/usr/bin/env python3
import os
import argparse
import numpy as np
from plyfile import PlyData

def read_ply(path):
    ply = PlyData.read(path)
    data = ply['vertex'].data
    out = {}
    for k in data.dtype.names:
        out[k] = data[k]
    return out

def analyze_file(path):
    print(f"\n=== Checking: {os.path.basename(path)} ===")

    d = read_ply(path)

    # ---- Check available fields ----
    required = ["pred", "conf", "entropy", "tta_var"]
    print("\nFields found:", list(d.keys()))
    missing = [r for r in required if r not in d]
    if missing:
        print("❌ Missing fields:", missing)
        return

    pred     = d["pred"]
    conf     = d["conf"]
    entropy  = d["entropy"]
    tta_var  = d["tta_var"]

    # ---- Basic sanity checks ----
    print("\n--- Sanity Checks ---")
    print(f"Pred classes: {np.unique(pred)}")

    print(f"conf: min={conf.min():.4f}, max={conf.max():.4f}, mean={conf.mean():.4f}, std={conf.std():.4f}")
    print(f"entropy: min={entropy.min():.4f}, max={entropy.max():.4f}, mean={entropy.mean():.4f}, std={entropy.std():.4f}")
    print(f"tta_var: min={tta_var.min():.6f}, max={tta_var.max():.6f}, mean={tta_var.mean():.6f}, std={tta_var.std():.6f}")

    # ---- Check ranges ----
    print("\n--- Range Validity ---")
    if conf.min() < 0 or conf.max() > 1:
        print("❌ conf out of valid range [0,1]")
    else:
        print("✔ conf in valid range [0,1]")

    if (entropy < 0).any():
        print("❌ entropy has negative values")
    else:
        print("✔ entropy >= 0")

    # ---- Correlation tests ----
    print("\n--- Correlation Tests ---")
    corr_conf_entropy = np.corrcoef(conf, entropy)[0,1]
    print(f"Correlation(conf, entropy) = {corr_conf_entropy:.4f} (should be strongly negative)")

    corr_conf_tta = np.corrcoef(conf, tta_var)[0,1]
    print(f"Correlation(conf, tta_var) = {corr_conf_tta:.4f} (should be negative or weak)")

    corr_entropy_tta = np.corrcoef(entropy, tta_var)[0,1]
    print(f"Correlation(entropy, tta_var) = {corr_entropy_tta:.4f} (should be positive or weak)")

    # ---- Non-zero TTA variance check ----
    print("\n--- TTA Variance Check ---")
    zero_ratio = np.mean(tta_var == 0)
    print(f"TTA zero-ratio: {zero_ratio:.3f}")
    if zero_ratio > 0.95:
        print("❌ Almost all TTA variance is zero — TTA may NOT be applied correctly.")
    else:
        print("✔ TTA variance contains real values")

    # ---- Special attention to RoboDog class (3) ----
    print("\n--- RoboDog Class (3) Stats ---")
    mask_dog = (pred == 3)
    if mask_dog.sum() == 0:
        print("⚠ No RoboDog points detected in predictions.")
    else:
        print(f"RoboDog points: {mask_dog.sum()}")
        print(f"Dog conf: mean={conf[mask_dog].mean():.4f}, std={conf[mask_dog].std():.4f}")
        print(f"Dog entropy: mean={entropy[mask_dog].mean():.4f}, std={entropy[mask_dog].std():.4f}")
        print(f"Dog tta_var: mean={tta_var[mask_dog].mean():.6f}, std={tta_var[mask_dog].std():.6f}")

    print("\n=== Done ===\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Single PLY file")
    parser.add_argument("--dir",  type=str, help="Directory with PLY files")

    args = parser.parse_args()

    if args.file:
        analyze_file(args.file)
        return

    if args.dir:
        for f in sorted(os.listdir(args.dir)):
            if f.endswith(".ply"):
                analyze_file(os.path.join(args.dir, f))
        return

    print("Please use --file <ply> or --dir <folder>")

if __name__ == "__main__":
    main()
