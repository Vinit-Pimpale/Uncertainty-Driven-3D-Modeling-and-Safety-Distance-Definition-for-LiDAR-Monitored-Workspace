import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors

from models.architectures import KPFCNN
from train_SEYOND import SeyondConfig
from utils.ply import read_ply, write_ply
from datasets.COVERED import COVEREDCollate


# --------------------- PATHS ---------------------
CHECKPOINT_PATH = (
    "training_logs_SEYOND_KPFCNN/Log_2025-11-04_02-22-11/"
    "checkpoints/best_chkp_adaptive.tar"
)
INPUT_DIR = "data_local/Input_RGB"
OUTPUT_DIR = "data_local/results_RGB"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ------------------------------------------------

# 0: Floor, 1: Wall, 2: Column, 3: Robo Dog, 4: Screen+Stand
CLASS_COLORS = np.array([
    [200, 200, 200],    # Floor
    [255,   0,   0],    # Wall
    [139,  69,  19],    # Column
    [255, 255,   0],    # Robo Dog
    [128,   0, 128],    # Screen+Stand
], dtype=np.uint8)


class InputRGBDataset(Dataset):
    """
    Minimal dataset for inference on data_local/Input_RGB.
    Returns (points, features, dummy_labels, cloud_index) in the same
    format expected by COVEREDCollate.
    """

    def __init__(self, config, root_dir):
        super().__init__()
        self.config = config
        self.name = "Input_RGB"
        self.path = root_dir
        self.set = "test"

        # Files in the folder
        self.files = sorted(
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".ply")
        )

        # Class meta (same as SeyondLiDARDataset)
        self.label_to_names = {
            0: "Floor",
            1: "Wall",
            2: "Column",
            3: "Robo Dog",
            4: "Screen+Stand",
        }
        self.label_values = np.sort(list(self.label_to_names.keys()))
        self.num_classes = len(self.label_values)
        self.ignored_labels = np.array([], dtype=np.int32)

        # Neighborhood limits from saved config
        # (COVEREDCollate uses these)
        if hasattr(config, "neighborhood_limits"):
            self.neighborhood_limits = np.array(config.neighborhood_limits)
        else:
            self.neighborhood_limits = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Return one cloud in the format:
          points: (N,3)
          features: (N, in_features_dim)  -> here [R,G,B,0,0]
          labels: (N,) dummy (all zeros, not used)
          cloud_index: int
        """
        f = self.files[idx]
        data = read_ply(f)

        pts = np.vstack((data["x"], data["y"], data["z"])).T.astype(np.float32)
        cols = np.vstack((data["red"], data["green"], data["blue"])).T / 255.0

        # Add two dummy channels just like SeyondLiDARDataset.get_cloud
        dummy = np.zeros((pts.shape[0], 2), dtype=np.float32)
        feats = np.concatenate([cols, dummy], axis=1).astype(np.float32)

        # Dummy labels – network does not need them for inference
        lbl = np.zeros(pts.shape[0], dtype=np.int32)

        return pts, feats, lbl, idx


def estimate_normals_fast(points, k=10, subsample=5):
    """Fast approximate normal estimation via subsampling + interpolation."""
    if len(points) < k:
        return np.zeros_like(points, dtype=np.float32)

    step = max(1, subsample)
    sub_pts = points[::step]

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(sub_pts)
    _, indices = nbrs.kneighbors(sub_pts)
    sub_normals = np.zeros_like(sub_pts, dtype=np.float32)

    for i in range(sub_pts.shape[0]):
        neigh = sub_pts[indices[i, 1:]]
        cov = np.cov(neigh - neigh.mean(axis=0), rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        sub_normals[i] = eigvecs[:, 0]
    sub_normals = np.where(sub_normals[:, 2:3] < 0, -sub_normals, sub_normals)

    # Interpolate normals back to all points
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(sub_pts)
    _, idx = nn.kneighbors(points)
    normals = sub_normals[idx[:, 0]]
    return normals.astype(np.float32)


if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # 1) Load config (from training log) and model
    print("\nLoading SEYOND configuration ...")
    log_dir = os.path.dirname(os.path.dirname(CHECKPOINT_PATH))
    config = SeyondConfig()
    config.load(log_dir)
    config.saving = False
    config.input_threads = 0

    print("\nPreparing model ...")
    net = KPFCNN(config, [0, 1, 2, 3, 4], np.array([], dtype=np.int32))
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device)
    net.eval()

    softmax = torch.nn.Softmax(dim=1)

    # 2) Dataset + DataLoader on data_local/Input_RGB
    print("\nData Preparation\n****************")
    ds = InputRGBDataset(config, INPUT_DIR)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=COVEREDCollate,
        num_workers=0,
        pin_memory=False,
    )

    # 3) Run inference
    print("\nStart SEYOND Test on Input_RGB\n********************************")
    all_pts, all_pred, all_conf = [], [], []

    with torch.no_grad():
        for batch in loader:
            # COVEREDCollate builds the full KPConv batch:
            #   batch.points, batch.neighbors, batch.pools, batch.upsamples, ...
            batch.to(device)

            out = net(batch, config)
            prob = softmax(out).cpu().numpy()
            pred = np.argmax(prob, axis=1).astype(np.int32)
            conf = prob.max(axis=1).astype(np.float32)

            pts = batch.points[0].cpu().numpy()

            all_pts.append(pts)
            all_pred.append(pred)
            all_conf.append(conf)

    # 4) Merge all frames
    all_pts = np.vstack(all_pts)
    all_pred = np.concatenate(all_pred)
    all_conf = np.concatenate(all_conf)

    # 5) Normals for visualization
    print("\nEstimating normals (fast) ...")
    normals = estimate_normals_fast(all_pts, k=10, subsample=5)

    # 6) Save outputs in data_local/results (no potentials)
    print("\nSaving results to data_local/results/\n")

    # a) Raw predictions
    pred_ply = os.path.join(OUTPUT_DIR, "predictions_RGB.ply")
    write_ply(
        pred_ply,
        [all_pts, normals, all_pred.astype(np.int32)],
        ["x", "y", "z", "nx", "ny", "nz", "preds"],
    )

    # b) Colored predictions
    colors = CLASS_COLORS[np.clip(all_pred, 0, len(CLASS_COLORS) - 1)]
    color_ply = os.path.join(OUTPUT_DIR, "predictions_RGB_colored.ply")
    write_ply(
        color_ply,
        [
            all_pts,
            normals,
            colors[:, 0].astype(np.uint8),
            colors[:, 1].astype(np.uint8),
            colors[:, 2].astype(np.uint8),
            all_pred.astype(np.int32),
        ],
        ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "preds"],
    )

    # c) Confidence
    conf_ply = os.path.join(OUTPUT_DIR, "confidence_RGB.ply")
    write_ply(
        conf_ply,
        [all_pts, all_conf.astype(np.float32)],
        ["x", "y", "z", "conf"],
    )

    print("Test completed successfully.")
    print(f"  Predictions         : {pred_ply}")
    print(f"  Colored predictions : {color_ply}")
    print(f"  Confidence map      : {conf_ply}")
