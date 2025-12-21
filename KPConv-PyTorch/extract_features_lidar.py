import os, sys, torch, numpy as np
from torch.utils.data import DataLoader
from models.architectures import KPFCNN
from utils.config import Config
from datasets.LidarDataset import LidarDataset

# ------------------------------------------------------------------------------
# Import low-level KPConv utilities (based on your compiled wrappers)
# ------------------------------------------------------------------------------
import cpp_wrappers.cpp_subsampling.grid_subsampling as gs
import cpp_wrappers.cpp_neighbors.neighbors as nb

# Your compiled module exposes 'subsample' and 'subsample_batch'
grid_subsampling = gs.subsample_batch   # correct function name for your build
batch_neighbors = nb.batch_neighbors if hasattr(nb, 'batch_neighbors') else nb.neighbors


# ------------------------------------------------------------------------------
# Helper class: build full KPConv batch hierarchy for raw point clouds
# ------------------------------------------------------------------------------
class KPConvBatch:
    """
    Stand-alone builder that mimics the KPConv SimpleBatch structure.
    Creates multiscale points, neighbor, pooling and upsample indices.
    """

    def __init__(self, cfg, points, features):
        self.points = []          # list of torch tensors (one per layer)
        self.neighbors = []       # neighbor indices per layer
        self.pools = []           # pooling indices per layer
        self.upsamples = []       # upsample indices per layer
        self.lengths = []         # number of points per layer
        self.features = features  # [N,C]
        self.input_inds = []
        self.cloud_inds = []

        # Convert to float32 if needed
        if not torch.is_tensor(points):
            points = torch.from_numpy(points).float()
        points = points.cuda()

        # --- 1. base layer points ---
        self.points.append(points)

        # --- 2. build hierarchy (subsampled clouds) ---
        sub_points = points.cpu().numpy()
        for layer in range(cfg.num_layers):
            sub_points = grid_subsampling(sub_points, sampleDl=cfg.subsampling_dl[layer])
            self.points.append(torch.from_numpy(sub_points).cuda().float())

        # --- 3. compute neighborhood, pool, and upsample indices ---
        for i in range(cfg.num_layers):
            r = cfg.kp_radius * cfg.dilation[i]

            neighb = batch_neighbors(
                self.points[i].cpu().numpy(), self.points[i].cpu().numpy(), radius=r
            )
            pool = batch_neighbors(
                self.points[i + 1].cpu().numpy(), self.points[i].cpu().numpy(), radius=r
            )
            up = batch_neighbors(
                self.points[i].cpu().numpy(), self.points[i + 1].cpu().numpy(), radius=r
            )

            self.neighbors.append(torch.from_numpy(neighb).int().cuda())
            self.pools.append(torch.from_numpy(pool).int().cuda())
            self.upsamples.append(torch.from_numpy(up).int().cuda())

        # --- 4. misc. bookkeeping (single cloud batch) ---
        self.lengths = [torch.tensor([p.shape[0]], dtype=torch.int32).cuda() for p in self.points]
        self.input_inds = [torch.arange(points.shape[0]).cuda()]
        self.cloud_inds = [torch.tensor([0]).cuda()]


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("results/features_lidar", exist_ok=True)

    # --- Load pretrained KPConv (S3DIS) ---
    cfg = Config()
    cfg.load("training_logs_S3DIS_KPFCNN")     # folder with parameters.txt
    lbl_values = np.sort([0])
    ign_lbls = np.array([], dtype=np.int32)
    net = KPFCNN(cfg, lbl_values, ign_lbls).cuda()

    # --- Load pretrained checkpoint (ignore classifier head) ---
    chkp_path = "training_logs_S3DIS_KPFCNN/checkpoints/current_chkp.tar"
    checkpoint = torch.load(chkp_path)
    filtered_dict = {
        k: v for k, v in checkpoint["model_state_dict"].items()
        if not k.startswith("head_softmax.")
    }
    net.load_state_dict(filtered_dict, strict=False)
    print(f"Loaded pretrained backbone weights (ignored classification head)")
    net.eval()

    # --- Dataset ---
    ds = LidarDataset(cfg, "data/lidar", train=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # --- Feature extraction ---
    with torch.no_grad():
        for i, sample in enumerate(dl):
            pts = sample["points"].cuda()
            feats_in = sample["features"].cuda()

            # Build full KPConv batch (neighbors, pools, etc.)
            batch = KPConvBatch(cfg, pts, feats_in)

            # Forward pass through KPConv backbone
            out = net(batch, cfg)

            # Extract feature tensor (robustly handle dict/tensor outputs)
            if isinstance(out, dict):
                if "features" in out:
                    feats = out["features"].detach().cpu().numpy()
                elif "x" in out:
                    feats = out["x"].detach().cpu().numpy()
                else:
                    first_tensor = [v for v in out.values() if isinstance(v, torch.Tensor)][0]
                    feats = first_tensor.detach().cpu().numpy()
            elif torch.is_tensor(out):
                feats = out.detach().cpu().numpy()
            else:
                raise RuntimeError(f"Unexpected model output type: {type(out)}")

            # Save features and corresponding points
            np.save(f"results/features_lidar/features_{i:03d}.npy", feats)
            np.save(f"results/features_lidar/points_{i:03d}.npy", sample["points"].numpy())
            print(f" Extracted features for cloud {i}  |  Shape: {feats.shape}")
