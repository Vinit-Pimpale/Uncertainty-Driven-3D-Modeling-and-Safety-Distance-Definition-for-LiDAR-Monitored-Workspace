#
#  COVERED Dataset integration for KPConv (full 7-layer pyramid with neighbor caps)
#  - Computes all KPConv layers (no dummy padding)
#  - Caps neighbors to 32 for memory safety
#  - Compatible with KPFCNN pretrained on S3DIS (5-D input)
#

import os
import numpy as np
from os.path import join
from torch.utils.data import Sampler
import torch
import gc
from sklearn.neighbors import KDTree
# KPConv utilities
from utils.config import Config
from datasets.common import PointCloudDataset, batch_neighbors, batch_grid_subsampling
from utils.ply import read_ply


# ==========================================================================================================
# Dataset Definition
# ==========================================================================================================
class COVEREDDataset(PointCloudDataset):
    """COVERED dataset loader compatible with KPConv cloud segmentation."""

    def __init__(self, config: Config, set='training', use_potentials=True):
        super().__init__(config)
        self.name = 'COVERED'
        self.path = join(config.data_path, 'COVERED')
        self.set = set
        self.use_potentials = use_potentials

        # ---- S3DIS-style split metadata expected by trainer ----
        self.all_splits = ['training', 'validation']
        self.validation_split = 'validation'
        self.train_split = 'training'
        self.test_split = 'validation'   # reuse validation as placeholder

        # Label mapping (update if needed)
        self.label_to_names = {
            0: 'Unlabeled',
            1: 'Floor',
            2: 'Wall',
            3: 'Robot',
            4: 'Human',
            5: 'AGV'
        }

        # Collect all .ply files
        all_files = []
        for root, _, files in os.walk(self.path):
            for f in files:
                if f.endswith('.ply'):
                    all_files.append(join(root, f))
        all_files = np.sort(all_files)

        # Train/Validation split (80/20)
        split_idx = int(0.8 * len(all_files))
        self.files = all_files[:split_idx] if set == 'training' else all_files[split_idx:]
        
        self.all_splits = ['training','validation']
        self.validation_split = 'validation'
        self.train_split = 'training'
        self.test_split = 'validation'

        self.label_values = np.sort([k for k in self.label_to_names.keys()]) 
        self.num_classes = len(self.label_values)
        self.ignored_labels = np.array([], dtype=np.int32)
        
        # --- CORRECTED INITIALIZATION ---
        # Initialize all lists needed by the trainer
        self.input_labels = []
        self.validation_labels = []
        self.test_proj = []
        self.pot_trees = []
        self.val_proportions = np.zeros(self.num_classes, dtype=np.int64)

        # Build KD-Trees and collect all metadata in ONE loop
        for f in self.files:
            data = read_ply(f)

            # Points
            pts = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)

            # Labels
            if 'class' in data.dtype.names:
                lbls = data['class'].astype(np.int32)
            else:
                lbls = np.zeros(pts.shape[0], np.int32)

            # Add labels to list (for validation)
            self.validation_labels.append(lbls)

            # Add identity projection map (for saving predictions)
            self.test_proj.append(np.arange(pts.shape[0], dtype=np.int32))

            # KDTree from points
            tree = KDTree(pts, leaf_size=10)
            self.pot_trees.append(tree)

            # Update class frequencies
            for c in np.unique(lbls):
                if c < self.num_classes:
                    self.val_proportions[c] += np.sum(lbls == c)

        self.val_proportions = self.val_proportions.astype(np.int64)
        self.use_potentials = True
        self.potentials = [torch.zeros(len(read_ply(f)['x']), dtype=torch.float32) for f in self.files]
        self.min_potentials = torch.zeros(len(self.files), dtype=torch.float32)
        
        # Copy validation labels to input_labels (required by trainer)
        self.input_labels = self.validation_labels
        
        # Initialize potentials (uses self.pot_trees)
        self.init_labels()


    def init_labels(self):
        """Initialize potential weights for balanced sampling."""
        if self.use_potentials:
            # self.potentials = np.random.rand(len(self.files)) * 1e-3  # <--- OLD, INCORRECT

            # NEW: Create potentials per-point in each tree, matching S3DIS structure
            self.potentials = []
            for tree in self.pot_trees:
                # Get the number of points in the corresponding KDTree
                num_points_in_tree = tree.data.shape[0]
                
                # Create a 1D PyTorch tensor of potentials for these points
                self.potentials.append(
                    torch.from_numpy(
                        np.random.rand(num_points_in_tree) * 1e-3
                    )
                )

    def get_cloud(self, cloud_idx):
        """Load a single .ply file (points + colors + labels)."""
        cloud_path = self.files[cloud_idx]
        data = read_ply(cloud_path)

        # Coordinates
        pts = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)

        # Labels
        lbl = data['class'].astype(np.int32) if 'class' in data.dtype.names else np.zeros(pts.shape[0], np.int32)

        # Colors (normalize to 0–1)
        if all(c in data.dtype.names for c in ('red', 'green', 'blue')):
            col = np.vstack((data['red'], data['green'], data['blue'])).T / 255.0
        else:
            col = np.zeros_like(pts)

        # Pad RGB with 2 dummy zeros → total 5D input (match S3DIS pretrained)
        dummy = np.zeros((pts.shape[0], 2), dtype=np.float32)
        feats = np.concatenate([col, dummy], axis=1)

        return pts, feats, lbl

    def __getitem__(self, idx):
        return self.get_cloud(idx) + (idx,)

    def __len__(self):
        return len(self.files)
    
    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """
        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T


# ==========================================================================================================
# Sampler
# ==========================================================================================================
class COVEREDSampler(Sampler):
    """Sequential sampler (no shuffling, compatible with KPConv)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


# ==========================================================================================================
# KPConv Batch (FULL 7-layer build, neighbor-capped)
# ==========================================================================================================
class COVEREDBatch:
    """Builds KPConv inputs (neighbors, pools, upsample indices) for all layers."""

    def __init__(self, batch_data, config: Config):
        pts, feats, lbls, cloud_idx = batch_data[0]

        # Containers (following S3DIS convention)
        self.points = []     # list of [N_l, 3]
        self.neighbors = []  # list of [N_l, K]
        self.pools = []      # list of [N_{l+1}, K]
        self.upsamples = []  # list of [N_l, K]
        self.lengths = []    # list of [1]

        # Initial cloud
        points = pts
        lengths = np.array([len(points)], dtype=np.int32)

        # How many layers to build (KPFCNN expects 7)
        requested_layers = max(getattr(config, "num_layers", 7), 7)

        # Base radius and progressive growth
        r = config.first_subsampling_dl * config.conv_radius
        KMAX = 60  # cap neighbors per query everywhere

        # Build all layers
        for _ in range(requested_layers):
            # Same-layer neighbors (query=support=current)
            neighb = batch_neighbors(points, points, lengths, lengths, radius=r)
            if neighb.shape[1] > KMAX:
                neighb = neighb[:, :KMAX]
            self.neighbors.append(torch.from_numpy(neighb).long())

            # Subsample to get next layer points
            dl = 2 * r / config.conv_radius
            pool_p, pool_b = batch_grid_subsampling(points, lengths, sampleDl=dl)

            # Pool indices: from next layer (queries) to current layer (support)
            pool_i = batch_neighbors(pool_p, points, pool_b, lengths, radius=r)
            if pool_i.shape[1] > KMAX:
                pool_i = pool_i[:, :KMAX]
            self.pools.append(torch.from_numpy(pool_i).long())

            # Upsample indices: from current to next layer
            up_i = batch_neighbors(points, pool_p, lengths, pool_b, radius=2 * r)
            if up_i.shape[1] > KMAX:
                up_i = up_i[:, :KMAX]
            self.upsamples.append(torch.from_numpy(up_i).long())

            # Save current points/lengths
            self.points.append(torch.from_numpy(points).float())
            self.lengths.append(torch.from_numpy(lengths).int())

            # Move to next layer
            points = pool_p
            lengths = pool_b
            r *= 2

        # Append final top-level points
        self.points.append(torch.from_numpy(points).float())
        self.lengths.append(torch.from_numpy(lengths).int())

        # Core tensors for the head/loss
        self.features = torch.from_numpy(feats).float()
        self.labels = torch.from_numpy(lbls).long()
        N = self.points[0].shape[0] if len(self.points) > 0 else 0
        self.input_inds = torch.arange(N, dtype=torch.long)
        self.cloud_inds = torch.tensor([cloud_idx], dtype=torch.long)   

    def to(self, device):
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.points = [p.to(device) for p in self.points]
        self.neighbors = [n.to(device) for n in self.neighbors]
        self.pools = [p.to(device) for p in self.pools]
        self.upsamples = [u.to(device) for u in self.upsamples]
        self.lengths = [l.to(device) for l in self.lengths]
        return self


# ==========================================================================================================
# Collate Function
# ==========================================================================================================
def COVEREDCollate(batch_data):
    """Memory-safe collate function for COVERED dataset."""
    import train_COVERED
    gc.collect()
    torch.cuda.empty_cache()

    config = getattr(train_COVERED, 'config', Config())
    try:
        return COVEREDBatch(batch_data, config)
    except Exception as e:
        print(" COVEREDCollate failed:", e)
        gc.collect()
        torch.cuda.empty_cache()
        raise
