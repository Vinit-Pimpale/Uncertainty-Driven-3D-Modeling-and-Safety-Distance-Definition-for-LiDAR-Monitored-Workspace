# datasets/Seyond_LiDAR.py
import os, numpy as np, torch, gc
from torch.utils.data import Sampler
from sklearn.neighbors import KDTree
from utils.ply import read_ply
from utils.config import Config
from datasets.common import PointCloudDataset, batch_neighbors, batch_grid_subsampling

class SeyondLiDARDataset(PointCloudDataset):
    def __init__(self, config: Config, set='training', use_potentials=True):
        super().__init__(config)
        self.name = 'Seyond_LiDAR'
        self.path = os.path.join(config.data_path, 'Seyond_LiDAR')
        self.set = set
        self.use_potentials = use_potentials

        self.label_to_names = {
            0: 'Floor',
            1: 'Wall',
            2: 'Column',
            3: 'Robo Dog',
            4: 'Screen+Stand',
        }
        
        self.label_values = np.sort(list(self.label_to_names.keys()))
        self.num_classes = len(self.label_values)  # 5
        self.ignored_labels = np.array([], dtype=np.int32)

        # Collect all .ply files
        all_files = sorted([os.path.join(self.path, f)
                            for f in os.listdir(self.path) if f.endswith('.ply')])
        split_idx = int(0.8 * len(all_files))
        self.files = all_files[:split_idx] if set == 'training' else all_files[split_idx:]

        self.validation_split = 'validation'
        self.all_splits = ['training', 'validation']

                # ----------------------------------------------------------------------
        # Compatibility fields for KPConv trainer validation
        # ----------------------------------------------------------------------
        self.input_labels = []           # store labels for each file
        self.validation_labels = []      # store validation labels (same as input_labels here)
        self.label_values = np.sort(list(self.label_to_names.keys()))
        self.num_classes = len(self.label_values)
        self.ignored_labels = np.array([], dtype=np.int32)

        for f in self.files:
            data = read_ply(f)
            pts = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
            lbls = data['label'].astype(np.int32)
            self.input_labels.append(lbls)
            self.validation_labels.append(lbls)

        self.use_potentials = True
        self.potentials = [torch.zeros(len(lbl), dtype=torch.float32) for lbl in self.input_labels]

        self.label_values = np.sort(list(self.label_to_names.keys()))
        self.num_classes = len(self.label_values)
        self.ignored_labels = np.array([], dtype=np.int32)

        # Metadata
        self.validation_labels, self.pot_trees, self.val_proportions = [], [], np.zeros(self.num_classes)
        for f in self.files:
            data = read_ply(f)
            pts = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
            lbls = data['label'].astype(np.int32)
            self.validation_labels.append(lbls)
            self.pot_trees.append(KDTree(pts, leaf_size=10))
            for c in np.unique(lbls):
                if c < self.num_classes: self.val_proportions[c] += np.sum(lbls == c)
        self.init_labels()

    def get_cloud(self, idx):
        data = read_ply(self.files[idx])
        pts = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
        lbl = data['label'].astype(np.int32)
        col = np.vstack((data['red'], data['green'], data['blue'])).T / 255.0
        dummy = np.zeros((pts.shape[0], 2), dtype=np.float32)
        #feats = np.concatenate([col, dummy], axis=1)
        feats = np.zeros((pts.shape[0], 5), dtype=np.float32)
        return pts, feats, lbl

    def __getitem__(self, idx): return self.get_cloud(idx) + (idx,)
    def __len__(self): return len(self.files)
