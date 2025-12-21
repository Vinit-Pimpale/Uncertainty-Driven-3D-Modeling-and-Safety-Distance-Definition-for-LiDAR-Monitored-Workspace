import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

from models.architectures import KPFCNN
from train_COVERED import COVEREDConfig
from datasets.COVERED import COVEREDDataset, COVEREDSampler, COVEREDCollate
from utils.ply import write_ply


# ---------- Output paths ----------
OUT_ROOT = os.path.join('test', 'training_logs_COVERED_KPFCNN')
PRED_DIR = os.path.join(OUT_ROOT, 'predictions')
PROB_DIR = os.path.join(OUT_ROOT, 'probs')
POTS_DIR = os.path.join(OUT_ROOT, 'potentials')

BASE_NAME = 'COVERED_val'

# ---------- Class color map ----------
# 0:Unlabeled, 1:Floor, 2:Wall, 3:Robot, 4:Human, 5:AGV
CLASS_COLORS = np.array([
    [  0,   0,   0],    # Unlabeled
    [200, 200, 200],    # Floor
    [255,   0,   0],    # Wall
    [  0,   0, 255],    # Robot (arm)
    [  0, 255,   0],    # Human
    [255, 255,   0],    # AGV
], dtype=np.uint8)


def estimate_normals(points, k=20):
    """Estimate normals for a point cloud using PCA."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    normals = np.zeros_like(points)

    for i in range(points.shape[0]):
        neighbors = points[indices[i, 1:]]  # skip itself
        cov = np.cov(neighbors - neighbors.mean(axis=0), rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue
    # Ensure normals face upward (optional, for visual consistency)
    normals = np.where(normals[:, 2:3] < 0, -normals, normals)
    return normals.astype(np.float32)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ----- checkpoint -----
    chosen_log = 'training_logs_COVERED_KPFCNN/Log_2025-10-27_22-28-06'
    chkp_path = os.path.join(chosen_log, 'checkpoints', 'best_chkp.tar')

    # ----- ensure output dirs -----
    for d in [OUT_ROOT, PRED_DIR, PROB_DIR, POTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ----- load config & dataset -----
    print('\nLoading COVERED configuration ...')
    config = COVEREDConfig()
    config.load(chosen_log)
    config.saving = False
    config.input_threads = 0

    print('\nData Preparation\n****************')
    ds = COVEREDDataset(config, set='validation', use_potentials=True)
    sampler = COVEREDSampler(ds)
    loader = DataLoader(ds, batch_size=1, sampler=sampler,
                        collate_fn=COVEREDCollate, num_workers=0, pin_memory=False)

    # ----- model -----
    print('\nModel Preparation\n*****************')
    net = KPFCNN(config, ds.label_values, ds.ignored_labels)
    ckpt = torch.load(chkp_path, map_location='cpu')
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval().cuda()
    softmax = torch.nn.Softmax(dim=1)

    # ----- run a single pass -----
    print('\nStart COVERED Test (single pass)\n********************************')
    all_pts, all_lbl, all_pred, all_conf = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            batch.to('cuda')
            out = net(batch, config)
            prob = softmax(out).cpu().numpy()
            pred = np.argmax(prob, axis=1).astype(np.int32)
            conf = prob.max(axis=1).astype(np.float32)
            pts = batch.points[0].cpu().numpy()
            lbl = batch.labels.cpu().numpy().astype(np.int32)

            all_pts.append(pts)
            all_lbl.append(lbl)
            all_pred.append(pred)
            all_conf.append(conf)

    # ----- merge all -----
    all_pts = np.vstack(all_pts)
    all_lbl = np.concatenate(all_lbl)
    all_pred = np.concatenate(all_pred)
    all_conf = np.concatenate(all_conf)

    # ----- compute normals -----
    print("\nEstimating normals ... (this may take ~30–60s for large clouds)")
    #normals = estimate_normals(all_pts, k=20)
    normals = estimate_normals(all_pts[::5], k=10)  # compute on 1/5 points
    # interpolate normals back to all points
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(all_pts[::5])
    _, idx = nbrs.kneighbors(all_pts)
    normals = normals[idx[:, 0]]

    # ----- predictions -----
    pred_ply = os.path.join(PRED_DIR, f'{BASE_NAME}.ply')
    write_ply(pred_ply,
              [all_pts, normals, all_pred.astype(np.int32), all_lbl.astype(np.int32)],
              ['x', 'y', 'z', 'nx', 'ny', 'nz', 'preds', 'class'])

    # ----- colored predictions -----
    colors = CLASS_COLORS[np.clip(all_pred, 0, len(CLASS_COLORS)-1)]
    pred_col_ply = os.path.join(PRED_DIR, f'{BASE_NAME}_colored.ply')
    write_ply(pred_col_ply,
              [all_pts,
               normals,
               colors[:, 0].astype(np.uint8),
               colors[:, 1].astype(np.uint8),
               colors[:, 2].astype(np.uint8),
               all_pred.astype(np.int32),
               all_lbl.astype(np.int32)],
              ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue', 'preds', 'class'])

    # ----- probs -----
    prob_ply = os.path.join(PROB_DIR, f'{BASE_NAME}.ply')
    write_ply(prob_ply,
              [all_pts, all_conf.astype(np.float32)],
              ['x', 'y', 'z', 'conf'])

    # ----- potentials -----
    pots = np.zeros((all_pts.shape[0],), dtype=np.float32)
    pots_ply = os.path.join(POTS_DIR, f'{BASE_NAME}.ply')
    write_ply(pots_ply,
              [all_pts, pots],
              ['x', 'y', 'z', 'pots'])

    print('\nCOVERED test complete with normals.')
    print(f'  predictions : {pred_ply}')
    print(f'  colored preds : {pred_col_ply}')
    print(f'  probs : {prob_ply}')
    print(f'  potentials : {pots_ply}')
