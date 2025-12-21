import os
import numpy as np
from utils.ply import read_ply, write_ply

IN_DIR  = 'ply_frames'
OUT_DIR = os.path.join('data_unlabeled', 'Seyond_LiDAR')

os.makedirs(OUT_DIR, exist_ok=True)

for name in sorted(os.listdir(IN_DIR)):
    if not name.endswith('.ply'):
        continue

    in_path  = os.path.join(IN_DIR, name)
    out_path = os.path.join(OUT_DIR, name)

    data = read_ply(in_path)
    x, y, z = data['x'], data['y'], data['z']
    n = x.shape[0]

    # --- always create RGB, even if missing ---
    if all(c in data.dtype.names for c in ['red', 'green', 'blue']):
        r, g, b = data['red'], data['green'], data['blue']
    else:
        r = np.zeros(n, dtype=np.uint8)
        g = np.zeros(n, dtype=np.uint8)
        b = np.zeros(n, dtype=np.uint8)

    labels = np.zeros(n, dtype=np.int32)

    write_ply(out_path,
              [x, y, z, r, g, b, labels],
              ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    print("Wrote", out_path)
