import os, numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from plyfile import PlyElement, PlyData

input_dir  = 'results/features_lidar'
output_dir = 'results/segmented_lidar'
os.makedirs(output_dir, exist_ok=True)

K = 5   # number of clusters – adjust per scene

for file in sorted(f for f in os.listdir(input_dir) if f.startswith('features_')):
    idx = file.split('_')[-1].split('.')[0]
    feats  = np.load(os.path.join(input_dir, f'features_{idx}.npy'))
    points = np.load(os.path.join(input_dir, f'points_{idx}.npy'))

    # reduce dimensionality for cleaner clustering
    feats_64 = PCA(n_components=64).fit_transform(feats)

    kmeans = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels = kmeans.fit_predict(feats_64)

    # --- attach cluster label & original intensity for visualization ---
    intensity = np.ones(points.shape[0], dtype=np.float32)
    verts = np.empty(points.shape[0],
                     dtype=[('x','f4'), ('y','f4'), ('z','f4'),
                            ('intensity','f4'), ('preds','i4')])
    verts['x'], verts['y'], verts['z'] = points[:,0], points[:,1], points[:,2]
    verts['intensity'] = intensity
    verts['preds'] = labels

    PlyData([PlyElement.describe(verts, 'vertex')]).write(
        os.path.join(output_dir, f'scan_{idx}_seg.ply')
    )

    print(f"Segmented scan_{idx}_seg.ply with {K} clusters")
