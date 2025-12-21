import os
import numpy as np
from plyfile import PlyData, PlyElement

# Path to your prediction file
pred_path = "test/training_logs_S3DIS_KPFCNN/predictions"
in_file = "Area_5.ply"
in_path = os.path.join(pred_path, in_file)

print(f"Processing {in_path}")

# Read the existing PLY file
plydata = PlyData.read(in_path)
v = plydata['vertex']

# Extract coordinates
x = v['x']
y = v['y']
z = v['z']

# Check which field to use for labeling
if 'preds' in v.data.dtype.names:
    labels = v['preds']
elif 'class' in v.data.dtype.names:
    labels = v['class']
else:
    raise KeyError("No 'preds' or 'class' field found in the PLY file.")

labels = np.asarray(labels, dtype=np.int32)

# Define a fixed color palette (RGB 0–255)
color_palette = np.array([
    [0, 0, 255],      # class 0 – blue
    [0, 255, 0],      # class 1 – green
    [255, 0, 0],      # class 2 – red
    [255, 255, 0],    # class 3 – yellow
    [255, 0, 255],    # class 4 – magenta
    [0, 255, 255],    # class 5 – cyan
    [255, 128, 0],    # class 6 – orange
    [128, 0, 255],    # class 7 – violet
    [128, 128, 128],  # class 8 – gray
    [0, 0, 0],        # class 9 – black (fallback)
], dtype=np.uint8)

# Assign colors based on labels
n_classes = len(color_palette)
colors = color_palette[labels % n_classes]

# Combine all vertex data
vertex_all = np.array(list(zip(
    x, y, z, colors[:, 0], colors[:, 1], colors[:, 2], labels
)), dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ('preds', 'i4')
])

# Output path
out_path = os.path.join(pred_path, in_file.replace(".ply", "_colored.ply"))

# Write as binary PLY (faster and smaller)
PlyData([PlyElement.describe(vertex_all, 'vertex')], text=False).write(out_path)

print(f"✅ Saved colored file: {out_path}")
