# fix_modelnet40_structure_recursive.py
import os, shutil

root = "data/ModelNet40"
moved = 0

for cls in os.listdir(root):
    cls_path = os.path.join(root, cls)
    if not os.path.isdir(cls_path):
        continue

    for dirpath, _, files in os.walk(cls_path):
        if dirpath == cls_path:
            continue  # skip top level
        # only process nested folders
        for f in files:
            if not f.endswith(".txt"):
                continue
            src = os.path.join(dirpath, f)
            dst = os.path.join(cls_path, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1
        # remove empty subfolders afterwards
        for subdir, _, _ in os.walk(dirpath, topdown=False):
            try:
                os.rmdir(subdir)
            except OSError:
                pass

print(f"✅ Moved {moved} .txt files to correct class folders.")
print("✅ Removed redundant nested directories.")
