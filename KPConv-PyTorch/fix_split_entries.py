import os
import random

def rebuild_splits(root="data/ModelNet40", test_ratio=0.2, seed=42):
    random.seed(seed)
    train_entries, test_entries = [], []

    for cls in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue

        files = sorted([f for f in os.listdir(cls_path) if f.endswith(".txt")])
        if not files:
            continue

        n_total = len(files)
        n_test = int(test_ratio * n_total)
        test_files = set(files[-n_test:])

        for f in files:
            name = f"{cls}/{os.path.splitext(f)[0]}"
            if f in test_files:
                test_entries.append(name)
            else:
                train_entries.append(name)

    with open(os.path.join(root, "modelnet40_train.txt"), "w") as f:
        f.write("\n".join(train_entries))
    with open(os.path.join(root, "modelnet40_test.txt"), "w") as f:
        f.write("\n".join(test_entries))

    print(f"✅ Train split: {len(train_entries)}")
    print(f"✅ Test split:  {len(test_entries)}")
    print(f"📦 Total:       {len(train_entries) + len(test_entries)}")

if __name__ == "__main__":
    rebuild_splits()
