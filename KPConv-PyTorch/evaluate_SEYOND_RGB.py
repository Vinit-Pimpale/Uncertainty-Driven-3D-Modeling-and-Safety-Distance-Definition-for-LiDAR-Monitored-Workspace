import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from utils.ply import read_ply
from utils.metrics import IoU_from_confusions
from sklearn.neighbors import KDTree


# -------------------------- PATHS --------------------------
GROUND_TRUTH_DIR = "data_local/Ground_Truth"
PRED_PATH = "data_local/results_RGB/predictions_RGB.ply"
CONF_PATH = "data_local/results_RGB/confidence_RGB.ply"
SAVE_DIR = "Evaluation/RGB"
os.makedirs(SAVE_DIR, exist_ok=True)
# -----------------------------------------------------------

CLASS_NAMES = ["Floor", "Wall", "Column", "Robo Dog", "Screen+Stand"]

def load_ply_points_labels(path):
    data = read_ply(path)
    pts = np.vstack((data["x"], data["y"], data["z"])).T
    if "class" in data.dtype.names:
        labels = data["class"]
    elif "label" in data.dtype.names:
        labels = data["label"]
    elif "preds" in data.dtype.names:
        labels = data["preds"]
    else:
        labels = None
    return pts, labels


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    print("\nStarting Evaluation: RGB Scenario (single merged prediction)\n")

    # ---------- Load predictions ----------
    pred_pts, pred_labels = load_ply_points_labels(PRED_PATH)
    if pred_labels is None:
        raise RuntimeError("Predictions file missing 'preds' or 'label' field.")
    print(f"Loaded predictions: {len(pred_pts)} points")

    # ---------- Load and merge all ground truth frames ----------
    gt_pts_list, gt_labels_list = [], []
    gt_files = sorted([f for f in os.listdir(GROUND_TRUTH_DIR) if f.endswith(".ply")])
    for f in gt_files:
        pts, lbls = load_ply_points_labels(os.path.join(GROUND_TRUTH_DIR, f))
        if lbls is None:
            continue
        gt_pts_list.append(pts)
        gt_labels_list.append(lbls)

    gt_pts = np.vstack(gt_pts_list)
    gt_labels = np.concatenate(gt_labels_list)
    print(f"Merged ground truth: {len(gt_pts)} points from {len(gt_files)} frames")

    # ---------- Match points globally ----------
    print("\nMatching prediction points to ground truth ...")
    tree = KDTree(gt_pts)
    dist, idx = tree.query(pred_pts, k=1)
    idx = idx.squeeze()
    mask = dist.squeeze() < 0.02  # 2 cm threshold
    pred_matched = pred_labels[mask]
    gt_matched = gt_labels[idx[mask]]
    print(f"Matched {len(pred_matched)} points for evaluation")

    # ---------- Compute Metrics ----------
    cm = confusion_matrix(gt_matched, pred_matched, labels=np.arange(len(CLASS_NAMES)))
    overall_acc = accuracy_score(gt_matched, pred_matched)
    per_class_iou = IoU_from_confusions(cm)
    precision = precision_score(gt_matched, pred_matched, average=None, zero_division=0)
    recall = recall_score(gt_matched, pred_matched, average=None, zero_division=0)
    f1 = f1_score(gt_matched, pred_matched, average=None, zero_division=0)
    macro_f1 = f1_score(gt_matched, pred_matched, average="macro")

    # ---------- Confidence & Entropy ----------
    if os.path.exists(CONF_PATH):
        conf_data = read_ply(CONF_PATH)
        conf_values = conf_data["conf"]
        conf_stats = {
            "min": float(np.min(conf_values)),
            "max": float(np.max(conf_values)),
            "mean": float(np.mean(conf_values)),
            "std": float(np.std(conf_values)),
        }
        entropy = -conf_values * np.log(conf_values + 1e-9)
        entropy_stats = {
            "min": float(np.min(entropy)),
            "max": float(np.max(entropy)),
            "mean": float(np.mean(entropy)),
            "std": float(np.std(entropy)),
        }
    else:
        conf_stats = entropy_stats = None

    # ---------- Save Metrics ----------
    metrics_txt = os.path.join(SAVE_DIR, "metrics_RGB.txt")
    with open(metrics_txt, "w") as f:
        f.write("=== RGB Inference Evaluation (Single Prediction) ===\n\n")
        f.write(f"Overall Accuracy: {overall_acc*100:.2f}%\n")
        f.write(f"Macro F1-score:  {macro_f1*100:.2f}%\n\n")

        f.write("Per-class Metrics:\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(
                f"  {name:<15s} | IoU: {per_class_iou[i]*100:5.2f}% "
                f"| Precision: {precision[i]*100:5.2f}% "
                f"| Recall: {recall[i]*100:5.2f}% "
                f"| F1: {f1[i]*100:5.2f}%\n"
            )

        if conf_stats:
            f.write("\nConfidence Statistics:\n")
            for k, v in conf_stats.items():
                f.write(f"  {k:>5s}: {v:.4f}\n")

        if entropy_stats:
            f.write("\nEntropy Statistics:\n")
            for k, v in entropy_stats.items():
                f.write(f"  {k:>5s}: {v:.4f}\n")

    print(f"\nMetrics saved to {metrics_txt}")

    # ---------- Save Confusion Matrix ----------
    cm_plot = os.path.join(SAVE_DIR, "confusion_matrix_RGB.png")
    plot_confusion_matrix(cm, CLASS_NAMES, cm_plot)
    print(f"Confusion matrix saved to {cm_plot}")

    print("\nEvaluation completed successfully.\n")
