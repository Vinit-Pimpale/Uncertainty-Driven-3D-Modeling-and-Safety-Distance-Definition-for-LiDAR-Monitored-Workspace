import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import imageio

from utils.ply import read_ply


INPUT_FRAMES_DIR = "Evaluation/RGB_PerFrame"
OUTPUT_DIR = "Phase4_Geometry_PerFrame"
ANIMATION_DIR = "Phase4_Animation_Frames"
VIDEO_PATH = "Phase4_Geometry_Animation.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANIMATION_DIR, exist_ok=True)


def load_pred_frame(path):
    data = read_ply(path)
    pts = np.vstack((data["x"], data["y"], data["z"])).T

    labels = data["preds"]
    conf = data["conf"]

    return pts, labels, conf


def extract_robot_points(pts, labels, conf, threshold=0.5):
    mask = (labels == 3) & (conf >= threshold)
    return pts[mask]


def remove_outliers(robot_pts):
    if robot_pts.shape[0] == 0:
        return robot_pts

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(robot_pts)
    filtered, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)
    return np.asarray(filtered.points)


def compute_aabb(robot_pts):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(robot_pts)
    return cloud.get_axis_aligned_bounding_box()


def compute_obb(robot_pts):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(robot_pts)
    return cloud.get_oriented_bounding_box()


def plot_frame(frame_id, robot_pts, aabb, obb, save_path):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    if robot_pts.shape[0] > 0:
        ax.scatter(robot_pts[:, 0], robot_pts[:, 1], robot_pts[:, 2], s=2, c="green")

        aabb_pts = np.asarray(aabb.get_box_points())
        ax.scatter(aabb_pts[:, 0], aabb_pts[:, 1], aabb_pts[:, 2], c="red")

        obb_pts = np.asarray(obb.get_box_points())
        ax.scatter(obb_pts[:, 0], obb_pts[:, 1], obb_pts[:, 2], c="blue")

    ax.set_title(f"Frame {frame_id}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def generate_video():
    frames = sorted([f for f in os.listdir(ANIMATION_DIR) if f.endswith(".png")])

    images = []
    for f in frames:
        img = imageio.imread(os.path.join(ANIMATION_DIR, f))
        images.append(img)

    imageio.mimsave(VIDEO_PATH, images, fps=10)
    print(f"Saved animation video to {VIDEO_PATH}")


if __name__ == "__main__":

    pred_files = sorted([f for f in os.listdir(INPUT_FRAMES_DIR) if f.endswith("_pred.ply")])

    print(f"Found {len(pred_files)} predicted frames.")

    for idx, pred_file in enumerate(pred_files):
        frame_name = pred_file.replace("_pred.ply", "")
        pred_path = os.path.join(INPUT_FRAMES_DIR, pred_file)

        pts, labels, conf = load_pred_frame(pred_path)

        robot_pts = extract_robot_points(pts, labels, conf)
        robot_pts = remove_outliers(robot_pts)

        frame_dir = os.path.join(OUTPUT_DIR, frame_name)
        os.makedirs(frame_dir, exist_ok=True)

        np.save(os.path.join(frame_dir, "robot_points.npy"), robot_pts)

        if robot_pts.shape[0] > 0:
            aabb = compute_aabb(robot_pts)
            obb = compute_obb(robot_pts)

            np.save(os.path.join(frame_dir, "aabb_points.npy"), np.asarray(aabb.get_box_points()))
            np.save(os.path.join(frame_dir, "obb_points.npy"), np.asarray(obb.get_box_points()))

            frame_img_path = os.path.join(ANIMATION_DIR, f"{frame_name}.png")
            plot_frame(frame_name, robot_pts, aabb, obb, frame_img_path)

            print(f"Processed {frame_name}")

        else:
            print(f"No robot points in {frame_name}")

    print("Creating animation video...")
    generate_video()
    print("Finished.")
