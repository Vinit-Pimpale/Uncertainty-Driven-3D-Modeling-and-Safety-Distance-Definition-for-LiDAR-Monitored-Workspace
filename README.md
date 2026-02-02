# Uncertainty-Driven 3D Modeling and Safety Distance Definition for LiDAR-Monitored Workspace

This repository provides an uncertainty-aware pipeline for processing LiDAR point clouds, performing semantic segmentation, and defining probabilistic safety envelopes for human-robot collaboration.

## Pipeline Flow

The execution follows a logical progression from raw data processing to final safety zone visualization:

1.  **`run_full_pipeline.py`**: The master script that orchestrates the initial data preparation and inference. It sequentially calls:
    * `rename_files.py`: Standardizes input frame naming.
    * `preprocess_rgb.py`: Prepares PLY frames for the network.
    * `generate_gt_rgb.py`: Generates ground truth labels for performance tracking.
    * `run_inference_rgb.py`: Executes standard KPConv semantic predictions.
    * `compare_inference_vs_gt.py`: Calculates initial accuracy and mIoU metrics.
    * `run_inference_rgb_uncertainty_mc.py`: Runs MC Dropout and TTA to quantify uncertainty.
    * `highlight_robot.py`: Extracts robot-only points using uncertainty-aware filtering.
2.  **`shape_estimation.py`**: Calculates the robot's centroid and PCA-based geometric dimensions.
3.  **`uncertainty_shells.py`**: Builds initial uncertainty-inflated radial shells based on standard deviations.
4.  **`per_frame_visualization.py`**: *(Optional)* Generates visual overlays for individual frames.
5.  **`make_video_from_ply.py`**: *(Optional)* Compiles processed PLY frames into a video.
6.  **`gp_radial_shell.py`**: Fits a Gaussian Process to the radial data to create a smooth, continuous surface model.
7.  **`temporal_shell_tracking.py`**: Implements sliding-window smoothing to ensure stable safety shells over time.
8.  **`stage3_2_visualize_shells.py`**: *(Optional)* Visualizes the GP-tracked shells.
9.  **`stage3_2_make_video.py`**: *(Optional)* Video generation for the temporal tracking stage.
10. **`safety_distance_shell.py`**: Generates the final static probabilistic safety shells (e.g., 95% and 99% confidence).
11. **`stage3_3_visualize_shells.py`**: *(Optional)* Final visualization of the safety envelopes.
12. **`stage3_3_make_video.py`**: *(Optional)* Video generation for the final safety shells.
13. **`overlap_video.py`**: Produces the final output video showing the robot, environment, and safety zone overlaps.

---

## Detailed File Descriptions

### Core Pipeline & Modeling
* **`run_full_pipeline.py`**: Automates stages 1 through 7, including PLY checking, renaming, preprocessing, GT generation, and inference.
* **`shape_estimation.py`**: Uses PCA to determine global orientation and OBB (Oriented Bounding Box) dimensions. It filters "good" frames to establish a master shape prior.
* **`uncertainty_shells.py`**: Converts local robot points into spherical coordinates to calculate radial mean and $3\sigma$ shells.
* **`gp_radial_shell.py`**: Fits a `GaussianProcessRegressor` to model the radial extent $r(\theta, \phi)$ and predicts the posterior on a fine grid.
* **`temporal_shell_tracking.py`**: Uses a 15-frame sliding window to smooth the radial mean and cap standard deviation, preventing "shell explosion" during occlusions.
* **`safety_distance_shell.py`**: Applies the formula $r_{safety} = \mu + k\sigma + d_{margin}$ to define final safety boundaries.

### Evaluation & Metrics
* **`compare_inference_vs_gt.py`**: Generates a detailed `metrics.txt` containing Overall Accuracy, Macro F1-score, and per-class IoU/Precision/Recall.
* **`benchmark_fps.py`**: Benchmarks the KD-Tree search latency to evaluate real-time capability (targeting >10Hz).
* **`check_uncertainty_metrics.py`**: Statistical analysis of confidence and entropy levels across the dataset.
* **`generate_calibration_plot.py`**: Visualizes the reliability of predicted probabilities against actual accuracy.

### Visualizations
* **`plot_uncertainty_map.py`**: Generates heatmaps indicating regions of high epistemic or aleatoric uncertainty.
* **`plot_trajectory_error.py`**: Analyzes the stability of the robot's centroid tracking over time.
* **`vis_safety_zones.py`**: Visualizes nested safety zones (95% and 99%) in a 3D environment.
