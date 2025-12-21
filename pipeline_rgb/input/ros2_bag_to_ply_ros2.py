#!/usr/bin/env python3

"""
ROS2-Native Bag to PLY Conversion Script
----------------------------------------

This script converts ROS2 bags (including MCAP format) into .ply frames.

REQUIRES:
    - ROS2 installed (source /opt/ros/<distro>/setup.bash)
    - rosbag2_py
    - sensor_msgs_py

USAGE:
    python ros2_bag_to_ply_ros2.py --bag /path/to/bag --topic /roi --output ply_frames/
"""

import os
import argparse
import numpy as np
import open3d as o3d

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def convert_ros2_bag_to_ply(bag_dir, topic, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_dir,
        storage_id="mcap"
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Filter connections by topic
    connections = reader.get_all_topics_and_types()
    topic_found = any(c.name == topic for c in connections)
    if not topic_found:
        raise RuntimeError(f"Topic '{topic}' not found in bag.")

    frame = 0

    while reader.has_next():
        (topic_name, data, t) = reader.read_next()

        if topic_name != topic:
            continue

        msg = deserialize_message(data, PointCloud2)

        # Convert ROS PointCloud2 → XYZ
        pts = []
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])

        pts = np.asarray(pts, dtype=np.float32)
        print(f"Frame {frame}: {pts.shape[0]} points")

        if pts.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            out = os.path.join(output_dir, f"frame_{frame:04d}.ply")
            o3d.io.write_point_cloud(out, pcd, write_ascii=False)

        frame += 1

    print("\n✓ ROS2 Native Bag → PLY conversion completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Path to the folder containing metadata.yaml and _0.mcap")
    parser.add_argument("--topic", default="/roi")
    parser.add_argument("--output", default="ply_frames")
    args = parser.parse_args()

    convert_ros2_bag_to_ply(args.bag, args.topic, args.output)


if __name__ == "__main__":
    main()