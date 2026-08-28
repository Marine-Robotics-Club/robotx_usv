#!/usr/bin/env python3
# RobotX 2024 world champion launch code ROS2 Launch File
# Created by: Caleb Wilson && Xavier Vicent
# Email: calebwilson@fau.edu

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def make_compressor(name: str, in_topic: str, out_topic: str, jpeg_quality: int = 60) -> Node:
    return Node(
        package="vision_tracker",
        executable="compressed_image",
        name=name,
        output="screen",
        parameters=[
            {"in_topic": in_topic},
            {"out_topic": out_topic},
            {"jpeg_quality": jpeg_quality},
        ],
    )


def generate_launch_description():
    vision_cpp_dir = get_package_share_directory("vision_cpp")
    asv_launch_dir = get_package_share_directory("asv_launch")
    asv_audio_dir = get_package_share_directory("asv_audio")

    ld = LaunchDescription()

    # --------------------------
    # Core detectors
    # --------------------------
    ld.add_action(Node(
        package="vision_cpp",
        executable="buoy_detector",
        name="buoy_detector",
        output="screen",
        parameters=[os.path.join(vision_cpp_dir, "config", "buoy_detector.yaml")],
    ))
    ld.add_action(Node(
        package="vision_cpp",
        executable="ball_detector",
        name="ball_detector",
        output="screen",
        parameters=[os.path.join(vision_cpp_dir, "config", "ball_detector.yaml")],
    ))
    ld.add_action(Node(
        package="vision_cpp",
        executable="boat_detector",
        name="boat_detector",
        output="screen",
        parameters=[os.path.join(vision_cpp_dir, "config", "boat_detector.yaml")],
    ))
    ld.add_action(Node(
        package="vision_cpp",
        executable="dock_detector",
        name="dock_detector",
        output="screen",
        parameters=[os.path.join(vision_cpp_dir, "config", "dock_detector.yaml")],
    ))
    ld.add_action(Node(
        package="vision_cpp",
        executable="light_buoy_detector",
        name="light_buoy_detector",
        output="screen",
        parameters=[os.path.join(vision_cpp_dir, "config", "light_buoy_detector.yaml")],
    ))
    ld.add_action(Node(
        package="vision_cpp",
        executable="task_2",
        name="task_2",
        output="screen",
    ))
    ld.add_action(Node(
        package="vision_cpp",
        executable="task_1",
        name="task_1",
        output="screen",
    ))
    ld.add_action(Node(
        package="vision_cpp",
        executable="test_cpp",
        name="test_cpp",
        output="screen",
    ))
    ld.add_action(Node(
        package="vision_cpp",
        executable="Camera_Estimater_CENTER",
        name="Camera_Estimater_CENTER",
        output="screen",
    ))
    ld.add_action(Node(
        package="vision_cpp",
        executable="Camera_Estimater_LEFT",
        name="Camera_Estimater_LEFT",
        output="screen",
    ))
    ld.add_action(Node(
        package="vision_cpp",
        executable="Camera_Estimater_RIGHT",
        name="Camera_Estimater_RIGHT",
        output="screen",
    ))
    # --------------------------
    # Sensor fusion nodes
    # --------------------------
    ld.add_action(Node(
        package="vision_cpp",
        executable="sensor_fusion",
        name="sensor_fusion",
        output="screen",
    ))

    ld.add_action(Node(
        package="vision_cpp",
        executable="sensor_fusion_left",
        name="sensor_fusion_left",
        output="screen",
    ))
    ld.add_action(Node(
        package="vision_cpp",
        executable="sensor_fusion_right",
        name="sensor_fusion_right",
        output="screen",
    ))
    # --------------------------
    # Include YOLO launch
    # --------------------------
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(asv_launch_dir, "launch", "yolov26.launch.py")
        )
    ))

    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(asv_launch_dir, "launch", "audio.launch.py")
        )
    ))
    

    # --------------------------
    # PointCloud Compression (Cloudini)
    # --------------------------
    cloudini_topics = [
        ("cloudini_compressor_velodyne",  "/velodyne_points",         "/velodyne_points/compressed"),
        ("cloudini_compressor_ball",      "/clustered_points/ball",   "/clustered_points/ball/compressed"),
        ("cloudini_compressor_buoy",      "/clustered_points/buoy",   "/clustered_points/buoy/compressed"),
        ("cloudini_compressor_boat",      "/clustered_points/boat",   "/clustered_points/boat/compressed"),
        ("cloudini_compressor_dock",      "/clustered_points/dock",   "/clustered_points/dock/compressed"),
        ("cloudini_compressor_lightbuoy", "/clustered_points/lightbuoy", "/clustered_points/lightbuoy/compressed"),
    ]

    for name, topic_in, topic_out in cloudini_topics:
        ld.add_action(Node(
            package="cloudini_ros",
            executable="cloudini_topic_converter",
            name=name,
            output="screen",
            parameters=[{
                "compressing": True,
                "topic_input": topic_in,
                "topic_output": topic_out,
                "resolution": 0.05,
            }],
        ))

    # --------------------------
    # Image Compression (Network Optimization)
    # --------------------------
    jpeg_q = 10

    ld.add_action(make_compressor(
        "compress_yolo_center",
        "/yolov26/annotated_image/center",
        "/yolov26/annotated_image/center/compressed",
        jpeg_q
    ))
    ld.add_action(make_compressor(
        "compress_yolo_left",
        "/yolov26/annotated_image/left",
        "/yolov26/annotated_image/left/compressed",
        jpeg_q
    ))
    ld.add_action(make_compressor(
        "compress_yolo_right",
        "/yolov26/annotated_image/right",
        "/yolov26/annotated_image/right/compressed",
        jpeg_q
    ))

    return ld