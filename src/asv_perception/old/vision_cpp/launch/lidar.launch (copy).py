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

def generate_launch_description():
    # Get package share directories for additional launch files
    vision_cpp_dir = get_package_share_directory('vision_cpp')
    velodyne_dir = get_package_share_directory('velodyne')
    zed2_dir = get_package_share_directory('zed_wrapper')
    yolov9_dir = get_package_share_directory('yolov9')  # Ensure this package is installed and named correctly

    return LaunchDescription([
        # Existing nodes from the original launch file
        Node(
            package='vision_cpp',
            executable='buoy_detector',
            name='buoy_detector',
            output='screen',
            parameters=[os.path.join(vision_cpp_dir, 'config', 'buoy_detector.yaml')]
        ),
        Node(
            package='vision_cpp',
            executable='light_buoy_detector',
            name='light_buoy_detector',
            output='screen',
            parameters=[os.path.join(vision_cpp_dir, 'config', 'light_buoy_detector.yaml')]
        ),
        Node(
            package='vision_cpp',
            executable='ball_detector',
            name='ball_detector',
            output='screen',
            parameters=[os.path.join(vision_cpp_dir, 'config', 'ball_detector.yaml')]
        ),
        Node(
            package='vision_cpp',
            executable='dock_detector',
            name='dock_detector',
            output='screen',
            parameters=[os.path.join(vision_cpp_dir, 'config', 'dock_detector.yaml')]
        ),
        Node(
            package='vision_cpp',
            executable='sensor_fusion',
            name='sensor_fusion',
            output='screen',
        ),
            Node(
            package='vision_cpp',
            executable='dock_vertices',
            name='sensorvertices',
            output='screen',
        ),

        # Additional launch files for Velodyne LiDAR, ZED2i, and YOLOv9
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(velodyne_dir, 'launch', 'velodyne-all-nodes-VLP16-launch.py')])
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(zed2_dir, 'launch', 'zed_camera.launch.py')]),
            launch_arguments={'camera_model': 'zed2i', 'camera_name': 'zed_hl1', 'node_name': 'z'}.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(yolov9_dir, 'yolov9.launch.py')])
        ),
    ])

