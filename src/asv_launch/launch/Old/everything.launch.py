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
    
    vision_cpp_dir = get_package_share_directory('asv_lidar')
    velodyne_dir = get_package_share_directory('velodyne')
    rtsp_gscam_dir = get_package_share_directory('rtsp_gscam_driver')

    
    return LaunchDescription([
        # Existing nodes from the original launch file
        Node(
            package='asv_lidar',
            executable='wamv_detector',
            name='wamv_detector',
            output='screen',
            parameters=[os.path.join(vision_cpp_dir, 'config', 'wamv_detector.yaml')]
        ),
        Node(
            package='asv_lidar',
            executable='sensor_fusion',
            name='sensor_fusion',
            output='screen',
        ),
        
        # Additional launch files for Velodyne LiDAR, ZED2i, and YOLOv9
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(velodyne_dir, 'launch', 'velodyne-all-nodes-VLP16-launch.py')])
        ),
                # ✅ NEW: POE cameras launch (gscam RTSP)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rtsp_gscam_dir, 'launch', 'POE_CAMERAS.launch.py')
            )
        ),

    ])

