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
    vision_cpp_dir = get_package_share_directory('wamv_lidar')
    yolov9_dir = get_package_share_directory('yolov9')
    return LaunchDescription([
        Node(
            package='wamv_lidar',
            executable='wamv_detector',
            name='wamv_detector',
            output='screen',
            parameters=[os.path.join(vision_cpp_dir, 'config', 'wamv_detector.yaml')]
        ),
        Node(
            package='wamv_lidar',
            executable='sensor_fusion',
            name='sensor_fusion',
            output='screen',
        ),
        Node(
            package='wamv_lidar',
            executable='right_cam',
            name='right_cam',
            output='screen',
        ),
        Node(
            package='wamv_lidar',
            executable='left_cam',
            name='left_cam',
            output='screen',
        ),
        
        #IncludeLaunchDescription(
        #    PythonLaunchDescriptionSource([os.path.join(yolov9_dir, 'yolov9.launch.py')])
        #),
    ])

