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
    return LaunchDescription([
        Node(
            package='vision_cpp',
            executable='dock_detector',
            name='dock_detector',
            output='screen',
            parameters=[os.path.join(vision_cpp_dir, 'config', 'dock_detector.yaml')]
        ),
    ])

