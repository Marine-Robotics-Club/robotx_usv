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
            executable='stop_light_detector',
            name='stop_light_detector',
            output='screen',
            parameters=[os.path.join(vision_cpp_dir, 'config', 'stop_light_detector.yaml')]
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
            executable='boat_detector',
            name='boat_detector',
            output='screen',
            parameters=[os.path.join(vision_cpp_dir, 'config', 'boat_detector.yaml')]
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
            executable='sensor_fusion_left',
            name='sensor_fusion_left',
            output='screen',
        ),
        Node(
            package='vision_cpp',
            executable='sensor_fusion_right',
            name='sensor_fusion_right',
            output='screen',
        ),
    	Node(
            package='vision_cpp',
            executable='Camera_Estimater_RIGHT',
            name='Camera_Estimator_RIGHT',
            output='screen',
        ),
    	Node(
            package='vision_cpp',
            executable='Camera_Estimater_CENTER',
            name='Camera_Estimator_CENTER',
            output='screen',
        ),
    	Node(
            package='vision_cpp',
            executable='Camera_Estimater_LEFT',
            name='Camera_Estimator_LEFT',
            output='screen',
        ),
        Node(
            package='vision_cpp',
            executable='task_2',
            name='task_2',
            output='screen',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(yolov9_dir, 'yolov9_3.launch.py')])
        ),

    ])

