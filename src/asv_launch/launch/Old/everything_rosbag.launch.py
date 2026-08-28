#!/usr/bin/env python3
# RobotX 2024 world champion launch code ROS2 Launch File
# Created by: Caleb Wilson && Xavier Vicent
# Email: calebwilson@fau.edu

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    vision_cpp_dir = get_package_share_directory('asv_lidar')
    asv = LaunchConfiguration('asv')
    params_file = os.path.expanduser(
        "~/roboboat_usv/src/asv_perception/yolov26_ros/config/yolov26.yaml"
    )
    use_sim_time = False

    return LaunchDescription([
        DeclareLaunchArgument(
            'asv',
            default_value='asv',
            description='ASV/WAMV namespace name'
        ),

        Node(
            package='asv_lidar',
            executable='pyramid_buoy_detector',
            name='pyramid_buoy_detector',
            output='screen',
            parameters=[
                os.path.join(vision_cpp_dir, 'config', 'pyramid_buoy_detector.yaml'),
                {'wamv': asv}
            ]
        ),

        Node(
            package="yolov26_ros",
            executable="detector",
            name="yolov26_detector",
            output="screen",
            parameters=[params_file], 
        ),



        #Node(
        #    package='asv_lidar',
        #    executable='sensor_fusion',
        #    name='sensor_fusion',
        #    output='screen',
        #),

        #Node(
        #    package='vision_tracker',
        #    executable='vision_tracker_2',
        #    name='vision_tracker_2',
        #    output='screen',
        #    parameters=[{"use_sim_time": use_sim_time}, {"wamv": asv}],
        #),
        #Node(
        #    package='vision_tracker',
        #    executable='fusion_map',
        #    name='fusion_map',
        #    output='screen',
        #    parameters=[{"use_sim_time": use_sim_time}, {"wamv": asv}],
        #),
    ])