#!/usr/bin/env python3
# RobotX 2024 world champion launch code ROS2 Launch File
# Created by: Caleb Wilson && Xavier Vicent
# Extended with ZED wrapper, zed_2d_box_overlay, compression,
# pyramid_buoy_detector, yolov26 detector, and AI-assisted buoy mapping

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution

from launch_ros.actions import Node


def make_compressor(
    name: str,
    in_topic: str,
    out_topic: str,
    jpeg_quality: int = 60,
) -> Node:
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


def launch_setup(context, *args, **kwargs):

    # ==========================================================
    # Launch configuration variables
    # ==========================================================
    start_zed_node = LaunchConfiguration("start_zed_node")
    camera_name = LaunchConfiguration("camera_name")
    camera_model = LaunchConfiguration("camera_model")
    publish_svo_clock = LaunchConfiguration("publish_svo_clock")
    config_path = LaunchConfiguration("config_path")
    object_detection_conf_path = LaunchConfiguration(
        "object_detection_conf_path"
    )
    asv = LaunchConfiguration("asv")
    yolov26_params_file = LaunchConfiguration("yolov26_params_file")

    # ==========================================================
    # Resolve camera name
    # ==========================================================
    camera_name_val = camera_name.perform(context)

    if camera_name_val == "":
        camera_name_val = "zed"

    # ==========================================================
    # Package directories
    # ==========================================================
    velodyne_dir = get_package_share_directory("velodyne")

    zed_wrapper_dir = get_package_share_directory(
        "zed_wrapper"
    )

    asv_lidar_dir = get_package_share_directory(
        "asv_lidar"
    )

    robust_buoy_mapping_dir = get_package_share_directory(
        "robust_buoy_mapping"
    )

    actions = []

    # ==========================================================
    # Velodyne LiDAR
    # ==========================================================
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    velodyne_dir,
                    "launch",
                    "velodyne-all-nodes-VLP16-launch.py",
                )
            )
        )
    )

    # ==========================================================
    # Pyramid Buoy Detector
    # ==========================================================
    actions.append(
        Node(
            package="asv_lidar",
            executable="pyramid_buoy_detector",
            name="pyramid_buoy_detector",
            output="screen",
            parameters=[
                os.path.join(
                    asv_lidar_dir,
                    "config",
                    "pyramid_buoy_detector.yaml",
                ),
                {
                    "wamv": asv,
                },
            ],
        )
    )

    # ==========================================================
    # ZED Wrapper
    # ==========================================================
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    zed_wrapper_dir,
                    "launch",
                    "zed_camera.launch.py",
                )
            ),
            launch_arguments={
                "camera_name": camera_name_val,
                "camera_model": camera_model.perform(context),
                "publish_svo_clock": publish_svo_clock.perform(context),
                "config_path": config_path.perform(context),
                "object_detection_conf_path":
                    object_detection_conf_path.perform(context),
            }.items(),
            condition=IfCondition(start_zed_node),
        )
    )

    # ==========================================================
    # YOLOv26 Detector
    # ==========================================================
    actions.append(
        Node(
            package="yolov26_ros",
            executable="detector",
            name="yolov26_detector",
            output="screen",
            parameters=[
                yolov26_params_file,
            ],
        )
    )

    # ==========================================================
    # ZED 2D Bounding Box Overlay
    # ==========================================================
    actions.append(
        Node(
            package="zed_custom",
            executable="zed_2d_box_overlay",
            name="zed_2d_box_overlay",
            namespace=camera_name_val,
            output="screen",
            parameters=[
                {
                    "image_topic":
                        f"/{camera_name_val}/"
                        f"{camera_name_val}_node/"
                        "rgb/color/rect/image",

                    "objects_topic":
                        f"/{camera_name_val}/"
                        f"{camera_name_val}_node/"
                        "obj_det/objects",

                    "output_image_topic":
                        f"/{camera_name_val}/"
                        f"{camera_name_val}_node/"
                        "obj_det/image_2d_boxes",
                }
            ],
        )
    )

    # ==========================================================
    # Image Compression
    # ==========================================================
    jpeg_q = 10

    actions.append(
        make_compressor(
            name="compress_zed_2d_boxes",

            in_topic=(
                f"/{camera_name_val}/"
                f"{camera_name_val}_node/"
                "obj_det/image_2d_boxes"
            ),

            out_topic=(
                f"/{camera_name_val}/"
                f"{camera_name_val}_node/"
                "obj_det/image_2d_boxes/compressed"
            ),

            jpeg_quality=jpeg_q,
        )
    )

    # ==========================================================
    # AI-Assisted Dynamic KF Buoy Mapping
    #
    # Equivalent command:
    #
    # ros2 launch robust_buoy_mapping \
    #   mapping_only_ai_assisted_dynamic_kf.launch.py
    #
    # ==========================================================
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    robust_buoy_mapping_dir,
                    "launch",
                    "mapping_only_ai_assisted_dynamic_kf.launch.py",
                )
            )
        )
    )

    return actions


def generate_launch_description():

    return LaunchDescription(
        [

            # ==================================================
            # ZED startup
            # ==================================================
            DeclareLaunchArgument(
                "start_zed_node",
                default_value="True",
                description=(
                    "Set to False if a ZED node "
                    "is already running."
                ),
            ),

            # ==================================================
            # Camera name
            # ==================================================
            DeclareLaunchArgument(
                "camera_name",
                default_value=TextSubstitution(
                    text="zed"
                ),
                description="Camera namespace/name.",
            ),

            # ==================================================
            # Camera model
            # ==================================================
            DeclareLaunchArgument(
                "camera_model",
                default_value="zed2i",
                description="ZED camera model.",
                choices=[
                    "zed",
                    "zedm",
                    "zed2",
                    "zed2i",
                    "zedx",
                    "zedxm",
                    "virtual",
                    "zedxonegs",
                    "zedxone4k",
                ],
            ),

            # ==================================================
            # SVO Clock
            # ==================================================
            DeclareLaunchArgument(
                "publish_svo_clock",
                default_value="false",
                description=(
                    "Publish SVO clock if needed."
                ),
            ),

            # ==================================================
            # ZED Config
            # ==================================================
            DeclareLaunchArgument(
                "config_path",
                default_value="",
                description=(
                    "Path to the common ZED wrapper "
                    "YAML config file."
                ),
            ),

            # ==================================================
            # ZED Object Detection Config
            # ==================================================
            DeclareLaunchArgument(
                "object_detection_conf_path",
                default_value="",
                description=(
                    "Path to the custom object "
                    "detection YAML config file."
                ),
            ),

            # ==================================================
            # ASV name
            # ==================================================
            DeclareLaunchArgument(
                "asv",
                default_value="asv",
                description=(
                    "Value passed to the "
                    "pyramid_buoy_detector "
                    "'wamv' parameter."
                ),
            ),

            # ==================================================
            # YOLOv26 parameters
            # ==================================================
            DeclareLaunchArgument(
                "yolov26_params_file",
                default_value=(
                    "/home/highlevel/roboboat_usv/"
                    "src/asv_perception/"
                    "yolov26_ros/config/"
                    "yolov26_params.yaml"
                ),
                description=(
                    "Path to the YOLOv26 detector "
                    "parameters YAML file."
                ),
            ),

            # ==================================================
            # Create launch actions
            # ==================================================
            OpaqueFunction(
                function=launch_setup
            ),
        ]
    )