#!/usr/bin/env python3

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


def launch_setup(context, *args, **kwargs):
    # ==========================================================
    # Launch configuration variables for ZED
    # ==========================================================
    start_zed_node = LaunchConfiguration("start_zed_node")
    camera_name = LaunchConfiguration("camera_name")
    camera_model = LaunchConfiguration("camera_model")
    publish_svo_clock = LaunchConfiguration("publish_svo_clock")
    config_path = LaunchConfiguration("config_path")
    object_detection_conf_path = LaunchConfiguration("object_detection_conf_path")

    camera_name_val = camera_name.perform(context)
    if camera_name_val == "":
        camera_name_val = "zed"

    # ==========================================================
    # Package directories
    # ==========================================================
    zed_wrapper_dir = get_package_share_directory("zed_wrapper")

    actions = []

    # ==========================================================
    # ZED Wrapper
    # ==========================================================
    zed_wrapper_launch = IncludeLaunchDescription(
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
            "object_detection_conf_path": object_detection_conf_path.perform(context),
        }.items(),
        condition=IfCondition(start_zed_node),
    )
    actions.append(zed_wrapper_launch)

    # ==========================================================
    # ZED 2D bounding box overlay node
    # ==========================================================
    actions.append(
        Node(
            package="zed_custom",
            executable="zed_2d_box_overlay",
            name="zed_2d_box_overlay",
            namespace=camera_name_val,
            output="screen",
            parameters=[{
                "image_topic": f"/{camera_name_val}/{camera_name_val}_node/rgb/color/rect/image",
                "objects_topic": f"/{camera_name_val}/{camera_name_val}_node/obj_det/objects",
                "output_image_topic": f"/{camera_name_val}/{camera_name_val}_node/obj_det/image_2d_boxes",
            }]
        )
    )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "start_zed_node",
            default_value="true",
            description="Set to false if a ZED node is already running."
        ),
        DeclareLaunchArgument(
            "camera_name",
            default_value=TextSubstitution(text="zed"),
            description="Camera namespace/name."
        ),
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
            ]
        ),
        DeclareLaunchArgument(
            "publish_svo_clock",
            default_value="false",
            description="Publish SVO clock if needed."
        ),
        DeclareLaunchArgument(
            "config_path",
            default_value="",
            description="Path to the common ZED wrapper YAML config file."
        ),
        DeclareLaunchArgument(
            "object_detection_conf_path",
            default_value="",
            description="Path to the custom object detection YAML config file."
        ),
        OpaqueFunction(function=launch_setup)
    ])