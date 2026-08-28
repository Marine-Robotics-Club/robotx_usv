from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():

    params_file = os.path.expanduser(
        "~/roboboat_usv/src/asv_perception/yolov26_ros/config/yolov26_params_vrx.yaml"
    )

    return LaunchDescription([
        Node(
            package="yolov26_ros",
            executable="detector",
            name="yolov26_detector",
            output="screen",
            parameters=[params_file], 
        )
    ])
