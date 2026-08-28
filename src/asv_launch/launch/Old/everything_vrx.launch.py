import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
import os


def launch_setup(context, *args, **kwargs):
    use_sim_time_str = LaunchConfiguration('use_sim_time').perform(context)
    asv = LaunchConfiguration('asv').perform(context)

    use_sim_time = use_sim_time_str.lower() == 'true'

    workspace_root = os.getenv('COLCON_CURRENT_PREFIX', os.getcwd())
    base_src_dir = os.path.join(workspace_root, 'roboboat_usv', 'src', 'asv_perception', 'asv_lidar')

    rviz_config_path = os.path.join(base_src_dir, 'config', 'wamv', 'map.rviz')
    buoy_detector_yaml_path = os.path.join(base_src_dir, 'config', 'buoy_detector.yaml')

    params_file = os.path.expanduser(
        "~/roboboat_usv/src/asv_perception/yolov26_ros/config/yolov26_params_vrx.yaml"
    )

    params_left_file = os.path.expanduser(
        "~/roboboat_usv/src/asv_perception/yolov26_ros/config/yolov26_left_params_vrx.yaml"
    )

    actions = [
        launch_ros.actions.Node(
            package="asv_lidar",
            executable="lidar_transform",
            name="lidar_transform",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}, {"wamv": asv}],
        ),

        launch_ros.actions.Node(
            package="asv_lidar",
            executable="clear_water",
            name="clear_water",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}, {"wamv": asv}],
        ),

        launch_ros.actions.Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen'
        ),

        launch_ros.actions.Node(
            package='asv_lidar',
            executable='buoy_detector_vrx',
            name='buoy_detector_vrx',
            output='screen',
            parameters=[buoy_detector_yaml_path, {"wamv": asv}],
        ),

        launch_ros.actions.Node(
            package="yolov26_ros",
            executable="detector",
            name="yolov26_detector",
            output="screen",
            parameters=[params_file],
        ),

        launch_ros.actions.Node(
             package='asv_lidar',
             executable='sensor_fusion_vrx',
             name='sensor_fusion_vrx',
            output='screen',
             parameters=[{"wamv": asv}],
         ),

         launch_ros.actions.Node(
             package='asv_lidar',
             executable='sensor_fusion_left_vrx',
             name='sensor_fusion_left_vrx',
             output='screen',
             parameters=[{"wamv": asv}],
         ),

        launch_ros.actions.Node(
            package='vision_tracker',
            executable='vision_tracker',
            name='vision_tracker',
            output='screen',
            parameters=[{"use_sim_time": use_sim_time}, {"wamv": asv}],
        ),
        launch_ros.actions.Node(
            package='vision_tracker',
            executable='vision_tracker_2',
            name='vision_tracker_2',
            output='screen',
            parameters=[{"use_sim_time": use_sim_time}, {"wamv": asv}],
        ),
        launch_ros.actions.Node(
            package='vision_tracker',
            executable='fusion_map',
            name='fusion_map',
            output='screen',
            parameters=[{"use_sim_time": use_sim_time}, {"wamv": asv}],
        ),

    ]

    return actions


def generate_launch_description():
    return launch.LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        DeclareLaunchArgument('asv', default_value='wamv', description='WAM-V name'),

        OpaqueFunction(function=launch_setup),
    ])
