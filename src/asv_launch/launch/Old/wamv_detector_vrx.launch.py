import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
import os

def launch_setup(context, *args, **kwargs):
    # Get the values of launch arguments at runtime
    use_sim_time_str = LaunchConfiguration('use_sim_time').perform(context)
    wamv = LaunchConfiguration('wamv').perform(context)

    # Convert use_sim_time to boolean (because rclcpp wants real bool type)
    use_sim_time = use_sim_time_str.lower() == 'true'

    # Get workspace root directory
    workspace_root = os.getenv('COLCON_CURRENT_PREFIX', os.getcwd())

    # Build the src directory path manually
    base_src_dir = os.path.join(workspace_root, 'roboboat_usv', 'src', 'asv_perception', 'asv_lidar')

    # RViz config and detector YAML paths
    rviz_config_path = os.path.join(base_src_dir, 'config', wamv, 'detector.rviz')
    detector_yaml_path = os.path.join(base_src_dir, 'config', wamv, 'wamv_detector.yaml')
    buoy_detector_yaml_path = os.path.join(base_src_dir, 'config', 'buoy_detector.yaml')

    # Now build the full launch description list
    return [

        # lidar_transform node
        launch_ros.actions.Node(
            package="asv_lidar",
            executable="lidar_transform",
            name="lidar_transform",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time},
                {"wamv": wamv}
            ]
        ),

        # clear_water node
        launch_ros.actions.Node(
            package="asv_lidar",
            executable="clear_water",
            name="clear_water",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time},
                {"wamv": wamv}
            ]
        ),

        # RViz2 visualization node
        launch_ros.actions.Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen'
        ),

        # buoy_detector_vrx node
        launch_ros.actions.Node(
            package='asv_lidar',
            executable='buoy_detector_vrx',
            name='buoy_detector_vrx',
            output='screen',
            parameters=[
                buoy_detector_yaml_path,
                {"wamv": wamv}
            ]
        ),
    ]

def generate_launch_description():
    return launch.LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        DeclareLaunchArgument('wamv', default_value='wamv', description='WAM-V name'),
        OpaqueFunction(function=launch_setup)
    ])
