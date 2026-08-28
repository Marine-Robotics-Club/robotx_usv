import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    wamv = LaunchConfiguration('wamv')

    # Get resolved WAMV name (fallback if environment doesn't provide it)
    wamv_name = os.environ.get('wamv', 'wamv1')

    # Get workspace root (e.g., /home/highlevel/wamv_nav)
    workspace_root = os.getenv('COLCON_CURRENT_PREFIX', os.getcwd())

    # Full path to the wamv_lidar source directory
    base_src_dir = os.path.join(
        workspace_root,
        'src',
        'wamv_perception',
        'wamv_lidar'
    )

    # Config paths from src directory
    detector_yaml_path = os.path.join(base_src_dir, 'config', wamv_name, 'wamv_detector.yaml')
    rviz_config_path = os.path.join(base_src_dir, 'config', wamv_name, 'detector.rviz')

    # Debug print (optional)
    print(f"[Launch] YAML config from: {detector_yaml_path}")
    print(f"[Launch] RViz config from: {rviz_config_path}")

    return launch.LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation time'),
        DeclareLaunchArgument('wamv', default_value='wamv1', description='WAM-V name'),

        launch_ros.actions.Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen'
        ),

        launch_ros.actions.Node(
            package='wamv_lidar',
            executable='wamv_detector',
            name='wamv_detector',
            output='screen',
            parameters=[
                detector_yaml_path,
                {'wamv': wamv}
            ]
        ),
    ])

