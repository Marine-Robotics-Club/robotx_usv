from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

        # Get workspace root (e.g., /home/highlevel/wamv_nav)
    workspace_root = os.getenv('COLCON_CURRENT_PREFIX', os.getcwd())

    base_src_dir = os.path.join(
        workspace_root,
        'src',
        'asv_state'
    )

    state_yaml_path = os.path.join(base_src_dir, 'config','state.yaml')
    asv = LaunchConfiguration('asv')
    mission_id = LaunchConfiguration('mission_id')
    velD = LaunchConfiguration('velD')

    return LaunchDescription([
        DeclareLaunchArgument('asv', default_value='asv', description='ASV robot name'),
        DeclareLaunchArgument('mission_id', default_value='2', description='Mission Number'),
        DeclareLaunchArgument('velD', default_value='1.5', description='Mission Number'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('sbg_ig500n_driver'),
                    'launch',
                    'ig500n_legacy.launch.py'
                )
            )
        ),
        
        Node(
            package="asv_state",
            executable="vehicle_state_vrx",
            parameters=[{'latRef': 26.055555,
                'lonRef': -80.113266,
                'Simulation': False,
                'use_sim_time': False, 
                'asv': asv}],
            output='screen'
        ),
        Node(
            package='ros2serial_py',
            executable='ros2serial_py',
            name='ros2serial_py',
            parameters=[{'asv': asv}],
            output='screen'
        ),
        Node(
            package='ros2serial_py',
            executable='usv_status_bridge',
            name='usv_status_bridge',
            parameters=[{
                'asv': asv,
                'teensy_status_topic': '/asv/teensy_status',
                'gps_topic': '/sbg_legacy/gps/fix',
                'rpy_topic': '/sbg_legacy/rpy',
                'output_topic': '/seaowls/usv1/status',
            }],
            output='screen',
        ),
    ])
