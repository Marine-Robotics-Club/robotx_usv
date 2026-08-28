from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    wamv = LaunchConfiguration('wamv')
    mission_id = LaunchConfiguration('mission_id')

    return LaunchDescription([
        DeclareLaunchArgument('wamv', default_value='wamv1', description='WAM-V robot name'),
        DeclareLaunchArgument('mission_id', default_value='1', description='Mission Number'),

        Node(
            package='the_planner',
            executable='usv_tracking',
            name='usv_tracking',
            parameters=[{'use_sim_time': True}, {'wamv': wamv}],
            output='screen'
        ),

        Node(
            package='wamv_state',
            executable='vehicle_state',
            parameters=[{
                'latRef': -33.72276539997444,
                'lonRef': 150.67399066878366,
                'Simulation': True,
                'use_sim_time': True,
                'wamv': wamv
            }],
            output='screen'
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('robot_localization'),
                    'launch',
                    'dual_ekf_navsat_example.launch.py'
                )
            )
        )
    ])
