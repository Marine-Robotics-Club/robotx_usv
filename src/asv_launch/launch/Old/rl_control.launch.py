from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    wamv = LaunchConfiguration('wamv')
    mission_id = LaunchConfiguration('mission_id')

        # Get workspace root (e.g., /home/highlevel/wamv_nav)
    workspace_root = os.getenv('COLCON_CURRENT_PREFIX', os.getcwd())

    base_src_dir = os.path.join(
        workspace_root,
        'src',
        'wamv_state'
    )

    state_yaml_path = os.path.join(base_src_dir, 'config','state.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('wamv', default_value='wamv1', description='WAM-V robot name'),
        DeclareLaunchArgument('mission_id', default_value='1', description='Mission Number'),
        
        Node(
            package="wamv_state",
            executable="vehicle_state",
            parameters=[state_yaml_path, {'wamv': wamv}],
            output='screen'
        ),

        Node(
            package="rl_control",
            executable="rl_control",
            name="rl_control",
            parameters=[{"use_sim_time": False},],
        ),
        Node(
            package='the_planner',
            executable='mission_planner',
            name='mission_planner',
            parameters=[{'use_sim_time': False}, {'wamv': wamv}, {'mission_id': mission_id}],
            output='screen'
        )


    ])
