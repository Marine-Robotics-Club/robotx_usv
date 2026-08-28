from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

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
        DeclareLaunchArgument('asv', default_value='wamv', description='ASV robot name'),
        DeclareLaunchArgument('mission_id', default_value='2', description='Mission Number'),
        DeclareLaunchArgument('velD', default_value='1.5', description='Mission Number'),
        
        Node(
            package="asv_state",
            executable="vehicle_state_vrx",
            parameters=[{'latRef': -33.722769835390636,  
                'lonRef': 150.6739636543141,
                'Simulation': True,
                'use_sim_time': True, 
                'asv': asv}],
            output='screen'
        ),
        Node(
           package='apf_grid_controller',
           executable='allocation',
           name='allocation',
           parameters=[{'asv': asv}],
           output='screen'
        ),
        Node(
           package='apf_grid_controller',
           executable='apf_controller',
           name='apf_controller',
           parameters=[{'asv': asv, 'use_sim_time': True}],
           output='screen'
        ),
        Node(
            package='asv_planner',
            executable='mission_planner',
            name='mission_planner',
            parameters=[{'asv': asv}, {'mission_id': mission_id}],
            output='screen'
        ),

        Node(
            package='asv_planner',
            executable='trajectory_logger',
            name='trajectory_logger',
            parameters=[{"save_path": "/home/xavi/logs/run1.csv"}],
            output='screen'
        ),
        Node(
            package='asv_planner',
            executable='planner_node',
            name='planner_node',
            parameters=[{'use_sim_time': True}, {'asv': asv}, {'mission_id': mission_id}],
            output='screen'
        ),
        Node(
            package='vision_tracker',
            executable='fusion_map',
            name='fusion_map',
            output='screen',
            parameters=[{"use_sim_time": True}, {"wamv": asv}],
        ),
    ])
