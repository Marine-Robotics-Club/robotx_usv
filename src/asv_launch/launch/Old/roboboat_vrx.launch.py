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
            package='pid_hs',
            executable='pid_hs_node',
            name='pid_hs_node',
            output='screen',
            parameters=[
                {'kp_xy': 0.7},
                {'kp_psi': 5.0},
                {'ki_xy': 0.0},
                {'ki_psi': 1.0},
                {'kd_xy': 1.0},
                {'kd_psi': 10.0},
                {'kp_v': 1.0},
                {'ki_v': 0.0},
                {'kd_v': 1.0},
                {'kp_psi_H': 10.0},
                {'kd_psi_H': 2.0},
                {'ki_psi_H': 0.1},
                {'asv': asv},
                {'velD': velD}
            ],
        ),
        Node(
           package='asv_allocation',
           executable='usv_allocation',
           name='usv_allocation',
           parameters=[{'asv': asv}],
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
    ])
