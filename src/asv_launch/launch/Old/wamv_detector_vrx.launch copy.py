import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Paths to the necessary launch files

    robot_localization_launch_file = os.path.join(
        get_package_share_directory("robot_localization"),
        "launch",
        "dual_ekf_navsat_example.launch.py"
    )

    # Declare the use_sim_time launch argument
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',  # Set 'false' for real hardware
        description='Use simulation time if true'
    )
    vision_cpp_dir = get_package_share_directory('wamv_lidar')
    rviz_config_path = os.path.join(
        os.getenv('COLCON_PREFIX_PATH').split(':')[0],  # fallback if using install
        '..', 'src','wamv_perception', 'wamv_lidar', 'config', 'wamv_detector.rviz'
    )
    rviz_config_path = os.path.abspath(rviz_config_path)

    # Construct the launch description
    return launch.LaunchDescription([
        # Declare the use_sim_time argument
        use_sim_time_arg,

        # PointCloud to LaserScan Node
        launch_ros.actions.Node(
            package="pointcloud_to_laserscan",
            executable="pointcloud_to_laserscan_node",
            name="pointcloud_to_laserscan",
            output="screen",
            remappings=[
                ("/cloud_in", "/lidar_wamv/points_no_water"),
                ("/scan", "/scan")
            ],
            parameters=[{
                "target_frame": "wamv/wamv/base_link",
                "transform_tolerance": 0.1,
                "min_height": -1.0,
                "max_height": 10.0,
                "angle_min": -3.14,
                "angle_max": 3.14,
                "angle_increment": 0.005,
                "scan_time": 0.033,
                "range_min": 2.5,
                "range_max": 100.0,
                "use_inf": True,
                "concurrency_level": 0,
                "qos_reliability": "best_effort"
            }]
        ),

        # Transform Lidar Node
        launch_ros.actions.Node(
            package="wamv_lidar",
            executable="lidar_transform",
            name="lidar_transform",
            output="screen",
            parameters=[{
                "use_sim_time": LaunchConfiguration('use_sim_time')
            }]
        ),

        # Clear Water Node
        launch_ros.actions.Node(
            package="wamv_lidar",
            executable="clear_water",
            name="clear_water",
            output="screen",
            parameters=[{
                "use_sim_time": LaunchConfiguration('use_sim_time')
            }]
        ),
        launch_ros.actions.Node(
            package='wamv_lidar',
            executable='map_detector',
            name='map_detector',
            output='screen',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(robot_localization_launch_file),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()
        ),

        launch_ros.actions.Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
        ),

        launch_ros.actions.Node(
            package='wamv_lidar',
            executable='wamv_detector',
            name='wamv_detector',
            output='screen',
            parameters=[os.path.join(vision_cpp_dir, 'config', 'wamv_detector.yaml')]
        ),
        launch_ros.actions.Node(
            package="fuzzy_logic",
            executable="plot_onr",
            emulate_tty=True
        ),




    ])
