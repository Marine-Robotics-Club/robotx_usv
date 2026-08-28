from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fused_topic = LaunchConfiguration('fused_topic')
    semantic_buoys_topic = LaunchConfiguration('semantic_buoys_topic')
    pose_topic = LaunchConfiguration('pose_topic')
    gt_csv = LaunchConfiguration('gt_csv')
    log_dir = LaunchConfiguration('log_dir')
    match_gate_m = LaunchConfiguration('match_gate_m')
    print_period_s = LaunchConfiguration('print_period_s')

    return LaunchDescription([
        DeclareLaunchArgument('fused_topic', default_value='/asv/perception/fused_buoy_detections'),
        DeclareLaunchArgument('semantic_buoys_topic', default_value='/asv/map/semantic_buoys'),
        DeclareLaunchArgument('pose_topic', default_value='/asv/vehicle_pose'),
        DeclareLaunchArgument('gt_csv', default_value='/home/highlevel/roboboat_vehicle_data/ground_truth/gt_buoys_test_day.csv'),
        DeclareLaunchArgument('log_dir', default_value='/home/highlevel/roboboat_vehicle_data/results/lidar_camera_fusion_mapping'),
        DeclareLaunchArgument('match_gate_m', default_value='5.0'),
        DeclareLaunchArgument('print_period_s', default_value='2.0'),

        Node(
            package='robust_buoy_mapping',
            executable='lidar_camera_fusion_results_logger_node',
            name='lidar_camera_fusion_results_logger_node',
            parameters=[{
                'fused_topic': fused_topic,
                'semantic_buoys_topic': semantic_buoys_topic,
                'pose_topic': pose_topic,
                'gt_csv': gt_csv,
                'log_dir': log_dir,
                'match_gate_m': match_gate_m,
                'log_fused_detections': True,
                'log_pose': True,
                'print_period_s': print_period_s,
            }],
            output='screen'
        ),
    ])