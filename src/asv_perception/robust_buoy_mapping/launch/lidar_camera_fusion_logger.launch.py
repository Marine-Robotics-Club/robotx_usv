from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    wamv = LaunchConfiguration('wamv')
    camera_topic = LaunchConfiguration('camera_topic')
    lidar_detection_source = LaunchConfiguration('lidar_detection_source')
    lidar_objects_topic = LaunchConfiguration('lidar_objects_topic')
    lidar_buoy_detected_topic = LaunchConfiguration('lidar_buoy_detected_topic')
    pose_topic = LaunchConfiguration('pose_topic')
    log_csv = LaunchConfiguration('log_csv')
    camera_y_is_left = LaunchConfiguration('camera_y_is_left')
    lidar_y_is_left = LaunchConfiguration('lidar_y_is_left')

    return LaunchDescription([
        DeclareLaunchArgument('wamv', default_value='wamv1'),
        DeclareLaunchArgument('camera_topic', default_value='/zed_custom_detections'),
        DeclareLaunchArgument('lidar_detection_source', default_value='fau_objects'),
        DeclareLaunchArgument('lidar_objects_topic', default_value='/wamv1/vision/output/buoy_objects'),
        DeclareLaunchArgument('lidar_buoy_detected_topic', default_value='/wamv1/vision/output/buoy_detected'),
        DeclareLaunchArgument('pose_topic', default_value='/asv/vehicle_pose'),
        DeclareLaunchArgument('log_csv', default_value='/home/highlevel/roboboat_vehicle_data/logs/lidar_camera_fusion_candidates.csv'),
        DeclareLaunchArgument('camera_y_is_left', default_value='true'),
        DeclareLaunchArgument('lidar_y_is_left', default_value='true'),

        Node(
            package='robust_buoy_mapping',
            executable='lidar_camera_fusion_logger_node',
            name='lidar_camera_fusion_logger_node',
            parameters=[{
                'wamv': wamv,
                'camera_topic': camera_topic,
                'lidar_detection_source': lidar_detection_source,
                'lidar_objects_topic': lidar_objects_topic,
                'lidar_buoy_detected_topic': lidar_buoy_detected_topic,
                'pose_topic': pose_topic,
                'log_csv': log_csv,

                # Inputs are sensor-frame NWU/FLU (+y left). Fusion flips y once to mapper body (+y right).
                'camera_x_offset_m': 0.6604,     # 26 in
                'camera_y_offset_m': 0.0,
                'camera_yaw_offset_rad': 0.061087,
                'camera_y_is_left': camera_y_is_left,   # NWU/FLU camera detections: +y left

                'lidar_x_offset_m': 0.5080,      # 20 in
                'lidar_y_offset_m': 0.0,
                'lidar_z_offset_m': 0.0,
                'lidar_yaw_offset_rad': 0.0,
                'lidar_y_is_left': lidar_y_is_left,    # NWU/FLU LiDAR detections: +y left

                'min_confidence': 45.0,
                'min_range_xy': 0.05,
                'max_range_xy': 25.0,

                'lidar_min_range_m': 0.3,
                'lidar_max_range_m': 35.0,
                'lidar_min_z_m': -5.0,
                'lidar_max_z_m': 3.0,

                'association_radius_m': 1.8,
                'lidar_blend_weight': 0.75,
                'max_lidar_age_s': 0.5,
            }],
            output='screen'
        ),
    ])
