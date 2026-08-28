from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    asv = LaunchConfiguration('asv')
    gt_csv = LaunchConfiguration('gt_csv')
    error_csv = LaunchConfiguration('error_csv')

    return LaunchDescription([
        DeclareLaunchArgument('asv', default_value='asv'),

        # GT only for evaluator, not used by mapper.
        DeclareLaunchArgument(
            'gt_csv',
            default_value='/home/highlevel/roboboat_vehicle_data/ground_truth/gt_buoys_test_day.csv'
        ),

        DeclareLaunchArgument(
            'error_csv',
            default_value='/home/highlevel/roboboat_vehicle_data/logs/dynamic_kf_mapping_error.csv'
        ),

        Node(
            package='robust_buoy_mapping',
            executable='dynamic_kf_ai_quality_logger_node',
            name='dynamic_kf_ai_quality_logger_node',
            parameters=[{
                'asv': asv,

                'camera_topic': '/zed_custom_detections',
                'pose_topic': '/asv/vehicle_pose',

                'semantic_buoys_topic': 'map/semantic_buoys',
                'map_topic': 'map/local_occupancy_2',

                'semantic_marker_topic': '/asv/camera_mapping/semantic_map_markers',
                'raw_marker_topic': '/asv/camera_mapping/raw_detection_markers',

                'frame_id': 'map',
                'occupancy_frame_id': 'map',

                # Calibrated transform
                'camera_x_offset_m': 0.6604,
                'camera_y_offset_m': 0.0,
                'camera_yaw_offset_rad': 0.061087,

                # ZED filtering
                'min_confidence': 45.0,
                'min_range_xy': 0.05,
                'max_range_xy': 22.0,

                # KF dynamics for slow drifting anchored buoys
                'initial_position_sigma': 1.5,
                'initial_velocity_sigma': 0.25,
                'process_accel_sigma': 0.025,
                'velocity_decay_time_s': 3.0,
                'max_speed_mps': 0.08,

                # ZED measurement noise
                'measurement_sigma_base': 0.90,
                'measurement_sigma_per_meter': 0.12,
                'measurement_sigma_min': 0.35,
                'measurement_sigma_max': 3.5,

                # Unknown number of buoys:
                # no max track count. Tracks are controlled by birth/merge/delete logic.
                'mahalanobis_gate_confirmed': 9.21,
                'mahalanobis_gate_tentative': 16.0,
                'confirm_hits': 8,
                'delete_tentative_after_missing_s': 2.0,
                'delete_confirmed_after_missing_s': 60.0,
                'birth_min_separation_m': 2.5,
                'merge_distance_m': 2.0,
                'merge_mahalanobis_gate': 4.0,

                'publish_tentative_tracks': False,

                # Occupancy grid
                'occupancy_resolution_m': 0.25,
                'occupancy_width_m': 80.0,
                'occupancy_height_m': 80.0,
                'occupancy_obstacle_radius_m': 0.25,
                'occupancy_inflation_radius_m': 0.15,
                'occupancy_covariance_scale': 0.0,
                'occupancy_use_fixed_radius': True,
                'occupancy_fixed_radius_m': 0.40,

                'publish_period_s': 0.5,
                'marker_diameter_m': 0.45,
                'raw_marker_diameter_m': 0.35,

                'ai_log_enabled': True,
                'ai_log_dir': '/home/highlevel/roboboat_vehicle_data/processed/ai_self_supervised',
                'ai_pair_logging_gate': 50.0,
                'snapshot_period_s': 0.5,
            }],
            output='screen'
        ),

        Node(
            package='robust_buoy_mapping',
            executable='gt_eval_node',
            name='gt_eval_node',
            parameters=[{
                'gt_csv': gt_csv,
                'semantic_buoys_topic': '/asv/map/semantic_buoys',
                'gt_marker_topic': '/asv/viz/gt_buoy_markers',
                'match_gate_m': 5.0,
                'frame_id': 'map',
                'publish_period_s': 0.5,
                'marker_diameter_m': 0.75,
                'error_csv': error_csv,
            }],
            output='screen'
        ),
    ])
