from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    asv = LaunchConfiguration('asv')
    camera_topic = LaunchConfiguration('camera_topic')
    lidar_detection_source = LaunchConfiguration('lidar_detection_source')
    lidar_objects_topic = LaunchConfiguration('lidar_objects_topic')
    lidar_buoy_detected_topic = LaunchConfiguration('lidar_buoy_detected_topic')
    pose_topic = LaunchConfiguration('pose_topic')
    fused_topic = LaunchConfiguration('fused_topic')
    gt_csv = LaunchConfiguration('gt_csv')
    error_csv = LaunchConfiguration('error_csv')
    fusion_model_dir = LaunchConfiguration('fusion_model_dir')
    mapping_model_dir = LaunchConfiguration('mapping_model_dir')
    camera_y_is_left = LaunchConfiguration('camera_y_is_left')
    lidar_y_is_left = LaunchConfiguration('lidar_y_is_left')
    association_radius_m = LaunchConfiguration('association_radius_m')
    allow_camera_only = LaunchConfiguration('allow_camera_only')
    p_reliable_threshold = LaunchConfiguration('p_reliable_threshold')
    birth_min_separation_m = LaunchConfiguration('birth_min_separation_m')
    merge_distance_m = LaunchConfiguration('merge_distance_m')
    confirm_hits = LaunchConfiguration('confirm_hits')

    return LaunchDescription([
        DeclareLaunchArgument('asv', default_value='asv'),
        DeclareLaunchArgument('camera_topic', default_value='/zed_custom_detections'),
        DeclareLaunchArgument('lidar_detection_source', default_value='fau_objects'),
        DeclareLaunchArgument('lidar_objects_topic', default_value='/wamv1/vision/output/buoy_objects'),
        DeclareLaunchArgument('lidar_buoy_detected_topic', default_value='/wamv1/vision/output/buoy_detected'),
        DeclareLaunchArgument('pose_topic', default_value='/asv/vehicle_pose'),
        DeclareLaunchArgument('fused_topic', default_value='/asv/perception/fused_buoy_detections'),
        DeclareLaunchArgument('fusion_model_dir', default_value='/home/highlevel/roboboat_usv/src/asv_perception/robust_buoy_mapping/models/lidar_camera_fusion'),
        DeclareLaunchArgument('mapping_model_dir', default_value='/home/highlevel/roboboat_usv/src/asv_perception/robust_buoy_mapping/models/self_supervised'),
        DeclareLaunchArgument('gt_csv', default_value='/home/highlevel/roboboat_vehicle_data/ground_truth/gt_buoys_test_day.csv'),
        DeclareLaunchArgument('error_csv', default_value='/home/highlevel/roboboat_vehicle_data/logs/lidar_camera_fusion_ai_mapping_error.csv'),
        DeclareLaunchArgument('camera_y_is_left', default_value='true'),
        DeclareLaunchArgument('lidar_y_is_left', default_value='true'),
        DeclareLaunchArgument('association_radius_m', default_value='1.20'),
        DeclareLaunchArgument('allow_camera_only', default_value='false'),
        DeclareLaunchArgument('p_reliable_threshold', default_value='0.70'),
        DeclareLaunchArgument('birth_min_separation_m', default_value='1.80'),
        DeclareLaunchArgument('merge_distance_m', default_value='1.80'),
        DeclareLaunchArgument('confirm_hits', default_value='12'),

        Node(
            package='robust_buoy_mapping',
            executable='lidar_camera_fusion_inference_node',
            name='lidar_camera_fusion_inference_node',
            parameters=[{
                'wamv': asv,
                'camera_topic': camera_topic,
                'lidar_detection_source': lidar_detection_source,
                'lidar_objects_topic': lidar_objects_topic,
                'lidar_buoy_detected_topic': lidar_buoy_detected_topic,
                'pose_topic': pose_topic,
                'fused_topic': fused_topic,
                'fused_marker_topic': '/asv/perception/fused_buoy_markers',

                'model_dir': fusion_model_dir,
                'p_reliable_threshold': p_reliable_threshold,
                'publish_without_model': True,
                'allow_camera_only': allow_camera_only,

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

                'association_radius_m': association_radius_m,
                'lidar_blend_weight': 0.75,
                'max_lidar_age_s': 0.5,
                'max_correction_m': 2.0,
                'sigma_min_m': 0.25,
                'sigma_max_m': 4.0,
            }],
            output='screen'
        ),

        Node(
            package='robust_buoy_mapping',
            executable='fused_dynamic_kf_ai_mapper_node',
            name='fused_dynamic_kf_ai_mapper_node',
            parameters=[{
                'asv': asv,

                # Do not feed raw camera into this node. It consumes fused_topic below.
                'camera_topic': '/unused/raw_camera_disabled_for_fusion_mapper',
                'fusion_topic': fused_topic,
                'pose_topic': pose_topic,

                'semantic_buoys_topic': 'map/semantic_buoys',
                'map_topic': 'map/local_occupancy_2',
                'semantic_marker_topic': '/asv/camera_mapping/semantic_map_markers',
                'raw_marker_topic': '/asv/camera_mapping/raw_detection_markers',
                'frame_id': 'map',
                'occupancy_frame_id': 'map',

                # Not used for fused detections, but kept for parent class compatibility.
                'camera_x_offset_m': 0.0,
                'camera_y_offset_m': 0.0,
                'camera_yaw_offset_rad': 0.0,

                # Fused measurement filtering/noise clamps.
                'min_confidence': 0.0,
                'min_range_xy': 0.05,
                'max_range_xy': 30.0,
                'measurement_sigma_base': 0.45,
                'measurement_sigma_per_meter': 0.04,
                'measurement_sigma_min': 0.25,
                'measurement_sigma_max': 4.0,

                # Slow drifting anchored buoys.
                'initial_position_sigma': 1.25,
                'initial_velocity_sigma': 0.20,
                'process_accel_sigma': 0.025,
                'velocity_decay_time_s': 3.0,
                'max_speed_mps': 0.08,

                'mahalanobis_gate_confirmed': 9.21,
                'mahalanobis_gate_tentative': 16.0,
                'confirm_hits': confirm_hits,
                'delete_tentative_after_missing_s': 2.0,
                'delete_confirmed_after_missing_s': 60.0,
                'birth_min_separation_m': birth_min_separation_m,
                'merge_distance_m': merge_distance_m,
                'merge_mahalanobis_gate': 4.0,
                'publish_tentative_tracks': False,

                'occupancy_resolution_m': 0.25,
                'occupancy_width_m': 80.0,
                'occupancy_height_m': 80.0,
                'occupancy_obstacle_radius_m': 0.25,
                'occupancy_inflation_radius_m': 0.15,
                'occupancy_covariance_scale': 0.0,
                'occupancy_use_fixed_radius': True,
                'occupancy_fixed_radius_m': 0.30,

                'publish_period_s': 0.5,
                'marker_diameter_m': 0.45,
                'raw_marker_diameter_m': 0.35,

                # Existing mapping AI. You probably need to retrain this on fused measurements.
                'ai_enabled': True,
                'pair_model_path': PathJoinSubstitution([mapping_model_dir, 'pair_reliable_update_rf.joblib']),
                'sigma_model_path': PathJoinSubstitution([mapping_model_dir, 'measurement_sigma_rf.joblib']),
                'birth_model_path': PathJoinSubstitution([mapping_model_dir, 'birth_reliable_rf.joblib']),
                'p_update_threshold': 0.60,
                'p_birth_threshold': 0.70,
                'ai_score_weight': 8.0,
                'sigma_blend_alpha': 0.75,
                'ai_sigma_min': 0.25,
                'ai_sigma_max': 4.0,
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