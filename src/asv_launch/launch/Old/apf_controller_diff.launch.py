from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    asv = LaunchConfiguration("asv")
    mission_id = LaunchConfiguration("mission_id")
    velD = LaunchConfiguration("velD")

    return LaunchDescription([
        DeclareLaunchArgument(
            "asv",
            default_value="asv",
            description="ASV robot name"
        ),

        DeclareLaunchArgument(
            "mission_id",
            default_value="2",
            description="Mission number"
        ),

        DeclareLaunchArgument(
            "velD",
            default_value="1.5",
            description="Desired velocity"
        ),

        # ------------------------------------------------------------
        # Raw semantic map from ZED detections
        # Publishes:
        #   /asv/map/semantic_buoys
        #   /asv/map/local_occupancy_2
        # ------------------------------------------------------------
        Node(
            package="zed_camera_mapping",
            executable="camera_raw_semantic_map_open_node",
            name="camera_raw_semantic_map_open_node",
            parameters=[
                {
                    "wamv": asv,

                    # Inputs
                    "camera_topic": "/zed_custom_detections",
                    "pose_topic": "/asv/vehicle_pose",

                    # Outputs
                    "semantic_buoys_topic": "map/semantic_buoys",
                    "map_topic": "map/local_occupancy_2",
                    "frame_id": "map",

                    # Detection filtering
                    "min_confidence": 80.0,
                    "min_range_xy": 0.25,
                    "max_range_xy": 20.0,
                    "reject_edge_detections": True,
                    "edge_margin_px": 40.0,
                    "camera_x_offset_m": 0.6604,
                    "camera_y_offset_m": 0.0,

                    # Landmark association / duplicate rejection
                    "match_distance_m": 3.0,
                    "update_alpha": 0.10,
                    "new_landmark_min_hits": 20,
                    "new_landmark_min_age_s": 2.0,
                    "confirmed_duplicate_distance": 3.0,
                    "publish_pending_landmarks": False,

                    # Occupancy grid
                    "occupancy_resolution_m": 0.25,
                    "occupancy_width_m": 80.0,
                    "occupancy_height_m": 80.0,
                    "occupancy_obstacle_radius_m": 0.22,
                    "occupancy_inflation_radius_m": 0.0,
                    "occupancy_publish_period_s": 0.50,
                    "semantic_buoys_publish_period_s": 0.50,

                    # RViz semantic map marker size
                    "marker_diameter_m": 0.45,
                }
            ],
            output="screen"
        ),

        # ------------------------------------------------------------
        # PWM allocation
        # ------------------------------------------------------------
        Node(
            package="apf_grid_controller",
            executable="allocation_diff",
            name="allocation_diff",
            parameters=[
                {
                    "asv": asv
                }
            ],
            output="screen"
        ),

        # ------------------------------------------------------------
        # APF controller
        # ------------------------------------------------------------
        Node(
            package="apf_grid_controller",
            executable="apf_controller_diff",
            name="apf_controller_diff",
            parameters=[
                {
                    "asv": asv,
                    "use_sim_time": False,

                    # IMPORTANT:
                    # These are relative topic names because apf_controller_diff
                    # internally prefixes them with /asv/.
                    "pose_topic": "vehicle_pose",
                    "odom_topic": "p3d_wamv_ned",
                    "map_topic": "map/local_occupancy_2",
                    "goal_topic": "nav/goal",
                    "control_effort_topic": "control_effort",

                    # Occupancy-grid obstacle extraction
                    "occ_threshold": 80,
                    "treat_unknown_as_occupied": False,
                    "window_radius_m": 10.0,
                    "max_obstacles": 800,
                    "downsample_stride": 1,

                    # Gate-mode APF tuning
                    "ka": 80.0,
                    "kr": 1.0,
                    "rho": 2.0,
                    "r_usv": 0.5,
                    "c_safe": 0.3,
                    "Fcap": 240.0,
                    "d_slide": 0.8,
                    "w_max": 0.20,
                    "k_wall": 0.0,
                    "K_yaw_apf": 120.0,

                    # Force limits
                    "F_max": 80.0,
                    "Mz_max": 35.0,

                    # Damping
                    "Kd_diag": [80.0, 0.0, 50.0],

                    # Timing
                    "dt": 0.1,

                    # RViz APF path
                    "apf_path_num_points": 80,
                    "apf_path_ds": 0.6,
                    "apf_path_max_yaw_rate": 0.6,

                    # RViz topics
                    "actual_path_topic": "viz/apf_actual_path",
                    "desired_path_topic": "viz/apf_desired_path",
                    "goal_marker_topic": "viz/apf_goal_marker",
                    "vehicle_pose_viz_topic": "viz/apf_vehicle_pose",
                    "vehicle_marker_topic": "viz/apf_vehicle_marker",
                    "vehicle_marker_length_m": 2.0,
                    "vehicle_marker_width_m": 0.35,
                    "path_max_len": 2000,
                    "viz_frame_id": "map",
                }
            ],
            output="screen"
        ),

        # ------------------------------------------------------------
        # Semantic red-green gate planner
        # ------------------------------------------------------------
        Node(
            package="apf_grid_controller",
            executable="path_planner_rx",
            name="path_planner_rx",
            parameters=[
                {
                    "asv": asv,

                    # Old parameters, keep them in case other parts still read them.
                    "mission_id": mission_id,
                    "velD": velD,

                    # Semantic-map planner input.
                    # With asv:=asv, this subscribes to:
                    # /asv/map/semantic_buoys
                    "wamv": asv,
                    "semantic_buoys_topic": "map/semantic_buoys",

                    # Pose input.
                    # This planner expects absolute topic names here.
                    "pose_topic": "/asv/vehicle_pose",

                    # Goal output to APF controller.
                    "goal_topic": "/asv/nav/goal",

                    # RViz markers.
                    "marker_topic": "/asv/viz/semantic_gate_goal",
                    "frame_id": "map",

                    # Gate selection
                    "semantic_timeout_s": 3.0,
                    "min_buoy_count": 2,
                    "min_gate_width_m": 1.0,
                    "max_gate_width_m": 8.0,
                    "gate_behind_allow_m": 0.0,

                    # Navigation behavior
                    "through_gate_distance_m": 5.0,
                    "reach_threshold_m": 2.0,
                    "publish_period_s": 0.10,

                    # Waypoints:
                    # True means each gate normally has:
                    #   1) midpoint
                    #   2) through point
                    "use_midpoint_first": True,

                    # New behavior:
                    # If the next gate is already detected after reaching the current
                    # gate midpoint, skip the current through-point and switch to next gate.
                    "skip_through_point_if_next_gate_detected": True,

                    # Number of gates
                    "total_gates": 2,

                    # Duplicate-gate rejection
                    "min_new_gate_separation_m": 7.0,
                    "completed_gate_match_distance_m": 6.0,

                    # Active-gate updates
                    "update_active_goal_from_semantic_map": True,
                    "active_gate_update_max_midpoint_jump_m": 6.0,

                    # Lookahead next gate
                    "lookahead_next_gate_enabled": True,
                    "pending_gate_update_max_midpoint_jump_m": 8.0,
                }
            ],
            output="screen"
        ),
    ])