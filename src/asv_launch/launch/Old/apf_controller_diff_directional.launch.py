from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    asv = LaunchConfiguration("asv")

    return LaunchDescription([
        DeclareLaunchArgument(
            "asv",
            default_value="asv",
            description="ASV robot name / namespace",
        ),

        # ------------------------------------------------------------
        # THRUSTER ALLOCATION
        # ------------------------------------------------------------
        Node(
            package="apf_grid_controller",
            executable="allocation_diff",
            name="allocation_diff",
            parameters=[
                {
                    "asv": asv,
                    "wamv": asv,
                }
            ],
            output="screen",
        ),

        # ------------------------------------------------------------
        # FIXED WAYPOINTS
        # ------------------------------------------------------------
        Node(
            package="apf_grid_controller",
            executable="fixed_points",
            name="fixed_points",
            output="screen",
            parameters=[{
                "asv": asv,
                "frame_id": "map",
                "rate_hz": 2.0,

                "waypoint0_x": -18.0,
                "waypoint0_y": 49.0,

                "waypoint1_x": -18.0,
                "waypoint1_y": 47.5,

                # Between B1 (-18.0, 41.0) and B2 (-15.3, 30.0)
                "waypoint2_x": -17.1,
                "waypoint2_y": 37.3,

                "waypoint3_x": -16.2,
                "waypoint3_y": 33.7,

                # Between B2 (-15.3, 30.0) and B3 (-16.5, 22.0)
                "waypoint4_x": -15.7,
                "waypoint4_y": 27.3,

                "waypoint5_x": -16.1,
                "waypoint5_y": 24.7,

                # After B3
                "waypoint6_x": -16.3,
                "waypoint6_y": 12.0,

                "waypoint7_x": -16.0,
                "waypoint7_y": 10.0,

                "switch_radius_m": 2.0,

                "require_start_heading": False,
                "switch_heading_tolerance_deg": 25.0,
            }],
        ),

        # ------------------------------------------------------------
        # DIRECT DIRECTIONAL / SEMANTIC APF CONTROLLER
        #
        # Green buoy:
        #   USV passes LEFT of buoy
        #   -> green remains on STARBOARD / RIGHT
        #
        # Red buoy:
        #   USV passes RIGHT of buoy
        #   -> red remains on PORT / LEFT
        #
        # Flow:
        #
        # /asv/map/local_occupancy_2
        #             +
        # /asv/map/semantic_buoys
        #             |
        #             v
        # apf_controller_diff_directional
        #             |
        #             v
        # /asv/control_effort
        #             |
        #             v
        # allocation_diff
        #             |
        #             v
        # /asv/motor_cmds
        # ------------------------------------------------------------
        Node(
            package="apf_grid_controller",
            executable="apf_controller_diff_directional",
            name="apf_controller_diff_directional",
            parameters=[
                {
                    # ------------------------------------------------
                    # Robot namespace
                    # ------------------------------------------------
                    "asv": asv,
                    "wamv": asv,
                    "use_sim_time": False,

                    # ------------------------------------------------
                    # Input topics
                    # ------------------------------------------------
                    "pose_topic": "vehicle_pose",
                    "odom_topic": "p3d_wamv_ned",
                    "map_topic": "map/local_occupancy_2",
                    "goal_topic": "nav/goal",

                    # Semantic buoy map:
                    # std_msgs/String JSON containing x, y, color,
                    # confirmed, track_id, etc.
                    "semantic_topic": "map/semantic_buoys",

                    # ------------------------------------------------
                    # Output
                    # ------------------------------------------------
                    "control_effort_topic": "control_effort",

                    # ------------------------------------------------
                    # Semantic directional bias
                    # ------------------------------------------------
                    "use_semantic_bias": False,

                    # Occupancy-grid points within this distance of a
                    # semantic buoy are assigned that buoy's color.
                    "semantic_match_radius_m": 0.3,

                    # Maximum rotation applied to the repulsive-force
                    # direction.
                    "semantic_bias_max_deg": 45.0,

                    # Only use confirmed semantic buoy tracks.
                    "semantic_confirmed_only": True,

                    # ------------------------------------------------
                    # Occupancy-grid obstacle extraction
                    # ------------------------------------------------
                    "occ_threshold": 80,
                    "treat_unknown_as_occupied": False,
                    "window_radius_m": 30.0,
                    "max_obstacles": 1000,
                    "downsample_stride": 2,

                    # ------------------------------------------------
                    # APF tuning
                    # KEEP SAME AS VALIDATED CONTROLLER
                    # ------------------------------------------------
                    "ka": 40.0,
                    "kr": 3.0,
                    "rho": 7.0,

                    # Effective safety boundary:
                    # R_eff = r_o + r_usv + c_safe
                    "r_usv": 0.5,
                    "c_safe": 0.7,

                    "Fcap": 150.0,

                    # ------------------------------------------------
                    # Wall-following / local-minimum escape
                    # ------------------------------------------------
                    "d_slide": 11.0,
                    "w_max": 1.2,
                    "k_wall": 11.0,
                    "K_yaw_apf": 120.0,

                    # ------------------------------------------------
                    # Force limits
                    # ------------------------------------------------
                    "F_max": 50.0,
                    "Mz_max": 45.0,

                    # ------------------------------------------------
                    # Damping
                    # ------------------------------------------------
                    "Kd_diag": [35.0, 0.0, 5.0],

                    # ------------------------------------------------
                    # APF lookahead yaw
                    # ------------------------------------------------
                    "use_lookahead_yaw": True,
                    "yaw_lookahead_m": 4.0,
                    "yaw_lookahead_blend": 0.7,
                    "yaw_lookahead_min_dist": 0.6,
                    "yaw_lookahead_min_field_strength": 0.30,

                    # ------------------------------------------------
                    # Timing
                    # ------------------------------------------------
                    "dt": 0.1,

                    # ------------------------------------------------
                    # RViz APF path
                    # ------------------------------------------------
                    "apf_path_num_points": 80,
                    "apf_path_ds": 0.6,
                    "apf_path_max_yaw_rate": 0.6,

                    # ------------------------------------------------
                    # RViz topics
                    # ------------------------------------------------
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
            output="screen",
        ),
    ])