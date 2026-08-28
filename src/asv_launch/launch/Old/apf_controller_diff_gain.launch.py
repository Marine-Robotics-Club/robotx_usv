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
        #
        # Buoys:
        #   B1 = (-18.5, 37.8)
        #   B2 = (-18.0, 28.0)
        #   B3 = (-18.0, 16.7)
        #
        # WP2/WP3 and WP4/WP5 are approximately 4 m from the
        # neighboring buoys along the buoy-to-buoy vectors.
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

                "waypoint0_x": -18.91,
                "waypoint0_y": 45.79,

                "waypoint1_x": -18.83,
                "waypoint1_y": 44.29,

                "waypoint2_x": -18.30,
                "waypoint2_y": 33.81,

                "waypoint3_x": -18.20,
                "waypoint3_y": 31.99,

                "waypoint4_x": -18.00,
                "waypoint4_y": 24.00,

                "waypoint5_x": -18.00,
                "waypoint5_y": 20.70,

                "waypoint6_x": -18.00,
                "waypoint6_y": 6.70,

                "waypoint7_x": -18.00,
                "waypoint7_y": 4.70,

                "switch_radius_m": 2.0,
                "require_start_heading": False,
                "switch_heading_tolerance_deg": 25.0,
            }],
        ),

        # ------------------------------------------------------------
        # DIRECT APF + SEMANTIC VORTEX, HYDRODYNAMIC D-MAPPING
        #
        # The semantic vortex is generated as a 2-D velocity field and
        # converted to an equivalent force with the identified damping
        # model. The complete APF force is then projected to the
        # differential-drive inputs [Tx, 0, Mz].
        # ------------------------------------------------------------
        Node(
            package="apf_grid_controller",
            executable="apf_controller_diff_vortex_D",
            name="apf_controller_diff_vortex_D",
            parameters=[
                {
                    # Robot namespace
                    "asv": asv,
                    "wamv": asv,
                    "use_sim_time": False,

                    # Input topics
                    "pose_topic": "vehicle_pose",
                    "odom_topic": "p3d_wamv_ned",
                    "map_topic": "map/local_occupancy_2",
                    "goal_topic": "nav/goal",
                    "semantic_topic": "map/semantic_buoys",

                    # Output
                    "control_effort_topic": "control_effort",

                    # ------------------------------------------------
                    # SEMANTIC VORTEX
                    # ------------------------------------------------
                    "use_semantic_vortex": True,
                    "vortex_gain": 3.0,
                    "vortex_u_inf": 1.0,
                    "vortex_max_distance_m": 12.0,
                    "iala_region": "B",
                    "vortex_ahead_only": True,
                    "vortex_behind_tol_m": 1.0,
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
                    # VALIDATED APF TUNING -- UNCHANGED
                    # ------------------------------------------------
                    "ka": 40.0,
                    "kr": 3.0,
                    "rho": 5.0,
                    "r_usv": 0.5,
                    "c_safe": 0.7,
                    "Fcap": 150.0,

                    # Wall-following / local-minimum escape
                    "d_slide": 11.0,
                    "w_max": 1.2,
                    "k_wall": 11.0,
                    "K_yaw_apf": 120.0,

                    # Force limits
                    "F_max": 50.0,
                    "Mz_max": 45.0,

                    # Damping injection
                    "Kd_diag": [35.0, 0.0, 5.0],

                    # APF lookahead yaw
                    "use_lookahead_yaw": True,
                    "yaw_lookahead_m": 4.0,
                    "yaw_lookahead_blend": 0.7,
                    "yaw_lookahead_min_dist": 0.6,
                    "yaw_lookahead_min_field_strength": 0.30,

                    # Timing
                    "dt": 0.1,

                    # RViz APF path. The D-vortex controller uses the
                    # forward-only differential-drive rollout.
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
            output="screen",
        ),
    ])