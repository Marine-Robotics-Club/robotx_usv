from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    asv = LaunchConfiguration("asv")
    velD = LaunchConfiguration("velD")

    return LaunchDescription([
        DeclareLaunchArgument(
            "asv",
            default_value="asv",
            description="ASV robot name"
        ),

        DeclareLaunchArgument(
            "velD",
            default_value="1.0",
            description="Desired velocity"
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

                    "pose_topic": "vehicle_pose",
                    "odom_topic": "p3d_wamv_ned",
                    "map_topic": "map/local_occupancy_2",
                    "goal_topic": "nav/goal",
                    "control_effort_topic": "control_effort",

                    "occ_threshold": 80,
                    "treat_unknown_as_occupied": False,
                    "window_radius_m": 10.0,
                    "max_obstacles": 800,
                    "downsample_stride": 1,

                    # Start conservative but passable.
                    "ka": 80.0,
                    "kr": 0.7,
                    "rho": 1.4,
                    "r_usv": 0.45,
                    "c_safe": 0.20,
                    "Fcap": 240.0,
                    "d_slide": 0.8,
                    "w_max": 0.20,
                    "k_wall": 0.0,
                    "K_yaw_apf": 120.0,

                    "F_max": 80.0,
                    "Mz_max": 35.0,

                    "Kd_diag": [80.0, 0.0, 50.0],
                    "dt": 0.1,

                    "apf_path_num_points": 80,
                    "apf_path_ds": 0.6,
                    "apf_path_max_yaw_rate": 0.6,

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
        # NEW semantic corridor planner
        # ------------------------------------------------------------
        Node(
            package="apf_grid_controller",
            executable="semantic_corridor_planner",
            name="semantic_corridor_planner",
            parameters=[
                {
                    "wamv": asv,

                    # Input from AI-assisted semantic mapper.
                    "semantic_buoys_topic": "map/semantic_buoys",
                    "pose_topic": "/asv/vehicle_pose",

                    # Output to APF.
                    "goal_topic": "/asv/nav/goal",

                    # RViz.
                    "marker_topic": "/asv/viz/semantic_corridor_planner",
                    "frame_id": "map",

                    "semantic_timeout_s": 3.0,
                    "publish_period_s": 0.10,
                    "min_buoy_count": 2,

                    # Gate/corridor selection.
                    "min_gate_width_m": 1.0,
                    "max_gate_width_m": 8.0,
                    "gate_behind_allow_m": 1.0,

                    # Corridor geometry.
                    # This is what forces the APF goal to stay on the passage centerline.
                    "approach_distance_m": 3.0,
                    "exit_distance_m": 7.0,
                    "lookahead_m": 3.0,
                    "goal_reach_threshold_m": 1.2,

                    # Gate scoring:
                    # prefer centered and wider red-green gate.
                    "score_lateral_weight": 1.0,
                    "score_distance_weight": 0.25,
                    "score_width_weight": 1.2,

                    # Keep the chosen gate stable.
                    "latch_gate": True,
                    "latch_max_midpoint_jump_m": 3.0,
                    "lost_gate_timeout_s": 3.0,

                    "hold_exit_goal": True,
                }
            ],
            output="screen"
        ),
    ])
