# RobotX USV

ROS 2 software for the RobotX unmanned surface vehicle. This guide is for students operating the boat and tuning the local path planner.

> **Safety first:** test with thrusters disabled or the boat secured before enabling control. Keep an emergency stop and manual override available.

## Operating sequence

In every terminal: `source /opt/ros/humble/setup.bash`, `cd ~/robotx_usv`, then `source install/setup.bash`. After changing source or launch files, run `colcon build --symlink-install` and source the workspace again.

### 1. Start sensors

```bash
ros2 launch asv_launch sensors.launch.py
```

This starts the SBG IG-500N GPS/IMU driver, Teensy serial driver/status bridge, and vehicle-state node. The state node converts GPS latitude/longitude into the local `map` frame.

#### Set the GPS `(0, 0)` origin

Edit `src/asv_launch/launch/sensors.launch.py`, in the `vehicle_state_vrx` node:

```python
'latRef': 26.055555,
'lonRef': -80.113266,
```

Replace these with the latitude and longitude of the point that should be `map = (0, 0)`, normally the reference point selected for the test area. Record it and keep the same origin for the entire run. Changing it moves every GPS-derived pose and map; it does not move the boat. The controller convention is approximately `+x = North`, `+y = East`.

### 2. Start vision, LiDAR, and mapping

```bash
ros2 launch asv_launch vision.launch.py
```

This starts the Velodyne VLP-16 driver and LiDAR buoy detector, ZED camera driver, YOLO buoy detector and 2-D overlay, and the AI-assisted dynamic Kalman-filter mapper. The mapper uses camera detections; LiDAR supplies obstacle occupancy and its own detector. Useful overrides are `camera_model:=zed2i`, `start_zed_node:=False` (only if ZED is already running), and `asv:=asv`.

### 3. Start the planner/controller

```bash
ros2 launch asv_launch apf_controller_diff_vortex_D.launch.py
```

This starts the differential-thrust allocator, fixed waypoint publisher, and `apf_controller_diff_vortex_D`. It combines goal attraction, occupancy-grid repulsion, wall-following/local-minimum escape, and a semantic vortex field around confirmed red/green buoys. Change `waypoint*_x/y` in the launch file when the course changes, using the same `map` origin.

## Tuning the path planner

The controller is operational. Focus tuning on vortex guidance, safety distance, obstacle repulsion, and waypoint geometry. Tune one parameter at a time at low speed and log every test.

| Parameter | Meaning | Tuning guidance |
|---|---|---|
| `vortex_gain` | Buoy circulation strength | Increase if the boat does not bend around buoys; decrease if it cuts sharply or oscillates. |
| `vortex_u_inf` | Reference speed scaling the vortex | Adjust with `vortex_gain`; higher means stronger guidance. |
| `vortex_max_distance_m` | Vortex influence range | Increase for earlier shaping; decrease for local behavior. |
| `iala_region` | Buoy convention (`A`/`B`) | Must match the course; this launch uses `B`. |
| `vortex_ahead_only` | Ignore buoys behind | Usually keep `True`. |
| `c_safe` | Extra obstacle/buoy clearance | Increase if too close; decrease only after checking localization and hull size. |
| `r_usv` | Vehicle collision radius | Set from the real hull footprint. |
| `ka` | Goal attraction | Increase if progress is weak; decrease if it overwhelms avoidance. |
| `kr` | Repulsion strength | Increase if obstacles are approached too closely; decrease if avoidance is excessive. |
| `rho` | Repulsion influence distance | Increase to react earlier; decrease if distant cells distract the boat. |
| `d_slide` | Wall-following blend distance | Increase if stuck near walls; decrease if it starts too early. |
| `w_max` | Maximum wall-following blend | Increase carefully to escape local minima. |
| `k_wall` | Wall-following gain | Starts at `0`; increase only when needed. |
| `K_yaw_apf` | Yaw moment from APF/lookahead | Increase for weak heading response; decrease for oscillation. |
| `yaw_lookahead_m` | Forward distance for desired heading | Increase for smoother anticipation; decrease for tighter tracking. |
| `yaw_lookahead_blend` | Lookahead/local-heading blend | Increase for smoothness; decrease for direct response. |
| `F_max`, `Mz_max` | Force/yaw command limits | Reduce for testing; do not increase without propulsion validation. |
| `waypoint*_x/y` | Course points in map metres | Change for the course; verify before enabling thrust. |
| `switch_radius_m` | Distance to advance waypoint | Increase if waypoints are missed; decrease for tighter tracking. |

### Keep unchanged unless formally re-identified

Do not casually change `Kd_diag`, `dt`, hydrodynamic coefficients (`m`, `Iz`, `xG`, `yG`, added-mass and damping terms), `Fcap`, topic names, frame IDs, or allocator settings. These describe the identified boat, timing, interfaces, or safety saturation. Change them only after calibration/vehicle-identification testing. `occ_threshold`, `window_radius_m`, `max_obstacles`, and `downsample_stride` are perception/performance settings, not primary path gains.

## Recommended tuning order

1. Verify GPS origin, frames, pose, occupancy grid, buoy map, and goal in RViz.
2. Set `r_usv` and `c_safe` from the hull and required clearance.
3. Tune `kr` and `rho` until obstacle avoidance is visible and stable.
4. Tune `vortex_gain`, `vortex_u_inf`, and `vortex_max_distance_m` around one buoy.
5. Tune `K_yaw_apf` and lookahead only after field direction is sensible.
6. Use `d_slide`, `w_max`, and `k_wall` only for wall-following/local-minimum issues.
7. Adjust waypoint coordinates and `switch_radius_m` for the course.

## Diagnostics

```bash
ros2 topic list
ros2 topic echo /asv/map/local_occupancy_2
ros2 topic echo /asv/map/semantic_buoys
ros2 topic echo /asv/nav/goal
ros2 topic echo /asv/control_effort
ros2 node list
```

Topic names may appear without `/asv` depending on namespace; use `ros2 topic info <topic>` to confirm. If there is no valid pose, map, or goal, stop and diagnose the upstream node before changing planner gains.
