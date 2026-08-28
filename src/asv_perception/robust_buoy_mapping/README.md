# robust_buoy_mapping

Mapping-only ROS 2 package for the four-buoy field test.

This package does **not** run APF and does **not** run the path planner. It only builds and evaluates a robust semantic buoy map.

## What it does

Pipeline:

```text
/zed_custom_detections + /asv/vehicle_pose
→ calibrated projection using 26 inch camera/GNSS offset and +3.5 deg yaw offset
→ rolling buffer of projected red/green buoy observations
→ robust color-aware clustering: 2 green + 2 red
→ stable semantic landmarks
→ /asv/map/semantic_buoys
→ /asv/map/local_occupancy_2
```

It also publishes GT markers and writes mapping error CSV.

## Main topics

Inputs:

```text
/zed_custom_detections
/asv/vehicle_pose
```

Outputs:

```text
/asv/map/semantic_buoys
/asv/map/local_occupancy_2
/asv/camera_mapping/semantic_map_markers
/asv/camera_mapping/raw_detection_markers
/asv/viz/gt_buoy_markers
```

## Install

```bash
cd ~/roboboat_usv/src/asv_perception
unzip /mnt/data/robust_buoy_mapping_package.zip

cd ~/roboboat_usv
source /opt/ros/humble/setup.bash
PYTHONNOUSERSITE=1 colcon build --packages-select robust_buoy_mapping
source install/setup.bash
```

## Ground truth file

Use your corrected GNSS buoy locations:

```bash
mkdir -p ~/roboboat_vehicle_data/ground_truth
cat > ~/roboboat_vehicle_data/ground_truth/gt_buoys_test_day.csv << 'CSV'
id,color,north_m,east_m
1,green,-27.545590,36.482525
2,red,-27.968922,38.923496
3,green,-27.696740,40.766982
4,red,-28.665106,44.472415
CSV
```

## Run mapping-only

```bash
pkill -f cluster4_mapper_node
pkill -f gt_eval_node
pkill -f semantic_map
pkill -f apf_controller_diff
pkill -f path_planner_rx
pkill -f allocation_diff

ros2 launch robust_buoy_mapping mapping_only_cluster4.launch.py
```

## Watch error

```bash
tail -f ~/roboboat_vehicle_data/logs/robust_cluster4_mapping_error.csv
```

Good result:

```text
live_count = 4
matched_count = 4
mean_error_m around 1.0–1.5 m
no growth to 6, 7, 8 tracks
```

## RViz displays

Add:

```text
/asv/camera_mapping/raw_detection_markers
/asv/camera_mapping/semantic_map_markers
/asv/viz/gt_buoy_markers
/asv/map/local_occupancy_2
```

## Notes

The package uses the calibration from today's data:

```text
camera_x_offset_m = 0.6604
camera_yaw_offset_rad = 0.061087
```

That is 26 inches from GNSS antenna to camera, and +3.5 degrees camera yaw offset.

---

# LiDAR + Camera Fusion Update

This version adds a two-stage AI pipeline:

```text
/zed_custom_detections + /wamv1/vision/output/buoy_objects + /asv/vehicle_pose
→ LiDAR-camera XGBoost fusion
→ /asv/perception/fused_buoy_detections
→ existing dynamic KF AI mapper
→ /asv/map/semantic_buoys
→ /asv/map/local_occupancy_2
```

The GPS antenna is treated as the body-frame origin.  The default extrinsics are:

```text
camera_x_offset_m = 0.6604 m    # 26 inches forward of GPS antenna
lidar_x_offset_m  = 0.5080 m    # 20 inches forward of GPS antenna
camera/LiDAR input detections = sensor-frame NWU/FLU
  x_sensor = forward, y_sensor = left, z_sensor = up

fusion/mapping body frame = GPS/body mapper frame
  x_body = forward, y_body = right

Conversion used before XGBoost fusion:
  x_body = x_sensor + sensor_x_offset_m
  y_body = -y_sensor + sensor_y_offset_m

map x = North
map y = East
```

The LiDAR-camera fusion model is the first AI stage.  Its output is then consumed by the existing mapping AI stage.  Because the fused measurements have different error statistics than the camera-only measurements, retrain the mapping AI after collecting fused data.

## New nodes

```text
lidar_camera_fusion_logger_node
train_lidar_camera_fusion_xgboost
lidar_camera_fusion_inference_node
fused_dynamic_kf_ai_mapper_node
```

## 1. Build

```bash
cd ~/roboboat_usv
source /opt/ros/humble/setup.bash
PYTHONNOUSERSITE=1 colcon build --packages-select robust_buoy_mapping
source install/setup.bash
```

Install Python ML dependencies if needed:

```bash
python3 -m pip install --user xgboost scikit-learn pandas joblib
```

If `xgboost` is not installed, the training script falls back to scikit-learn histogram gradient boosting.  The filenames remain `*_xgb.joblib` so the launch files do not change.

## 2. Log LiDAR-camera fusion candidates

Run your camera detector, your LiDAR buoy detector package, and vehicle pose publisher first.  The fusion logger now subscribes to the LiDAR buoy-detection outputs from your C++ detector; it does **not** cluster `/unilidar/cloud` again.

Default LiDAR detection input:

```text
/wamv1/vision/output/buoy_objects       # fau_msgs/msg/ObjectPosition
```

Fallback LiDAR detection input:

```text
/wamv1/vision/output/buoy_detected      # lidar_msgs/msg/BuoyDetected
```

Then run:

```bash
ros2 launch robust_buoy_mapping lidar_camera_fusion_logger.launch.py \
  camera_topic:=/zed_custom_detections \
  lidar_detection_source:=fau_objects \
  lidar_objects_topic:=/wamv1/vision/output/buoy_objects \
  lidar_buoy_detected_topic:=/wamv1/vision/output/buoy_detected \
  pose_topic:=/asv/vehicle_pose \
  log_csv:=/home/highlevel/roboboat_vehicle_data/logs/lidar_camera_fusion_candidates.csv
```

Useful checks:

```bash
ros2 topic echo /zed_custom_detections --once
ros2 topic echo /wamv1/vision/output/buoy_objects --once
ros2 topic echo /wamv1/vision/output/buoy_detected --once
ros2 topic echo /asv/vehicle_pose --once

tail -f /home/highlevel/roboboat_vehicle_data/logs/lidar_camera_fusion_candidates.csv
```

Important columns:

```text
camera_x_body_m, camera_y_body_m
lidar_x_body_m, lidar_y_body_m
lidar_radius_m
cam_lidar_dist_m
raw_fused_map_x_m, raw_fused_map_y_m
lidar_source
```

If `lidar_has_cluster` is always zero, increase `association_radius_m` and verify the LiDAR detection topic.  For your current setup, keep `camera_y_is_left:=true` and `lidar_y_is_left:=true` because both detector outputs are NWU/FLU.  Only set them to `false` if a detection topic is already converted to +y-right body coordinates.

## 3. Train the LiDAR-camera fusion XGBoost model

Use the same corrected GT buoy file:

```bash
mkdir -p /home/highlevel/roboboat_usv/src/asv_perception/robust_buoy_mapping/models/lidar_camera_fusion

ros2 run robust_buoy_mapping train_lidar_camera_fusion_xgboost \
  --fusion-log /home/highlevel/roboboat_vehicle_data/logs/lidar_camera_fusion_candidates.csv \
  --gt-csv /home/highlevel/roboboat_vehicle_data/ground_truth/gt_buoys_test_day.csv \
  --out-dir /home/highlevel/roboboat_usv/src/asv_perception/robust_buoy_mapping/models/lidar_camera_fusion \
  --reliable-gate-m 1.25 \
  --max-negative-range-m 8.0
```

This creates:

```text
fusion_reliable_xgb.joblib      # classifier: should this fused measurement be trusted?
fusion_dx_xgb.joblib            # map-frame x/North residual correction
fusion_dy_xgb.joblib            # map-frame y/East residual correction
fusion_sigma_xgb.joblib         # fused measurement sigma for the KF
lidar_camera_fusion_supervised_dataset.csv
```

## 4. Run LiDAR-camera fusion + existing mapping AI

```bash
pkill -f dynamic_kf_ai_assisted_mapper_node
pkill -f dynamic_kf_buoy_mapper_node
pkill -f fused_dynamic_kf_ai_mapper_node
pkill -f lidar_camera_fusion_inference_node
pkill -f gt_eval_node

ros2 launch robust_buoy_mapping lidar_camera_fusion_ai_mapping.launch.py \
  camera_topic:=/zed_custom_detections \
  lidar_detection_source:=fau_objects \
  lidar_objects_topic:=/wamv1/vision/output/buoy_objects \
  lidar_buoy_detected_topic:=/wamv1/vision/output/buoy_detected \
  pose_topic:=/asv/vehicle_pose
```

RViz topics:

```text
/asv/perception/fused_buoy_markers
/asv/camera_mapping/raw_detection_markers
/asv/camera_mapping/semantic_map_markers
/asv/viz/gt_buoy_markers
/asv/map/local_occupancy_2
```

The inference node publishes:

```text
/asv/perception/fused_buoy_detections
```

The mapper consumes that topic and still publishes:

```text
/asv/map/semantic_buoys
/asv/map/local_occupancy_2
```

## 5. Retrain the second-stage mapping AI for fused detections

The old mapping AI was trained from camera-only measurement statistics.  After fusion is working, collect mapping logs with the fused mapper and retrain the existing pair/sigma/birth models.  The current package already has the self-supervised mapping dataset/training scripts; use the fused mapper output as the new teacher data.

Recommended order:

```text
1. Train LiDAR-camera fusion AI.
2. Run lidar_camera_fusion_ai_mapping.launch.py.
3. Log mapping pair/birth/snapshot data using the fused detections.
4. Rebuild the self-supervised mapping datasets.
5. Retrain pair_reliable_update, measurement_sigma, and birth_reliable models.
```

Until the new mapping AI is trained, the fused mapper can still run, but the existing mapping models may be slightly conservative or biased because the input noise changed.

## Tuning notes

If LiDAR and camera do not line up:

```text
camera_x_offset_m = 0.6604
lidar_x_offset_m  = 0.5080
camera_y_is_left  = True   # your camera detections are NWU/FLU: +y left
lidar_y_is_left   = True   # your LiDAR detections are NWU/FLU: +y left
```

These defaults flip y once before fusion so the existing mapper receives +y-right body measurements.  Do not flip y in your LiDAR detector and again in this fusion node.

If the fusion rejects too many detections:

```text
p_reliable_threshold: lower from 0.50 to 0.35
association_radius_m: increase from 1.8 to 2.5
lidar_detection_source: try fau_objects first, then buoy_detected if needed
```

If the map creates duplicate buoys:

```text
birth_min_separation_m: increase
merge_distance_m: increase
p_birth_threshold: increase
```
