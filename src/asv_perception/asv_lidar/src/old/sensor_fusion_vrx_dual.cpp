// sensor_fusion_vrx_dual.cpp
//
// One node that fuses LiDAR buoy points with YOLO detections from BOTH cameras:
//   - Right camera topic: /yolov26/detections
//   - Left  camera topic: /yolov26/detections/left
//
// Key change (fix for “phantom blacks in the middle”):
//   ✅ Each LiDAR buoy point can be matched to AT MOST ONE YOLO detection per fusion cycle,
//      per camera. We do matching by choosing the best YOLO whose x_center is closest to the
//      LiDAR-projected pixel (or equivalently smallest angular error).
//
// Notes:
// - This version uses a pinhole model to convert YOLO x_center ↔ bearing.
// - It supports mirror options and yaw offsets per camera.
// - It publishes a single fused ObjectPositionArray and an RViz PoseArray.
//
// You MUST tune:
//   camera_hfov_deg_right, camera_hfov_deg_left
//   mirror_right, mirror_left
//   yaw_offset_right_deg, yaw_offset_left_deg
//   max_match_err_deg
//
// Build as a normal ROS2 C++ node.

#include "rclcpp/rclcpp.hpp"

#include "lidar_msgs/msg/buoy_detected.hpp"
#include "yolov26_msgs/msg/yolo_detection.hpp"
#include "fau_msgs/msg/object_position_array.hpp"

#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>

class SensorFusionNode : public rclcpp::Node
{
public:
  SensorFusionNode() : Node("sensor_fusion_vrx_dual")
  {
    // -------- Parameters --------
    this->declare_parameter<std::string>("wamv", "wamv1");
    this->get_parameter("wamv", wamv_name_);

    // Camera model (set your real camera HFOVs)
    this->declare_parameter<double>("image_width_px", 1280.0);

    this->declare_parameter<double>("camera_hfov_deg_right", 60.0);
    this->declare_parameter<double>("camera_hfov_deg_left", 60.0);

    // Mirror YOLO x_center if the image is mirrored
    this->declare_parameter<bool>("mirror_right", false);
    this->declare_parameter<bool>("mirror_left",  true);

    // Small yaw offsets (deg) to align LiDAR bearing to camera bearing
    this->declare_parameter<double>("yaw_offset_right_deg", 0.0);
    this->declare_parameter<double>("yaw_offset_left_deg",  7.8);

    // Max acceptable match error (deg)
    this->declare_parameter<double>("max_match_err_deg", 12.0);

    // LiDAR acceptance region (per your two original nodes)
    // Right side wedge
    this->declare_parameter<double>("right_min_angle_deg", -45.0);
    this->declare_parameter<double>("right_max_angle_deg",  45.0);
    this->declare_parameter<double>("right_max_length_m",   30.0);

    // Left side wedge
    this->declare_parameter<double>("left_min_angle_deg",   45.0);
    this->declare_parameter<double>("left_max_angle_deg",  135.0);
    this->declare_parameter<double>("left_max_length_m",    10.0);

    // Shared limits
    this->declare_parameter<double>("min_height_m", -3.0);
    this->declare_parameter<double>("max_height_m",  3.0);

    // Output frame for RViz PoseArray (GLOBAL frame)
    // IMPORTANT: Your fused points are GLOBAL NED, so publish in your global frame (odom/map).
    this->declare_parameter<std::string>("rviz_frame", "odom");

    // Topics
    this->declare_parameter<std::string>("lidar_topic", "vision/output/buoy_detected");
    this->declare_parameter<std::string>("yolo_right_topic", "/yolov26/detections");
    this->declare_parameter<std::string>("yolo_left_topic",  "/yolov26/detections/left");

    this->declare_parameter<std::string>("fusion_topic", "vision/output/fusion");
    this->declare_parameter<std::string>("rviz_topic",  "vision/output/rviz_fusion");

    // -------- Read params --------
    image_width_px_ = this->get_parameter("image_width_px").as_double();

    camera_hfov_right_rad_ = deg2rad(this->get_parameter("camera_hfov_deg_right").as_double());
    camera_hfov_left_rad_  = deg2rad(this->get_parameter("camera_hfov_deg_left").as_double());

    mirror_right_ = this->get_parameter("mirror_right").as_bool();
    mirror_left_  = this->get_parameter("mirror_left").as_bool();

    yaw_offset_right_rad_ = deg2rad(this->get_parameter("yaw_offset_right_deg").as_double());
    yaw_offset_left_rad_  = deg2rad(this->get_parameter("yaw_offset_left_deg").as_double());

    max_match_err_rad_ = deg2rad(this->get_parameter("max_match_err_deg").as_double());

    right_min_angle_rad_ = deg2rad(this->get_parameter("right_min_angle_deg").as_double());
    right_max_angle_rad_ = deg2rad(this->get_parameter("right_max_angle_deg").as_double());
    right_max_length_m_  = this->get_parameter("right_max_length_m").as_double();

    left_min_angle_rad_ = deg2rad(this->get_parameter("left_min_angle_deg").as_double());
    left_max_angle_rad_ = deg2rad(this->get_parameter("left_max_angle_deg").as_double());
    left_max_length_m_  = this->get_parameter("left_max_length_m").as_double();

    min_height_m_ = this->get_parameter("min_height_m").as_double();
    max_height_m_ = this->get_parameter("max_height_m").as_double();

    rviz_frame_ = this->get_parameter("rviz_frame").as_string();

    lidar_topic_      = this->get_parameter("lidar_topic").as_string();
    yolo_right_topic_ = this->get_parameter("yolo_right_topic").as_string();
    yolo_left_topic_  = this->get_parameter("yolo_left_topic").as_string();
    fusion_topic_     = this->get_parameter("fusion_topic").as_string();
    rviz_topic_       = this->get_parameter("rviz_topic").as_string();

    // -------- Subscriptions --------
    wamv_sub_ = this->create_subscription<lidar_msgs::msg::BuoyDetected>(
        ("/" + wamv_name_ + "/" + lidar_topic_),
        10,
        std::bind(&SensorFusionNode::wamv_callback, this, std::placeholders::_1));

    yolo_right_sub_ = this->create_subscription<yolov26_msgs::msg::YoloDetection>(
        yolo_right_topic_,
        10,
        std::bind(&SensorFusionNode::yolo_right_callback, this, std::placeholders::_1));

    yolo_left_sub_ = this->create_subscription<yolov26_msgs::msg::YoloDetection>(
        yolo_left_topic_,
        10,
        std::bind(&SensorFusionNode::yolo_left_callback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        ("/" + wamv_name_ + "/p3d_wamv_ned"),
        10,
        std::bind(&SensorFusionNode::odom_callback, this, std::placeholders::_1));

    // -------- Publishers --------
    fusion_pub_ = this->create_publisher<fau_msgs::msg::ObjectPositionArray>(
        ("/" + wamv_name_ + "/" + fusion_topic_),
        10);

    rviz_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
        ("/" + wamv_name_ + "/" + rviz_topic_),
        10);

    RCLCPP_INFO(this->get_logger(),
                "SensorFusionNode dual started for %s | lidar=/%s/%s | yoloR=%s | yoloL=%s",
                wamv_name_.c_str(),
                wamv_name_.c_str(),
                lidar_topic_.c_str(),
                yolo_right_topic_.c_str(),
                yolo_left_topic_.c_str());
  }

private:
  // ---------------- Utils ----------------
  static inline double deg2rad(double d) { return d * M_PI / 180.0; }

  static inline double wrap_pi(double a)
  {
    return std::atan2(std::sin(a), std::cos(a));
  }

  inline bool is_buoy_class(const std::string &cls) const
  {
    return (cls == "red_buoy" || cls == "green_buoy" || cls == "black_buoy" || cls == "yellow_buoy");
  }

  // Pinhole model: convert pixel x_center -> camera bearing (rad)
  inline double pixel_to_angle(double x_px, double image_width, double hfov_rad) const
  {
    const double cx = image_width * 0.5;
    const double half = hfov_rad * 0.5;
    const double fx = cx / std::tan(half);
    return std::atan((x_px - cx) / fx);
  }

  // Convert camera bearing -> predicted pixel
  inline double angle_to_pixel(double ang, double image_width, double hfov_rad) const
  {
    const double cx = image_width * 0.5;
    const double half = hfov_rad * 0.5;
    const double fx = cx / std::tan(half);
    return cx + fx * std::tan(ang);
  }

  // ---------------- ODOM: global NED pose ----------------
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    x_usv_NED_ = msg->pose.pose.position.x;
    y_usv_NED_ = msg->pose.pose.position.y;

    const auto &q = msg->pose.pose.orientation;
    tf2::Quaternion quat(q.x, q.y, q.z, q.w);
    tf2::Matrix3x3 m(quat);

    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    psi_usv_NED_ = yaw;
    have_odom_ = true;
  }

  // Convert BODY (sensor) point -> GLOBAL NED using odom pose/yaw.
  inline void body_to_global_ned(double xb, double yb, double zb,
                                 double &xg, double &yg, double &zg) const
  {
    // body NWU -> body NED
    const double x_body = xb;
    const double y_body = -yb;

    const double c = std::cos(psi_usv_NED_);
    const double s = std::sin(psi_usv_NED_);

    const double dN = c * x_body - s * y_body;
    const double dE = s * x_body + c * y_body;

    xg = x_usv_NED_ + dN;
    yg = y_usv_NED_ + dE;
    zg = zb;
  }

  // ---------------- LIDAR BUOYS ----------------
  void wamv_callback(const lidar_msgs::msg::BuoyDetected::SharedPtr msg)
  {
    lidar_names_.clear();
    lidar_x_.clear();
    lidar_y_.clear();
    lidar_z_.clear();

    const size_t n = std::min({msg->name.size(), msg->x.size(), msg->y.size(), msg->z.size()});
    lidar_names_.reserve(n);
    lidar_x_.reserve(n);
    lidar_y_.reserve(n);
    lidar_z_.reserve(n);

    for (size_t i = 0; i < n; ++i)
    {
      const double angle = std::atan2(msg->y[i], msg->x[i]);
      const double dist  = std::hypot(msg->x[i], msg->y[i]);

      // We keep all LiDAR buoys; later we will filter by right/left wedges per camera.
      // Height filtering here:
      if (msg->z[i] < min_height_m_ || msg->z[i] > max_height_m_)
        continue;

      lidar_names_.push_back(msg->name[i]);
      lidar_x_.push_back(msg->x[i]);
      lidar_y_.push_back(msg->y[i]);
      lidar_z_.push_back(msg->z[i]);
      (void)angle;
      (void)dist;
    }
  }

  // ---------------- YOLO callbacks ----------------
  void yolo_right_callback(const yolov26_msgs::msg::YoloDetection::SharedPtr msg)
  {
    yolo_right_ = *msg;
    try_fuse();
  }

  void yolo_left_callback(const yolov26_msgs::msg::YoloDetection::SharedPtr msg)
  {
    yolo_left_ = *msg;
    try_fuse();
  }

  // Fuse when we have odom + at least one yolo message (either side) + lidar points
  void try_fuse()
  {
    if (!have_odom_)
      return;
    if (lidar_x_.empty())
      return;

    // Build fused output from both cameras
    fused_.object_names.clear();
    fused_.x_object.clear();
    fused_.y_object.clear();
    fused_.z_object.clear();

    // Fuse right camera detections against right wedge
    fuse_one_camera(yolo_right_,
                    /*is_left=*/false,
                    right_min_angle_rad_, right_max_angle_rad_, right_max_length_m_,
                    camera_hfov_right_rad_, mirror_right_, yaw_offset_right_rad_);

    // Fuse left camera detections against left wedge
    fuse_one_camera(yolo_left_,
                    /*is_left=*/true,
                    left_min_angle_rad_, left_max_angle_rad_, left_max_length_m_,
                    camera_hfov_left_rad_, mirror_left_, yaw_offset_left_rad_);

    if (!fused_.object_names.empty())
    {
      fusion_pub_->publish(fused_);
      publish_to_rviz(fused_);
    }
  }

  // Core: match YOLO detections to LiDAR points with unique assignment (one LiDAR -> one YOLO)
  void fuse_one_camera(const yolov26_msgs::msg::YoloDetection &yolo,
                       bool is_left_camera,
                       double min_ang, double max_ang, double max_len,
                       double cam_hfov_rad,
                       bool mirror,
                       double cam_yaw_offset_rad)
  {
    if (yolo.class_name.empty())
      return;

    const double W  = image_width_px_;
    const double cx = W * 0.5;

    // Prepare candidate LiDAR indices in this camera wedge
    std::vector<int> lidar_idx;
    lidar_idx.reserve(lidar_x_.size());

    for (size_t j = 0; j < lidar_x_.size(); ++j)
    {
      const double xb = lidar_x_[j];
      const double yb = lidar_y_[j];
      const double ang = std::atan2(yb, xb);
      const double dist = std::hypot(xb, yb);

      if (ang < min_ang || ang > max_ang)
        continue;
      if (dist > max_len)
        continue;

      lidar_idx.push_back(static_cast<int>(j));
    }

    if (lidar_idx.empty())
      return;

    // We'll enforce unique assignment of LiDAR points to YOLO detections
    std::vector<bool> lidar_used(lidar_x_.size(), false);

    // For each YOLO detection, select best LiDAR point (not used yet)
    for (size_t i = 0; i < yolo.class_name.size(); ++i)
    {
      const std::string &cls = yolo.class_name[i];
      if (!is_buoy_class(cls))
        continue;

      // YOLO x_center -> bearing
      const double raw_px = yolo.x_center[i];
      const double px = mirror ? (W - raw_px) : raw_px;
      const double yolo_ang = pixel_to_angle(px, W, cam_hfov_rad);

      int best_j = -1;
      double best_err = std::numeric_limits<double>::infinity();
      double best_cam_ang = 0.0;

      for (int j : lidar_idx)
      {
        if (lidar_used[j])
          continue;

        const double xb = lidar_x_[j];
        const double yb = lidar_y_[j];

        // Predict what angle the LiDAR point should appear at in this camera
        // Right camera: assume camera forward ≈ body +X, right ≈ body -Y (depends on your mounting).
        // Left  camera: your proven model: camera forward ≈ body +Y, right ≈ body -X.
        double cam_ang = 0.0;

        if (is_left_camera)
        {
          // left camera model from your working snippet
          const double x_cam =  yb;   // forward
          const double y_cam = -xb;   // right
          cam_ang = std::atan2(y_cam, x_cam);
        }
        else
        {
          // right camera default model (forward +X, right -Y)
          // If your right camera is mounted differently, adjust here.
          const double x_cam =  xb;   // forward
          const double y_cam = -yb;   // right
          cam_ang = std::atan2(y_cam, x_cam);
        }

        cam_ang = wrap_pi(cam_ang + cam_yaw_offset_rad);

        const double err = std::fabs(wrap_pi(yolo_ang - cam_ang));

        if (err < best_err)
        {
          best_err = err;
          best_j = j;
          best_cam_ang = cam_ang;
        }
      }

      if (best_j < 0)
        continue;

      // Reject if mismatch too large
      if (best_err > max_match_err_rad_)
        continue;

      // Lock this LiDAR point so it can't become multiple colors in this frame
      lidar_used[best_j] = true;

      // Convert accepted LiDAR point -> GLOBAL NED
      double xg, yg, zg;
      body_to_global_ned(lidar_x_[best_j], lidar_y_[best_j], lidar_z_[best_j], xg, yg, zg);

      fused_.object_names.push_back(cls);
      fused_.x_object.push_back(xg);
      fused_.y_object.push_back(yg);
      fused_.z_object.push_back(zg);

      // Optional debug:
      // double pred_px = angle_to_pixel(best_cam_ang, W, cam_hfov_rad);
      // RCLCPP_INFO(this->get_logger(), "[%s] %s match err=%.1f deg raw_px=%.1f px=%.1f pred_px=%.1f",
      //             is_left_camera ? "L" : "R",
      //             cls.c_str(),
      //             best_err * 180.0 / M_PI,
      //             raw_px, px, pred_px);
      (void)cx;
      (void)best_cam_ang;
    }
  }

  // ---------------- RVIZ ----------------
  void publish_to_rviz(const fau_msgs::msg::ObjectPositionArray &fusion_data)
  {
    geometry_msgs::msg::PoseArray pose_array_msg;
    pose_array_msg.header.stamp = this->get_clock()->now();
    pose_array_msg.header.frame_id = rviz_frame_;   // ✅ GLOBAL

    pose_array_msg.poses.clear();
    pose_array_msg.poses.reserve(fusion_data.x_object.size());

    for (size_t i = 0; i < fusion_data.x_object.size(); ++i)
    {
      geometry_msgs::msg::Pose pose;
      pose.position.x = fusion_data.x_object[i];
      pose.position.y = fusion_data.y_object[i];
      pose.position.z = fusion_data.z_object[i];
      pose.orientation.w = 1.0;
      pose_array_msg.poses.push_back(pose);
    }

    rviz_pub_->publish(pose_array_msg);
  }

private:
  // Params
  std::string wamv_name_;

  double image_width_px_{1280.0};

  double camera_hfov_right_rad_{deg2rad(60.0)};
  double camera_hfov_left_rad_{deg2rad(60.0)};

  bool mirror_right_{false};
  bool mirror_left_{true};

  double yaw_offset_right_rad_{0.0};
  double yaw_offset_left_rad_{deg2rad(7.8)};

  double max_match_err_rad_{deg2rad(12.0)};

  double right_min_angle_rad_{deg2rad(-45.0)};
  double right_max_angle_rad_{deg2rad(45.0)};
  double right_max_length_m_{30.0};

  double left_min_angle_rad_{deg2rad(45.0)};
  double left_max_angle_rad_{deg2rad(135.0)};
  double left_max_length_m_{10.0};

  double min_height_m_{-3.0};
  double max_height_m_{3.0};

  std::string rviz_frame_{"odom"};

  std::string lidar_topic_{"vision/output/buoy_detected"};
  std::string yolo_right_topic_{"/yolov26/detections"};
  std::string yolo_left_topic_{"/yolov26/detections/left"};
  std::string fusion_topic_{"vision/output/fusion"};
  std::string rviz_topic_{"vision/output/rviz_fusion"};

  // Subs/Pubs
  rclcpp::Subscription<lidar_msgs::msg::BuoyDetected>::SharedPtr wamv_sub_;
  rclcpp::Subscription<yolov26_msgs::msg::YoloDetection>::SharedPtr yolo_right_sub_;
  rclcpp::Subscription<yolov26_msgs::msg::YoloDetection>::SharedPtr yolo_left_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  rclcpp::Publisher<fau_msgs::msg::ObjectPositionArray>::SharedPtr fusion_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr rviz_pub_;

  // LiDAR data (body frame)
  std::vector<std::string> lidar_names_;
  std::vector<double> lidar_x_, lidar_y_, lidar_z_;

  // Latest YOLO messages
  yolov26_msgs::msg::YoloDetection yolo_right_;
  yolov26_msgs::msg::YoloDetection yolo_left_;

  // Fused output
  fau_msgs::msg::ObjectPositionArray fused_;

  // Odom state (GLOBAL NED)
  bool have_odom_{false};
  double x_usv_NED_{0.0};
  double y_usv_NED_{0.0};
  double psi_usv_NED_{0.0};
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorFusionNode>());
  rclcpp::shutdown();
  return 0;
}
