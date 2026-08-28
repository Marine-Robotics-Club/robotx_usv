#include <geometry_msgs/msg/pose2_d.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include <message_filters/subscriber.h>
#include <nav_msgs/msg/odometry.hpp>
#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "time.h"
#include <sstream>
#include <iostream>
#include "math.h"
#include "stdio.h"
#include <tf2/LinearMath/Quaternion.h>
//#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>  // this is the missing one!
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include "std_msgs/msg/float32.hpp"
#include "rclcpp/clock.hpp"
#include <deque>

#include <string>

#define PI 3.14159265
using std::placeholders::_1;
//using namespace std::chrono_literals;

using std::cout;
using std::endl;
using namespace std;

typedef struct {
  double north;
  double east;
} NED_struct;

class state_to_ned : public rclcpp::Node
{
 public:
 
 state_to_ned();
  
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr posit_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr quat_sub;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr heading_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr rpy_sub;
  
  void position_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg);
  void quaternions_callback(
      const sensor_msgs::msg::Imu::SharedPtr msg);
  void compute_enu2ned_mat();
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void heading_callback(const std_msgs::msg::Float32::SharedPtr msg);
  void rpy_callback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);


  int loop();
  double twopiwrap(double angle);
 

    // Publish static transforms once at startup
    //this->make_transforms(transformation);

 private:
  // MessageFilter for simulation-odometry data
  //tf2_ros::Buffer buff_sim;
  //tf2_ros::TransformListener listener_sim;
  //message_filters::Subscriber<nav_msgs::msg::Odometry> sim_odom_enu_sub;
  //tf2_ros::MessageFilter<nav_msgs::msg::Odometry> filter_sim;

  // MessageFilter for real-quaternions data
  // tf2_ros::Buffer buff_quat
  // tf2_ros::TransformListener listener_quat;
  // message_filters::Subscriber<geometry_msgs::QuaternionStamped> quat_sub;
  // tf2_ros::MessageFilter<geometry_msgs::QuaternionStamped> filter_quat;

  geometry_msgs::msg::Pose2D the_pose_ned,the_goal_pose_ned;
  geometry_msgs::msg::PoseArray the_waypoint_pose_ned;
  geometry_msgs::msg::Twist the_twist_ned;
  geometry_msgs::msg::Vector3 the_bfvelo_ned;
  nav_msgs::msg::Odometry the_odom_ned, the_goal_odom_ned;
  NED_struct nedPoint;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time start_time, current_time, last_time;  // creates time variables
  rclcpp::TimerBase::SharedPtr timer;
 
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr baselink_pose_ned_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr baselink_odom_ned_pub;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr baselink_bfvelo_ned_pub;
  

  // Some functions here
  void send_static_transform(double x, double y, double z, double roll,
                             double pitch, double yaw,
                             const std::string frame_id,
                             const std::string child_frame_id,
                             bool intrinsic_rot = true);
  void send_transform(double x, double y, double z, double roll, double pitch,
                      double yaw, const std::string frame_id,
                      const std::string child_frame_id,
                      bool intrinsic_rot = false);
  bool send_all_transforms();
  void publish_all_messages();
  double moving_average(const std::deque<double>& buffer);
  NED_struct Geo2NED(double lat, double lon, double latref, double lonref);

  double the_yaw_enu;
  double the_yaw_ned;
  float lidarXOffset;
  float lidarYOffset;
  float lidarZOffset;
  float cameraXOffset;
  float cameraYOffset;
  float cameraZOffset;
  bool use_simulation = false;  // true for VRX simulations
  bool gotPosit = false;
  bool gotBFVelo = false;
  bool gotTwistys = false;
  bool gotQuat =
      false;  // these two are true in order to make message_filter work
  bool gotSim =
      false;  // these two are true in order to make message_filter work
  bool use_sim = false;
  bool initialized = false;
  bool clock_ready = false;
  bool gotRef;
  tf2::Matrix3x3 Rot;

  rclcpp::Time prev_time;

  // Pose history for velocity/acceleration


  const size_t filter_window = 10;  // or 5 for faster response

  std::deque<double> vx_buf, vy_buf, vyaw_buf;
  std::deque<double> ax_buf, ay_buf, ayaw_buf;

  geometry_msgs::msg::Pose2D prev_pose_ned;
  double prev_altitude = 0.0;

  double prev_yaw_ned = 0.0;
  rclcpp::Time prev_time_gps, prev_time_imu;

  double prev_vx = 0.0, prev_vy = 0.0, prev_vyaw = 0.0;
  bool first_gps = true, first_imu = true;

  // Buffers for moving average
  std::deque<double> vz_buf;
  std::deque<double> wx_buf;
  std::deque<double> wy_buf;
  std::deque<double> wz_buf;

  // Previous values for derivative estimation
  double prev_vz;
  double prev_vroll;
  double prev_vpitch;
  double prev_ayaw;

  bool use_yaw_offset_ = true;          // enable offset logic
  bool have_yaw_offset_ = false;
  double yaw_offset_ = 0.0;

  // Mode A: fixed reference
  double desired_zero_heading_deg_ = 0.0; // only used if calibrate_north_now_ == false

  // Mode B: set North once at startup (recommended for field quick calibration)
  bool calibrate_north_now_ = false;     // set true => when node starts, face North and it locks heading=0

  double gps_heading_rad_ = 0.0;
  bool   gps_heading_valid_ = false;
  double gps_speed_mps_ = 0.0;
  double min_speed_for_heading_ = 0.3;   // tune (0.3–1.0 typical)

  double gps_yaw_ned_ = 0.0;     // radians, NED frame
  bool   gps_yaw_valid_ = false;

  // Compass heading state
bool gotHeading = false;
rclcpp::Time prev_time_heading;
double prev_heading_ned = 0.0;

// Estimated yaw rate from compass
double yaw_rate_ned_ = 0.0;
double yaw_accel_ned_ = 0.0;

// Estimated linear acceleration from GPS velocity
bool gotGpsVelocity = false;
double prev_vx_ned_ = 0.0;
double prev_vy_ned_ = 0.0;

double ax_ned_ = 0.0;
double ay_ned_ = 0.0;

double ax_body_ = 0.0;
double ay_body_ = 0.0;






  
  
};

