#include <asv_state/state_to_ned_vrx.h>
  double latRef;
  double lonRef;
/* Program written by Xavier Vicent for his Dissertation
 * May 20, 2025.*/

 double wrap_to_pi(double angle)
{
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }

    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }

    return angle;
}

state_to_ned::state_to_ned()
: Node("state_to_ned_1",
       rclcpp::NodeOptions()
           .allow_undeclared_parameters(true)
           .automatically_declare_parameters_from_overrides(true)
           .parameter_overrides({rclcpp::Parameter("use_sim_time", true)}))
{
  // Simulation?
  bool use_sim_time = this->get_parameter("use_sim_time").as_bool();
  this->set_parameter(rclcpp::Parameter("use_sim_time", false));  // Activate sim time

    // Now you can safely read your own parameter
  use_sim = get_parameter("Simulation").as_bool();

  if (use_sim) {
    // Wait for simulation time to be non-zero before using it
    RCLCPP_INFO(this->get_logger(), "Waiting for /clock to become active...");
    while (rclcpp::ok() && this->get_clock()->now().seconds() == 0.0) {
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
    RCLCPP_INFO(this->get_logger(), "/clock is now active.");
  }

  // Frame conversion matrix
  compute_enu2ned_mat();
  if (!this->has_parameter("asv")) {
  this->declare_parameter("asv", "asv1");
  }

  std::string asv_name = this->get_parameter("asv").as_string();


  // Subscribers
  if (use_sim)
  {
    quat_sub =  this->create_subscription<sensor_msgs::msg::Imu>(		
     "/wamv/sensors/imu/imu/data", 10, std::bind(&state_to_ned::quaternions_callback, this, _1));
      
    posit_sub = this->create_subscription<sensor_msgs::msg::NavSatFix>(		
     "/wamv/sensors/gps/gps/fix", 10, std::bind(&state_to_ned::position_callback, this, _1));

   // this->set_parameter(rclcpp::Parameter("use_sim_time", true));
  }
  else 
  {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    /*quat_sub =  this->create_subscription<sensor_msgs::msg::Imu>(		
    "/" + asv_name + "/imu/data", qos, std::bind(&state_to_ned::quaternions_callback, this, _1));*/
    
    /*quat_sub =  this->create_subscription<sensor_msgs::msg::Imu>(		
    "/sbg_legacy/imu/data", qos, std::bind(&state_to_ned::quaternions_callback, this, _1));*/
    rpy_sub = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
    "/sbg_legacy/rpy",
    qos,
    std::bind(&state_to_ned::rpy_callback, this, std::placeholders::_1)
    );
      
    posit_sub = this->create_subscription<sensor_msgs::msg::NavSatFix>(		
    "/sbg_legacy/gps/fix", qos, std::bind(&state_to_ned::position_callback, this, _1));

    /*heading_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "/asv/heading",
        qos,
        std::bind(&state_to_ned::heading_callback, this, std::placeholders::_1)
    );*/

    

    //this->set_parameter(rclcpp::Parameter("use_sim_time", false));
  }
        
  // Publishers
  baselink_pose_ned_pub =
      this->create_publisher<geometry_msgs::msg::Pose2D>( "/" + asv_name + "/vehicle_pose", 10);
  baselink_odom_ned_pub =
      this->create_publisher<nav_msgs::msg::Odometry>( "/" + asv_name + "/p3d_wamv_ned", 10);
  baselink_bfvelo_ned_pub =
      this->create_publisher<geometry_msgs::msg::Vector3>( "/" + asv_name + "/bfvelo_ned", 10);
  timer = this->create_wall_timer(50ms, std::bind(&state_to_ned::loop, this));
      
  // Initialize odom_ned odometry message
  the_odom_ned.header.frame_id = "ned_origin";
  the_odom_ned.child_frame_id = "base_link_ned";
  // read params
  //this->declare_parameter("latRef", rclcpp::ParameterValue(0.0));
  //this->declare_parameter("lonRef", rclcpp::ParameterValue(0.0));

  //this->get_parameter("latRef",latRef);
  //this->get_parameter("lonRef",lonRef);
  
  
  
  latRef = get_parameter("latRef").as_double();
  lonRef = get_parameter("lonRef").as_double();
  gotRef = false;  // <-- Add this as a class member bool

  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "The latRef is %f and the lonRef is %f ", latRef, lonRef);

  prev_pose_ned.x = 0.0;
  prev_pose_ned.y = 0.0;
  prev_yaw_ned = 0.0;
  prev_vx = prev_vy = prev_vyaw = 0.0;
  prev_altitude = 0.0;
  prev_ayaw = 0.0;

  state_to_ned::loop();
}


void state_to_ned::compute_enu2ned_mat() {
  tf2::Matrix3x3 Rx(1, 0, 0, 0, cos(M_PI), sin(M_PI), 0, -sin(M_PI), cos(M_PI));
  tf2::Matrix3x3 Rz(cos(M_PI / 2), -sin(M_PI / 2), 0, sin(M_PI / 2),
                    cos(M_PI / 2), 0, 0, 0, 1);
  Rot = Rz * Rx;
}

void state_to_ned::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  /*
    the_odom_ned.twist.twist.linear.x = msg->twist.twist.linear.x;
  the_odom_ned.twist.twist.linear.y = msg->twist.twist.linear.y;
  the_odom_ned.twist.twist.linear.z = msg->twist.twist.linear.z;
  the_odom_ned.twist.twist.angular.x = msg->twist.twist.angular.x;
  the_odom_ned.twist.twist.angular.y = msg->twist.twist.angular.y;
  the_odom_ned.twist.twist.angular.z = msg->twist.twist.angular.z;
  */

  // Optional: Store or process these values
  gotTwistys = true;
}

void state_to_ned::rpy_callback(
    const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
{
    rclcpp::Time current_time = this->get_clock()->now();

    // SBG RPY topic assumption:
    // x = roll
    // y = pitch
    // z = yaw
    //
    // Usually this is already in radians. If your echo shows degrees,
    // convert with * M_PI / 180.0.
    double roll = msg->vector.x;
    double pitch = msg->vector.y;
    double yaw = msg->vector.z;

    // If needed, enable this instead:
    // roll  = roll  * M_PI / 180.0;
    // pitch = pitch * M_PI / 180.0;
    // yaw   = yaw   * M_PI / 180.0;

    // Wrap yaw
    yaw = twopiwrap(yaw);

    // Store heading in your NED pose
    the_pose_ned.theta = yaw;
    the_yaw_ned = yaw;

    // Create odometry orientation from RPY
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    q.normalize();

    the_odom_ned.pose.pose.orientation = tf2::toMsg(q);

    // Compute yaw rate from yaw derivative
    if (gotHeading) {
        double dt = (current_time - prev_time_heading).seconds();

        if (dt > 0.01) {
            double dtheta = wrap_to_pi(yaw - prev_heading_ned);
            double yaw_rate = dtheta / dt;

            vyaw_buf.push_back(yaw_rate);
            if (vyaw_buf.size() > filter_window) {
                vyaw_buf.pop_front();
            }

            yaw_rate_ned_ = moving_average(vyaw_buf);

            the_odom_ned.twist.twist.angular.x = 0.0;
            the_odom_ned.twist.twist.angular.y = 0.0;
            the_odom_ned.twist.twist.angular.z = yaw_rate_ned_;

            double ayaw = (yaw_rate_ned_ - prev_vyaw) / dt;

            ayaw_buf.push_back(ayaw);
            if (ayaw_buf.size() > filter_window) {
                ayaw_buf.pop_front();
            }

            yaw_accel_ned_ = moving_average(ayaw_buf);

            prev_vyaw = yaw_rate_ned_;
        }
    }

    prev_heading_ned = yaw;
    prev_time_heading = current_time;

    // These are important for your existing loop logic
    gotHeading = true;
    gotQuat = true;
}

void state_to_ned::heading_callback(const std_msgs::msg::Float32::SharedPtr msg)
{
    rclcpp::Time current_time = this->get_clock()->now();

    // ----------------------------------------------------
    // 1) Compass heading
    // Assumption:
    //   msg->data is in degrees
    //   0 deg = North
    //   90 deg = East
    // This matches your NED-style heading convention.
    // ----------------------------------------------------
    double heading_rad = static_cast<double>(msg->data);
    double heading_deg = heading_rad * 180.0 / M_PI;

    // Wrap to [0, 2*pi)
    heading_rad = twopiwrap(heading_rad);

    // Store heading in Pose2D
    the_pose_ned.theta = heading_rad;
    the_yaw_ned = heading_rad;

    // ----------------------------------------------------
    // 2) Create odometry orientation from compass heading
    // In your NED convention:
    //   x = North
    //   y = East
    //   theta = 0 means North
    //   theta = +90 deg means East
    // ----------------------------------------------------
    tf2::Quaternion q_ned;
    q_ned.setRPY(0.0, 0.0, heading_rad);
    q_ned.normalize();

    the_odom_ned.pose.pose.orientation = tf2::toMsg(q_ned);

    // ----------------------------------------------------
    // 3) Compute angular velocity from heading derivative
    // yaw_rate = d(theta) / dt
    // ----------------------------------------------------
    if (gotHeading) {
        double dt = (current_time - prev_time_heading).seconds();

        if (dt > 0.01) {
            // Use wrapped difference to avoid 359 -> 0 deg spike
            double dtheta = wrap_to_pi(heading_rad - prev_heading_ned);

            double wz_ned = dtheta / dt;

            // Buffer yaw rate
            vyaw_buf.push_back(wz_ned);
            if (vyaw_buf.size() > filter_window) {
                vyaw_buf.pop_front();
            }

            auto mean = [](const std::deque<double>& d) -> double {
                if (d.empty()) return 0.0;

                double s = 0.0;
                for (double v : d) {
                    s += v;
                }

                return s / static_cast<double>(d.size());
            };

            double wz_f = mean(vyaw_buf);

            yaw_rate_ned_ = wz_f;

            // Store in odometry twist
            the_odom_ned.twist.twist.angular.x = 0.0;
            the_odom_ned.twist.twist.angular.y = 0.0;
            the_odom_ned.twist.twist.angular.z = yaw_rate_ned_;

            // ------------------------------------------------
            // 4) Optional yaw acceleration
            // yaw_accel = d(yaw_rate) / dt
            // ------------------------------------------------
            double ayaw = (yaw_rate_ned_ - prev_vyaw) / dt;

            ayaw_buf.push_back(ayaw);
            if (ayaw_buf.size() > filter_window) {
                ayaw_buf.pop_front();
            }

            yaw_accel_ned_ = mean(ayaw_buf);
            prev_ayaw = yaw_accel_ned_;

            prev_vyaw = yaw_rate_ned_;
        }
    }

    prev_heading_ned = heading_rad;
    prev_time_heading = current_time;
    gotHeading = true;

    /*RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "Compass heading: %.2f deg | yaw_rate: %.4f rad/s | yaw_accel: %.4f rad/s^2",
        heading_deg,
        yaw_rate_ned_,
        yaw_accel_ned_
    );*/
}
void state_to_ned::position_callback(
    const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
  rclcpp::Time current_time = this->get_clock()->now();

  // ----------------------------
  // Convert GPS to NED
  // ----------------------------
  nedPoint = Geo2NED(msg->latitude, msg->longitude, latRef, lonRef);

  the_pose_ned.x = nedPoint.north;  // North
  the_pose_ned.y = nedPoint.east;   // East

  // Also store position in odometry
  the_odom_ned.header.stamp = current_time;
  the_odom_ned.pose.pose.position.x = the_pose_ned.x;
  the_odom_ned.pose.pose.position.y = the_pose_ned.y;
  the_odom_ned.pose.pose.position.z = 0.0;

  if (gotPosit) {
    double dt = (current_time - prev_time_gps).seconds();

    if (dt > 0.01) {

      // ----------------------------
      // Raw velocities in NED
      // ----------------------------
      double vx = (the_pose_ned.x - prev_pose_ned.x) / dt;  // North velocity
      double vy = (the_pose_ned.y - prev_pose_ned.y) / dt;  // East velocity

      vx_buf.push_back(vx);
      vy_buf.push_back(vy);

      if (vx_buf.size() > filter_window) {
        vx_buf.pop_front();
      }

      if (vy_buf.size() > filter_window) {
        vy_buf.pop_front();
      }

      // ----------------------------
      // Mean filter helper
      // ----------------------------
      auto mean = [](const std::deque<double>& d) -> double {
        if (d.empty()) {
          return 0.0;
        }

        double s = 0.0;
        for (double v : d) {
          s += v;
        }

        return s / static_cast<double>(d.size());
      };

      // ----------------------------
      // Filtered velocity in NED
      // ----------------------------
      double vx_f = mean(vx_buf);   // North velocity
      double vy_f = mean(vy_buf);   // East velocity

      // ----------------------------
      // Store linear velocity in odometry
      // ----------------------------
      the_odom_ned.twist.twist.linear.x = vx_f;
      the_odom_ned.twist.twist.linear.y = vy_f;
      the_odom_ned.twist.twist.linear.z = 0.0;

      // ----------------------------
      // Speed magnitude
      // ----------------------------
      double speed = std::hypot(vx_f, vy_f);
      gps_speed_mps_ = speed;

      // ----------------------------
      // Optional GPS course heading
      // This is NOT the compass heading.
      // This is only the direction of motion from GPS.
      // ----------------------------
      constexpr double MIN_SPEED_FOR_HEADING = 0.5;

      if (speed > MIN_SPEED_FOR_HEADING) {
        gps_yaw_ned_ = std::atan2(vy_f, vx_f);  // atan2(East, North)
        gps_yaw_valid_ = true;
      } else {
        gps_yaw_valid_ = false;
      }

      // ----------------------------------------------------
      // Linear acceleration from GPS velocity derivative
      //
      // ax = d(vx) / dt
      // ay = d(vy) / dt
      //
      // This is acceleration in the NED/local navigation frame.
      // ----------------------------------------------------
      if (gotGpsVelocity) {
        double ax_raw = (vx_f - prev_vx_ned_) / dt;
        double ay_raw = (vy_f - prev_vy_ned_) / dt;

        ax_buf.push_back(ax_raw);
        ay_buf.push_back(ay_raw);

        if (ax_buf.size() > filter_window) {
          ax_buf.pop_front();
        }

        if (ay_buf.size() > filter_window) {
          ay_buf.pop_front();
        }

        ax_ned_ = mean(ax_buf);
        ay_ned_ = mean(ay_buf);

        // ------------------------------------------------
        // Rotate NED acceleration into body frame
        //
        // NED:
        //   x = North
        //   y = East
        //
        // Body:
        //   x = forward / surge
        //   y = right / sway
        //
        // the_pose_ned.theta should come from your compass
        // heading_callback(), not from quaternion anymore.
        // ------------------------------------------------
        double psi = the_pose_ned.theta;

        ax_body_ =  ax_ned_ * std::cos(psi) + ay_ned_ * std::sin(psi);
        ay_body_ = -ax_ned_ * std::sin(psi) + ay_ned_ * std::cos(psi);

        /*RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "GPS vel NED: vx=%.3f vy=%.3f speed=%.3f | Accel NED: ax=%.3f ay=%.3f | Body: surge=%.3f sway=%.3f",
            vx_f,
            vy_f,
            speed,
            ax_ned_,
            ay_ned_,
            ax_body_,
            ay_body_
        );*/
      }

      // ----------------------------
      // Store previous filtered velocity
      // ----------------------------
      prev_vx_ned_ = vx_f;
      prev_vy_ned_ = vy_f;
      gotGpsVelocity = true;
    }
  }

  // ----------------------------
  // Update previous GPS state
  // ----------------------------
  prev_pose_ned = the_pose_ned;
  prev_time_gps = current_time;
  gotPosit = true;
}
void state_to_ned::quaternions_callback(
    const sensor_msgs::msg::Imu::SharedPtr msg)
{
    rclcpp::Time current_time = this->get_clock()->now();

    // ----------------------------
    // 1) Convert incoming quaternion (assumed ENU)
    // ----------------------------
    tf2::Quaternion q_enu;
    tf2::convert(msg->orientation, q_enu);
    q_enu.normalize();

    // ----------------------------
    // 2) Define ENU -> NED rotation
    // ----------------------------
    tf2::Quaternion q_rot;
    q_rot.setRPY(M_PI, 0.0, M_PI / 2.0);
    q_rot.normalize();

    // ----------------------------
    // 3) Rotate orientation into NED
    // ----------------------------
    tf2::Quaternion q_ned = q_rot * q_enu;
    q_ned.normalize();

    // ----------------------------
    // 4) Extract yaw in NED and apply constant correction
    // ----------------------------
    tf2::Matrix3x3 m_ned(q_ned);
    double roll_ned, pitch_ned, yaw_ned;
    m_ned.getRPY(roll_ned, pitch_ned, yaw_ned);

    // Always apply constant correction so that North -> 0
    // Raw data: when pointing North you see about -154.7 deg
    // So correction is +154.7 deg
    constexpr double YAW_OFFSET_DEG = 0.0;
    const double yaw_offset = YAW_OFFSET_DEG * M_PI / 180.0;

    // Publish as compass-style heading [0, 2pi)
    yaw_ned = twopiwrap(yaw_ned + yaw_offset);

    // ----------------------------
    // 6) Rotate angular velocity ENU -> NED
    // ----------------------------
    tf2::Vector3 w_enu(msg->angular_velocity.x,
                       msg->angular_velocity.y,
                       msg->angular_velocity.z);

    tf2::Vector3 w_ned = tf2::quatRotate(q_rot, w_enu);

    const double wx_ned = w_ned.x();
    const double wy_ned = w_ned.y();
    const double wz_ned = w_ned.z();

    // ----------------------------
    // 7) Rotate linear acceleration ENU -> NED
    // ----------------------------
    tf2::Vector3 a_enu(msg->linear_acceleration.x,
                       msg->linear_acceleration.y,
                       msg->linear_acceleration.z);

    tf2::Vector3 a_ned = tf2::quatRotate(q_rot, a_enu);

    const double ax_ned = a_ned.x();
    const double ay_ned = a_ned.y();
    const double az_ned = a_ned.z();

    (void)ax_ned; (void)ay_ned; (void)az_ned;

    // ----------------------------
    // 8) Store NED orientation in odom message
    // ----------------------------
    the_odom_ned.pose.pose.orientation = tf2::toMsg(q_ned);

    // ----------------------------
    // 9) Buffer angular velocities (NED-consistent)
    // ----------------------------
    wx_buf.push_back(wx_ned);
    wy_buf.push_back(wy_ned);
    vyaw_buf.push_back(wz_ned);

    if (wx_buf.size() > filter_window) wx_buf.pop_front();
    if (wy_buf.size() > filter_window) wy_buf.pop_front();
    if (vyaw_buf.size() > filter_window) vyaw_buf.pop_front();

    // ----------------------------
    // 10) Compute yaw acceleration (NED-consistent)
    // ----------------------------
    if (gotQuat) {
        const double dt = (current_time - prev_time_imu).seconds();
        if (dt > 0.01) {
            const double ayaw = (wz_ned - prev_vyaw) / dt;
            ayaw_buf.push_back(ayaw);
            if (ayaw_buf.size() > filter_window) ayaw_buf.pop_front();
            prev_ayaw = ayaw;
        }
    }

    // Publish heading (NED compass-style, corrected)
    //the_pose_ned.theta = yaw_ned;
    // (optional if you use this elsewhere)
    // the_yaw_ned = yaw_ned;

    prev_time_imu = current_time;
    gotQuat = true;
    prev_yaw_ned = yaw_ned;
    prev_vyaw = wz_ned;
}


NED_struct state_to_ned::Geo2NED(double lat, double lon, double latref,
                                 double lonref) {
  NED_struct inner_struct;
  double Rne[3][3];
  float Pned[2];
  unsigned long long Rea = 6378137;  // radius of earth in m
  double e = 0.08181919;

  // Convert degrees to radians
  lat = lat * M_PI / 180;
  lon = lon * M_PI / 180;
  latref = latref * M_PI / 180;
  lonref = lonref * M_PI / 180;

  // Imperfect ellipsoid
  double Ne = Rea / (sqrt((1 - pow(e, 2) * pow(sin(lat), 2))));
  //double Neref = Rea / (sqrt((1 - pow(e, 2) * pow(sin(latref), 2))));

  // ECEF coordinates
  double Pe_xe = Ne * cos(lat) * cos(lon);
  double Pe_ye = Ne * cos(lat) * sin(lon);
  double Pe_ze = Ne * (1 - pow(e, 2)) * sin(lat);

  double Peref_xe = Ne * cos(latref) * cos(lonref);
  double Peref_ye = Ne * cos(latref) * sin(lonref);
  double Peref_ze = Ne * (1 - pow(e, 2)) * sin(latref);

  // Rne transformation matrix
  Rne[0][0] = -1 * sin(latref) * cos(lonref);
  Rne[0][1] = -1 * sin(latref) * sin(lonref);
  Rne[0][2] = cos(latref);
  Rne[1][0] = -1 * sin(lonref);
  Rne[1][1] = cos(lonref);
  Rne[1][2] = 0;
  Rne[2][0] = -1 * cos(latref) * cos(lonref);
  Rne[2][1] = -1 * cos(latref) * sin(lonref);
  Rne[2][2] = -1 * sin(latref);

  // NED coordinates
  Pned[0] = (Pe_xe - Peref_xe) * Rne[0][0] + (Pe_ye - Peref_ye) * Rne[0][1] +
            (Pe_ze - Peref_ze) * Rne[0][2];
  Pned[1] = (Pe_xe - Peref_xe) * Rne[1][0] + (Pe_ye - Peref_ye) * Rne[1][1] + 0;

  inner_struct.north = Pned[0];
  inner_struct.east = Pned[1];

  return inner_struct;
}

bool state_to_ned::send_all_transforms() {
  if (gotSim == true || (gotPosit && gotQuat && gotTwistys) == true) {
    //ROS_INFO("(gotSim, gotPosit, gotQuat, gotTwistys) = (%d, %d, %d, %d)", gotSim, gotPosit, gotQuat, gotTwistys);

    
      // origin_enu to origin_ned
      send_static_transform(0.0, 0.0, 0.0, M_PI, 0.0, M_PI / 2, "enu_odom",
                            "ned_origin");
    
    // origin_ned to baselink_ned
    send_transform(the_pose_ned.x, the_pose_ned.y,
                   the_odom_ned.pose.pose.position.z, 0.0, 0.0, the_yaw_ned,
                   "ned_origin", "base_link_ned");
   //goal pose for Task 1
   // origin_ned to baselink_ned
    send_transform(the_goal_pose_ned.x, the_goal_pose_ned.y,
                   the_goal_odom_ned.pose.pose.position.z, 0.0, 0.0, the_yaw_ned,
                   "ned_origin", "base_link_ned");

    // baselink_ned to lidar_nwu
    send_static_transform(lidarXOffset, lidarYOffset, lidarZOffset, M_PI, 0.0,
                          0.0, "base_link_ned", "lidar_nwu");

    // baselink_ned to camera
    send_static_transform(cameraXOffset, cameraYOffset, cameraZOffset, M_PI / 2,
                          0.0, -M_PI / 8, "base_link_ned", "camera_optical");
    gotSim = false;
    gotPosit = false;
    gotQuat = false;
    gotTwistys = false;

    return true;
  } else {
    // ROS_WARN("From send_all_transforms(): Transforms failed to be
    // sent!"); ROS_INFO("(gotSim, gotPosit, gotQuat, gotTwistys) = (%d, %d,
    // %d, %d)", gotSim, gotPosit, gotQuat, gotTwistys);

    return false;
  }
}

void state_to_ned::send_static_transform(double x, double y, double z,
                                         double roll, double pitch, double yaw,
                                         const std::string frame_id,
                                         const std::string child_frame_id,
                                         bool intrinsic_rot) {
  geometry_msgs::msg::TransformStamped static_transformStamped;
 // static tf2_ros::StaticTransformBroadcaster static_broadcaster;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster;
  
  static_broadcaster = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

  static_transformStamped.header.stamp = this->get_clock()->now();
  static_transformStamped.header.frame_id = frame_id;
  static_transformStamped.child_frame_id = child_frame_id;
  static_transformStamped.transform.translation.x = x;
  static_transformStamped.transform.translation.y = y;
  static_transformStamped.transform.translation.z = z;
  tf2::Quaternion quat;
  if (intrinsic_rot == true) {
    quat.setEuler(yaw, pitch, roll);  // intrinsic rotations
  } else {
    quat.setRPY(roll, pitch, yaw);  // extrinsic rotations
  }
  static_transformStamped.transform.rotation.x = quat.x();
  static_transformStamped.transform.rotation.y = quat.y();
  static_transformStamped.transform.rotation.z = quat.z();
  static_transformStamped.transform.rotation.w = quat.w();

  static_broadcaster->sendTransform(static_transformStamped);
}

void state_to_ned::send_transform(double x, double y, double z, double roll,
                                  double pitch, double yaw,
                                  const std::string frame_id,
                                  const std::string child_frame_id,
                                  bool intrinsic_rot) {
  geometry_msgs::msg::TransformStamped transformStamped;
 // static tf2_ros::TransformBroadcaster broadcaster;
  
  std::shared_ptr<tf2_ros::TransformBroadcaster> broadcaster;
  
  broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  transformStamped.header.stamp = this->get_clock()->now();
  transformStamped.header.frame_id = frame_id;
  transformStamped.child_frame_id = child_frame_id;
  transformStamped.transform.translation.x = x;
  transformStamped.transform.translation.y = y;
  transformStamped.transform.translation.z = z;
  tf2::Quaternion quat;
  if (intrinsic_rot == true) {
    quat.setEuler(yaw, pitch, roll);  // intrinsic rotations
  } else {
    quat.setRPY(roll, pitch, yaw);  // extrinsic rotations
  }
  transformStamped.transform.rotation.x = quat.x();
  transformStamped.transform.rotation.y = quat.y();
  transformStamped.transform.rotation.z = quat.z();
  transformStamped.transform.rotation.w = quat.w();

  broadcaster->sendTransform(transformStamped);
}

double state_to_ned::moving_average(const std::deque<double>& buffer) {
  if (buffer.empty()) return 0.0;
  double sum = 0.0;
  for (double val : buffer) sum += val;
  return sum / buffer.size();
}


void state_to_ned::publish_all_messages() {
  
  the_odom_ned.twist.twist.linear.x = moving_average(vx_buf);
  the_odom_ned.twist.twist.linear.y = moving_average(vy_buf);
  the_odom_ned.twist.twist.linear.z = moving_average(vz_buf);
  the_odom_ned.twist.twist.angular.x = moving_average(wx_buf);
  the_odom_ned.twist.twist.angular.y = moving_average(wy_buf);
  the_odom_ned.twist.twist.angular.z = moving_average(vyaw_buf);

  the_odom_ned.twist.covariance[0] = moving_average(ax_buf);
  the_odom_ned.twist.covariance[1] = moving_average(ay_buf);
  the_odom_ned.twist.covariance[2] = moving_average(ayaw_buf);

  baselink_pose_ned_pub->publish(the_pose_ned);
  baselink_bfvelo_ned_pub->publish(the_bfvelo_ned);
  the_odom_ned.header.stamp = this->get_clock()->now();
  baselink_odom_ned_pub->publish(the_odom_ned);
}


int state_to_ned::loop() {
  // Wait until simulation time has started
  if (!clock_ready && this->now().seconds() == 0.0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      2000,
      "Waiting for /clock to start publishing..."
    );
    return 0;
  }

  clock_ready = true;

  // ----------------------------------------------------
  // Initialize using GPS position + compass heading
  // NOT quaternion anymore
  // ----------------------------------------------------
  if (!initialized && gotPosit && gotQuat) {
    prev_time = this->get_clock()->now();
    prev_pose_ned = the_pose_ned;
    prev_yaw_ned = the_pose_ned.theta;
    initialized = true;

    RCLCPP_INFO(
      this->get_logger(),
      "Initialized timing and pose tracking using GPS + compass heading."
    );

    return 0;
  }

  if (!initialized) {
    return 0;
  }

  compute_enu2ned_mat();
  publish_all_messages();

  return 0;
}


double state_to_ned::twopiwrap(double angle) {
  angle = fmod(angle + M_PI, 2.0 * M_PI);
  if (angle < 0)
    angle += 2.0 * M_PI;
  return angle - M_PI;
}

