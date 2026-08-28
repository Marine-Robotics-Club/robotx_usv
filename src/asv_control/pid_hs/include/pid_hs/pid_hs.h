//  Filename:  pid_hs.h
//  Creation Date:  20/09/2025
//  Last Revision Date:  20/08/2026
//  Author [email]:  Xavier Vicent [xavi@aquairyx.com]
//  Organization:  Aquairyx LLC

#ifndef PID_HS_H
#define PID_HS_H

// Standard libraries
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>

// ROS2
#include "rclcpp/rclcpp.hpp"

// Messages
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/twist.hpp"

#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/int64.hpp"
#include "std_msgs/msg/string.hpp"

#include "nav_msgs/msg/odometry.hpp"

#include "fau_msgs/msg/control_effort.hpp"
#include "fau_msgs/msg/teensy_status.hpp"

// TF
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"

#define PI 3.14159265358979323846

using std::placeholders::_1;

namespace FAU
{

class PID_HS : public rclcpp::Node
{
public:
    PID_HS();

private:

    // ============================================================
    // Subscribers
    // ============================================================

    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr velD_sub;
    rclcpp::Subscription<fau_msgs::msg::TeensyStatus>::SharedPtr pcu_status_sub;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr state_des_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr curr_pose_sub;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr control_sub;
    rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr pose_sub;

    // Kept for compatibility/debugging.
    // PID speed feedback is taken from /p3d_wamv_ned.
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr gps_vel_sub;


    // ============================================================
    // Publisher
    // ============================================================

    rclcpp::Publisher<fau_msgs::msg::ControlEffort>::SharedPtr control_pub;

    fau_msgs::msg::ControlEffort control_effort_msg;


    // ============================================================
    // Timer
    // ============================================================

    rclcpp::TimerBase::SharedPtr timer;


    // ============================================================
    // Callbacks
    // ============================================================

    void velD_callback(
        const std_msgs::msg::Float64::SharedPtr msg);

    void pcu_status_callback(
        const fau_msgs::msg::TeensyStatus::SharedPtr msg);

    void desiredStateCallback(
        const std_msgs::msg::Float32MultiArray::SharedPtr msg);

    void control_callback(
        const std_msgs::msg::String::SharedPtr msg);

    void pose_callback(
        const nav_msgs::msg::Odometry::SharedPtr msg);

    void position_callback(
        const geometry_msgs::msg::Pose2D::SharedPtr msg);

    void gps_velocity_callback(
        const geometry_msgs::msg::Twist::SharedPtr msg);


    // ============================================================
    // Functions
    // ============================================================

    void display_gains();
    void parameters_function();
    void loop();

    double wrap_angle(double angle);


    // ============================================================
    // Timing
    // ============================================================

    double dt = 0.1;


    // ============================================================
    // State
    // ============================================================

    bool run_pid = false;
    bool gotGoal = false;
    bool first_control_update = true;

    double psi_usv_NED = 0.0;
    double psi_goal = 0.0;

    // Body-frame surge velocity [m/s]
    double vel_USV = 0.0;

    // Desired speed [m/s]
    double velD = 0.8;


    // ============================================================
    // Errors
    // ============================================================

    double e_psi = 0.0;
    double e_Vel = 0.0;

    double e_psi_prev = 0.0;
    double e_Vel_prev = 0.0;

    double e_psi_total = 0.0;
    double e_Vel_total = 0.0;


    // ============================================================
    // Physical controller outputs
    // ============================================================

    // Surge force [N]
    double T_x = 0.0;

    // Differential drive cannot command sway.
    double T_y = 0.0;

    // Yaw moment [N m]
    double M_z = 0.0;


    // ============================================================
    // Physical control limits
    // Same limits used for the Direct APF comparison
    // ============================================================

    double Tx_max = 50.0;   // [N]
    double Mz_max = 45.0;   // [N m]


    // ============================================================
    // PID gains
    // ============================================================

    // Speed PID -> Tx [N]
    double Kp_v = 60.0;
    double Ki_v = 0.0;
    double Kd_v = 0.0;

    // Heading PID -> Mz [N m]
    double Kp_psi = 30.0;
    double Ki_psi = 0.0;
    double Kd_psi = 0.0;
};

}  // namespace FAU

#endif