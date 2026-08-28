#include "pid_hs/pid_hs.h"

std::string controller = "HS";


FAU::PID_HS::PID_HS()
: Node("pid_hs_node")
{
    std::string asv_name;

    // ============================================================
    // Parameters
    // ============================================================

    this->declare_parameter("asv", "asv");

    // Loop timing
    this->declare_parameter("dt", 0.1);

    // Desired speed
    this->declare_parameter("velD", 0.8);

    // ------------------------------------------------------------
    // SPEED PID
    //
    // Controller output:
    //      Tx [N]
    // ------------------------------------------------------------

    this->declare_parameter("kp_v", 60.0);
    this->declare_parameter("ki_v", 0.0);
    this->declare_parameter("kd_v", 0.0);

    // ------------------------------------------------------------
    // HEADING PID
    //
    // Controller output:
    //      Mz [N m]
    // ------------------------------------------------------------

    this->declare_parameter("kp_psi", 30.0);
    this->declare_parameter("ki_psi", 0.0);
    this->declare_parameter("kd_psi", 0.0);

    // ------------------------------------------------------------
    // Physical output limits
    //
    // Match Direct APF controller limits.
    // ------------------------------------------------------------

    this->declare_parameter("Tx_max", 50.0);
    this->declare_parameter("Mz_max", 45.0);

    // Read ASV namespace
    this->get_parameter("asv", asv_name);

    // Read all controller parameters
    parameters_function();


    // ============================================================
    // Subscribers
    // ============================================================

    pcu_status_sub =
        this->create_subscription<fau_msgs::msg::TeensyStatus>(
            "/" + asv_name + "/teensy_status",
            10,
            std::bind(
                &PID_HS::pcu_status_callback,
                this,
                _1));


    state_des_sub =
        this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/" + asv_name + "/traj_desired_state",
            10,
            std::bind(
                &PID_HS::desiredStateCallback,
                this,
                std::placeholders::_1));


    curr_pose_sub =
        this->create_subscription<nav_msgs::msg::Odometry>(
            "/" + asv_name + "/p3d_wamv_ned",
            10,
            std::bind(
                &PID_HS::pose_callback,
                this,
                std::placeholders::_1));


    velD_sub =
        this->create_subscription<std_msgs::msg::Float64>(
            "/" + asv_name + "/velD",
            10,
            std::bind(
                &PID_HS::velD_callback,
                this,
                _1));


    control_sub =
        this->create_subscription<std_msgs::msg::String>(
            "/" + asv_name + "/controller",
            10,
            std::bind(
                &PID_HS::control_callback,
                this,
                _1));


    pose_sub =
        this->create_subscription<geometry_msgs::msg::Pose2D>(
            "/" + asv_name + "/vehicle_pose",
            10,
            std::bind(
                &PID_HS::position_callback,
                this,
                std::placeholders::_1));


    // Kept so the rest of your stack does not have to change.
    // This topic is NOT used for PID speed feedback anymore.
    gps_vel_sub =
        this->create_subscription<geometry_msgs::msg::Twist>(
            "/" + asv_name + "/gps/velocity",
            rclcpp::SensorDataQoS(),
            std::bind(
                &PID_HS::gps_velocity_callback,
                this,
                std::placeholders::_1));


    // ============================================================
    // Publisher
    // ============================================================

    control_pub =
        this->create_publisher<fau_msgs::msg::ControlEffort>(
            "/" + asv_name + "/control_effort",
            10);


    // ============================================================
    // Timer
    // ============================================================

    auto period =
        std::chrono::duration<double>(dt);

    auto period_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            period);

    timer =
        this->create_wall_timer(
            period_ms,
            std::bind(
                &PID_HS::loop,
                this));


    // Preserve your previous behavior.
    // Teensy status callback will subsequently enable/disable it.
    run_pid = true;


    RCLCPP_INFO(
        this->get_logger(),
        "PID Heading-Speed controller started.\n"
        "  Outputs are PHYSICAL control effort:\n"
        "  Tx [N], Ty [N], Mz [N m]\n"
        "  dt = %.3f s\n"
        "  velD = %.2f m/s\n"
        "  Speed gains: Kp=%.2f Ki=%.2f Kd=%.2f\n"
        "  Heading gains: Kp=%.2f Ki=%.2f Kd=%.2f\n"
        "  Tx_max = %.2f N\n"
        "  Mz_max = %.2f N m",
        dt,
        velD,
        Kp_v,
        Ki_v,
        Kd_v,
        Kp_psi,
        Ki_psi,
        Kd_psi,
        Tx_max,
        Mz_max);
}


// ============================================================
// Parameter loading
// ============================================================

void FAU::PID_HS::parameters_function()
{
    Kp_psi = this->get_parameter("kp_psi").as_double();
    Ki_psi = this->get_parameter("ki_psi").as_double();
    Kd_psi = this->get_parameter("kd_psi").as_double();

    Kp_v = this->get_parameter("kp_v").as_double();
    Ki_v = this->get_parameter("ki_v").as_double();
    Kd_v = this->get_parameter("kd_v").as_double();

    velD = this->get_parameter("velD").as_double();

    dt = this->get_parameter("dt").as_double();

    Tx_max = this->get_parameter("Tx_max").as_double();
    Mz_max = this->get_parameter("Mz_max").as_double();

    if (dt <= 0.0)
    {
        RCLCPP_WARN(
            this->get_logger(),
            "Invalid dt. Resetting to 0.1 s.");

        dt = 0.1;
    }

    Tx_max = std::abs(Tx_max);
    Mz_max = std::abs(Mz_max);
}


// ============================================================
// Utility
// ============================================================

double FAU::PID_HS::wrap_angle(double angle)
{
    while (angle > PI)
    {
        angle -= 2.0 * PI;
    }

    while (angle < -PI)
    {
        angle += 2.0 * PI;
    }

    return angle;
}


// ============================================================
// Desired speed callback
// ============================================================

void FAU::PID_HS::velD_callback(
    const std_msgs::msg::Float64::SharedPtr msg)
{
    velD = msg->data;
}


// ============================================================
// Controller selector callback
// ============================================================

void FAU::PID_HS::control_callback(
    const std_msgs::msg::String::SharedPtr msg)
{
    controller = msg->data;
}


// ============================================================
// GPS velocity callback
// ============================================================

void FAU::PID_HS::gps_velocity_callback(
    const geometry_msgs::msg::Twist::SharedPtr msg)
{
    // Intentionally not used for the PID speed feedback.
    //
    // Previously:
    //
    //     vel_USV = msg->linear.x;
    //
    // This was only the North/world component of velocity and therefore
    // was not the vessel's surge velocity when the boat changed heading.
    //
    // Body-frame surge is now computed from /p3d_wamv_ned in pose_callback().

    (void)msg;
}


// ============================================================
// Teensy / safety callback
// ============================================================

void FAU::PID_HS::pcu_status_callback(
    const fau_msgs::msg::TeensyStatus::SharedPtr msg)
{
    const bool isAutoFlag = msg->auto_flag;
    const bool isKilledFlag = msg->kill_flag;

    if (isAutoFlag && !isKilledFlag)
    {
        run_pid = true;
    }
    else
    {
        run_pid = false;

        // Reset controller state when disabled.
        e_Vel_total = 0.0;
        e_psi_total = 0.0;

        first_control_update = true;
    }
}


// ============================================================
// Desired state callback
// ============================================================

void FAU::PID_HS::desiredStateCallback(
    const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    if (msg->data.size() != 9)
    {
        RCLCPP_WARN(
            this->get_logger(),
            "Received desired state with incorrect size. Expected 9.");

        return;
    }

    const double psi = msg->data[2];
    const double vx = msg->data[3];
    const double vy = msg->data[4];

    // Desired speed magnitude.
    velD = std::sqrt(
        vx * vx +
        vy * vy);

    psi_goal = wrap_angle(psi);

    gotGoal = true;
}


// ============================================================
// Odometry callback
// ============================================================

void FAU::PID_HS::pose_callback(
    const nav_msgs::msg::Odometry::SharedPtr msg)
{
    /*
     * /p3d_wamv_ned:
     *
     *     linear.x = North velocity [m/s]
     *     linear.y = East velocity  [m/s]
     *
     * Heading convention:
     *
     *     x = North
     *     y = East
     *
     * Convert world/NED velocity into body-frame surge:
     *
     *     u = vN cos(psi) + vE sin(psi)
     */

    const double vN =
        msg->twist.twist.linear.x;

    const double vE =
        msg->twist.twist.linear.y;

    vel_USV =
        vN * std::cos(psi_usv_NED)
        +
        vE * std::sin(psi_usv_NED);
}


// ============================================================
// Vehicle pose callback
// ============================================================

void FAU::PID_HS::position_callback(
    const geometry_msgs::msg::Pose2D::SharedPtr msg)
{
    psi_usv_NED =
        wrap_angle(
            static_cast<double>(msg->theta));
}


// ============================================================
// Display gains
// ============================================================

void FAU::PID_HS::display_gains()
{
    RCLCPP_INFO(
        this->get_logger(),
        "---------- PID HS GAINS ----------");

    RCLCPP_INFO(
        this->get_logger(),
        "Speed: Kp_v=%.3f Ki_v=%.3f Kd_v=%.3f",
        Kp_v,
        Ki_v,
        Kd_v);

    RCLCPP_INFO(
        this->get_logger(),
        "Heading: Kp_psi=%.3f Ki_psi=%.3f Kd_psi=%.3f",
        Kp_psi,
        Ki_psi,
        Kd_psi);

    RCLCPP_INFO(
        this->get_logger(),
        "Limits: Tx=%.2f N, Mz=%.2f N m",
        Tx_max,
        Mz_max);
}


// ============================================================
// Main PID loop
// ============================================================

void FAU::PID_HS::loop()
{
    // ============================================================
    // Controller disabled / no goal
    // ============================================================

    if (!(gotGoal && run_pid))
    {
        T_x = 0.0;
        T_y = 0.0;
        M_z = 0.0;

        control_effort_msg.type = "HS";

        control_effort_msg.tau.clear();
        control_effort_msg.tau.push_back(0.0);
        control_effort_msg.tau.push_back(0.0);
        control_effort_msg.tau.push_back(0.0);

        control_pub->publish(
            control_effort_msg);

        first_control_update = true;

        return;
    }


    // ============================================================
    // SPEED PID
    //
    // Error:
    //
    //      e_v = v_d - u
    //
    // Output:
    //
    //      Tx [N]
    // ============================================================

    e_Vel =
        velD -
        vel_USV;


    // Small dead-zone for integral only.
    if (std::abs(e_Vel) < 0.05)
    {
        e_Vel_total = 0.0;
    }


    double de_Vel = 0.0;

    if (!first_control_update)
    {
        de_Vel =
            (e_Vel - e_Vel_prev) /
            dt;
    }


    // Candidate trapezoidal integral.
    double e_Vel_total_candidate =
        e_Vel_total;

    if (!first_control_update)
    {
        e_Vel_total_candidate +=
            0.5 *
            (e_Vel_prev + e_Vel) *
            dt;
    }


    // Unsaturated physical surge command [N].
    const double Tx_unsat =
        Kp_v * e_Vel
        +
        Ki_v * e_Vel_total_candidate
        +
        Kd_v * de_Vel;


    // Same physical force limit as APF.
    T_x =
        std::clamp(
            Tx_unsat,
            -Tx_max,
            Tx_max);


    // ------------------------------------------------------------
    // Anti-windup
    //
    // Only integrate while output is not saturated.
    // ------------------------------------------------------------

    if (std::abs(Tx_unsat - T_x) < 1e-6)
    {
        e_Vel_total =
            e_Vel_total_candidate;
    }


    // ============================================================
    // HEADING PID
    //
    // Error:
    //
    //      e_psi = psi_d - psi
    //
    // Output:
    //
    //      Mz [N m]
    // ============================================================

    e_psi =
        wrap_angle(
            psi_goal -
            psi_usv_NED);


    // Small dead-zone for integral only.
    if (std::abs(e_psi) < 0.0)
    {
        e_psi_total = 0.0;
    }


    double de_psi = 0.0;

    if (!first_control_update)
    {
        // Wrap the difference too, preventing a derivative spike
        // when crossing +/- pi.
        const double delta_error =
            wrap_angle(
                e_psi -
                e_psi_prev);

        de_psi =
            delta_error /
            dt;
    }


    // Candidate trapezoidal integral.
    double e_psi_total_candidate =
        e_psi_total;

    if (!first_control_update)
    {
        e_psi_total_candidate +=
            0.5 *
            (e_psi_prev + e_psi) *
            dt;
    }


    // Unsaturated physical yaw moment [N m].
    const double Mz_unsat =
        Kp_psi * e_psi
        +
        Ki_psi * e_psi_total_candidate
        +
        Kd_psi * de_psi;


    // Same physical yaw-moment limit as APF.
    M_z =
        std::clamp(
            Mz_unsat,
            -Mz_max,
            Mz_max);


    // Anti-windup.
    if (std::abs(Mz_unsat - M_z) < 1e-6)
    {
        e_psi_total =
            e_psi_total_candidate;
    }


    // ============================================================
    // Differential-drive configuration
    // ============================================================

    T_y = 0.0;


    // ============================================================
    // Save errors
    // ============================================================

    e_Vel_prev =
        e_Vel;

    e_psi_prev =
        e_psi;

    first_control_update =
        false;


    // ============================================================
    // Publish physical generalized control effort
    //
    // tau:
    //
    //      [Tx, Ty, Mz]
    //      [N,  N,  N m]
    // ============================================================

    control_effort_msg.type =
        "HS";

    control_effort_msg.tau.clear();


    if (controller == "H")
    {
        // Heading-only controller.
        control_effort_msg.tau.push_back(
            0.0);

        control_effort_msg.tau.push_back(
            0.0);

        control_effort_msg.tau.push_back(
            M_z);
    }
    else
    {
        // Heading + speed controller.
        control_effort_msg.tau.push_back(
            T_x);

        control_effort_msg.tau.push_back(
            0.0);

        control_effort_msg.tau.push_back(
            M_z);
    }


    control_pub->publish(
        control_effort_msg);
}