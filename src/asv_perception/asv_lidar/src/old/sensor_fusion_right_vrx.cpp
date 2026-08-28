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

class SensorFusionNode : public rclcpp::Node
{
public:
    SensorFusionNode() : Node("sensor_fusion_vrx")
    {
        // ---- Params ----
        this->declare_parameter<std::string>("wamv", "wamv");
        this->declare_parameter<bool>("use_sim_time", true);
        this->get_parameter("wamv", wamv_name_);

        // ---- Subscriptions (namespaced) ----
        lidar_buoy_sub_ = this->create_subscription<lidar_msgs::msg::BuoyDetected>(
            "/" + wamv_name_ + "/vision/output/buoy_detected",
            10,
            std::bind(&SensorFusionNode::lidar_buoy_callback, this, std::placeholders::_1));

        // YOLO
        yolo_sub_ = this->create_subscription<yolov26_msgs::msg::YoloDetection>(
            "/yolov26/detections/right",
            10,
            std::bind(&SensorFusionNode::yolo_callback, this, std::placeholders::_1));

        // Odom (global NED)
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/" + wamv_name_ + "/p3d_wamv_ned",
            10,
            std::bind(&SensorFusionNode::odom_callback, this, std::placeholders::_1));

        // ---- Publishers ----
        fusion_pub_ = this->create_publisher<fau_msgs::msg::ObjectPositionArray>(
            "/" + wamv_name_ + "/vision/output/fusion",
            10);

        rviz_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
            "/" + wamv_name_ + "/vision/output/rviz_fusion",
            10);

        // ---- Sensor limits ----
        min_angle_   = -45.0 * (M_PI / 180.0);
        max_angle_   =  45.0 * (M_PI / 180.0);
        max_angle_2_ =  90.0 * (M_PI / 180.0);

        min_height_  = -3.0;
        max_height_  =  3.0;
        max_length_  = 50.0;

        RCLCPP_INFO(this->get_logger(), "SensorFusionNode started for %s", wamv_name_.c_str());
    }

private:
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

        psi_usv_NED_ = yaw;  // radians
        have_odom_ = true;
    }

    // Convert BODY point -> GLOBAL NED using odom pose/yaw.
    // Assumption: incoming lidar points are BODY frame with x forward, y left (NWU-ish).
    // Convert to BODY NED (y right) then rotate+translate.
    inline void body_to_global_ned(double xb, double yb, double zb,
                                   double &xg, double &yg, double &zg) const
    {
        const double x_body = xb;
        const double y_body = -yb;  // left->right

        const double c = std::cos(psi_usv_NED_);
        const double s = std::sin(psi_usv_NED_);

        const double dN = c * x_body - s * y_body;
        const double dE = s * x_body + c * y_body;

        xg = x_usv_NED_ + dN;
        yg = y_usv_NED_ + dE;
        zg = zb;
    }

    // Generic filter helper (keeps within angle/height/range)
    void filter_lidar(const lidar_msgs::msg::BuoyDetected::SharedPtr msg,
                      fau_msgs::msg::ObjectPositionArray &out)
    {
        out.object_names.clear();
        out.x_object.clear();
        out.y_object.clear();
        out.z_object.clear();

        const size_t n = std::min({msg->name.size(), msg->x.size(), msg->y.size(), msg->z.size()});

        for (size_t i = 0; i < n; ++i) {
            const double angle = std::atan2(msg->y[i], msg->x[i]);
            const double dist  = std::sqrt(msg->x[i] * msg->x[i] + msg->y[i] * msg->y[i]);

            if (angle >= min_angle_ && angle <= max_angle_ &&
                msg->z[i] >= min_height_ && msg->z[i] <= max_height_ &&
                dist <= max_length_)
            {
                out.object_names.push_back(msg->name[i]);
                out.x_object.push_back(msg->x[i]);
                out.y_object.push_back(msg->y[i]);
                out.z_object.push_back(msg->z[i]);
            }
        }
    }

    // ---------------- LIDAR callbacks ----------------
    void lidar_buoy_callback(const lidar_msgs::msg::BuoyDetected::SharedPtr msg) { filter_lidar(msg, buoy_); }

    // ---------------- YOLO ----------------
    void yolo_callback(const yolov26_msgs::msg::YoloDetection::SharedPtr msg)
    {
        objects_ = *msg;
        if (!objects_.class_name.empty()) {
            sensor_fusion_decision();
        }
    }

    // Utility: compute pixel gate from lidar point
    inline void compute_pixel_gate(double x, double y, double search_size,
                                   double &left_px, double &right_px) const
    {
        // Keep your original style: expand in x by +/- search_size, then map angle -> pixel.
        const double left_angle  = std::atan2(y, x - search_size);
        const double right_angle = std::atan2(y, x + search_size);

        // NOTE: Your original code had +1920 and other offsets. That depends on your image width and ROI.
        // In the VRX-style example, 640 assumed 1280 width. Here we keep 640 for 1280px.
        left_px  = ((1280.0 / -max_angle_2_) * left_angle)  + 640.0;
        right_px = ((1280.0 / -max_angle_2_) * right_angle) + 640.0;

        if (left_px > right_px) std::swap(left_px, right_px);
    }

    inline bool in_gate(double x_center, double left_px, double right_px) const
    {
        return (x_center >= left_px) && (x_center <= right_px);
    }

    // ---------------- FUSION ----------------
    void sensor_fusion_decision()
    {
        if (!have_odom_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                "No odom yet; skipping fusion.");
            return;
        }

        fusion_.object_names.clear();
        fusion_.x_object.clear();
        fusion_.y_object.clear();
        fusion_.z_object.clear();

        for (size_t i = 0; i < objects_.class_name.size(); ++i) {

            const std::string &cls = objects_.class_name[i];
            const double xc = objects_.x_center[i];

            // ✅ ONLY BUOYS
            if (cls == "red_buoy" || cls == "green_buoy" ||
                cls == "black_buoy")
            {
                for (size_t j = 0; j < buoy_.object_names.size(); ++j) {

                    double left_px, right_px;
                    compute_pixel_gate(
                        buoy_.x_object[j],
                        buoy_.y_object[j],
                        1.5,
                        left_px,
                        right_px
                    );

                    if (in_gate(xc, left_px, right_px)) {

                        double xg, yg, zg;
                        body_to_global_ned(
                            buoy_.x_object[j],
                            buoy_.y_object[j],
                            buoy_.z_object[j],
                            xg, yg, zg
                        );

                        fusion_.object_names.push_back(cls);
                        fusion_.x_object.push_back(xg);
                        fusion_.y_object.push_back(yg);
                        fusion_.z_object.push_back(zg);
                    }
                }
            }
        }

        if (!fusion_.object_names.empty()) {
            fusion_pub_->publish(fusion_);
            publish_to_rviz(fusion_);
        }
    }


    void match_and_push(const std::string &cls, double x_center,
                        const fau_msgs::msg::ObjectPositionArray &lidar_bucket)
    {
        const double search_size = 1.5;

        for (size_t j = 0; j < lidar_bucket.object_names.size(); ++j) {

            double left_px, right_px;
            compute_pixel_gate(lidar_bucket.x_object[j], lidar_bucket.y_object[j], search_size, left_px, right_px);

            if (in_gate(x_center, left_px, right_px)) {

                // ✅ Transform BODY -> GLOBAL NED
                double xg, yg, zg;
                body_to_global_ned(lidar_bucket.x_object[j], lidar_bucket.y_object[j], lidar_bucket.z_object[j],
                                   xg, yg, zg);

                fusion_.object_names.push_back(cls);
                fusion_.x_object.push_back(xg);
                fusion_.y_object.push_back(yg);
                fusion_.z_object.push_back(zg);
            }
        }
    }

    // ---------------- RVIZ ----------------
    void publish_to_rviz(const fau_msgs::msg::ObjectPositionArray &fusion_data)
    {
        geometry_msgs::msg::PoseArray pose_array;
        pose_array.header.stamp = this->get_clock()->now();

        // ✅ IMPORTANT:
        // Since fusion_ is GLOBAL NED, RViz should use a GLOBAL frame (usually "odom" or "map").
        // Replace with your actual global frame.
        pose_array.header.frame_id = "wamv/wamv/base_link/lidar_wamv_sensor";

        pose_array.poses.clear();
        pose_array.poses.reserve(fusion_data.x_object.size());

        for (size_t i = 0; i < fusion_data.x_object.size(); ++i) {
            geometry_msgs::msg::Pose p;
            p.position.x = fusion_data.x_object[i];
            p.position.y = fusion_data.y_object[i];
            p.position.z = fusion_data.z_object[i];
            pose_array.poses.push_back(p);
        }

        rviz_pub_->publish(pose_array);
    }

private:
    // Params
    std::string wamv_name_;

    // Subs/Pubs
    rclcpp::Subscription<lidar_msgs::msg::BuoyDetected>::SharedPtr lidar_buoy_sub_;
    rclcpp::Subscription<yolov26_msgs::msg::YoloDetection>::SharedPtr yolo_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::Publisher<fau_msgs::msg::ObjectPositionArray>::SharedPtr fusion_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr rviz_pub_;

    // Data buckets (BODY frame)
    fau_msgs::msg::ObjectPositionArray buoy_;
    fau_msgs::msg::ObjectPositionArray ball_;
    fau_msgs::msg::ObjectPositionArray boat_;
    fau_msgs::msg::ObjectPositionArray stop_light_;

    // YOLO and fusion
    yolov26_msgs::msg::YoloDetection objects_;
    fau_msgs::msg::ObjectPositionArray fusion_;

    // Sensor limits
    double min_angle_{0.0};
    double max_angle_{0.0};
    double min_height_{0.0};
    double max_height_{0.0};
    double max_length_{0.0};
    double max_angle_2_{0.0};

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
