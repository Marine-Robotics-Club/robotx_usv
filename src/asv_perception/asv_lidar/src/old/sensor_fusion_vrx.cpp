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
#include <unordered_set>

class SensorFusionNode : public rclcpp::Node
{
public:
    SensorFusionNode() : Node("sensor_fusion_vrx")
    {
        this->declare_parameter<std::string>("wamv", "wamv1");
        this->get_parameter("wamv", wamv_name_);

        // ---- Subscriptions ----
        wamv_sub_ = this->create_subscription<lidar_msgs::msg::BuoyDetected>(
            "/" + wamv_name_ + "/vision/output/buoy_detected",
            10,
            std::bind(&SensorFusionNode::wamv_callback, this, std::placeholders::_1));

        yolo_sub_ = this->create_subscription<yolov26_msgs::msg::YoloDetection>(
            "/yolov26/detections",
            10,
            std::bind(&SensorFusionNode::yolo_callback, this, std::placeholders::_1));

        // ✅ Global pose (NED) from odom (change topic if yours is different)
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

        // ---- Sensor limits ---- (kept EXACTLY as you had)
        min_angle_   = -45.0 * (M_PI / 180.0);
        max_angle_   =  45.0 * (M_PI / 180.0);
        max_angle_2_ =  90.0 * (M_PI / 180.0);

        min_height_  = -3.0;
        max_height_  =  3.0;
        max_length_  = 30.0;

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

        psi_usv_NED_ = yaw; // radians
        have_odom_ = true;
    }

    // Convert BODY (sensor) point -> GLOBAL NED using odom pose/yaw.
    inline void body_to_global_ned(double xb, double yb, double zb,
                                   double &xg, double &yg, double &zg) const
    {
        // body NWU -> body NED (kept EXACTLY as you had)
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
        wamv.object_names.clear();
        wamv.x_object.clear();
        wamv.y_object.clear();
        wamv.z_object.clear();

        const size_t n = std::min({msg->name.size(), msg->x.size(), msg->y.size(), msg->z.size()});

        for (size_t i = 0; i < n; ++i) {
            const double angle = std::atan2(msg->y[i], msg->x[i]);
            const double distance = std::sqrt(msg->x[i] * msg->x[i] + msg->y[i] * msg->y[i]);

            if (angle >= min_angle_ && angle <= max_angle_ &&
                msg->z[i] >= min_height_ && msg->z[i] <= max_height_ &&
                distance <= max_length_)
            {
                wamv.object_names.push_back(msg->name[i]);
                wamv.x_object.push_back(msg->x[i]);
                wamv.y_object.push_back(msg->y[i]);
                wamv.z_object.push_back(msg->z[i]);
            }
        }
    }

    // ---------------- YOLO ----------------
    void yolo_callback(const yolov26_msgs::msg::YoloDetection::SharedPtr msg)
    {
        objects = *msg;

        if (!objects.class_name.empty()) {
            sensor_fusion_decision();
        }
    }

    // ---- pixel mapping helper (same mapping as your original code) ----
    inline double angle_to_pixel(double angle_rad) const
    {
        // kept same constants: 1280 width, cx=640, max_angle_2_ used as half-FOV in your mapping
        return ((1280.0 / -max_angle_2_) * angle_rad) + 640.0;
    }

    // ---- helper: compute lidar->pixel window for a lidar point (same math you had) ----
    inline void lidar_point_to_pixel_window(size_t j,
                                            double search_size,
                                            double &left_pixel,
                                            double &right_pixel) const
    {
        const double left_angle  = std::atan2((wamv.y_object[j] + search_size), wamv.x_object[j]);
        const double right_angle = std::atan2((wamv.y_object[j] - search_size), wamv.x_object[j]);

        left_pixel  = angle_to_pixel(left_angle);
        right_pixel = angle_to_pixel(right_angle);

        if (left_pixel > right_pixel) std::swap(left_pixel, right_pixel);
    }

    // ---- helper: pick ONE best lidar index for a given detection pixel, optionally skipping already-used ----
    // Uses predicted pixel of LiDAR point (NOT window center) + a mild distance penalty.
    int find_best_lidar_for_detection(double det_px,
                                     const std::unordered_set<int> &used_lidar,
                                     double search_size,
                                     double &best_err_px) const
    {
        int best_j = -1;
        double best_cost = 1e9;
        best_err_px = 1e9;

        for (size_t j = 0; j < wamv.x_object.size(); ++j) {
            if (used_lidar.count(static_cast<int>(j)) > 0)
                continue;

            double lp, rp;
            lidar_point_to_pixel_window(j, search_size, lp, rp);

            // gate: detection must fall inside lidar pixel window
            if (det_px < lp || det_px > rp)
                continue;

            // predicted pixel from actual lidar point angle
            const double ang = std::atan2(wamv.y_object[j], wamv.x_object[j]);
            const double pred_px = angle_to_pixel(ang);

            const double err_px = std::fabs(det_px - pred_px);
            const double dist = std::hypot(wamv.x_object[j], wamv.y_object[j]);

            // cost: pixel error + mild distance penalty
            const double cost = err_px + 0.02 * dist;

            if (cost < best_cost) {
                best_cost = cost;
                best_j = static_cast<int>(j);
                best_err_px = err_px;
            }
        }

        return best_j;
    }

    // ---------------- FUSION ----------------
    void sensor_fusion_decision()
    {
        if (!have_odom_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "No odom yet; skipping global fusion.");
            return;
        }

        fusion.object_names.clear();
        fusion.x_object.clear();
        fusion.y_object.clear();
        fusion.z_object.clear();

        // Pre-collect red/green pixel centers to suppress black when overlapping (your main failure mode)
        std::vector<double> rg_px;
        rg_px.reserve(objects.class_name.size());
        for (size_t i = 0; i < objects.class_name.size(); ++i) {
            if (objects.class_name[i] == "red_buoy" || objects.class_name[i] == "green_buoy") {
                rg_px.push_back(objects.x_center[i]);
            }
        }

        // one LiDAR point used once per callback
        std::unordered_set<int> used_lidar;

        auto process_class = [&](const std::string &target_cls)
        {
            for (size_t i = 0; i < objects.class_name.size(); ++i) {
                if (objects.class_name[i] != target_cls) continue;

                const double det_px = objects.x_center[i];

                // kept EXACTLY as you had
                const double search_size = 1.5;

                // Extra suppression ONLY for black: if a red/green is near the same pixel column, skip black
                if (target_cls == "black_buoy") {
                    bool near_rg = false;
                    for (double px : rg_px) {
                        if (std::fabs(px - det_px) < 60.0) { // tune 40–80 if needed
                            near_rg = true;
                            break;
                        }
                    }
                    if (near_rg) {
                        continue;
                    }
                }

                double best_err_px = 1e9;
                const int best_j = find_best_lidar_for_detection(det_px, used_lidar, search_size, best_err_px);
                if (best_j < 0) continue;

                // Extra strictness ONLY for black
                if (target_cls == "black_buoy") {
                    const double black_max_err_px = 35.0; // tune 30–50
                    if (best_err_px > black_max_err_px) {
                        continue;
                    }
                }

                used_lidar.insert(best_j);

                // ✅ Transform to GLOBAL NED
                double xg, yg, zg;
                body_to_global_ned(wamv.x_object[best_j], wamv.y_object[best_j], wamv.z_object[best_j], xg, yg, zg);

                fusion.object_names.push_back(target_cls);
                fusion.x_object.push_back(xg);
                fusion.y_object.push_back(yg);
                fusion.z_object.push_back(zg);
            }
        };

        // priority order: red/green first, then black (keeps black from stealing)
        process_class("red_buoy");
        process_class("green_buoy");
        process_class("black_buoy");

        if (!fusion.object_names.empty()) {
            fusion_pub_->publish(fusion);
            publish_to_rviz(fusion);
        }
    }

    // ---------------- RVIZ ----------------
    void publish_to_rviz(const fau_msgs::msg::ObjectPositionArray &fusion_data)
    {
        geometry_msgs::msg::PoseArray pose_array_msg;
        pose_array_msg.header.stamp = this->get_clock()->now();

        // kept EXACTLY as you had it
        pose_array_msg.header.frame_id = "wamv/wamv/base_link/lidar_wamv_sensor";

        pose_array_msg.poses.clear();
        pose_array_msg.poses.reserve(fusion_data.x_object.size());

        for (size_t i = 0; i < fusion_data.x_object.size(); ++i) {
            geometry_msgs::msg::Pose pose;
            pose.position.x = fusion_data.x_object[i];
            pose.position.y = fusion_data.y_object[i];
            pose.position.z = fusion_data.z_object[i];
            pose_array_msg.poses.push_back(pose);
        }

        rviz_pub_->publish(pose_array_msg);
    }

private:
    // Params
    std::string wamv_name_;

    // Subs/Pubs
    rclcpp::Subscription<lidar_msgs::msg::BuoyDetected>::SharedPtr wamv_sub_;
    rclcpp::Subscription<yolov26_msgs::msg::YoloDetection>::SharedPtr yolo_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::Publisher<fau_msgs::msg::ObjectPositionArray>::SharedPtr fusion_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr rviz_pub_;

    // Data
    fau_msgs::msg::ObjectPositionArray wamv;
    yolov26_msgs::msg::YoloDetection objects;
    fau_msgs::msg::ObjectPositionArray fusion;

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
    double psi_usv_NED_{0.0}; // yaw rad
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SensorFusionNode>());
    rclcpp::shutdown();
    return 0;
}
