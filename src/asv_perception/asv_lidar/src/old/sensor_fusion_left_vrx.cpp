#include "rclcpp/rclcpp.hpp"

#include "lidar_msgs/msg/buoy_detected.hpp"          // Custom message from the lidar
#include "yolov26_msgs/msg/yolo_detection.hpp"       // Custom YOLO message for bounding boxes
#include "fau_msgs/msg/object_position_array.hpp"

#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

#include <vector>
#include <string>
#include <cmath>

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
            "/yolov26/detections/left",
            10,
            std::bind(&SensorFusionNode::yolo_callback, this, std::placeholders::_1));

        // ✅ Global pose (NED) from odom (change topic if yours is different)
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/" + wamv_name_ + "/p3d_wamv_ned",
            10,
            std::bind(&SensorFusionNode::odom_callback, this, std::placeholders::_1));

        // ---- Publishers ----
        // ✅ This will now publish GLOBAL (NED) positions
        fusion_pub_ = this->create_publisher<fau_msgs::msg::ObjectPositionArray>(
            "/" + wamv_name_ + "/vision/output/fusion",
            10);

        // ✅ RViz should subscribe in the SAME global frame (odom/map)
        rviz_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
            "/" + wamv_name_ + "/vision/output/rviz_fusion",
            10);

        // ---- Sensor limits ----
        min_angle_   =  45.0 * (M_PI / 180.0);
        max_angle_   =  135.0 * (M_PI / 180.0);
        max_angle_2_ =  90.0 * (M_PI / 180.0);

        min_height_  = -3.0;
        max_height_  =  3.0;
        max_length_  = 10.0;

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
    // Assumes incoming buoy coords are BODY NWU-ish: x forward, y left.
    // Convert to BODY NED (y right) then rotate+translate into GLOBAL NED.
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
                /*
                RCLCPP_INFO(
                this->get_logger(),
                "PASS [%zu] name=%s | angle=%.2f deg | dist=%.2f | x=%.2f y=%.2f z=%.2f",
                i,
                msg->name[i].c_str(),
                angle * 180.0 / M_PI,
                distance,
                msg->x[i],
                msg->y[i],
                msg->z[i]
            );*/
                
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

    inline double angle_to_pixel(double angle_rad, double image_width, double half_fov_rad) const
    {
        const double cx = image_width * 0.5;
        return cx + (angle_rad / half_fov_rad) * cx;
    }

    inline double pixel_to_angle(double x_px, double image_width, double half_fov_rad) const
    {
        const double cx = image_width * 0.5;
        return (x_px - cx) * (half_fov_rad / cx);
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

    // ---------------- Camera model ----------------
    const double W  = 1280.0;
    const double cx = W * 0.5;

    // IMPORTANT:
    // DO NOT use max_angle_2_ here if max_angle_2_ is your LiDAR half-plane (~pi/2).
    // Set your real camera HFOV, e.g. 60 deg -> half = 30 deg.
    // Tune this to your VRX camera topic HFOV.
    const double camera_hfov_deg = 60.0;  // <-- CHANGE to your camera HFOV
    const double cam_half_fov = (camera_hfov_deg * M_PI / 180.0) * 0.5;

    // Pinhole focal length in pixels
    const double fx = cx / std::tan(cam_half_fov);

    auto wrap_pi = [&](double a) {
        return std::atan2(std::sin(a), std::cos(a));
    };

    auto pixel_to_angle = [&](double x_px) {
        // camera bearing, center=0
        return std::atan((x_px - cx) / fx);
    };

    auto angle_to_pixel = [&](double ang) {
        return cx + fx * std::tan(ang);
    };

    // ---------------- Tuning knobs ----------------
    const bool mirror_yolo = true;                       // flip if your image is mirrored
    const double cam_yaw_offset = 7.8 * M_PI / 180.0;    // from your logs
    const double max_match_err  = 12.0 * M_PI / 180.0;   // accept threshold (deg -> rad)

    // Body/LiDAR -> LEFT CAMERA bearing model (this was the one that made RED perfect)
    auto lidar_to_leftcam_angle = [&](double xb, double yb) {
        // left camera "forward" points to body +Y
        // left camera "right" points to body -X
        const double x_cam =  yb;
        const double y_cam = -xb;

        double ang = std::atan2(y_cam, x_cam);           // rad
        ang = wrap_pi(ang + cam_yaw_offset);             // add yaw offset
        return ang;
    };

    // ---------------- YOLO loop ----------------
    for (size_t i = 0; i < objects.class_name.size(); ++i) {

        const std::string &cls = objects.class_name[i];

        if (!(cls == "red_buoy" || cls == "green_buoy" || cls == "black_buoy"))
            continue;

        // YOLO pixel -> camera bearing
        const double yolo_raw = objects.x_center[i];
        const double yolo_px  = mirror_yolo ? (W - yolo_raw) : yolo_raw;
        const double yolo_ang = pixel_to_angle(yolo_px);

        // Find best LiDAR point for this detection
        int best_j = -1;
        double best_err = 1e9;

        double best_cam_ang = 0.0;
        double best_pred_px = 0.0;

        for (size_t j = 0; j < wamv.object_names.size(); ++j) {

            const double xb = wamv.x_object[j];
            const double yb = wamv.y_object[j];

            const double cam_ang = lidar_to_leftcam_angle(xb, yb);
            const double err = std::fabs(wrap_pi(yolo_ang - cam_ang));

            if (err < best_err) {
                best_err = err;
                best_j = static_cast<int>(j);
                best_cam_ang = cam_ang;
                best_pred_px = angle_to_pixel(cam_ang);
            }
        }

        if (best_j < 0) continue;

        const double xb_best = wamv.x_object[best_j];
        const double yb_best = wamv.y_object[best_j];

        // Debug print
        /*RCLCPP_INFO(
            this->get_logger(),
            "%s cand(best): yolo_raw=%.1f yolo_px=%.1f yolo_ang=%.1f deg | "
            "best_cam_ang=%.1f deg | pred_px=%.1f | err=%.1f deg | lidar_xy=(%.2f,%.2f)",
            cls.c_str(),
            yolo_raw, yolo_px,
            yolo_ang * 180.0 / M_PI,
            best_cam_ang * 180.0 / M_PI,
            best_pred_px,
            best_err * 180.0 / M_PI,
            xb_best, yb_best
        );*/

        // Reject if error too large
        if (best_err > max_match_err) {

            /*RCLCPP_INFO(
                this->get_logger(),
                "%s reject: yolo_raw=%.1f yolo_px=%.1f yolo_ang=%.1f deg | best_cam_ang=%.1f deg | err=%.1f deg",
                cls.c_str(),
                yolo_raw, yolo_px,
                yolo_ang * 180.0 / M_PI,
                best_cam_ang * 180.0 / M_PI,
                best_err * 180.0 / M_PI
            );*/
            continue;
        }

        // ✅ Transform accepted LiDAR point to GLOBAL NED
        double xg, yg, zg;
        body_to_global_ned(xb_best, yb_best, wamv.z_object[best_j], xg, yg, zg);

        fusion.object_names.push_back(cls);
        fusion.x_object.push_back(xg);
        fusion.y_object.push_back(yg);
        fusion.z_object.push_back(zg);
    }

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

        // ✅ Because fusion is GLOBAL NED now, RViz frame must be the global frame.
        // Change to "map" if your nav stack uses map; otherwise "odom" is typical.
        pose_array_msg.header.frame_id = "wamv/wamv/base_link/lidar_wamv_sensor";

        pose_array_msg.poses.clear();
        pose_array_msg.poses.reserve(fusion_data.x_object.size());

        for (size_t i = 0; i < fusion_data.x_object.size(); ++i) {
            geometry_msgs::msg::Pose pose;
            pose.position.x = fusion_data.x_object[i];
            pose.position.y = fusion_data.y_object[i];
            pose.position.z = fusion_data.z_object[i];
            // orientation left default (0) for points
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
