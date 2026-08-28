#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include <nav_msgs/msg/odometry.hpp>
#include <cmath>
#include <tuple>  // for std::tuple
class PointCloudTransformer : public rclcpp::Node {
public:
    PointCloudTransformer() : Node("pointcloud_transformer") {
        // Subscriber to raw point cloud (Use Reliable QoS for consistency)
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "velodyne_points",
            rclcpp::SensorDataQoS(),
            std::bind(&PointCloudTransformer::cloud_callback, this, std::placeholders::_1));

        nav_ned_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/wamv1/NA_nav_ned", 1,
        std::bind(&PointCloudTransformer::nav_ned_callback, this, std::placeholders::_1));

        // Publisher for transformed point cloud (ensure consistent QoS)
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/transformed_cloud", rclcpp::SensorDataQoS());

        RCLCPP_INFO(this->get_logger(), "PointCloud Transformer Node Started");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr nav_ned_sub_;
    double x_usv_NED_ = 0.0;
    double y_usv_NED_ = 0.0;
    double z_usv_NED_ = 0.0;
    double psi_usv_NED_ = 0.0;
    double vel_x_ = 0.0;
    double vel_y_ = 0.0;
    double vel_USV_ = 0.0;

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
        pcl::PointCloud<pcl::PointXYZ> cloud, pitch_adjusted_cloud;
        pcl::fromROSMsg(*cloud_msg, cloud);

        // Step 1: Apply static transform: -8° pitch (Y-axis) + 2.0m Z offset
        Eigen::Affine3f static_transform = Eigen::Affine3f::Identity();
        float angle_rad = 8.0f * M_PI / 180.0f;
        static_transform.rotate(Eigen::AngleAxisf(angle_rad, Eigen::Vector3f::UnitY()));
        static_transform.translation() << 0.0f, 0.0f, 2.0f;

        pcl::transformPointCloud(cloud, pitch_adjusted_cloud, static_transform);

        // Step 2: Apply dynamic transform to NED frame using transform_nwu_ned()
        pcl::PointCloud<pcl::PointXYZ> final_cloud;
        final_cloud.points.reserve(pitch_adjusted_cloud.size());

        for (const auto& pt : pitch_adjusted_cloud.points) {
            if (!pcl::isFinite(pt)) continue;

            auto [x_ned, y_ned, z_ned] = transform_nwu_ned(pt.x, pt.y, pt.z);

            pcl::PointXYZ transformed_point;
            transformed_point.x = x_ned;
            transformed_point.y = y_ned;
            transformed_point.z = z_ned;

            final_cloud.points.push_back(transformed_point);
        }

        // Finalize point cloud format for publishing
        final_cloud.width = final_cloud.points.size();
        final_cloud.height = 1;
        final_cloud.is_dense = true;
        final_cloud.header = pitch_adjusted_cloud.header;

        // Convert to ROS2 message
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(final_cloud, output);
        output.header.stamp = cloud_msg->header.stamp;
        output.header.frame_id = "map";  // NED/global frame

        publisher_->publish(output);
    }

    void nav_ned_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        x_usv_NED_ = msg->pose.pose.position.x;
        y_usv_NED_ = msg->pose.pose.position.y;
        z_usv_NED_ = msg->pose.pose.position.z;
        psi_usv_NED_ = msg->pose.pose.orientation.z;  // NOTE: this is usually a quaternion, consider converting if necessary

        vel_x_ = msg->twist.twist.linear.x;
        vel_y_ = msg->twist.twist.linear.y;
        vel_USV_ = std::sqrt(vel_x_ * vel_x_ + vel_y_ * vel_y_);
    }

    std::tuple<double, double, double> transform_nwu_ned(double x_local, double y_local, double z_local) {
        Eigen::Vector3d d_L_G(x_usv_NED_, y_usv_NED_, z_usv_NED_);

        // Rotation matrix for yaw only (around Z)
        Eigen::Matrix3d R_L_G;
        R_L_G << std::cos(psi_usv_NED_), -std::sin(psi_usv_NED_), 0,
                 std::sin(psi_usv_NED_),  std::cos(psi_usv_NED_), 0,
                 0,                       0,                      1;

        // Convert from NWU to NED (invert Y and Z)
        Eigen::Vector3d d_P_L(x_local, -y_local, -z_local);

        Eigen::Matrix3d R_P_L = Eigen::Matrix3d::Identity();

        Eigen::Matrix4d H_L_G = Eigen::Matrix4d::Identity();
        H_L_G.block<3,3>(0,0) = R_L_G;
        H_L_G.block<3,1>(0,3) = d_L_G;

        Eigen::Matrix4d H_P_L = Eigen::Matrix4d::Identity();
        H_P_L.block<3,3>(0,0) = R_P_L;
        H_P_L.block<3,1>(0,3) = d_P_L;

        Eigen::Matrix4d H_P_G = H_L_G * H_P_L;

        double x_global = H_P_G(0, 3);
        double y_global = H_P_G(1, 3);
        double z_global = H_P_G(2, 3);

        return std::make_tuple(x_global, y_global, z_global);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudTransformer>());
    rclcpp::shutdown();
    return 0;
}