#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>

class PointCloudTransformer : public rclcpp::Node {
public:
    PointCloudTransformer() : Node("pointcloud_transformer") {

        std::string wamv_name;
        this->declare_parameter("wamv", "wamv1");
        this->get_parameter("wamv", wamv_name);
        // Subscriber to raw point cloud (Use Reliable QoS for consistency)
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/" + wamv_name+"/sensors/lidars/lidar_wamv_sensor/points",
            rclcpp::SensorDataQoS(),
            std::bind(&PointCloudTransformer::cloud_callback, this, std::placeholders::_1));

        // Publisher for transformed point cloud (ensure consistent QoS)
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/" + wamv_name + "/transformed_cloud", rclcpp::SensorDataQoS());

        RCLCPP_INFO(this->get_logger(), "PointCloud Transformer Node Started");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
        pcl::PointCloud<pcl::PointXYZ> cloud, transformed_cloud;
        pcl::fromROSMsg(*cloud_msg, cloud);

        // Define transformation matrix: -8° pitch rotation (Y-axis) + 1.8m Z translation
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        float angle_rad = 1.5 * M_PI / 180.0;  // Convert degrees to radians (negative for correct pitch)
        transform.rotate(Eigen::AngleAxisf(angle_rad, Eigen::Vector3f::UnitY())); // -8° Pitch
        transform.translation() << 0.0, 0.0, -0.2;  // Move 1.8 meters up in Z

        // Apply transformation
        pcl::transformPointCloud(cloud, transformed_cloud, transform);

        // Convert back to ROS PointCloud2
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(transformed_cloud, output);
        
        // Ensure timestamps match
        output.header.stamp = cloud_msg->header.stamp; // Keep original timestamp
        output.header.frame_id = cloud_msg->header.frame_id; // Keep original frame (if needed)

        // Publish transformed cloud
        publisher_->publish(output);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudTransformer>());
    rclcpp::shutdown();
    return 0;
}
