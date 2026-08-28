#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <cmath>
#include <vector>

class ClearWaterNode : public rclcpp::Node {
public:
    ClearWaterNode() : Node("clear_water") {

        std::string wamv_name;
        this->declare_parameter("wamv", "wamv1");
        this->get_parameter("wamv", wamv_name);

        // Subscriber to transformed point cloud (consistent QoS)
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/" + wamv_name + "/transformed_cloud", rclcpp::SensorDataQoS(),
            std::bind(&ClearWaterNode::cloud_callback, this, std::placeholders::_1));

        // Publisher for filtered point cloud (consistent QoS)
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/" + wamv_name + "/lidar_wamv/points_no_water", rclcpp::SensorDataQoS());

        RCLCPP_INFO(this->get_logger(), "ClearWaterNode has started");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*cloud_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ> filtered_cloud;

        for (const auto& point : cloud->points) {
            double x = point.x, y = point.y, z = point.z;
            double distance = std::sqrt(x * x + y * y);
            double angle = std::atan2(y, x); // Compute heading angle (yaw)

            const double alpha = 110.0 * M_PI / 180.0;  // widen beyond 90°
            if (distance > 1.0 && distance < 100.0 &&
                angle >= -alpha && angle <= alpha &&
                z > 0.0 && z < 3.0)
            {
                filtered_cloud.points.push_back(point);
            }

        }
        if (filtered_cloud.points.empty()) {
            RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty. Publishing a placeholder to avoid crashes.");

            // Create an empty placeholder cloud
            sensor_msgs::msg::PointCloud2 output;
            output.header.frame_id = cloud_msg->header.frame_id;
            output.header.stamp = cloud_msg->header.stamp;
            output.width = 1;  // At least one point to prevent issues
            output.height = 1;
            output.is_dense = true;

            // Add a dummy point at infinity to avoid crashes
            pcl::PointCloud<pcl::PointXYZ> dummy_cloud;
            dummy_cloud.push_back(pcl::PointXYZ(std::numeric_limits<float>::quiet_NaN(),
                                                std::numeric_limits<float>::quiet_NaN(),
                                                std::numeric_limits<float>::quiet_NaN()));
            pcl::toROSMsg(dummy_cloud, output);
            
            publisher_->publish(output);
        } else {
            // Normal point cloud processing
            filtered_cloud.width = filtered_cloud.points.size();
            filtered_cloud.height = 1;
            filtered_cloud.is_dense = true;
            
            sensor_msgs::msg::PointCloud2 output;
            pcl::toROSMsg(filtered_cloud, output);
            output.header.frame_id = cloud_msg->header.frame_id;
            output.header.stamp = cloud_msg->header.stamp;
            publisher_->publish(output);
        }


    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ClearWaterNode>());
    rclcpp::shutdown();
    return 0;
}
