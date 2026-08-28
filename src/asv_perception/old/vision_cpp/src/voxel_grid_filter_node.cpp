#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/transforms.hpp>

class VoxelGridFilterNode : public rclcpp::Node {
public:
    VoxelGridFilterNode() : Node("voxel_grid_filter_node") {
        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 0, std::bind(&VoxelGridFilterNode::pointCloudCallback, this, std::placeholders::_1));
        point_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/vision/pc2/voxel", 0);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Convert ROS2 PointCloud2 message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        // Apply Voxel Grid filtering using PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(cloud);
        voxel_filter.setLeafSize(0.5f, 0.5f, 0.5f);  // Set the voxel size
        voxel_filter.filter(*cloud_filtered);

        // Convert the filtered point cloud back to ROS2 PointCloud2 message
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud_filtered, output);
        output.header = msg->header;

        // Publish the filtered point cloud
        point_cloud_pub_->publish(output);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VoxelGridFilterNode>());
    rclcpp::shutdown();
    return 0;
}

