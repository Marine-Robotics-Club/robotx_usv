#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/search/kdtree.h"
#include "pcl/segmentation/extract_clusters.h"
#include <pcl/common/centroid.h>  // For centroid calculation
#include <pcl/common/common.h>    // For getMinMax3D
#include <pcl/filters/filter.h>   // For removeNaNFromPointCloud
#include <Eigen/Dense>
#include <iostream>
#include <random>  // For random color generation

class PointCloudClusterer : public rclcpp::Node {
public:
    PointCloudClusterer() : Node("point_cloud_clusterer") {
        // Subscriber to the voxel-filtered point cloud
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne/velodyne_points", 1,
            std::bind(&PointCloudClusterer::pointcloud_callback, this, std::placeholders::_1)
        );

        // Publisher for clustered point clouds
        cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("clustered_points", 10);
    }

private:
    // Callback for PointCloud data
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Convert the ROS2 message to a PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // Remove NaN values from the point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, indices);

        if (cloud_filtered->empty()) {
            RCLCPP_WARN(this->get_logger(), "Filtered cloud is empty after removing NaNs!");
            return;
        }

        // Create a KdTree object for clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_filtered);

        // Euclidean clustering object
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.3);  // Set distance threshold (30 cm)
        ec.setMinClusterSize(20);     // Minimum number of points in a cluster
        ec.setMaxClusterSize(25000);  // Maximum number of points in a cluster
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_filtered);
        ec.extract(cluster_indices);

        // Create a point cloud to hold all clusters with RGB
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusters_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        // Random color generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        // Iterate over clusters and process each one
        int cluster_id = 0;
        for (const auto& indices : cluster_indices) {
            // Create a point cloud to hold the current cluster with RGB
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
            for (const auto& index : indices.indices) {
                pcl::PointXYZRGB point;
                point.x = cloud_filtered->points[index].x;
                point.y = cloud_filtered->points[index].y;
                point.z = cloud_filtered->points[index].z;
                cluster->points.push_back(point);
            }
            cluster->width = cluster->points.size();
            cluster->height = 1;
            cluster->is_dense = true;

            // Calculate centroid
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cluster, centroid);

            // Calculate bounding box with pcl::PointXYZRGB
            pcl::PointXYZRGB min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);

            // Calculate the dimensions of the cluster
            float width_x = max_pt.x - min_pt.x;
            float width_y = max_pt.y - min_pt.y;
            float height_z = max_pt.z - min_pt.z;

            // Set the size threshold for filtering (0.5m in X and Y, 1m in Z)
            const float min_x_size = 0.2f;
            const float min_y_size = 0.2f;
            const float min_z_size = 0.8f;
            const float max_x_size = 0.8f;
            const float max_y_size = 0.8f;
            const float max_z_size = 2.8f;

            // Filter based on the size criteria
            if (width_x >= min_x_size && max_x_size >= width_x 
                && width_y >= min_y_size && max_y_size >= width_y 
                && height_z >= min_z_size && max_z_size >= height_z) {
                // If the cluster meets the size criteria, add it to the output cloud

                // Generate a random color for the current cluster
                uint8_t r = static_cast<uint8_t>(dis(gen));
                uint8_t g = static_cast<uint8_t>(dis(gen));
                uint8_t b = static_cast<uint8_t>(dis(gen));

                // Add the current cluster to the combined clusters cloud
                for (auto& point : cluster->points) {
                    point.r = 16; //this is neon green 
                    point.g = 255;
                    point.b = 32;
                }

                *clusters_cloud += *cluster;

                // Cluster size category
                std::string cluster_size = categorize_cluster(cluster->points.size());

                // Print cluster information
                RCLCPP_INFO(this->get_logger(),
                    "Cluster ID: %d | Size: %s | Dimensions (X, Y, Z): [%.2f, %.2f, %.2f] | Centroid: [%.2f, %.2f, %.2f] | Color: RGB(%d, %d, %d)",
                    cluster_id, cluster_size.c_str(), width_x, width_y, height_z, centroid[0], centroid[1], centroid[2], r, g, b);

                cluster_id++;
            }
        }

        // Publish the concatenated clusters point cloud
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*clusters_cloud, output);
        output.header.frame_id = msg->header.frame_id;  // Retain the original frame
        output.header.stamp = this->get_clock()->now();

        cluster_pub_->publish(output);
    }

    // Categorize cluster by size
    std::string categorize_cluster(int cluster_size) {
        if (cluster_size < 500) {
            return "Small";
        } else if (cluster_size >= 500 && cluster_size <= 5000) {
            return "Medium";
        } else {
            return "Large";
        }
    }

    // ROS2 Subscriber and Publisher
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cluster_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudClusterer>());
    rclcpp::shutdown();
    return 0;
}

