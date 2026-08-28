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
#include <cmath>  // For std::abs()
#include "lidar_msgs/msg/buoy_detected.hpp"

//could try clustering in local frame and then find the probable frame of the clustered WAMV this allows you to then have a reference
// that you can filer the detected wamv in its own x,y,z frame. this is how u could only see the wamVs.
//cluster "probable WAMVs" -> deduce there own local frame -> filter upon that local frame -> 

class PointCloudClusterer : public rclcpp::Node {
public:
    PointCloudClusterer() : Node("wamv_detector_vrx") {
        declare_parameters();

        // Get parameters from YAML or use defaults
        min_x_size_ = this->get_parameter("min_x_size").as_double();
        min_y_size_ = this->get_parameter("min_y_size").as_double();
        min_z_size_ = this->get_parameter("min_z_size").as_double();
        max_x_size_ = this->get_parameter("max_x_size").as_double();
        max_y_size_ = this->get_parameter("max_y_size").as_double();
        max_z_size_ = this->get_parameter("max_z_size").as_double();
        min_detect_dist_ = this->get_parameter("min_detect_dist").as_double();
        max_detect_dist_ = this->get_parameter("max_detect_dist").as_double();
        cluster_tolerance_ = this->get_parameter("cluster_tolerance").as_double();
        min_cluster_size_ = this->get_parameter("min_cluster_size").as_int();
        max_cluster_size_ = this->get_parameter("max_cluster_size").as_int();

        RCLCPP_INFO(this->get_logger(), "Parameters:");
        RCLCPP_INFO(this->get_logger(), "  min_x_size: %.2f", min_x_size_);
        RCLCPP_INFO(this->get_logger(), "  min_y_size: %.2f", min_y_size_);
        RCLCPP_INFO(this->get_logger(), "  min_z_size: %.2f", min_z_size_);
        RCLCPP_INFO(this->get_logger(), "  max_x_size: %.2f", max_x_size_);
        RCLCPP_INFO(this->get_logger(), "  max_y_size: %.2f", max_y_size_);
        RCLCPP_INFO(this->get_logger(), "  max_z_size: %.2f", max_z_size_);
        RCLCPP_INFO(this->get_logger(), "  min_detect_dist: %.2f", min_detect_dist_);
        RCLCPP_INFO(this->get_logger(), "  max_detect_dist: %.2f", max_detect_dist_);
        RCLCPP_INFO(this->get_logger(), "  cluster_tolerance: %.2f", cluster_tolerance_);
        RCLCPP_INFO(this->get_logger(), "  min_cluster_size: %d", min_cluster_size_);
        RCLCPP_INFO(this->get_logger(), "  max_cluster_size: %d", max_cluster_size_);

        std::string wamv_name;
        this->declare_parameter("wamv", "wamv1");
        this->get_parameter("wamv", wamv_name);

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/" + wamv_name + "/lidar_wamv/points_no_water",
            rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
            std::bind(&PointCloudClusterer::pointcloud_callback, this, std::placeholders::_1)
        );


        // Publisher for clustered point clouds
        cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/" + wamv_name + "/clustered_wamv_points", 10);
        wamv_detected_pub= this->create_publisher<lidar_msgs::msg::BuoyDetected>("/" + wamv_name + "/vision/output/wamv_detected",0);
    }
    
private:
    // Declare parameters
    void declare_parameters() {
        this->declare_parameter<double>("min_x_size", 0.2);
        this->declare_parameter<double>("min_y_size", 0.2);
        this->declare_parameter<double>("min_z_size", 0.8);
        this->declare_parameter<double>("max_x_size", 0.8);
        this->declare_parameter<double>("max_y_size", 0.8);
        this->declare_parameter<double>("max_z_size", 2.8);
        this->declare_parameter<double>("min_detect_dist", 1.8);
        this->declare_parameter<double>("max_detect_dist", 30.0);
        this->declare_parameter<double>("cluster_tolerance", 0.3);
        this->declare_parameter<int>("min_cluster_size", 20);
        this->declare_parameter<int>("max_cluster_size", 25000);
    }

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
        ec.setClusterTolerance(cluster_tolerance_);  // Set distance threshold (30 cm)
        ec.setMinClusterSize(min_cluster_size_);     // Minimum number of points in a cluster
        ec.setMaxClusterSize(max_cluster_size_);  // Maximum number of points in a cluster
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
            float dist = sqrt(std::abs((centroid[0] * centroid[0]) + (centroid[1] * centroid[1])));

            // Filter based on the size criteria
            if (width_x >= min_x_size_ && max_x_size_ >= width_x 
                && width_y >= min_y_size_ && max_y_size_ >= width_y 
                && height_z >= min_z_size_ && max_z_size_ >= height_z
                && dist >= min_detect_dist_ && dist <= max_detect_dist_) { //&& min_pt.x >= 4.0
                // If the cluster meets the size criteria, add it to the output cloud

                // Generate a random color for the current cluster
                uint8_t r = static_cast<uint8_t>(dis(gen));
                uint8_t g = static_cast<uint8_t>(dis(gen));
                uint8_t b = static_cast<uint8_t>(dis(gen));

                // Add the current cluster to the combined clusters cloud
                for (auto& point : cluster->points) {
                    point.r = 118;
                    point.g = 37;
                    point.b = 251;
                }

                *clusters_cloud += *cluster;

                // Cluster size category
                std::string cluster_size = "NORMAL";

                // Print cluster information
                //RCLCPP_INFO(this->get_logger(),"Cluster ID: %d | Size: %s | Dimensions (X, Y, Z): [%.2f, %.2f, %.2f] | Centroid: [%.2f, %.2f, %.2f] | Color: RGB(%d, %d, %d)",cluster_id, cluster_size.c_str(), width_x, width_y, height_z, centroid[0], centroid[1], centroid[2], r, g, b);
                //RCLCPP_WARN(this->get_logger(), "Getting clusters");
                wamv_detected_msg.name.push_back("WAMV[" + std::to_string(cluster_id) + "]");
                wamv_detected_msg.x.push_back(centroid[0]);
                wamv_detected_msg.y.push_back(centroid[1]);
                wamv_detected_msg.z.push_back(centroid[2]);
                
                cluster_id++;

            }
        }

        // Publish the concatenated clusters point cloud
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*clusters_cloud, output);
        output.header.frame_id = msg->header.frame_id;  // Retain the original frame
        output.header.stamp = this->get_clock()->now();

        cluster_pub_->publish(output);
        wamv_detected_pub->publish(wamv_detected_msg);
        
        wamv_detected_msg.name.clear();
        wamv_detected_msg.x.clear();
        wamv_detected_msg.y.clear();
        wamv_detected_msg.z.clear();
    }

    // ROS2 Subscriber and Publisher
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cluster_pub_;
    rclcpp::Publisher<lidar_msgs::msg::BuoyDetected>::SharedPtr wamv_detected_pub;

    // Parameters
    std::string input_ros_cloud_;
    double min_x_size_, min_y_size_, min_z_size_;
    double max_x_size_, max_y_size_, max_z_size_;
    double min_detect_dist_, max_detect_dist_, cluster_tolerance_;
    int min_cluster_size_, max_cluster_size_;
    lidar_msgs::msg::BuoyDetected wamv_detected_msg;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudClusterer>());
    rclcpp::shutdown();
    return 0;
}

