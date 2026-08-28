#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/search/kdtree.h"
#include "pcl/segmentation/extract_clusters.h"
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cmath>
#include "lidar_msgs/msg/buoy_detected.hpp"

class PointCloudClusterer : public rclcpp::Node {
public:
    PointCloudClusterer() : Node("ball_detector") {
        declare_parameters();

        // Get parameters from YAML or use defaults
        input_ros_cloud_   = this->get_parameter("input_ros_cloud").as_string();
        min_x_size_        = this->get_parameter("min_x_size").as_double();
        min_y_size_        = this->get_parameter("min_y_size").as_double();
        min_z_size_        = this->get_parameter("min_z_size").as_double();
        max_x_size_        = this->get_parameter("max_x_size").as_double();
        max_y_size_        = this->get_parameter("max_y_size").as_double();
        max_z_size_        = this->get_parameter("max_z_size").as_double();
        min_detect_dist_   = this->get_parameter("min_detect_dist").as_double();
        max_detect_dist_   = this->get_parameter("max_detect_dist").as_double();
        cluster_tolerance_ = this->get_parameter("cluster_tolerance").as_double();
        min_cluster_size_  = this->get_parameter("min_cluster_size").as_int();
        max_cluster_size_  = this->get_parameter("max_cluster_size").as_int();
        min_detect_height_ = this->get_parameter("min_detect_height").as_double();
        max_detect_height_ = this->get_parameter("max_detect_height").as_double();

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_ros_cloud_, 10,
            std::bind(&PointCloudClusterer::pointcloud_callback, this, std::placeholders::_1)
        );

        cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("clustered_points", 10);
        buoy_detected_pub_ = this->create_publisher<lidar_msgs::msg::BuoyDetected>("/vision/output/ball_detected", 1);
    }

private:
    void declare_parameters() {
        this->declare_parameter<std::string>("input_ros_cloud", "/velodyne_points");
        this->declare_parameter<double>("min_x_size", 0.2);
        this->declare_parameter<double>("min_y_size", 0.2);
        this->declare_parameter<double>("min_z_size", 0.2);
        this->declare_parameter<double>("max_x_size", 2.0);
        this->declare_parameter<double>("max_y_size", 2.0);
        this->declare_parameter<double>("max_z_size", 2.8);
        this->declare_parameter<double>("min_detect_dist", 1.8);
        this->declare_parameter<double>("max_detect_dist", 30.0);
        this->declare_parameter<double>("cluster_tolerance", 0.3);
        this->declare_parameter<int>("min_cluster_size", 1);
        this->declare_parameter<int>("max_cluster_size", 25000);
        this->declare_parameter<double>("min_detect_height", -5.0);
        this->declare_parameter<double>("max_detect_height", 1.0);
    }

    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, indices);

        if (cloud_filtered->empty()) {
            RCLCPP_WARN(this->get_logger(), "Filtered cloud is empty after removing NaNs!");
            return;
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_filtered);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(cluster_tolerance_);
        ec.setMinClusterSize(min_cluster_size_);
        ec.setMaxClusterSize(max_cluster_size_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_filtered);
        ec.extract(cluster_indices);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusters_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        int cluster_id = 0;

        for (const auto& indices : cluster_indices) {
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

            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cluster, centroid);

            pcl::PointXYZRGB min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);

            float width_x  = max_pt.x - min_pt.x;
            float width_y  = max_pt.y - min_pt.y;
            float height_z = max_pt.z - min_pt.z;
            float dist = std::sqrt((centroid[0] * centroid[0]) + (centroid[1] * centroid[1]));

            if (width_x >= min_x_size_ && width_x <= max_x_size_ &&
                width_y >= min_y_size_ && width_y <= max_y_size_ &&
                height_z >= min_z_size_ && height_z <= max_z_size_ &&
                dist >= min_detect_dist_ && dist <= max_detect_dist_ &&
                centroid[2] >= min_detect_height_ && centroid[2] <= max_detect_height_) {

                for (auto& point : cluster->points) {
                    point.r = 255;
                    point.g = 92;
                    point.b = 0;
                }

                *clusters_cloud += *cluster;

                buoy_detected_msg.name.push_back("Ball[" + std::to_string(cluster_id) + "]");
                buoy_detected_msg.x.push_back(centroid[0]);
                buoy_detected_msg.y.push_back(centroid[1]);
                buoy_detected_msg.z.push_back(centroid[2]);

                cluster_id++;
            }
        }

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*clusters_cloud, output);
        output.header.frame_id = msg->header.frame_id;
        output.header.stamp = this->get_clock()->now();

        cluster_pub_->publish(output);
        buoy_detected_pub_->publish(buoy_detected_msg);

        buoy_detected_msg.name.clear();
        buoy_detected_msg.x.clear();
        buoy_detected_msg.y.clear();
        buoy_detected_msg.z.clear();
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cluster_pub_;
    rclcpp::Publisher<lidar_msgs::msg::BuoyDetected>::SharedPtr buoy_detected_pub_;

    std::string input_ros_cloud_;
    double min_x_size_, min_y_size_, min_z_size_;
    double max_x_size_, max_y_size_, max_z_size_;
    double min_detect_dist_, max_detect_dist_, cluster_tolerance_;
    double min_detect_height_, max_detect_height_;
    int min_cluster_size_, max_cluster_size_;
    lidar_msgs::msg::BuoyDetected buoy_detected_msg;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudClusterer>());
    rclcpp::shutdown();
    return 0;
}