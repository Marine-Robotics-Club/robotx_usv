#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/search/kdtree.h"
#include "pcl/segmentation/extract_clusters.h"
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/common/io.h>

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "lidar_msgs/msg/buoy_detected.hpp"
#include "fau_msgs/msg/object_position.hpp"

class PointCloudClusterer : public rclcpp::Node {
public:
    PointCloudClusterer() : Node("buoy_detector_vrx") {
        declare_parameters();

        // Get parameters
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
        min_detect_height_ = this->get_parameter("min_detect_height").as_double();
        max_detect_height_ = this->get_parameter("max_detect_height").as_double();
        radius_scale_ = this->get_parameter("radius_scale").as_double();
        min_radius_m_ = this->get_parameter("min_radius_m").as_double();

        std::string wamv_name;
        this->get_parameter("wamv", wamv_name);
        wamv_ = wamv_name;

        auto qos = rclcpp::SensorDataQoS();

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/" + wamv_name + "/lidar_wamv/points_no_water",
            qos,
            std::bind(&PointCloudClusterer::pointcloud_callback, this, std::placeholders::_1)
        );

        cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/" + wamv_name + "/clustered_buoys", 10);

        wamv_detected_pub_ = this->create_publisher<lidar_msgs::msg::BuoyDetected>(
            "/" + wamv_name + "/vision/output/buoy_detected", 10);

        fau_objects_pub_ = this->create_publisher<fau_msgs::msg::ObjectPosition>(
            "/" + wamv_name + "/vision/output/buoy_objects", 10);

        RCLCPP_INFO(this->get_logger(), "buoy_detector_vrx subscribed to: %s",
                    ("/" + wamv_name + "/lidar_wamv/points_no_water").c_str());
        RCLCPP_INFO(this->get_logger(), "publishing clustered cloud: /%s/clustered_buoys", wamv_name.c_str());
        RCLCPP_INFO(this->get_logger(), "publishing lidar_msgs detections: /%s/vision/output/buoy_detected", wamv_name.c_str());
        RCLCPP_INFO(this->get_logger(), "publishing fau_msgs objects: /%s/vision/output/buoy_objects", wamv_name.c_str());
    }

private:
    void declare_parameters() {
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

        // New parameters
        this->declare_parameter<double>("radius_scale", 1.0);
        this->declare_parameter<double>("min_radius_m", 0.10);
        this->declare_parameter<std::string>("wamv", "wamv1");
    }

    static inline bool finite3(float x, float y, float z) {
        return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
    }

    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (msg->header.frame_id.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Incoming PointCloud2 has empty frame_id. Dropping.");
            return;
        }

        if ((msg->width * msg->height) == 0) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Incoming cloud has 0 points. Skipping.");
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        if (!cloud || cloud->empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "PCL cloud empty after conversion. Skipping.");
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> nan_indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, nan_indices);

        if (!cloud_filtered || cloud_filtered->empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Cloud empty after NaN removal. Skipping.");
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

        wamv_detected_msg_.name.clear();
        wamv_detected_msg_.x.clear();
        wamv_detected_msg_.y.clear();
        wamv_detected_msg_.z.clear();

        fau_msgs::msg::ObjectPosition fau_msg;
        fau_msg.object_names.clear();
        fau_msg.x_object.clear();
        fau_msg.y_object.clear();
        fau_msg.z_object.clear();
        fau_msg.radii_object.clear();

        int cluster_id = 0;

        for (const auto& cluster_idx : cluster_indices) {
            if (cluster_idx.indices.empty()) {
                continue;
            }

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
            cluster->points.reserve(cluster_idx.indices.size());

            for (const auto& idx : cluster_idx.indices) {
                const auto& src = cloud_filtered->points[idx];
                if (!pcl::isFinite(src)) {
                    continue;
                }

                pcl::PointXYZRGB p;
                p.x = src.x;
                p.y = src.y;
                p.z = src.z;
                cluster->points.push_back(p);
            }

            if (cluster->points.empty()) {
                continue;
            }

            cluster->width = static_cast<uint32_t>(cluster->points.size());
            cluster->height = 1;
            cluster->is_dense = true;

            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cluster, centroid);

            if (!finite3(centroid[0], centroid[1], centroid[2])) {
                continue;
            }

            if (centroid[2] < min_detect_height_ || centroid[2] > max_detect_height_) {
                continue;
            }

            pcl::PointXYZRGB min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);

            const float width_x  = max_pt.x - min_pt.x;
            const float width_y  = max_pt.y - min_pt.y;
            const float height_z = max_pt.z - min_pt.z;
            const float dist = std::sqrt((centroid[0] * centroid[0]) + (centroid[1] * centroid[1]));

            if (!(width_x >= min_x_size_ && width_x <= max_x_size_ &&
                  width_y >= min_y_size_ && width_y <= max_y_size_ &&
                  height_z >= min_z_size_ && height_z <= max_z_size_ &&
                  dist >= min_detect_dist_ && dist <= max_detect_dist_)) {
                continue;
            }

            // Radius from XY half-diagonal of bbox
            const float rx = 0.5f * width_x;
            const float ry = 0.5f * width_y;
            float r_xy = std::sqrt(rx * rx + ry * ry);

            if (!std::isfinite(r_xy)) {
                continue;
            }

            r_xy = std::max(static_cast<float>(min_radius_m_), r_xy);
            r_xy = static_cast<float>(radius_scale_) * r_xy;

            for (auto& p : cluster->points) {
                p.r = 118;
                p.g = 37;
                p.b = 251;
            }

            *clusters_cloud += *cluster;

            const std::string name = "BUOY[" + std::to_string(cluster_id) + "]";

            // lidar_msgs output
            wamv_detected_msg_.name.push_back(name);
            wamv_detected_msg_.x.push_back(centroid[0]);
            wamv_detected_msg_.y.push_back(centroid[1]);
            wamv_detected_msg_.z.push_back(centroid[2]);

            // fau_msgs output
            fau_msg.object_names.push_back(name);
            fau_msg.x_object.push_back(static_cast<double>(centroid[0]));
            fau_msg.y_object.push_back(static_cast<double>(centroid[1]));
            fau_msg.z_object.push_back(static_cast<double>(centroid[2]));
            fau_msg.radii_object.push_back(static_cast<double>(r_xy));

            cluster_id++;
        }

        if (clusters_cloud->empty()) {
            return;
        }

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*clusters_cloud, output);
        output.header.frame_id = msg->header.frame_id;
        output.header.stamp = msg->header.stamp;
        cluster_pub_->publish(output);

        if (!wamv_detected_msg_.name.empty()) {
            wamv_detected_pub_->publish(wamv_detected_msg_);
        }

        if (!fau_msg.object_names.empty()) {
            fau_objects_pub_->publish(fau_msg);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cluster_pub_;
    rclcpp::Publisher<lidar_msgs::msg::BuoyDetected>::SharedPtr wamv_detected_pub_;
    rclcpp::Publisher<fau_msgs::msg::ObjectPosition>::SharedPtr fau_objects_pub_;

    double min_x_size_, min_y_size_, min_z_size_;
    double max_x_size_, max_y_size_, max_z_size_;
    double min_detect_dist_, max_detect_dist_, cluster_tolerance_;
    int min_cluster_size_, max_cluster_size_;
    double min_detect_height_, max_detect_height_;

    double radius_scale_{1.0};
    double min_radius_m_{0.10};

    std::string wamv_;

    lidar_msgs::msg::BuoyDetected wamv_detected_msg_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudClusterer>());
    rclcpp::shutdown();
    return 0;
}