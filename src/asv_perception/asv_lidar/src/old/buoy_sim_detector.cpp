#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <lidar_msgs/msg/buoy_detected.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cmath>
#include <std_msgs/msg/int64.hpp>


class ObjectDetector : public rclcpp::Node {
public:
    ObjectDetector() : Node("object_detector") {
        rclcpp::QoS qos_profile = rclcpp::QoS(10).best_effort().durability_volatile();

        std::string wamv_name;
        this->declare_parameter("wamv", "wamv1");
        this->get_parameter("wamv", wamv_name);
        // Subscribe to LaserScan topic
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/" + wamv_name + "/scan", qos_profile, std::bind(&ObjectDetector::scan_callback, this, std::placeholders::_1));


        // Publishers for buoys and docks
        buoy_pub_ = this->create_publisher<lidar_msgs::msg::BuoyDetected>("/vision/output/buoy_detected", 10);
        dock_pub_ = this->create_publisher<lidar_msgs::msg::BuoyDetected>("/vision/output/dock_detected", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vision/output/object_markers", 10);
    }

private:
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        double angle = msg->angle_min;

        // Convert LaserScan to PointCloud with filtering

        for (const auto& r : msg->ranges) {
            if (r >= msg->range_min && r <= 20.0) {
                double x = r * std::cos(angle);
                double y = r * std::sin(angle);
                if (angle >= -M_PI_2 && angle <= M_PI_2) {
                    cloud->points.emplace_back(x, y, 0.0);
                }
            }
            angle += msg->angle_increment;
        }
        

        if (cloud->points.empty()) return;

        // Cluster extraction
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(3.0);
        ec.setMinClusterSize(2);
        ec.setMaxClusterSize(1000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        // Messages and markers
        lidar_msgs::msg::BuoyDetected buoy_msg;
        lidar_msgs::msg::BuoyDetected dock_msg;
        visualization_msgs::msg::MarkerArray marker_array;
        int marker_id = 0;

        int buoy_count = 0;
        int dock_count = 0;
        for (const auto& indices : cluster_indices) {
            double sum_x = 0.0, sum_y = 0.0;
            for (auto idx : indices.indices) {
                sum_x += cloud->points[idx].x;
                sum_y += cloud->points[idx].y;
            }

            double centroid_x = sum_x / indices.indices.size();
            double centroid_y = sum_y / indices.indices.size();
            double distance = std::sqrt(centroid_x * centroid_x + centroid_y * centroid_y);

            if (distance <= 60.0) {
                if (indices.indices.size() < 10) { // Buoy condition
                    buoy_msg.name.push_back("buoy_" + std::to_string(buoy_count++));
                    buoy_msg.x.push_back(centroid_x);
                    buoy_msg.y.push_back(centroid_y);
                    buoy_msg.z.push_back(0.0f);
                    add_marker(marker_array, centroid_x, centroid_y, marker_id++, "Buoy", 0.0, 1.0, 0.0);
                } else if (indices.indices.size() > 15) { // Dock condition
                    dock_msg.name.push_back("dock_" + std::to_string(dock_count++));
                    dock_msg.x.push_back(centroid_x);
                    dock_msg.y.push_back(centroid_y);
                    dock_msg.z.push_back(0.0f);
                    add_marker(marker_array, centroid_x, centroid_y, marker_id++, "Dock", 0.0, 0.0, 1.0);
                }
            }
        }

        // Publish detections
        if (!buoy_msg.name.empty()) buoy_pub_->publish(buoy_msg);
        if (!dock_msg.name.empty()) dock_pub_->publish(dock_msg);
        if (!marker_array.markers.empty()) marker_pub_->publish(marker_array);
    }

    void add_marker(visualization_msgs::msg::MarkerArray& marker_array, double x, double y, int id, const std::string& label, float r, float g, float b) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "wamv/wamv/base_link";
        marker.header.stamp = this->get_clock()->now();
        marker.ns = label + "_marker";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = 0.0;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.color.a = 1.0;
        marker_array.markers.push_back(marker);

        visualization_msgs::msg::Marker text_marker = marker;
        text_marker.id = id + 1000;
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.text = label;
        text_marker.scale.z = 0.3;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        marker_array.markers.push_back(text_marker);
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr pp_state_sub;
    rclcpp::Publisher<lidar_msgs::msg::BuoyDetected>::SharedPtr buoy_pub_;
    rclcpp::Publisher<lidar_msgs::msg::BuoyDetected>::SharedPtr dock_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    int PP_state = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObjectDetector>());
    rclcpp::shutdown();
    return 0;
}
