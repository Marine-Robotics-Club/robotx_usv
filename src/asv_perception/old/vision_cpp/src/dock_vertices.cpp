#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/common/pca.h"
#include "pcl/filters/statistical_outlier_removal.h"
#include <Eigen/Dense>
#include <vector>
#include <limits>

class BoundingBoxCalculator : public rclcpp::Node {
public:
    BoundingBoxCalculator() : Node("bounding_box_calculator") {
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "clustered_points_dock", 1,
            std::bind(&BoundingBoxCalculator::pointcloud_callback, this, std::placeholders::_1)
        );
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("bounding_box_marker", 10);
        points_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("bounding_box_points", 10);
    }

private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->points.empty()) {
          //  RCLCPP_WARN(this->get_logger(), "Received empty point cloud!");
            return;
        }

        // Step 1: Filter outliers and points within 3 meters of the origin
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*filtered_cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& point : filtered_cloud->points) {
            float distance_to_origin = std::sqrt(point.x * point.x + point.y * point.y);
            if (distance_to_origin > 3.0) {
                clustered_cloud->points.push_back(point);
            }
        }

        if (clustered_cloud->points.empty()) {
            //RCLCPP_WARN(this->get_logger(), "Filtered point cloud is empty!");
            return;
        }

        // Step 2: Perform PCA on the XY plane for bounding box alignment
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*clustered_cloud, centroid);
        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(clustered_cloud);
        Eigen::Matrix3f eigenvectors = pca.getEigenVectors();

        // Rotate cloud to align with PCA axes (XY only)
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translation() << -centroid[0], -centroid[1], 0;
        transform.linear().block<2, 2>(0, 0) = eigenvectors.block<2, 2>(0, 0).transpose(); // Only rotate XY
        pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*clustered_cloud, *rotated_cloud, transform);

        // Calculate min and max bounds on the XY plane
        pcl::PointXYZ min_pt, max_pt;
        min_pt.x = min_pt.y = std::numeric_limits<float>::max();
        max_pt.x = max_pt.y = std::numeric_limits<float>::lowest();

        for (const auto& point : rotated_cloud->points) {
            if (point.x < min_pt.x) min_pt.x = point.x;
            if (point.y < min_pt.y) min_pt.y = point.y;
            if (point.x > max_pt.x) max_pt.x = point.x;
            if (point.y > max_pt.y) max_pt.y = point.y;
        }

        // Step 3: Ensure the longer side is at least 15m and the shorter side is at least 8m
        float length_x = max_pt.x - min_pt.x;
        float length_y = max_pt.y - min_pt.y;
        bool x_is_longer = length_x > length_y;

        if (x_is_longer) {
            if (length_x < 15.0) {
                float adjustment = (15.0 - length_x) / 2.0;
                min_pt.x -= adjustment;
                max_pt.x += adjustment;
            }
            if (length_y < 8.0) {
                float adjustment = (8.0 - length_y) / 2.0;
                min_pt.y -= adjustment;
                max_pt.y += adjustment;
            }
        } else {
            if (length_y < 15.0) {
                float adjustment = (15.0 - length_y) / 2.0;
                min_pt.y -= adjustment;
                max_pt.y += adjustment;
            }
            if (length_x < 8.0) {
                float adjustment = (8.0 - length_x) / 2.0;
                min_pt.x -= adjustment;
                max_pt.x += adjustment;
            }
        }

        // Define 4 corners in rotated frame (XY only)
        std::vector<Eigen::Vector3f> box_corners(4);
        box_corners[0] = Eigen::Vector3f(min_pt.x, min_pt.y, 0);
        box_corners[1] = Eigen::Vector3f(max_pt.x, min_pt.y, 0);
        box_corners[2] = Eigen::Vector3f(max_pt.x, max_pt.y, 0);
        box_corners[3] = Eigen::Vector3f(min_pt.x, max_pt.y, 0);

        // Transform corners back to the original frame
        Eigen::Affine3f inverse_transform = transform.inverse();
        for (auto &corner : box_corners) {
            corner = inverse_transform * corner;
        }

        // Publish bounding box as a line strip marker in RViz
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = msg->header.frame_id;
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "bounding_box";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.1;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        for (const auto &corner : box_corners) {
            geometry_msgs::msg::Point p;
            p.x = corner[0];
            p.y = corner[1];
            p.z = 0.0;
            marker.points.push_back(p);
        }
        marker.points.push_back(marker.points[0]);
        marker_pub_->publish(marker);

        // Step 4: Publish points at specific percentages along the long edge
        geometry_msgs::msg::PoseArray points_msg;
        points_msg.header.frame_id = msg->header.frame_id;
        points_msg.header.stamp = this->get_clock()->now();
        std::vector<float> percentages = {0.25, 0.4, 0.75};

        Eigen::Vector3f long_edge_start, long_edge_direction;
        if (x_is_longer) {
            long_edge_start = box_corners[0];
            long_edge_direction = (box_corners[1] - box_corners[0]);
        } else {
            long_edge_start = box_corners[0];
            long_edge_direction = (box_corners[3] - box_corners[0]);
        }

        for (float percentage : percentages) {
            Eigen::Vector3f point_position = long_edge_start + percentage * long_edge_direction;

            geometry_msgs::msg::Pose point_pose;
            point_pose.position.x = point_position[0];
            point_pose.position.y = point_position[1];
            point_pose.position.z = 0.0;
            points_msg.poses.push_back(point_pose);
        }

        points_pub_->publish(points_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr points_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BoundingBoxCalculator>());
    rclcpp::shutdown();
    return 0;
}

