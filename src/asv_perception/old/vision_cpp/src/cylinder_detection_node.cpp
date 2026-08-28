#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

class SemiCylinderDetectionNode : public rclcpp::Node {
public:
    SemiCylinderDetectionNode()
    : Node("semi_cylinder_detection_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 1, std::bind(&SemiCylinderDetectionNode::pointCloudCallback, this, std::placeholders::_1));
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Parse the point cloud data
        std::vector<geometry_msgs::msg::Point> points;
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        while (iter_x != iter_x.end()) {
            geometry_msgs::msg::Point point;
            point.x = *iter_x;
            point.y = *iter_y;
            point.z = *iter_z;
            points.push_back(point);

            ++iter_x;
            ++iter_y;
            ++iter_z;
        }

        // Detect semi-cylinders (posts, buoys, etc.)
        std::vector<std::vector<geometry_msgs::msg::Point>> detected_objects = detectSemiCylinders(points);

        // Print the centroids of detected objects
        for (const auto& object : detected_objects) {
            if (!object.empty()) {
                geometry_msgs::msg::Point centroid = computeCentroid(object);
                RCLCPP_INFO(this->get_logger(), "Object detected at centroid (x: %.2f, y: %.2f, z: %.2f)",
                            centroid.x, centroid.y, centroid.z);
            }
        }
    }

    std::vector<std::vector<geometry_msgs::msg::Point>> detectSemiCylinders(const std::vector<geometry_msgs::msg::Point>& points) {
        std::vector<std::vector<geometry_msgs::msg::Point>> objects;
        std::vector<geometry_msgs::msg::Point> current_object;

        for (const auto& point : points) {
            // Calculate the radial distance from the origin
            float radius = std::sqrt(point.x * point.x + point.y * point.y);

            // Check if the point fits the semi-cylinder radius criteria (e.g., 0.1 to 0.5 meters)
            if (radius >= 0.1 && radius <= 0.5) {
                current_object.push_back(point);
            }
        }

        // Group points by height (Z-axis) to detect distinct semi-cylinders
        if (!current_object.empty()) {
            std::sort(current_object.begin(), current_object.end(), [](const auto& a, const auto& b) {
                return a.z < b.z;
            });

            std::vector<geometry_msgs::msg::Point> current_group;
            float height_threshold = 0.5; // Height tolerance for grouping

            for (size_t i = 0; i < current_object.size(); ++i) {
                if (i > 0 && (std::abs(current_object[i].z - current_object[i - 1].z) > height_threshold)) {
                    // Save the previous group as a detected object if it meets height requirements
                    if (isValidSemiCylinder(current_group)) {
                        objects.push_back(current_group);
                    }
                    current_group.clear();
                }
                current_group.push_back(current_object[i]);
            }

            // Check the last group
            if (!current_group.empty() && isValidSemiCylinder(current_group)) {
                objects.push_back(current_group);
            }
        }

        return objects;
    }

    bool isValidSemiCylinder(const std::vector<geometry_msgs::msg::Point>& points) {
        // Simple heuristic: Check if points form a vertical cluster with a certain curvature

        if (points.size() < 5) {  // Not enough points for a meaningful detection
            return false;
        }

        // Calculate the angular spread in the XY plane
        float min_angle = std::atan2(points.front().y, points.front().x);
        float max_angle = std::atan2(points.back().y, points.back().x);
        float angular_spread = std::abs(max_angle - min_angle);

        // Check if the points are spread out enough to represent a semi-cylinder (arc)
        return (angular_spread > M_PI / 6); // Roughly 30 degrees spread
    }

    geometry_msgs::msg::Point computeCentroid(const std::vector<geometry_msgs::msg::Point>& points) {
        geometry_msgs::msg::Point centroid;
        float sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

        for (const auto& point : points) {
            sum_x += point.x;
            sum_y += point.y;
            sum_z += point.z;
        }

        centroid.x = sum_x / points.size();
        centroid.y = sum_y / points.size();
        centroid.z = sum_z / points.size();

        return centroid;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SemiCylinderDetectionNode>());
    rclcpp::shutdown();
    return 0;
}

