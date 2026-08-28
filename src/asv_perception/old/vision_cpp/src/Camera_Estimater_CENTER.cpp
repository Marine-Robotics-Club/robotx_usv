#include "rclcpp/rclcpp.hpp"
#include "yolov26_msgs/msg/yolo_detection.hpp"
#include "fau_msgs/msg/object_position_array.hpp"
#include <cmath>
#include <unordered_map>

class YoloEstimatorNode : public rclcpp::Node
{
public:
    YoloEstimatorNode() : Node("yolo_estimator_node_center")
    {
        // Initialize reference values for specific object types (width in pixels, reference distance in meters)
        reference_data_ = {
            {"black_cross", {270.0, 1.0}},
            {"black_triangle", {270.0, 1.0}},
            {"green_pole_buoy", {130.0, 1.0}},
            {"red_pole_buoy", {130.0, 1.0}},
            {"red_buoy", {90.0, 1.0}},
            {"green_buoy", {90.0, 1.0}},
            {"yellow_buoy", {135.0, 1.0}}, //testing right now. 
            {"black_buoy", {90.0, 1.0}}
        };

        // Subscribe to YOLO detection topic
        yolo_sub_ = this->create_subscription<yolov26_msgs::msg::YoloDetection>(
            "/yolov9/center/detections", 10,
            std::bind(&YoloEstimatorNode::yoloCallback, this, std::placeholders::_1));

        // Publisher for object position estimates
        position_pub_ = this->create_publisher<fau_msgs::msg::ObjectPositionArray>(
            "/vision/output/position_estimates", 10);
    }

private:
    void yoloCallback(const yolov26_msgs::msg::YoloDetection::SharedPtr msg)
    {
        fau_msgs::msg::ObjectPositionArray position_estimates;

        for (size_t i = 0; i < msg->class_name.size(); ++i)
        {
            std::string object_type = msg->class_name[i];

            // Process only objects that exist in reference_data_
            if (reference_data_.find(object_type) == reference_data_.end())
                continue;

            // Retrieve reference width and distance
            double reference_width = reference_data_[object_type].first;
            double reference_distance = reference_data_[object_type].second;

            double angle, magnitude;
            calculateAngleMagnitude(*msg, i, angle, magnitude, reference_width, reference_distance);

            // Scale magnitude
            magnitude *= 2.0;

            // Store estimated positions
            position_estimates.object_names.push_back(object_type);
            position_estimates.x_object.push_back(magnitude * cos(-1 * angle));
            position_estimates.y_object.push_back(magnitude * sin(-1 * angle));
            position_estimates.z_object.push_back(0.0); // Assuming 2D estimation in NWU frame

            RCLCPP_INFO(this->get_logger(), "Detection [%zu] (%s): Angle: %.2f°, Magnitude: %.2f meters", 
                        i, object_type.c_str(), -1 * angle * (180.0 / M_PI), magnitude);
        }

        // Only publish if valid objects were detected
        if (!position_estimates.object_names.empty())
        {
            position_pub_->publish(position_estimates);
        }
    }

    void calculateAngleMagnitude(const yolov26_msgs::msg::YoloDetection &detection, size_t index,
                                 double &angle, double &magnitude, double reference_width, double reference_distance)
    {
        // ZED 2i camera parameters (1280x640 resolution)
        const double fov_x = 110.0 * (M_PI / 180.0); // Horizontal FOV in radians
        const double cx = 640.0;                     // Center x-coordinate (half of 1280)

        // Compute azimuth angle from bounding box center
        double px = detection.x_center[index];
        angle = (px - cx) / cx * (fov_x / 2.0);

        // Compute bounding box width for magnitude estimation
        double width = detection.x_max[index] - detection.x_min[index];
        magnitude = reference_distance * (reference_width / width);
    }

    rclcpp::Subscription<yolov26_msgs::msg::YoloDetection>::SharedPtr yolo_sub_;
    rclcpp::Publisher<fau_msgs::msg::ObjectPositionArray>::SharedPtr position_pub_;

    // Reference width and distance per object type
    std::unordered_map<std::string, std::pair<double, double>> reference_data_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloEstimatorNode>());
    rclcpp::shutdown();
    return 0;
}

