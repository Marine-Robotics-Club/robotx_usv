#include <rclcpp/rclcpp.hpp>
#include <yolov26_msgs/msg/yolo_detection.hpp>
#include <std_msgs/msg/string.hpp> // For publishing the output as a string message
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cmath>
#include <limits>

class StereoFusionNode : public rclcpp::Node {
public:
    StereoFusionNode() : Node("stereo_fusion_node") {
        // Set up subscribers
        left_sub_ = std::make_shared<message_filters::Subscriber<yolov26_msgs::msg::YoloDetection>>(this, "/yolov9/detections/left");
        right_sub_ = std::make_shared<message_filters::Subscriber<yolov26_msgs::msg::YoloDetection>>(this, "/yolov9/detections");

        // Define ApproximateTime policy and synchronizer
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), *left_sub_, *right_sub_);
        sync_->registerCallback(std::bind(&StereoFusionNode::callback, this, 
                                          std::placeholders::_1, std::placeholders::_2));
        
        // Set up publisher for the fused output as a string
        fused_pub_ = this->create_publisher<std_msgs::msg::String>("/fused_detections", 10);
    }

private:
    void callback(const std::shared_ptr<const yolov26_msgs::msg::YoloDetection> left_msg, 
                  const std::shared_ptr<const yolov26_msgs::msg::YoloDetection> right_msg) {
        const double baseline = 0.12;        // Baseline in meters
        const double focal_length = 1066.0;  // Focal length in pixels for HD1080
        const double image_center_x = 1280 / 2;
        const double matching_threshold = 50.0; // Pixel threshold for detection matching

        for (size_t i = 0; i < left_msg->class_name.size(); ++i) {
            int best_match_index = -1;
            double smallest_disparity = std::numeric_limits<double>::max();

            // Find the closest match in right detections
            for (size_t j = 0; j < right_msg->class_name.size(); ++j) {
                if (left_msg->class_name[i] == right_msg->class_name[j]) {
                    double disparity = right_msg->x_center[j] - left_msg->x_center[i];

                    // Check for valid disparity and match within pixel threshold
                    if (disparity > 0 && disparity < smallest_disparity &&
                        std::abs(right_msg->x_center[j] - left_msg->x_center[i]) < matching_threshold) {
                        smallest_disparity = disparity;
                        best_match_index = j;
                    }
                }
            }

            if (best_match_index == -1) {
                RCLCPP_WARN(this->get_logger(), "No valid match found for detection %zu", i);
                continue;
            }

            // Calculate distance (magnitude) and angle using matched disparity
            double distance = (baseline * focal_length) / smallest_disparity;
            double x_center_avg = (left_msg->x_center[i] + right_msg->x_center[best_match_index]) / 2;
            double angle = atan2(x_center_avg - image_center_x, focal_length) * (180.0 / M_PI);

            // Format the output as a string
            auto fused_msg = std_msgs::msg::String();
            fused_msg.data = "Class: " + left_msg->class_name[i] + 
                             ", Angle: " + std::to_string(angle) + 
                             " degrees, Distance: " + std::to_string(distance) + " meters";

            fused_pub_->publish(fused_msg);

            RCLCPP_INFO(this->get_logger(), "%s", fused_msg.data.c_str());
        }
    }

    // Subscribers, Sync Policy, Synchronizer, and Publisher
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<yolov26_msgs::msg::YoloDetection, yolov26_msgs::msg::YoloDetection>;
    std::shared_ptr<message_filters::Subscriber<yolov26_msgs::msg::YoloDetection>> left_sub_;
    std::shared_ptr<message_filters::Subscriber<yolov26_msgs::msg::YoloDetection>> right_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fused_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoFusionNode>());
    rclcpp::shutdown();
    return 0;
}

