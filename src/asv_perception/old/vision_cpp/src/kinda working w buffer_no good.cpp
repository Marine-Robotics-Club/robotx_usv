#include <rclcpp/rclcpp.hpp>
#include <yolov26_msgs/msg/yolo_detection.hpp>
#include <std_msgs/msg/string.hpp>
#include <cmath>
#include <vector>
#include <unordered_map>

struct Detection {
    std::string class_name;
    double x_center;
    double y_center;
    double timestamp;
};

class StereoFusionNode : public rclcpp::Node {
public:
    StereoFusionNode() : Node("stereo_fusion_node") {
        left_sub_ = this->create_subscription<yolov26_msgs::msg::YoloDetection>(
            "/yolov9/detections/left", 10,
            std::bind(&StereoFusionNode::left_callback, this, std::placeholders::_1));
        
        right_sub_ = this->create_subscription<yolov26_msgs::msg::YoloDetection>(
            "/yolov9/detections", 10,
            std::bind(&StereoFusionNode::right_callback, this, std::placeholders::_1));

        fused_pub_ = this->create_publisher<std_msgs::msg::String>("/fused_detections", 10);
    }

private:
    void left_callback(const yolov26_msgs::msg::YoloDetection::SharedPtr msg) {
        update_buffer(msg, "left");
    }

    void right_callback(const yolov26_msgs::msg::YoloDetection::SharedPtr msg) {
        update_buffer(msg, "right");
    }

    void update_buffer(const yolov26_msgs::msg::YoloDetection::SharedPtr msg, const std::string& source) {
        const double time_now = this->now().seconds();
        for (size_t i = 0; i < msg->class_name.size(); ++i) {
            Detection det = {msg->class_name[i], msg->x_center[i], msg->y_center[i], time_now};
            if (source == "left") {
                left_buffer_.push_back(det);
            } else {
                right_buffer_.push_back(det);
            }
        }
        fuse_detections();
    }

    void fuse_detections() {
        const double baseline = 0.12; // meters
        const double focal_length = 1066.0; // pixels
        const double image_center_x = 1280 / 2;
        const double time_threshold = 0.1; // seconds for matching

        for (const auto& left_det : left_buffer_) {
            for (const auto& right_det : right_buffer_) {
                if (left_det.class_name == right_det.class_name &&
                    std::abs(left_det.timestamp - right_det.timestamp) <= time_threshold) {
                    double disparity = right_det.x_center - left_det.x_center;
                    if (disparity > 0) {
                        double distance = (baseline * focal_length) / disparity;
                        double x_center_avg = (left_det.x_center + right_det.x_center) / 2;
                        double angle = atan2(x_center_avg - image_center_x, focal_length) * (180.0 / M_PI);

                        auto fused_msg = std_msgs::msg::String();
                        fused_msg.data = "Class: " + left_det.class_name + 
                                         ", Angle: " + std::to_string(angle) + 
                                         " degrees, Distance: " + std::to_string(distance) + " meters";
                        fused_pub_->publish(fused_msg);
                        RCLCPP_INFO(this->get_logger(), "%s", fused_msg.data.c_str());
                    }
                }
            }
        }
        left_buffer_.clear();
        right_buffer_.clear();
    }

    // Subscribers and Publisher
    rclcpp::Subscription<yolov26_msgs::msg::YoloDetection>::SharedPtr left_sub_;
    rclcpp::Subscription<yolov26_msgs::msg::YoloDetection>::SharedPtr right_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fused_pub_;

    // Buffers to hold detections
    std::vector<Detection> left_buffer_;
    std::vector<Detection> right_buffer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoFusionNode>());
    rclcpp::shutdown();
    return 0;
}

