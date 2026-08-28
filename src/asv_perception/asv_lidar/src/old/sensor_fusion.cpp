#include "rclcpp/rclcpp.hpp"
#include "lidar_msgs/msg/buoy_detected.hpp"  // Custom message from the lidar
#include "yolov26_msgs/msg/yolo_detection.hpp"  // Custom YOLO message for bounding boxes
#include "fau_msgs/msg/object_position_array.hpp"
#include <vector>
#include <string>
#include <cmath>
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"

class SensorFusionNode : public rclcpp::Node
{
public:
    SensorFusionNode() : Node("sensor_fusion_node")
    {

        std::string wamv_name;
        this->declare_parameter("wamv", "wamv1");
        this->get_parameter("wamv", wamv_name);

        // Subscriptions to lidar topics
        wamv_sub_ = this->create_subscription<lidar_msgs::msg::BuoyDetected>
            ("/" + wamv_name + "/vision/output/buoy_detected", 10, std::bind(&SensorFusionNode::wamv_callback, this, std::placeholders::_1));

        // Subscription to YOLO bounding box topic
        yolo_sub_ = this->create_subscription<yolov26_msgs::msg::YoloDetection>(
            "/yolov26/detections", 10, std::bind(&SensorFusionNode::yolo_callback, this, std::placeholders::_1));
            
        fusion_pub_ = this->create_publisher<fau_msgs::msg::ObjectPositionArray>("/" + wamv_name + "/vision/output/fusion", 10);
        rviz_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/" + wamv_name + "/vision/output/rviz_fusion", 10);
        
        // Initialize sensor limits
        min_angle_ = -45* (M_PI/180) ;  // -45 degrees
        max_angle_ = 45* (M_PI/180) ;  // +45 degrees
        max_angle_2 = 90* (M_PI/180) ;
        min_height_ = -3.0;       // Lower bound for height
        max_height_ = 3.0;        // Upper bound for height
        max_length_ = 150.0;       // Maximum distance from the LiDAR
    }

private:
    // Callback for lidar buoy data
    void wamv_callback(const lidar_msgs::msg::BuoyDetected::SharedPtr msg)
    {
        wamv.object_names.clear();
        wamv.x_object.clear();
        wamv.y_object.clear();
        wamv.z_object.clear();

        for (size_t i = 0; i < msg->name.size(); ++i) {
            double angle = atan2(msg->y[i], msg->x[i]);
            double distance = sqrt(msg->x[i] * msg->x[i] + msg->y[i] * msg->y[i]);

            if (angle >= min_angle_ && angle <= max_angle_ &&
                msg->z[i] >= min_height_ && msg->z[i] <= max_height_ &&
                distance <= max_length_) 
            {
                wamv.object_names.push_back(msg->name[i]);
                wamv.x_object.push_back(msg->x[i]);
                wamv.y_object.push_back(msg->y[i]);
                wamv.z_object.push_back(msg->z[i]);
            }
        }
    }
    


    // Callback for YOLO data
    void yolo_callback(const yolov26_msgs::msg::YoloDetection::SharedPtr msg)
    {
        objects.class_name.clear();
        objects.confidence.clear();
        objects.x_min.clear();
        objects.y_min.clear();
        objects.x_max.clear();
        objects.y_max.clear();
        objects.x_center.clear();
        objects.y_center.clear();

        objects = *msg;

        if (!objects.class_name.empty()) {
            sensor_fusion_decision();
        }
    }

    // Sensor fusion decision-making
    void sensor_fusion_decision()
    {
        fusion.object_names.clear();
        fusion.x_object.clear();
        fusion.y_object.clear();
        fusion.z_object.clear();

        for (size_t i = 0; i < objects.class_name.size(); ++i) {
        
            // Check for stop light matches
            for (size_t j = 0; j < wamv.object_names.size(); ++j) {
                if (objects.class_name[i] == "WAM-V") {
                    double search_size = 1.5;
                    double left_angle = atan2((wamv.y_object[j] + search_size), wamv.x_object[j]);
                    double right_angle = atan2((wamv.y_object[j] - search_size), wamv.x_object[j]);
                    double left_pixel = ((1280.0 / -max_angle_2) * left_angle) + 640;
                    double right_pixel = ((1280.0 / -max_angle_2) * right_angle) + 640;

                    if ((objects.x_center[i] >= left_pixel) && (objects.x_center[i] <= right_pixel)) {
                        RCLCPP_INFO(this->get_logger(), "SF:WAM-V!!! xmin: %f xmax: %f", left_pixel, right_pixel);
                        fusion.object_names.push_back(objects.class_name[i]);
                        fusion.x_object.push_back(wamv.x_object[j]);
                        fusion.y_object.push_back(wamv.y_object[j]);
                        fusion.z_object.push_back(wamv.z_object[j]);
                    }
                }
            }

        }
        if (!fusion.object_names.empty()) {
            fusion_pub_->publish(fusion);
            publish_to_rviz(fusion);
        }
    }
    
            // Convert fused object positions to NWU and publish for RViz
    void publish_to_rviz(const fau_msgs::msg::ObjectPositionArray &fusion_data)
    {
        geometry_msgs::msg::PoseArray pose_array_msg;
        pose_array_msg.header.stamp = this->get_clock()->now();
        pose_array_msg.header.frame_id = "velodyne"; // Change as needed for RViz
        
        pose_array_msg.poses.clear();

        for (size_t i = 0; i < fusion_data.x_object.size(); ++i) {
            geometry_msgs::msg::Pose pose;
            
            // Apply -90 degree rotation for NWU transformation
            pose.position.x = fusion_data.x_object[i];  // 
            pose.position.y = fusion_data.y_object[i];   // 
            pose.position.z = fusion_data.z_object[i];   // z remains unchanged
            
            pose_array_msg.poses.push_back(pose);
        }

        rviz_pub_->publish(pose_array_msg);
    }

    rclcpp::Subscription<lidar_msgs::msg::BuoyDetected>::SharedPtr wamv_sub_;
    rclcpp::Subscription<yolov26_msgs::msg::YoloDetection>::SharedPtr yolo_sub_;
    rclcpp::Publisher<fau_msgs::msg::ObjectPositionArray>::SharedPtr fusion_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr rviz_pub_;
    
    // Data structures
    fau_msgs::msg::ObjectPositionArray wamv;
    yolov26_msgs::msg::YoloDetection objects;
    fau_msgs::msg::ObjectPositionArray fusion;
    
    // Sensor limits
    double min_angle_;
    double max_angle_;
    double min_height_;
    double max_height_;
    double max_length_;
    double max_angle_2;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SensorFusionNode>());
    rclcpp::shutdown();
    return 0;
}

