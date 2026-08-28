#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace cv;
using namespace std;

class CameraDriver : public rclcpp::Node
{
public:
    CameraDriver() : Node("camera_driver")
    {
        // ✅ Update this with your new ID_PATH:
        //const std::string target_camera_path = "platform-3610000.usb-usb-0:4.1.3:1.0";
        //const std::string target_camera_path = "platform-3610000.usb-usb-0:4.1.2:1.0";
        const std::string target_camera_path = "platform-3610000.usb-usb-0:4.1.4:1.0";

        std::string camera_device = find_camera(target_camera_path);
        if (camera_device.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No matching camera found.");
            return;
        }

        // Open the camera
        camera_.open(camera_device, cv::CAP_V4L2);
        if (!camera_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open camera at %s", camera_device.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Camera opened successfully at %s", camera_device.c_str());

        // Set camera parameters
        camera_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        camera_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        camera_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        camera_.set(cv::CAP_PROP_FPS, 120);

        // ✅ Initialize ROS2 Publishers
        raw_pub_ = this->create_publisher<sensor_msgs::msg::Image>("vision/cam/right/raw", 1);
        lower_res_pub_ = this->create_publisher<sensor_msgs::msg::Image>("vision/cam/right/lower_res", 1);

        // Create a timer to capture and publish frames
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&CameraDriver::capture_and_publish, this));
    }

private:
    std::string find_camera(const std::string &target_path)
    {
        for (int i = 0; i < 10; ++i) {
            std::string device_path = "/dev/video" + std::to_string(i);

            std::string cmd = "udevadm info --query=all --name=" + device_path;
            FILE *pipe = popen(cmd.c_str(), "r");
            if (!pipe) continue;

            char buffer[512];
            std::string result = "";
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                result += buffer;
            }
            pclose(pipe);

            // ✅ Check for ID_PATH
            if (result.find(target_path) != std::string::npos) {
                return device_path;
            }
        }
        return "";
    }

    void capture_and_publish()
    {
        Mat frame;
        camera_ >> frame;

        if (frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "Empty frame captured");
            return;
        }

        // Publish raw image
        publish_image(frame, raw_pub_);

        // Publish lower resolution image
        Mat lower_res;
        resize(frame, lower_res, Size(frame.cols / 2, frame.rows / 2));
        publish_image(lower_res, lower_res_pub_);
    }

    void publish_image(const Mat &image, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub)
    {
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();
        pub->publish(*msg);
    }

    // ✅ ROS2 Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr lower_res_pub_;

    cv::VideoCapture camera_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraDriver>());
    rclcpp::shutdown();
    return 0;
}

