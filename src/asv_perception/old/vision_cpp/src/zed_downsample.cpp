#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.hpp>

class DownsampleNode : public rclcpp::Node
{
public:
    DownsampleNode() : Node("downsample_node")
    {
        // Create a subscriber to the ZED camera topic
        image_sub_ = image_transport::create_subscription(
            this, "/zed_hl1/z/right_raw/image_raw_color",
            std::bind(&DownsampleNode::imageCallback, this, std::placeholders::_1),
            "raw");

        // Create a publisher for the downsampled image
        image_pub_ = image_transport::create_publisher(this, "/vision/downsampled_image");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        // Convert ROS2 image to OpenCV image
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Downsample the image (resize to half its size as an example)
        cv::Mat downsampled_image;
        cv::resize(cv_ptr->image, downsampled_image, cv::Size(), 0.02, 0.02, cv::INTER_LINEAR);

        // Convert back to ROS2 message
        cv_bridge::CvImage out_msg;
        out_msg.header = msg->header;  // Keep the original timestamp and frame_id
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = downsampled_image;

        // Publish the downsampled image
        image_pub_.publish(out_msg.toImageMsg());
    }

    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DownsampleNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

