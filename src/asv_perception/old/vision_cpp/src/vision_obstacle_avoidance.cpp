#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <fau_msgs/msg/vision.hpp>
#include <cmath>
#include <vector>
#include <tuple>
#include <limits>
#include <algorithm>

using std::placeholders::_1;

class VisionObstacleAvoidance : public rclcpp::Node {
public:
    VisionObstacleAvoidance()
    : Node("vision_obstacle_avoidance"),
      occ_grid_range_(40.0),
      grid_size_(0.5),
      lidar_sensitivity_(100.0),
      vehicle_dim_width_(2.0),
      vehicle_dim_length_(3.0),
      max_index_x_((2 * (occ_grid_range_ / grid_size_))),
      x_array_f_(max_index_x_ * max_index_x_, 0.0),
      y_array_f_(max_index_x_ * max_index_x_, 0.0),
      x_array_b_(max_index_x_ * max_index_x_, 0.0),
      y_array_b_(max_index_x_ * max_index_x_, 0.0),
      x_array_l_(max_index_x_ * max_index_x_, 0.0),
      y_array_l_(max_index_x_ * max_index_x_, 0.0),
      x_array_r_(max_index_x_ * max_index_x_, 0.0),
      y_array_r_(max_index_x_ * max_index_x_, 0.0),
      array_(max_index_x_ * max_index_x_, 0),
      array_old_(max_index_x_ * max_index_x_, 0)
    {
        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/vision/pc2/voxel", 1, std::bind(&VisionObstacleAvoidance::pointCloudCallback, this, _1));
        
        pub_occ_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("vision/occupancy_grid", 1);
        pub_vis_f_ = this->create_publisher<fau_msgs::msg::Vision>("vision/obstacle_avoidance/front", 1);
        pub_vis_b_ = this->create_publisher<fau_msgs::msg::Vision>("vision/obstacle_avoidance/back", 1);
        pub_vis_l_ = this->create_publisher<fau_msgs::msg::Vision>("vision/obstacle_avoidance/left", 1);
        pub_vis_r_ = this->create_publisher<fau_msgs::msg::Vision>("vision/obstacle_avoidance/right", 1);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        // Clear previous data before starting new processing
        std::fill(array_old_.begin(), array_old_.end(), 0);

        int count_obj_f = 0, count_obj_b = 0, count_obj_l = 0, count_obj_r = 0;

        while (iter_x != iter_x.end()) {
            float x = *iter_x, y = *iter_y;
            if (std::abs(x) <= occ_grid_range_ && std::abs(y) <= occ_grid_range_) {
                // Perform grid size rounding
                int grid_x = std::round((x + occ_grid_range_) / grid_size_);
                int grid_y = std::round((y + occ_grid_range_) / grid_size_);
                int index = grid_x + grid_y * max_index_x_;

                // Update grid occupancy
                array_old_[index] += (1 * lidar_sensitivity_ / 10.0);
                array_old_[index] = std::min(100, array_old_[index]);

                // Determine object positions for obstacle avoidance
                if (std::abs(x) >= vehicle_dim_length_ / 2.0 && std::abs(y) >= vehicle_dim_width_ / 2.0) {
                    if (x >= 2.0 && std::abs(y) <= 2.0) {
                        x_array_f_[count_obj_f] = x;
                        y_array_f_[count_obj_f] = -y;
                        count_obj_f++;
                    } else if (x <= -2.0 && std::abs(y) <= 2.0) {
                        x_array_b_[count_obj_b] = x;
                        y_array_b_[count_obj_b] = -y;
                        count_obj_b++;
                    } else if (y >= 2.0) {
                        x_array_l_[count_obj_l] = x;
                        y_array_l_[count_obj_l] = -y;
                        count_obj_l++;
                    } else if (y <= -2.0) {
                        x_array_r_[count_obj_r] = x;
                        y_array_r_[count_obj_r] = -y;
                        count_obj_r++;
                    }
                }
            }

            ++iter_x;
            ++iter_y;
            ++iter_z;
        }

        publishOccupancyGrid();
        publishVisionData(pub_vis_f_, x_array_f_, y_array_f_, count_obj_f, "front");
        publishVisionData(pub_vis_b_, x_array_b_, y_array_b_, count_obj_b, "back");
        publishVisionData(pub_vis_l_, x_array_l_, y_array_l_, count_obj_l, "left");
        publishVisionData(pub_vis_r_, x_array_r_, y_array_r_, count_obj_r, "right");
    }

    void publishOccupancyGrid() {
        auto msg = nav_msgs::msg::OccupancyGrid();
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "velodyne";

        // Convert std::vector<int> to std::vector<signed char>
        std::vector<signed char> occupancy_data(array_old_.begin(), array_old_.end());
        msg.data = occupancy_data;

        msg.info.resolution = grid_size_;
        msg.info.width = static_cast<unsigned int>(std::ceil(max_index_x_));
        msg.info.height = static_cast<unsigned int>(std::ceil(max_index_x_));

        geometry_msgs::msg::Pose pose;
        pose.position.x = -occ_grid_range_;
        pose.position.y = -occ_grid_range_;
        pose.position.z = 0.0;
        pose.orientation.w = 1.0;
        msg.info.origin = pose;

        pub_occ_->publish(msg);
    }

    void publishVisionData(rclcpp::Publisher<fau_msgs::msg::Vision>::SharedPtr &pub,
                           std::vector<float> &x_array, std::vector<float> &y_array,
                           int count_obj, const std::string &direction) {
        if (count_obj > 0) {
            auto msg = fau_msgs::msg::Vision();
            msg.object_avoid = "DANGER";
            msg.quantity_avoid = count_obj;
            msg.x_avoid.resize(count_obj);
            msg.y_avoid.resize(count_obj);
            std::copy(x_array.begin(), x_array.begin() + count_obj, msg.x_avoid.begin());
            std::copy(y_array.begin(), y_array.begin() + count_obj, msg.y_avoid.begin());
            pub->publish(msg);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_occ_;
    rclcpp::Publisher<fau_msgs::msg::Vision>::SharedPtr pub_vis_f_;
    rclcpp::Publisher<fau_msgs::msg::Vision>::SharedPtr pub_vis_b_;
    rclcpp::Publisher<fau_msgs::msg::Vision>::SharedPtr pub_vis_l_;
    rclcpp::Publisher<fau_msgs::msg::Vision>::SharedPtr pub_vis_r_;

    double occ_grid_range_;
    double grid_size_;
    double lidar_sensitivity_;
    double vehicle_dim_width_;
    double vehicle_dim_length_;
    int max_index_x_;

    std::vector<int> array_;
    std::vector<int> array_old_;
    std::vector<float> x_array_f_;
    std::vector<float> y_array_f_;
    std::vector<float> x_array_b_;
    std::vector<float> y_array_b_;
    std::vector<float> x_array_l_;
    std::vector<float> y_array_l_;
    std::vector<float> x_array_r_;
    std::vector<float> y_array_r_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisionObstacleAvoidance>());
    rclcpp::shutdown();
    return 0;
}

