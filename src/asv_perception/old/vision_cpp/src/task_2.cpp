#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "lidar_msgs/msg/buoy_detected.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

class OccupancyGridPathGenerator : public rclcpp::Node
{
public:
    OccupancyGridPathGenerator()
        : Node("occupancy_grid_path_generator"),
          grid_resolution_(0.1),  // 10 cm per cell
          grid_width_(295),       // 29.5m range (x: 0.5m to 30m, 295 cells)
          grid_height_(140),      // 14m range (y: -7m to 7m, 140 cells)
          min_safe_distance_(3.0) // Minimum safe distance for path points
    {
        // Subscriber for Velodyne point cloud
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10,
            std::bind(&OccupancyGridPathGenerator::pointcloud_callback, this, std::placeholders::_1));

        // Subscriber for Lidar boat detection
        boat_sub_ = this->create_subscription<lidar_msgs::msg::BuoyDetected>(
            "/vision/output/boat_detected", 10,
            std::bind(&OccupancyGridPathGenerator::boat_callback, this, std::placeholders::_1));

        // Publisher for occupancy grid
        grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/vision/output/occupancy_grid", 10);

        // Publisher for path visualization
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/path_vision_2", 10);

        buoy_detected_pub_= this->create_publisher<lidar_msgs::msg::BuoyDetected>("/vision/output/task2_buoy_detected",0);
        
        RCLCPP_INFO(this->get_logger(), "Occupancy Grid & Path Node Initialized with angle filtering.");
    }

private:
    struct Object {
        std::string name;
        float x, y, z;
    };

    std::vector<Object> detected_boats_;

    void boat_callback(const lidar_msgs::msg::BuoyDetected::SharedPtr msg)
    {
        detected_boats_.clear();

        for (size_t i = 0; i < msg->name.size(); ++i) {
            double angle = std::atan2(msg->y[i], msg->x[i]) * (180.0 / M_PI); // Convert to degrees
            double distance = std::sqrt(msg->x[i] * msg->x[i] + msg->y[i] * msg->y[i]);
            float max_length_ = 15.0; // Max detection range

            // Only store boats within max range
            if (distance <= max_length_) {
                detected_boats_.push_back({msg->name[i], msg->x[i], msg->y[i], msg->z[i]});
            }
        }
    }

    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Create an empty occupancy grid
        nav_msgs::msg::OccupancyGrid grid_msg;
        grid_msg.header.stamp = this->now();
        grid_msg.header.frame_id = "velodyne";  // Using "map" for NWU compatibility

        // Set up grid metadata
        grid_msg.info.resolution = grid_resolution_;
        grid_msg.info.width = grid_width_;
        grid_msg.info.height = grid_height_;
        grid_msg.info.origin.position.x = 0.5;  // x_min in NWU
        grid_msg.info.origin.position.y = -7.0; // y_min in NWU
        grid_msg.info.origin.position.z = 0.0;
        grid_msg.info.origin.orientation.w = 1.0;

        // Initialize the grid with unknown (-1)
        grid_msg.data.assign(grid_width_ * grid_height_, -1);

        std::vector<std::pair<float, float>> detected_objects;

        // Iterate through point cloud
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y)
        {
            float x = *iter_x;
            float y = *iter_y;
            double angle = std::atan2(y, x) * (180.0 / M_PI); // Convert to degrees

            // Filter points within the expanded range in NWU and angle range
            if (x >= 0.1 && x <= 5.0 && y >= -4.0 && y <= 4.0 && angle >= -60 && angle <= 60)
            {
                // Convert to grid index
                int grid_x = static_cast<int>((x - 0.5) / grid_resolution_);
                int grid_y = static_cast<int>((y - (-7.0)) / grid_resolution_);

                if (grid_x >= 0 && grid_x < grid_width_ && grid_y >= 0 && grid_y < grid_height_)
                {
                    grid_msg.data[grid_y * grid_width_ + grid_x] = 100; // Mark as occupied
                    detected_objects.push_back({x, y});
                }
            }
        }

        // Publish occupancy grid
        grid_pub_->publish(grid_msg);

        // Process detected objects into a path
        process_path(detected_objects);
    }

    bool is_safe_path(float x, float y)
    {
        for (const auto& obj : detected_boats_) {
            double distance = std::sqrt(std::pow(x - obj.x, 2) + std::pow(y - obj.y, 2));
            if (distance < min_safe_distance_) {
                return false;
            }
        }
        return true;
    }

    void process_path(const std::vector<std::pair<float, float>> &objects)
    {
        if (objects.size() < 2)
        {
            //RCLCPP_WARN(this->get_logger(), "Not enough objects detected for path generation.");
            return;
        }

        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->now();
        path_msg.header.frame_id = "velodyne"; // Using "map" for NWU frame

        bool found_path = false;
        geometry_msgs::msg::PoseStamped pose1, pose2;
        float min_distance = std::numeric_limits<float>::max();
        std::pair<float, float> best_midpoint;
        float x_i_1 = 0;
        float x_p_1;
        float y_i_1;
        float y_p_1;

        RCLCPP_INFO(this->get_logger(), "Detected %lu objects, attempting path generation.", objects.size());

        for (size_t i = 0; i < objects.size(); ++i)
        {
            float x_i = objects[i].first, y_i = objects[i].second;

            for (size_t p = i + 1; p < objects.size(); ++p) // Start at i+1 to avoid self-comparison
            {
                float x_p = objects[p].first, y_p = objects[p].second;

                // Ensure objects are not too close
                float dx = std::abs(x_i - x_p);
                float dy = std::abs(y_i - y_p);
                x_i_1 = x_i;
                y_i_1 = y_i;
                x_p_1 = x_p;
                y_p_1 = y_p;               
                float dist = std::sqrt(std::pow(x_p - x_i, 2) + std::pow(y_p - y_i, 2));

                if (dx >= 0.5 && dx < 3.0 && dy < 6.0 && dy > 1.0 && dist < min_distance)  // Ensure valid spacing
                {
                    float x_mid = (x_i + x_p) / 2.0;
                    float y_mid = (y_i + y_p) / 2.0;
                    double mid_angle = std::atan2(y_mid, x_mid) * (180.0 / M_PI); // Convert to degrees

                    // Ensure path is within the -60 to 60-degree range and is safe
                    if (is_safe_path(x_mid, y_mid) && mid_angle >= -60 && mid_angle <= 60) {
                        min_distance = dist;
                        best_midpoint = {x_mid, y_mid};
                        found_path = true;
                    }
                }
            }
        }

        if (found_path)
        {
            pose1.pose.position.x = 0.0;
            pose1.pose.position.y = 0.0;

            pose2.pose.position.x = best_midpoint.first;
            pose2.pose.position.y = best_midpoint.second;

            path_msg.poses.push_back(pose1);
            path_msg.poses.push_back(pose2);
    		buoy_detected_msg.name.push_back("Buoy1");
			buoy_detected_msg.x.push_back(x_i_1);
			buoy_detected_msg.y.push_back(y_i_1);
			buoy_detected_msg.z.push_back(0);
			buoy_detected_msg.name.push_back("Buoy2");
			buoy_detected_msg.x.push_back(x_p_1);
			buoy_detected_msg.y.push_back(y_p_1);
			buoy_detected_msg.z.push_back(0);
            buoy_detected_pub_->publish(buoy_detected_msg);
        
			buoy_detected_msg.name.clear();
			buoy_detected_msg.x.clear();
			buoy_detected_msg.y.clear();
			buoy_detected_msg.z.clear();
            
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "No valid path found, publishing empty path.");
        }

        path_pub_->publish(path_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<lidar_msgs::msg::BuoyDetected>::SharedPtr boat_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<lidar_msgs::msg::BuoyDetected>::SharedPtr buoy_detected_pub_;
    lidar_msgs::msg::BuoyDetected buoy_detected_msg;

    double grid_resolution_;
    int grid_width_;
    int grid_height_;
    float min_safe_distance_;
};

// ** MAIN FUNCTION **
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OccupancyGridPathGenerator>());
    rclcpp::shutdown();
    return 0;
}