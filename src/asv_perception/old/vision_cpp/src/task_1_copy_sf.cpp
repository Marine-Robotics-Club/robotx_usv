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
#include <limits>
#include <string>
#include <unordered_map>
#include <cstdint>

class OccupancyGridPathGenerator : public rclcpp::Node
{
public:
    OccupancyGridPathGenerator()
        : Node("occupancy_grid_path_generator"),
          grid_resolution_(0.1),
          grid_width_(295),
          grid_height_(140),
          min_safe_distance_(2.0),
          z_min_m_(0.0),
          min_obj_points_(3)
    {
        // Optional: make tangent distance configurable
        this->declare_parameter<double>("tangent_len_m", 5.0);
        tangent_len_ = this->get_parameter("tangent_len_m").as_double();

        // Minimum Z threshold for points to be considered (local frame)
        this->declare_parameter<double>("z_min_m", 0.2);
        z_min_m_ = this->get_parameter("z_min_m").as_double();

        // NEW: Minimum number of points in a grid-cell "object" to count as an object
        // You asked: "more than 4 points" -> default 5
        this->declare_parameter<int>("min_obj_points", 5);
        min_obj_points_ = this->get_parameter("min_obj_points").as_int();

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10,
            std::bind(&OccupancyGridPathGenerator::pointcloud_callback, this, std::placeholders::_1));

        boat_sub_ = this->create_subscription<lidar_msgs::msg::BuoyDetected>(
            "/vision/output/boat_detected", 10,
            std::bind(&OccupancyGridPathGenerator::boat_callback, this, std::placeholders::_1));

        grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/vision/output/occupancy_grid", 10);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/path_vision_1", 10);
        buoy_detected_pub_ = this->create_publisher<lidar_msgs::msg::BuoyDetected>("/vision/output/task1_buoy_detected", 10);

        RCLCPP_INFO(this->get_logger(),
                    "Node Initialized. tangent_len_m=%.2f, z_min_m=%.3f, min_obj_points=%d",
                    tangent_len_, z_min_m_, min_obj_points_);
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
            double distance = std::sqrt(msg->x[i] * msg->x[i] + msg->y[i] * msg->y[i]);
            if (distance <= 30.0) {
                detected_boats_.push_back({msg->name[i], msg->x[i], msg->y[i], msg->z[i]});
            }
        }
    }

    // Helper to pack grid_x, grid_y into one 64-bit key for unordered_map
    static inline uint64_t pack_key(int gx, int gy)
    {
        return (static_cast<uint64_t>(static_cast<uint32_t>(gx)) << 32) |
               (static_cast<uint64_t>(static_cast<uint32_t>(gy)));
    }

    struct CellAccum {
        int count = 0;
        double sum_x = 0.0;
        double sum_y = 0.0;
    };

    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        nav_msgs::msg::OccupancyGrid grid_msg;
        grid_msg.header.stamp = this->now();
        grid_msg.header.frame_id = "velodyne";

        grid_msg.info.resolution = grid_resolution_;
        grid_msg.info.width = grid_width_;
        grid_msg.info.height = grid_height_;
        grid_msg.info.origin.position.x = 0.5;
        grid_msg.info.origin.position.y = -7.0;
        grid_msg.info.origin.position.z = 0.0;
        grid_msg.info.origin.orientation.w = 1.0;

        grid_msg.data.assign(grid_width_ * grid_height_, -1);

        // NEW: per-cell accumulators (instead of pushing every point)
        std::unordered_map<uint64_t, CellAccum> cell_map;
        cell_map.reserve(static_cast<size_t>(msg->width * msg->height / 20));

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
        {
            const float x = *iter_x;
            const float y = *iter_y;
            const float z = *iter_z;

            // Z FILTER (your request)
            if (z < static_cast<float>(z_min_m_)) {
                continue;
            }

            const double angle = std::atan2(y, x) * (180.0 / M_PI);

            if (x >= 1.5f && x <= 25.0f &&
                y >= -6.0f && y <= 6.0f &&
                angle >= -60.0 && angle <= 60.0)
            {
                const int grid_x = static_cast<int>((x - 0.5f) / grid_resolution_);
                const int grid_y = static_cast<int>((y - (-7.0f)) / grid_resolution_);

                if (grid_x >= 0 && grid_x < grid_width_ && grid_y >= 0 && grid_y < grid_height_)
                {
                    // mark occupancy
                    grid_msg.data[grid_y * grid_width_ + grid_x] = 100;

                    // accumulate into this cell
                    const uint64_t key = pack_key(grid_x, grid_y);
                    auto &acc = cell_map[key];
                    acc.count += 1;
                    acc.sum_x += x;
                    acc.sum_y += y;
                }
            }
        }

        // Build "objects" from cells with enough points
        std::vector<std::pair<float, float>> detected_objects;
        detected_objects.reserve(cell_map.size());

        for (const auto &kv : cell_map)
        {
            const CellAccum &acc = kv.second;
            if (acc.count >= min_obj_points_)  // your request: >4 points
            {
                const float cx = static_cast<float>(acc.sum_x / acc.count);
                const float cy = static_cast<float>(acc.sum_y / acc.count);
                detected_objects.emplace_back(cx, cy);
            }
        }

        grid_pub_->publish(grid_msg);

        // Optional debug:
        // RCLCPP_INFO(this->get_logger(), "Cells=%zu, Objects(count>=%d)=%zu",
        //             cell_map.size(), min_obj_points_, detected_objects.size());

        process_path(detected_objects);
    }

    bool is_safe_path(float x, float y) const
    {
        for (const auto& obj : detected_boats_) {
            double distance = std::sqrt((x - obj.x)*(x - obj.x) + (y - obj.y)*(y - obj.y));
            if (distance < min_safe_distance_) {
                return false;
            }
        }
        return true;
    }

    bool is_in_region(float x, float y) const
    {
        if (x < 0.5f || x > 15.0f) return false;
        if (y < -6.0f || y > 6.0f) return false;
        double ang = std::atan2(y, x) * (180.0 / M_PI);
        return (ang >= -60.0 && ang <= 60.0);
    }

    bool compute_perpendicular_tangent_point(
        float x1, float y1,
        float x2, float y2,
        float mx, float my,
        float target_len,
        float min_len,
        float &out_x3, float &out_y3)
    {
        float vx = x2 - x1;
        float vy = y2 - y1;
        float norm = std::sqrt(vx*vx + vy*vy);
        if (norm < 1e-3f) return false;

        float nx = -vy / norm;
        float ny =  vx / norm;

        const float step = 0.25f;
        for (float len = target_len; len >= min_len; len -= step)
        {
            float c1x = mx + len * nx;
            float c1y = my + len * ny;
            float c2x = mx - len * nx;
            float c2y = my - len * ny;

            bool c1_ok = (c1x >= mx) && is_in_region(c1x, c1y) && is_safe_path(c1x, c1y);
            bool c2_ok = (c2x >= mx) && is_in_region(c2x, c2y) && is_safe_path(c2x, c2y);

            if (!c1_ok && !c2_ok) continue;

            if (c1_ok && c2_ok) {
                if (c1x >= c2x) { out_x3 = c1x; out_y3 = c1y; }
                else            { out_x3 = c2x; out_y3 = c2y; }
                return true;
            }
            if (c1_ok) { out_x3 = c1x; out_y3 = c1y; return true; }
            if (c2_ok) { out_x3 = c2x; out_y3 = c2y; return true; }
        }

        return false;
    }

    void process_path(const std::vector<std::pair<float, float>> &objects)
    {
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->now();
        path_msg.header.frame_id = "velodyne";

        if (objects.size() < 2)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough objects detected for path generation.");
            path_pub_->publish(path_msg);
            return;
        }

        bool found_path = false;
        float min_distance = std::numeric_limits<float>::max();
        std::pair<float, float> best_midpoint{0.0f, 0.0f};

        float best_x1 = 0.0f, best_y1 = 0.0f;
        float best_x2 = 0.0f, best_y2 = 0.0f;

        for (size_t i = 0; i < objects.size(); ++i)
        {
            float x_i = objects[i].first, y_i = objects[i].second;

            for (size_t p = i + 1; p < objects.size(); ++p)
            {
                float x_p = objects[p].first, y_p = objects[p].second;

                float dx = std::abs(x_i - x_p);
                float dy = std::abs(y_i - y_p);
                float dist = std::sqrt((x_p - x_i)*(x_p - x_i) + (y_p - y_i)*(y_p - y_i));

                if (dx >= 0.5f && dx < 2.5f && dy < 6.0f && dy > 1.0f && dist < min_distance)
                {
                    float x_mid = (x_i + x_p) * 0.5f;
                    float y_mid = (y_i + y_p) * 0.5f;
                    double mid_angle = std::atan2(y_mid, x_mid) * (180.0 / M_PI);

                    if (is_safe_path(x_mid, y_mid) && mid_angle >= -60.0 && mid_angle <= 60.0)
                    {
                        min_distance = dist;
                        best_midpoint = {x_mid, y_mid};
                        best_x1 = x_i; best_y1 = y_i;
                        best_x2 = x_p; best_y2 = y_p;
                        found_path = true;
                    }
                }
            }
        }

        if (!found_path)
        {
            RCLCPP_WARN(this->get_logger(), "No valid buoy pair found for midpoint, publishing empty path.");
            path_pub_->publish(path_msg);
            return;
        }

        geometry_msgs::msg::PoseStamped pose1, pose2, pose3;
        pose1.header = path_msg.header;
        pose1.pose.position.x = 0.0;
        pose1.pose.position.y = 0.0;
        pose1.pose.position.z = 0.0;
        pose1.pose.orientation.w = 1.0;

        pose2.header = path_msg.header;
        pose2.pose.position.x = best_midpoint.first;
        pose2.pose.position.y = best_midpoint.second;
        pose2.pose.position.z = 0.0;
        pose2.pose.orientation.w = 1.0;

        path_msg.poses.push_back(pose1);
        path_msg.poses.push_back(pose2);

        float x3 = 0.0f, y3 = 0.0f;

        if (compute_perpendicular_tangent_point(
                best_x1, best_y1,
                best_x2, best_y2,
                best_midpoint.first, best_midpoint.second,
                static_cast<float>(tangent_len_),
                0.75f,
                x3, y3))
        {
            pose3.header = path_msg.header;
            pose3.pose.position.x = x3;
            pose3.pose.position.y = y3;
            pose3.pose.position.z = 0.0;
            pose3.pose.orientation.w = 1.0;

            path_msg.poses.push_back(pose3);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(),
                        "Midpoint found, but could not place perpendicular tangent point (forward + safe + bounds).");
        }

        buoy_detected_msg.name.clear();
        buoy_detected_msg.x.clear();
        buoy_detected_msg.y.clear();
        buoy_detected_msg.z.clear();

        buoy_detected_msg.name.push_back("Buoy1");
        buoy_detected_msg.x.push_back(best_x1);
        buoy_detected_msg.y.push_back(best_y1);
        buoy_detected_msg.z.push_back(0);

        buoy_detected_msg.name.push_back("Buoy2");
        buoy_detected_msg.x.push_back(best_x2);
        buoy_detected_msg.y.push_back(best_y2);
        buoy_detected_msg.z.push_back(0);

        buoy_detected_pub_->publish(buoy_detected_msg);

        path_pub_->publish(path_msg);
    }

private:
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
    double tangent_len_;

    double z_min_m_;
    int min_obj_points_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OccupancyGridPathGenerator>());
    rclcpp::shutdown();
    return 0;
}