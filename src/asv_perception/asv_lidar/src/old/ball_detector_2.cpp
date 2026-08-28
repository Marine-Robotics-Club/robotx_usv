#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"

#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/common/io.h>

#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <deque>
#include <limits>
#include <algorithm>

#include "lidar_msgs/msg/buoy_detected.hpp"
#include "fau_msgs/msg/object_position.hpp"

class PointCloudClusterer : public rclcpp::Node {
public:
    PointCloudClusterer() : Node("ball_detector") {
        declare_parameters();

        min_x_size_ = this->get_parameter("min_x_size").as_double();
        min_y_size_ = this->get_parameter("min_y_size").as_double();
        min_z_size_ = this->get_parameter("min_z_size").as_double();

        max_x_size_ = this->get_parameter("max_x_size").as_double();
        max_y_size_ = this->get_parameter("max_y_size").as_double();
        max_z_size_ = this->get_parameter("max_z_size").as_double();

        min_detect_dist_ = this->get_parameter("min_detect_dist").as_double();
        max_detect_dist_ = this->get_parameter("max_detect_dist").as_double();

        min_detect_height_ = this->get_parameter("min_detect_height").as_double();
        max_detect_height_ = this->get_parameter("max_detect_height").as_double();

        min_cluster_size_ = this->get_parameter("min_cluster_size").as_int();
        max_cluster_size_ = this->get_parameter("max_cluster_size").as_int();

        radius_scale_ = this->get_parameter("radius_scale").as_double();
        min_radius_m_ = this->get_parameter("min_radius_m").as_double();

        rows_ = this->get_parameter("range_rows").as_int();
        cols_ = this->get_parameter("range_cols").as_int();
        v_angle_min_deg_ = this->get_parameter("v_angle_min_deg").as_double();
        v_angle_max_deg_ = this->get_parameter("v_angle_max_deg").as_double();
        horiz_fov_deg_ = this->get_parameter("horiz_fov_deg").as_double();

        range_jump_thresh_m_ = this->get_parameter("range_jump_thresh_m").as_double();
        neighbor_col_window_ = this->get_parameter("neighbor_col_window").as_int();
        neighbor_row_window_ = this->get_parameter("neighbor_row_window").as_int();
        min_range_m_ = this->get_parameter("min_range_m").as_double();
        max_range_m_ = this->get_parameter("max_range_m").as_double();

        accumulation_scans_ = this->get_parameter("accumulation_scans").as_int();
        accumulate_only_when_static_ = this->get_parameter("accumulate_only_when_static").as_bool();
        max_accum_translation_m_ = this->get_parameter("max_accum_translation_m").as_double();
        max_accum_yaw_deg_ = this->get_parameter("max_accum_yaw_deg").as_double();

        std::string wamv_name;
        this->get_parameter("wamv", wamv_name);
        wamv_ = wamv_name;

        auto qos = rclcpp::SensorDataQoS();

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points",
            qos,
            std::bind(&PointCloudClusterer::pointcloud_callback, this, std::placeholders::_1)
        );

        pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose2D>(
            "/" + wamv_name + "/vehicle_pose",
            10,
            std::bind(&PointCloudClusterer::pose_callback, this, std::placeholders::_1)
        );

        cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/" + wamv_name + "/clustered_buoys", 10);

        wamv_detected_pub_ = this->create_publisher<lidar_msgs::msg::BuoyDetected>(
            "/" + wamv_name + "/vision/output/buoy_detected", 10);

        fau_objects_pub_ = this->create_publisher<fau_msgs::msg::ObjectPosition>(
            "/" + wamv_name + "/vision/output/buoy_objects", 10);

        RCLCPP_INFO(this->get_logger(), "Range-image buoy detector with motion-gated accumulation started");
        RCLCPP_INFO(this->get_logger(), "accumulation_scans=%d static_only=%s max_trans=%.3f max_yaw_deg=%.3f",
                    accumulation_scans_,
                    accumulate_only_when_static_ ? "true" : "false",
                    max_accum_translation_m_,
                    max_accum_yaw_deg_);
    }

private:
    struct PixelCell {
        bool valid = false;
        float range = std::numeric_limits<float>::infinity();
        int point_idx = -1;
    };

    struct BufferedScan {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        double x{0.0};
        double y{0.0};
        double yaw{0.0};
    };

    void declare_parameters() {
        this->declare_parameter<std::string>("wamv", "asv");

        this->declare_parameter<double>("min_x_size", 0.005);
        this->declare_parameter<double>("min_y_size", 0.005);
        this->declare_parameter<double>("min_z_size", 0.005);

        this->declare_parameter<double>("max_x_size", 0.80);
        this->declare_parameter<double>("max_y_size", 0.80);
        this->declare_parameter<double>("max_z_size", 0.60);

        this->declare_parameter<double>("min_detect_dist", 0.0);
        this->declare_parameter<double>("max_detect_dist", 10.0);

        this->declare_parameter<double>("min_detect_height", -10.0);
        this->declare_parameter<double>("max_detect_height", 10.0);

        this->declare_parameter<int>("min_cluster_size", 1);
        this->declare_parameter<int>("max_cluster_size", 150);

        this->declare_parameter<double>("radius_scale", 1.3);
        this->declare_parameter<double>("min_radius_m", 0.10);

        this->declare_parameter<int>("range_rows", 16);
        this->declare_parameter<int>("range_cols", 900);
        this->declare_parameter<double>("v_angle_min_deg", -15.0);
        this->declare_parameter<double>("v_angle_max_deg", 15.0);
        this->declare_parameter<double>("horiz_fov_deg", 360.0);

        this->declare_parameter<double>("range_jump_thresh_m", 0.10);
        this->declare_parameter<int>("neighbor_row_window", 1);
        this->declare_parameter<int>("neighbor_col_window", 2);
        this->declare_parameter<double>("min_range_m", 0.0);
        this->declare_parameter<double>("max_range_m", 60.0);

        this->declare_parameter<int>("accumulation_scans", 2);

        // Motion gating
        this->declare_parameter<bool>("accumulate_only_when_static", true);
        this->declare_parameter<double>("max_accum_translation_m", 0.20);
        this->declare_parameter<double>("max_accum_yaw_deg", 5.0);
    }

    static inline bool finite3(float x, float y, float z) {
        return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
    }

    static double wrap_angle(double a) {
        while (a > M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }

    int wrap_col(int c) const {
        while (c < 0) c += cols_;
        while (c >= cols_) c -= cols_;
        return c;
    }

    void pose_callback(const geometry_msgs::msg::Pose2D::SharedPtr msg) {
        current_x_ = msg->x;
        current_y_ = msg->y;
        current_yaw_ = msg->theta;
        have_pose_ = true;
    }

    bool point_to_image_index(const pcl::PointXYZ& p, int& row, int& col, float& range) const {
        if (!pcl::isFinite(p)) {
            return false;
        }

        range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (!std::isfinite(range) || range < min_range_m_ || range > max_range_m_) {
            return false;
        }

        const double az_deg = std::atan2(p.y, p.x) * 180.0 / M_PI;
        const double el_deg = std::atan2(p.z, std::sqrt(p.x * p.x + p.y * p.y)) * 180.0 / M_PI;

        const double v_norm = (el_deg - v_angle_min_deg_) /
                              std::max(1e-6, (v_angle_max_deg_ - v_angle_min_deg_));
        row = static_cast<int>(std::round(v_norm * (rows_ - 1)));

        double az_shift = az_deg + 180.0;
        if (horiz_fov_deg_ < 359.9) {
            az_shift = std::clamp(az_shift, 0.0, horiz_fov_deg_);
        }
        const double h_norm = az_shift / std::max(1e-6, horiz_fov_deg_);
        col = static_cast<int>(std::round(h_norm * (cols_ - 1)));

        if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            return false;
        }
        return true;
    }

    std::vector<std::vector<int>> range_image_segment(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) const
    {
        std::vector<PixelCell> grid(rows_ * cols_);
        auto idx_rc = [&](int r, int c) { return r * cols_ + c; };

        for (int i = 0; i < static_cast<int>(cloud->points.size()); ++i) {
            const auto& p = cloud->points[i];

            int row, col;
            float range;
            if (!point_to_image_index(p, row, col, range)) {
                continue;
            }

            auto& cell = grid[idx_rc(row, col)];
            if (!cell.valid || range < cell.range) {
                cell.valid = true;
                cell.range = range;
                cell.point_idx = i;
            }
        }

        std::vector<int> labels(rows_ * cols_, -1);
        std::vector<std::vector<int>> clusters;
        int cluster_id = 0;

        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                const int start_idx = idx_rc(r, c);
                if (!grid[start_idx].valid || labels[start_idx] != -1) {
                    continue;
                }

                std::queue<std::pair<int, int>> q;
                q.push({r, c});
                labels[start_idx] = cluster_id;

                std::vector<int> cluster_point_indices;

                while (!q.empty()) {
                    auto [cr, cc] = q.front();
                    q.pop();

                    const int cur_flat = idx_rc(cr, cc);
                    const auto& cur_cell = grid[cur_flat];
                    if (!cur_cell.valid) {
                        continue;
                    }

                    cluster_point_indices.push_back(cur_cell.point_idx);

                    for (int dr = -neighbor_row_window_; dr <= neighbor_row_window_; ++dr) {
                        for (int dc = -neighbor_col_window_; dc <= neighbor_col_window_; ++dc) {
                            if (dr == 0 && dc == 0) {
                                continue;
                            }

                            const int nr = cr + dr;
                            if (nr < 0 || nr >= rows_) {
                                continue;
                            }

                            const int nc = wrap_col(cc + dc);
                            const int nb_flat = idx_rc(nr, nc);

                            if (labels[nb_flat] != -1 || !grid[nb_flat].valid) {
                                continue;
                            }

                            const float r1 = cur_cell.range;
                            const float r2 = grid[nb_flat].range;

                            if (std::fabs(r1 - r2) <= range_jump_thresh_m_) {
                                labels[nb_flat] = cluster_id;
                                q.push({nr, nc});
                            }
                        }
                    }
                }

                if (!cluster_point_indices.empty()) {
                    clusters.push_back(cluster_point_indices);
                    cluster_id++;
                }
            }
        }

        return clusters;
    }

    bool motion_small_enough(double last_x, double last_y, double last_yaw) const {
        const double dx = current_x_ - last_x;
        const double dy = current_y_ - last_y;
        const double dtrans = std::sqrt(dx * dx + dy * dy);
        const double dyaw_deg = std::abs(wrap_angle(current_yaw_ - last_yaw)) * 180.0 / M_PI;

        return (dtrans <= max_accum_translation_m_) && (dyaw_deg <= max_accum_yaw_deg_);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulate_clouds() const {
        auto accumulated = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& scan : cloud_buffer_) {
            *accumulated += *(scan.cloud);
        }
        accumulated->width = static_cast<uint32_t>(accumulated->points.size());
        accumulated->height = 1;
        accumulated->is_dense = false;
        return accumulated;
    }

    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (msg->header.frame_id.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Incoming PointCloud2 has empty frame_id. Dropping.");
            return;
        }

        if ((msg->width * msg->height) == 0) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Incoming cloud has 0 points. Skipping.");
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        if (!cloud || cloud->empty()) {
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> nan_indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, nan_indices);

        if (!cloud_filtered || cloud_filtered->empty()) {
            return;
        }

        if (!have_pose_) {
            cloud_buffer_.clear();
            BufferedScan scan;
            scan.cloud = cloud_filtered;
            cloud_buffer_.push_back(scan);
        } else {
            bool can_accumulate = true;

            if (accumulate_only_when_static_ && !cloud_buffer_.empty()) {
                const auto& last = cloud_buffer_.back();
                can_accumulate = motion_small_enough(last.x, last.y, last.yaw);
            }

            if (!can_accumulate) {
                cloud_buffer_.clear();
            }

            BufferedScan scan;
            scan.cloud = cloud_filtered;
            scan.x = current_x_;
            scan.y = current_y_;
            scan.yaw = current_yaw_;
            cloud_buffer_.push_back(scan);

            while (static_cast<int>(cloud_buffer_.size()) > accumulation_scans_) {
                cloud_buffer_.pop_front();
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr working_cloud = accumulate_clouds();
        if (!working_cloud || working_cloud->empty()) {
            return;
        }

        auto cluster_indices = range_image_segment(working_cloud);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusters_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        wamv_detected_msg_.name.clear();
        wamv_detected_msg_.x.clear();
        wamv_detected_msg_.y.clear();
        wamv_detected_msg_.z.clear();

        fau_msgs::msg::ObjectPosition fau_msg;
        fau_msg.object_names.clear();
        fau_msg.x_object.clear();
        fau_msg.y_object.clear();
        fau_msg.z_object.clear();
        fau_msg.radii_object.clear();

        int cluster_id = 0;

        for (const auto& idxs : cluster_indices) {
            if (static_cast<int>(idxs.size()) < min_cluster_size_ ||
                static_cast<int>(idxs.size()) > max_cluster_size_) {
                continue;
            }

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
            cluster->points.reserve(idxs.size());

            for (int idx : idxs) {
                const auto& src = working_cloud->points[idx];
                if (!pcl::isFinite(src)) {
                    continue;
                }

                pcl::PointXYZRGB p;
                p.x = src.x;
                p.y = src.y;
                p.z = src.z;
                cluster->points.push_back(p);
            }

            if (cluster->points.empty()) {
                continue;
            }

            cluster->width = static_cast<uint32_t>(cluster->points.size());
            cluster->height = 1;
            cluster->is_dense = true;

            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cluster, centroid);

            if (!finite3(centroid[0], centroid[1], centroid[2])) {
                continue;
            }

            if (centroid[0] <= 0.0) {
                continue;
            }

            if (centroid[2] < min_detect_height_ || centroid[2] > max_detect_height_) {
                continue;
            }

            pcl::PointXYZRGB min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);

            const float width_x  = max_pt.x - min_pt.x;
            const float width_y  = max_pt.y - min_pt.y;
            const float height_z = max_pt.z - min_pt.z;
            const float dist = std::sqrt((centroid[0] * centroid[0]) + (centroid[1] * centroid[1]));

            if (!(width_x >= min_x_size_ && width_x <= max_x_size_ &&
                  width_y >= min_y_size_ && width_y <= max_y_size_ &&
                  height_z >= min_z_size_ && height_z <= max_z_size_ &&
                  dist >= min_detect_dist_ && dist <= max_detect_dist_)) {
                continue;
            }

            const float rx = 0.5f * width_x;
            const float ry = 0.5f * width_y;
            float r_xy = std::sqrt(rx * rx + ry * ry);

            if (!std::isfinite(r_xy)) {
                continue;
            }

            r_xy = std::max(static_cast<float>(min_radius_m_), r_xy);
            r_xy = static_cast<float>(radius_scale_) * r_xy;

            for (auto& p : cluster->points) {
                p.r = 118;
                p.g = 37;
                p.b = 251;
            }

            *clusters_cloud += *cluster;

            const std::string name = "BUOY[" + std::to_string(cluster_id) + "]";

            wamv_detected_msg_.name.push_back(name);
            wamv_detected_msg_.x.push_back(centroid[0]);
            wamv_detected_msg_.y.push_back(centroid[1]);
            wamv_detected_msg_.z.push_back(centroid[2]);

            fau_msg.object_names.push_back(name);
            fau_msg.x_object.push_back(static_cast<double>(centroid[0]));
            fau_msg.y_object.push_back(static_cast<double>(centroid[1]));
            fau_msg.z_object.push_back(static_cast<double>(centroid[2]));
            fau_msg.radii_object.push_back(static_cast<double>(r_xy));

            cluster_id++;
        }

        if (clusters_cloud->empty()) {
            return;
        }

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*clusters_cloud, output);
        output.header.frame_id = msg->header.frame_id;
        output.header.stamp = msg->header.stamp;
        cluster_pub_->publish(output);

        if (!wamv_detected_msg_.name.empty()) {
            wamv_detected_pub_->publish(wamv_detected_msg_);
        }

        if (!fau_msg.object_names.empty()) {
            fau_objects_pub_->publish(fau_msg);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr pose_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cluster_pub_;
    rclcpp::Publisher<lidar_msgs::msg::BuoyDetected>::SharedPtr wamv_detected_pub_;
    rclcpp::Publisher<fau_msgs::msg::ObjectPosition>::SharedPtr fau_objects_pub_;

    std::deque<BufferedScan> cloud_buffer_;

    double current_x_{0.0};
    double current_y_{0.0};
    double current_yaw_{0.0};
    bool have_pose_{false};

    double min_x_size_, min_y_size_, min_z_size_;
    double max_x_size_, max_y_size_, max_z_size_;
    double min_detect_dist_, max_detect_dist_;
    double min_detect_height_, max_detect_height_;

    int min_cluster_size_, max_cluster_size_;

    double radius_scale_{1.0};
    double min_radius_m_{0.10};

    int rows_{16};
    int cols_{900};
    double v_angle_min_deg_{-15.0};
    double v_angle_max_deg_{15.0};
    double horiz_fov_deg_{360.0};

    double range_jump_thresh_m_{0.10};
    int neighbor_col_window_{2};
    int neighbor_row_window_{1};
    double min_range_m_{0.0};
    double max_range_m_{60.0};

    int accumulation_scans_{2};
    bool accumulate_only_when_static_{true};
    double max_accum_translation_m_{0.20};
    double max_accum_yaw_deg_{5.0};

    std::string wamv_;

    lidar_msgs::msg::BuoyDetected wamv_detected_msg_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudClusterer>());
    rclcpp::shutdown();
    return 0;
}