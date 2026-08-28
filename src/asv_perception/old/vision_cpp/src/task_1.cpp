#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "lidar_msgs/msg/buoy_detected.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <deque>

class OccupancyGridPathGenerator : public rclcpp::Node
{
public:
  OccupancyGridPathGenerator()
  : Node("occupancy_grid_path_generator")
  {
    // ---------------- Parameters ----------------
    // Topics
    this->declare_parameter<std::string>("topic_points", "/velodyne_points");
    this->declare_parameter<std::string>("topic_obstacles", "/vision/output/boat_detected");
    this->declare_parameter<std::string>("topic_grid", "/vision/output/occupancy_grid");
    this->declare_parameter<std::string>("topic_path", "/path_vision_1");
    this->declare_parameter<std::string>("topic_task1_buoys", "/vision/output/task1_buoy_detected");
    this->declare_parameter<std::string>("frame_id", "velodyne");

    // ROI / filters
    this->declare_parameter<double>("x_min_m", 1.5);
    this->declare_parameter<double>("x_max_m", 35.0);
    this->declare_parameter<double>("y_min_m", -6.0);
    this->declare_parameter<double>("y_max_m",  6.0);
    this->declare_parameter<double>("fov_deg", 80.0);   // +/- fov_deg
    this->declare_parameter<double>("z_min_m", 0.0);

    // Grid (optional, for debug)
    this->declare_parameter<bool>("publish_grid", true);
    this->declare_parameter<double>("grid_resolution_m", 0.10);
    this->declare_parameter<int>("grid_width", 295);
    this->declare_parameter<int>("grid_height", 140);
    this->declare_parameter<double>("grid_origin_x", 0.5);
    this->declare_parameter<double>("grid_origin_y", -7.0);

    // Object formation
    this->declare_parameter<int>("min_obj_points", 3);          // min points per *cell* to consider it occupied
    this->declare_parameter<double>("cluster_radius_m", 1.0);   // cluster cells -> buoy centroid
    this->declare_parameter<int>("min_cluster_cells", 2);       // min occupied cells per buoy cluster

    // Temporal hold
    this->declare_parameter<double>("hold_seconds", 0.10);      // keep buoy centroids for this long

    // Gate logic (Task 1)
    this->declare_parameter<double>("gate_dx_max_m", 2.0);      // buoys in a gate should be near same x
    this->declare_parameter<double>("gate_sep_min_m", 1.0);     // separation bounds (tune for pole buoys)
    this->declare_parameter<double>("gate_sep_max_m", 6.0);
    this->declare_parameter<double>("second_gate_min_forward_m", 6.0); // how far ahead second gate must be
    this->declare_parameter<double>("tangent_len_m", 12.0);     // optional lookahead point

    // Safety against "boats" / obstacles list
    this->declare_parameter<double>("min_safe_distance_m", 1.0);

    // ---------------- Load params ----------------
    topic_points_     = this->get_parameter("topic_points").as_string();
    topic_obstacles_  = this->get_parameter("topic_obstacles").as_string();
    topic_grid_       = this->get_parameter("topic_grid").as_string();
    topic_path_       = this->get_parameter("topic_path").as_string();
    topic_task1_buoys_= this->get_parameter("topic_task1_buoys").as_string();
    frame_id_         = this->get_parameter("frame_id").as_string();

    x_min_ = this->get_parameter("x_min_m").as_double();
    x_max_ = this->get_parameter("x_max_m").as_double();
    y_min_ = this->get_parameter("y_min_m").as_double();
    y_max_ = this->get_parameter("y_max_m").as_double();
    fov_deg_ = this->get_parameter("fov_deg").as_double();
    z_min_m_ = this->get_parameter("z_min_m").as_double();

    publish_grid_ = this->get_parameter("publish_grid").as_bool();
    grid_resolution_ = this->get_parameter("grid_resolution_m").as_double();
    grid_width_  = this->get_parameter("grid_width").as_int();
    grid_height_ = this->get_parameter("grid_height").as_int();
    grid_origin_x_ = this->get_parameter("grid_origin_x").as_double();
    grid_origin_y_ = this->get_parameter("grid_origin_y").as_double();

    min_obj_points_ = this->get_parameter("min_obj_points").as_int();
    cluster_radius_ = this->get_parameter("cluster_radius_m").as_double();
    min_cluster_cells_ = this->get_parameter("min_cluster_cells").as_int();

    hold_seconds_ = this->get_parameter("hold_seconds").as_double();

    gate_dx_max_ = this->get_parameter("gate_dx_max_m").as_double();
    gate_sep_min_ = this->get_parameter("gate_sep_min_m").as_double();
    gate_sep_max_ = this->get_parameter("gate_sep_max_m").as_double();
    second_gate_min_forward_ = this->get_parameter("second_gate_min_forward_m").as_double();
    tangent_len_ = this->get_parameter("tangent_len_m").as_double();

    min_safe_distance_ = this->get_parameter("min_safe_distance_m").as_double();

    // ---------------- ROS I/O ----------------
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      topic_points_, rclcpp::SensorDataQoS(),
      std::bind(&OccupancyGridPathGenerator::pointcloud_callback, this, std::placeholders::_1));

    obstacle_sub_ = this->create_subscription<lidar_msgs::msg::BuoyDetected>(
      topic_obstacles_, 10,
      std::bind(&OccupancyGridPathGenerator::obstacle_callback, this, std::placeholders::_1));

    grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(topic_grid_, 10);
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(topic_path_, 10);
    task1_buoy_pub_ = this->create_publisher<lidar_msgs::msg::BuoyDetected>(topic_task1_buoys_, 10);

    RCLCPP_INFO(this->get_logger(),
      "Initialized %s\n  points=%s\n  obstacles=%s\n  path=%s\n  z_min=%.2f, cluster_r=%.2f, hold=%.2fs",
      this->get_name(), topic_points_.c_str(), topic_obstacles_.c_str(), topic_path_.c_str(),
      z_min_m_, cluster_radius_, hold_seconds_);
  }

private:
  // ---------------- Types ----------------
  struct Obstacle {
    std::string name;
    float x, y, z;
  };

  struct CellAccum {
    int count = 0;
    double sum_x = 0.0;
    double sum_y = 0.0;
  };

  struct TimedCentroids {
    rclcpp::Time stamp;
    std::vector<std::pair<float,float>> centroids;
  };

  // ---------------- State ----------------
  std::vector<Obstacle> obstacles_;
  std::deque<TimedCentroids> centroid_history_;

  // ---------------- Params ----------------
  std::string topic_points_, topic_obstacles_, topic_grid_, topic_path_, topic_task1_buoys_, frame_id_;

  double x_min_{1.5}, x_max_{25.0}, y_min_{-6.0}, y_max_{6.0}, fov_deg_{60.0};
  double z_min_m_{0.2};
  

  bool publish_grid_{true};
  double grid_resolution_{0.1};
  int grid_width_{295}, grid_height_{140};
  double grid_origin_x_{0.5}, grid_origin_y_{-7.0};

  int min_obj_points_{5};
  double cluster_radius_{1.0};
  int min_cluster_cells_{2};

  double hold_seconds_{0.10};

  double gate_dx_max_{1.5};
  double gate_sep_min_{1.0};
  double gate_sep_max_{6.0};
  double second_gate_min_forward_{6.0};
  double tangent_len_{12.0};

  double min_safe_distance_{2.0};

  // ---------------- ROS ----------------
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Subscription<lidar_msgs::msg::BuoyDetected>::SharedPtr obstacle_sub_;

  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<lidar_msgs::msg::BuoyDetected>::SharedPtr task1_buoy_pub_;

  // ---------------- Helpers ----------------
  static inline uint64_t pack_key(int gx, int gy)
  {
    return (static_cast<uint64_t>(static_cast<uint32_t>(gx)) << 32) |
           (static_cast<uint64_t>(static_cast<uint32_t>(gy)));
  }

  inline bool in_roi(float x, float y) const
  {
    if (x < x_min_ || x > x_max_) return false;
    if (y < y_min_ || y > y_max_) return false;
    const double ang = std::atan2(y, x) * 180.0 / M_PI;
    return (ang >= -fov_deg_ && ang <= fov_deg_);
  }

  bool is_safe(float x, float y) const
  {
    for (const auto& ob : obstacles_) {
      const double d = std::hypot(x - ob.x, y - ob.y);
      if (d < min_safe_distance_) return false;
    }
    return true;
  }

  void obstacle_callback(const lidar_msgs::msg::BuoyDetected::SharedPtr msg)
  {
    obstacles_.clear();
    for (size_t i = 0; i < msg->name.size(); ++i) {
      const float x = msg->x[i];
      const float y = msg->y[i];
      const float z = msg->z[i];
      if (std::hypot(x,y) <= 50.0) {
        obstacles_.push_back({msg->name[i], x, y, z});
      }
    }
  }

  // Simple radius-based clustering of buoy candidates
  static std::vector<std::vector<int>> radius_cluster(
      const std::vector<std::pair<float,float>>& pts,
      double radius)
  {
    const float r2 = static_cast<float>(radius * radius);
    const int n = static_cast<int>(pts.size());
    std::vector<char> used(n, 0);
    std::vector<std::vector<int>> clusters;
    clusters.reserve(n/2);

    for (int i = 0; i < n; ++i) {
      if (used[i]) continue;
      used[i] = 1;
      std::vector<int> cluster;
      cluster.push_back(i);

      // BFS style expansion
      for (size_t k = 0; k < cluster.size(); ++k) {
        const int idx = cluster[k];
        const float x = pts[idx].first;
        const float y = pts[idx].second;

        for (int j = 0; j < n; ++j) {
          if (used[j]) continue;
          const float dx = pts[j].first - x;
          const float dy = pts[j].second - y;
          if (dx*dx + dy*dy <= r2) {
            used[j] = 1;
            cluster.push_back(j);
          }
        }
      }
      clusters.push_back(std::move(cluster));
    }
    return clusters;
  }

  static std::vector<std::pair<float,float>> cluster_centroids(
      const std::vector<std::pair<float,float>>& pts,
      double radius,
      int min_cluster_size)
  {
    std::vector<std::pair<float,float>> out;
    if (pts.empty()) return out;

    auto clusters = radius_cluster(pts, radius);
    out.reserve(clusters.size());

    for (const auto& c : clusters) {
      if (static_cast<int>(c.size()) < min_cluster_size) continue;
      double sx = 0.0, sy = 0.0;
      for (int idx : c) {
        sx += pts[idx].first;
        sy += pts[idx].second;
      }
      out.emplace_back(static_cast<float>(sx / c.size()),
                       static_cast<float>(sy / c.size()));
    }
    return out;
  }

  void prune_history()
  {
    const rclcpp::Time now = this->now();
    while (!centroid_history_.empty()) {
      const double age = (now - centroid_history_.front().stamp).seconds();
      if (age <= hold_seconds_) break;
      centroid_history_.pop_front();
    }
  }

  std::vector<std::pair<float,float>> merged_held_centroids()
  {
    prune_history();
    std::vector<std::pair<float,float>> merged;
    size_t total = 0;
    for (const auto& h : centroid_history_) total += h.centroids.size();
    merged.reserve(total);
    for (const auto& h : centroid_history_) {
      merged.insert(merged.end(), h.centroids.begin(), h.centroids.end());
    }

    // re-cluster merged to avoid duplicates across frames
    return cluster_centroids(merged, cluster_radius_, /*min_cluster_size=*/1);
  }

  // Gate-finding: return best (x1,y1,x2,y2, xmid,ymid)
  bool find_best_gate(
      const std::vector<std::pair<float,float>>& buoys,
      double min_x_mid,
      float &x1, float &y1, float &x2, float &y2, float &xm, float &ym) const
  {
    bool found = false;
    float best_xmid = std::numeric_limits<float>::max();

    for (size_t i = 0; i < buoys.size(); ++i) {
      for (size_t j = i + 1; j < buoys.size(); ++j) {
        const float ax = buoys[i].first, ay = buoys[i].second;
        const float bx = buoys[j].first, by = buoys[j].second;

        // Must be roughly same "gate plane"
        if (std::fabs(ax - bx) > gate_dx_max_) continue;

        const float sep = std::hypot(bx - ax, by - ay);
        if (sep < gate_sep_min_ || sep > gate_sep_max_) continue;

        const float cx = 0.5f * (ax + bx);
        const float cy = 0.5f * (ay + by);

        if (cx <= static_cast<float>(min_x_mid)) continue;
        if (!in_roi(cx, cy)) continue;
        if (!is_safe(cx, cy)) continue;

        // Prefer nearest gate ahead (smallest xmid)
        if (cx < best_xmid) {
          best_xmid = cx;
          x1 = ax; y1 = ay;
          x2 = bx; y2 = by;
          xm = cx; ym = cy;
          found = true;
        }
      }
    }
    return found;
  }

  bool compute_perp_lookahead(
      float x1, float y1,
      float x2, float y2,
      float mx, float my,
      float target_len,
      float &out_x, float &out_y) const
  {
    float vx = x2 - x1;
    float vy = y2 - y1;
    float n = std::sqrt(vx*vx + vy*vy);
    if (n < 1e-3f) return false;

    // Perp unit normal
    float nx = -vy / n;
    float ny =  vx / n;

    // We want *forward* in +x preference, try both sides and pick bigger x
    float c1x = mx + target_len * nx;
    float c1y = my + target_len * ny;
    float c2x = mx - target_len * nx;
    float c2y = my - target_len * ny;

    bool ok1 = in_roi(c1x, c1y) && is_safe(c1x, c1y);
    bool ok2 = in_roi(c2x, c2y) && is_safe(c2x, c2y);

    if (!ok1 && !ok2) return false;
    if (ok1 && ok2) {
      if (c1x >= c2x) { out_x = c1x; out_y = c1y; }
      else            { out_x = c2x; out_y = c2y; }
      return true;
    }
    if (ok1) { out_x = c1x; out_y = c1y; return true; }
    out_x = c2x; out_y = c2y; return true;
  }

  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // ---- Build occupancy + per-cell accumulators ----
    nav_msgs::msg::OccupancyGrid grid;
    if (publish_grid_) {
      grid.header.stamp = this->now();
      grid.header.frame_id = frame_id_;
      grid.info.resolution = grid_resolution_;
      grid.info.width = static_cast<uint32_t>(grid_width_);
      grid.info.height = static_cast<uint32_t>(grid_height_);
      grid.info.origin.position.x = grid_origin_x_;
      grid.info.origin.position.y = grid_origin_y_;
      grid.info.origin.position.z = 0.0;
      grid.info.origin.orientation.w = 1.0;
      grid.data.assign(static_cast<size_t>(grid_width_ * grid_height_), -1);
    }

    std::unordered_map<uint64_t, CellAccum> cell_map;
    cell_map.reserve(static_cast<size_t>(msg->width * msg->height / 20));

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      const float x = *iter_x;
      const float y = *iter_y;
      const float z = *iter_z;

      if (z < static_cast<float>(z_min_m_)) continue;
      if (!in_roi(x, y)) continue;

      const int gx = static_cast<int>((x - static_cast<float>(grid_origin_x_)) / grid_resolution_);
      const int gy = static_cast<int>((y - static_cast<float>(grid_origin_y_)) / grid_resolution_);

      if (gx < 0 || gx >= grid_width_ || gy < 0 || gy >= grid_height_) continue;

      const uint64_t key = pack_key(gx, gy);
      auto &acc = cell_map[key];
      acc.count += 1;
      acc.sum_x += x;
      acc.sum_y += y;

      if (publish_grid_) {
        grid.data[static_cast<size_t>(gy * grid_width_ + gx)] = 100;
      }
    }

    // ---- Convert occupied cells -> candidate points (cell centroids) ----
    std::vector<std::pair<float,float>> cell_pts;
    cell_pts.reserve(cell_map.size());
    for (const auto &kv : cell_map) {
      const CellAccum &acc = kv.second;
      if (acc.count < min_obj_points_) continue;
      cell_pts.emplace_back(static_cast<float>(acc.sum_x / acc.count),
                            static_cast<float>(acc.sum_y / acc.count));
    }

    // ---- Cluster into buoy centroids ----
    auto buoy_centroids = cluster_centroids(cell_pts, cluster_radius_, min_cluster_cells_);

    // ---- Hold briefly to reduce flicker ----
    centroid_history_.push_back({this->now(), buoy_centroids});
    auto held_buoys = merged_held_centroids();

    if (publish_grid_) grid_pub_->publish(grid);

    // ---- Find gates ----
    nav_msgs::msg::Path path;
    path.header.stamp = this->now();
    path.header.frame_id = frame_id_;

    if (held_buoys.size() < 2) {
      path_pub_->publish(path);
      return;
    }

    float g1_x1=0,g1_y1=0,g1_x2=0,g1_y2=0,g1_xm=0,g1_ym=0;
    if (!find_best_gate(held_buoys, /*min_x_mid=*/0.0, g1_x1,g1_y1,g1_x2,g1_y2,g1_xm,g1_ym)) {
      path_pub_->publish(path);
      return;
    }

    // Optional second gate
    float g2_x1=0,g2_y1=0,g2_x2=0,g2_y2=0,g2_xm=0,g2_ym=0;
    const double second_min_x = static_cast<double>(g1_xm) + second_gate_min_forward_;
    const bool has_second = find_best_gate(held_buoys, second_min_x, g2_x1,g2_y1,g2_x2,g2_y2,g2_xm,g2_ym);

    // ---- Publish path: origin -> gate1 mid -> gate2 mid (if found) -> optional lookahead ----
    auto push_pose = [&](float x, float y) {
      geometry_msgs::msg::PoseStamped p;
      p.header = path.header;
      p.pose.position.x = x;
      p.pose.position.y = y;
      p.pose.position.z = 0.0;
      p.pose.orientation.w = 1.0;
      path.poses.push_back(p);
    };

    push_pose(0.0f, 0.0f);
    push_pose(g1_xm, g1_ym);
    if (has_second) push_pose(g2_xm, g2_ym);

    // optional lookahead perpendicular to first gate line
    float lx=0, ly=0;
    if (compute_perp_lookahead(g1_x1,g1_y1,g1_x2,g1_y2, g1_xm,g1_ym,
                              static_cast<float>(tangent_len_), lx, ly)) {
      push_pose(lx, ly);
    }

    path_pub_->publish(path);

    // ---- Publish chosen buoys for gate1 ----
    lidar_msgs::msg::BuoyDetected out;
    out.name = {"Gate1_BuoyA", "Gate1_BuoyB"};
    out.x = {g1_x1, g1_x2};
    out.y = {g1_y1, g1_y2};
    out.z = {0.0f, 0.0f};
    task1_buoy_pub_->publish(out);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OccupancyGridPathGenerator>());
  rclcpp::shutdown();
  return 0;
}