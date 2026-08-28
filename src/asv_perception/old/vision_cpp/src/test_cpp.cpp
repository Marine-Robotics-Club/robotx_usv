#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "lidar_msgs/msg/buoy_detected.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <deque>

class OccupancyGridPathGenerator : public rclcpp::Node
{
public:
  OccupancyGridPathGenerator()
  : Node("occupancy_grid_path_generator")
  {
    frame_id_ = "velodyne";

    // -------- ROI ----------
    x_min_ = 0.5;
    x_max_ = 25.0;
    y_min_ = -8.0;
    y_max_ = 8.0;
    fov_deg_ = 85.0;

    // -------- Noise Rejection ----------
    min_cell_points_ = 12;        // stronger wave rejection
    cluster_radius_ = 0.5;
    min_cluster_cells_ = 4;
    cluster_min_extent_ = 0.15;
    cluster_max_extent_ = 2.5;

    hold_seconds_ = 0.05;         // <<< YOUR REQUEST

    // -------- Gate Logic ----------
    gate_dx_max_ = 1.5;
    gate_w_min_ = 1.5;
    gate_w_max_ = 10.0;
    min_gate_step_ = 2.0;
    max_waypoints_ = 10;

    // -------- Safety ----------
    min_safe_distance_ = 4.0;
    min_mid_clear_ = 2.0;
    avoid_shift_ = 1.5;

    pointcloud_sub_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points",
        rclcpp::SensorDataQoS(),
        std::bind(&OccupancyGridPathGenerator::pointcloud_callback,
                  this, std::placeholders::_1));

    obstacle_sub_ =
      this->create_subscription<lidar_msgs::msg::BuoyDetected>(
        "/vision/output/boat_detected",
        10,
        std::bind(&OccupancyGridPathGenerator::obstacle_callback,
                  this, std::placeholders::_1));

    path_pub_ =
      this->create_publisher<nav_msgs::msg::Path>(
        "/path_vision_2_test", 10);

    RCLCPP_INFO(this->get_logger(),
      "Channel nav started. Temporal hold = %.2f sec",
      hold_seconds_);
  }

private:

  struct Obstacle { float x,y; };
  std::vector<Obstacle> obstacles_;

  struct TimedCentroids {
    rclcpp::Time stamp;
    std::vector<std::pair<float,float>> centroids;
  };

  std::deque<TimedCentroids> history_;

  // ---------------- Parameters ----------------
  std::string frame_id_;

  double x_min_, x_max_, y_min_, y_max_, fov_deg_;
  int min_cell_points_;
  double cluster_radius_;
  int min_cluster_cells_;
  double cluster_min_extent_, cluster_max_extent_;
  double hold_seconds_;

  double gate_dx_max_, gate_w_min_, gate_w_max_;
  double min_gate_step_;
  int max_waypoints_;

  double min_safe_distance_;
  double min_mid_clear_;
  double avoid_shift_;

  // ---------------- ROS ----------------
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Subscription<lidar_msgs::msg::BuoyDetected>::SharedPtr obstacle_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

  // ----------------------------------------------------

  bool in_roi(float x, float y)
  {
    if (x < x_min_ || x > x_max_) return false;
    if (y < y_min_ || y > y_max_) return false;
    double ang = std::atan2(y,x) * 180.0/M_PI;
    return (ang >= -fov_deg_ && ang <= fov_deg_);
  }

  bool is_safe(float x, float y)
  {
    for (auto &o : obstacles_)
      if (std::hypot(x-o.x, y-o.y) < min_safe_distance_)
        return false;
    return true;
  }

  void obstacle_callback(const lidar_msgs::msg::BuoyDetected::SharedPtr msg)
  {
    obstacles_.clear();
    for (size_t i=0;i<msg->x.size();++i)
      obstacles_.push_back({msg->x[i], msg->y[i]});
  }

  // -------- Radius clustering --------
  std::vector<std::vector<int>>
  radius_cluster(const std::vector<std::pair<float,float>>& pts)
  {
    const float r2 = cluster_radius_ * cluster_radius_;
    std::vector<char> used(pts.size(),0);
    std::vector<std::vector<int>> clusters;

    for (size_t i=0;i<pts.size();++i)
    {
      if (used[i]) continue;
      used[i]=1;

      std::vector<int> c;
      c.push_back(i);

      for (size_t k=0;k<c.size();++k)
      {
        int idx = c[k];
        for (size_t j=0;j<pts.size();++j)
        {
          if (used[j]) continue;
          float dx = pts[j].first - pts[idx].first;
          float dy = pts[j].second - pts[idx].second;
          if (dx*dx + dy*dy <= r2)
          {
            used[j]=1;
            c.push_back(j);
          }
        }
      }
      clusters.push_back(c);
    }
    return clusters;
  }

  // --------- Extract buoy centroids ----------
  std::vector<std::pair<float,float>>
  extract_buoys(const std::vector<std::pair<float,float>>& pts)
  {
    std::vector<std::pair<float,float>> out;
    auto clusters = radius_cluster(pts);

    for (auto &c : clusters)
    {
      if ((int)c.size() < min_cluster_cells_) continue;

      float minx=1e9, miny=1e9, maxx=-1e9, maxy=-1e9;
      double sx=0, sy=0;

      for (int idx : c)
      {
        float x = pts[idx].first;
        float y = pts[idx].second;
        sx+=x; sy+=y;
        minx=std::min(minx,x);
        maxx=std::max(maxx,x);
        miny=std::min(miny,y);
        maxy=std::max(maxy,y);
      }

      float extent = std::max(maxx-minx, maxy-miny);

      if (extent < cluster_min_extent_) continue;
      if (extent > cluster_max_extent_) continue;

      out.emplace_back(sx/c.size(), sy/c.size());
    }
    return out;
  }

  // --------- Temporal hold logic -----------
  std::vector<std::pair<float,float>> temporal_filter(
      const std::vector<std::pair<float,float>>& now)
  {
    history_.push_back({this->now(), now});

    while (!history_.empty())
    {
      if ((this->now() - history_.front().stamp).seconds() <= hold_seconds_)
        break;
      history_.pop_front();
    }

    std::vector<std::pair<float,float>> merged;
    for (auto &h : history_)
      merged.insert(merged.end(),
                    h.centroids.begin(),
                    h.centroids.end());

    return extract_buoys(merged);
  }

  // --------- Main callback -----------
  void pointcloud_callback(
      const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    std::unordered_map<int, std::pair<int,std::pair<double,double>>> cell_map;

    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg,"x");
    sensor_msgs::PointCloud2ConstIterator<float> iy(*msg,"y");

    for (; ix!=ix.end(); ++ix, ++iy)
    {
      float x=*ix, y=*iy;
      if (!in_roi(x,y)) continue;

      int key = (int)(x*10) * 10000 + (int)(y*10);

      auto &cell = cell_map[key];
      cell.first++;
      cell.second.first += x;
      cell.second.second += y;
    }

    std::vector<std::pair<float,float>> cell_pts;
    for (auto &kv : cell_map)
    {
      if (kv.second.first < min_cell_points_) continue;
      cell_pts.emplace_back(
        kv.second.second.first / kv.second.first,
        kv.second.second.second / kv.second.first);
    }

    auto buoys_now = extract_buoys(cell_pts);
    auto buoys = temporal_filter(buoys_now);

    if (buoys.size() < 2) return;

    std::vector<std::pair<float,float>> left,right;
    for (auto &b : buoys)
      (b.second < 0 ? left : right).push_back(b);

    if (left.empty() || right.empty()) return;

    std::sort(left.begin(), left.end());
    std::sort(right.begin(), right.end());

    nav_msgs::msg::Path path;
    path.header.stamp = this->now();
    path.header.frame_id = frame_id_;

    geometry_msgs::msg::PoseStamped origin;
    origin.pose.position.x=0;
    origin.pose.position.y=0;
    origin.pose.orientation.w=1;
    path.poses.push_back(origin);

    int count=0;
    for (size_t i=0;i<std::min(left.size(),right.size());++i)
    {
      float xm = 0.5f*(left[i].first + right[i].first);
      float ym = 0.5f*(left[i].second + right[i].second);

      float dL = std::hypot(xm-left[i].first, ym-left[i].second);
      float dR = std::hypot(xm-right[i].first, ym-right[i].second);
      if (dL < min_mid_clear_ || dR < min_mid_clear_) continue;

      if (!is_safe(xm,ym)) continue;

      geometry_msgs::msg::PoseStamped p;
      p.pose.position.x=xm;
      p.pose.position.y=ym;
      p.pose.orientation.w=1;
      path.poses.push_back(p);

      if (++count >= max_waypoints_) break;
    }

    path_pub_->publish(path);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc,argv);
  rclcpp::spin(std::make_shared<OccupancyGridPathGenerator>());
  rclcpp::shutdown();
  return 0;
}