#pragma once

#include <cstddef>
#include <vector>

#include "line_follower/finals_types.hpp"

namespace line_follower
{

struct ObstacleMemoryConfig
{
  double merge_radius_m{0.16};
  double memory_ttl_sec{1.8};
  double car_length_m{0.28};
  double rear_keep_margin_m{0.15};   // Keep obstacles down to d >= -(car_length + margin)
  double max_forward_dist_m{2.5};
  double ema_alpha{0.45};
  std::size_t max_entries{32};
};

// ============================================================
// ObstacleMemory (pure C++17, zero OpenCV/ROS2 deps)
//
// Persists detected red_cone and red_no_entry footprints in the
// 2D EKF/odom world frame so that obstacles passing into the
// camera's near blind zone or beside the rear wheels
// (d_bumper in [-0.30m, 0.0m]) remain active for Ackermann
// rear-wheel cut-in protection.
// ============================================================
class ObstacleMemory
{
public:
  ObstacleMemory() = default;
  explicit ObstacleMemory(const ObstacleMemoryConfig & config);

  void set_config(const ObstacleMemoryConfig & config);
  const ObstacleMemoryConfig & config() const;

  void reset();

  void update(
    const std::vector<GroundDetection> & detections,
    const VehiclePose2D & pose,
    double now_sec);

  std::vector<LocalObstacle> get_local_obstacles(
    const VehiclePose2D & pose,
    double now_sec) const;

  const std::vector<TrackedObstacle> & entries() const;
  std::size_t size() const;

  static void bumper_to_world(
    const VehiclePose2D & pose,
    double d_bumper_m,
    double y_lat_m,
    double & wx,
    double & wy);

  static void world_to_bumper(
    const VehiclePose2D & pose,
    double wx,
    double wy,
    double & d_bumper_m,
    double & y_lat_m);

private:
  void prune(const VehiclePose2D & pose, double now_sec);

  ObstacleMemoryConfig config_{};
  std::vector<TrackedObstacle> entries_{};
  int next_id_{1};
};

}  // namespace line_follower
