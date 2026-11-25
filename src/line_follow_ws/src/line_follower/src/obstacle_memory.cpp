#include "line_follower/obstacle_memory.hpp"

#include <algorithm>
#include <cmath>

namespace line_follower
{

ObstacleMemory::ObstacleMemory(const ObstacleMemoryConfig & config)
{
  set_config(config);
}

void ObstacleMemory::set_config(const ObstacleMemoryConfig & config)
{
  config_ = config;
  config_.merge_radius_m = std::max(0.02, config_.merge_radius_m);
  config_.memory_ttl_sec = std::max(0.1, config_.memory_ttl_sec);
  config_.car_length_m = std::max(0.05, config_.car_length_m);
  config_.rear_axle_to_bumper_m = std::max(0.0, config_.rear_axle_to_bumper_m);
  config_.rear_keep_margin_m = std::max(0.0, config_.rear_keep_margin_m);
  config_.max_forward_dist_m = std::max(0.3, config_.max_forward_dist_m);
  config_.ema_alpha = std::clamp(config_.ema_alpha, 0.05, 1.0);
  if (config_.max_entries < 4) {
    config_.max_entries = 4;
  }
}

const ObstacleMemoryConfig & ObstacleMemory::config() const
{
  return config_;
}

void ObstacleMemory::reset()
{
  entries_.clear();
  next_id_ = 1;
}

void ObstacleMemory::bumper_to_world(
  const VehiclePose2D & pose,
  double d_bumper_m,
  double y_lat_m,
  double & wx,
  double & wy,
  double rear_axle_to_bumper_m)
{
  // EKF pose is at the rear axle; shift bumper point into rear-axle frame before rotating.
  const double d_axle = d_bumper_m + rear_axle_to_bumper_m;
  const double c = std::cos(pose.yaw);
  const double s = std::sin(pose.yaw);
  wx = pose.x + d_axle * c - y_lat_m * s;
  wy = pose.y + d_axle * s + y_lat_m * c;
}

void ObstacleMemory::world_to_bumper(
  const VehiclePose2D & pose,
  double wx,
  double wy,
  double & d_bumper_m,
  double & y_lat_m,
  double rear_axle_to_bumper_m)
{
  const double dx = wx - pose.x;
  const double dy = wy - pose.y;
  const double c = std::cos(pose.yaw);
  const double s = std::sin(pose.yaw);
  const double d_axle = dx * c + dy * s;
  d_bumper_m = d_axle - rear_axle_to_bumper_m;
  y_lat_m = -dx * s + dy * c;
}

void ObstacleMemory::update(
  const std::vector<GroundDetection> & detections,
  const VehiclePose2D & pose,
  double now_sec)
{
  VehiclePose2D ref_pose = pose;
  if (!ref_pose.valid) {
    // Fallback: without EKF pose, keep only current-frame observations in local origin
    entries_.clear();
    ref_pose.x = 0.0;
    ref_pose.y = 0.0;
    ref_pose.yaw = 0.0;
  }

  for (const auto & det : detections) {
    if (det.type != SemanticObjectType::RED_CONE &&
      det.type != SemanticObjectType::RED_NO_ENTRY)
    {
      continue;
    }

    if (det.d_bumper_m < -(config_.car_length_m + config_.rear_keep_margin_m) ||
      det.d_bumper_m > config_.max_forward_dist_m)
    {
      continue;
    }

    double wx = 0.0;
    double wy = 0.0;
    bumper_to_world(
      ref_pose, det.d_bumper_m, det.y_lat_m, wx, wy, config_.rear_axle_to_bumper_m);

    int best_idx = -1;
    double best_dist = config_.merge_radius_m;

    for (std::size_t i = 0; i < entries_.size(); ++i) {
      if (entries_[i].type != det.type) {
        continue;
      }
      const double dist = std::hypot(wx - entries_[i].wx, wy - entries_[i].wy);
      if (dist <= best_dist) {
        best_dist = dist;
        best_idx = static_cast<int>(i);
      }
    }

    if (best_idx >= 0) {
      auto & e = entries_[static_cast<std::size_t>(best_idx)];
      const double a = config_.ema_alpha;
      e.wx = a * wx + (1.0 - a) * e.wx;
      e.wy = a * wy + (1.0 - a) * e.wy;
      e.half_len_m = a * std::max(0.02, det.half_len_m) + (1.0 - a) * e.half_len_m;
      e.half_wid_m = a * std::max(0.02, det.half_wid_m) + (1.0 - a) * e.half_wid_m;
      e.confidence = std::max(e.confidence * 0.85, det.confidence);
      e.hit_count++;
      e.last_seen_sec = now_sec;
    } else {
      TrackedObstacle entry{};
      entry.id = next_id_++;
      entry.type = det.type;
      entry.wx = wx;
      entry.wy = wy;
      entry.half_len_m = std::max(0.02, det.half_len_m);
      entry.half_wid_m = std::max(0.02, det.half_wid_m);
      entry.confidence = det.confidence;
      entry.hit_count = 1;
      entry.first_seen_sec = now_sec;
      entry.last_seen_sec = now_sec;
      entries_.push_back(entry);
    }
  }

  prune(pose, now_sec);
}

void ObstacleMemory::prune(const VehiclePose2D & pose, double now_sec)
{
  const double min_rear_d = -(config_.car_length_m + config_.rear_keep_margin_m);
  const double max_fwd_d = config_.max_forward_dist_m + 0.5;

  entries_.erase(
    std::remove_if(
      entries_.begin(),
      entries_.end(),
      [&](const TrackedObstacle & e) {
        if ((now_sec - e.last_seen_sec) > config_.memory_ttl_sec) {
          return true;
        }
        if (pose.valid) {
          double d_bumper = 0.0;
          double y_lat = 0.0;
          world_to_bumper(
            pose, e.wx, e.wy, d_bumper, y_lat, config_.rear_axle_to_bumper_m);
          // Expire only once the ENTIRE obstacle footprint is behind the rear-keep limit
          if ((d_bumper + e.half_len_m) < min_rear_d) {
            return true;
          }
          if ((d_bumper - e.half_len_m) > max_fwd_d) {
            return true;
          }
        }
        return false;
      }),
    entries_.end());

  if (entries_.size() > config_.max_entries) {
    std::sort(
      entries_.begin(),
      entries_.end(),
      [](const TrackedObstacle & a, const TrackedObstacle & b) {
        return a.last_seen_sec > b.last_seen_sec;
      });
    entries_.resize(config_.max_entries);
  }
}

std::vector<LocalObstacle> ObstacleMemory::get_local_obstacles(
  const VehiclePose2D & pose,
  double now_sec) const
{
  std::vector<LocalObstacle> out;
  out.reserve(entries_.size());

  const double min_rear_d = -(config_.car_length_m + config_.rear_keep_margin_m);

  VehiclePose2D ref_pose = pose;
  if (!ref_pose.valid) {
    ref_pose.x = 0.0;
    ref_pose.y = 0.0;
    ref_pose.yaw = 0.0;
  }

  for (const auto & e : entries_) {
    const double age = now_sec - e.last_seen_sec;
    if (age > config_.memory_ttl_sec) {
      continue;
    }

    double d_bumper = 0.0;
    double y_lat = 0.0;
    world_to_bumper(
      ref_pose, e.wx, e.wy, d_bumper, y_lat, config_.rear_axle_to_bumper_m);

    if ((d_bumper + e.half_len_m) < min_rear_d) {
      continue;
    }

    LocalObstacle loc{};
    loc.id = e.id;
    loc.type = e.type;
    loc.d_bumper_m = d_bumper;
    loc.d_min_m = d_bumper - e.half_len_m;
    loc.d_max_m = d_bumper + e.half_len_m;
    loc.y_lat_m = y_lat;
    loc.y_min_m = y_lat - e.half_wid_m;
    loc.y_max_m = y_lat + e.half_wid_m;
    loc.confidence = e.confidence;
    loc.age_sec = std::max(0.0, age);
    out.push_back(loc);
  }

  return out;
}

const std::vector<TrackedObstacle> & ObstacleMemory::entries() const
{
  return entries_;
}

std::size_t ObstacleMemory::size() const
{
  return entries_.size();
}

}  // namespace line_follower
