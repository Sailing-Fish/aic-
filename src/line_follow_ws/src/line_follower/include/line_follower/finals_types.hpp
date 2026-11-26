#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace line_follower
{

// ============================================================
// National Finals Pure C++17 Data Types (Zero OpenCV/ROS2 deps)
//
// Coordinate conventions:
//   Vehicle bumper frame:
//     Origin (0, 0) at front foam bumper center.
//     +d_bumper_m : forward ahead of bumper [m] (negative = beside/behind bumper)
//     +y_lat_m    : left of vehicle centerline [m] (ROS REP-103; negative = right)
//   World / EKF frame (/odom_combined):
//     (wx, wy) in meters, yaw in radians (+CCW / left turn).
// ============================================================

enum class SemanticObjectType
{
  UNKNOWN = 0,
  RED_CONE,
  RED_NO_ENTRY,
  BONUS_DIGIT,
  YELLOW_STOP_LINE
};

inline const char * semantic_object_type_to_string(SemanticObjectType type)
{
  switch (type) {
    case SemanticObjectType::RED_CONE:
      return "red_cone";
    case SemanticObjectType::RED_NO_ENTRY:
      return "red_no_entry";
    case SemanticObjectType::BONUS_DIGIT:
      return "bonus_digit";
    case SemanticObjectType::YELLOW_STOP_LINE:
      return "yellow_stop_line";
    case SemanticObjectType::UNKNOWN:
    default:
      return "unknown";
  }
}

struct VehiclePose2D
{
  bool valid{false};
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double linear_v{0.0};
  double angular_w{0.0};
  double stamp_sec{0.0};
};

struct GroundDetection
{
  SemanticObjectType type{SemanticObjectType::UNKNOWN};
  int digit{0};          // 1..8 when BONUS_DIGIT, else 0
  double confidence{0.0};
  double d_bumper_m{0.0};
  double y_lat_m{0.0};
  double half_len_m{0.0675};
  double half_wid_m{0.0675};
  int track_id{-1};
  double stamp_sec{0.0};
};

struct TrackedObstacle
{
  int id{0};
  SemanticObjectType type{SemanticObjectType::RED_CONE};
  double wx{0.0};
  double wy{0.0};
  double half_len_m{0.0675};
  double half_wid_m{0.0675};
  double obs_yaw_rad{0.0};
  double confidence{0.0};
  int hit_count{1};
  double first_seen_sec{0.0};
  double last_seen_sec{0.0};
};

struct LocalObstacle
{
  int id{0};
  SemanticObjectType type{SemanticObjectType::RED_CONE};
  double d_bumper_m{0.0};
  double d_min_m{0.0};
  double d_max_m{0.0};
  double y_lat_m{0.0};
  double y_min_m{0.0};   // right edge (+y is left)
  double y_max_m{0.0};   // left edge
  double rel_yaw_rad{0.0};
  double confidence{0.0};
  double age_sec{0.0};
};

struct LateralInterval
{
  double y_min{0.0};
  double y_max{0.0};

  double width() const
  {
    return y_max - y_min;
  }

  double center() const
  {
    return 0.5 * (y_min + y_max);
  }
};

struct CorridorPlan
{
  double target_y_m{0.0};            // Desired vehicle lateral offset in lane frame (+y left)
  double speed_cap_mps{1.0};         // Maximum safe longitudinal speed
  int preferred_branch{0};           // -1 = left branch, +1 = right branch, 0 = none
  bool has_blocking_obstacle{false}; // True if no collision-free corridor exists
  bool digit_target_active{false};   // True when actively steering for a straight Bonus digit
  int target_digit{0};               // 1..8 when targeting a digit
  double safe_y_min_m{-0.095};       // Chosen feasible interval right bound
  double safe_y_max_m{0.095};        // Chosen feasible interval left bound
  int near_obstacle_count{0};        // Obstacles in immediate steering horizon
  int far_obstacle_count{0};         // Obstacles in far preview decel horizon
  std::string active_mode_tag{"cruise"};
};

struct YellowStopAdvice
{
  bool pre_brake{false};
  bool trigger_stop{false};
  bool is_finish_line{false};
  double speed_cap_mps{1.0};
  double stop_trigger_dist_m{0.0};
  double d_bumper_m{0.0};
};

// ============================================================
// Red Obstacle OR-Fusion & Source Attribution (YOLO || CV)
//
// Policy:
//   - Logical OR: if either YOLO or white-lane CV sees a red obstacle,
//     it enters ObstacleMemory / CorridorPlanner.
//   - Deduplication: when both see the same obstacle (within match_radius_m),
//     keep the higher-precision YOLO box/class and attribute source = BOTH.
//   - Attribution: cleanly separates YOLO_ONLY, BOTH, CV_ONLY so logs prove
//     whether YOLO or CV is doing the work during track runs.
// ============================================================

enum class RedDetectionSource
{
  NONE = 0,
  YOLO_ONLY,
  CV_ONLY,
  BOTH
};

inline const char * red_detection_source_to_string(RedDetectionSource src)
{
  switch (src) {
    case RedDetectionSource::YOLO_ONLY:
      return "yolo_only";
    case RedDetectionSource::CV_ONLY:
      return "cv_only";
    case RedDetectionSource::BOTH:
      return "both";
    case RedDetectionSource::NONE:
    default:
      return "none";
  }
}

struct RedFusionSummary
{
  RedDetectionSource overall_source{RedDetectionSource::NONE};
  int yolo_red_count{0};
  int cv_red_count{0};
  int matched_both_count{0};
  int yolo_only_count{0};
  int cv_only_count{0};
  double nearest_d_bumper_m{0.0};
  double nearest_y_lat_m{0.0};
  SemanticObjectType nearest_type{SemanticObjectType::UNKNOWN};
  std::vector<GroundDetection> fused_red_detections{};
  std::vector<RedDetectionSource> fused_sources{};
  std::vector<bool> cv_matched_flags{};
};

inline RedFusionSummary fuse_red_obstacles_or(
  const std::vector<GroundDetection> & yolo_red,
  const std::vector<GroundDetection> & cv_red,
  double match_radius_m = 0.24)
{
  RedFusionSummary out{};
  out.yolo_red_count = static_cast<int>(yolo_red.size());
  out.cv_red_count = static_cast<int>(cv_red.size());
  out.cv_matched_flags.assign(cv_red.size(), false);

  const double r2 = std::max(0.02, match_radius_m) * std::max(0.02, match_radius_m);

  // 1) Keep all YOLO red detections; greedily match closest unmatched CV detection
  for (const auto & yd : yolo_red) {
    int best_cv = -1;
    double best_d2 = r2;
    for (size_t j = 0; j < cv_red.size(); ++j) {
      if (out.cv_matched_flags[j]) {
        continue;
      }
      const double dd = yd.d_bumper_m - cv_red[j].d_bumper_m;
      const double dy = yd.y_lat_m - cv_red[j].y_lat_m;
      const double d2 = dd * dd + dy * dy;
      if (d2 <= best_d2) {
        best_d2 = d2;
        best_cv = static_cast<int>(j);
      }
    }
    out.fused_red_detections.push_back(yd);
    if (best_cv >= 0) {
      out.cv_matched_flags[static_cast<size_t>(best_cv)] = true;
      out.fused_sources.push_back(RedDetectionSource::BOTH);
      out.matched_both_count++;
    } else {
      out.fused_sources.push_back(RedDetectionSource::YOLO_ONLY);
      out.yolo_only_count++;
    }
  }

  // 2) OR-fusion: inject any unmatched CV red detections as CV_ONLY fallback
  for (size_t j = 0; j < cv_red.size(); ++j) {
    if (!out.cv_matched_flags[j]) {
      out.fused_red_detections.push_back(cv_red[j]);
      out.fused_sources.push_back(RedDetectionSource::CV_ONLY);
      out.cv_only_count++;
    }
  }

  if (out.yolo_red_count > 0 && out.cv_red_count > 0) {
    out.overall_source = RedDetectionSource::BOTH;
  } else if (out.yolo_red_count > 0) {
    out.overall_source = RedDetectionSource::YOLO_ONLY;
  } else if (out.cv_red_count > 0) {
    out.overall_source = RedDetectionSource::CV_ONLY;
  } else {
    out.overall_source = RedDetectionSource::NONE;
  }

  // 3) Nearest obstacle summary for concise logging
  bool has_nearest = false;
  double min_d = 1e9;
  for (const auto & gd : out.fused_red_detections) {
    if (!has_nearest || gd.d_bumper_m < min_d) {
      has_nearest = true;
      min_d = gd.d_bumper_m;
      out.nearest_d_bumper_m = gd.d_bumper_m;
      out.nearest_y_lat_m = gd.y_lat_m;
      out.nearest_type = gd.type;
    }
  }

  return out;
}

}  // namespace line_follower
