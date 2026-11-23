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

}  // namespace line_follower
