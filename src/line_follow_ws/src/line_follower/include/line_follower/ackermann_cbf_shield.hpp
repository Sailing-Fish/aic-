#pragma once

#include <array>
#include <string>
#include <vector>

#include "line_follower/finals_types.hpp"

namespace line_follower
{

struct AckermannCbfConfig
{
  // Vehicle geometry (280x191mm chassis, rear-axle EKF reference)
  double car_length_m{0.280};
  double car_width_m{0.191};
  double rear_axle_to_bumper_m{0.180};
  double wheelbase_m{0.142};
  double max_steer_rad{0.6108652381980153};  // 35 deg
  double max_yaw_rate{1.6};

  // Fallback lane width & clearance margins
  double default_lane_width_m{0.430};
  double border_margin_m{0.025};
  double cone_margin_m{0.025};
  double no_entry_margin_m{0.030};

  // Multi-step Discrete CBF rollout parameters
  int horizon_steps{6};
  double step_dt_sec{0.06};
  double cbf_gamma{0.60};
  int yaw_sample_count{25};
};

struct CbfShieldResult
{
  double linear_speed{0.0};
  double yaw_rate{0.0};
  bool intervened{false};
  double min_clearance_m{0.0};
  std::string limiting_source{"none"};
};

// ============================================================
// AckermannCbfShield (pure C++17, zero OpenCV/ROS2 deps)
//
// 3-disc vehicle footprint covering the 280x191mm chassis with
// rear/mid/front discs along the longitudinal centerline,
// rolled out over horizon_steps via Ackermann kinematics.
// Enforces Discrete Control Barrier Function (DCBF) constraints
// against:
//   1. Oriented Bounding Box (OBB) red no-entry & red cone obstacles
//   2. Left & right white-lane boundary hard limits
// ============================================================
class AckermannCbfShield
{
public:
  AckermannCbfShield() = default;
  explicit AckermannCbfShield(const AckermannCbfConfig & config);

  void set_config(const AckermannCbfConfig & config);
  const AckermannCbfConfig & config() const;

  CbfShieldResult filter(
    double nom_linear_speed,
    double nom_yaw_rate,
    double y_lane_left_m,
    double y_lane_right_m,
    const std::vector<LocalObstacle> & obstacles) const;

  // Analytical signed distance from point (px, py) to OBB centered at (cx, cy)
  // with half-extents (half_len, half_wid) and orientation yaw_rad.
  // Negative inside the OBB, zero on boundary, positive outside.
  static double point_to_obb_sdf(
    double px,
    double py,
    double cx,
    double cy,
    double half_len,
    double half_wid,
    double yaw_rad);

  // Longitudinal offsets [m] of the 3 covering discs in the rear-axle frame.
  static std::array<double, 3> compute_disc_offsets(const AckermannCbfConfig & config);

private:
  AckermannCbfConfig config_{};
};

}  // namespace line_follower
