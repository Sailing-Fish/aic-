#pragma once

#include "line_follower/line_follower_types.hpp"

namespace line_follower
{

struct MotionControllerConfig
{
  double target_speed{0.6};
  double max_speed{1.0};
  double min_speed{0.0};
  double max_yaw_rate{1.2};
  double speed_gain_damping{0.5};

  double deviation_weight{0.30};
  double curve_weight{0.90};

  // Low-confidence speed limit & yaw-hold policy
  double low_confidence_threshold{0.0};
  double low_confidence_speed{0.26};
  double low_confidence_yaw_hold{0.75};

  double line_lost_forward_speed{0.06};
  double line_lost_yaw_hold_factor{0.85};
  double line_lost_slow_time{0.4};
  double line_lost_slow_factor{0.4};
  double line_lost_full_stop_time{1.2};

  // Finals safe lost-line override (default false preserves unit tests)
  bool lost_line_safe_mode{false};
  double lost_line_safe_speed{0.12};
  double lost_line_safe_stop_time{0.35};

  // Finals metric lateral corridor tracking gains
  double corridor_kp_y{3.8};
  double corridor_heading_gain{0.85};
};

class MotionController
{
public:
  MotionController();
  explicit MotionController(const MotionControllerConfig & config);

  void set_config(const MotionControllerConfig & config);
  const MotionControllerConfig & config() const;

  // Finals metric lateral corridor command (+y_m is LEFT, +yaw_rate is LEFT)
  MotionCommand compute_corridor_command(
    double target_y_m,
    double current_y_m,
    double heading_proxy,
    double curvature_proxy,
    double confidence,
    double speed_cap_mps,
    const CurveAssistState & curve_state,
    const MotionState & motion_state) const;

  MotionCommand compute_lost_line_command(
    double since_detection_sec,
    const MotionState & motion_state) const;

  static void record_motion(
    bool yellow_stop_now,
    double linear_speed,
    double yaw_rate,
    MotionState & motion_state);

private:
  MotionControllerConfig config_{};
};

}  // namespace line_follower
