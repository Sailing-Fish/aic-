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

  // Clear-path physics-envelope adaptive speed
  bool enable_clear_path_speed{true};
  double clear_straight_speed{0.62};
  double clear_preview_dist_m{0.95};
  double max_accel_mps2{0.90};
  double max_decel_mps2{2.20};
  double active_decel_kp{0.45};
};

struct ClearPathInput
{
  bool has_active_obstacle{false};
  bool has_no_entry_ahead{false};
  bool has_yellow_ahead{false};
  bool has_digit_target{false};
  bool has_fork_context{false};
  double confidence{0.0};
  double curvature_proxy{0.0};
  double heading_proxy{0.0};
  double lateral_err_m{0.0};
  double lane_width_ratio{0.58};
  bool reverse_curve_active{false};
  double v_safe_cap_mps{1e9};
  double odom_linear_x{0.0};
  bool odom_valid{false};
  double dt_sec{0.0};
  double prev_speed_cmd_mps{-1.0};
};

struct AdaptiveSpeedResult
{
  MotionCommand command{};
  double clear_path_score{0.0};
  double effective_speed_cap_mps{0.0};
  double blended_cruise_speed_mps{0.0};
};

class MotionController
{
public:
  MotionController();
  explicit MotionController(const MotionControllerConfig & config);

  void set_config(const MotionControllerConfig & config);
  const MotionControllerConfig & config() const;
  void reset();
  void notify_external_speed(double linear_speed_mps, double now_sec);

  double compute_clear_path_score(const ClearPathInput & clear_input) const;

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

  // Stateless clear-path adaptive speed + physics ceiling + asymmetric slew + active decel
  AdaptiveSpeedResult compute_adaptive_corridor_command(
    double target_y_m,
    double current_y_m,
    double heading_proxy,
    double curvature_proxy,
    double confidence,
    double speed_cap_mps,
    const CurveAssistState & curve_state,
    const MotionState & motion_state,
    const ClearPathInput & clear_input) const;

  // Stateful wrapper managing dt & previous speed command automatically
  AdaptiveSpeedResult step_adaptive_corridor_command(
    double target_y_m,
    double current_y_m,
    double heading_proxy,
    double curvature_proxy,
    double confidence,
    double speed_cap_mps,
    const CurveAssistState & curve_state,
    const MotionState & motion_state,
    const ClearPathInput & clear_input,
    double now_sec);

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
  double last_speed_cmd_mps_{-1.0};
  double last_step_time_sec_{-1e9};
  bool has_step_time_{false};
};

}  // namespace line_follower
