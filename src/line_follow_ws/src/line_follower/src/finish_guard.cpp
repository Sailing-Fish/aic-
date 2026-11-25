#include "line_follower/finish_guard.hpp"

#include <algorithm>
#include <cmath>

namespace line_follower
{

FinishGuard::FinishGuard(const FinishGuardConfig & config)
{
  set_config(config);
}

void FinishGuard::set_config(const FinishGuardConfig & config)
{
  config_ = config;
  config_.odom_scale_factor = std::max(0.1, config_.odom_scale_factor);
  config_.finish_min_dist_m = std::max(0.5, config_.finish_min_dist_m);
  config_.finish_force_dist_m = std::max(config_.finish_min_dist_m, config_.finish_force_dist_m);
  config_.min_cumulative_abs_yaw_rad = std::max(0.0, config_.min_cumulative_abs_yaw_rad);
  config_.pre_brake_dist_m = std::max(0.15, config_.pre_brake_dist_m);
  config_.creep_speed_mps = std::max(0.05, config_.creep_speed_mps);
  config_.brake_decel_mps2 = std::max(0.2, config_.brake_decel_mps2);
  config_.control_latency_s = std::max(0.0, config_.control_latency_s);
  config_.bumper_margin_m = std::max(0.0, config_.bumper_margin_m);
  config_.min_stop_trigger_dist_m = std::max(0.02, config_.min_stop_trigger_dist_m);
  config_.max_stop_trigger_dist_m =
    std::max(config_.min_stop_trigger_dist_m, config_.max_stop_trigger_dist_m);
  config_.rearm_travel_dist_m = std::max(0.05, config_.rearm_travel_dist_m);
}

const FinishGuardConfig & FinishGuard::config() const
{
  return config_;
}

void FinishGuard::reset()
{
  s_total_m_ = 0.0;
  cum_abs_yaw_rad_ = 0.0;
  dist_since_last_yellow_m_ = 1e6;
  fork_passed_ = false;
}

void FinishGuard::update_motion(double ds_fwd_m, double dyaw_rad)
{
  const double ds = std::max(0.0, ds_fwd_m) * config_.odom_scale_factor;
  s_total_m_ += ds;
  dist_since_last_yellow_m_ += ds;
  cum_abs_yaw_rad_ += std::abs(dyaw_rad);
}

void FinishGuard::mark_fork_passed()
{
  fork_passed_ = true;
}

void FinishGuard::mark_yellow_stop_triggered()
{
  dist_since_last_yellow_m_ = 0.0;
}

bool FinishGuard::is_yellow_rearmed() const
{
  return dist_since_last_yellow_m_ >= config_.rearm_travel_dist_m;
}

bool FinishGuard::is_finish_armed() const
{
  return s_total_m_ >= config_.finish_min_dist_m &&
         cum_abs_yaw_rad_ >= config_.min_cumulative_abs_yaw_rad;
}

bool FinishGuard::evaluate_is_finish(bool visual_dead_end_ahead) const
{
  if (!is_finish_armed()) {
    return false;
  }
  return visual_dead_end_ahead || (s_total_m_ >= config_.finish_force_dist_m);
}

double FinishGuard::compute_stop_trigger_dist(double current_speed_mps) const
{
  const double v = std::max(0.0, std::abs(current_speed_mps));
  const double raw =
    (v * v) / (2.0 * config_.brake_decel_mps2) +
    v * config_.control_latency_s +
    config_.bumper_margin_m;
  return std::clamp(raw, config_.min_stop_trigger_dist_m, config_.max_stop_trigger_dist_m);
}

YellowStopAdvice FinishGuard::evaluate_yellow_stop(
  bool yellow_detected,
  bool has_metric_distance,
  double d_bumper_m,
  double current_speed_mps,
  bool visual_dead_end_ahead) const
{
  YellowStopAdvice advice{};
  advice.d_bumper_m = d_bumper_m;
  advice.stop_trigger_dist_m = compute_stop_trigger_dist(current_speed_mps);
  advice.speed_cap_mps = 1e9;

  if (!yellow_detected) {
    return advice;
  }

  advice.is_finish_line = evaluate_is_finish(visual_dead_end_ahead);

  // While not yet re-armed after a temporary stop, never emit pre-brake or stop triggers
  // (prevents 0.0 m/s deadlock right in front of the yellow line just stopped for).
  if (!is_yellow_rearmed()) {
    return advice;
  }

  if (!has_metric_distance) {
    advice.pre_brake = true;
    advice.trigger_stop = true;
    advice.speed_cap_mps = 0.0;
    return advice;
  }

  if (d_bumper_m <= advice.stop_trigger_dist_m) {
    advice.pre_brake = true;
    advice.trigger_stop = true;
    advice.speed_cap_mps = 0.0;
  } else if (d_bumper_m <= config_.pre_brake_dist_m) {
    advice.pre_brake = true;
    advice.trigger_stop = false;
    advice.speed_cap_mps = config_.creep_speed_mps;
  }

  return advice;
}

}  // namespace line_follower
