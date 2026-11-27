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
  config_.yellow_final_ramp_dist_m = std::max(0.0, config_.yellow_final_ramp_dist_m);
  config_.yellow_final_creep_speed_mps = std::clamp(
    config_.yellow_final_creep_speed_mps, 0.05, config_.creep_speed_mps);
  config_.brake_decel_mps2 = std::max(0.2, config_.brake_decel_mps2);
  config_.control_latency_s = std::max(0.0, config_.control_latency_s);
  config_.bumper_margin_m = std::max(0.0, config_.bumper_margin_m);
  config_.min_stop_trigger_dist_m = std::max(0.02, config_.min_stop_trigger_dist_m);
  config_.max_stop_trigger_dist_m =
    std::max(config_.min_stop_trigger_dist_m, config_.max_stop_trigger_dist_m);
  config_.voltage_nominal_v = std::clamp(config_.voltage_nominal_v, 9.0, 15.0);
  config_.voltage_timeout_sec = std::max(0.1, config_.voltage_timeout_sec);
  config_.brake_adapt_ema_alpha = std::clamp(config_.brake_adapt_ema_alpha, 0.05, 1.0);
  config_.friction_scale_min = std::clamp(config_.friction_scale_min, 0.30, 1.0);
  config_.friction_scale_max = std::clamp(
    config_.friction_scale_max, config_.friction_scale_min, 2.0);
  config_.brake_damping_gain = std::clamp(config_.brake_damping_gain, 0.0, 5.0);
  config_.brake_zero_cutoff_mps = std::clamp(config_.brake_zero_cutoff_mps, 0.0, 0.20);
  config_.rearm_travel_dist_m = std::max(0.05, config_.rearm_travel_dist_m);
  friction_scale_ = std::clamp(
    friction_scale_, config_.friction_scale_min, config_.friction_scale_max);
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
  battery_voltage_v_ = config_.voltage_nominal_v;
  last_voltage_sec_ = -1e9;
  voltage_valid_ = false;
  friction_scale_ = 1.0;
  brake_adapt_updates_ = 0;
}

void FinishGuard::update_motion(double ds_fwd_m, double dyaw_rad)
{
  const double ds = std::max(0.0, ds_fwd_m) * config_.odom_scale_factor;
  s_total_m_ += ds;
  dist_since_last_yellow_m_ += ds;
  cum_abs_yaw_rad_ += std::abs(dyaw_rad);
}

void FinishGuard::update_battery_voltage(double voltage_v, double now_sec)
{
  if (!std::isfinite(voltage_v) || voltage_v < 9.0 || voltage_v > 15.0) {
    return;
  }
  battery_voltage_v_ = voltage_v;
  last_voltage_sec_ = now_sec;
  voltage_valid_ = true;
}

double FinishGuard::active_battery_voltage(double now_sec) const
{
  if (!voltage_valid_) {
    return config_.voltage_nominal_v;
  }
  if (now_sec >= 0.0 && (now_sec - last_voltage_sec_) > config_.voltage_timeout_sec) {
    return config_.voltage_nominal_v;
  }
  return battery_voltage_v_;
}

double FinishGuard::effective_brake_decel_mps2(double now_sec) const
{
  const double v_bat = active_battery_voltage(now_sec);
  const double base_decel =
    config_.brake_decel_mps2 +
    config_.voltage_decel_slope * (v_bat - config_.voltage_nominal_v);
  const double scale = config_.enable_brake_adaptation ? friction_scale_ : 1.0;
  return std::clamp(base_decel * scale, 0.5, 4.0);
}

bool FinishGuard::update_measured_yellow_stop(
  double v_entry_mps,
  double stop_travel_m,
  double now_sec)
{
  if (!config_.enable_brake_adaptation) {
    return false;
  }
  if (!std::isfinite(v_entry_mps) || !std::isfinite(stop_travel_m)) {
    return false;
  }
  if (v_entry_mps < 0.12 || stop_travel_m <= 0.005) {
    return false;
  }

  const double v_bat = active_battery_voltage(now_sec);
  const double nominal_decel = std::clamp(
    config_.brake_decel_mps2 +
    config_.voltage_decel_slope * (v_bat - config_.voltage_nominal_v),
    0.5,
    4.0);
  const double measured_decel = (v_entry_mps * v_entry_mps) / (2.0 * stop_travel_m);
  const double raw_scale = std::clamp(
    measured_decel / nominal_decel,
    config_.friction_scale_min,
    config_.friction_scale_max);

  const double a = config_.brake_adapt_ema_alpha;
  friction_scale_ = std::clamp(
    (1.0 - a) * friction_scale_ + a * raw_scale,
    config_.friction_scale_min,
    config_.friction_scale_max);
  brake_adapt_updates_++;
  return true;
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

double FinishGuard::compute_stop_trigger_dist(double current_speed_mps, double now_sec) const
{
  const double v = std::max(0.0, std::abs(current_speed_mps));
  const double a_eff = effective_brake_decel_mps2(now_sec);
  const double raw =
    (v * v) / (2.0 * a_eff) +
    v * config_.control_latency_s +
    config_.bumper_margin_m;
  return std::clamp(raw, config_.min_stop_trigger_dist_m, config_.max_stop_trigger_dist_m);
}

double FinishGuard::compute_closed_loop_brake_cmd(
  double odom_linear_x,
  double elapsed_s,
  double max_brake_speed,
  double max_time_s) const
{
  if (!std::isfinite(odom_linear_x) || !std::isfinite(elapsed_s) ||
    !std::isfinite(max_brake_speed) || !std::isfinite(max_time_s))
  {
    return 0.0;
  }
  const double v_max = std::max(0.0, max_brake_speed);
  if (v_max <= 0.0 || max_time_s <= 0.0 || elapsed_s < 0.0 || elapsed_s >= max_time_s) {
    return 0.0;
  }
  if (odom_linear_x <= config_.brake_zero_cutoff_mps) {
    return 0.0;
  }
  const double mag = std::clamp(config_.brake_damping_gain * odom_linear_x, 0.0, v_max);
  return (mag > 1e-9) ? -mag : 0.0;
}

double FinishGuard::compute_safe_speed_ceiling(
  double preview_dist_m,
  double target_end_speed_mps,
  double now_sec) const
{
  const double a_eff = effective_brake_decel_mps2(now_sec);
  const double t_lat = config_.control_latency_s;
  const double margin = config_.bumper_margin_m;
  const double v_end =
    (std::isfinite(target_end_speed_mps) && target_end_speed_mps > 0.0) ? target_end_speed_mps : 0.0;
  const double d_avail =
    std::isfinite(preview_dist_m) ? std::max(0.0, preview_dist_m - margin) : 0.0;

  // Exact positive root of: v^2/(2*a_eff) + v*t_lat = d_avail + v_end^2/(2*a_eff)
  const double at = a_eff * t_lat;
  const double disc = at * at + 2.0 * a_eff * d_avail + v_end * v_end;
  const double v_root = std::max(0.0, -at + std::sqrt(std::max(0.0, disc)));
  return std::max(v_end, v_root);
}

YellowStopAdvice FinishGuard::evaluate_yellow_stop(
  bool yellow_detected,
  bool has_metric_distance,
  double d_bumper_m,
  double current_speed_mps,
  bool visual_dead_end_ahead,
  double now_sec) const
{
  YellowStopAdvice advice{};
  advice.d_bumper_m = d_bumper_m;
  advice.stop_trigger_dist_m = compute_stop_trigger_dist(current_speed_mps, now_sec);
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
    const double ramp_dist = std::min(config_.yellow_final_ramp_dist_m, config_.pre_brake_dist_m);
    const double v_final = std::min(config_.yellow_final_creep_speed_mps, config_.creep_speed_mps);
    double pre_brake_cap = config_.creep_speed_mps;
    if (ramp_dist > advice.stop_trigger_dist_m + 1e-6 && d_bumper_m < ramp_dist) {
      const double alpha = std::clamp(
        (d_bumper_m - advice.stop_trigger_dist_m) / (ramp_dist - advice.stop_trigger_dist_m),
        0.0,
        1.0);
      pre_brake_cap = v_final + alpha * (config_.creep_speed_mps - v_final);
    }
    if (config_.enable_sqrt_brake_envelope) {
      const double a_eff = effective_brake_decel_mps2(now_sec);
      const double d_rem = std::max(0.0, d_bumper_m - advice.stop_trigger_dist_m);
      const double v_env = std::sqrt(2.0 * a_eff * d_rem);
      advice.speed_cap_mps = std::clamp(v_env, v_final, pre_brake_cap);
    } else {
      advice.speed_cap_mps = pre_brake_cap;
    }
  }

  return advice;
}

}  // namespace line_follower
