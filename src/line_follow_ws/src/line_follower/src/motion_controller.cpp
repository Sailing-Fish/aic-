#include "line_follower/motion_controller.hpp"

#include <algorithm>
#include <cmath>

namespace line_follower
{

MotionController::MotionController()
{
  set_config(MotionControllerConfig{});
}

MotionController::MotionController(const MotionControllerConfig & config)
{
  set_config(config);
}

void MotionController::set_config(const MotionControllerConfig & config)
{
  config_ = config;
  config_.clear_straight_speed = std::max(0.0, config_.clear_straight_speed);
  config_.clear_preview_dist_m = std::max(0.10, config_.clear_preview_dist_m);
  config_.max_accel_mps2 = std::max(0.05, config_.max_accel_mps2);
  config_.max_decel_mps2 = std::max(0.05, config_.max_decel_mps2);
  config_.active_decel_kp = clamp_value(config_.active_decel_kp, 0.0, 2.0);
}

const MotionControllerConfig & MotionController::config() const
{
  return config_;
}

void MotionController::reset()
{
  last_speed_cmd_mps_ = -1.0;
  last_step_time_sec_ = -1e9;
  has_step_time_ = false;
}

void MotionController::notify_external_speed(double linear_speed_mps, double now_sec)
{
  last_speed_cmd_mps_ = std::isfinite(linear_speed_mps) ? std::max(0.0, linear_speed_mps) : 0.0;
  if (std::isfinite(now_sec)) {
    last_step_time_sec_ = now_sec;
    has_step_time_ = true;
  }
}

double MotionController::compute_clear_path_score(const ClearPathInput & cp) const
{
  if (!config_.enable_clear_path_speed) {
    return 0.0;
  }
  if (cp.has_active_obstacle || cp.has_no_entry_ahead || cp.has_yellow_ahead ||
    cp.has_digit_target || cp.has_fork_context || cp.reverse_curve_active)
  {
    return 0.0;
  }
  if (!std::isfinite(cp.confidence) || !std::isfinite(cp.curvature_proxy) ||
    !std::isfinite(cp.heading_proxy) || !std::isfinite(cp.lateral_err_m) ||
    !std::isfinite(cp.lane_width_ratio))
  {
    return 0.0;
  }

  // 1. High lane confidence gate
  const double conf_lo = std::max(0.65, config_.low_confidence_threshold + 0.05);
  const double conf_hi = std::max(conf_lo + 0.05, 0.85);
  const double f_conf = clamp_value((cp.confidence - conf_lo) / (conf_hi - conf_lo), 0.0, 1.0);

  // 2. Small curvature & heading proxy
  const double f_curv = clamp_value((0.22 - std::abs(cp.curvature_proxy)) / (0.22 - 0.06), 0.0, 1.0);
  const double f_hdg = clamp_value((0.20 - std::abs(cp.heading_proxy)) / (0.20 - 0.06), 0.0, 1.0);

  // 3. Centered lateral offset inside 43cm lane
  const double f_lat = clamp_value((0.09 - std::abs(cp.lateral_err_m)) / (0.09 - 0.03), 0.0, 1.0);

  // 4. Normal lane width (suppresses boost in wide Y-junction / roundabout flare or degenerate narrow mask)
  const double f_w_hi = clamp_value((0.82 - cp.lane_width_ratio) / (0.82 - 0.72), 0.0, 1.0);
  const double f_w_lo = clamp_value((cp.lane_width_ratio - 0.30) / (0.42 - 0.30), 0.0, 1.0);
  const double f_width = f_w_hi * f_w_lo;

  return clamp_value(f_conf * f_curv * f_hdg * f_lat * f_width, 0.0, 1.0);
}

MotionCommand MotionController::compute_corridor_command(
  double target_y_m,
  double current_y_m,
  double heading_proxy,
  double curvature_proxy,
  double confidence,
  double speed_cap_mps,
  const CurveAssistState & curve_state,
  const MotionState & motion_state) const
{
  // +y is LEFT, +yaw_rate is LEFT; heading_proxy > 0 means far lane bends RIGHT.
  // Keep 45% heading guidance even during reverse_curve_active so the car never stops
  // turning mid-fork or mid-hairpin.
  const double lateral_err_m = target_y_m - current_y_m;
  const double eff_heading_gain =
    curve_state.reverse_curve_active ?
    (0.45 * config_.corridor_heading_gain) :
    config_.corridor_heading_gain;

  const double lat_term = config_.corridor_kp_y * lateral_err_m;
  double hdg_term = -eff_heading_gain * heading_proxy;

  // Anti-off-track heading guard:
  // near_error < 0 means near lane is on LEFT (car is close to RIGHT border), while
  // hdg_term < 0 steers RIGHT (further off that close border). When the car is already
  // displaced toward an edge and the corridor target is NOT pushing toward that edge
  // (e.g. recovering from obstacle dodge or hugging an inner corner), prevent heading_proxy
  // (= far - near) from overpowering lateral recovery and pushing the car off-track.
  if (curve_state.near_valid && curve_state.last_lane_width_ratio < 0.80) {
    const double abs_near = std::abs(curve_state.near_error);
    const bool pushing_close_edge = (hdg_term * curve_state.near_error) > 0.0;
    const bool corridor_not_pushing_edge = (lat_term * curve_state.near_error) <= 0.0;
    if (abs_near > 0.20 && pushing_close_edge && corridor_not_pushing_edge) {
      const bool genuine_opposite_far =
        curve_state.far_valid && ((curve_state.near_error * curve_state.far_error) < 0.0);
      if (!genuine_opposite_far) {
        const double scale = clamp_value((0.35 - abs_near) / 0.15, 0.0, 1.0);
        hdg_term *= scale;
      } else {
        const double scale = clamp_value((0.55 - abs_near) / 0.35, 0.0, 1.0);
        hdg_term *= scale;
        if (abs_near >= 0.35 && std::abs(lat_term) > 1e-6) {
          const double max_opp = 0.50 * std::abs(lat_term);
          hdg_term = clamp_value(hdg_term, -max_opp, max_opp);
        }
      }
    }
  }

  double yaw_cmd = clamp_value(
    lat_term + hdg_term,
    -config_.max_yaw_rate,
    config_.max_yaw_rate);

  // Orthogonal speed law:
  //   1. Cruise + proportional curve damping: target_speed * (1 - speed_gain_damping * slow_factor)
  //   2. Independent event ceilings (obstacle / no_entry / digit / yellow creep): min(..., speed_cap_mps)
  const double norm_lat = clamp_value(std::abs(lateral_err_m) / 0.215, 0.0, 1.0);
  const double curve_proxy = std::max(
    std::abs(curve_state.far_error - curve_state.near_error),
    std::abs(curvature_proxy));
  const double slow_factor = std::min(
    1.0,
    config_.deviation_weight * norm_lat + config_.curve_weight * curve_proxy);
  const double speed_scale = 1.0 - config_.speed_gain_damping * slow_factor;

  double speed_cmd =
    clamp_value(config_.target_speed * speed_scale, config_.min_speed, config_.max_speed);
  speed_cmd = std::min(speed_cmd, std::max(0.0, speed_cap_mps));

  const double low_conf_thresh = clamp_value(config_.low_confidence_threshold, 0.0, 1.0);
  if (low_conf_thresh > 0.0 && confidence < low_conf_thresh) {
    speed_cmd = std::min(
      speed_cmd,
      clamp_value(config_.low_confidence_speed, config_.min_speed, config_.max_speed));
    if (motion_state.valid) {
      const double hold_factor = clamp_value(config_.low_confidence_yaw_hold, 0.0, 1.0);
      const double held_yaw = clamp_value(
        motion_state.last_yaw * hold_factor,
        -config_.max_yaw_rate,
        config_.max_yaw_rate);
      if (std::abs(yaw_cmd) < std::abs(held_yaw)) {
        const double deficit = clamp_value(
          (low_conf_thresh - confidence) / std::max(1e-3, low_conf_thresh),
          0.0,
          1.0);
        yaw_cmd = clamp_value(
          (1.0 - deficit) * yaw_cmd + deficit * held_yaw,
          -config_.max_yaw_rate,
          config_.max_yaw_rate);
      }
    }
  }

  return MotionCommand{speed_cmd, yaw_cmd, false};
}

AdaptiveSpeedResult MotionController::compute_adaptive_corridor_command(
  double target_y_m,
  double current_y_m,
  double heading_proxy,
  double curvature_proxy,
  double confidence,
  double speed_cap_mps,
  const CurveAssistState & curve_state,
  const MotionState & motion_state,
  const ClearPathInput & clear_input) const
{
  // Compute steering & legacy baseline speed identically to compute_corridor_command
  const MotionCommand base_cmd = compute_corridor_command(
    target_y_m,
    current_y_m,
    heading_proxy,
    curvature_proxy,
    confidence,
    speed_cap_mps,
    curve_state,
    motion_state);

  AdaptiveSpeedResult out{};
  out.command = base_cmd;
  out.clear_path_score = 0.0;
  out.blended_cruise_speed_mps = config_.target_speed;

  double eff_cap = std::max(0.0, speed_cap_mps);
  if (std::isfinite(clear_input.v_safe_cap_mps) && clear_input.v_safe_cap_mps >= 0.0) {
    eff_cap = std::min(eff_cap, clear_input.v_safe_cap_mps);
  }
  const double low_conf_thresh = clamp_value(config_.low_confidence_threshold, 0.0, 1.0);
  if (low_conf_thresh > 0.0 && confidence < low_conf_thresh) {
    eff_cap = std::min(
      eff_cap,
      clamp_value(config_.low_confidence_speed, config_.min_speed, config_.max_speed));
  }
  out.effective_speed_cap_mps = eff_cap;

  // Backward-compatible legacy mode when clear-path adaptive speed is disabled
  if (!config_.enable_clear_path_speed) {
    out.command.linear_speed = std::min(base_cmd.linear_speed, eff_cap);
    return out;
  }

  const double lateral_err_m = target_y_m - current_y_m;
  const double norm_lat = clamp_value(std::abs(lateral_err_m) / 0.215, 0.0, 1.0);
  const double curve_proxy = std::max(
    std::abs(curve_state.far_error - curve_state.near_error),
    std::abs(curvature_proxy));
  const double slow_factor = std::min(
    1.0,
    config_.deviation_weight * norm_lat + config_.curve_weight * curve_proxy);
  const double speed_scale = 1.0 - config_.speed_gain_damping * slow_factor;

  const double clear_score = compute_clear_path_score(clear_input);
  out.clear_path_score = clear_score;

  const double boost_target = std::max(config_.target_speed, config_.clear_straight_speed);
  const double blended_cruise =
    config_.target_speed + clear_score * (boost_target - config_.target_speed);
  out.blended_cruise_speed_mps = blended_cruise;

  const double base_desired =
    clamp_value(config_.target_speed * speed_scale, config_.min_speed, config_.max_speed);
  const double desired_speed =
    clamp_value(blended_cruise * speed_scale, config_.min_speed, config_.max_speed);

  double speed_cmd = std::min(desired_speed, eff_cap);

  // Asymmetric accel/decel rate limiting (smooth ramp up on clear straights, fast drop for hazards)
  const double dt = (std::isfinite(clear_input.dt_sec) && clear_input.dt_sec > 0.0) ?
    std::min(clear_input.dt_sec, 0.50) : 0.0;
  const bool has_prev =
    std::isfinite(clear_input.prev_speed_cmd_mps) && clear_input.prev_speed_cmd_mps >= 0.0;
  const double prev_v = has_prev ?
    clear_input.prev_speed_cmd_mps :
    std::min(base_desired, eff_cap);

  if (dt > 0.0) {
    const double max_up = prev_v + config_.max_accel_mps2 * dt;
    const double max_dn = prev_v - config_.max_decel_mps2 * dt;
    if (speed_cmd > max_up) {
      speed_cmd = max_up;
    } else if (speed_cmd < max_dn) {
      speed_cmd = max_dn;
    }
  } else if (!has_prev) {
    // First tracking frame with no elapsed dt yet: anchor smoothly at base cruise speed
    speed_cmd = std::min(base_desired, eff_cap);
  }

  // Optional mild proportional active deceleration when actual odom speed exceeds target ceiling
  // (never commands negative speed during normal line tracking)
  if (clear_input.odom_valid && std::isfinite(clear_input.odom_linear_x) &&
    config_.active_decel_kp > 0.0)
  {
    const double decel_ref = std::min(eff_cap, desired_speed);
    const double overspeed = clear_input.odom_linear_x - decel_ref;
    if (overspeed > 0.02) {
      speed_cmd = std::max(0.0, speed_cmd - config_.active_decel_kp * overspeed);
    }
  }

  const double upper_bound = std::min(config_.max_speed, eff_cap);
  speed_cmd = clamp_value(speed_cmd, 0.0, std::max(0.0, upper_bound));
  out.command.linear_speed = speed_cmd;
  return out;
}

AdaptiveSpeedResult MotionController::step_adaptive_corridor_command(
  double target_y_m,
  double current_y_m,
  double heading_proxy,
  double curvature_proxy,
  double confidence,
  double speed_cap_mps,
  const CurveAssistState & curve_state,
  const MotionState & motion_state,
  const ClearPathInput & clear_input,
  double now_sec)
{
  ClearPathInput local_in = clear_input;
  double dt = 0.0;
  if (has_step_time_ && std::isfinite(now_sec)) {
    const double raw_dt = now_sec - last_step_time_sec_;
    if (raw_dt > 0.0 && raw_dt <= 0.50) {
      dt = raw_dt;
    }
  }
  local_in.dt_sec = dt;
  local_in.prev_speed_cmd_mps = last_speed_cmd_mps_;

  const AdaptiveSpeedResult res = compute_adaptive_corridor_command(
    target_y_m,
    current_y_m,
    heading_proxy,
    curvature_proxy,
    confidence,
    speed_cap_mps,
    curve_state,
    motion_state,
    local_in);

  last_speed_cmd_mps_ = res.command.linear_speed;
  if (std::isfinite(now_sec)) {
    last_step_time_sec_ = now_sec;
    has_step_time_ = true;
  }
  return res;
}

MotionCommand MotionController::compute_lost_line_command(
  double since_detection_sec,
  const MotionState & motion_state) const
{
  double base_lost_speed = std::max(0.01, config_.line_lost_forward_speed);
  double slow_time = std::max(0.0, config_.line_lost_slow_time);
  double full_stop_time = std::max(slow_time, config_.line_lost_full_stop_time);
  const double slow_factor = clamp_value(config_.line_lost_slow_factor, 0.0, 1.0);

  if (config_.lost_line_safe_mode) {
    base_lost_speed = std::min(base_lost_speed, std::max(0.0, config_.lost_line_safe_speed));
    full_stop_time = std::min(full_stop_time, std::max(0.05, config_.lost_line_safe_stop_time));
    slow_time = std::min(slow_time, 0.5 * full_stop_time);
  }

  double lost_speed = base_lost_speed;
  if (since_detection_sec > full_stop_time) {
    lost_speed = 0.0;
  } else if (since_detection_sec > slow_time) {
    lost_speed = base_lost_speed * slow_factor;
  }

  const double yaw_hold_factor =
    clamp_value(config_.line_lost_yaw_hold_factor, 0.0, 1.0);
  const double yaw_rate =
    motion_state.valid ?
    clamp_value(
      motion_state.last_yaw * yaw_hold_factor,
      -config_.max_yaw_rate,
      config_.max_yaw_rate) :
    0.0;

  return MotionCommand{lost_speed, yaw_rate, false};
}

void MotionController::record_motion(
  bool yellow_stop_now,
  double linear_speed,
  double yaw_rate,
  MotionState & motion_state)
{
  if (!yellow_stop_now && linear_speed > 0.0) {
    motion_state.valid = true;
    motion_state.last_linear = linear_speed;
    motion_state.last_yaw = yaw_rate;
  }
}

}  // namespace line_follower
