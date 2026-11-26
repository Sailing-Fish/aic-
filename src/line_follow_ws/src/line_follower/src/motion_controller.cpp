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
}

const MotionControllerConfig & MotionController::config() const
{
  return config_;
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
