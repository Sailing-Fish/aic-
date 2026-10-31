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

MotionCommand MotionController::compute_tracking_command(
  double error,
  const CurveAssistState & curve_state) const
{
  const double yaw_cmd =
    clamp_value(-(config_.kp * error), -config_.max_yaw_rate, config_.max_yaw_rate);

  const double composite_err = std::abs(error);
  const double curve_proxy =
    std::abs(curve_state.far_error - curve_state.near_error);
  const double slow_factor = std::min(
    1.0,
    config_.deviation_weight * composite_err + config_.curve_weight * curve_proxy);
  const double speed_scale = 1.0 - config_.speed_gain_damping * slow_factor;
  double speed_cmd =
    clamp_value(config_.target_speed * speed_scale, config_.min_speed, config_.max_speed);

  if (config_.enable_continuous_curve_assist) {
    const double reverse_curve_speed = std::max(0.0, config_.reverse_curve_speed);
    const double sharp_curve_threshold = std::max(0.0, config_.sharp_curve_threshold);
    const double sharp_curve_speed = std::max(0.0, config_.sharp_curve_speed);

    if (curve_state.reverse_curve_active) {
      speed_cmd = std::min(speed_cmd, reverse_curve_speed);
    } else if (curve_state.curve_level >= sharp_curve_threshold) {
      speed_cmd = std::min(speed_cmd, sharp_curve_speed);
    }
  }

  return MotionCommand{speed_cmd, yaw_cmd, false};
}

MotionCommand MotionController::compute_lane_command(
  const LaneObservation & obs,
  const CurveAssistState & curve_state,
  const MotionState & motion_state) const
{
  const double effective_heading_gain =
    curve_state.reverse_curve_active ? 0.0 : config_.heading_gain;

  double yaw_cmd = clamp_value(
    -(config_.kp * obs.lateral_error + effective_heading_gain * obs.heading_proxy),
    -config_.max_yaw_rate,
    config_.max_yaw_rate);

  const double composite_err =
    std::abs(obs.lateral_error) + 0.35 * std::abs(effective_heading_gain * obs.heading_proxy);
  const double curve_proxy = std::max(
    std::abs(curve_state.far_error - curve_state.near_error),
    obs.curvature_proxy);
  const double slow_factor = std::min(
    1.0,
    config_.deviation_weight * composite_err + config_.curve_weight * curve_proxy);
  const double speed_scale = 1.0 - config_.speed_gain_damping * slow_factor;
  double speed_cmd =
    clamp_value(config_.target_speed * speed_scale, config_.min_speed, config_.max_speed);

  if (config_.enable_continuous_curve_assist) {
    const double reverse_curve_speed = std::max(0.0, config_.reverse_curve_speed);
    const double sharp_curve_threshold = std::max(0.0, config_.sharp_curve_threshold);
    const double sharp_curve_speed = std::max(0.0, config_.sharp_curve_speed);
    const double effective_curve_level = std::max(curve_state.curve_level, obs.curvature_proxy);

    if (curve_state.reverse_curve_active) {
      speed_cmd = std::min(speed_cmd, reverse_curve_speed);
    } else if (effective_curve_level >= sharp_curve_threshold) {
      speed_cmd = std::min(speed_cmd, sharp_curve_speed);
    }
  }

  const double low_conf_thresh = clamp_value(config_.low_confidence_threshold, 0.0, 1.0);
  if (low_conf_thresh > 0.0 && obs.confidence < low_conf_thresh) {
    const double low_conf_speed =
      clamp_value(config_.low_confidence_speed, config_.min_speed, config_.max_speed);
    speed_cmd = std::min(speed_cmd, low_conf_speed);

    if (motion_state.valid) {
      const double hold_factor = clamp_value(config_.low_confidence_yaw_hold, 0.0, 1.0);
      const double held_yaw = clamp_value(
        motion_state.last_yaw * hold_factor,
        -config_.max_yaw_rate,
        config_.max_yaw_rate);

      if (std::abs(yaw_cmd) < std::abs(held_yaw)) {
        const double both_wide =
          (obs.near_width_ratio >= 0.88 && obs.far_width_ratio >= 0.88) ? 1.0 : 0.0;
        const double deficit = clamp_value(
          (low_conf_thresh - obs.confidence) / std::max(1e-3, low_conf_thresh),
          0.0,
          1.0);
        const double blend = std::max(both_wide, deficit);
        yaw_cmd = clamp_value(
          (1.0 - blend) * yaw_cmd + blend * held_yaw,
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
  const double base_lost_speed = std::max(0.01, config_.line_lost_forward_speed);
  const double slow_time = std::max(0.0, config_.line_lost_slow_time);
  const double full_stop_time = std::max(slow_time, config_.line_lost_full_stop_time);
  const double slow_factor = clamp_value(config_.line_lost_slow_factor, 0.0, 1.0);

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

void MotionController::fuse_soft_red_avoidance(
  double soft_red_yaw_bias,
  const CurveAssistState & curve_state,
  RedAvoidState & red_state,
  double & linear_speed,
  double & yaw_rate) const
{
  const double red_near_ratio = clamp_value(config_.red_near_ratio, 0.0, 1.0);
  const double red_very_close_ratio = clamp_value(config_.red_very_close_ratio, 0.0, 1.0);
  const double red_soft_speed = std::max(0.0, config_.red_soft_speed);
  const double red_min_forward_speed = std::max(0.01, config_.red_min_forward_speed);

  const double bottom_ratio = clamp_value(red_state.bottom_ratio, 0.0, 1.0);
  const double risk_denominator = std::max(1e-3, red_very_close_ratio - red_near_ratio);
  const double risk = clamp_value(
    (bottom_ratio - red_near_ratio) / risk_denominator,
    0.0,
    1.0);

  red_state.risk = risk;

  const double repulsion_gain = std::max(0.0, config_.red_repulsion_gain);
  const double lane_keep_min_factor =
    clamp_value(config_.red_lane_keep_min_factor, 0.0, 1.0);

  const bool red_on_left = soft_red_yaw_bias < 0.0;

  double repulsion_strength =
    std::abs(soft_red_yaw_bias) * (1.0 + repulsion_gain * risk);

  if (curve_state.reverse_curve_active) {
    const double curve_bias_limit =
      clamp_value(config_.red_soft_curve_yaw_limit, 0.0, config_.max_yaw_rate);
    const double dynamic_limit =
      curve_bias_limit + risk * (config_.max_yaw_rate - curve_bias_limit);
    repulsion_strength = std::min(repulsion_strength, dynamic_limit);
  }

  const double lane_keep_factor = 1.0 - (1.0 - lane_keep_min_factor) * risk;

  if (red_on_left) {
    if (yaw_rate > 0.0) {
      yaw_rate *= lane_keep_factor;
    }
    yaw_rate -= repulsion_strength;
  } else {
    if (yaw_rate < 0.0) {
      yaw_rate *= lane_keep_factor;
    }
    yaw_rate += repulsion_strength;
  }

  yaw_rate = clamp_value(yaw_rate, -config_.max_yaw_rate, config_.max_yaw_rate);

  const double red_base_speed =
    red_state.is_red_line ?
    std::max(0.0, config_.red_line_speed) :
    red_soft_speed;

  const double safe_speed =
    std::max(
      red_min_forward_speed,
      red_base_speed * (1.0 - 0.60 * risk));

  linear_speed =
    std::max(
      red_min_forward_speed,
      std::min(linear_speed, safe_speed));

  red_state.very_close = bottom_ratio >= red_very_close_ratio;
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
