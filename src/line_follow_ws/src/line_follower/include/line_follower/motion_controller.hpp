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
  double kp{1.8};
  double speed_gain_damping{0.5};

  double deviation_weight{0.30};
  double curve_weight{0.90};

  bool enable_continuous_curve_assist{false};
  double reverse_curve_speed{0.16};
  double sharp_curve_threshold{0.50};
  double sharp_curve_speed{0.20};

  // 航向代理增益与低置信度限速/转向保持策略（默认关闭，保持 fork 兼容）
  double heading_gain{0.0};
  double low_confidence_threshold{0.0};
  double low_confidence_speed{0.26};
  double low_confidence_yaw_hold{0.75};

  double line_lost_forward_speed{0.06};
  double line_lost_yaw_hold_factor{0.85};
  double line_lost_slow_time{0.4};
  double line_lost_slow_factor{0.4};
  double line_lost_full_stop_time{1.2};

  double red_near_ratio{0.35};
  double red_very_close_ratio{0.72};
  double red_soft_speed{0.15};
  double red_line_speed{0.12};
  double red_min_forward_speed{0.06};
  double red_soft_curve_yaw_limit{0.15};
  double red_repulsion_gain{1.5};
  double red_lane_keep_min_factor{0.40};
};

class MotionController
{
public:
  MotionController();
  explicit MotionController(const MotionControllerConfig & config);

  void set_config(const MotionControllerConfig & config);
  const MotionControllerConfig & config() const;

  MotionCommand compute_tracking_command(
    double error,
    const CurveAssistState & curve_state) const;

  MotionCommand compute_lane_command(
    const LaneObservation & obs,
    const CurveAssistState & curve_state,
    const MotionState & motion_state) const;

  MotionCommand compute_lost_line_command(
    double since_detection_sec,
    const MotionState & motion_state) const;

  void fuse_soft_red_avoidance(
    double soft_red_yaw_bias,
    const CurveAssistState & curve_state,
    RedAvoidState & red_state,
    double & linear_speed,
    double & yaw_rate) const;

  static void record_motion(
    bool yellow_stop_now,
    double linear_speed,
    double yaw_rate,
    MotionState & motion_state);

private:
  MotionControllerConfig config_{};
};

using LaneController = MotionController;
using LaneControllerConfig = MotionControllerConfig;

}  // namespace line_follower
