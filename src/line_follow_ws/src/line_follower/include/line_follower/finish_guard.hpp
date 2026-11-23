#pragma once

#include "line_follower/finals_types.hpp"

namespace line_follower
{

struct FinishGuardConfig
{
  // Calibrated arc-length scale (field-calibrated odom multiplier)
  double odom_scale_factor{1.0};

  // Finish gating (decoupled from yellow_count == 4!)
  double finish_min_dist_m{8.5};
  double finish_force_dist_m{11.5};
  double min_cumulative_abs_yaw_rad{0.0};

  // Two-stage zero-touch yellow line stopping law
  double pre_brake_dist_m{0.70};
  double creep_speed_mps{0.24};
  double brake_decel_mps2{1.8};
  double control_latency_s{0.10};
  double bumper_margin_m{0.08};
  double min_stop_trigger_dist_m{0.10};
  double max_stop_trigger_dist_m{0.55};

  // Distance traveled after a yellow stop before re-arming (prevents double-counting)
  double rearm_travel_dist_m{0.35};
};

// ============================================================
// FinishGuard (pure C++17, zero OpenCV/ROS2 deps)
//
// 1. Tracks monotonic calibrated cumulative arc-length s_total_m
//    (NEVER reset by random straight yellow stop lines!) and
//    cumulative |dyaw| + fork-passed landmark flag.
// 2. Distinguishes true finish yellow line from random straight
//    yellow stop lines using distance + visual dead-end-ahead.
// 3. Implements two-stage zero-touch stopping law based on
//    metric bumper-to-line distance d_bumper_m.
// ============================================================
class FinishGuard
{
public:
  FinishGuard() = default;
  explicit FinishGuard(const FinishGuardConfig & config);

  void set_config(const FinishGuardConfig & config);
  const FinishGuardConfig & config() const;

  void reset();

  void update_motion(double ds_raw_m, double dyaw_rad);
  void mark_fork_passed();
  void mark_yellow_stop_triggered();

  bool is_yellow_rearmed() const;
  bool is_finish_armed() const;
  bool evaluate_is_finish(bool visual_dead_end_ahead) const;

  double compute_stop_trigger_dist(double current_speed_mps) const;

  YellowStopAdvice evaluate_yellow_stop(
    bool yellow_detected,
    bool has_metric_distance,
    double d_bumper_m,
    double current_speed_mps,
    bool visual_dead_end_ahead) const;

  double total_arc_m() const {return s_total_m_;}
  double cumulative_abs_yaw_rad() const {return cum_abs_yaw_rad_;}
  double dist_since_last_yellow_m() const {return dist_since_last_yellow_m_;}
  bool fork_passed() const {return fork_passed_;}

private:
  FinishGuardConfig config_{};
  double s_total_m_{0.0};
  double cum_abs_yaw_rad_{0.0};
  double dist_since_last_yellow_m_{1e6};
  bool fork_passed_{false};
};

}  // namespace line_follower
