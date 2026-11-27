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

  // Two-stage zero-touch yellow line stopping law + near-line ramp
  double pre_brake_dist_m{0.70};
  double creep_speed_mps{0.24};
  double yellow_final_ramp_dist_m{0.35};
  double yellow_final_creep_speed_mps{0.18};
  double brake_decel_mps2{1.8};
  double control_latency_s{0.10};
  double bumper_margin_m{0.08};
  double min_stop_trigger_dist_m{0.10};
  double max_stop_trigger_dist_m{0.55};

  // Static battery voltage feedforward + 1-param clamped friction EMA adaptation
  double voltage_decel_slope{0.0};
  double voltage_nominal_v{12.0};
  double voltage_timeout_sec{3.0};
  bool enable_brake_adaptation{true};
  double brake_adapt_ema_alpha{0.35};
  double friction_scale_min{0.65};
  double friction_scale_max{1.25};

  // Dual-layer closed-loop yellow braking (sqrt physics envelope + proportional damping)
  bool enable_sqrt_brake_envelope{true};
  bool enable_closed_loop_brake{true};
  double brake_damping_gain{0.85};
  double brake_zero_cutoff_mps{0.025};

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

  void update_motion(double ds_fwd_m, double dyaw_rad);
  void update_battery_voltage(double voltage_v, double now_sec);
  bool update_measured_yellow_stop(
    double v_entry_mps,
    double stop_travel_m,
    double now_sec = -1.0);
  void mark_fork_passed();
  void mark_yellow_stop_triggered();

  bool is_yellow_rearmed() const;
  bool is_finish_armed() const;
  bool evaluate_is_finish(bool visual_dead_end_ahead) const;

  double active_battery_voltage(double now_sec = -1.0) const;
  double effective_brake_decel_mps2(double now_sec = -1.0) const;
  double compute_stop_trigger_dist(double current_speed_mps, double now_sec = -1.0) const;
  double compute_closed_loop_brake_cmd(
    double odom_linear_x,
    double elapsed_s,
    double max_brake_speed,
    double max_time_s) const;
  double compute_safe_speed_ceiling(
    double preview_dist_m,
    double target_end_speed_mps = 0.0,
    double now_sec = -1.0) const;

  YellowStopAdvice evaluate_yellow_stop(
    bool yellow_detected,
    bool has_metric_distance,
    double d_bumper_m,
    double current_speed_mps,
    bool visual_dead_end_ahead,
    double now_sec = -1.0) const;

  double total_arc_m() const {return s_total_m_;}
  double cumulative_abs_yaw_rad() const {return cum_abs_yaw_rad_;}
  double dist_since_last_yellow_m() const {return dist_since_last_yellow_m_;}
  bool fork_passed() const {return fork_passed_;}
  double friction_scale() const {return friction_scale_;}
  double battery_voltage_v() const {return battery_voltage_v_;}
  bool battery_voltage_valid() const {return voltage_valid_;}
  int brake_adapt_updates() const {return brake_adapt_updates_;}

private:
  FinishGuardConfig config_{};
  double s_total_m_{0.0};
  double cum_abs_yaw_rad_{0.0};
  double dist_since_last_yellow_m_{1e6};
  bool fork_passed_{false};
  double battery_voltage_v_{12.0};
  double last_voltage_sec_{-1e9};
  bool voltage_valid_{false};
  double friction_scale_{1.0};
  int brake_adapt_updates_{0};
};

}  // namespace line_follower
