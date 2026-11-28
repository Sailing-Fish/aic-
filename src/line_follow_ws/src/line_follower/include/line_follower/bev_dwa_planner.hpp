#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "line_follower/finals_types.hpp"

namespace line_follower
{

// ============================================================================
// Pure C++17 First-Principles BEV Occupancy + Topology Compass + Ackermann DWA
// + Real-Car Execution Layer (Zero OpenCV / Zero ROS2 dependencies)
//
// Grid coordinate convention (bumper-centered, ROS REP-103):
//   d_m in [kBevDMin, kBevDMax) = [-0.35m, +1.25m], +d = forward ahead of bumper
//   y_m in [kBevYMin, kBevYMax) = [-0.80m, +0.80m], +y = LEFT, -y = RIGHT
//   Resolution = 0.01 m/cell -> 160 x 160 cells (25,600 bytes)
//     row = floor((d_m - kBevDMin) / kBevRes)   (0 = rear -0.35m, 159 = +1.24m)
//     col = floor((kBevYMax - y_m) / kBevRes)   (0 = left +0.80m, 159 = right -0.79m)
// ============================================================================

constexpr int kBevRows = 160;
constexpr int kBevCols = 160;
constexpr int kBevCells = kBevRows * kBevCols;
constexpr double kBevRes = 0.01;
constexpr double kBevDMin = -0.35;
constexpr double kBevDMax = kBevDMin + kBevRows * kBevRes;  // +1.25 m
constexpr double kBevYMin = -0.80;
constexpr double kBevYMax = kBevYMin + kBevCols * kBevRes;  // +0.80 m

enum BevCellType : uint8_t
{
  BEV_UNKNOWN = 0,
  BEV_FREE = 1,
  BEV_WALL_BLUE = 2,
  BEV_OBS_CONE = 3,
  BEV_OBS_NO_ENTRY = 4
};

enum class TopologyPhase
{
  CRUISE_START = 0,
  FORK_RIGHT_ENTER,
  FORK_RIGHT_STEM,
  FORK_T_RIGHT_TURN,
  FORK_LEFT_UTURN,
  FORK_S_CURVE,
  RB1_APPROACH,
  RB1_ACTIVE,
  RB_UTURN,
  RB2_APPROACH,
  RB2_ACTIVE,
  RB2_EXIT
};

inline const char * topology_phase_to_string(TopologyPhase phase)
{
  switch (phase) {
    case TopologyPhase::CRUISE_START: return "cruise_start";
    case TopologyPhase::FORK_RIGHT_ENTER: return "fork_R_enter";
    case TopologyPhase::FORK_RIGHT_STEM: return "fork_R_stem";
    case TopologyPhase::FORK_T_RIGHT_TURN: return "fork_T_right";
    case TopologyPhase::FORK_LEFT_UTURN: return "fork_L_uturn";
    case TopologyPhase::FORK_S_CURVE: return "fork_S_curve";
    case TopologyPhase::RB1_APPROACH: return "rb1_approach";
    case TopologyPhase::RB1_ACTIVE: return "rb1_active";
    case TopologyPhase::RB_UTURN: return "rb_uturn";
    case TopologyPhase::RB2_APPROACH: return "rb2_approach";
    case TopologyPhase::RB2_ACTIVE: return "rb2_active";
    case TopologyPhase::RB2_EXIT: return "rb2_exit";
    default: return "unknown";
  }
}

struct BevDwaConfig
{
  // Track mode ("fork" | "roundabout")
  std::string track_mode{"fork"};

  // Real-measured Wheeltec mini_akm physical geometry & asymmetric steering limits
  double wheelbase_m{0.1395};
  double car_width_m{0.191};
  double car_length_m{0.280};
  double rear_axle_to_bumper_m{0.180};
  double kappa_max_left{2.38};   // Measured +18.0 deg -> R_min_left  ~= 0.42 m
  double kappa_max_right{2.80};  // Measured -21.4 deg -> R_min_right ~= 0.36 m
  double max_yaw_rate{1.60};
  double servo_slew_rps2{2.70};

  // Control latency compensation & IMU yaw-rate inner loop
  double control_latency_s{0.09};
  double imu_yaw_kp{0.35};
  double large_angle_boost_max{0.18};
  double steer_launch_v_floor{0.16};

  // Speed envelope (single clean physics pipeline)
  double cruise_speed_mps{0.44};
  double clear_straight_speed_mps{0.56};
  bool enable_clear_straight_boost{false};
  double min_speed_mps{0.0};
  double max_speed_mps{0.80};
  double max_lat_accel_mps2{0.42};
  double max_accel_mps2{0.90};
  double max_decel_mps2{2.00};
  double obstacle_speed_mps{0.26};
  double no_entry_speed_mps{0.22};
  double corner_approach_speed_mps{0.26};
  double creep_speed_mps{0.14};

  // BEV short-horizon EKF memory (protects rear inner wheel, erased instantly by white)
  double bev_memory_ttl_sec{0.55};

  // DWA rollout geometry & asymmetric penalty economics (+30s no_entry > +15s/s wall > +10s cone)
  int num_arc_samples{29};
  double rollout_horizon_m{0.85};
  double rollout_step_m{0.025};
  double wall_hard_margin_m{0.008};
  double cone_hard_margin_m{0.010};
  double no_entry_hard_margin_m{0.015};
  double wall_soft_margin_m{0.055};
  double cone_soft_margin_m{0.050};
  double no_entry_soft_margin_m{0.065};
  double wall_soft_weight{20.0};
  double cone_soft_weight{12.0};
  double no_entry_soft_weight{35.0};
  double safe_len_weight{4.2};
  double heading_weight{1.8};
  double wrong_way_penalty{4.5};
  double branch_bias_weight{1.2};
  double smooth_weight{0.12};
  double center_bias_weight{0.25};

  // Fork & straight bonus digit strategy
  int fork_outer_branch{-1};
  double fork_outer_time_penalty_s{3.0};
  int fork_min_votes{2};
  double digit_min_conf{0.50};
  double digit_bonus_weight{0.90};
  double digit_repel_weight{2.50};
  bool allow_dual_digit_cover{false};
  double max_digit_lateral_shift_m{0.12};

  // Finish gate (zero yellow-count dependence; immune to random yellow lines & +/-25% odom drift)
  double finish_min_cum_yaw_rad{2.2};
  double finish_force_dist_m{11.5};
  double yellow_rearm_dist_m{0.25};

  // Anti-stall Level-3 escape
  bool enable_anti_stall_escape{true};
  double stall_detect_sec{1.5};
  double escape_reverse_speed_mps{0.18};
  double escape_reverse_dur_sec{0.50};
  int max_escape_retries{2};

  // Pre-flight homography self-check bounds
  double nominal_lane_width_m{0.430};
  double preflight_width_tol_m{0.045};
  double preflight_center_tol_m{0.060};
};

struct PreFlightStatus
{
  bool checked{false};
  bool homography_ok{true};
  bool gyro_calibrated{false};
  double measured_lane_width_m{0.0};
  double measured_center_y_m{0.0};
  double measured_edge_angle_deg{0.0};
  double gyro_bias_rps{0.0};
  int standstill_samples{0};
  std::string warning_msg{};
};

struct BevArcCandidate
{
  double kappa{0.0};
  double safe_len_m{0.0};
  double min_wall_clr_m{0.0};
  double min_obs_clr_m{0.0};
  double score{-1e9};
  bool hard_feasible{false};
  // 白盒代价分解（面板 HUD 用；与 score 严格同源）
  double cost_safe_len{0.0};
  double cost_soft{0.0};
  double cost_heading{0.0};
  double cost_branch{0.0};
  double cost_digit{0.0};
  double cost_clearance{0.0};
  double cost_smooth{0.0};
};

struct BevDwaPlanResult
{
  double linear_speed_mps{0.0};
  double yaw_rate_rps{0.0};
  double chosen_kappa{0.0};
  double chosen_safe_len_m{0.0};
  double min_wall_clearance_m{0.0};
  double min_obs_clearance_m{0.0};
  double speed_cap_mps{0.0};

  bool has_feasible_arc{false};
  bool has_near_obstacle{false};
  bool has_no_entry_near{false};
  bool digit_target_active{false};
  int target_digit{0};
  bool escape_active{false};
  bool visual_dead_end_ahead{false};
  bool is_finish_line{false};

  TopologyPhase phase{TopologyPhase::CRUISE_START};
  int committed_branch{0};     // -1 left, 0 none, +1 right
  double rel_yaw_rad{0.0};
  double goal_yaw_rad{0.0};
  bool goal_yaw_active{false};
  double measured_lane_width_m{0.43};
  double straight_free_len_m{0.0};

  int fork_left_digit{0};
  int fork_left_votes{0};
  int fork_right_digit{0};
  int fork_right_votes{0};

  std::string mode_tag{"bev_cruise"};
  std::vector<BevArcCandidate> candidates{};

  // 选中弧快照（面板白盒 HUD：代价分解 + 可行弧统计）
  BevArcCandidate chosen_candidate{};
  int feasible_arc_count{0};
};

class BevDwaPlanner
{
public:
  BevDwaPlanner();
  explicit BevDwaPlanner(const BevDwaConfig & config);

  void set_config(const BevDwaConfig & config);
  const BevDwaConfig & config() const { return config_; }

  // Reset full lap state (preserves calibrated pre-flight gyro bias!)
  void reset(double now_sec, const VehiclePose2D & start_pose = VehiclePose2D{});

  // Reset pre-flight standstill accumulators (on node startup or stop->idle)
  void reset_preflight();

  // Coordinate conversions
  static inline bool metric_to_cell(double d_m, double y_m, int & row, int & col)
  {
    row = static_cast<int>(std::floor((d_m - kBevDMin) / kBevRes));
    col = static_cast<int>(std::floor((kBevYMax - y_m) / kBevRes));
    return row >= 0 && row < kBevRows && col >= 0 && col < kBevCols;
  }

  static inline void cell_to_metric(int row, int col, double & d_m, double & y_m)
  {
    d_m = kBevDMin + (static_cast<double>(row) + 0.5) * kBevRes;
    y_m = kBevYMax - (static_cast<double>(col) + 0.5) * kBevRes;
  }

  // Standstill 2s self-check (homography lane width + stationary gyro bias)
  void update_standstill_check(
    const std::array<uint8_t, kBevCells> & live_grid,
    const VehiclePose2D & pose);

  const PreFlightStatus & preflight_status() const { return preflight_; }

  // Accumulate fork digit votes (both during approach and stationary yellow stop!)
  void update_fork_votes(
    const std::vector<GroundDetection> & detections,
    bool is_stationary_at_yellow);

  void clear_fork_votes();

  // Finalize fork branch decision using net-value economics + single-side E=4.5 fallback
  int finalize_fork_decision();

  // Notify yellow stop lifecycle (dual visual+distance rearm)
  void notify_yellow_stop_triggered();
  void update_yellow_visual_state(bool yellow_visible, bool yellow_dist_valid, double d_yellow_m);
  bool is_yellow_rearmed() const;

  // Check if BEV shows blue dead-end beyond a detected yellow line
  bool check_bev_dead_end_ahead(double d_yellow_m) const;

  // Main 30 Hz step: fuses live BEV grid with short-window EKF memory, updates topology
  // compass, rolls out Ackermann DWA arcs, and computes execution-compensated (v, w).
  BevDwaPlanResult step(
    const std::array<uint8_t, kBevCells> & live_grid,
    const VehiclePose2D & pose,
    const std::vector<GroundDetection> & valid_digits,
    bool yellow_ahead,
    bool yellow_dist_valid,
    double yellow_d_bumper_m,
    double yellow_speed_cap_mps,
    bool is_holding_yellow_stop,
    double now_sec);

  // Access fused BEV grid (for live debug panel visualization)
  const std::array<uint8_t, kBevCells> & fused_grid() const { return fused_grid_; }
  double total_arc_m() const { return total_arc_m_; }
  double cum_abs_yaw_rad() const { return cum_abs_yaw_rad_; }
  double rel_yaw_rad() const { return rel_yaw_rad_; }
  int committed_branch() const { return committed_branch_; }
  TopologyPhase phase() const { return phase_; }

  static double wrap_angle_rad(double a);

private:
  void warp_and_fuse_bev(
    const std::array<uint8_t, kBevCells> & live_grid,
    const VehiclePose2D & pose,
    double dt);

  void filter_connected_free_space();
  void compute_distance_fields();
  void measure_bev_lane_geometry(
    double & out_width_m,
    double & out_center_y_m,
    double & out_visual_yaw_rad,
    bool & out_clean_straight,
    double & out_straight_free_len_m,
    bool & out_t_junction_right_open,
    bool & out_left_branch_open,
    bool & out_right_branch_open) const;

  void update_heading_and_topology(
    const VehiclePose2D & pose,
    double dt,
    double width_m,
    double visual_yaw_rad,
    bool clean_straight,
    double straight_free_len_m,
    bool t_junction_right_open,
    bool is_holding_yellow_stop);

  void evaluate_arc(
    double x0,
    double y0,
    double th0,
    double kappa,
    double goal_yaw,
    bool goal_yaw_active,
    bool forbid_reverse_yaw,
    double reverse_yaw,
    int branch_bias,
    const std::vector<GroundDetection> & straight_digits,
    bool allow_digit_bonus,
    BevArcCandidate & out_cand,
    int & out_hit_digit) const;

  double sample_dist_field(const std::array<float, kBevCells> & field, double d_m, double y_m) const;

  BevDwaConfig config_{};
  PreFlightStatus preflight_{};

  // Fused BEV memory & distance fields
  std::array<uint8_t, kBevCells> mem_grid_{};
  std::array<float, kBevCells> mem_age_{};
  std::array<uint8_t, kBevCells> fused_grid_{};
  std::array<float, kBevCells> dist_wall_m_{};
  std::array<float, kBevCells> dist_cone_m_{};
  std::array<float, kBevCells> dist_no_entry_m_{};

  // Pose & self-zeroing heading compass
  bool has_prev_pose_{false};
  VehiclePose2D prev_pose_{};
  double last_step_sec_{-1e9};
  double rel_yaw_rad_{0.0};
  double cum_abs_yaw_rad_{0.0};
  double total_arc_m_{0.0};

  // Yellow dual-condition rearm
  double dist_since_yellow_m_{1e6};
  int yellow_absent_frames_{10};
  bool yellow_visual_exited_{true};

  // Topology state & fork votes
  TopologyPhase phase_{TopologyPhase::CRUISE_START};
  int committed_branch_{0};
  double phase_arc_m_{0.0};
  double phase_abs_yaw_rad_{0.0};
  std::array<int, 9> left_digit_votes_{};
  std::array<int, 9> right_digit_votes_{};
  int best_left_digit_{0};
  int best_left_votes_{0};
  int best_right_digit_{0};
  int best_right_votes_{0};

  // Control & anti-stall state
  double prev_kappa_{0.0};
  double last_speed_cmd_{0.0};
  double stall_accum_sec_{0.0};
  bool escape_active_{false};
  double escape_elapsed_sec_{0.0};
  int escape_retries_used_{0};

  // Standstill calibration accumulators
  double gyro_bias_sum_{0.0};
  int gyro_bias_count_{0};
};

}  // namespace line_follower
