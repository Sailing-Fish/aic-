#pragma once

#include <array>
#include <string>
#include <vector>

#include "line_follower/finals_types.hpp"

namespace line_follower
{

struct CorridorPlannerConfig
{
  // Physical car & track dimensions
  double car_width_m{0.191};
  double car_length_m{0.280};
  double default_lane_width_m{0.430};

  // Lateral clearance margins
  double border_margin_m{0.025};
  double cone_margin_m{0.025};
  double no_entry_margin_m{0.030};

  // Two-zone obstacle horizon:
  //   - Far preview [eff_steer_horizon, obstacle_lookahead_m]: smooth pre-decel only (no curve lateral distortion)
  //   - Near steering [-obstacle_rear_keep_m, eff_steer_horizon]: immediate corridor carving (staggered gap split)
  double obstacle_lookahead_m{1.10};
  double obstacle_steer_horizon_m{0.65};
  double obstacle_curve_min_horizon_m{0.45};
  double obstacle_stagger_gap_m{0.25};
  double obstacle_rear_keep_m{0.28};

  // Speed caps by active hazard/mode
  double cruise_speed_mps{0.50};
  double obstacle_speed_mps{0.26};
  double no_entry_speed_mps{0.22};
  double blocked_creep_speed_mps{0.12};

  // Fork static-stop voting & DAG branch cost
  std::string track_mode{"fork"};       // "fork" or "roundabout"
  int fork_outer_branch{-1};            // -1 = left is outer, +1 = right is outer, 0 = symmetric
  double fork_outer_time_penalty_s{3.0};
  double fork_no_entry_penalty_s{30.0};
  double fork_cone_penalty_s{10.0};
  double bonus_score_weight_s{5.0};
  int fork_min_votes{3};
  double fork_commit_dist_m{1.10};

  // Straight dual-digit Bonus targeting
  double digit_min_conf{0.55};
  int digit_min_frames{2};
  double digit_lookahead_min_m{0.05};
  double digit_lookahead_max_m{1.15};
  double max_digit_lateral_shift_m{0.12};
  double digit_approach_speed_mps{0.30};
  bool allow_dual_digit_cover{false};
  double digit_sibling_avoid_offset_m{0.025};
  double digit_track_ttl_sec{0.45};
  double digit_obstacle_longitudinal_buffer_m{0.35};
};

struct ForkVoteSummary
{
  int left_digit{0};
  int right_digit{0};
  int left_votes{0};
  int right_votes{0};
  bool left_has_no_entry{false};
  bool right_has_no_entry{false};
  bool left_has_cone{false};
  bool right_has_cone{false};
  int chosen_branch{0};  // -1 left, +1 right, 0 undecided
};

// ============================================================
// CorridorPlanner (pure C++17, zero OpenCV/ROS2 deps)
//
// Strict safety priority hierarchy:
//   1. Lane Boundary (+15s/s, >=3 wheels DNF)
//   2. Red No-Entry (+30s)
//   3. Red Cone (+10s)
//   4. Bonus Digit (-5*d)
//   5. Cruise Speed
// ============================================================
class CorridorPlanner
{
public:
  CorridorPlanner() = default;
  explicit CorridorPlanner(const CorridorPlannerConfig & config);

  void set_config(const CorridorPlannerConfig & config);
  const CorridorPlannerConfig & config() const;

  void reset();
  void reset_fork_votes();

  // Called on every frame while stationary at a fork yellow stop line
  void update_fork_static_votes(
    const std::vector<GroundDetection> & detections,
    double now_sec);

  // Computes DAG-optimal branch (-1 left, +1 right, 0 none) and commits it
  int finalize_fork_decision(double current_arc_s_m);
  void clear_fork_commit();
  int committed_branch(double current_arc_s_m) const;
  ForkVoteSummary fork_summary() const;

  // Main per-frame lateral corridor & speed planner
  CorridorPlan plan(
    double y_lane_left_m,
    double y_lane_right_m,
    double current_y_m,
    const std::vector<LocalObstacle> & obstacles,
    const std::vector<GroundDetection> & raw_detections,
    bool is_stopped_at_yellow,
    double current_arc_s_m,
    double now_sec,
    double curve_level = 0.0);

  // Pure helper exposed for unit testing
  static std::vector<LateralInterval> subtract_interval(
    const std::vector<LateralInterval> & intervals,
    double ex_min,
    double ex_max);

private:
  struct DigitTrack
  {
    int digit{0};
    int track_id{-1};
    int frames{0};
    double mean_conf{0.0};
    double d_bumper_m{0.0};
    double y_lat_m{0.0};
    double last_seen_sec{-1e9};
  };

  void update_straight_digit_tracks(
    const std::vector<GroundDetection> & detections,
    double now_sec);

  CorridorPlannerConfig config_{};

  // Fork voting accumulators (index 1..8)
  std::array<int, 9> left_digit_votes_{};
  std::array<double, 9> left_digit_conf_sum_{};
  std::array<int, 9> right_digit_votes_{};
  std::array<double, 9> right_digit_conf_sum_{};
  int left_no_entry_votes_{0};
  int right_no_entry_votes_{0};
  int left_cone_votes_{0};
  int right_cone_votes_{0};

  int committed_branch_{0};
  double commit_start_s_m_{-1e9};

  std::vector<DigitTrack> digit_tracks_{};
};

}  // namespace line_follower
