#include "line_follower/bev_dwa_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>

namespace line_follower
{

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr float kInfDist = 2.5f;

double clamp_d(double v, double lo, double hi)
{
  return std::max(lo, std::min(v, hi));
}

void compute_chamfer_dt(
  const std::array<uint8_t, kBevCells> & grid,
  uint8_t target_cell,
  std::array<float, kBevCells> & dist_out)
{
  constexpr float kOrtho = static_cast<float>(kBevRes);
  constexpr float kDiag = static_cast<float>(kBevRes * 1.41421356);

  for (int i = 0; i < kBevCells; ++i) {
    dist_out[static_cast<size_t>(i)] = (grid[static_cast<size_t>(i)] == target_cell) ? 0.0f : kInfDist;
  }

  // Forward pass (top-left to bottom-right)
  for (int r = 0; r < kBevRows; ++r) {
    const int row_off = r * kBevCols;
    for (int c = 0; c < kBevCols; ++c) {
      const size_t idx = static_cast<size_t>(row_off + c);
      float d = dist_out[idx];
      if (d == 0.0f) {
        continue;
      }
      if (r > 0) {
        d = std::min(d, dist_out[idx - static_cast<size_t>(kBevCols)] + kOrtho);
        if (c > 0) {
          d = std::min(d, dist_out[idx - static_cast<size_t>(kBevCols) - 1U] + kDiag);
        }
        if (c + 1 < kBevCols) {
          d = std::min(d, dist_out[idx - static_cast<size_t>(kBevCols) + 1U] + kDiag);
        }
      }
      if (c > 0) {
        d = std::min(d, dist_out[idx - 1U] + kOrtho);
      }
      dist_out[idx] = d;
    }
  }

  // Backward pass (bottom-right to top-left)
  for (int r = kBevRows - 1; r >= 0; --r) {
    const int row_off = r * kBevCols;
    for (int c = kBevCols - 1; c >= 0; --c) {
      const size_t idx = static_cast<size_t>(row_off + c);
      float d = dist_out[idx];
      if (d == 0.0f) {
        continue;
      }
      if (r + 1 < kBevRows) {
        d = std::min(d, dist_out[idx + static_cast<size_t>(kBevCols)] + kOrtho);
        if (c > 0) {
          d = std::min(d, dist_out[idx + static_cast<size_t>(kBevCols) - 1U] + kDiag);
        }
        if (c + 1 < kBevCols) {
          d = std::min(d, dist_out[idx + static_cast<size_t>(kBevCols) + 1U] + kDiag);
        }
      }
      if (c + 1 < kBevCols) {
        d = std::min(d, dist_out[idx + 1U] + kOrtho);
      }
      dist_out[idx] = d;
    }
  }
}

}  // namespace

BevDwaPlanner::BevDwaPlanner()
{
  set_config(BevDwaConfig{});
  reset(0.0);
}

BevDwaPlanner::BevDwaPlanner(const BevDwaConfig & config)
{
  set_config(config);
  reset(0.0);
}

void BevDwaPlanner::set_config(const BevDwaConfig & config)
{
  config_ = config;
  config_.wheelbase_m = std::max(0.05, config_.wheelbase_m);
  config_.car_width_m = std::max(0.08, config_.car_width_m);
  config_.car_length_m = std::max(0.10, config_.car_length_m);
  config_.kappa_max_left = std::max(0.5, config_.kappa_max_left);
  config_.kappa_max_right = std::max(0.5, config_.kappa_max_right);
  config_.max_yaw_rate = std::max(0.3, config_.max_yaw_rate);
  config_.num_arc_samples = std::clamp(config_.num_arc_samples, 9, 61);
  if (config_.num_arc_samples % 2 == 0) {
    config_.num_arc_samples += 1;
  }
  config_.rollout_horizon_m = std::clamp(config_.rollout_horizon_m, 0.40, 1.20);
  config_.rollout_step_m = std::clamp(config_.rollout_step_m, 0.015, 0.05);
  config_.bev_memory_ttl_sec = std::clamp(config_.bev_memory_ttl_sec, 0.10, 1.50);
  config_.cruise_speed_mps = std::clamp(config_.cruise_speed_mps, 0.10, config_.max_speed_mps);
  config_.clear_straight_speed_mps = std::clamp(
    config_.clear_straight_speed_mps, config_.cruise_speed_mps, config_.max_speed_mps);
  config_.max_lat_accel_mps2 = std::max(0.15, config_.max_lat_accel_mps2);
  config_.max_accel_mps2 = std::max(0.15, config_.max_accel_mps2);
  config_.max_decel_mps2 = std::max(0.30, config_.max_decel_mps2);
  config_.finish_min_cum_yaw_rad = std::max(0.0, config_.finish_min_cum_yaw_rad);
  config_.yellow_rearm_dist_m = std::max(0.10, config_.yellow_rearm_dist_m);
  config_.lookahead_min_m = std::clamp(config_.lookahead_min_m, 0.16, 0.55);
  config_.lookahead_time_s = std::clamp(config_.lookahead_time_s, 0.0, 1.50);
  config_.curv_ff_gain = std::clamp(config_.curv_ff_gain, 0.0, 1.50);
  config_.stanley_k_y = std::clamp(config_.stanley_k_y, 0.0, 4.0);
  config_.border_margin_m = std::clamp(config_.border_margin_m, 0.0, 0.08);
  config_.cone_margin_m = std::clamp(config_.cone_margin_m, 0.0, 0.10);
  config_.no_entry_margin_m = std::clamp(config_.no_entry_margin_m, 0.0, 0.12);
}

void BevDwaPlanner::reset(double now_sec, const VehiclePose2D & start_pose)
{
  mem_grid_.fill(BEV_UNKNOWN);
  mem_age_.fill(10.0f);
  fused_grid_.fill(BEV_UNKNOWN);
  dist_wall_m_.fill(kInfDist);
  dist_free_m_.fill(kInfDist);
  dist_cone_m_.fill(kInfDist);
  dist_no_entry_m_.fill(kInfDist);
  road_progress_m_.fill(0.0f);
  blocked_branch_grid_.fill(0);

  locked_digit_val_ = 0;
  locked_digit_y_ = 0.0;
  locked_digit_d_ = 0.0;
  locked_digit_hits_ = 0;
  locked_digit_misses_ = 0;
  cand_digit_val_ = 0;
  cand_digit_streak_ = 0;

  has_prev_pose_ = start_pose.valid;
  prev_pose_ = start_pose;
  last_step_sec_ = now_sec;

  rel_yaw_rad_ = 0.0;
  cum_abs_yaw_rad_ = 0.0;
  total_arc_m_ = 0.0;

  dist_since_yellow_m_ = 1e6;
  yellow_absent_frames_ = 10;
  yellow_visual_exited_ = true;
  yellow_stop_begin_sec_ = -1e9;

  phase_ = (config_.track_mode == "roundabout") ?
    TopologyPhase::RB1_APPROACH : TopologyPhase::CRUISE_START;
  committed_branch_ = 0;
  phase_arc_m_ = 0.0;
  phase_abs_yaw_rad_ = 0.0;

  clear_fork_votes();

  prev_kappa_ = 0.0;
  last_speed_cmd_ = 0.0;
  stall_accum_sec_ = 0.0;
  escape_active_ = false;
  escape_elapsed_sec_ = 0.0;
  escape_retries_used_ = 0;

  prev_obs_pass_side_ = 0;
  prev_obs_d_ = 0.0;
  prev_obs_y_ = 0.0;
  prev_obs_miss_frames_ = 10;
}

void BevDwaPlanner::reset_preflight()
{
  preflight_ = PreFlightStatus{};
  gyro_bias_sum_ = 0.0;
  gyro_bias_count_ = 0;
}

double BevDwaPlanner::wrap_angle_rad(double a)
{
  while (a > kPi) {
    a -= 2.0 * kPi;
  }
  while (a < -kPi) {
    a += 2.0 * kPi;
  }
  return a;
}

void BevDwaPlanner::clear_fork_votes()
{
  left_digit_votes_.fill(0.0);
  right_digit_votes_.fill(0.0);
  left_digit_hits_.fill(0);
  right_digit_hits_.fill(0);
  best_left_digit_ = 0;
  best_left_votes_ = 0;
  best_right_digit_ = 0;
  best_right_votes_ = 0;
}

void BevDwaPlanner::update_standstill_check(
  const std::array<uint8_t, kBevCells> & live_grid,
  const VehiclePose2D & pose)
{
  // 1) Stationary IMU gyro bias calibration
  if (pose.valid && std::abs(pose.linear_v) < 0.015 && std::isfinite(pose.angular_w)) {
    if (std::abs(pose.angular_w) < 0.25 && gyro_bias_count_ < 120) {
      gyro_bias_sum_ += pose.angular_w;
      gyro_bias_count_++;
      preflight_.standstill_samples = gyro_bias_count_;
      preflight_.gyro_bias_rps = clamp_d(
        gyro_bias_sum_ / static_cast<double>(gyro_bias_count_), -0.12, 0.12);
      preflight_.gyro_calibrated = (gyro_bias_count_ >= 15);
    }
  }

  // 2) Live BEV homography sanity check on the starting straight (d in [0.20, 0.45]m)
  double width_sum = 0.0;
  double center_sum = 0.0;
  int valid_rows = 0;
  double near_y = 0.0;
  double far_y = 0.0;
  int near_n = 0;
  int far_n = 0;

  for (int r = 0; r < kBevRows; ++r) {
    double d_m = 0.0;
    double y_tmp = 0.0;
    cell_to_metric(r, 0, d_m, y_tmp);
    if (d_m < 0.18 || d_m > 0.50) {
      continue;
    }
    int c_left = -1;
    int c_right = -1;
    // Find widest contiguous FREE run crossing center
    int best_len = 0;
    int cur_start = -1;
    for (int c = 0; c <= kBevCols; ++c) {
      const bool is_free = (c < kBevCols) &&
        (live_grid[static_cast<size_t>(r * kBevCols + c)] == BEV_FREE);
      if (is_free) {
        if (cur_start < 0) {
          cur_start = c;
        }
      } else if (cur_start >= 0) {
        const int len = c - cur_start;
        if (len > best_len) {
          best_len = len;
          c_left = cur_start;
          c_right = c - 1;
        }
        cur_start = -1;
      }
    }
    if (best_len >= 20 && c_left > 1 && c_right < kBevCols - 2) {
      double dl = 0.0, yl = 0.0, dr = 0.0, yr = 0.0;
      cell_to_metric(r, c_left, dl, yl);
      cell_to_metric(r, c_right, dr, yr);
      const double w = yl - yr;
      const double yc = 0.5 * (yl + yr);
      width_sum += w;
      center_sum += yc;
      valid_rows++;
      if (d_m < 0.30) {
        near_y += yc;
        near_n++;
      } else {
        far_y += yc;
        far_n++;
      }
    }
  }

  if (valid_rows >= 8) {
    preflight_.checked = true;
    preflight_.measured_lane_width_m = width_sum / static_cast<double>(valid_rows);
    preflight_.measured_center_y_m = center_sum / static_cast<double>(valid_rows);
    double angle_deg = 0.0;
    if (near_n >= 3 && far_n >= 3) {
      const double dy = (far_y / static_cast<double>(far_n)) - (near_y / static_cast<double>(near_n));
      angle_deg = std::atan2(dy, 0.18) * (180.0 / kPi);
    }
    preflight_.measured_edge_angle_deg = angle_deg;

    const double w_err = std::abs(preflight_.measured_lane_width_m - config_.nominal_lane_width_m);
    const bool w_ok = (w_err <= config_.preflight_width_tol_m);
    const bool c_ok = (std::abs(preflight_.measured_center_y_m) <= config_.preflight_center_tol_m);
    const bool a_ok = (std::abs(angle_deg) <= 10.0);
    preflight_.homography_ok = w_ok && a_ok;

    if (!w_ok) {
      char buf[128];
      std::snprintf(
        buf, sizeof(buf), "BEV width=%.3fm (expect %.3f+/-%.3fm): check cam pitch/H",
        preflight_.measured_lane_width_m, config_.nominal_lane_width_m, config_.preflight_width_tol_m);
      preflight_.warning_msg = buf;
    } else if (!a_ok) {
      char buf[128];
      std::snprintf(
        buf, sizeof(buf), "BEV road yaw=%.1fdeg: car/cam misaligned", angle_deg);
      preflight_.warning_msg = buf;
    } else if (!c_ok) {
      char buf[128];
      std::snprintf(
        buf, sizeof(buf), "BEV center=%+.3fm (off-center start)", preflight_.measured_center_y_m);
      preflight_.warning_msg = buf;
    } else {
      preflight_.warning_msg.clear();
    }
  }
}

void BevDwaPlanner::update_fork_votes(
  const std::vector<GroundDetection> & detections,
  bool is_stationary_at_yellow)
{
  std::vector<const GroundDetection *> valid;
  valid.reserve(detections.size());
  double y_min = 1e9;
  double y_max = -1e9;

  for (const auto & gd : detections) {
    if (gd.type != SemanticObjectType::BONUS_DIGIT) {
      continue;
    }
    if (gd.digit < 1 || gd.digit > 8 || gd.confidence < config_.digit_min_conf) {
      continue;
    }
    if (gd.d_bumper_m < 0.12 || gd.d_bumper_m > 1.25) {
      continue;
    }
    // Stationary stop bonus weights must come strictly from post-stop frames!
    if (is_stationary_at_yellow && yellow_stop_begin_sec_ > 0.0 &&
      gd.stamp_sec > 0.0 && gd.stamp_sec < yellow_stop_begin_sec_)
    {
      continue;
    }
    valid.push_back(&gd);
    y_min = std::min(y_min, gd.y_lat_m);
    y_max = std::max(y_max, gd.y_lat_m);
  }

  const bool use_midpoint_split = (valid.size() >= 2U) && ((y_max - y_min) >= 0.08);
  const double y_split = 0.5 * (y_min + y_max);
  const int hit_inc = is_stationary_at_yellow ? 2 : 1;

  for (const GroundDetection * gd : valid) {
    const double w = static_cast<double>(hit_inc) * clamp_d(gd->confidence, 0.20, 1.0);
    if (use_midpoint_split) {
      if (gd->y_lat_m >= y_split) {
        left_digit_votes_[static_cast<size_t>(gd->digit)] += w;
        left_digit_hits_[static_cast<size_t>(gd->digit)] += hit_inc;
      } else {
        right_digit_votes_[static_cast<size_t>(gd->digit)] += w;
        right_digit_hits_[static_cast<size_t>(gd->digit)] += hit_inc;
      }
    } else {
      if (gd->y_lat_m > 0.025) {
        left_digit_votes_[static_cast<size_t>(gd->digit)] += w;
        left_digit_hits_[static_cast<size_t>(gd->digit)] += hit_inc;
      } else if (gd->y_lat_m < -0.025) {
        right_digit_votes_[static_cast<size_t>(gd->digit)] += w;
        right_digit_hits_[static_cast<size_t>(gd->digit)] += hit_inc;
      }
    }
  }

  best_left_digit_ = 0;
  best_left_votes_ = 0;
  best_right_digit_ = 0;
  best_right_votes_ = 0;
  double best_lw = 0.0;
  double best_rw = 0.0;
  // Scan descending 8..1 so ties strictly favor the LARGER bonus digit!
  for (int d = 8; d >= 1; --d) {
    const double lw = left_digit_votes_[static_cast<size_t>(d)];
    if (lw > best_lw + 1e-6) {
      best_lw = lw;
      best_left_digit_ = d;
      best_left_votes_ = left_digit_hits_[static_cast<size_t>(d)];
    }
    const double rw = right_digit_votes_[static_cast<size_t>(d)];
    if (rw > best_rw + 1e-6) {
      best_rw = rw;
      best_right_digit_ = d;
      best_right_votes_ = right_digit_hits_[static_cast<size_t>(d)];
    }
  }
}

int BevDwaPlanner::finalize_fork_decision()
{
  const bool left_ok = (best_left_digit_ >= 1 && best_left_votes_ >= config_.fork_min_votes);
  const bool right_ok = (best_right_digit_ >= 1 && best_right_votes_ >= config_.fork_min_votes);

  if (!left_ok && !right_ok) {
    // No fork digits observed at this stop (e.g. random straight yellow line!)
    return 0;
  }

  // Single-side missing-digit fallback uses E[missing] = 4.5
  const double eff_left = left_ok ? static_cast<double>(best_left_digit_) : 4.5;
  const double eff_right = right_ok ? static_cast<double>(best_right_digit_) : 4.5;

  const double left_pen = (config_.fork_outer_branch < 0) ? config_.fork_outer_time_penalty_s : 0.0;
  const double right_pen = (config_.fork_outer_branch > 0) ? config_.fork_outer_time_penalty_s : 0.0;

  const double net_left = 5.0 * eff_left - left_pen;
  const double net_right = 5.0 * eff_right - right_pen;

  const int chosen = (net_left > net_right + 1e-6) ? -1 : +1;
  committed_branch_ = chosen;

  // Transition topology phase on fork commitment (zero yellow-count dependence!)
  if (config_.track_mode == "fork") {
    if (phase_ == TopologyPhase::CRUISE_START) {
      phase_ = (chosen > 0) ? TopologyPhase::FORK_RIGHT_ENTER : TopologyPhase::FORK_LEFT_UTURN;
      phase_arc_m_ = 0.0;
      phase_abs_yaw_rad_ = 0.0;
    }
  } else {
    // Roundabout mode: distinguish RB1 vs RB2 strictly by current phase (never jump out of RB_UTURN!)
    if (phase_ == TopologyPhase::RB1_APPROACH) {
      phase_ = TopologyPhase::RB1_ACTIVE;
      phase_arc_m_ = 0.0;
      phase_abs_yaw_rad_ = 0.0;
    } else if (phase_ == TopologyPhase::RB2_APPROACH) {
      phase_ = TopologyPhase::RB2_ACTIVE;
      phase_arc_m_ = 0.0;
      phase_abs_yaw_rad_ = 0.0;
    }
  }

  return chosen;
}

void BevDwaPlanner::notify_yellow_stop_triggered(double now_sec)
{
  dist_since_yellow_m_ = 0.0;
  yellow_absent_frames_ = 0;
  yellow_visual_exited_ = false;
  yellow_stop_begin_sec_ = now_sec;
  last_speed_cmd_ = 0.0;
  stall_accum_sec_ = 0.0;
  escape_active_ = false;
}

void BevDwaPlanner::update_yellow_visual_state(
  bool yellow_visible,
  bool yellow_dist_valid,
  double d_yellow_m)
{
  if (!yellow_visible) {
    yellow_absent_frames_++;
    if (yellow_absent_frames_ >= 3) {
      yellow_visual_exited_ = true;
    }
  } else {
    yellow_absent_frames_ = 0;
    // A brand-new yellow line appearing far ahead (> 0.35m) after moving >= 0.25m also counts as exited
    if (yellow_dist_valid && d_yellow_m > 0.35 && dist_since_yellow_m_ >= config_.yellow_rearm_dist_m) {
      yellow_visual_exited_ = true;
    }
  }
}

bool BevDwaPlanner::is_yellow_rearmed() const
{
  return (dist_since_yellow_m_ >= config_.yellow_rearm_dist_m) && yellow_visual_exited_;
}

bool BevDwaPlanner::check_bev_dead_end_ahead(double d_yellow_m) const
{
  // Only confirm visual dead-end when yellow stop line is reasonably close (<= 0.38m)
  // so mid-track corner yellow lines viewed from 0.55m never trigger premature finish!
  if (d_yellow_m > 0.38) {
    return false;
  }
  const double d0 = std::max(0.05, d_yellow_m + 0.14);
  const double d1 = std::min(kBevDMax - 0.02, d_yellow_m + 0.46);
  if (d1 <= d0 + 0.05) {
    return false;
  }
  // Wide continuation check: if white drivable road continues left, right, or ahead beyond
  // the yellow strip (e.g., curved entrance or T-bar), it is NOT a finish dead-end!
  int wide_free_cnt = 0;
  for (double d = d0 + 0.02; d <= d1; d += 0.02) {
    for (double y = -0.48; y <= 0.48; y += 0.02) {
      int r = 0, c = 0;
      if (!metric_to_cell(d, y, r, c)) {
        continue;
      }
      if (fused_grid_[static_cast<size_t>(r * kBevCols + c)] == BEV_FREE) {
        wide_free_cnt++;
      }
    }
  }
  if (wide_free_cnt >= 25) {
    return false;
  }

  int total = 0;
  int wall_cnt = 0;
  int free_cnt = 0;
  for (double d = d0; d <= d1; d += 0.02) {
    for (double y = -0.18; y <= 0.18; y += 0.02) {
      int r = 0, c = 0;
      if (!metric_to_cell(d, y, r, c)) {
        continue;
      }
      const uint8_t cell = fused_grid_[static_cast<size_t>(r * kBevCols + c)];
      if (cell == BEV_UNKNOWN) {
        continue;
      }
      total++;
      if (cell == BEV_WALL_BLUE) {
        wall_cnt++;
      } else if (cell == BEV_FREE) {
        free_cnt++;
      }
    }
  }
  if (total < 15) {
    return false;
  }
  const double wall_ratio = static_cast<double>(wall_cnt) / static_cast<double>(total);
  const double free_ratio = static_cast<double>(free_cnt) / static_cast<double>(total);
  return (wall_ratio >= 0.60 && free_ratio <= 0.25);
}

void BevDwaPlanner::warp_and_fuse_bev(
  const std::array<uint8_t, kBevCells> & live_grid,
  const VehiclePose2D & pose,
  double dt)
{
  std::array<uint8_t, kBevCells> warped_grid{};
  std::array<float, kBevCells> warped_age{};
  warped_grid.fill(BEV_UNKNOWN);
  warped_age.fill(10.0f);

  const float dt_f = static_cast<float>(clamp_d(dt, 0.0, 0.25));
  const float ttl_f = static_cast<float>(config_.bev_memory_ttl_sec);

  if (has_prev_pose_ && pose.valid && dt > 0.0 && dt < 0.35) {
    // Transform from current bumper frame to previous bumper frame
    const double dth = wrap_angle_rad(pose.yaw - prev_pose_.yaw);
    const double dx_w = pose.x - prev_pose_.x;
    const double dy_w = pose.y - prev_pose_.y;
    const double cp = std::cos(prev_pose_.yaw);
    const double sp = std::sin(prev_pose_.yaw);
    // Motion of rear axle in previous vehicle frame
    const double dx_prev = dx_w * cp + dy_w * sp;
    const double dy_prev = -dx_w * sp + dy_w * cp;
    const double cd = std::cos(dth);
    const double sd = std::sin(dth);
    const double r2b = config_.rear_axle_to_bumper_m;

    // Keep short-window rolling memory primarily for RED OBSTACLES in the rear blind spot
    // (d in [-0.28, +0.26]m) so rear wheels never cut back into passed cones/no-entry, while
    // keeping blue wall memory strictly behind the live camera FOV (d <= +0.02m, max 0.25s)
    // so stale blue walls never trap the front bumper.
    const float wall_ttl_f = std::min(ttl_f, 0.25f);
    for (int r = 0; r < kBevRows; ++r) {
      const double d_cur = kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes;
      if (d_cur < -0.28 || d_cur > 0.26) {
        continue;
      }
      const double x_axle_cur = d_cur + r2b;
      for (int c = 0; c < kBevCols; ++c) {
        const double y_cur = kBevYMax - (static_cast<double>(c) + 0.5) * kBevRes;
        // Point in previous rear-axle frame
        const double x_axle_prev = dx_prev + cd * x_axle_cur - sd * y_cur;
        const double y_prev = dy_prev + sd * x_axle_cur + cd * y_cur;
        const double d_prev = x_axle_prev - r2b;

        int pr = 0, pc = 0;
        if (metric_to_cell(d_prev, y_prev, pr, pc)) {
          const size_t pidx = static_cast<size_t>(pr * kBevCols + pc);
          const uint8_t prev_cell = mem_grid_[pidx];
          const float new_age = mem_age_[pidx] + dt_f;
          const bool is_red_obs =
            (prev_cell == BEV_OBS_CONE || prev_cell == BEV_OBS_NO_ENTRY);
          if (is_red_obs && new_age <= ttl_f) {
            const size_t idx = static_cast<size_t>(r * kBevCols + c);
            warped_grid[idx] = prev_cell;
            warped_age[idx] = new_age;
          } else if (!is_red_obs && prev_cell != BEV_UNKNOWN &&
            d_cur <= 0.02 && new_age <= wall_ttl_f)
          {
            const size_t idx = static_cast<size_t>(r * kBevCols + c);
            warped_grid[idx] = prev_cell;
            warped_age[idx] = new_age;
          }
        }
      }
    }
  }

  // Fuse current live camera observation (instant free-space erasure!)
  for (int i = 0; i < kBevCells; ++i) {
    const size_t idx = static_cast<size_t>(i);
    const uint8_t obs = live_grid[idx];
    if (obs != BEV_UNKNOWN) {
      warped_grid[idx] = obs;
      warped_age[idx] = 0.0f;
    }
  }

  mem_grid_ = warped_grid;
  mem_age_ = warped_age;
  fused_grid_ = warped_grid;
}

void BevDwaPlanner::filter_connected_free_space()
{
  // Keep ONLY the single connected FREE road component anchored near the front bumper!
  // Physically prevents jumping across a blue roundabout island or locking onto off-track glare.
  std::array<uint8_t, kBevCells> visited{};
  visited.fill(0);

  std::vector<int> queue;
  queue.reserve(8192);

  // Primary anchor box right ahead of/around front bumper
  int r0 = 0, r1 = 0, c0 = 0, c1 = 0;
  metric_to_cell(-0.04, 0.24, r0, c0);
  metric_to_cell(0.32, -0.24, r1, c1);
  r0 = std::clamp(r0, 0, kBevRows - 1);
  r1 = std::clamp(r1, 0, kBevRows - 1);
  c0 = std::clamp(c0, 0, kBevCols - 1);
  c1 = std::clamp(c1, 0, kBevCols - 1);

  for (int r = r0; r <= r1; ++r) {
    for (int c = c0; c <= c1; ++c) {
      const int idx = r * kBevCols + c;
      const uint8_t cell = fused_grid_[static_cast<size_t>(idx)];
      if (cell == BEV_FREE) {
        visited[static_cast<size_t>(idx)] = 1;
        queue.push_back(idx);
      }
    }
  }

  // Fallback: if car drifted slightly off-track, seed from closest FREE cell to (d=0.15, y=0)
  if (queue.empty()) {
    int best_idx = -1;
    double best_d2 = 1e9;
    for (int r = 0; r < kBevRows; ++r) {
      const double d_m = kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes;
      if (d_m < 0.0 || d_m > 0.70) {
        continue;
      }
      for (int c = 0; c < kBevCols; ++c) {
        const int idx = r * kBevCols + c;
        if (fused_grid_[static_cast<size_t>(idx)] != BEV_FREE) {
          continue;
        }
        const double y_m = kBevYMax - (static_cast<double>(c) + 0.5) * kBevRes;
        const double d2 = (d_m - 0.15) * (d_m - 0.15) + y_m * y_m;
        if (d2 < best_d2) {
          best_d2 = d2;
          best_idx = idx;
        }
      }
    }
    if (best_idx >= 0) {
      visited[static_cast<size_t>(best_idx)] = 1;
      queue.push_back(best_idx);
    }
  }

  // 4-connected BFS through FREE cells (also bridge tiny 1-cell red obstacle gaps only via FREE)
  size_t head = 0;
  constexpr int dr[4] = {-1, 1, 0, 0};
  constexpr int dc[4] = {0, 0, -1, 1};
  while (head < queue.size()) {
    const int cur = queue[head++];
    const int r = cur / kBevCols;
    const int c = cur % kBevCols;
    for (int k = 0; k < 4; ++k) {
      const int nr = r + dr[k];
      const int nc = c + dc[k];
      if (nr < 0 || nr >= kBevRows || nc < 0 || nc >= kBevCols) {
        continue;
      }
      const int nidx = nr * kBevCols + nc;
      if (!visited[static_cast<size_t>(nidx)] &&
        fused_grid_[static_cast<size_t>(nidx)] == BEV_FREE)
      {
        visited[static_cast<size_t>(nidx)] = 1;
        queue.push_back(nidx);
      }
    }
  }

  // Turn any disconnected FREE blob (e.g. opposite side of blue island) into WALL_BLUE
  if (!queue.empty()) {
    for (int i = 0; i < kBevCells; ++i) {
      const size_t idx = static_cast<size_t>(i);
      if (fused_grid_[idx] == BEV_FREE && !visited[idx]) {
        fused_grid_[idx] = BEV_WALL_BLUE;
      }
    }
  }

  // Heal tiny enclosed WALL_BLUE islands (< 120 cells ~= 11x11cm, NOT touching FOV/grid border)
  // caused by dark-blue printed digit stickers or specular chromatic noise inside white road.
  // Real outer blue mat and roundabout islands (2000+ cells or touching border) stay solid WALL_BLUE!
  std::array<uint8_t, kBevCells> wall_vis{};
  wall_vis.fill(0);
  std::vector<int> comp;
  comp.reserve(256);
  for (int i = 0; i < kBevCells; ++i) {
    const size_t start_idx = static_cast<size_t>(i);
    if (fused_grid_[start_idx] != BEV_WALL_BLUE || wall_vis[start_idx]) {
      continue;
    }
    comp.clear();
    wall_vis[start_idx] = 1;
    comp.push_back(i);
    bool touches_outer_or_unknown = false;
    size_t c_head = 0;
    while (c_head < comp.size()) {
      const int cur = comp[c_head++];
      const int r = cur / kBevCols;
      const int c = cur % kBevCols;
      if (r == 0 || r == kBevRows - 1 || c == 0 || c == kBevCols - 1) {
        touches_outer_or_unknown = true;
      }
      for (int k = 0; k < 4; ++k) {
        const int nr = r + dr[k];
        const int nc = c + dc[k];
        if (nr < 0 || nr >= kBevRows || nc < 0 || nc >= kBevCols) {
          continue;
        }
        const int nidx = nr * kBevCols + nc;
        const uint8_t ncell = fused_grid_[static_cast<size_t>(nidx)];
        if (ncell == BEV_UNKNOWN) {
          touches_outer_or_unknown = true;
        } else if (ncell == BEV_WALL_BLUE && !wall_vis[static_cast<size_t>(nidx)]) {
          wall_vis[static_cast<size_t>(nidx)] = 1;
          comp.push_back(nidx);
        }
      }
    }
    // Heal enclosed blue digit/noise islands (< 120 cells) and tiny bottom-edge specks (< 45 cells)
    // while preserving all real outer blue mat borders & roundabout islands (500-5000+ cells).
    const bool tiny_enclosed = (!touches_outer_or_unknown && comp.size() < 120U);
    const bool tiny_speck = (comp.size() < 45U);
    if (tiny_enclosed || tiny_speck) {
      for (int idx_i : comp) {
        fused_grid_[static_cast<size_t>(idx_i)] = BEV_FREE;
      }
    }
  }
}

void BevDwaPlanner::compute_distance_fields()
{
  compute_chamfer_dt(fused_grid_, BEV_WALL_BLUE, dist_wall_m_);
  compute_chamfer_dt(fused_grid_, BEV_FREE, dist_free_m_);
  compute_chamfer_dt(fused_grid_, BEV_OBS_CONE, dist_cone_m_);
  compute_chamfer_dt(fused_grid_, BEV_OBS_NO_ENTRY, dist_no_entry_m_);
}

double BevDwaPlanner::sample_dist_field(
  const std::array<float, kBevCells> & field,
  double d_m,
  double y_m) const
{
  int r = 0, c = 0;
  if (!metric_to_cell(d_m, y_m, r, c)) {
    return 0.0;
  }
  return static_cast<double>(field[static_cast<size_t>(r * kBevCols + c)]);
}

void BevDwaPlanner::measure_bev_lane_geometry(
  double & out_width_m,
  double & out_center_y_m,
  double & out_visual_yaw_rad,
  bool & out_clean_straight,
  double & out_straight_free_len_m,
  bool & out_t_junction_right_open,
  bool & out_left_branch_open,
  bool & out_right_branch_open) const
{
  out_width_m = config_.nominal_lane_width_m;
  out_center_y_m = 0.0;
  out_visual_yaw_rad = 0.0;
  out_clean_straight = false;
  out_straight_free_len_m = 0.0;
  out_t_junction_right_open = false;
  out_left_branch_open = false;
  out_right_branch_open = false;

  // 1) Measure straight-ahead wall-free distance along y = 0
  double straight_len = 0.0;
  for (double d = 0.05; d <= 1.05; d += 0.02) {
    int r = 0, c = 0;
    if (!metric_to_cell(d, 0.0, r, c)) {
      break;
    }
    const uint8_t cell = fused_grid_[static_cast<size_t>(r * kBevCols + c)];
    if (cell == BEV_WALL_BLUE) {
      break;
    }
    if (cell == BEV_FREE || cell == BEV_OBS_CONE || cell == BEV_OBS_NO_ENTRY) {
      straight_len = d;
    }
  }
  out_straight_free_len_m = straight_len;

  // 2) Measure lane width & centerline slope over d in [0.18, 0.55]m
  double w_sum = 0.0;
  double c_sum = 0.0;
  int n_rows = 0;
  double sum_d = 0.0, sum_y = 0.0, sum_dd = 0.0, sum_dy = 0.0;
  double max_w = 0.0;
  double min_w = 10.0;

  for (int r = 0; r < kBevRows; ++r) {
    const double d_m = kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes;
    if (d_m < 0.18 || d_m > 0.55) {
      continue;
    }
    int c_left = -1, c_right = -1;
    for (int c = 0; c < kBevCols; ++c) {
      if (fused_grid_[static_cast<size_t>(r * kBevCols + c)] == BEV_FREE) {
        if (c_left < 0) {
          c_left = c;
        }
        c_right = c;
      }
    }
    if (c_left >= 2 && c_right <= kBevCols - 3 && (c_right - c_left) >= 18) {
      double dl = 0.0, yl = 0.0, dr = 0.0, yr = 0.0;
      cell_to_metric(r, c_left, dl, yl);
      cell_to_metric(r, c_right, dr, yr);
      const double w = yl - yr;
      const double yc = 0.5 * (yl + yr);
      w_sum += w;
      c_sum += yc;
      max_w = std::max(max_w, w);
      min_w = std::min(min_w, w);
      sum_d += d_m;
      sum_y += yc;
      sum_dd += d_m * d_m;
      sum_dy += d_m * yc;
      n_rows++;
    }
  }

  if (n_rows >= 8) {
    out_width_m = w_sum / static_cast<double>(n_rows);
    out_center_y_m = c_sum / static_cast<double>(n_rows);
    const double denom = static_cast<double>(n_rows) * sum_dd - sum_d * sum_d;
    if (std::abs(denom) > 1e-6) {
      const double slope = (static_cast<double>(n_rows) * sum_dy - sum_d * sum_y) / denom;
      out_visual_yaw_rad = std::atan(slope);
      out_clean_straight =
        (out_width_m >= 0.34 && out_width_m <= 0.52) &&
        ((max_w - min_w) <= 0.14) &&
        (std::abs(out_visual_yaw_rad) <= 0.30);
    }
  }

  // 3) Check side openings in d in [0.22, 0.72]m for T-junction / fork awareness
  int left_free = 0;
  int right_free = 0;
  for (double d = 0.22; d <= 0.72; d += 0.03) {
    for (double y = 0.25; y <= 0.62; y += 0.03) {
      int r = 0, c = 0;
      if (metric_to_cell(d, y, r, c) &&
        fused_grid_[static_cast<size_t>(r * kBevCols + c)] == BEV_FREE)
      {
        left_free++;
      }
      if (metric_to_cell(d, -y, r, c) &&
        fused_grid_[static_cast<size_t>(r * kBevCols + c)] == BEV_FREE)
      {
        right_free++;
      }
    }
  }
  out_left_branch_open = (left_free >= 24);
  out_right_branch_open = (right_free >= 24);
  out_t_junction_right_open = (straight_len < 0.72) && (right_free >= 22);
}

void BevDwaPlanner::update_heading_and_topology(
  const VehiclePose2D & pose,
  double dt,
  double width_m,
  double visual_yaw_rad,
  bool clean_straight,
  double straight_free_len_m,
  bool t_junction_right_open,
  bool is_holding_yellow_stop)
{
  // 1) Update relative heading & arc length from EKF/odom with stationary gyro bias removed
  double dyaw = 0.0;
  double ds_fwd = 0.0;
  if (has_prev_pose_ && pose.valid && dt > 0.0 && dt < 0.35) {
    const double raw_dyaw = wrap_angle_rad(pose.yaw - prev_pose_.yaw);
    const double bias_corr = preflight_.gyro_calibrated ? (preflight_.gyro_bias_rps * dt) : 0.0;
    dyaw = raw_dyaw - bias_corr;
    if (std::abs(pose.linear_v) >= 0.015 || std::abs(raw_dyaw) >= 0.004) {
      rel_yaw_rad_ = wrap_angle_rad(rel_yaw_rad_ + dyaw);
      cum_abs_yaw_rad_ += std::abs(dyaw);
      phase_abs_yaw_rad_ += std::abs(dyaw);
    }

    const double dx = pose.x - prev_pose_.x;
    const double dy = pose.y - prev_pose_.y;
    const double ds_pos = std::hypot(dx, dy);
    if (ds_pos > 1e-4 && ds_pos < 0.5) {
      ds_fwd = std::max(0.0, dx * std::cos(pose.yaw) + dy * std::sin(pose.yaw));
    } else if (pose.linear_v > 0.015) {
      ds_fwd = pose.linear_v * dt;
    }
    total_arc_m_ += ds_fwd;
    phase_arc_m_ += ds_fwd;
    dist_since_yellow_m_ += ds_fwd;
  }

  // 2) Self-zeroing visual heading re-alignment on clean straights or while stopped at yellow
  if ((clean_straight && std::abs(prev_kappa_) < 0.45) ||
    (is_holding_yellow_stop && clean_straight))
  {
    const double road_world_yaw = wrap_angle_rad(rel_yaw_rad_ + visual_yaw_rad);
    constexpr double kAxes[4] = {0.0, 0.5 * kPi, -0.5 * kPi, kPi};
    double best_axis = 0.0;
    double best_err = 1e9;
    for (double ax : kAxes) {
      const double e = wrap_angle_rad(ax - road_world_yaw);
      if (std::abs(e) < std::abs(best_err)) {
        best_err = e;
        best_axis = ax;
      }
    }
    if (std::abs(best_err) < 0.30) {
      // Gently pull rel_yaw toward the true physical track axis (bounds gyro drift < 2 deg!)
      const double target_rel = wrap_angle_rad(best_axis - visual_yaw_rad);
      const double corr = wrap_angle_rad(target_rel - rel_yaw_rad_);
      rel_yaw_rad_ = wrap_angle_rad(rel_yaw_rad_ + 0.08 * corr);
    }
  }

  // 3) Zero-yellow-count topology phase transitions (pure BEV geometry + self-zeroed heading!)
  const double err_east = std::abs(wrap_angle_rad(rel_yaw_rad_ - kPi));
  const double err_west = std::abs(wrap_angle_rad(rel_yaw_rad_));
  const double err_north = std::abs(wrap_angle_rad(rel_yaw_rad_ + 0.5 * kPi));

  if (config_.track_mode == "fork") {
    switch (phase_) {
      case TopologyPhase::CRUISE_START:
        // If car already turned strongly north without yellow stop decision (edge safety)
        if (err_north < 0.45 && cum_abs_yaw_rad_ > 0.9) {
          phase_ = TopologyPhase::FORK_RIGHT_STEM;
          phase_arc_m_ = 0.0;
        }
        break;
      case TopologyPhase::FORK_RIGHT_ENTER:
        // Keep committed_branch_=+1 until physically turned North AND entered narrow stem!
        if ((err_north < 0.48 && width_m < 0.56 && phase_arc_m_ >= 0.40) || phase_arc_m_ > 2.20) {
          phase_ = TopologyPhase::FORK_RIGHT_STEM;
          committed_branch_ = 0;
          phase_arc_m_ = 0.0;
        }
        break;
      case TopologyPhase::FORK_RIGHT_STEM:
        // T-JUNCTION DETECTION: must have driven >= 0.20m up vertical stem facing North
        // AND genuinely see top horizontal bar (never false-trigger on L=0 wall-graze!)
        if (err_north < 0.55 && phase_arc_m_ >= 0.20 &&
          ((t_junction_right_open && straight_free_len_m < 0.72) ||
          (straight_free_len_m >= 0.06 && straight_free_len_m < 0.58)))
        {
          phase_ = TopologyPhase::FORK_T_RIGHT_TURN;
          phase_arc_m_ = 0.0;
        }
        break;
      case TopologyPhase::FORK_T_RIGHT_TURN:
        // Completed 90 deg right turn into East-bound S-curve
        if (err_east < 0.50) {
          phase_ = TopologyPhase::FORK_S_CURVE;
          phase_arc_m_ = 0.0;
        }
        break;
      case TopologyPhase::FORK_LEFT_UTURN:
        if (phase_abs_yaw_rad_ > 0.65 || (phase_arc_m_ > 0.90 && width_m < 0.52)) {
          committed_branch_ = 0;  // Release initial fork entry bias only once inside outer lane
        }
        // After sweeping through outer U-turn to East (+180 deg), enter S-curve past T-junction
        if (err_east < 0.55 && cum_abs_yaw_rad_ > 2.0) {
          phase_ = TopologyPhase::FORK_S_CURVE;
          committed_branch_ = 0;
          phase_arc_m_ = 0.0;
        }
        break;
      case TopologyPhase::FORK_S_CURVE:
      default:
        committed_branch_ = 0;
        break;
    }
  } else {
    // Roundabout mode
    switch (phase_) {
      case TopologyPhase::RB1_APPROACH:
        break;
      case TopologyPhase::RB1_ACTIVE:
        // Release branch mask only after turning onto the chosen side of the island
        if (phase_abs_yaw_rad_ > 0.65 || (phase_arc_m_ > 0.90 && width_m < 0.52)) {
          committed_branch_ = 0;
        }
        // Exited RB1 once swept > 103 deg around island and returned to West (0 deg) narrow straight
        if (phase_abs_yaw_rad_ > 1.80 && err_west < 0.45 && width_m < 0.54 && phase_arc_m_ > 1.35) {
          phase_ = TopologyPhase::RB_UTURN;
          committed_branch_ = 0;
          phase_arc_m_ = 0.0;
          phase_abs_yaw_rad_ = 0.0;
        }
        break;
      case TopologyPhase::RB_UTURN:
        // Completed big 180 deg U-turn onto East-bound straight approaching RB2
        if (err_east < 0.52 && phase_abs_yaw_rad_ > 2.20 && phase_arc_m_ > 1.50) {
          phase_ = TopologyPhase::RB2_APPROACH;
          phase_arc_m_ = 0.0;
          phase_abs_yaw_rad_ = 0.0;
        }
        break;
      case TopologyPhase::RB2_APPROACH:
        break;
      case TopologyPhase::RB2_ACTIVE:
        if (phase_abs_yaw_rad_ > 0.65 || (phase_arc_m_ > 0.90 && width_m < 0.52)) {
          committed_branch_ = 0;
        }
        if (phase_abs_yaw_rad_ > 1.80 && err_east < 0.45 && width_m < 0.54 && phase_arc_m_ > 1.35) {
          phase_ = TopologyPhase::RB2_EXIT;
          committed_branch_ = 0;
          phase_arc_m_ = 0.0;
        }
        break;
      case TopologyPhase::RB2_EXIT:
      default:
        committed_branch_ = 0;
        break;
    }
  }
}

std::vector<GroundDetection> BevDwaPlanner::filter_and_lock_straight_digits(
  const std::vector<GroundDetection> & valid_digits,
  bool allow_digit_bonus)
{
  std::vector<GroundDetection> out;
  if (!allow_digit_bonus) {
    locked_digit_val_ = 0;
    locked_digit_hits_ = 0;
    locked_digit_misses_ = 0;
    cand_digit_val_ = 0;
    cand_digit_streak_ = 0;
    return out;
  }

  // Pick highest valid in-range straight digit this frame
  int best_val = 0;
  double best_y = 0.0;
  double best_d = 0.0;
  for (const auto & gd : valid_digits) {
    if (gd.type != SemanticObjectType::BONUS_DIGIT || gd.digit < 1 || gd.digit > 8) {
      continue;
    }
    if (gd.confidence < config_.digit_min_conf) {
      continue;
    }
    if (gd.d_bumper_m < 0.18 || gd.d_bumper_m > 0.90) {
      continue;
    }
    if (std::abs(gd.y_lat_m) > config_.max_digit_lateral_shift_m + 0.06) {
      continue;
    }
    if (gd.digit > best_val) {
      best_val = gd.digit;
      best_y = gd.y_lat_m;
      best_d = gd.d_bumper_m;
    }
  }

  // Multi-frame lock: require 3 consecutive frames to acquire; once locked, NEVER jump
  // to a lower same-row digit when YOLO flickers for 1-5 frames!
  if (locked_digit_val_ > 0) {
    bool seen_locked = false;
    for (const auto & gd : valid_digits) {
      if (gd.type == SemanticObjectType::BONUS_DIGIT && gd.digit == locked_digit_val_ &&
        gd.confidence >= config_.digit_min_conf && gd.d_bumper_m >= 0.14 && gd.d_bumper_m <= 0.95)
      {
        locked_digit_y_ = 0.65 * locked_digit_y_ + 0.35 * gd.y_lat_m;
        locked_digit_d_ = gd.d_bumper_m;
        locked_digit_misses_ = 0;
        locked_digit_hits_++;
        seen_locked = true;
        break;
      }
    }
    if (!seen_locked) {
      // Only promote to a strictly HIGHER digit if seen 3 frames in a row
      if (best_val > locked_digit_val_) {
        if (best_val == cand_digit_val_) {
          cand_digit_streak_++;
        } else {
          cand_digit_val_ = best_val;
          cand_digit_streak_ = 1;
        }
        if (cand_digit_streak_ >= 3) {
          locked_digit_val_ = best_val;
          locked_digit_y_ = best_y;
          locked_digit_d_ = best_d;
          locked_digit_hits_ = 3;
          locked_digit_misses_ = 0;
          cand_digit_val_ = 0;
          cand_digit_streak_ = 0;
          seen_locked = true;
        }
      }
      if (!seen_locked) {
        locked_digit_misses_++;
        if (locked_digit_misses_ >= 6 || locked_digit_d_ < 0.16) {
          locked_digit_val_ = 0;
          locked_digit_hits_ = 0;
          locked_digit_misses_ = 0;
        }
      }
    }
  } else if (best_val > 0) {
    if (best_val == cand_digit_val_) {
      cand_digit_streak_++;
    } else {
      cand_digit_val_ = best_val;
      cand_digit_streak_ = 1;
    }
    if (cand_digit_streak_ >= 3) {
      locked_digit_val_ = best_val;
      locked_digit_y_ = best_y;
      locked_digit_d_ = best_d;
      locked_digit_hits_ = 3;
      locked_digit_misses_ = 0;
      cand_digit_val_ = 0;
      cand_digit_streak_ = 0;
    }
  } else {
    cand_digit_val_ = 0;
    cand_digit_streak_ = 0;
  }

  if (locked_digit_val_ > 0) {
    GroundDetection locked{};
    locked.type = SemanticObjectType::BONUS_DIGIT;
    locked.digit = locked_digit_val_;
    locked.confidence = 0.90;
    locked.d_bumper_m = locked_digit_d_;
    locked.y_lat_m = locked_digit_y_;
    out.push_back(locked);

    // Also pass through any genuinely visible smaller same-row digit for anti-dual-cover
    for (const auto & gd : valid_digits) {
      if (gd.type == SemanticObjectType::BONUS_DIGIT &&
        gd.digit > 0 && gd.digit < locked_digit_val_ &&
        gd.confidence >= config_.digit_min_conf &&
        std::abs(gd.d_bumper_m - locked_digit_d_) <= 0.22)
      {
        out.push_back(gd);
      }
    }
  }
  return out;
}

void BevDwaPlanner::extract_safe_corridor(
  double y0,
  int branch_bias,
  double goal_yaw,
  bool goal_yaw_active,
  bool allow_glare_guard,
  std::vector<BevCorridorSlice> & out_slices) const
{
  out_slices.clear();
  const double r_car = 0.5 * config_.car_width_m;
  const double wall_shrink = r_car + config_.border_margin_m;
  const double nom_w = config_.nominal_lane_width_m;
  const double d_step = 0.02;
  const double d_max = std::min(kBevDMax - 0.02, config_.rollout_horizon_m);

  struct Run
  {
    double y_right{0.0};
    double y_left{0.0};
    double width() const { return y_left - y_right; }
    double center() const { return 0.5 * (y_left + y_right); }
  };

  bool has_prev = false;
  double prev_yr = -0.5 * nom_w;
  double prev_yl = 0.5 * nom_w;
  double prev_yc = y0;

  for (double d = 0.04; d <= d_max + 1e-6; d += d_step) {
    BevCorridorSlice sl{};
    sl.d_m = d;
    int r = 0, c_dummy = 0;
    if (!metric_to_cell(d, 0.0, r, c_dummy)) {
      out_slices.push_back(sl);
      continue;
    }

    // Collect contiguous drivable road runs (FREE + interior RED obstacles so a centered cone
    // does not fracture the road before Stage D obstacle carving!).
    std::vector<Run> runs;
    int run_start = -1;
    const int row_off = r * kBevCols;
    for (int c = 0; c <= kBevCols; ++c) {
      bool drivable = false;
      if (c < kBevCols) {
        const uint8_t cell = fused_grid_[static_cast<size_t>(row_off + c)];
        drivable = (cell == BEV_FREE || cell == BEV_OBS_CONE || cell == BEV_OBS_NO_ENTRY);
      }
      if (drivable) {
        if (run_start < 0) {
          run_start = c;
        }
      } else if (run_start >= 0) {
        const int len = c - run_start;
        if (len >= 12) {
          double dl = 0.0, yl = 0.0, dr_m = 0.0, yr = 0.0;
          cell_to_metric(r, run_start, dl, yl);
          cell_to_metric(r, c - 1, dr_m, yr);
          runs.push_back(Run{yr, yl});
        }
        run_start = -1;
      }
    }

    if (runs.empty()) {
      out_slices.push_back(sl);
      continue;
    }

    Run chosen = runs[0];
    if (runs.size() == 1U) {
      // Wide throat handling (fork entry, T-junction, or glare blowout)
      const double w = chosen.width();
      if (w > 0.56 && branch_bias != 0) {
        if (branch_bias < 0) {
          chosen.y_right = std::max(chosen.y_right, chosen.y_left - nom_w);
        } else {
          chosen.y_left = std::min(chosen.y_left, chosen.y_right + nom_w);
        }
      } else if (w > 0.58 && phase_ == TopologyPhase::FORK_T_RIGHT_TURN) {
        chosen.y_left = std::min(chosen.y_left, chosen.y_right + nom_w);
      } else if (allow_glare_guard && w > 0.62 && has_prev) {
        // Single-sided glare blowout guard: if only ONE edge jumped outward abnormally (> 0.14m),
        // lock the continuous physical border and reconstruct the blown-out side with nominal width.
        const double jump_l = chosen.y_left - prev_yl;
        const double jump_r = prev_yr - chosen.y_right;
        if (jump_l > 0.14 && jump_r < 0.08) {
          chosen.y_left = chosen.y_right + nom_w;
        } else if (jump_r > 0.14 && jump_l < 0.08) {
          chosen.y_right = chosen.y_left - nom_w;
        }
      }
    } else {
      // Multi-run split (roundabout island or fork divider): select strictly by topology / continuity!
      if (branch_bias < 0) {
        // Leftmost run (largest y)
        chosen = *std::max_element(
          runs.begin(), runs.end(),
          [](const Run & a, const Run & b) { return a.center() < b.center(); });
      } else if (branch_bias > 0) {
        // Rightmost run (smallest y)
        chosen = *std::min_element(
          runs.begin(), runs.end(),
          [](const Run & a, const Run & b) { return a.center() < b.center(); });
      } else {
        double best_score = -1e9;
        for (const auto & run : runs) {
          const double cont = -std::abs(run.center() - prev_yc);
          double h_bonus = 0.0;
          if (goal_yaw_active) {
            const double run_yaw = wrap_angle_rad(rel_yaw_rad_ + std::atan2(run.center(), d));
            h_bonus = 0.25 * std::cos(wrap_angle_rad(run_yaw - goal_yaw));
          }
          const double sc = cont + h_bonus + 0.05 * std::min(0.45, run.width());
          if (sc > best_score) {
            best_score = sc;
            chosen = run;
          }
        }
      }
    }

    sl.valid = true;
    sl.y_raw_right = chosen.y_right;
    sl.y_raw_left = chosen.y_left;
    double sr = chosen.y_right + wall_shrink;
    double sl_b = chosen.y_left - wall_shrink;
    if (sr > sl_b) {
      const double mid = 0.5 * (chosen.y_right + chosen.y_left);
      sr = mid;
      sl_b = mid;
    }
    sl.y_safe_right = sr;
    sl.y_safe_left = sl_b;
    sl.y_target = 0.5 * (sr + sl_b);
    out_slices.push_back(sl);

    prev_yr = chosen.y_right;
    prev_yl = chosen.y_left;
    prev_yc = sl.y_target;
    has_prev = true;
  }
}

std::vector<BevDwaPlanner::ObstacleCluster> BevDwaPlanner::extract_obstacle_clusters() const
{
  std::vector<ObstacleCluster> clusters;
  std::array<uint8_t, kBevCells> vis{};
  vis.fill(0);

  int r_min = 0, r_max = 0, c_dummy = 0;
  metric_to_cell(-0.24, 0.0, r_min, c_dummy);
  metric_to_cell(std::min(kBevDMax - 0.02, config_.rollout_horizon_m + 0.10), 0.0, r_max, c_dummy);
  r_min = std::clamp(r_min, 0, kBevRows - 1);
  r_max = std::clamp(r_max, 0, kBevRows - 1);

  std::vector<int> q;
  q.reserve(256);
  constexpr int dr[4] = {-1, 1, 0, 0};
  constexpr int dc[4] = {0, 0, -1, 1};

  for (int r = r_min; r <= r_max; ++r) {
    for (int c = 0; c < kBevCols; ++c) {
      const int idx = r * kBevCols + c;
      if (vis[static_cast<size_t>(idx)]) {
        continue;
      }
      const uint8_t cell = fused_grid_[static_cast<size_t>(idx)];
      if (cell != BEV_OBS_CONE && cell != BEV_OBS_NO_ENTRY) {
        continue;
      }

      ObstacleCluster cl{};
      cl.d_min = 1e9;
      cl.d_max = -1e9;
      cl.y_min = 1e9;
      cl.y_max = -1e9;
      cl.is_no_entry = (cell == BEV_OBS_NO_ENTRY);

      q.clear();
      vis[static_cast<size_t>(idx)] = 1;
      q.push_back(idx);
      size_t head = 0;
      while (head < q.size()) {
        const int cur = q[head++];
        const int cr = cur / kBevCols;
        const int cc = cur % kBevCols;
        double dm = 0.0, ym = 0.0;
        cell_to_metric(cr, cc, dm, ym);
        cl.d_min = std::min(cl.d_min, dm);
        cl.d_max = std::max(cl.d_max, dm);
        cl.y_min = std::min(cl.y_min, ym);
        cl.y_max = std::max(cl.y_max, ym);
        cl.cell_count++;
        if (fused_grid_[static_cast<size_t>(cur)] == BEV_OBS_NO_ENTRY) {
          cl.is_no_entry = true;
        }
        for (int k = 0; k < 4; ++k) {
          const int nr = cr + dr[k];
          const int nc = cc + dc[k];
          if (nr < r_min || nr > r_max || nc < 0 || nc >= kBevCols) {
            continue;
          }
          const int nidx = nr * kBevCols + nc;
          if (!vis[static_cast<size_t>(nidx)]) {
            const uint8_t nc_val = fused_grid_[static_cast<size_t>(nidx)];
            if (nc_val == BEV_OBS_CONE || nc_val == BEV_OBS_NO_ENTRY) {
              vis[static_cast<size_t>(nidx)] = 1;
              q.push_back(nidx);
            }
          }
        }
      }

      if (cl.cell_count >= 3) {
        clusters.push_back(cl);
      }
    }
  }
  return clusters;
}

void BevDwaPlanner::carve_obstacles_into_corridor(
  const std::vector<ObstacleCluster> & clusters,
  std::vector<BevCorridorSlice> & slices)
{
  if (slices.empty() || clusters.empty()) {
    prev_obs_miss_frames_ = std::min(100, prev_obs_miss_frames_ + 1);
    if (prev_obs_miss_frames_ > 8) {
      prev_obs_pass_side_ = 0;
    }
    return;
  }

  const double r_car = 0.5 * config_.car_width_m;
  const size_t n = slices.size();
  std::vector<double> right_prio(n, 1.0);
  std::vector<double> left_prio(n, 1.0);
  std::vector<double> orig_safe_r(n, 0.0);
  std::vector<double> orig_safe_l(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    orig_safe_r[i] = slices[i].y_safe_right;
    orig_safe_l[i] = slices[i].y_safe_left;
  }

  // Sort clusters near-to-far so hysteresis tracks the primary closest obstacle cleanly
  std::vector<ObstacleCluster> sorted = clusters;
  std::sort(
    sorted.begin(), sorted.end(),
    [](const ObstacleCluster & a, const ObstacleCluster & b) { return a.d_min < b.d_min; });

  bool updated_primary = false;

  for (const auto & obs : sorted) {
    const double d_mid = clamp_d(0.5 * (obs.d_min + obs.d_max), 0.04, config_.rollout_horizon_m);
    const double y_mid = 0.5 * (obs.y_min + obs.y_max);

    // Find closest valid corridor slice to measure left/right passage gaps in the curved lane!
    int ref_idx = -1;
    double best_dd = 1e9;
    for (size_t i = 0; i < n; ++i) {
      if (!slices[i].valid) {
        continue;
      }
      const double dd = std::abs(slices[i].d_m - d_mid);
      if (dd < best_dd) {
        best_dd = dd;
        ref_idx = static_cast<int>(i);
      }
    }
    if (ref_idx < 0) {
      continue;
    }

    const auto & ref = slices[static_cast<size_t>(ref_idx)];
    // Skip obstacles strictly outside the active corridor walls
    if (obs.y_max < ref.y_raw_right - 0.04 || obs.y_min > ref.y_raw_left + 0.04) {
      continue;
    }

    const double gap_L = ref.y_raw_left - obs.y_max;
    const double gap_R = obs.y_min - ref.y_raw_right;

    // Pass-side selection (+1 = pass LEFT of obstacle, -1 = pass RIGHT) with hysteresis memory
    double score_L = gap_L;
    double score_R = gap_R;
    if (!updated_primary && prev_obs_pass_side_ != 0 && prev_obs_miss_frames_ <= 6 &&
      std::abs(obs.d_min - prev_obs_d_) < 0.28 && std::abs(y_mid - prev_obs_y_) < 0.24)
    {
      if (prev_obs_pass_side_ > 0) {
        score_L += 0.040;
      } else {
        score_R += 0.040;
      }
    }
    const int pass_side = (score_L >= score_R) ? +1 : -1;

    if (!updated_primary && obs.d_max >= -0.15 && obs.d_min <= 0.80) {
      prev_obs_pass_side_ = pass_side;
      prev_obs_d_ = obs.d_min;
      prev_obs_y_ = y_mid;
      prev_obs_miss_frames_ = 0;
      updated_primary = true;
    }

    const double margin = obs.is_no_entry ? config_.no_entry_margin_m : config_.cone_margin_m;
    const double obs_req = r_car + margin;
    const double prio = obs.is_no_entry ? 4.0 : 2.0;

    // Smooth S-curve longitudinal envelope:
    //   lead-in  : [d_min - 0.30, d_min - 0.05]  (smooth cosine ramp into bypass offset)
    //   full hold: [d_min - 0.05, d_max + 0.22]  (protects BOTH front bumper & rear axle 0.18m behind!)
    //   lead-out : [d_max + 0.22, d_max + 0.38]  (smooth cosine return to curved lane centerline)
    const double d_in_start = obs.d_min - 0.30;
    const double d_hold_start = obs.d_min - 0.05;
    const double d_hold_end = obs.d_max + 0.22;
    const double d_out_end = obs.d_max + 0.38;

    for (size_t i = 0; i < n; ++i) {
      if (!slices[i].valid) {
        continue;
      }
      const double d = slices[i].d_m;
      if (d < d_in_start || d > d_out_end) {
        continue;
      }
      double w_env = 1.0;
      if (d < d_hold_start) {
        const double t = clamp_d((d - d_in_start) / std::max(0.05, d_hold_start - d_in_start), 0.0, 1.0);
        w_env = 0.5 * (1.0 - std::cos(kPi * t));
      } else if (d > d_hold_end) {
        const double t = clamp_d((d_out_end - d) / std::max(0.05, d_out_end - d_hold_end), 0.0, 1.0);
        w_env = 0.5 * (1.0 - std::cos(kPi * t));
      }

      if (pass_side > 0) {
        // Pass LEFT of obstacle -> raise y_safe_right
        const double y_req_r = obs.y_max + obs_req;
        const double y_carved = orig_safe_r[i] + w_env * (y_req_r - orig_safe_r[i]);
        if (y_carved > slices[i].y_safe_right) {
          slices[i].y_safe_right = y_carved;
          right_prio[i] = std::max(right_prio[i], prio);
        }
      } else {
        // Pass RIGHT of obstacle -> lower y_safe_left
        const double y_req_l = obs.y_min - obs_req;
        const double y_carved = orig_safe_l[i] + w_env * (y_req_l - orig_safe_l[i]);
        if (y_carved < slices[i].y_safe_left) {
          slices[i].y_safe_left = y_carved;
          left_prio[i] = std::max(left_prio[i], prio);
        }
      }
    }
  }

  if (!updated_primary) {
    prev_obs_miss_frames_ = std::min(100, prev_obs_miss_frames_ + 1);
    if (prev_obs_miss_frames_ > 8) {
      prev_obs_pass_side_ = 0;
    }
  }

  // Recompute corridor target centerline; if squeezed, resolve asymmetrically (+30s no_entry first!)
  for (size_t i = 0; i < n; ++i) {
    if (!slices[i].valid) {
      continue;
    }
    if (slices[i].y_safe_right <= slices[i].y_safe_left) {
      slices[i].y_target = 0.5 * (slices[i].y_safe_right + slices[i].y_safe_left);
    } else {
      const double denom = right_prio[i] + left_prio[i];
      const double y_comp =
        (slices[i].y_safe_right * right_prio[i] + slices[i].y_safe_left * left_prio[i]) / denom;
      slices[i].y_safe_right = y_comp;
      slices[i].y_safe_left = y_comp;
      slices[i].y_target = y_comp;
    }
  }
}

void BevDwaPlanner::apply_digit_bias_to_corridor(
  const std::vector<GroundDetection> & locked_digits,
  bool allow_digit_bonus,
  std::vector<BevCorridorSlice> & slices,
  int & out_hit_digit,
  double & out_digit_shift_m) const
{
  out_hit_digit = 0;
  out_digit_shift_m = 0.0;
  if (!allow_digit_bonus || locked_digits.empty() || slices.empty()) {
    return;
  }

  const GroundDetection & target = locked_digits[0];
  const double r_car = 0.5 * config_.car_width_m;
  double y_des = clamp_d(
    target.y_lat_m, -config_.max_digit_lateral_shift_m, config_.max_digit_lateral_shift_m);

  // Anti-dual-cover: push away from any smaller same-row digit if dual coverage is forbidden
  if (!config_.allow_dual_digit_cover && locked_digits.size() > 1U) {
    const double min_sep = r_car + 0.045;
    for (size_t k = 1; k < locked_digits.size(); ++k) {
      const double y_small = locked_digits[k].y_lat_m;
      const double diff = y_des - y_small;
      if (std::abs(diff) < min_sep) {
        const double push_dir = (diff >= 0.0) ? 1.0 : -1.0;
        y_des = clamp_d(
          y_small + push_dir * min_sep,
          -config_.max_digit_lateral_shift_m,
          config_.max_digit_lateral_shift_m);
      }
    }
  }

  double max_applied_shift = 0.0;
  bool clamped_hit = false;
  for (auto & sl : slices) {
    if (!sl.valid) {
      continue;
    }
    const double dd = sl.d_m - target.d_bumper_m;
    if (std::abs(dd) > 0.30) {
      continue;
    }
    const double g = std::exp(-0.5 * (dd / 0.14) * (dd / 0.14));
    const double y_prev = sl.y_target;
    const double y_cand = y_prev + g * (y_des - y_prev);
    // HARD CLAMP inside safe corridor: digit bonus can NEVER pull the car into wall or obstacle!
    sl.y_target = clamp_d(y_cand, sl.y_safe_right, sl.y_safe_left);
    const double applied = sl.y_target - y_prev;
    if (std::abs(applied) > std::abs(max_applied_shift)) {
      max_applied_shift = applied;
    }
    if (std::abs(dd) <= 0.08 && std::abs(sl.y_target - target.y_lat_m) < 0.11) {
      clamped_hit = true;
    }
  }

  if (clamped_hit || std::abs(max_applied_shift) > 0.005) {
    out_hit_digit = target.digit;
    out_digit_shift_m = max_applied_shift;
  }
}

void BevDwaPlanner::smooth_corridor_centerline(std::vector<BevCorridorSlice> & slices) const
{
  if (slices.size() < 3U) {
    return;
  }
  std::vector<double> tmp(slices.size(), 0.0);
  for (int pass = 0; pass < 3; ++pass) {
    for (size_t i = 0; i < slices.size(); ++i) {
      tmp[i] = slices[i].y_target;
    }
    for (size_t i = 1; i + 1 < slices.size(); ++i) {
      if (slices[i - 1].valid && slices[i].valid && slices[i + 1].valid) {
        const double sm = 0.25 * tmp[i - 1] + 0.50 * tmp[i] + 0.25 * tmp[i + 1];
        // Re-clamp inside safe bounds after every smoothing pass!
        slices[i].y_target = clamp_d(sm, slices[i].y_safe_right, slices[i].y_safe_left);
      }
    }
  }
}

void BevDwaPlanner::fit_corridor_curve_arc(
  const std::vector<BevCorridorSlice> & slices,
  double x0,
  double y0,
  double th0,
  double & out_road_kappa,
  double & out_lat_err_m,
  double & out_heading_err_rad,
  double & out_corridor_len_m) const
{
  out_road_kappa = 0.0;
  out_lat_err_m = 0.0;
  out_heading_err_rad = 0.0;
  out_corridor_len_m = 0.0;

  // Measure contiguous corridor horizon length (tolerating up to 2 blank rows)
  int gap_rows = 0;
  bool started = false;
  int valid_n = 0;
  for (const auto & sl : slices) {
    if (sl.valid) {
      started = true;
      gap_rows = 0;
      out_corridor_len_m = sl.d_m;
      valid_n++;
    } else if (started) {
      gap_rows++;
      if (gap_rows > 2) {
        break;
      }
    }
  }

  // Near-bumper lateral error e_y at d_near = max(0.06, x0 + 0.06)
  const double d_near = std::max(0.06, x0 + 0.06);
  bool found_near = false;
  double near_dd = 1e9;
  for (const auto & sl : slices) {
    if (!sl.valid) {
      continue;
    }
    const double dd = std::abs(sl.d_m - d_near);
    if (dd < near_dd) {
      near_dd = dd;
      out_lat_err_m = sl.y_target - y0;
      found_near = true;
    }
  }
  (void)found_near;

  if (valid_n < 5) {
    return;
  }

  // Weighted least-squares quadratic fit: y(d) = a0 + a1*d + a2*d^2
  // Gives exact road arc curvature k_road = 2*a2 / (1 + a1^2)^(3/2)!
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
  double b0 = 0.0, b1 = 0.0, b2 = 0.0;
  for (const auto & sl : slices) {
    if (!sl.valid || sl.d_m > out_corridor_len_m + 0.01) {
      continue;
    }
    const double d = sl.d_m;
    const double y = sl.y_target;
    const double w = std::exp(-1.1 * d);
    const double d2 = d * d;
    s0 += w;
    s1 += w * d;
    s2 += w * d2;
    s3 += w * d2 * d;
    s4 += w * d2 * d2;
    b0 += w * y;
    b1 += w * d * y;
    b2 += w * d2 * y;
  }

  // Tiny ridge regularization on curvature term to prevent ill-conditioning on short horizons
  s4 += 1e-4;
  s2 += 1e-6;

  // Solve 3x3 symmetric system via Gaussian elimination with partial pivoting
  double M[3][4] = {
    {s0, s1, s2, b0},
    {s1, s2, s3, b1},
    {s2, s3, s4, b2}
  };
  for (int col = 0; col < 3; ++col) {
    int piv = col;
    for (int r = col + 1; r < 3; ++r) {
      if (std::abs(M[r][col]) > std::abs(M[piv][col])) {
        piv = r;
      }
    }
    if (std::abs(M[piv][col]) < 1e-10) {
      return;
    }
    if (piv != col) {
      for (int j = col; j < 4; ++j) {
        std::swap(M[col][j], M[piv][j]);
      }
    }
    const double diag = M[col][col];
    for (int r = col + 1; r < 3; ++r) {
      const double f = M[r][col] / diag;
      for (int j = col; j < 4; ++j) {
        M[r][j] -= f * M[col][j];
      }
    }
  }
  double a[3] = {0.0, 0.0, 0.0};
  for (int r = 2; r >= 0; --r) {
    double sum = M[r][3];
    for (int j = r + 1; j < 3; ++j) {
      sum -= M[r][j] * a[j];
    }
    a[r] = sum / M[r][r];
  }

  const double slope = a[1] + 2.0 * a[2] * x0;
  const double denom = std::pow(1.0 + slope * slope, 1.5);
  out_road_kappa = clamp_d((2.0 * a[2]) / std::max(1.0, denom), -3.5, 3.5);
  out_heading_err_rad = wrap_angle_rad(std::atan(slope) - th0);
}

void BevDwaPlanner::check_chosen_arc_clearance(
  double x0,
  double y0,
  double th0,
  double kappa,
  double & out_safe_len_m,
  double & out_min_wall_clr_m,
  double & out_min_obs_clr_m) const
{
  const double r_car = 0.5 * config_.car_width_m;
  const double wall_hard = r_car + config_.wall_hard_margin_m;
  const double cone_hard = r_car + config_.cone_hard_margin_m;
  const double ne_hard = r_car + config_.no_entry_hard_margin_m;
  constexpr double kBodyOffsets[3] = {0.0, -0.10, -0.22};

  const double ds = 0.025;
  const int max_steps = static_cast<int>(std::ceil(config_.rollout_horizon_m / ds));
  double x = x0;
  double y = y0;
  double th = th0;

  out_safe_len_m = 0.0;
  out_min_wall_clr_m = 1.0;
  out_min_obs_clr_m = 1.0;
  bool hit = false;

  for (int step = 1; step <= max_steps; ++step) {
    const double dth = kappa * ds;
    x += ds * std::cos(th + 0.5 * dth);
    y += ds * std::sin(th + 0.5 * dth);
    th += dth;

    double dw = 1.0;
    double dc = 1.0;
    double dne = 1.0;
    bool out_front = false;
    const double cth = std::cos(th);
    const double sth = std::sin(th);
    for (int b = 0; b < 3; ++b) {
      const double bx = x + kBodyOffsets[b] * cth;
      const double by = y + kBodyOffsets[b] * sth;
      int r = 0, c = 0;
      if (!metric_to_cell(bx, by, r, c)) {
        if (b == 0) {
          out_front = true;
        }
        continue;
      }
      const size_t idx = static_cast<size_t>(r * kBevCols + c);
      const uint8_t cell = fused_grid_[idx];
      // Check wall distance strictly in live forward FOV (bx >= 0.02) to avoid blind-spot traps
      if (bx >= 0.02 && cell != BEV_UNKNOWN) {
        const double signed_w = (cell == BEV_WALL_BLUE) ?
          -static_cast<double>(dist_free_m_[idx]) :
          static_cast<double>(dist_wall_m_[idx]);
        dw = std::min(dw, signed_w);
      }
      // Check red obstacles across full 3-circle chassis (protects rear wheels!)
      if (bx >= -0.24) {
        dc = std::min(dc, static_cast<double>(dist_cone_m_[idx]));
        dne = std::min(dne, static_cast<double>(dist_no_entry_m_[idx]));
      }
    }
    if (out_front) {
      break;
    }

    out_min_wall_clr_m = std::min(out_min_wall_clr_m, dw - r_car);
    out_min_obs_clr_m = std::min(out_min_obs_clr_m, std::min(dc, dne) - r_car);

    if (!hit && (dw < wall_hard || dc < cone_hard || dne < ne_hard)) {
      hit = true;
      break;
    }
    out_safe_len_m = static_cast<double>(step) * ds;
  }
}

void BevDwaPlanner::compute_geodesic_progress_field(
  int branch_bias,
  double goal_yaw,
  bool goal_yaw_active,
  bool forbid_reverse_yaw,
  double reverse_yaw)
{
  (void)goal_yaw;
  (void)goal_yaw_active;
  (void)forbid_reverse_yaw;
  (void)reverse_yaw;

  road_progress_m_.fill(-1.0f);
  blocked_branch_grid_.fill(0);

  // 1) Topological wrong-way branch masking
  if (phase_ == TopologyPhase::FORK_T_RIGHT_TURN) {
    // At T-junction crossbar, mask wrong left arm (going toward world yaw 0)
    for (int r = 0; r < kBevRows; ++r) {
      const double dm = kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes;
      if (dm <= 0.22) {
        continue;
      }
      const int row_off = r * kBevCols;
      for (int c = 0; c < kBevCols; ++c) {
        const double ym = kBevYMax - (static_cast<double>(c) + 0.5) * kBevRes;
        if (ym > 0.14) {
          blocked_branch_grid_[static_cast<size_t>(row_off + c)] = 1;
        }
      }
    }
  }

  if (branch_bias != 0) {
    // Only split branches across genuine blue islands/borders (>= 3 WALL_BLUE or UNKNOWN),
    // keeping cones/no-entry INSIDE a single road span connected!
    for (int r = 0; r < kBevRows; ++r) {
      const double dm = kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes;
      if (dm < 0.14) {
        continue;
      }
      const int row_off = r * kBevCols;
      bool road_like[kBevCols]{};
      for (int c = 0; c < kBevCols; ++c) {
        const uint8_t cell = fused_grid_[static_cast<size_t>(row_off + c)];
        road_like[c] = (cell == BEV_FREE || cell == BEV_OBS_CONE || cell == BEV_OBS_NO_ENTRY);
      }
      // Bridge 1..2 cell WALL_BLUE specks inside the same branch
      for (int c = 1; c + 1 < kBevCols; ++c) {
        if (!road_like[c] && fused_grid_[static_cast<size_t>(row_off + c)] == BEV_WALL_BLUE) {
          if ((road_like[c - 1] && road_like[c + 1]) ||
            (c + 2 < kBevCols && road_like[c - 1] && road_like[c + 2] &&
              fused_grid_[static_cast<size_t>(row_off + c + 1)] == BEV_WALL_BLUE))
          {
            road_like[c] = true;
          }
        }
      }

      std::vector<std::pair<int, int>> intervals;
      int c = 0;
      while (c < kBevCols) {
        if (!road_like[c]) {
          ++c;
          continue;
        }
        const int start_c = c;
        while (c < kBevCols && road_like[c]) {
          ++c;
        }
        const int end_c = c - 1;
        if (end_c - start_c + 1 >= 4) {
          intervals.emplace_back(start_c, end_c);
        }
      }

      if (intervals.size() >= 2U) {
        // c=0 is leftmost (+y), c=kBevCols-1 is rightmost (-y)
        const size_t keep_idx = (branch_bias > 0) ? (intervals.size() - 1U) : 0U;
        for (size_t k = 0; k < intervals.size(); ++k) {
          if (k == keep_idx) {
            continue;
          }
          for (int cc = intervals[k].first; cc <= intervals[k].second; ++cc) {
            const size_t idx = static_cast<size_t>(row_off + cc);
            if (fused_grid_[idx] == BEV_FREE) {
              blocked_branch_grid_[idx] = 1;
            }
          }
        }
      }
    }
  } else if (phase_ != TopologyPhase::FORK_T_RIGHT_TURN) {
    // Single-sided window glare guard: ONLY trigger on a genuine straight corridor continuing
    // past d >= 0.78m where one edge stays steady to d >= 0.75m while the opposite edge jumps > 16cm.
    double glare_straight_len = 0.0;
    for (double dm = 0.10; dm <= 0.90; dm += 0.02) {
      int rr = 0, cc = 0;
      if (!metric_to_cell(dm, 0.0, rr, cc)) {
        break;
      }
      if (fused_grid_[static_cast<size_t>(rr * kBevCols + cc)] == BEV_WALL_BLUE) {
        break;
      }
      glare_straight_len = dm;
    }
    if (glare_straight_len >= 0.76) {
      bool has_prev_edges = false;
      double prev_yl = 0.0, prev_yr = 0.0;
      bool glare_left = false;
      bool glare_right = false;
      for (int r = 0; r < kBevRows; ++r) {
        const double dm = kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes;
        if (dm < 0.14 || dm > 0.95) {
          continue;
        }
        const int row_off = r * kBevCols;
        int c_l = -1, c_r = -1;
        for (int c = 0; c < kBevCols; ++c) {
          const uint8_t cell = fused_grid_[static_cast<size_t>(row_off + c)];
          if (cell == BEV_FREE || cell == BEV_OBS_CONE || cell == BEV_OBS_NO_ENTRY) {
            if (c_l < 0) {
              c_l = c;
            }
            c_r = c;
          }
        }
        if (c_l >= 0 && (c_r - c_l) >= 12) {
          double dl = 0.0, yl = 0.0, dr = 0.0, yr = 0.0;
          cell_to_metric(r, c_l, dl, yl);
          cell_to_metric(r, c_r, dr, yr);
          if (has_prev_edges) {
            const double d_yl = yl - prev_yl;
            const double d_yr = yr - prev_yr;
            if (!glare_left && !glare_right) {
              if (d_yl > 0.16 && std::abs(d_yr) < 0.03) {
                glare_left = true;
              } else if (-d_yr > 0.16 && std::abs(d_yl) < 0.03) {
                glare_right = true;
              }
            }
            if (glare_left) {
              const double max_yl = yr + config_.nominal_lane_width_m + 0.04;
              for (int c = c_l; c <= c_r; ++c) {
                double dd = 0.0, yy = 0.0;
                cell_to_metric(r, c, dd, yy);
                if (yy > max_yl && fused_grid_[static_cast<size_t>(row_off + c)] == BEV_FREE) {
                  blocked_branch_grid_[static_cast<size_t>(row_off + c)] = 1;
                }
              }
              yl = std::min(yl, max_yl);
            } else if (glare_right) {
              const double min_yr = yl - config_.nominal_lane_width_m - 0.04;
              for (int c = c_l; c <= c_r; ++c) {
                double dd = 0.0, yy = 0.0;
                cell_to_metric(r, c, dd, yy);
                if (yy < min_yr && fused_grid_[static_cast<size_t>(row_off + c)] == BEV_FREE) {
                  blocked_branch_grid_[static_cast<size_t>(row_off + c)] = 1;
                }
              }
              yr = std::max(yr, min_yr);
            }
          }
          prev_yl = yl;
          prev_yr = yr;
          has_prev_edges = true;
        }
      }
    }
  }

  // 2) 8-connected Euclidean Dijkstra progress field along unblocked BEV_FREE road
  struct QNode
  {
    float d;
    int idx;
    bool operator>(const QNode & other) const { return d > other.d; }
  };

  auto run_dijkstra = [&](float min_clr_thresh) -> float {
    road_progress_m_.fill(-1.0f);
    std::priority_queue<QNode, std::vector<QNode>, std::greater<QNode>> pq;

    int seed_count = 0;
    for (int r = 0; r < kBevRows; ++r) {
      const double dm = kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes;
      if (dm < -0.08 || dm > 0.16) {
        continue;
      }
      const int row_off = r * kBevCols;
      for (int c = 0; c < kBevCols; ++c) {
        const double ym = kBevYMax - (static_cast<double>(c) + 0.5) * kBevRes;
        if (std::abs(ym) > 0.24) {
          continue;
        }
        const int idx = row_off + c;
        const size_t uidx = static_cast<size_t>(idx);
        if (fused_grid_[uidx] != BEV_FREE || blocked_branch_grid_[uidx] != 0) {
          continue;
        }
        const float clr = std::min(
          dist_wall_m_[uidx], std::min(dist_cone_m_[uidx], dist_no_entry_m_[uidx]));
        if (clr >= min_clr_thresh) {
          road_progress_m_[uidx] = 0.0f;
          pq.push(QNode{0.0f, idx});
          seed_count++;
        }
      }
    }

    if (seed_count == 0) {
      int best_idx = -1;
      double best_d2 = 1e9;
      for (int r = 0; r < kBevRows; ++r) {
        const double dm = kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes;
        if (dm < -0.15 || dm > 0.35) {
          continue;
        }
        for (int c = 0; c < kBevCols; ++c) {
          const int idx = r * kBevCols + c;
          const size_t uidx = static_cast<size_t>(idx);
          if (fused_grid_[uidx] != BEV_FREE || blocked_branch_grid_[uidx] != 0) {
            continue;
          }
          const double ym = kBevYMax - (static_cast<double>(c) + 0.5) * kBevRes;
          const double d2 = (dm - 0.10) * (dm - 0.10) + ym * ym;
          if (d2 < best_d2) {
            best_d2 = d2;
            best_idx = idx;
          }
        }
      }
      if (best_idx >= 0) {
        road_progress_m_[static_cast<size_t>(best_idx)] = 0.0f;
        pq.push(QNode{0.0f, best_idx});
      }
    }

    constexpr int dr8[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    constexpr int dc8[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    constexpr float kOrtho = static_cast<float>(kBevRes);
    constexpr float kDiag = static_cast<float>(kBevRes * 1.41421356);

    float max_prog = 0.0f;
    while (!pq.empty()) {
      const QNode top = pq.top();
      pq.pop();
      const size_t u = static_cast<size_t>(top.idx);
      if (top.d > road_progress_m_[u] + 1e-6f) {
        continue;
      }
      max_prog = std::max(max_prog, top.d);
      const int r = top.idx / kBevCols;
      const int c = top.idx % kBevCols;
      for (int k = 0; k < 8; ++k) {
        const int nr = r + dr8[k];
        const int nc = c + dc8[k];
        if (nr < 0 || nr >= kBevRows || nc < 0 || nc >= kBevCols) {
          continue;
        }
        const int nidx = nr * kBevCols + nc;
        const size_t nu = static_cast<size_t>(nidx);
        if (fused_grid_[nu] != BEV_FREE || blocked_branch_grid_[nu] != 0) {
          continue;
        }
        if (min_clr_thresh > 0.0f) {
          const float clr = std::min(
            dist_wall_m_[nu], std::min(dist_cone_m_[nu], dist_no_entry_m_[nu]));
          if (clr < min_clr_thresh) {
            continue;
          }
        }
        const float step_cost = (k < 4) ? kOrtho : kDiag;
        const float nd = top.d + step_cost;
        if (road_progress_m_[nu] < 0.0f || nd < road_progress_m_[nu]) {
          road_progress_m_[nu] = nd;
          pq.push(QNode{nd, nidx});
        }
      }
    }
    return max_prog;
  };

  // Primary pass blocks sub-car pinholes (< 5.5cm clearance) beside cones/walls; fallback if too pinched
  const float max_prog = run_dijkstra(0.055f);
  if (max_prog < 0.20f) {
    run_dijkstra(0.0f);
  }
}

void BevDwaPlanner::evaluate_two_stage_candidate(
  double kappa_init,
  double kappa1,
  double kappa2,
  double s_split,
  double s_lag,
  bool allow_digit_bonus,
  const std::vector<GroundDetection> & straight_digits,
  bool clean_straight,
  bool forbid_reverse_yaw,
  double reverse_yaw,
  int branch_bias,
  double goal_yaw,
  bool goal_yaw_active,
  BevArcCandidate & out_cand,
  int & out_hit_digit,
  double & out_near_min_clr,
  double & out_clr_start,
  double & out_clr_near_end,
  double & out_soft_ne,
  double & out_soft_wall,
  double & out_soft_cone) const
{
  (void)branch_bias;
  out_cand = BevArcCandidate{};
  out_cand.kappa = kappa1;
  out_cand.kappa2 = kappa2;
  out_cand.s_split_m = s_split;
  out_hit_digit = 0;

  const double r_car = 0.5 * config_.car_width_m;
  const double horizon = std::max(0.65, config_.rollout_horizon_m);
  const double ds = 0.035;
  const int max_steps = std::max(1, static_cast<int>(std::floor((horizon + 1e-6) / ds)));
  constexpr double kBodyOffsets[3] = {0.0, -0.10, -0.22};

  // Exact rear-axle Ackermann kinematics: rear axle at -L_r moves tangent to th;
  // front bumper at +L_r ahead captures inner rear-wheel off-tracking on tight bends.
  const double L_r = std::max(0.05, config_.rear_axle_to_bumper_m);
  const double wall_soft_lim = clean_straight ?
    std::max(0.105, config_.border_margin_m + 0.085) :
    std::max(0.075, config_.border_margin_m + 0.055);
  const double cone_soft_lim = std::max(0.065, config_.cone_margin_m + 0.040);
  const double ne_soft_lim = std::max(0.085, config_.no_entry_margin_m + 0.050);

  double rx = -L_r;
  double ry = 0.0;
  double th = 0.0;

  double safe_len = 0.0;
  double rolled_len = 0.0;
  bool first_hit = false;

  double min_wall_clr = 1.0;
  double min_obs_clr = 1.0;
  double min_eff_clr = 1.0;
  double min_safe_wall_clr = 1.0;
  double min_safe_cone_clr = 1.0;
  double min_safe_ne_clr = 1.0;
  double near_min_clr = 1.0;
  double clr_start = 1.0;
  double clr_near_end = 1.0;

  double soft_wall_sum = 0.0;
  double soft_cone_sum = 0.0;
  double soft_ne_sum = 0.0;
  int soft_steps = 0;

  double best_progress = 0.0;
  double last_safe_progress = 0.0;
  double rec_progress = 0.0;
  double last_safe_th = 0.0;
  double last_eval_th = 0.0;

  double best_digit_bonus = 0.0;
  int best_digit_id = 0;

  for (int step = 1; step <= max_steps; ++step) {
    const double s = static_cast<double>(step) * ds;
    const double prev_s = static_cast<double>(step - 1) * ds;
    const double k_seg = (s <= s_lag + 1e-6) ? kappa_init : ((s <= s_split + 1e-6) ? kappa1 : kappa2);
    const double dth = k_seg * ds;
    rx += ds * std::cos(th + 0.5 * dth);
    ry += ds * std::sin(th + 0.5 * dth);
    th += dth;
    const double x = rx + L_r * std::cos(th);
    const double y = ry + L_r * std::sin(th);

    double dw = 1.0;
    double dc = 1.0;
    double dne = 1.0;
    bool out_front = false;
    const double cth = std::cos(th);
    const double sth = std::sin(th);
    for (int b = 0; b < 3; ++b) {
      const double bx = x + kBodyOffsets[b] * cth;
      const double by = y + kBodyOffsets[b] * sth;
      int r = 0, c = 0;
      if (!metric_to_cell(bx, by, r, c)) {
        if (b == 0) {
          out_front = true;
        }
        continue;
      }
      const size_t idx = static_cast<size_t>(r * kBevCols + c);
      const uint8_t cell = fused_grid_[idx];
      if (bx >= 0.02 && cell != BEV_UNKNOWN) {
        double signed_w = 1.0;
        if (bx >= 0.14 && blocked_branch_grid_[idx] != 0) {
          signed_w = -0.05;
        } else {
          signed_w = (cell == BEV_WALL_BLUE) ?
            -static_cast<double>(dist_free_m_[idx]) :
            static_cast<double>(dist_wall_m_[idx]);
        }
        dw = std::min(dw, signed_w);
      }
      if (bx >= -0.24) {
        dc = std::min(dc, static_cast<double>(dist_cone_m_[idx]));
        dne = std::min(dne, static_cast<double>(dist_no_entry_m_[idx]));
      }
    }
    if (out_front) {
      break;
    }

    rolled_len = s;
    const double w_clr = dw - r_car;
    const double c_clr = dc - r_car;
    const double ne_clr = dne - r_car;
    const double step_eff = std::min({
      w_clr - config_.wall_hard_margin_m,
      c_clr - config_.cone_hard_margin_m,
      ne_clr - config_.no_entry_hard_margin_m});

    if (s <= 0.11) {
      clr_start = std::min(clr_start, step_eff);
    }
    if (s > 0.11 && s <= 0.42) {
      clr_near_end = step_eff;
    }
    if (s <= 0.45) {
      near_min_clr = std::min(near_min_clr, step_eff);
    }

    min_wall_clr = std::min(min_wall_clr, w_clr);
    min_obs_clr = std::min(min_obs_clr, std::min(c_clr, ne_clr));
    min_eff_clr = std::min(min_eff_clr, step_eff);
    last_eval_th = th;

    if (step_eff < 0.0 && !first_hit) {
      first_hit = true;
      safe_len = prev_s;
    }

    // Sample geodesic road progress around (x, y) both while safe and during short recovery
    if (!first_hit || s <= 0.42) {
      int pr = 0, pc = 0;
      if (metric_to_cell(x, y, pr, pc)) {
        double p_here = static_cast<double>(road_progress_m_[static_cast<size_t>(pr * kBevCols + pc)]);
        if (p_here < 0.0) {
          for (int dr = -1; dr <= 1; ++dr) {
            for (int dc_i = -1; dc_i <= 1; ++dc_i) {
              const int rr = pr + dr;
              const int cc = pc + dc_i;
              if (rr >= 0 && rr < kBevRows && cc >= 0 && cc < kBevCols) {
                p_here = std::max(
                  p_here,
                  static_cast<double>(road_progress_m_[static_cast<size_t>(rr * kBevCols + cc)]));
              }
            }
          }
        }
        if (p_here >= 0.0) {
          if (!first_hit) {
            last_safe_progress = p_here;
            best_progress = std::max(best_progress, p_here);
          } else {
            rec_progress = std::max(rec_progress, p_here);
          }
        }
      }
    }

    if (!first_hit) {
      safe_len = s;
      last_safe_th = th;
      min_safe_wall_clr = std::min(min_safe_wall_clr, w_clr);
      min_safe_cone_clr = std::min(min_safe_cone_clr, c_clr);
      min_safe_ne_clr = std::min(min_safe_ne_clr, ne_clr);

      if (w_clr < wall_soft_lim) {
        const double pen = (wall_soft_lim - std::max(0.0, w_clr)) / wall_soft_lim;
        soft_wall_sum += pen * pen;
      }
      if (c_clr < cone_soft_lim) {
        const double pen = (cone_soft_lim - std::max(0.0, c_clr)) / cone_soft_lim;
        soft_cone_sum += pen * pen;
      }
      if (ne_clr < ne_soft_lim) {
        const double pen = (ne_soft_lim - std::max(0.0, ne_clr)) / ne_soft_lim;
        soft_ne_sum += pen * pen;
      }
      soft_steps++;

      if (allow_digit_bonus && !straight_digits.empty() && x >= 0.14 && x <= 0.85) {
        for (const auto & gd : straight_digits) {
          const double d2 = (x - gd.d_bumper_m) * (x - gd.d_bumper_m) +
            (y - gd.y_lat_m) * (y - gd.y_lat_m);
          if (d2 < 0.015) {
            const double rew = config_.digit_bonus_weight *
              std::max(0.0, 1.0 - std::sqrt(d2) / 0.125);
            if (rew > best_digit_bonus) {
              best_digit_bonus = rew;
              best_digit_id = gd.digit;
            }
          }
        }
      }
    } else if (s > 0.42) {
      // Already evaluated recovery response [0..0.42]; stop rolling past hard collision
      break;
    }
  }

  const double inv_steps = 1.0 / static_cast<double>(std::max(1, soft_steps));
  const double pw_75 = std::max(0.0, (0.075 - std::max(0.0, min_safe_wall_clr)) / 0.075);
  const double pw_35 = std::max(0.0, (0.035 - std::max(0.0, min_safe_wall_clr)) / 0.035);
  const double pc_65 = std::max(0.0, (0.065 - std::max(0.0, min_safe_cone_clr)) / 0.065);
  const double pne_85 = std::max(0.0, (0.085 - std::max(0.0, min_safe_ne_clr)) / 0.085);

  out_soft_wall = 2.2 * soft_wall_sum * inv_steps + 11.0 * (pw_75 * pw_75) + 28.0 * (pw_35 * pw_35);
  out_soft_cone = 6.0 * soft_cone_sum * inv_steps + 12.0 * (pc_65 * pc_65);
  out_soft_ne = 10.0 * soft_ne_sum * inv_steps + 18.0 * (pne_85 * pne_85);

  const double progress_m = 0.70 * last_safe_progress + 0.30 * best_progress;
  const bool hard_ok = (!first_hit && rolled_len >= std::min(0.32, horizon - 0.15));

  double cost_branch = 0.0;
  double cost_heading = 0.0;
  const double th_ref = hard_ok ? last_safe_th : last_eval_th;
  const double th_proj = wrap_angle_rad(rel_yaw_rad_ + th_ref);
  if (forbid_reverse_yaw) {
    const double rev_align = std::cos(wrap_angle_rad(th_proj - reverse_yaw));
    if (rev_align > 0.45) {
      cost_branch = config_.wrong_way_penalty * ((rev_align - 0.45) / 0.55);
    }
  }
  if (phase_ == TopologyPhase::FORK_T_RIGHT_TURN) {
    cost_branch += 2.5 * std::max(0.0, kappa1 + 0.35) + 2.5 * std::max(0.0, kappa2 + 0.35);
  }
  if (goal_yaw_active) {
    const double goal_err = std::abs(wrap_angle_rad(th_proj - goal_yaw));
    cost_heading = 0.45 * goal_err;
  }

  const double smooth_pen =
    0.07 * std::abs(kappa1 - prev_kappa_) +
    0.03 * std::abs(kappa2 - kappa1) +
    0.025 * std::abs(kappa1);

  const double soft_total = out_soft_wall + out_soft_cone + out_soft_ne;
  double score = 0.0;
  if (hard_ok) {
    score = 20.0 +
      6.5 * progress_m +
      2.2 * safe_len -
      soft_total -
      cost_heading -
      cost_branch +
      best_digit_bonus -
      smooth_pen;
  } else {
    // Lexicographic recovery: reward continuous clearance gain & getting back onto BEV_FREE
    const double eff_prog = std::max(progress_m, rec_progress);
    const double rec_delta = clr_near_end - clr_start;
    score = 16.0 * clr_near_end +
      10.0 * rec_delta +
      8.0 * min_eff_clr +
      5.5 * safe_len +
      4.0 * eff_prog -
      0.5 * soft_total -
      3.5 * cost_heading -
      cost_branch -
      0.3 * smooth_pen;
  }

  out_cand.progress_m = progress_m;
  out_cand.safe_len_m = safe_len;
  out_cand.min_wall_clr_m = min_wall_clr;
  out_cand.min_obs_clr_m = min_obs_clr;
  out_cand.score = score;
  out_cand.hard_feasible = hard_ok;
  out_cand.cost_safe_len = progress_m;
  out_cand.cost_soft = soft_total;
  out_cand.cost_heading = cost_heading;
  out_cand.cost_branch = cost_branch;
  out_cand.cost_digit = best_digit_bonus;
  out_cand.cost_clearance = min_eff_clr;
  out_cand.cost_smooth = smooth_pen;

  out_hit_digit = best_digit_id;
  out_near_min_clr = near_min_clr;
  out_clr_start = clr_start;
  out_clr_near_end = clr_near_end;
}

BevDwaPlanResult BevDwaPlanner::step(
  const std::array<uint8_t, kBevCells> & live_grid,
  const VehiclePose2D & pose,
  const std::vector<GroundDetection> & valid_digits,
  bool yellow_ahead,
  bool yellow_dist_valid,
  double yellow_d_bumper_m,
  double yellow_speed_cap_mps,
  bool is_holding_yellow_stop,
  double now_sec)
{
  BevDwaPlanResult res{};

  double dt = 0.033;
  if (std::isfinite(now_sec) && last_step_sec_ > -1e8) {
    const double raw_dt = now_sec - last_step_sec_;
    if (raw_dt > 0.001 && raw_dt < 0.35) {
      dt = raw_dt;
    }
  }

  // 1) Warp short-window EKF BEV obstacle memory & overlay current live frame
  warp_and_fuse_bev(live_grid, pose, dt);
  filter_connected_free_space();
  compute_distance_fields();

  // 2) Measure BEV lane geometry & update self-zeroing heading compass + topology phase
  double width_m = config_.nominal_lane_width_m;
  double center_y_m = 0.0;
  double visual_yaw_rad = 0.0;
  bool clean_straight = false;
  double straight_free_len_m = 0.0;
  bool t_junction_right_open = false;
  bool left_branch_open = false;
  bool right_branch_open = false;
  measure_bev_lane_geometry(
    width_m, center_y_m, visual_yaw_rad, clean_straight,
    straight_free_len_m, t_junction_right_open, left_branch_open, right_branch_open);

  update_heading_and_topology(
    pose, dt, width_m, visual_yaw_rad, clean_straight,
    straight_free_len_m, t_junction_right_open, is_holding_yellow_stop);

  // Update dual visual+distance yellow rearm state
  update_yellow_visual_state(yellow_ahead, yellow_dist_valid, yellow_d_bumper_m);

  // Check finish line directly in BEV (zero yellow-count dependence!)
  if (yellow_ahead && yellow_dist_valid) {
    res.visual_dead_end_ahead = check_bev_dead_end_ahead(yellow_d_bumper_m);
    const bool yaw_armed = (cum_abs_yaw_rad_ >= config_.finish_min_cum_yaw_rad);
    res.is_finish_line = (res.visual_dead_end_ahead && yaw_armed && total_arc_m_ >= 6.5) ||
      (total_arc_m_ >= config_.finish_force_dist_m);
  }

  // Update fork votes during yellow stop or wide split approach
  if (is_holding_yellow_stop || width_m > 0.54 || left_branch_open || right_branch_open) {
    update_fork_votes(valid_digits, is_holding_yellow_stop);
  }

  // Autonomous Fork / Roundabout entry backup (decoupled from yellow_stop_end!)
  // If yellow line was missed in shadow/dim lighting while YOLO already collected votes (or the car
  // physically enters the wide fork/roundabout throat), commit cleanly so the car NEVER stays stuck
  // in CRUISE_START / RB1_APPROACH!
  const bool waiting_decision_phase =
    (phase_ == TopologyPhase::CRUISE_START) ||
    (phase_ == TopologyPhase::RB1_APPROACH) ||
    (phase_ == TopologyPhase::RB2_APPROACH);
  if (waiting_decision_phase && !is_holding_yellow_stop && !yellow_ahead) {
    const bool split_open = (width_m > 0.56) || (left_branch_open && right_branch_open);
    const bool votes_ready =
      (best_left_votes_ >= config_.fork_min_votes) ||
      (best_right_votes_ >= config_.fork_min_votes);
    if (split_open && votes_ready) {
      finalize_fork_decision();
    } else if (split_open) {
      if (config_.track_mode == "roundabout") {
        if (phase_ == TopologyPhase::RB1_APPROACH && total_arc_m_ > 0.35 && width_m > 0.60) {
          committed_branch_ = +1;
          phase_ = TopologyPhase::RB1_ACTIVE;
          phase_arc_m_ = 0.0;
          phase_abs_yaw_rad_ = 0.0;
        } else if (phase_ == TopologyPhase::RB2_APPROACH && phase_arc_m_ > 0.35 && width_m > 0.60) {
          committed_branch_ = +1;
          phase_ = TopologyPhase::RB2_ACTIVE;
          phase_arc_m_ = 0.0;
          phase_abs_yaw_rad_ = 0.0;
        }
      } else if (phase_ == TopologyPhase::CRUISE_START && total_arc_m_ > 1.80 && width_m > 0.60) {
        committed_branch_ = +1;
        phase_ = TopologyPhase::FORK_RIGHT_ENTER;
        phase_arc_m_ = 0.0;
        phase_abs_yaw_rad_ = 0.0;
      }
    }
  }

  res.phase = phase_;
  res.committed_branch = committed_branch_;
  res.rel_yaw_rad = rel_yaw_rad_;
  res.measured_lane_width_m = width_m;
  res.straight_free_len_m = straight_free_len_m;
  res.fork_left_digit = best_left_digit_;
  res.fork_left_votes = best_left_votes_;
  res.fork_right_digit = best_right_digit_;
  res.fork_right_votes = best_right_votes_;

  // Update pose bookmark
  if (pose.valid) {
    prev_pose_ = pose;
    has_prev_pose_ = true;
  }
  if (std::isfinite(now_sec)) {
    last_step_sec_ = now_sec;
  }

  // If currently holding at yellow stop line, zero velocity and skip planning
  if (is_holding_yellow_stop) {
    last_speed_cmd_ = 0.0;
    stall_accum_sec_ = 0.0;
    escape_active_ = false;
    res.mode_tag = "bev_yellow_stop";
    return res;
  }

  // 3) Determine Topology Compass target heading & reverse-forbidden heading
  double goal_yaw = 0.0;
  bool goal_yaw_active = false;
  bool forbid_reverse_yaw = false;
  double reverse_yaw = kPi;
  int branch_bias = 0;

  switch (phase_) {
    case TopologyPhase::CRUISE_START:
      goal_yaw = 0.0;
      goal_yaw_active = false;
      forbid_reverse_yaw = true;
      reverse_yaw = kPi;
      break;
    case TopologyPhase::FORK_RIGHT_ENTER:
      goal_yaw = -0.5 * kPi;
      goal_yaw_active = (phase_arc_m_ >= 0.20);
      forbid_reverse_yaw = true;
      reverse_yaw = 0.5 * kPi;
      branch_bias = +1;
      break;
    case TopologyPhase::FORK_RIGHT_STEM:
      goal_yaw = -0.5 * kPi;
      goal_yaw_active = true;
      forbid_reverse_yaw = true;
      reverse_yaw = 0.5 * kPi;
      break;
    case TopologyPhase::FORK_T_RIGHT_TURN:
      goal_yaw = kPi;
      goal_yaw_active = true;
      forbid_reverse_yaw = true;
      reverse_yaw = 0.0;
      branch_bias = +1;
      break;
    case TopologyPhase::FORK_LEFT_UTURN:
      branch_bias = (committed_branch_ < 0) ? -1 : 0;
      if (std::abs(wrap_angle_rad(rel_yaw_rad_ - kPi)) < 0.85 && cum_abs_yaw_rad_ > 1.8) {
        goal_yaw = kPi;
        goal_yaw_active = true;
        forbid_reverse_yaw = true;
        reverse_yaw = 0.5 * kPi;
      }
      break;
    case TopologyPhase::FORK_S_CURVE:
      forbid_reverse_yaw = true;
      reverse_yaw = 0.0;
      break;
    case TopologyPhase::RB1_APPROACH:
      forbid_reverse_yaw = true;
      reverse_yaw = kPi;
      break;
    case TopologyPhase::RB1_ACTIVE:
      branch_bias = committed_branch_;
      forbid_reverse_yaw = true;
      reverse_yaw = kPi;
      if (phase_abs_yaw_rad_ > 0.75) {
        goal_yaw = 0.0;
        goal_yaw_active = true;
      }
      break;
    case TopologyPhase::RB_UTURN:
      break;
    case TopologyPhase::RB2_APPROACH:
      forbid_reverse_yaw = true;
      reverse_yaw = 0.0;
      break;
    case TopologyPhase::RB2_ACTIVE:
      branch_bias = committed_branch_;
      forbid_reverse_yaw = true;
      reverse_yaw = 0.0;
      if (phase_abs_yaw_rad_ > 0.75) {
        goal_yaw = kPi;
        goal_yaw_active = true;
      }
      break;
    case TopologyPhase::RB2_EXIT:
      goal_yaw = kPi;
      goal_yaw_active = true;
      forbid_reverse_yaw = true;
      reverse_yaw = 0.0;
      break;
  }

  res.goal_yaw_rad = goal_yaw;
  res.goal_yaw_active = goal_yaw_active;

  // 4) Extract obstacle clusters in BEV (d in [-0.24, rollout_horizon_m + 0.10]m)
  const std::vector<ObstacleCluster> obs_clusters = extract_obstacle_clusters();
  bool near_cone = false;
  bool near_no_entry = false;
  for (const auto & cl : obs_clusters) {
    const double y_mid = 0.5 * (cl.y_min + cl.y_max);
    if (cl.d_max >= -0.20 && cl.d_min <= 0.90 && std::abs(y_mid) <= 0.42) {
      if (cl.is_no_entry) {
        near_no_entry = true;
      } else {
        near_cone = true;
      }
    }
  }
  res.has_near_obstacle = near_cone || near_no_entry;
  res.has_no_entry_near = near_no_entry;

  // Gate straight bonus digits: OFF whenever yellow ahead, red obstacle near, or in fork/T transition!
  const bool in_fork_or_t =
    (branch_bias != 0) ||
    (phase_ == TopologyPhase::FORK_RIGHT_ENTER) ||
    (phase_ == TopologyPhase::FORK_T_RIGHT_TURN) ||
    (phase_ == TopologyPhase::RB1_ACTIVE) ||
    (phase_ == TopologyPhase::RB2_ACTIVE);
  const bool yellow_near = yellow_ahead || (yellow_dist_valid && yellow_d_bumper_m < 0.95);
  const bool allow_digit_bonus =
    (!yellow_near) && (!res.has_near_obstacle) && (!in_fork_or_t) &&
    (config_.max_digit_lateral_shift_m > 0.01) && (width_m < 0.58);
  const std::vector<GroundDetection> locked_digits =
    filter_and_lock_straight_digits(valid_digits, allow_digit_bonus);

  // 5) Latency forward-prediction (Smith predictor compensating STM32 Smooth_control lag)
  const double v_meas = (pose.valid && std::isfinite(pose.linear_v)) ?
    pose.linear_v : last_speed_cmd_;
  const double w_meas = (pose.valid && std::isfinite(pose.angular_w)) ?
    (pose.angular_w - (preflight_.gyro_calibrated ? preflight_.gyro_bias_rps : 0.0)) :
    (last_speed_cmd_ * prev_kappa_);
  const double tau = clamp_d(config_.control_latency_s, 0.0, 0.30);
  const double x0 = std::max(0.0, v_meas) * tau;
  const double th0 = clamp_d(w_meas * tau, -0.45, 0.45);
  const double y0 = 0.5 * x0 * std::sin(th0);

  // 6) 1D Corridor extraction retained strictly for whitebox HUD/test telemetry
  const bool allow_glare_guard = (branch_bias == 0) && (phase_ != TopologyPhase::FORK_T_RIGHT_TURN);
  extract_safe_corridor(
    y0, branch_bias, goal_yaw, goal_yaw_active, allow_glare_guard, res.corridor_slices);

  carve_obstacles_into_corridor(obs_clusters, res.corridor_slices);

  int hud_hit_digit = 0;
  double hud_digit_shift_m = 0.0;
  apply_digit_bias_to_corridor(
    locked_digits, allow_digit_bonus, res.corridor_slices, hud_hit_digit, hud_digit_shift_m);

  smooth_corridor_centerline(res.corridor_slices);

  int valid_slices = 0;
  for (const auto & sl : res.corridor_slices) {
    if (sl.valid) {
      valid_slices++;
    }
  }
  res.total_slice_count = static_cast<int>(res.corridor_slices.size());
  res.valid_slice_count = valid_slices;

  double road_kappa = 0.0;
  double lat_err_m = 0.0;
  double heading_err_rad = 0.0;
  double corridor_len_m = 0.0;
  fit_corridor_curve_arc(
    res.corridor_slices, x0, y0, th0,
    road_kappa, lat_err_m, heading_err_rad, corridor_len_m);

  res.road_kappa_fit = road_kappa;
  res.lat_err_m = lat_err_m;
  res.heading_err_rad = heading_err_rad;

  // 7) 2D BEV Geodesic Progress Field + 2-Stage Ackermann State Lattice (kappa1, kappa2)
  compute_geodesic_progress_field(
    branch_bias, goal_yaw, goal_yaw_active, forbid_reverse_yaw, reverse_yaw);

  const double k_left = std::max(0.5, config_.kappa_max_left);
  const double k_right = std::max(0.5, config_.kappa_max_right);
  const double s_lag = std::max(0.0, v_meas) * tau;
  const double kappa_init = (v_meas > 0.12) ?
    clamp_d(w_meas / v_meas, -k_right, k_left) :
    clamp_d(prev_kappa_, -k_right, k_left);

  std::vector<double> k1_vals;
  k1_vals.reserve(14);
  const double neg_fracs[6] = {1.0, 0.82, 0.62, 0.42, 0.24, 0.10};
  const double pos_fracs[6] = {0.10, 0.24, 0.42, 0.62, 0.82, 1.0};
  for (double f : neg_fracs) {
    k1_vals.push_back(-f * k_right);
  }
  k1_vals.push_back(0.0);
  for (double f : pos_fracs) {
    k1_vals.push_back(f * k_left);
  }
  // Include continuous warm-start prev_kappa_ if outside lattice resolution
  bool has_prev_k = false;
  for (double kv : k1_vals) {
    if (std::abs(kv - prev_kappa_) < 0.06) {
      has_prev_k = true;
      break;
    }
  }
  if (!has_prev_k) {
    k1_vals.push_back(clamp_d(prev_kappa_, -k_right, k_left));
  }

  const double k2_vals[9] = {
    -k_right, -0.70 * k_right, -0.40 * k_right, -0.15 * k_right,
    0.0,
    0.15 * k_left, 0.40 * k_left, 0.70 * k_left, k_left
  };

  std::vector<double> split_vals;
  const double s_split_0 = clamp_d(s_lag + 0.16, 0.18, 0.24);
  const double s_split_1 = clamp_d(std::max(s_lag + 0.14, 0.30), 0.26, 0.44);
  const double s_split_2 = clamp_d(std::max(s_lag + 0.30, 0.48), 0.38, 0.62);
  if ((res.has_near_obstacle || std::abs(road_kappa) > 0.8) && s_split_0 + 0.04 < s_split_1) {
    split_vals.push_back(s_split_0);
  }
  split_vals.push_back(s_split_1);
  if (s_split_2 > s_split_1 + 0.08) {
    split_vals.push_back(s_split_2);
  }

  res.candidates.clear();
  res.candidates.reserve(k1_vals.size() * 9U * split_vals.size());

  int best_hard_idx = -1;
  double best_hard_score = -1e18;
  int best_rec_idx = -1;
  double best_rec_score = -1e18;
  int hard_count = 0;

  int best_hard_digit = 0;
  int best_rec_digit = 0;

  for (double s_sp : split_vals) {
    for (double k1 : k1_vals) {
      for (double k2 : k2_vals) {
        BevArcCandidate cand{};
        int cand_hit_digit = 0;
        double near_min_clr = 0.0;
        double clr_start = 0.0;
        double clr_near_end = 0.0;
        double soft_ne = 0.0;
        double soft_wall = 0.0;
        double soft_cone = 0.0;

        evaluate_two_stage_candidate(
          kappa_init, k1, k2, s_sp, s_lag,
          allow_digit_bonus, locked_digits, clean_straight,
          forbid_reverse_yaw, reverse_yaw, branch_bias, goal_yaw, goal_yaw_active,
          cand, cand_hit_digit, near_min_clr, clr_start, clr_near_end,
          soft_ne, soft_wall, soft_cone);

        const int idx = static_cast<int>(res.candidates.size());
        res.candidates.push_back(cand);
        if (cand.hard_feasible) {
          hard_count++;
          if (cand.score > best_hard_score) {
            best_hard_score = cand.score;
            best_hard_idx = idx;
            best_hard_digit = cand_hit_digit;
          }
        }
        if (cand.score > best_rec_score) {
          best_rec_score = cand.score;
          best_rec_idx = idx;
          best_rec_digit = cand_hit_digit;
        }
      }
    }
  }

  const bool has_hard = (best_hard_idx >= 0);
  const int win_idx = has_hard ? best_hard_idx : best_rec_idx;
  BevArcCandidate chosen_snap{};
  int hit_digit = 0;
  if (win_idx >= 0) {
    chosen_snap = res.candidates[static_cast<size_t>(win_idx)];
    hit_digit = has_hard ? best_hard_digit : best_rec_digit;
  } else {
    chosen_snap.kappa = (branch_bias > 0 || phase_ == TopologyPhase::FORK_T_RIGHT_TURN) ?
      -k_right : ((branch_bias < 0) ? k_left : prev_kappa_);
    chosen_snap.kappa2 = chosen_snap.kappa;
    chosen_snap.s_split_m = s_split_1;
  }

  // Topology emergency T-junction right-turn guard when straight corridor is closing fast
  double kappa_cmd = chosen_snap.kappa;
  if (phase_ == TopologyPhase::FORK_T_RIGHT_TURN && straight_free_len_m < 0.42 && !has_hard) {
    kappa_cmd = std::min(kappa_cmd, -1.6);
  }
  const double kappa_star = clamp_d(kappa_cmd, -k_right, k_left);
  prev_kappa_ = kappa_star;

  // Integrate winning 2-stage trajectory out to preview distance for HUD lookahead marker
  const double Ld = clamp_d(std::max(0.24, 0.70 * chosen_snap.safe_len_m), 0.22, 0.55);
  res.lookahead_m = Ld;
  {
    const double L_r = std::max(0.05, config_.rear_axle_to_bumper_m);
    double prx = -L_r, pry = 0.0, pth = 0.0;
    double px = 0.0, py = 0.0;
    const double pds = 0.025;
    for (double s = pds; s <= Ld + 1e-5; s += pds) {
      const double ks = (s <= s_lag) ? kappa_init :
        ((s <= chosen_snap.s_split_m) ? kappa_star : chosen_snap.kappa2);
      const double dth = ks * pds;
      prx += pds * std::cos(pth + 0.5 * dth);
      pry += pds * std::sin(pth + 0.5 * dth);
      pth += dth;
      px = prx + L_r * std::cos(pth);
      py = pry + L_r * std::sin(pth);
    }
    res.lookahead_x_m = px;
    res.lookahead_y_m = py;
  }

  res.kappa_pp = kappa_star;
  res.kappa_lat = lat_err_m * config_.stanley_k_y;
  res.kappa_curv_ff = chosen_snap.kappa2 - kappa_star;
  res.kappa_digit_shift = hud_digit_shift_m;

  res.chosen_kappa = kappa_star;
  res.chosen_kappa2 = chosen_snap.kappa2;
  res.chosen_s_split_m = chosen_snap.s_split_m;
  res.chosen_safe_len_m = std::max(0.0, chosen_snap.safe_len_m);
  res.min_wall_clearance_m = chosen_snap.min_wall_clr_m;
  res.min_obs_clearance_m = chosen_snap.min_obs_clr_m;
  res.has_feasible_arc = has_hard;
  res.feasible_arc_count = hard_count;
  res.digit_target_active = (hit_digit > 0);
  res.target_digit = hit_digit;
  res.chosen_candidate = chosen_snap;

  // 8) Level-3 Anti-Stall Escape (only if physically wedged for > stall_detect_sec away from yellow)
  if (config_.enable_anti_stall_escape && !yellow_ahead) {
    if (escape_active_) {
      escape_elapsed_sec_ += dt;
      if (escape_elapsed_sec_ >= config_.escape_reverse_dur_sec) {
        escape_active_ = false;
        escape_elapsed_sec_ = 0.0;
        stall_accum_sec_ = 0.0;
        last_speed_cmd_ = 0.0;
      } else {
        res.escape_active = true;
        res.linear_speed_mps = -config_.escape_reverse_speed_mps;
        const double rev_kappa = (kappa_star >= 0.0) ? -1.2 : +1.2;
        res.yaw_rate_rps = clamp_d(
          res.linear_speed_mps * rev_kappa, -config_.max_yaw_rate, config_.max_yaw_rate);
        res.mode_tag = "bev_escape_rev";
        return res;
      }
    } else if (pose.valid && last_speed_cmd_ > 0.06 && std::abs(pose.linear_v) < 0.02) {
      stall_accum_sec_ += dt;
      if (stall_accum_sec_ >= config_.stall_detect_sec &&
        escape_retries_used_ < config_.max_escape_retries)
      {
        escape_active_ = true;
        escape_elapsed_sec_ = 0.0;
        escape_retries_used_++;
        res.escape_active = true;
        res.linear_speed_mps = -config_.escape_reverse_speed_mps;
        res.yaw_rate_rps = 0.0;
        res.mode_tag = "bev_escape_rev";
        return res;
      }
    } else {
      stall_accum_sec_ = std::max(0.0, stall_accum_sec_ - 2.0 * dt);
    }
  }

  // 9) Single Clean Physics Speed Envelope (proactive 2-stage curve-arc slowdown!)
  double v_target = config_.cruise_speed_mps;
  if (config_.enable_clear_straight_boost &&
    clean_straight && !res.has_near_obstacle && !yellow_ahead &&
    !in_fork_or_t && corridor_len_m >= 0.72 &&
    std::abs(kappa_star) < 0.35 && std::abs(chosen_snap.kappa2) < 0.45 &&
    std::abs(road_kappa) < 0.35)
  {
    v_target = config_.clear_straight_speed_mps;
  }

  // Centripetal lateral acceleration limit using max(|kappa1|, 0.95*|kappa2|, 0.85*|road_kappa|, 0.65*|kappa_init|)
  // plus wall-clearance speed taper so the car decelerates BEFORE entering tight bends under actuator lag!
  const double k_speed = std::max({
    std::abs(kappa_star),
    0.95 * std::abs(chosen_snap.kappa2),
    0.85 * std::abs(road_kappa),
    0.65 * std::abs(kappa_init)});
  double v_curve = std::sqrt(std::min(config_.max_lat_accel_mps2, 0.24) / (k_speed + 0.15));
  if (std::abs(chosen_snap.kappa2) > 1.6) {
    v_curve = std::min(v_curve, config_.corner_approach_speed_mps);
  } else if (k_speed > 1.25) {
    v_curve = std::min(v_curve, config_.corner_approach_speed_mps + 0.04);
  }
  v_target = std::min(v_target, v_curve);

  if (chosen_snap.min_wall_clr_m < 0.065) {
    const double clr_frac = clamp_d(chosen_snap.min_wall_clr_m / 0.065, 0.0, 1.0);
    const double v_clr = 0.22 + (config_.cruise_speed_mps - 0.22) * clr_frac;
    v_target = std::min(v_target, v_clr);
  }

  // Horizon limit using both rolled safe_len and geodesic progress
  const double eff_horizon = std::max(res.chosen_safe_len_m, 0.80 * chosen_snap.progress_m);
  if (eff_horizon < 0.50) {
    const double v_hor = std::sqrt(2.0 * 0.65 * std::max(0.03, eff_horizon - 0.10));
    v_target = std::min(v_target, std::max(config_.creep_speed_mps, v_hor));
  }
  if (phase_ == TopologyPhase::FORK_T_RIGHT_TURN || phase_ == TopologyPhase::FORK_RIGHT_ENTER) {
    v_target = std::min(v_target, config_.corner_approach_speed_mps);
  }

  // Obstacle & recovery speed caps
  if (near_no_entry) {
    v_target = std::min(v_target, config_.no_entry_speed_mps);
  } else if (near_cone) {
    v_target = std::min(v_target, config_.obstacle_speed_mps);
  }
  if (!res.has_feasible_arc) {
    v_target = std::min(v_target, config_.creep_speed_mps);
  }

  // Yellow line smooth pre-brake envelope
  if (yellow_ahead && is_yellow_rearmed() && std::isfinite(yellow_speed_cap_mps)) {
    v_target = std::min(v_target, std::max(0.0, yellow_speed_cap_mps));
  }

  v_target = clamp_d(v_target, config_.min_speed_mps, config_.max_speed_mps);
  res.speed_cap_mps = v_target;

  // Single monotonic asymmetric slew rate limiter
  const double max_up = last_speed_cmd_ + config_.max_accel_mps2 * dt;
  const double max_dn = last_speed_cmd_ - config_.max_decel_mps2 * dt;
  double v_cmd = clamp_d(v_target, max_dn, max_up);
  v_cmd = clamp_d(v_cmd, 0.0, config_.max_speed_mps);
  last_speed_cmd_ = v_cmd;

  // 10) Real-Car Ackermann Yaw-Rate Execution Layer (Feedforward + Launch Pre-Steer + IMU Inner Loop)
  const double v_for_steer = (v_cmd > 0.01) ? std::max(v_cmd, config_.steer_launch_v_floor) : 0.0;

  const double abs_k = std::abs(kappa_star);
  const double k_ref_max = (kappa_star >= 0.0) ? k_left : k_right;
  const double k_frac = clamp_d(abs_k / std::max(0.5, k_ref_max), 0.0, 1.0);
  const double nl_boost = 1.0 + config_.large_angle_boost_max * std::max(0.0, (k_frac - 0.35) / 0.65);

  double omega_cmd = v_for_steer * kappa_star * nl_boost;

  if (pose.valid && std::isfinite(pose.angular_w) && v_cmd > 0.05 && config_.imu_yaw_kp > 0.0) {
    const double omega_target = v_cmd * kappa_star;
    const double omega_imu = pose.angular_w -
      (preflight_.gyro_calibrated ? preflight_.gyro_bias_rps : 0.0);
    if (k_frac < 0.94) {
      omega_cmd += config_.imu_yaw_kp * (omega_target - omega_imu);
    }
  }

  omega_cmd = clamp_d(omega_cmd, -config_.max_yaw_rate, config_.max_yaw_rate);

  res.linear_speed_mps = v_cmd;
  res.yaw_rate_rps = omega_cmd;

  if (!res.has_feasible_arc) {
    res.mode_tag = "bev_tight_creep";
  } else if (phase_ == TopologyPhase::FORK_T_RIGHT_TURN) {
    res.mode_tag = "bev_T_right";
  } else if (near_no_entry) {
    res.mode_tag = "bev_avoid_ne";
  } else if (near_cone) {
    res.mode_tag = "bev_avoid_cone";
  } else if (branch_bias < 0) {
    res.mode_tag = "bev_fork_L";
  } else if (branch_bias > 0) {
    res.mode_tag = "bev_fork_R";
  } else if (res.digit_target_active) {
    res.mode_tag = "bev_digit";
  } else {
    res.mode_tag = "bev_cruise";
  }

  return res;
}

}  // namespace line_follower
