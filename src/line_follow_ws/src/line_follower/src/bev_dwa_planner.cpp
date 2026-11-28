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
  left_digit_votes_.fill(0);
  right_digit_votes_.fill(0);
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
  const int vote_weight = is_stationary_at_yellow ? 2 : 1;
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
    if (gd.y_lat_m > 0.025) {
      left_digit_votes_[static_cast<size_t>(gd.digit)] += vote_weight;
    } else if (gd.y_lat_m < -0.025) {
      right_digit_votes_[static_cast<size_t>(gd.digit)] += vote_weight;
    }
  }

  best_left_digit_ = 0;
  best_left_votes_ = 0;
  best_right_digit_ = 0;
  best_right_votes_ = 0;
  for (int d = 1; d <= 8; ++d) {
    const int lv = left_digit_votes_[static_cast<size_t>(d)];
    if (lv > best_left_votes_) {
      best_left_votes_ = lv;
      best_left_digit_ = d;
    }
    const int rv = right_digit_votes_[static_cast<size_t>(d)];
    if (rv > best_right_votes_) {
      best_right_votes_ = rv;
      best_right_digit_ = d;
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
    // Roundabout mode: distinguish RB1 (heading near 0) vs RB2 (heading near +/-pi)
    if (phase_ == TopologyPhase::RB1_APPROACH) {
      phase_ = TopologyPhase::RB1_ACTIVE;
      phase_arc_m_ = 0.0;
      phase_abs_yaw_rad_ = 0.0;
    } else if (phase_ == TopologyPhase::RB2_APPROACH || phase_ == TopologyPhase::RB_UTURN) {
      phase_ = TopologyPhase::RB2_ACTIVE;
      phase_arc_m_ = 0.0;
      phase_abs_yaw_rad_ = 0.0;
    }
  }

  return chosen;
}

void BevDwaPlanner::notify_yellow_stop_triggered()
{
  dist_since_yellow_m_ = 0.0;
  yellow_absent_frames_ = 0;
  yellow_visual_exited_ = false;
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
  const double d0 = std::max(0.05, d_yellow_m + 0.14);
  const double d1 = std::min(kBevDMax - 0.02, d_yellow_m + 0.44);
  if (d1 <= d0 + 0.05) {
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
        // Transition to vertical stem only after truly rotating North (<-64 deg) AND traveling >= 0.40m
        if ((err_north < 0.45 && width_m < 0.56 && phase_arc_m_ >= 0.40) || phase_arc_m_ > 0.95) {
          phase_ = TopologyPhase::FORK_RIGHT_STEM;
          committed_branch_ = 0;
          phase_arc_m_ = 0.0;
        }
        break;
      case TopologyPhase::FORK_RIGHT_STEM:
        // T-JUNCTION DETECTION: must have driven >= 0.25m up vertical stem facing North (< 29 deg)
        // AND genuinely see top horizontal bar (never false-trigger on L=0 wall-graze!)
        if (err_north < 0.50 && phase_arc_m_ >= 0.25 &&
          ((t_junction_right_open && straight_free_len_m < 0.68) ||
          (straight_free_len_m >= 0.08 && straight_free_len_m < 0.55)))
        {
          phase_ = TopologyPhase::FORK_T_RIGHT_TURN;
          phase_arc_m_ = 0.0;
        }
        break;
      case TopologyPhase::FORK_T_RIGHT_TURN:
        // Completed 90 deg right turn into East-bound S-curve
        if (err_east < 0.55) {
          phase_ = TopologyPhase::FORK_S_CURVE;
          phase_arc_m_ = 0.0;
        }
        break;
      case TopologyPhase::FORK_LEFT_UTURN:
        if (phase_arc_m_ > 0.55 && width_m < 0.54) {
          committed_branch_ = 0;  // Release initial fork entry bias once inside outer lane
        }
        // After sweeping through outer U-turn to East (+180 deg), enter S-curve past T-junction
        if (err_east < 0.60 && cum_abs_yaw_rad_ > 2.0) {
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
        // Release initial branch bias after entering the ring
        if (phase_arc_m_ > 0.55 && width_m < 0.56) {
          committed_branch_ = 0;
        }
        // Exited RB1 once swept > 60 deg around island and returned to West (0 deg) narrow straight
        if (phase_abs_yaw_rad_ > 1.1 && err_west < 0.45 && width_m < 0.54 && phase_arc_m_ > 1.2) {
          phase_ = TopologyPhase::RB_UTURN;
          committed_branch_ = 0;
          phase_arc_m_ = 0.0;
          phase_abs_yaw_rad_ = 0.0;
        }
        break;
      case TopologyPhase::RB_UTURN:
        // Completed big 180 deg U-turn onto East-bound straight approaching RB2
        if (err_east < 0.55 && cum_abs_yaw_rad_ > 2.2) {
          phase_ = TopologyPhase::RB2_APPROACH;
          phase_arc_m_ = 0.0;
          phase_abs_yaw_rad_ = 0.0;
        }
        break;
      case TopologyPhase::RB2_APPROACH:
        break;
      case TopologyPhase::RB2_ACTIVE:
        if (phase_arc_m_ > 0.55 && width_m < 0.56) {
          committed_branch_ = 0;
        }
        if (phase_abs_yaw_rad_ > 1.1 && err_east < 0.45 && width_m < 0.54 && phase_arc_m_ > 1.2) {
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
    res.is_finish_line = (res.visual_dead_end_ahead && yaw_armed) ||
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

  // 6) Unified 1D Safe Corridor + Obstacle S-Curve Carving + Digit Clamping + Curve-Arc Pure Pursuit
  const bool allow_glare_guard = (branch_bias == 0) && (phase_ != TopologyPhase::FORK_T_RIGHT_TURN);
  extract_safe_corridor(
    y0, branch_bias, goal_yaw, goal_yaw_active, allow_glare_guard, res.corridor_slices);

  carve_obstacles_into_corridor(obs_clusters, res.corridor_slices);

  int hit_digit = 0;
  double digit_shift_m = 0.0;
  apply_digit_bias_to_corridor(
    locked_digits, allow_digit_bonus, res.corridor_slices, hit_digit, digit_shift_m);

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

  // Curvature-adaptive Regulated Pure Pursuit lookahead distance
  double ld_raw =
    (config_.lookahead_min_m + config_.lookahead_time_s * std::max(0.0, v_meas)) /
    (1.0 + 0.30 * std::abs(road_kappa));
  if (res.has_near_obstacle) {
    ld_raw = std::min(ld_raw, 0.38);
  }
  if (phase_ == TopologyPhase::FORK_T_RIGHT_TURN || phase_ == TopologyPhase::FORK_RIGHT_ENTER) {
    ld_raw = std::min(ld_raw, 0.36);
  }
  const double ld_lo = std::min(0.18, config_.lookahead_min_m);
  const double ld_hi = std::min(0.72, std::max(0.22, corridor_len_m));
  const double Ld = clamp_d(ld_raw, ld_lo, ld_hi);
  res.lookahead_m = Ld;

  double kappa_pp = 0.0;
  double kappa_lat = 0.0;
  double kappa_curv_ff = 0.0;
  double kappa_cmd = 0.0;
  double tx = x0 + Ld;
  double ty = 0.0;

  if (valid_slices >= 3) {
    // Interpolate lookahead target (tx, ty) along the smoothed safe corridor centerline
    const double d_query = x0 + Ld;
    int idx_lo = -1;
    int idx_hi = -1;
    int first_valid = -1;
    int last_valid = -1;
    for (size_t i = 0; i < res.corridor_slices.size(); ++i) {
      if (!res.corridor_slices[i].valid) {
        continue;
      }
      if (first_valid < 0) {
        first_valid = static_cast<int>(i);
      }
      last_valid = static_cast<int>(i);
      if (res.corridor_slices[i].d_m <= d_query) {
        idx_lo = static_cast<int>(i);
      } else if (idx_hi < 0) {
        idx_hi = static_cast<int>(i);
      }
    }

    if (idx_lo >= 0 && idx_hi >= 0) {
      const auto & a = res.corridor_slices[static_cast<size_t>(idx_lo)];
      const auto & b = res.corridor_slices[static_cast<size_t>(idx_hi)];
      const double alpha = clamp_d(
        (d_query - a.d_m) / std::max(1e-4, b.d_m - a.d_m), 0.0, 1.0);
      tx = a.d_m + alpha * (b.d_m - a.d_m);
      ty = a.y_target + alpha * (b.y_target - a.y_target);
    } else if (last_valid >= 0) {
      const auto & b = res.corridor_slices[static_cast<size_t>(last_valid)];
      tx = b.d_m;
      ty = b.y_target;
    } else if (first_valid >= 0) {
      const auto & a = res.corridor_slices[static_cast<size_t>(first_valid)];
      tx = a.d_m;
      ty = a.y_target;
    }

    // Transform target into Smith-predicted bumper frame (x0, y0, th0)
    const double dx_loc = (tx - x0) * std::cos(th0) + (ty - y0) * std::sin(th0);
    const double dy_loc = -(tx - x0) * std::sin(th0) + (ty - y0) * std::cos(th0);
    const double L2 = std::max(0.04, dx_loc * dx_loc + dy_loc * dy_loc);
    kappa_pp = (2.0 * dy_loc) / L2;

    // Near-bumper Stanley cross-track centering pull (evaluated in Smith-predicted frame)
    kappa_lat = clamp_d(config_.stanley_k_y * lat_err_m / std::max(0.28, Ld), -1.6, 1.6);

    // Curve-Arc Curvature Feedforward: feeds forward excess road arc curvature ahead WITHOUT
    // double-counting on steady circles (where road_kappa == kappa_pp -> excess == 0)!
    const double k_excess = road_kappa - kappa_pp;
    const double obs_scale = res.has_near_obstacle ? 0.35 : 1.0;
    if (std::abs(road_kappa) > 0.25 && k_excess * road_kappa > 0.0) {
      kappa_curv_ff = clamp_d(obs_scale * config_.curv_ff_gain * k_excess, -1.6, 1.6);
    } else if (std::abs(road_kappa) <= 0.25 && std::abs(lat_err_m) < 0.06) {
      // Gentle straightening on bend exit onto a straight
      kappa_curv_ff = clamp_d(0.30 * obs_scale * config_.curv_ff_gain * k_excess, -0.6, 0.6);
    }

    kappa_cmd = kappa_pp + kappa_lat + kappa_curv_ff;
  } else {
    // Blind / extreme off-track fallback: steer toward closest BEV_FREE cell (never deadlock!)
    int best_r = -1, best_c = -1;
    double best_d2 = 1e9;
    for (int r = 0; r < kBevRows; ++r) {
      const double dm = kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes;
      if (dm < 0.02 || dm > 0.90) {
        continue;
      }
      for (int c = 0; c < kBevCols; ++c) {
        if (fused_grid_[static_cast<size_t>(r * kBevCols + c)] != BEV_FREE) {
          continue;
        }
        const double ym = kBevYMax - (static_cast<double>(c) + 0.5) * kBevRes;
        const double d2 = (dm - 0.25) * (dm - 0.25) + ym * ym;
        if (d2 < best_d2) {
          best_d2 = d2;
          best_r = r;
          best_c = c;
        }
      }
    }
    if (best_r >= 0) {
      cell_to_metric(best_r, best_c, tx, ty);
      kappa_pp = (ty >= 0.0) ? config_.kappa_max_left : -config_.kappa_max_right;
      kappa_cmd = kappa_pp;
    } else if (phase_ == TopologyPhase::FORK_T_RIGHT_TURN || branch_bias > 0) {
      kappa_cmd = -config_.kappa_max_right;
    } else if (branch_bias < 0) {
      kappa_cmd = config_.kappa_max_left;
    } else {
      kappa_cmd = prev_kappa_;
    }
  }

  // Topology Compass guard: prevent T-junction wrong-way turn or reverse-looping
  if (phase_ == TopologyPhase::FORK_T_RIGHT_TURN && straight_free_len_m < 0.55) {
    kappa_cmd = std::min(kappa_cmd, -1.5);
  }
  if (forbid_reverse_yaw) {
    const double th_proj = wrap_angle_rad(rel_yaw_rad_ + th0 + kappa_cmd * 0.45);
    const double rev_align = std::cos(wrap_angle_rad(th_proj - reverse_yaw));
    if (rev_align > 0.55) {
      const double err_to_goal = goal_yaw_active ?
        wrap_angle_rad(goal_yaw - rel_yaw_rad_) :
        -wrap_angle_rad(reverse_yaw - rel_yaw_rad_);
      const double nudge = (err_to_goal >= 0.0) ? +1.4 : -1.4;
      kappa_cmd = 0.5 * kappa_cmd + 0.5 * nudge;
    }
  }

  // Clamp to real measured asymmetric Ackermann limits [-kappa_max_right, +kappa_max_left]
  const double kappa_star = clamp_d(kappa_cmd, -config_.kappa_max_right, config_.kappa_max_left);
  const double delta_kappa = kappa_star - prev_kappa_;
  prev_kappa_ = kappa_star;

  // Single fast 3-circle clearance check along chosen kappa_star for whitebox telemetry & speed cap
  double safe_len_m = 0.0;
  double min_wall_clr_m = 0.20;
  double min_obs_clr_m = 1.0;
  check_chosen_arc_clearance(
    x0, y0, th0, kappa_star, safe_len_m, min_wall_clr_m, min_obs_clr_m);

  res.lookahead_x_m = tx;
  res.lookahead_y_m = ty;
  res.kappa_pp = kappa_pp;
  res.kappa_lat = kappa_lat;
  res.kappa_curv_ff = kappa_curv_ff;
  res.kappa_digit_shift = digit_shift_m;

  res.chosen_kappa = kappa_star;
  res.chosen_safe_len_m = std::max(safe_len_m, 0.0);
  res.min_wall_clearance_m = min_wall_clr_m;
  res.min_obs_clearance_m = min_obs_clr_m;
  res.has_feasible_arc = (valid_slices >= 3);
  res.feasible_arc_count = valid_slices;
  res.digit_target_active = (hit_digit > 0);
  res.target_digit = hit_digit;

  // Populate chosen_candidate & candidates[0] for unified whitebox telemetry / HUD compatibility
  BevArcCandidate chosen_snap{};
  chosen_snap.kappa = kappa_star;
  chosen_snap.safe_len_m = res.chosen_safe_len_m;
  chosen_snap.min_wall_clr_m = min_wall_clr_m;
  chosen_snap.min_obs_clr_m = min_obs_clr_m;
  chosen_snap.hard_feasible = res.has_feasible_arc;
  chosen_snap.cost_safe_len = road_kappa;
  chosen_snap.cost_soft = kappa_pp;
  chosen_snap.cost_heading = kappa_lat;
  chosen_snap.cost_branch = kappa_curv_ff;
  chosen_snap.cost_digit = digit_shift_m;
  chosen_snap.cost_clearance = lat_err_m;
  chosen_snap.cost_smooth = delta_kappa;
  chosen_snap.score = kappa_star;
  res.chosen_candidate = chosen_snap;
  res.candidates.clear();
  res.candidates.push_back(chosen_snap);

  // 7) Level-3 Anti-Stall Escape (only if physically wedged for > stall_detect_sec away from yellow)
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

  // 8) Single Clean Physics Speed Envelope (proactive curve-arc slowdown!)
  double v_target = config_.cruise_speed_mps;
  if (config_.enable_clear_straight_boost &&
    clean_straight && !res.has_near_obstacle && !yellow_ahead &&
    !in_fork_or_t && corridor_len_m >= 0.72 &&
    std::abs(kappa_star) < 0.35 && std::abs(road_kappa) < 0.35)
  {
    v_target = config_.clear_straight_speed_mps;
  }

  // Centripetal lateral acceleration limit using max(|kappa_cmd|, 0.9*|road_kappa_fit|)
  // so the car slows down BEFORE entering sharp 90 deg bends!
  const double k_speed = std::max(std::abs(kappa_star), 0.90 * std::abs(road_kappa));
  const double v_curve = std::sqrt(config_.max_lat_accel_mps2 / (k_speed + 0.15));
  v_target = std::min(v_target, v_curve);

  // Corridor horizon limit (uses actual curved corridor length so sharp curves don't choke to 0.14m/s!)
  const double eff_horizon = std::max(res.chosen_safe_len_m, 0.85 * corridor_len_m);
  if (eff_horizon < 0.55) {
    const double v_hor = std::sqrt(2.0 * 0.65 * std::max(0.03, eff_horizon - 0.10));
    v_target = std::min(v_target, std::max(config_.creep_speed_mps, v_hor));
  }
  if (phase_ == TopologyPhase::FORK_T_RIGHT_TURN || phase_ == TopologyPhase::FORK_RIGHT_ENTER) {
    v_target = std::min(v_target, config_.corner_approach_speed_mps);
  }

  // Obstacle speed caps
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

  // 9) Real-Car Ackermann Yaw-Rate Execution Layer (Feedforward + Launch Pre-Steer + IMU Inner Loop)
  const double v_for_steer = (v_cmd > 0.01) ? std::max(v_cmd, config_.steer_launch_v_floor) : 0.0;

  // Gentle large-angle nonlinearity boost (compensates measured ~20% under-turn above 10 deg)
  const double abs_k = std::abs(kappa_star);
  const double k_ref_max = (kappa_star >= 0.0) ? config_.kappa_max_left : config_.kappa_max_right;
  const double k_frac = clamp_d(abs_k / std::max(0.5, k_ref_max), 0.0, 1.0);
  const double nl_boost = 1.0 + config_.large_angle_boost_max * std::max(0.0, (k_frac - 0.35) / 0.65);

  double omega_cmd = v_for_steer * kappa_star * nl_boost;

  // IMU gyro inner feedback loop (cancels mechanical right offset & servo under-turn!)
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

  // Concise human-readable mode tag
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
