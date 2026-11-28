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

    // Only keep short-window rolling memory in the near/rear blind-spot zone (d in [-0.35, +0.35]m)
    for (int r = 0; r < kBevRows; ++r) {
      const double d_cur = kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes;
      if (d_cur > 0.38) {
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
          const float new_age = mem_age_[pidx] + dt_f;
          if (mem_grid_[pidx] != BEV_UNKNOWN && new_age <= ttl_f) {
            const size_t idx = static_cast<size_t>(r * kBevCols + c);
            warped_grid[idx] = mem_grid_[pidx];
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

void BevDwaPlanner::evaluate_arc(
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
  int & out_hit_digit) const
{
  out_cand = BevArcCandidate{};
  out_cand.kappa = kappa;
  out_hit_digit = 0;

  const double r_car = 0.5 * config_.car_width_m;  // 0.0955 m
  const double wall_hard = r_car + config_.wall_hard_margin_m;
  const double cone_hard = r_car + config_.cone_hard_margin_m;
  const double ne_hard = r_car + config_.no_entry_hard_margin_m;

  const double wall_soft = r_car + config_.wall_soft_margin_m;
  const double cone_soft = r_car + config_.cone_soft_margin_m;
  const double ne_soft = r_car + config_.no_entry_soft_margin_m;

  // 3 circles along the 0.280m chassis behind front bumper (protects front bumper & rear inner wheel!)
  constexpr double kBodyOffsets[3] = {-0.04, -0.14, -0.24};

  const double ds = config_.rollout_step_m;
  const int max_steps = static_cast<int>(std::ceil(config_.rollout_horizon_m / ds));

  double x = x0;
  double y = y0;
  double th = th0;

  double safe_len = 0.0;
  double min_wall_clr = 1.0;
  double min_obs_clr = 1.0;
  double soft_cost_sum = 0.0;
  int eval_steps = 0;
  int recovery_free_steps = 0;
  double y_at_mid = y0;
  bool mid_captured = false;
  bool hit_hard = false;

  // Signed Distance Field (SDF) evaluation for 3-circle chassis:
  //   Inside white road (BEV_FREE):      dw = +dist_wall_m_ >= 0
  //   Inside blue wall  (BEV_WALL_BLUE): dw = -dist_free_m_ <= 0 (smooth gradient back to white road!)
  auto eval_body_dists = [&](double px, double py, double pth,
                             double & dw, double & dw_mean,
                             double & dc, double & dne, bool & out_front) {
    dw = 1.0;
    double w_acc = 0.0;
    int w_cnt = 0;
    dc = 1.0;
    dne = 1.0;
    out_front = false;
    const double cth_l = std::cos(pth);
    const double sth_l = std::sin(pth);
    for (int b = 0; b < 3; ++b) {
      const double bx = px + kBodyOffsets[b] * cth_l;
      const double by = py + kBodyOffsets[b] * sth_l;
      int r = 0, c = 0;
      if (!metric_to_cell(bx, by, r, c)) {
        if (b == 0) {
          out_front = true;
        }
        continue;
      }
      const size_t idx = static_cast<size_t>(r * kBevCols + c);
      const uint8_t cell = fused_grid_[idx];
      if (b > 0 && cell == BEV_UNKNOWN && bx < 0.02) {
        continue;
      }
      const double signed_w = (cell == BEV_WALL_BLUE) ?
        -static_cast<double>(dist_free_m_[idx]) :
        static_cast<double>(dist_wall_m_[idx]);
      dw = std::min(dw, signed_w);
      w_acc += signed_w;
      w_cnt++;
      dc = std::min(dc, static_cast<double>(dist_cone_m_[idx]));
      dne = std::min(dne, static_cast<double>(dist_no_entry_m_[idx]));
    }
    dw_mean = (w_cnt > 0) ? (w_acc / static_cast<double>(w_cnt)) : dw;
  };

  double prev_wall_d = 1.0, prev_wall_mean = 1.0, prev_cone_d = 1.0, prev_ne_d = 1.0;
  bool dummy_out = false;
  eval_body_dists(x0, y0, th0, prev_wall_d, prev_wall_mean, prev_cone_d, prev_ne_d, dummy_out);
  (void)prev_wall_mean;

  // Digit bonus & anti-double-cover tracking along this arc
  double digit_bonus_score = 0.0;
  int highest_digit_val = 0;
  for (const auto & gd : straight_digits) {
    if (gd.digit > highest_digit_val) {
      highest_digit_val = gd.digit;
    }
  }

  for (int step = 1; step <= max_steps; ++step) {
    // Exact bicycle/Ackermann arc step
    const double dth = kappa * ds;
    const double th_mid = th + 0.5 * dth;
    x += ds * std::cos(th_mid);
    y += ds * std::sin(th_mid);
    th += dth;

    if (!mid_captured && step * ds >= 0.42) {
      y_at_mid = y;
      mid_captured = true;
    }

    double step_wall_d = 1.0;
    double step_wall_mean = 1.0;
    double step_cone_d = 1.0;
    double step_ne_d = 1.0;
    bool out_of_bev = false;
    eval_body_dists(x, y, th, step_wall_d, step_wall_mean, step_cone_d, step_ne_d, out_of_bev);

    if (out_of_bev) {
      break;
    }

    const double obs_d = std::min(step_cone_d, step_ne_d);
    const bool wall_bad = (step_wall_d < wall_hard) &&
      !(step <= 6 && step_wall_d >= prev_wall_d - 0.002 && step_wall_d >= r_car - 0.035);
    const bool cone_bad = (step_cone_d < cone_hard) &&
      !(step <= 6 && step_cone_d >= prev_cone_d - 0.002 && step_cone_d >= r_car - 0.030);
    const bool ne_bad = (step_ne_d < ne_hard) &&
      !(step <= 6 && step_ne_d >= prev_ne_d - 0.002 && step_ne_d >= r_car - 0.030);
    prev_wall_d = step_wall_d;
    prev_cone_d = step_cone_d;
    prev_ne_d = step_ne_d;

    if (!hit_hard && (wall_bad || cone_bad || ne_bad)) {
      hit_hard = true;
      // If the arc already ran cleanly >= 0.15m in white road, stop at the boundary
      // (a constant-curvature circle naturally hits the far wall on sharp bends/T-junctions!).
      // Only when the car is ALREADY grazing/inside a wall (safe_len < 0.15m) do we keep
      // rolling out full 0.85m through the Signed Distance Field to pull back onto white road!
      if (safe_len >= 0.15) {
        break;
      }
    }

    if (!hit_hard) {
      safe_len = static_cast<double>(step) * ds;
    } else if (step_wall_d >= wall_hard && step_cone_d >= cone_hard && step_ne_d >= ne_hard) {
      recovery_free_steps++;
    }

    min_wall_clr = std::min(min_wall_clr, step_wall_d - r_car);
    min_obs_clr = std::min(min_obs_clr, obs_d - r_car);

    // Continuous asymmetric penalty across BOTH soft buffer AND full-horizon SDF penetration!
    // Blend worst circle + 3-circle mean during wall penetration so 1-2cm rear overhang swing
    // never vetoes a front-first recovery arc back into white road.
    const double eff_wall_d = (step_wall_d < wall_hard) ?
      (0.55 * step_wall_d + 0.45 * step_wall_mean) : step_wall_d;
    double step_pen = 0.0;
    if (eff_wall_d < wall_hard) {
      const double pen = (wall_hard - eff_wall_d) / 0.05;
      step_pen += config_.wall_soft_weight * (1.2 + 2.8 * pen + 2.0 * pen * pen);
    } else if (eff_wall_d < wall_soft) {
      const double frac = (wall_soft - eff_wall_d) / std::max(0.01, wall_soft - wall_hard);
      step_pen += config_.wall_soft_weight * frac * frac;
    }
    if (step_cone_d < cone_hard) {
      const double pen = (cone_hard - step_cone_d) / 0.05;
      step_pen += config_.cone_soft_weight * (1.2 + 2.5 * pen + 1.8 * pen * pen);
    } else if (step_cone_d < cone_soft) {
      const double frac = (cone_soft - step_cone_d) / std::max(0.01, cone_soft - cone_hard);
      step_pen += config_.cone_soft_weight * frac * frac;
    }
    if (step_ne_d < ne_hard) {
      const double pen = (ne_hard - step_ne_d) / 0.05;
      step_pen += config_.no_entry_soft_weight * (1.4 + 3.0 * pen + 2.2 * pen * pen);
    } else if (step_ne_d < ne_soft) {
      const double frac = (ne_soft - step_ne_d) / std::max(0.01, ne_soft - ne_hard);
      step_pen += config_.no_entry_soft_weight * frac * frac;
    }

    soft_cost_sum += step_pen;
    eval_steps++;
  }

  // Straight Bonus Digit evaluation (debounced by filter_and_lock_straight_digits!)
  if (allow_digit_bonus && !straight_digits.empty() && safe_len >= 0.35) {
    for (const auto & gd : straight_digits) {
      if (gd.d_bumper_m < 0.15 || gd.d_bumper_m > safe_len + 0.10) {
        continue;
      }
      const double d_eval = clamp_d(gd.d_bumper_m, 0.0, safe_len);
      const double y_arc = y0 + d_eval * std::sin(th0 + 0.5 * kappa * d_eval);
      const double lat_dist = std::abs(y_arc - gd.y_lat_m);
      if (gd.digit == highest_digit_val &&
        std::abs(gd.y_lat_m) <= config_.max_digit_lateral_shift_m + 0.06)
      {
        const double bonus = config_.digit_bonus_weight *
          (static_cast<double>(gd.digit) / 8.0) *
          std::exp(-(lat_dist * lat_dist) / (2.0 * 0.07 * 0.07));
        if (bonus > digit_bonus_score) {
          digit_bonus_score = bonus;
          if (lat_dist < 0.09) {
            out_hit_digit = gd.digit;
          }
        }
      } else if (!config_.allow_dual_digit_cover && gd.digit < highest_digit_val) {
        const double cover_thresh = r_car + 0.045;
        if (lat_dist < cover_thresh) {
          const double overlap = (cover_thresh - lat_dist) / cover_thresh;
          digit_bonus_score -= config_.digit_repel_weight * overlap * overlap;
        }
      }
    }
  }

  const double mean_soft_pen = (eval_steps > 0) ?
    (soft_cost_sum / static_cast<double>(eval_steps)) : 30.0;

  // Global heading alignment at arc end
  const double world_th_end = wrap_angle_rad(rel_yaw_rad_ + th);
  double heading_term = 0.0;
  if (goal_yaw_active) {
    const double align = std::cos(wrap_angle_rad(world_th_end - goal_yaw));
    heading_term += config_.heading_weight * align;
    if (align < -0.15) {
      heading_term -= config_.wrong_way_penalty * (-align);
    }
  }
  if (forbid_reverse_yaw) {
    const double rev_align = std::cos(wrap_angle_rad(world_th_end - reverse_yaw));
    if (rev_align > 0.45) {
      heading_term -= config_.wrong_way_penalty * (rev_align - 0.45);
    }
  }

  // Fork entry lateral preference (-1 = left -> +y, +1 = right -> -y)
  // Scale matches soft wall penalty so committed fork branch NEVER gets overruled by minor edge graze!
  double branch_term = 0.0;
  if (branch_bias != 0) {
    const double desired_y_sign = (branch_bias < 0) ? 1.0 : -1.0;
    const double y_blend = 0.55 * (mid_captured ? y_at_mid : y) + 0.45 * y;
    branch_term = config_.branch_bias_weight * clamp_d(y_blend * desired_y_sign, -0.45, 0.45);
  }

  // Smoothness penalty
  const double dkappa = kappa - prev_kappa_;
  const double smooth_pen = config_.smooth_weight * dkappa * dkappa;

  // Clearance + white-road recovery bonus (strongly guides grazing car back into BEV_FREE!)
  const double rec_bonus = 0.20 * static_cast<double>(recovery_free_steps);
  const double clr_bonus = config_.center_bias_weight * clamp_d(min_wall_clr, -0.25, 0.14) + rec_bonus;

  out_cand.safe_len_m = safe_len;
  out_cand.min_wall_clr_m = min_wall_clr;
  out_cand.min_obs_clr_m = min_obs_clr;
  out_cand.hard_feasible = (!hit_hard) || (safe_len >= 0.20);
  out_cand.cost_safe_len = config_.safe_len_weight * safe_len;
  out_cand.cost_soft = -mean_soft_pen;
  out_cand.cost_heading = heading_term;
  out_cand.cost_branch = branch_term;
  out_cand.cost_digit = digit_bonus_score;
  out_cand.cost_clearance = clr_bonus;
  out_cand.cost_smooth = -smooth_pen;
  out_cand.score =
    out_cand.cost_safe_len + out_cand.cost_soft + out_cand.cost_heading +
    out_cand.cost_branch + out_cand.cost_digit + out_cand.cost_clearance +
    out_cand.cost_smooth;
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

  // 1) Warp short-window EKF BEV memory & overlay current live frame
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
  if (is_holding_yellow_stop || width_m > 0.56 || left_branch_open || right_branch_open) {
    update_fork_votes(valid_digits, is_holding_yellow_stop);
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
      goal_yaw_active = false;  // Free lane follow on start straight
      forbid_reverse_yaw = true;
      reverse_yaw = kPi;
      break;
    case TopologyPhase::FORK_RIGHT_ENTER:
      // Avoid premature apex-cutting on the first 0.20m out of the yellow stop line;
      // branch_bias=+1 guides right cleanly, then North heading engages smoothly.
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
      // CRITICAL: Force RIGHT turn (-180 deg East) into S-curve, forbid LEFT (0 deg backwards!)
      goal_yaw = kPi;
      goal_yaw_active = true;
      forbid_reverse_yaw = true;
      reverse_yaw = 0.0;
      branch_bias = +1;  // Bias right into S-curve entrance
      break;
    case TopologyPhase::FORK_LEFT_UTURN:
      branch_bias = (committed_branch_ < 0) ? -1 : 0;
      // Once rotated past +135 deg, keep heading East (+pi) past T-junction stem
      if (std::abs(wrap_angle_rad(rel_yaw_rad_ - kPi)) < 0.85 && cum_abs_yaw_rad_ > 1.8) {
        goal_yaw = kPi;
        goal_yaw_active = true;
        forbid_reverse_yaw = true;
        reverse_yaw = 0.5 * kPi;  // Do not dive South down vertical shortcut!
      }
      break;
    case TopologyPhase::FORK_S_CURVE:
      forbid_reverse_yaw = true;
      reverse_yaw = 0.0;  // Never reverse West back into fork!
      break;
    case TopologyPhase::RB1_APPROACH:
      forbid_reverse_yaw = true;
      reverse_yaw = kPi;
      break;
    case TopologyPhase::RB1_ACTIVE:
      branch_bias = committed_branch_;
      forbid_reverse_yaw = true;
      reverse_yaw = kPi;  // Never loop backwards East inside RB1!
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
      reverse_yaw = 0.0;  // Never loop backwards West inside RB2!
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

  // 4) Scan for nearby red obstacles in BEV (d in [-0.20, +0.90]m, |y| <= 0.40m)
  bool near_cone = false;
  bool near_no_entry = false;
  for (int r = 0; r < kBevRows; ++r) {
    const double d_m = kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes;
    if (d_m < -0.20 || d_m > 0.90) {
      continue;
    }
    for (int c = 0; c < kBevCols; ++c) {
      const double y_m = kBevYMax - (static_cast<double>(c) + 0.5) * kBevRes;
      if (std::abs(y_m) > 0.40) {
        continue;
      }
      const uint8_t cell = fused_grid_[static_cast<size_t>(r * kBevCols + c)];
      if (cell == BEV_OBS_NO_ENTRY) {
        near_no_entry = true;
      } else if (cell == BEV_OBS_CONE) {
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

  // 5) Latency forward-prediction (Smith predictor, tau ~= 0.09s)
  const double v_meas = (pose.valid && std::isfinite(pose.linear_v)) ?
    pose.linear_v : last_speed_cmd_;
  const double w_meas = (pose.valid && std::isfinite(pose.angular_w)) ?
    (pose.angular_w - (preflight_.gyro_calibrated ? preflight_.gyro_bias_rps : 0.0)) :
    (last_speed_cmd_ * prev_kappa_);
  const double tau = clamp_d(config_.control_latency_s, 0.0, 0.18);
  const double x0 = std::max(0.0, v_meas) * tau;
  const double th0 = clamp_d(w_meas * tau, -0.35, 0.35);
  const double y0 = 0.5 * x0 * std::sin(th0);

  // 6) Roll out N physically feasible Ackermann arcs across REAL asymmetric [-kappa_R, +kappa_L]
  //    Single unified SDF score selects BOTH normal cruise arcs AND smooth wall-recovery arcs
  //    (zero hard-cutoff blindness, zero Level-1 tie-to-i=0 deadlock!).
  const int N = config_.num_arc_samples;
  res.candidates.resize(static_cast<size_t>(N));

  int best_idx = N / 2;
  double best_score = -1e18;
  int best_digit = 0;
  bool any_feasible = false;

  for (int i = 0; i < N; ++i) {
    // Non-uniform spacing: denser near 0 and exact hits at -kappa_max_right and +kappa_max_left
    const double u = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(N - 1);
    const double kappa = (u >= 0.0) ?
      (u * config_.kappa_max_left) : (u * config_.kappa_max_right);

    int hit_digit = 0;
    evaluate_arc(
      x0, y0, th0, kappa,
      goal_yaw, goal_yaw_active,
      forbid_reverse_yaw, reverse_yaw,
      branch_bias, locked_digits, allow_digit_bonus,
      res.candidates[static_cast<size_t>(i)],
      hit_digit);

    const auto & cand = res.candidates[static_cast<size_t>(i)];
    if (cand.hard_feasible) {
      any_feasible = true;
    }
    if (cand.score > best_score) {
      best_score = cand.score;
      best_idx = i;
      best_digit = hit_digit;
    }
  }

  const auto & chosen = res.candidates[static_cast<size_t>(best_idx)];
  const double kappa_star = chosen.kappa;
  prev_kappa_ = kappa_star;

  res.chosen_kappa = kappa_star;
  res.chosen_safe_len_m = chosen.safe_len_m;
  res.min_wall_clearance_m = chosen.min_wall_clr_m;
  res.min_obs_clearance_m = chosen.min_obs_clr_m;
  res.has_feasible_arc = any_feasible;
  res.chosen_candidate = chosen;
  res.feasible_arc_count = static_cast<int>(std::count_if(
    res.candidates.begin(), res.candidates.end(),
    [](const BevArcCandidate & c) { return c.hard_feasible; }));
  res.digit_target_active = (best_digit > 0);
  res.target_digit = best_digit;

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
        // Counter-steer slightly while backing up to open front turning angle
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

  // 8) Single Clean Physics Speed Envelope (ZERO active_decel_kp double-integration bug!)
  double v_target = config_.cruise_speed_mps;
  if (config_.enable_clear_straight_boost &&
    clean_straight && !res.has_near_obstacle && !yellow_ahead &&
    !in_fork_or_t && chosen.safe_len_m >= 0.78 && std::abs(kappa_star) < 0.35)
  {
    v_target = config_.clear_straight_speed_mps;
  }

  // Centripetal lateral acceleration limit: v <= sqrt(a_lat_max / |kappa|)
  const double v_curve = std::sqrt(config_.max_lat_accel_mps2 / (std::abs(kappa_star) + 0.15));
  v_target = std::min(v_target, v_curve);

  // Horizon / corner-entry lookahead speed limit (slows BEFORE 90 deg bends / T-junction!)
  if (chosen.safe_len_m < 0.72) {
    const double v_hor = std::sqrt(2.0 * 0.65 * std::max(0.03, chosen.safe_len_m - 0.12));
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
  if (!any_feasible) {
    v_target = std::min(v_target, config_.creep_speed_mps);
  }

  // Yellow line smooth pre-brake envelope
  if (yellow_ahead && is_yellow_rearmed() && std::isfinite(yellow_speed_cap_mps)) {
    v_target = std::min(v_target, std::max(0.0, yellow_speed_cap_mps));
  }

  v_target = clamp_d(v_target, config_.min_speed_mps, config_.max_speed_mps);
  res.speed_cap_mps = v_target;

  // Single monotonic asymmetric slew rate limiter (never feeds odom error back into state!)
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

  // IMU gyro inner feedback loop (cancels 1.7 deg mechanical right offset & servo under-turn!)
  if (pose.valid && std::isfinite(pose.angular_w) && v_cmd > 0.05 && config_.imu_yaw_kp > 0.0) {
    const double omega_target = v_cmd * kappa_star;
    const double omega_imu = pose.angular_w -
      (preflight_.gyro_calibrated ? preflight_.gyro_bias_rps : 0.0);
    // Avoid windup when already requesting hard steering saturation
    if (k_frac < 0.94) {
      omega_cmd += config_.imu_yaw_kp * (omega_target - omega_imu);
    }
  }

  omega_cmd = clamp_d(omega_cmd, -config_.max_yaw_rate, config_.max_yaw_rate);

  res.linear_speed_mps = v_cmd;
  res.yaw_rate_rps = omega_cmd;

  // Concise human-readable mode tag
  if (!any_feasible) {
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
