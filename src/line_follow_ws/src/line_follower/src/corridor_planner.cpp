#include "line_follower/corridor_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace line_follower
{

CorridorPlanner::CorridorPlanner(const CorridorPlannerConfig & config)
{
  set_config(config);
}

void CorridorPlanner::set_config(const CorridorPlannerConfig & config)
{
  config_ = config;
  config_.car_width_m = std::max(0.05, config_.car_width_m);
  config_.car_length_m = std::max(0.05, config_.car_length_m);
  config_.default_lane_width_m = std::max(0.20, config_.default_lane_width_m);
  config_.border_margin_m = std::max(0.0, config_.border_margin_m);
  config_.cone_margin_m = std::max(0.0, config_.cone_margin_m);
  config_.no_entry_margin_m = std::max(0.0, config_.no_entry_margin_m);
  config_.obstacle_lookahead_m = std::max(0.25, config_.obstacle_lookahead_m);
  config_.obstacle_steer_horizon_m = std::clamp(
    config_.obstacle_steer_horizon_m, 0.25, config_.obstacle_lookahead_m);
  config_.obstacle_curve_min_horizon_m = std::clamp(
    config_.obstacle_curve_min_horizon_m, 0.20, config_.obstacle_steer_horizon_m);
  config_.obstacle_stagger_gap_m = std::max(0.05, config_.obstacle_stagger_gap_m);
  config_.obstacle_rear_keep_m = std::max(0.0, config_.obstacle_rear_keep_m);
  config_.fork_min_votes = std::max(1, config_.fork_min_votes);
  config_.digit_min_frames = std::max(1, config_.digit_min_frames);
}

const CorridorPlannerConfig & CorridorPlanner::config() const
{
  return config_;
}

void CorridorPlanner::reset()
{
  reset_fork_votes();
  clear_fork_commit();
  digit_tracks_.clear();
  locked_digit_ = 0;
  locked_digit_y_m_ = 0.0;
  locked_digit_d_m_ = 1e9;
  locked_digit_time_sec_ = -1e9;
}

void CorridorPlanner::reset_fork_votes()
{
  left_digit_votes_.fill(0);
  left_digit_conf_sum_.fill(0.0);
  right_digit_votes_.fill(0);
  right_digit_conf_sum_.fill(0.0);
  left_no_entry_votes_ = 0;
  right_no_entry_votes_ = 0;
  left_cone_votes_ = 0;
  right_cone_votes_ = 0;
}

void CorridorPlanner::clear_fork_commit()
{
  committed_branch_ = 0;
  commit_start_s_m_ = -1e9;
}

int CorridorPlanner::committed_branch(double current_arc_s_m) const
{
  if (committed_branch_ == 0) {
    return 0;
  }
  if ((current_arc_s_m - commit_start_s_m_) >= config_.fork_commit_dist_m) {
    return 0;
  }
  return committed_branch_;
}

void CorridorPlanner::update_fork_static_votes(
  const std::vector<GroundDetection> & detections,
  double /*now_sec*/)
{
  // Collect valid bonus digits first to handle paired left/right splitting cleanly
  std::vector<const GroundDetection *> digits;
  for (const auto & det : detections) {
    if (det.type == SemanticObjectType::BONUS_DIGIT &&
      det.digit >= 1 && det.digit <= 8 &&
      det.confidence >= config_.digit_min_conf &&
      det.d_bumper_m >= 0.0 && det.d_bumper_m <= 2.2)
    {
      digits.push_back(&det);
    } else if (det.type == SemanticObjectType::RED_NO_ENTRY &&
      det.d_bumper_m >= 0.0 && det.d_bumper_m <= 2.2)
    {
      if (det.y_lat_m >= 0.0) {
        left_no_entry_votes_++;
      } else {
        right_no_entry_votes_++;
      }
    } else if (det.type == SemanticObjectType::RED_CONE &&
      det.d_bumper_m >= 0.0 && det.d_bumper_m <= 2.2)
    {
      if (det.y_lat_m >= 0.0) {
        left_cone_votes_++;
      } else {
        right_cone_votes_++;
      }
    }
  }

  if (digits.size() >= 2) {
    // Sort by y_lat_m descending: highest y is LEFT (+y), lowest y is RIGHT (-y)
    std::sort(
      digits.begin(),
      digits.end(),
      [](const GroundDetection * a, const GroundDetection * b) {
        return a->y_lat_m > b->y_lat_m;
      });
    const auto * left_det = digits.front();
    const auto * right_det = digits.back();
    if ((left_det->y_lat_m - right_det->y_lat_m) >= 0.04) {
      left_digit_votes_[static_cast<std::size_t>(left_det->digit)]++;
      left_digit_conf_sum_[static_cast<std::size_t>(left_det->digit)] += left_det->confidence;
      right_digit_votes_[static_cast<std::size_t>(right_det->digit)]++;
      right_digit_conf_sum_[static_cast<std::size_t>(right_det->digit)] += right_det->confidence;
      return;
    }
  }

  for (const auto * d : digits) {
    if (d->y_lat_m >= 0.0) {
      left_digit_votes_[static_cast<std::size_t>(d->digit)]++;
      left_digit_conf_sum_[static_cast<std::size_t>(d->digit)] += d->confidence;
    } else {
      right_digit_votes_[static_cast<std::size_t>(d->digit)]++;
      right_digit_conf_sum_[static_cast<std::size_t>(d->digit)] += d->confidence;
    }
  }
}

ForkVoteSummary CorridorPlanner::fork_summary() const
{
  ForkVoteSummary s{};

  int total_l_votes = 0;
  int best_l_votes = 0;
  double best_l_conf = -1.0;
  for (int d = 1; d <= 8; ++d) {
    const int v = left_digit_votes_[static_cast<std::size_t>(d)];
    const double c = left_digit_conf_sum_[static_cast<std::size_t>(d)];
    total_l_votes += v;
    if (v > best_l_votes || (v == best_l_votes && v > 0 && c > best_l_conf)) {
      best_l_votes = v;
      best_l_conf = c;
      s.left_digit = d;
    }
  }
  s.left_votes = best_l_votes;
  if (best_l_votes > 0 && total_l_votes > 0) {
    const double mean_c = best_l_conf / static_cast<double>(best_l_votes);
    const double consistency =
      static_cast<double>(best_l_votes) / static_cast<double>(total_l_votes);
    s.left_conf_eff = std::clamp(mean_c, 0.2, 1.0) * consistency;
  }

  int total_r_votes = 0;
  int best_r_votes = 0;
  double best_r_conf = -1.0;
  for (int d = 1; d <= 8; ++d) {
    const int v = right_digit_votes_[static_cast<std::size_t>(d)];
    const double c = right_digit_conf_sum_[static_cast<std::size_t>(d)];
    total_r_votes += v;
    if (v > best_r_votes || (v == best_r_votes && v > 0 && c > best_r_conf)) {
      best_r_votes = v;
      best_r_conf = c;
      s.right_digit = d;
    }
  }
  s.right_votes = best_r_votes;
  if (best_r_votes > 0 && total_r_votes > 0) {
    const double mean_c = best_r_conf / static_cast<double>(best_r_votes);
    const double consistency =
      static_cast<double>(best_r_votes) / static_cast<double>(total_r_votes);
    s.right_conf_eff = std::clamp(mean_c, 0.2, 1.0) * consistency;
  }

  s.left_has_no_entry = (left_no_entry_votes_ >= config_.fork_min_votes);
  s.right_has_no_entry = (right_no_entry_votes_ >= config_.fork_min_votes);
  s.left_has_cone = (left_cone_votes_ >= config_.fork_min_votes);
  s.right_has_cone = (right_cone_votes_ >= config_.fork_min_votes);
  s.chosen_branch = committed_branch_;
  return s;
}

int CorridorPlanner::finalize_fork_decision(double current_arc_s_m)
{
  const ForkVoteSummary s = fork_summary();
  const bool left_valid = s.left_votes >= config_.fork_min_votes;
  const bool right_valid = s.right_votes >= config_.fork_min_votes;

  // Hard ban any branch with a red no-entry zone (+30s) when the other branch is clear
  if (s.left_has_no_entry && !s.right_has_no_entry) {
    committed_branch_ = +1;
    commit_start_s_m_ = current_arc_s_m;
    return committed_branch_;
  }
  if (s.right_has_no_entry && !s.left_has_no_entry) {
    committed_branch_ = -1;
    commit_start_s_m_ = current_arc_s_m;
    return committed_branch_;
  }

  // If no fork digits or obstacles were confirmed, do not commit an arbitrary branch
  if (!left_valid && !right_valid && !s.left_has_cone && !s.right_has_cone) {
    return 0;
  }

  const int d_left = left_valid ? s.left_digit : 0;
  const int d_right = right_valid ? s.right_digit : 0;

  const bool is_roundabout = (config_.track_mode == "roundabout");
  const double outer_pen = is_roundabout ? 0.0 : config_.fork_outer_time_penalty_s;

  double t_left = 0.0;
  double t_right = 0.0;
  if (config_.fork_outer_branch < 0) {
    t_left = outer_pen;
  } else if (config_.fork_outer_branch > 0) {
    t_right = outer_pen;
  }

  const double p_left =
    (s.left_has_no_entry ? config_.fork_no_entry_penalty_s : 0.0) +
    (s.left_has_cone ? config_.fork_cone_penalty_s : 0.0);
  const double p_right =
    (s.right_has_no_entry ? config_.fork_no_entry_penalty_s : 0.0) +
    (s.right_has_cone ? config_.fork_cone_penalty_s : 0.0);

  // J = T_branch + P_obs - 5 * digit * p_conf_eff (lower score is better!)
  const double eff_left = config_.enable_risk_aware_decision ? s.left_conf_eff : 1.0;
  const double eff_right = config_.enable_risk_aware_decision ? s.right_conf_eff : 1.0;
  const double j_left =
    t_left + p_left - config_.bonus_score_weight_s * static_cast<double>(d_left) * eff_left;
  const double j_right =
    t_right + p_right - config_.bonus_score_weight_s * static_cast<double>(d_right) * eff_right;

  if (j_left < j_right - 1e-6) {
    committed_branch_ = -1;
  } else if (j_right < j_left - 1e-6) {
    committed_branch_ = +1;
  } else {
    // Exact tie: prefer shorter branch (inner), or right if symmetric
    committed_branch_ = (t_left <= t_right) ? -1 : +1;
  }

  commit_start_s_m_ = current_arc_s_m;
  return committed_branch_;
}

std::vector<LateralInterval> CorridorPlanner::subtract_interval(
  const std::vector<LateralInterval> & intervals,
  double ex_min,
  double ex_max)
{
  if (ex_min >= ex_max) {
    return intervals;
  }

  std::vector<LateralInterval> out;
  out.reserve(intervals.size() + 1);

  for (const auto & iv : intervals) {
    if (ex_max <= iv.y_min || ex_min >= iv.y_max) {
      out.push_back(iv);
    } else {
      if (ex_min > iv.y_min) {
        out.push_back(LateralInterval{iv.y_min, ex_min});
      }
      if (ex_max < iv.y_max) {
        out.push_back(LateralInterval{ex_max, iv.y_max});
      }
    }
  }
  return out;
}

void CorridorPlanner::update_straight_digit_tracks(
  const std::vector<GroundDetection> & detections,
  double now_sec)
{
  // 1. Prune expired tracks
  digit_tracks_.erase(
    std::remove_if(
      digit_tracks_.begin(),
      digit_tracks_.end(),
      [&](const DigitTrack & t) {
        return (now_sec - t.last_seen_sec) > config_.digit_track_ttl_sec;
      }),
    digit_tracks_.end());

  // 2. Collect valid straight-road digit detections
  std::vector<const GroundDetection *> valid_dets;
  for (const auto & det : detections) {
    if (det.type != SemanticObjectType::BONUS_DIGIT ||
      det.digit < 1 || det.digit > 8 ||
      det.confidence < config_.digit_min_conf)
    {
      continue;
    }
    if (det.d_bumper_m < config_.digit_lookahead_min_m ||
      det.d_bumper_m > config_.digit_lookahead_max_m)
    {
      continue;
    }
    valid_dets.push_back(&det);
  }

  std::vector<bool> track_matched(digit_tracks_.size(), false);
  std::vector<bool> det_matched(valid_dets.size(), false);

  auto apply_update = [&](DigitTrack & trk, const GroundDetection & obs) {
      trk.frames++;
      trk.mean_conf =
        (trk.mean_conf * static_cast<double>(trk.frames - 1) + obs.confidence) /
        static_cast<double>(trk.frames);
      trk.d_bumper_m = 0.5 * trk.d_bumper_m + 0.5 * obs.d_bumper_m;
      trk.y_lat_m = 0.5 * trk.y_lat_m + 0.5 * obs.y_lat_m;
      if (obs.track_id > 0) {
        trk.track_id = obs.track_id;
      }
      trk.last_seen_sec = now_sec;
    };

  // Pass 1: exact (digit, track_id) match when track_id > 0
  for (std::size_t di = 0; di < valid_dets.size(); ++di) {
    const GroundDetection & obs = *valid_dets[di];
    if (obs.track_id <= 0) {
      continue;
    }
    for (std::size_t ti = 0; ti < digit_tracks_.size(); ++ti) {
      if (track_matched[ti]) {
        continue;
      }
      if (digit_tracks_[ti].digit == obs.digit && digit_tracks_[ti].track_id == obs.track_id) {
        apply_update(digit_tracks_[ti], obs);
        track_matched[ti] = true;
        det_matched[di] = true;
        break;
      }
    }
  }

  // Pass 2: spatial nearest-neighbor match (same digit, hypot(dd, dy) <= 0.18m)
  constexpr double kMaxMatchDistM = 0.18;
  for (std::size_t di = 0; di < valid_dets.size(); ++di) {
    if (det_matched[di]) {
      continue;
    }
    const GroundDetection & obs = *valid_dets[di];
    int best_ti = -1;
    double best_dist = kMaxMatchDistM;
    for (std::size_t ti = 0; ti < digit_tracks_.size(); ++ti) {
      if (track_matched[ti] || digit_tracks_[ti].digit != obs.digit) {
        continue;
      }
      const double dist = std::hypot(
        obs.d_bumper_m - digit_tracks_[ti].d_bumper_m,
        obs.y_lat_m - digit_tracks_[ti].y_lat_m);
      if (dist <= best_dist) {
        best_dist = dist;
        best_ti = static_cast<int>(ti);
      }
    }
    if (best_ti >= 0) {
      const std::size_t ti = static_cast<std::size_t>(best_ti);
      apply_update(digit_tracks_[ti], obs);
      track_matched[ti] = true;
      det_matched[di] = true;
    }
  }

  // Pass 3: create new tracks for unmatched detections
  for (std::size_t di = 0; di < valid_dets.size(); ++di) {
    if (det_matched[di]) {
      continue;
    }
    const GroundDetection & obs = *valid_dets[di];
    DigitTrack trk{};
    trk.digit = obs.digit;
    trk.track_id = obs.track_id;
    trk.frames = 1;
    trk.mean_conf = obs.confidence;
    trk.d_bumper_m = obs.d_bumper_m;
    trk.y_lat_m = obs.y_lat_m;
    trk.last_seen_sec = now_sec;
    digit_tracks_.push_back(trk);
  }
}

CorridorPlan CorridorPlanner::plan(
  double y_lane_left_m,
  double y_lane_right_m,
  double current_y_m,
  const std::vector<LocalObstacle> & obstacles,
  const std::vector<GroundDetection> & raw_detections,
  bool is_stopped_at_yellow,
  double current_arc_s_m,
  double now_sec,
  double curve_level)
{
  CorridorPlan out{};
  out.speed_cap_mps = config_.cruise_speed_mps;

  if (is_stopped_at_yellow) {
    update_fork_static_votes(raw_detections, now_sec);
  }

  // Ensure valid lane bounds (+y left > -y right)
  if (y_lane_left_m <= y_lane_right_m + 0.10) {
    const double half_w = 0.5 * config_.default_lane_width_m;
    y_lane_left_m = half_w;
    y_lane_right_m = -half_w;
  }

  const double half_car = 0.5 * config_.car_width_m;
  const double lane_mid = 0.5 * (y_lane_left_m + y_lane_right_m);

  auto compute_gaps = [&](double border_m, double cone_m, double no_entry_m,
      const std::vector<const LocalObstacle *> & obs_list) {
      double y_min_base = y_lane_right_m + half_car + border_m;
      double y_max_base = y_lane_left_m - half_car - border_m;
      if (y_min_base > y_max_base) {
        y_min_base = lane_mid;
        y_max_base = lane_mid;
      }
      std::vector<LateralInterval> ivs{LateralInterval{y_min_base, y_max_base}};
      for (const auto * obs : obs_list) {
        const double m =
          (obs->type == SemanticObjectType::RED_NO_ENTRY) ? no_entry_m : cone_m;
        const double ex_min = obs->y_min_m - half_car - m;
        const double ex_max = obs->y_max_m + half_car + m;
        ivs = subtract_interval(ivs, ex_min, ex_max);
      }
      return ivs;
    };

  // Two-zone longitudinal window:
  //   - Tighten steering horizon smoothly in curves to prevent far-around-the-bend lateral interference
  const double c_norm = std::clamp(std::abs(curve_level), 0.0, 1.0);
  const double eff_steer_horizon =
    config_.obstacle_steer_horizon_m -
    c_norm * (config_.obstacle_steer_horizon_m - config_.obstacle_curve_min_horizon_m);

  std::vector<const LocalObstacle *> near_obs;
  std::vector<const LocalObstacle *> far_obs;
  bool any_no_entry_in_lookahead = false;
  bool any_obs_in_lookahead = false;

  for (const auto & obs : obstacles) {
    if (obs.type != SemanticObjectType::RED_CONE &&
      obs.type != SemanticObjectType::RED_NO_ENTRY)
    {
      continue;
    }
    if (obs.d_max_m < -config_.obstacle_rear_keep_m ||
      obs.d_min_m > config_.obstacle_lookahead_m)
    {
      continue;
    }
    any_obs_in_lookahead = true;
    if (obs.type == SemanticObjectType::RED_NO_ENTRY) {
      any_no_entry_in_lookahead = true;
    }
    if (obs.d_min_m <= eff_steer_horizon) {
      near_obs.push_back(&obs);
    } else {
      far_obs.push_back(&obs);
    }
  }

  // Sort near obstacles closest-first so staggered front/back obstacles are solved sequentially
  std::sort(
    near_obs.begin(),
    near_obs.end(),
    [](const LocalObstacle * a, const LocalObstacle * b) {
      return a->d_min_m < b->d_min_m;
    });

  // Staggered S-slalom separation: only carve lateral corridor for the immediate longitudinal cluster
  // (obstacles overlapping or within obstacle_stagger_gap_m behind the closest obstacle).
  // Farther staggered obstacles still apply speed caps without locking out the immediate escape gap!
  std::vector<const LocalObstacle *> active_obs;
  bool has_no_entry = false;
  if (!near_obs.empty()) {
    const double cluster_d_limit = near_obs.front()->d_max_m + config_.obstacle_stagger_gap_m;
    for (const auto * obs : near_obs) {
      if (obs->d_min_m <= cluster_d_limit) {
        active_obs.push_back(obs);
        if (obs->type == SemanticObjectType::RED_NO_ENTRY) {
          has_no_entry = true;
        }
      } else {
        far_obs.push_back(obs);
      }
    }
  }

  out.near_obstacle_count = static_cast<int>(active_obs.size());
  out.far_obstacle_count = static_cast<int>(far_obs.size());

  std::vector<LateralInterval> gaps =
    compute_gaps(config_.border_margin_m, config_.cone_margin_m, config_.no_entry_margin_m, active_obs);

  // Relax border & cone margins slightly before declaring blocked, keeping no-entry (+30s) strict
  if (gaps.empty() && !active_obs.empty()) {
    gaps = compute_gaps(
      std::min(0.008, config_.border_margin_m),
      std::min(0.008, config_.cone_margin_m),
      config_.no_entry_margin_m,
      active_obs);
  }
  if (gaps.empty() && !active_obs.empty()) {
    gaps = compute_gaps(
      0.002,
      0.002,
      std::min(0.010, config_.no_entry_margin_m),
      active_obs);
  }
  // If multiple overlapping near obstacles mutually block the corridor, solve the single closest one
  // first rather than freezing in the middle of an S-slalom!
  if (gaps.empty() && active_obs.size() >= 2) {
    std::vector<const LocalObstacle *> nearest_only{active_obs.front()};
    gaps = compute_gaps(
      std::min(0.008, config_.border_margin_m),
      std::min(0.008, config_.cone_margin_m),
      std::min(0.015, config_.no_entry_margin_m),
      nearest_only);
  }

  LateralInterval chosen_gap{lane_mid, lane_mid};
  if (!gaps.empty()) {
    // Pick the widest feasible gap; if multiple gaps are wide (>= 4cm), prefer closest to current_y
    double best_score = -1e9;
    for (const auto & g : gaps) {
      const double w = g.width();
      const double clamped_cur = std::clamp(current_y_m, g.y_min, g.y_max);
      const double dist_to_cur = std::abs(clamped_cur - current_y_m);
      const double score = 3.0 * w - dist_to_cur;
      if (score > best_score) {
        best_score = score;
        chosen_gap = g;
      }
    }
    out.has_blocking_obstacle = false;
    out.safe_y_min_m = chosen_gap.y_min;
    out.safe_y_max_m = chosen_gap.y_max;

    if (active_obs.empty()) {
      out.target_y_m = std::clamp(lane_mid, chosen_gap.y_min, chosen_gap.y_max);
      if (any_obs_in_lookahead) {
        // Far preview zone: pre-decelerate smoothly without premature lateral steering
        out.speed_cap_mps = any_no_entry_in_lookahead ?
          config_.no_entry_speed_mps : config_.obstacle_speed_mps;
        out.active_mode_tag = "far_obs_slow";
      } else {
        out.speed_cap_mps = config_.cruise_speed_mps;
        out.active_mode_tag = "cruise";
      }
    } else {
      // Maximize symmetric clearance inside the feasible gap
      out.target_y_m = chosen_gap.center();
      out.speed_cap_mps = (has_no_entry || any_no_entry_in_lookahead) ?
        config_.no_entry_speed_mps : config_.obstacle_speed_mps;
      out.active_mode_tag = has_no_entry ? "avoid_no_entry" : "avoid_cone";
    }
  } else {
    // No strictly collision-free gap: prioritize staying out of +30s red no-entry & inside lane
    out.has_blocking_obstacle = true;
    out.speed_cap_mps = config_.blocked_creep_speed_mps;
    out.active_mode_tag = "blocked";

    const double hard_min = std::min(lane_mid, y_lane_right_m + half_car);
    const double hard_max = std::max(lane_mid, y_lane_left_m - half_car);
    out.safe_y_min_m = hard_min;
    out.safe_y_max_m = hard_max;

    double best_y = lane_mid;
    double best_clear = -1e9;
    constexpr int kSamples = 17;
    for (int i = 0; i < kSamples; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(kSamples - 1);
      const double y = hard_min + t * (hard_max - hard_min);
      double worst_penalty_clear = 1e9;
      for (const auto * obs : active_obs) {
        const double clear = std::max(obs->y_min_m - (y + half_car), (y - half_car) - obs->y_max_m);
        // Weight no-entry clearance 3x higher (+30s vs +10s)
        const double weighted =
          (obs->type == SemanticObjectType::RED_NO_ENTRY) ? (clear - 0.05) : clear;
        worst_penalty_clear = std::min(worst_penalty_clear, weighted);
      }
      if (worst_penalty_clear > best_clear) {
        best_clear = worst_penalty_clear;
        best_y = y;
      }
    }
    out.target_y_m = best_y;
  }

  out.preferred_branch = committed_branch(current_arc_s_m);
  if (out.preferred_branch != 0 && active_obs.empty()) {
    out.active_mode_tag = (out.preferred_branch < 0) ? "fork_left" : "fork_right";
  }

  // Straight-road Bonus digit targeting (lowest safety priority: NEVER violates lane/obstacle corridor)
  if (!is_stopped_at_yellow && !out.has_blocking_obstacle && out.preferred_branch == 0) {
    update_straight_digit_tracks(raw_detections, now_sec);

    std::vector<const DigitTrack *> confirmed;
    for (const auto & trk : digit_tracks_) {
      if (trk.frames >= config_.digit_min_frames &&
        trk.mean_conf >= config_.digit_min_conf)
      {
        confirmed.push_back(&trk);
      }
    }

    if (!confirmed.empty()) {
      auto risk_aware_score = [&](const DigitTrack * t) {
          const double shift = std::abs(t->y_lat_m - current_y_m);
          const double r_shift = (config_.max_digit_lateral_shift_m > 1e-6) ?
            std::clamp(shift / config_.max_digit_lateral_shift_m, 0.0, 1.0) : 0.0;
          const double p_shift = 1.0 - 0.35 * r_shift * r_shift;
          const double edge_clear = std::min(
            t->y_lat_m - chosen_gap.y_min,
            chosen_gap.y_max - t->y_lat_m);
          double p_edge = 1.0;
          if (edge_clear < 0.0) {
            p_edge = 0.0;
          } else if (edge_clear < 0.04) {
            p_edge = 0.35 + 0.65 * (edge_clear / 0.04);
          }
          const double p_cov = p_shift * p_edge;
          const double eff_conf = std::clamp(t->mean_conf, 0.2, 1.0);
          return config_.bonus_score_weight_s * static_cast<double>(t->digit) * eff_conf * p_cov;
        };

      if (config_.enable_risk_aware_decision) {
        std::sort(
          confirmed.begin(),
          confirmed.end(),
          [&](const DigitTrack * a, const DigitTrack * b) {
            const double sa = risk_aware_score(a);
            const double sb = risk_aware_score(b);
            if (std::abs(sa - sb) > 1e-9) {
              return sa > sb;
            }
            if (a->digit != b->digit) {
              return a->digit > b->digit;
            }
            return a->mean_conf > b->mean_conf;
          });
      } else {
        // Legacy: sort descending by digit value (-5 * digit -> target the LARGER digit!)
        std::sort(
          confirmed.begin(),
          confirmed.end(),
          [](const DigitTrack * a, const DigitTrack * b) {
            if (a->digit != b->digit) {
              return a->digit > b->digit;
            }
            return a->mean_conf > b->mean_conf;
          });
      }

      const bool has_close_digit_lock =
        (locked_digit_ > 0) &&
        ((now_sec - locked_digit_time_sec_) <= 0.60) &&
        (locked_digit_d_m_ <= 0.35);

      for (const auto * cand : confirmed) {
        // Prevent last-second swerve into a smaller sibling after the committed larger digit
        // leaves the bottom of the FOV (still allows upgrading to a strictly larger digit
        // or engaging a fresh distant pair ahead >= 0.45m).
        if (has_close_digit_lock &&
          cand->d_bumper_m < 0.45 &&
          cand->digit <= locked_digit_ &&
          std::abs(cand->y_lat_m - locked_digit_y_m_) >= 0.08)
        {
          continue;
        }

        // Check if any obstacle conflicts near this digit longitudinally
        bool blocked_near_digit = false;
        for (const auto * obs : active_obs) {
          const double d_gap = std::max(
            obs->d_min_m - cand->d_bumper_m,
            cand->d_bumper_m - obs->d_max_m);
          if (d_gap <= config_.digit_obstacle_longitudinal_buffer_m) {
            const double m = (obs->type == SemanticObjectType::RED_NO_ENTRY) ?
              config_.no_entry_margin_m : config_.cone_margin_m;
            if (cand->y_lat_m >= (obs->y_min_m - half_car - m) &&
              cand->y_lat_m <= (obs->y_max_m + half_car + m))
            {
              blocked_near_digit = true;
              break;
            }
          }
        }
        if (blocked_near_digit) {
          continue;
        }

        // Digit center must lie inside our safe feasible corridor and lateral shift budget
        if (cand->y_lat_m < chosen_gap.y_min || cand->y_lat_m > chosen_gap.y_max) {
          continue;
        }
        if (std::abs(cand->y_lat_m - current_y_m) > config_.max_digit_lateral_shift_m) {
          continue;
        }

        double y_target = cand->y_lat_m;

        // Check for a nearby smaller sibling digit (10cm straight dual-digit pair)
        for (const auto * sib : confirmed) {
          if (sib == cand) {
            continue;
          }
          if (std::abs(sib->d_bumper_m - cand->d_bumper_m) <= 0.25 &&
            std::abs(sib->y_lat_m - cand->y_lat_m) <= 0.22)
          {
            if (config_.allow_dual_digit_cover) {
              // Optional switch: if car width (19.1cm) can cover both cards separated by ~10cm
              const double mid_y = 0.5 * (cand->y_lat_m + sib->y_lat_m);
              if (std::abs(cand->y_lat_m - sib->y_lat_m) <= (config_.car_width_m - 0.05) &&
                mid_y >= chosen_gap.y_min && mid_y <= chosen_gap.y_max &&
                std::abs(mid_y - current_y_m) <= config_.max_digit_lateral_shift_m)
              {
                y_target = mid_y;
              }
            } else {
              // Default: cleanly cover ONLY the larger digit by biasing slightly away from sibling
              const double dir_away = (cand->y_lat_m >= sib->y_lat_m) ? 1.0 : -1.0;
              y_target = cand->y_lat_m + dir_away * config_.digit_sibling_avoid_offset_m;
            }
            break;
          }
        }

        // Strict final clamp inside obstacle-free & lane-safe corridor
        out.target_y_m = std::clamp(y_target, chosen_gap.y_min, chosen_gap.y_max);
        out.speed_cap_mps = std::min(out.speed_cap_mps, config_.digit_approach_speed_mps);
        out.digit_target_active = true;
        out.target_digit = cand->digit;
        locked_digit_ = cand->digit;
        locked_digit_y_m_ = cand->y_lat_m;
        locked_digit_d_m_ = cand->d_bumper_m;
        locked_digit_time_sec_ = now_sec;
        if (active_obs.empty()) {
          out.active_mode_tag = "digit_bonus";
        }
        break;
      }
    }
  }

  return out;
}

}  // namespace line_follower
