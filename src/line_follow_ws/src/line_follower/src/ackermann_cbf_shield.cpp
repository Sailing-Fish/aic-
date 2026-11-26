#include "line_follower/ackermann_cbf_shield.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace line_follower
{

namespace
{

struct ObbObstacle
{
  double cx{0.0};
  double cy{0.0};
  double half_len{0.05};
  double half_wid{0.05};
  double yaw_rad{0.0};
  double margin_m{0.025};
  const char * source_tag{"red_cone"};
};

struct EvalOutcome
{
  bool safe{true};
  double min_clearance_m{1e9};
  const char * limiting_source{"none"};
};

void step_ackermann(
  double x,
  double y,
  double theta,
  double v,
  double omega,
  double dt,
  double & x_next,
  double & y_next,
  double & theta_next)
{
  if (std::abs(omega) > 1e-6) {
    const double dtheta = omega * dt;
    const double r = v / omega;
    x_next = x + r * (std::sin(theta + dtheta) - std::sin(theta));
    y_next = y - r * (std::cos(theta + dtheta) - std::cos(theta));
    theta_next = theta + dtheta;
  } else {
    x_next = x + v * dt * std::cos(theta);
    y_next = y + v * dt * std::sin(theta);
    theta_next = theta;
  }
}

void compute_barrier_values(
  double x,
  double y,
  double theta,
  const std::array<double, 3> & disc_offsets,
  double r_disc,
  double y_lane_left_m,
  double y_lane_right_m,
  double border_margin_m,
  const std::vector<ObbObstacle> & obbs,
  std::vector<double> & out_h,
  std::vector<const char *> & out_tags)
{
  out_h.clear();
  out_tags.clear();
  out_h.reserve(2 + obbs.size());
  out_tags.reserve(2 + obbs.size());

  const double c = std::cos(theta);
  const double s = std::sin(theta);

  std::array<double, 3> dcx{};
  std::array<double, 3> dcy{};
  for (std::size_t i = 0; i < 3; ++i) {
    dcx[i] = x + disc_offsets[i] * c;
    dcy[i] = y + disc_offsets[i] * s;
  }

  // 1) Left white-lane boundary hard constraint
  double h_left = 1e9;
  for (std::size_t i = 0; i < 3; ++i) {
    const double h = (y_lane_left_m - border_margin_m) - (dcy[i] + r_disc);
    h_left = std::min(h_left, h);
  }
  out_h.push_back(h_left);
  out_tags.push_back("lane_left");

  // 2) Right white-lane boundary hard constraint
  double h_right = 1e9;
  for (std::size_t i = 0; i < 3; ++i) {
    const double h = (dcy[i] - r_disc) - (y_lane_right_m + border_margin_m);
    h_right = std::min(h_right, h);
  }
  out_h.push_back(h_right);
  out_tags.push_back("lane_right");

  // 3) OBB obstacle hard constraints
  for (const auto & obb : obbs) {
    double h_obs = 1e9;
    for (std::size_t i = 0; i < 3; ++i) {
      const double sdf = AckermannCbfShield::point_to_obb_sdf(
        dcx[i], dcy[i], obb.cx, obb.cy, obb.half_len, obb.half_wid, obb.yaw_rad);
      const double h = sdf - r_disc - obb.margin_m;
      h_obs = std::min(h_obs, h);
    }
    out_h.push_back(h_obs);
    out_tags.push_back(obb.source_tag);
  }
}

EvalOutcome evaluate_candidate(
  double v,
  double omega,
  const AckermannCbfConfig & cfg,
  const std::array<double, 3> & disc_offsets,
  double r_disc,
  double y_lane_left_m,
  double y_lane_right_m,
  const std::vector<ObbObstacle> & obbs,
  const std::vector<double> & h0,
  const std::vector<const char *> & tags)
{
  EvalOutcome res{};
  const std::size_t n_c = h0.size();
  const double one_minus_gamma = 1.0 - cfg.cbf_gamma;

  double x = 0.0;
  double y = 0.0;
  double theta = 0.0;

  std::vector<double> h_prev = h0;
  std::vector<double> h_curr;
  std::vector<const char *> dummy_tags;

  double worst_violation = 0.0;
  const char * worst_viol_tag = "none";
  double min_clr = 1e9;
  const char * min_clr_tag = "none";

  for (int step = 0; step < cfg.horizon_steps; ++step) {
    double xn = 0.0;
    double yn = 0.0;
    double tn = 0.0;
    step_ackermann(x, y, theta, v, omega, cfg.step_dt_sec, xn, yn, tn);
    x = xn;
    y = yn;
    theta = tn;

    compute_barrier_values(
      x, y, theta, disc_offsets, r_disc,
      y_lane_left_m, y_lane_right_m, cfg.border_margin_m,
      obbs, h_curr, dummy_tags);

    for (std::size_t c = 0; c < n_c; ++c) {
      const double hc = h_curr[c];
      if (hc < min_clr) {
        min_clr = hc;
        min_clr_tag = tags[c];
      }

      if (h0[c] >= 0.0) {
        // Strict discrete CBF invariance: h(x_{k+1}) >= (1 - gamma) * h(x_k) and h >= 0
        const double req = std::max(0.0, one_minus_gamma * h_prev[c]);
        if (hc < req - 1e-9) {
          res.safe = false;
          const double viol = req - hc;
          if (viol > worst_violation) {
            worst_violation = viol;
            worst_viol_tag = tags[c];
          }
        }
      } else {
        // Already slightly inside margin at k=0: must not worsen and must recover by end of horizon
        if (hc < h_prev[c] - 2e-3) {
          res.safe = false;
          const double viol = (h_prev[c] - hc);
          if (viol > worst_violation) {
            worst_violation = viol;
            worst_viol_tag = tags[c];
          }
        }
      }
    }

    h_prev.swap(h_curr);
  }

  // Verify recovery by horizon end for any initially negative barrier
  for (std::size_t c = 0; c < n_c; ++c) {
    if (h0[c] < 0.0) {
      const double target_h = one_minus_gamma * h0[c];
      if (h_prev[c] < target_h - 1e-9) {
        res.safe = false;
        const double viol = target_h - h_prev[c];
        if (viol > worst_violation) {
          worst_violation = viol;
          worst_viol_tag = tags[c];
        }
      }
    }
  }

  res.min_clearance_m = (n_c > 0) ? min_clr : 0.0;
  res.limiting_source = res.safe ? min_clr_tag : worst_viol_tag;
  return res;
}

}  // namespace

AckermannCbfShield::AckermannCbfShield(const AckermannCbfConfig & config)
{
  set_config(config);
}

void AckermannCbfShield::set_config(const AckermannCbfConfig & config)
{
  config_ = config;
  config_.car_length_m = std::max(0.05, config_.car_length_m);
  config_.car_width_m = std::max(0.05, config_.car_width_m);
  config_.rear_axle_to_bumper_m = std::max(0.0, config_.rear_axle_to_bumper_m);
  config_.wheelbase_m = std::max(0.04, config_.wheelbase_m);
  config_.max_steer_rad = std::clamp(config_.max_steer_rad, 0.05, 1.25);
  config_.max_yaw_rate = std::max(0.1, config_.max_yaw_rate);
  config_.default_lane_width_m = std::max(0.20, config_.default_lane_width_m);
  config_.border_margin_m = std::max(0.0, config_.border_margin_m);
  config_.cone_margin_m = std::max(0.0, config_.cone_margin_m);
  config_.no_entry_margin_m = std::max(0.0, config_.no_entry_margin_m);
  config_.horizon_steps = std::clamp(config_.horizon_steps, 1, 30);
  config_.step_dt_sec = std::clamp(config_.step_dt_sec, 0.01, 0.25);
  config_.cbf_gamma = std::clamp(config_.cbf_gamma, 0.05, 1.0);
  config_.yaw_sample_count = std::clamp(config_.yaw_sample_count, 5, 101);
}

const AckermannCbfConfig & AckermannCbfShield::config() const
{
  return config_;
}

double AckermannCbfShield::point_to_obb_sdf(
  double px,
  double py,
  double cx,
  double cy,
  double half_len,
  double half_wid,
  double yaw_rad)
{
  const double dx = px - cx;
  const double dy = py - cy;
  const double c = std::cos(yaw_rad);
  const double s = std::sin(yaw_rad);
  const double lx = dx * c + dy * s;
  const double ly = -dx * s + dy * c;

  const double qx = std::abs(lx) - std::max(0.0, half_len);
  const double qy = std::abs(ly) - std::max(0.0, half_wid);
  const double out_x = std::max(qx, 0.0);
  const double out_y = std::max(qy, 0.0);
  const double outside_dist = std::hypot(out_x, out_y);
  const double inside_dist = std::min(std::max(qx, qy), 0.0);
  return outside_dist + inside_dist;
}

std::array<double, 3> AckermannCbfShield::compute_disc_offsets(const AckermannCbfConfig & config)
{
  const double r_disc = 0.5 * config.car_width_m;
  const double x_mid = config.rear_axle_to_bumper_m - 0.5 * config.car_length_m;
  const double half_span = std::max(0.0, 0.5 * config.car_length_m - r_disc);
  return {x_mid - half_span, x_mid, x_mid + half_span};
}

CbfShieldResult AckermannCbfShield::filter(
  double nom_linear_speed,
  double nom_yaw_rate,
  double y_lane_left_m,
  double y_lane_right_m,
  const std::vector<LocalObstacle> & obstacles) const
{
  CbfShieldResult out{};
  out.linear_speed = nom_linear_speed;
  out.yaw_rate = nom_yaw_rate;

  if (!std::isfinite(y_lane_left_m) || !std::isfinite(y_lane_right_m) ||
    y_lane_left_m <= y_lane_right_m + 0.10)
  {
    const double half_w = 0.5 * config_.default_lane_width_m;
    y_lane_left_m = half_w;
    y_lane_right_m = -half_w;
  }

  const double r_disc = 0.5 * config_.car_width_m;
  const std::array<double, 3> disc_offsets = compute_disc_offsets(config_);

  // Convert active red obstacles into initial rear-axle frame OBBs
  std::vector<ObbObstacle> obbs;
  obbs.reserve(obstacles.size());
  for (const auto & obs : obstacles) {
    if (obs.type != SemanticObjectType::RED_NO_ENTRY &&
      obs.type != SemanticObjectType::RED_CONE)
    {
      continue;
    }
    if (!std::isfinite(obs.d_bumper_m) || !std::isfinite(obs.y_lat_m) ||
      !std::isfinite(obs.rel_yaw_rad) || !std::isfinite(obs.d_min_m) ||
      !std::isfinite(obs.d_max_m) || !std::isfinite(obs.y_min_m) ||
      !std::isfinite(obs.y_max_m))
    {
      continue;
    }
    ObbObstacle o{};
    o.cx = obs.d_bumper_m + config_.rear_axle_to_bumper_m;
    o.cy = obs.y_lat_m;
    o.half_len = std::max(0.01, 0.5 * (obs.d_max_m - obs.d_min_m));
    o.half_wid = std::max(0.01, 0.5 * (obs.y_max_m - obs.y_min_m));
    o.yaw_rad = obs.rel_yaw_rad;
    if (obs.type == SemanticObjectType::RED_NO_ENTRY) {
      o.margin_m = config_.no_entry_margin_m;
      o.source_tag = "red_no_entry";
    } else {
      o.margin_m = config_.cone_margin_m;
      o.source_tag = "red_cone";
    }
    obbs.push_back(o);
  }

  std::vector<double> h0;
  std::vector<const char *> tags;
  compute_barrier_values(
    0.0, 0.0, 0.0, disc_offsets, r_disc,
    y_lane_left_m, y_lane_right_m, config_.border_margin_m,
    obbs, h0, tags);

  double h0_min = 1e9;
  const char * h0_min_tag = "none";
  for (std::size_t i = 0; i < h0.size(); ++i) {
    if (h0[i] < h0_min) {
      h0_min = h0[i];
      h0_min_tag = tags[i];
    }
  }
  if (h0.empty()) {
    h0_min = 0.0;
  }

  // Guard against NaN/Inf control inputs (fail-safe full stop)
  if (!std::isfinite(nom_linear_speed) || !std::isfinite(nom_yaw_rate)) {
    out.linear_speed = 0.0;
    out.yaw_rate = 0.0;
    out.intervened = true;
    out.min_clearance_m = h0_min;
    out.limiting_source = "invalid_cmd";
    return out;
  }

  // Stationary or reverse commands are passed through cleanly (Ackermann cannot pivot at v=0)
  if (nom_linear_speed <= 1e-6) {
    out.linear_speed = nom_linear_speed;
    out.yaw_rate = (std::abs(nom_linear_speed) <= 1e-6) ? 0.0 : nom_yaw_rate;
    out.intervened = (std::abs(nom_linear_speed) <= 1e-6 && std::abs(nom_yaw_rate) > 1e-6);
    out.min_clearance_m = h0_min;
    out.limiting_source = out.intervened ? "kinematic_limit" : h0_min_tag;
    return out;
  }

  const double tan_max_steer = std::tan(config_.max_steer_rad);
  const double nom_omega_lim = std::min(
    config_.max_yaw_rate,
    (std::abs(nom_linear_speed) / config_.wheelbase_m) * tan_max_steer);
  const double clamped_nom_omega = std::clamp(nom_yaw_rate, -nom_omega_lim, nom_omega_lim);

  // Check nominal command first
  const EvalOutcome nom_eval = evaluate_candidate(
    nom_linear_speed, clamped_nom_omega, config_, disc_offsets, r_disc,
    y_lane_left_m, y_lane_right_m, obbs, h0, tags);

  const bool nom_kin_clamped = std::abs(clamped_nom_omega - nom_yaw_rate) > 1e-6;
  if (nom_eval.safe) {
    out.linear_speed = nom_linear_speed;
    out.yaw_rate = clamped_nom_omega;
    out.intervened = nom_kin_clamped;
    out.min_clearance_m = nom_eval.min_clearance_m;
    out.limiting_source = nom_kin_clamped ? "kinematic_limit" : "none";
    return out;
  }

  const char * cause_tag = nom_eval.limiting_source;

  // Search candidate yaw rates across speed-reduction ladder: 1.0*v -> 0.6*v -> 0.3*v -> 0.0
  constexpr std::array<double, 3> kSpeedScales = {1.0, 0.6, 0.3};
  std::vector<double> candidates;
  candidates.reserve(static_cast<std::size_t>(config_.yaw_sample_count) + 2);

  for (const double scale : kSpeedScales) {
    const double v_cand = nom_linear_speed * scale;
    const double omega_lim = std::min(
      config_.max_yaw_rate,
      (std::abs(v_cand) / config_.wheelbase_m) * tan_max_steer);

    candidates.clear();
    candidates.push_back(std::clamp(nom_yaw_rate, -omega_lim, omega_lim));
    candidates.push_back(0.0);
    const int n = config_.yaw_sample_count;
    for (int i = 0; i < n; ++i) {
      const double t = (n > 1) ?
        (-1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n - 1)) : 0.0;
      candidates.push_back(t * omega_lim);
    }

    bool found_safe = false;
    double best_omega = 0.0;
    double best_dist = std::numeric_limits<double>::infinity();
    double best_clr = -1e9;

    for (const double w_cand : candidates) {
      const EvalOutcome ev = evaluate_candidate(
        v_cand, w_cand, config_, disc_offsets, r_disc,
        y_lane_left_m, y_lane_right_m, obbs, h0, tags);
      if (!ev.safe) {
        continue;
      }
      const double dist = std::abs(w_cand - nom_yaw_rate);
      if (!found_safe || dist < best_dist - 1e-9 ||
        (std::abs(dist - best_dist) <= 1e-9 && ev.min_clearance_m > best_clr))
      {
        found_safe = true;
        best_omega = w_cand;
        best_dist = dist;
        best_clr = ev.min_clearance_m;
      }
    }

    if (found_safe) {
      out.linear_speed = v_cand;
      out.yaw_rate = best_omega;
      out.intervened = true;
      out.min_clearance_m = best_clr;
      out.limiting_source = cause_tag;
      return out;
    }
  }

  // Final safe fallback: full stop
  out.linear_speed = 0.0;
  out.yaw_rate = 0.0;
  out.intervened = true;
  out.min_clearance_m = h0_min;
  out.limiting_source = (std::string(cause_tag) != "none") ? cause_tag : h0_min_tag;
  return out;
}

}  // namespace line_follower
