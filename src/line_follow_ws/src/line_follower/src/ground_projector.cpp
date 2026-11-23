#include "line_follower/ground_projector.hpp"

#include <algorithm>
#include <cmath>

namespace line_follower
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

bool invert_3x3(const std::array<double, 9> & m, std::array<double, 9> & inv)
{
  for (double v : m) {
    if (!std::isfinite(v)) {
      return false;
    }
  }

  const double c00 = m[4] * m[8] - m[5] * m[7];
  const double c01 = -(m[3] * m[8] - m[5] * m[6]);
  const double c02 = m[3] * m[7] - m[4] * m[6];

  const double c10 = -(m[1] * m[8] - m[2] * m[7]);
  const double c11 = m[0] * m[8] - m[2] * m[6];
  const double c12 = -(m[0] * m[7] - m[1] * m[6]);

  const double c20 = m[1] * m[5] - m[2] * m[4];
  const double c21 = -(m[0] * m[5] - m[2] * m[3]);
  const double c22 = m[0] * m[4] - m[1] * m[3];

  const double det = m[0] * c00 + m[1] * c01 + m[2] * c02;
  if (!std::isfinite(det) || std::abs(det) < 1e-12) {
    return false;
  }

  const double inv_det = 1.0 / det;
  inv = {
    c00 * inv_det, c10 * inv_det, c20 * inv_det,
    c01 * inv_det, c11 * inv_det, c21 * inv_det,
    c02 * inv_det, c12 * inv_det, c22 * inv_det};

  for (double v : inv) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

}  // namespace

GroundProjector::GroundProjector()
{
  set_config(GroundProjectorConfig{});
}

GroundProjector::GroundProjector(const GroundProjectorConfig & config)
{
  set_config(config);
}

void GroundProjector::set_config(const GroundProjectorConfig & config)
{
  config_ = config;
  config_.cam_height_m = std::max(0.01, config_.cam_height_m);
  config_.cam_pitch_deg = std::clamp(config_.cam_pitch_deg, -45.0, 85.0);
  config_.focal_px = std::max(1.0, config_.focal_px);
  config_.image_width = std::max(1, config_.image_width);
  config_.image_height = std::max(1, config_.image_height);
  if (config_.max_valid_d_m < config_.min_valid_d_m) {
    std::swap(config_.min_valid_d_m, config_.max_valid_d_m);
  }
  update_cached_math();
}

const GroundProjectorConfig & GroundProjector::config() const
{
  return config_;
}

bool GroundProjector::set_homography_from_vector(
  const std::vector<double> & h9,
  bool enable)
{
  if (h9.size() != 9) {
    return false;
  }
  std::array<double, 9> candidate{};
  for (size_t i = 0; i < 9; ++i) {
    candidate[i] = h9[i];
  }
  std::array<double, 9> candidate_inv{};
  if (!invert_3x3(candidate, candidate_inv)) {
    return false;
  }
  config_.homography_matrix = candidate;
  config_.use_homography = enable;
  inv_homography_ = candidate_inv;
  homography_valid_ = true;
  return true;
}

bool GroundProjector::using_homography() const
{
  return config_.use_homography && homography_valid_;
}

bool GroundProjector::homography_valid() const
{
  return homography_valid_;
}

double GroundProjector::pinhole_horizon_v() const
{
  if (std::abs(cos_pitch_) < 1e-6) {
    return -1e9;
  }
  return config_.cy - config_.focal_px * (sin_pitch_ / cos_pitch_);
}

void GroundProjector::update_cached_math()
{
  const double pitch_rad = config_.cam_pitch_deg * (kPi / 180.0);
  sin_pitch_ = std::sin(pitch_rad);
  cos_pitch_ = std::cos(pitch_rad);
  homography_valid_ = invert_3x3(config_.homography_matrix, inv_homography_);
}

bool GroundProjector::pixel_to_ground(
  double u,
  double v,
  double & d_bumper_m,
  double & y_lat_m) const
{
  if (!std::isfinite(u) || !std::isfinite(v)) {
    return false;
  }

  if (using_homography()) {
    const auto & h = config_.homography_matrix;
    const double w = h[6] * u + h[7] * v + h[8];
    if (!std::isfinite(w) || std::abs(w) < 1e-9) {
      return false;
    }
    const double d = (h[0] * u + h[1] * v + h[2]) / w;
    const double y = (h[3] * u + h[4] * v + h[5]) / w;
    if (!std::isfinite(d) || !std::isfinite(y)) {
      return false;
    }
    if (d < config_.min_valid_d_m || d > config_.max_valid_d_m) {
      return false;
    }
    d_bumper_m = d;
    y_lat_m = y;
    return true;
  }

  // Pinhole camera model:
  //   Camera axes in ground frame (+X fwd, +Y left, +Z up):
  //     e_xc (right) = (0, -1, 0)
  //     e_yc (down)  = (-sin_pitch, 0, -cos_pitch)
  //     e_zc (fwd)   = ( cos_pitch, 0, -sin_pitch)
  //   Ray direction r = x_n * e_xc + y_n * e_yc + e_zc
  //     rz = -(sin_pitch + y_n * cos_pitch) < 0 hits ground plane z = 0 from z = cam_height_m
  const double x_n = (u - config_.cx) / config_.focal_px;
  const double y_n = (v - config_.cy) / config_.focal_px;
  const double denom = sin_pitch_ + y_n * cos_pitch_;
  if (!std::isfinite(denom) || denom <= 1e-6) {
    return false;
  }

  const double d_cam_m = config_.cam_height_m * (cos_pitch_ - y_n * sin_pitch_) / denom;
  const double y = -config_.cam_height_m * x_n / denom;
  const double d = d_cam_m - config_.front_overhang_m;

  if (!std::isfinite(d_cam_m) || !std::isfinite(d) || !std::isfinite(y)) {
    return false;
  }
  if (d_cam_m <= 0.0 || d < config_.min_valid_d_m || d > config_.max_valid_d_m) {
    return false;
  }

  d_bumper_m = d;
  y_lat_m = y;
  return true;
}

bool GroundProjector::ground_to_pixel(
  double d_bumper_m,
  double y_lat_m,
  double & u,
  double & v) const
{
  if (!std::isfinite(d_bumper_m) || !std::isfinite(y_lat_m)) {
    return false;
  }

  if (using_homography()) {
    const auto & hi = inv_homography_;
    const double w = hi[6] * d_bumper_m + hi[7] * y_lat_m + hi[8];
    if (!std::isfinite(w) || std::abs(w) < 1e-9) {
      return false;
    }
    const double px_u = (hi[0] * d_bumper_m + hi[1] * y_lat_m + hi[2]) / w;
    const double px_v = (hi[3] * d_bumper_m + hi[4] * y_lat_m + hi[5]) / w;
    if (!std::isfinite(px_u) || !std::isfinite(px_v)) {
      return false;
    }
    u = px_u;
    v = px_v;
    return true;
  }

  const double d_cam_m = d_bumper_m + config_.front_overhang_m;
  const double x_c = -y_lat_m;
  const double y_c = -d_cam_m * sin_pitch_ + config_.cam_height_m * cos_pitch_;
  const double z_c = d_cam_m * cos_pitch_ + config_.cam_height_m * sin_pitch_;
  if (!std::isfinite(z_c) || z_c <= 1e-6) {
    return false;
  }

  const double px_u = config_.cx + config_.focal_px * (x_c / z_c);
  const double px_v = config_.cy + config_.focal_px * (y_c / z_c);
  if (!std::isfinite(px_u) || !std::isfinite(px_v)) {
    return false;
  }

  u = px_u;
  v = px_v;
  return true;
}

double GroundProjector::lane_width_px_at_row(double v, double lane_width_m) const
{
  if (!std::isfinite(v) || !std::isfinite(lane_width_m) || lane_width_m <= 0.0) {
    return 0.0;
  }

  double d = 0.0;
  double y0 = 0.0;
  if (!pixel_to_ground(config_.cx, v, d, y0)) {
    return 0.0;
  }

  double u_left = 0.0;
  double v_left = 0.0;
  double u_right = 0.0;
  double v_right = 0.0;
  if (!ground_to_pixel(d, y0 + 0.5 * lane_width_m, u_left, v_left) ||
      !ground_to_pixel(d, y0 - 0.5 * lane_width_m, u_right, v_right))
  {
    return 0.0;
  }

  return std::abs(u_right - u_left);
}

}  // namespace line_follower
