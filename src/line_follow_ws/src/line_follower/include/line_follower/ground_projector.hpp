#pragma once

#include <array>
#include <vector>

namespace line_follower
{

// ============================================================================
// GroundProjector — Pure C++17 STL (zero OpenCV / ROS2 dependencies)
//
// Vehicle ground-plane coordinate convention (ROS REP-103):
//   Origin (d_bumper_m = 0, y_lat_m = 0):
//     Ground projection of the front foam bumper center.
//   +d_bumper_m : forward ahead of front bumper (meters).
//                 Negative values lie beside/behind the front bumper.
//   +y_lat_m    : LEFT of the vehicle centerline (meters).
//                 Negative values lie to the RIGHT.
//
// Image pixel coordinate convention:
//   u : horizontal pixel [0, image_width), increasing rightward.
//   v : vertical pixel   [0, image_height), increasing downward.
//
// Supports two projection backends:
//   1. Calibrated 3x3 homography H_img2ground (row-major 9 doubles) mapping
//      [u, v, 1]^T -> [w * d_bumper_m, w * y_lat_m, w]^T.
//   2. Automatic distortion-free pinhole fallback using physical car camera
//      extrinsics (cam_height_m, cam_pitch_deg, focal_px, cx, cy, front_overhang_m).
// ============================================================================
struct GroundProjectorConfig
{
  bool use_homography{false};
  std::array<double, 9> homography_matrix{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0};

  // Pinhole fallback defaults matched to AIC 2026 car (2.8mm rectilinear 105-deg lens)
  double cam_height_m{0.11};
  double cam_pitch_deg{12.5};
  double focal_px{325.0};
  double cx{320.0};
  double cy{240.0};
  double front_overhang_m{0.06};

  int image_width{640};
  int image_height{480};

  // Sanity range for valid ground projections (meters from front bumper)
  double min_valid_d_m{-0.5};
  double max_valid_d_m{6.0};
};

class GroundProjector
{
public:
  GroundProjector();
  explicit GroundProjector(const GroundProjectorConfig & config);

  void set_config(const GroundProjectorConfig & config);
  const GroundProjectorConfig & config() const;

  // Convenience helper for loading 9-element row-major homography from ROS2 param array.
  // Returns true if h9 has 9 finite values and an invertible 3x3 matrix.
  bool set_homography_from_vector(const std::vector<double> & h9, bool enable = true);

  bool using_homography() const;
  bool homography_valid() const;

  // Horizon row v (pixels) in pinhole model (pixels above horizon cannot hit ground).
  double pinhole_horizon_v() const;

  // Project image pixel (u, v) onto ground plane (d_bumper_m forward, y_lat_m left).
  bool pixel_to_ground(double u, double v, double & d_bumper_m, double & y_lat_m) const;

  // Project ground point (d_bumper_m forward, y_lat_m left) back to image pixel (u, v).
  bool ground_to_pixel(double d_bumper_m, double y_lat_m, double & u, double & v) const;

  // Compute expected pixel span of a lane of width `lane_width_m` centered at row `v`.
  // Returns 0.0 if row `v` cannot be projected onto the ground plane.
  double lane_width_px_at_row(double v, double lane_width_m = 0.43) const;

private:
  void update_cached_math();

  GroundProjectorConfig config_{};

  // Cached pinhole trig
  double sin_pitch_{0.0};
  double cos_pitch_{1.0};

  // Cached analytic inverse of homography_matrix (ground -> image)
  bool homography_valid_{false};
  std::array<double, 9> inv_homography_{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0};
};

}  // namespace line_follower
