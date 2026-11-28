#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "line_follower/bev_dwa_planner.hpp"
#include "line_follower/ground_projector.hpp"
#include "line_follower/image_processor.hpp"

namespace line_follower
{

struct BevMapperConfig
{
  // Blue mat detection (matches relative_border chroma rules)
  int blue_margin{22};
  int lab_blue_b_max{116};

  // HSV red obstacle thresholds (NEVER masked by YOLO!)
  int red_h_low_1{0};
  int red_h_high_1{10};
  int red_h_low_2{160};
  int red_h_high_2{180};
  int red_s_min{90};
  int red_v_min{70};
  double min_red_area_px{65.0};

  // Physical YOLO digit gate thresholds
  double digit_max_red_ratio{0.08};
  double digit_max_blue_ratio{0.68};
  double digit_min_ground_w_m{0.035};
  double digit_max_ground_w_m{0.30};
};

struct BevRedOverlayBox
{
  cv::Rect bbox_img{};
  bool is_no_entry{false};
  double d_bumper_m{0.0};
  double y_lat_m{0.0};
};

class BevMapper
{
public:
  BevMapper() = default;
  explicit BevMapper(const BevMapperConfig & config);

  void set_config(const BevMapperConfig & config);
  const BevMapperConfig & config() const { return config_; }

  // Build / refresh static 160x160 BEV->(u,v) LUT when GroundProjector or image size changes
  void ensure_lut(const GroundProjector & projector, int img_w, int img_h);

  // Build live 160x160 BEV grid (FREE / WALL_BLUE / OBS_CONE / OBS_NO_ENTRY / UNKNOWN)
  // - Non-blue dark digit ink inside white road span stays FREE (never digs holes!)
  // - Tall 3D red cones project ONLY via bottom ground-contact edge (zero 1m comet tail!)
  // - Flat red no-entry patches project full ground footprint
  void build_live_grid(
    const cv::Mat & frame_bgr,
    const cv::Mat & roi_white_mask,
    int roi_top,
    int roi_bottom,
    const GroundProjector & projector,
    std::array<uint8_t, kBevCells> & out_grid,
    std::vector<BevRedOverlayBox> & out_red_boxes);

  // Physical pixel & ground-size sanity gate for YOLO digit detections
  bool validate_yolo_digit(
    const cv::Mat & frame_bgr,
    const cv::Rect & bbox_img,
    const GroundProjector & projector,
    double & out_d_m,
    double & out_y_m) const;

  // Render crisp 200x200 live top-down BEV + 29 candidate Ackermann arcs + compass HUD
  static cv::Mat render_bev_inset(
    const std::array<uint8_t, kBevCells> & fused_grid,
    const BevDwaPlanResult & plan,
    const PreFlightStatus & preflight,
    bool yellow_visible,
    double yellow_d_m,
    int inset_size = 220);

private:
  BevMapperConfig config_{};
  bool lut_ready_{false};
  int lut_img_w_{0};
  int lut_img_h_{0};
  bool lut_use_homography_{false};
  std::array<double, 9> lut_h_{};
  std::array<int32_t, kBevCells> lut_u_{};
  std::array<int32_t, kBevCells> lut_v_{};
  cv::Mat last_road_span_{};
  int last_roi_top_{0};
};

}  // namespace line_follower
