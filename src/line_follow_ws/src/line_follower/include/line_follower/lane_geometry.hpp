#pragma once

#include <optional>
#include <opencv2/core.hpp>

#include "line_follower/line_follower_types.hpp"

namespace line_follower
{

struct LaneGeometryConfig
{
  double far_band_top_ratio{0.34};
  double far_band_bottom_ratio{0.56};
  double near_band_top_ratio{0.62};
  double near_band_bottom_ratio{0.88};

  double near_band_weight{0.70};
  double far_band_weight{0.30};

  double column_white_ratio{0.30};
  int min_run_width{20};

  bool enable_continuous_curve_assist{false};
  double reverse_curve_min_error{0.12};
  double reverse_curve_threshold{0.35};
  int reverse_curve_confirm_frames{3};
  int reverse_curve_clear_frames{4};

  // 道路宽度模型、边界补偿与连续帧滤波（默认关闭宽道自适应，保持 fork 兼容）
  double nominal_near_width_ratio{0.72};
  double nominal_far_width_ratio{0.52};
  double wide_road_ratio_threshold{0.85};
  double wide_band_min_reliability{0.15};
  bool enable_width_adaptive_fusion{false};
  double wide_lane_boundary_blend{0.0};
  double observation_filter_alpha{1.0};
};

class LaneGeometry
{
public:
  LaneGeometry();
  explicit LaneGeometry(const LaneGeometryConfig & config);

  void set_config(const LaneGeometryConfig & config);
  const LaneGeometryConfig & config() const;

  static std::optional<double> find_band_center(
    const cv::Mat & binary,
    int y0,
    int y1,
    double reference_x,
    double column_white_ratio = 0.30,
    int min_run_width = 20);

  BandObservation observe_band(
    const cv::Mat & binary,
    int y0,
    int y1,
    double reference_x,
    double nominal_width_ratio,
    double dir_hint_norm) const;

  LaneObservation observe(
    const SegmentedFrame & seg,
    CurveAssistState & curve_state) const;

  ProcessingResult evaluate(
    const SegmentedFrame & seg,
    CurveAssistState & curve_state) const;

private:
  LaneGeometryConfig config_{};
};

}  // namespace line_follower
