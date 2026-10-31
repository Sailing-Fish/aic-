#pragma once

#include <optional>
#include <opencv2/core.hpp>

#include "line_follower/line_follower_types.hpp"

namespace line_follower
{

struct ImageProcessorConfig
{
  double roi_top_ratio{0.5};
  double roi_bottom_ratio{0.05};

  int gaussian_kernel{5};
  double binary_threshold{160.0};
  int morph_kernel{5};
  int morph_iterations{2};
  double min_contour_area{2000.0};
  bool invert_binary{false};

  bool use_white_filter{false};
  int white_h_low{0};
  int white_h_high{180};
  int white_s_max{60};
  int white_v_min{180};

  bool use_yellow_filter{false};
  int yellow_h_low{20};
  int yellow_h_high{40};
  int yellow_s_min{80};
  int yellow_v_min{120};
  double yellow_area_ratio_threshold{0.1};
};

class ImageProcessor
{
public:
  ImageProcessor();
  explicit ImageProcessor(const ImageProcessorConfig & config);

  void set_config(const ImageProcessorConfig & config);
  const ImageProcessorConfig & config() const;

  std::optional<SegmentedFrame> segment(const cv::Mat & frame_bgr) const;

private:
  ImageProcessorConfig config_{};
  cv::Mat morph_element_;
};

}  // namespace line_follower
