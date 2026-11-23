#pragma once

#include <optional>
#include <string>
#include <vector>

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

  // ==========================================================================
  // Finals pluggable lane segmentation ("legacy_hsv" | "relative_border" | "hybrid")
  // ==========================================================================
  std::string lane_seg_mode{"legacy_hsv"};

  // Relative border / illumination-invariant parameters (zero manual calibration)
  int blue_margin{22};            // B - max(R, G) > blue_margin -> blue mat exterior
  int lab_blue_b_max{116};        // OpenCV Lab b < lab_blue_b_max -> blue reject
  int black_block_size{31};       // odd adaptiveThreshold block size for black border
  double black_c_offset{14.0};    // adaptiveThreshold C offset for dark ridge
  int black_v_max{115};           // pixels brighter than this are not black borders
  double lab_chroma_radius{24.0}; // max Lab (a,b) distance from intra-frame track median
  double lab_l_min_offset{38.0};  // min Lab L drop below intra-frame track median L
  int hybrid_white_s_max{75};     // relaxed HSV saturation bound in "hybrid" mode
  int hybrid_white_v_min{65};     // relaxed HSV value bound in "hybrid" mode

  // Semantic hole-filling: paint digit cards / yellow bars white before band extraction
  bool enable_semantic_hole_fill{false};
  int semantic_hole_fill_dilate_px{2};

  // ==========================================================================
  // Geometric yellow bar, YOLO-guided CV refinement & dead-end-ahead detection
  // yellow_detect_mode: "legacy_area" | "geometric_cv" | "yolo_cv_hybrid"
  // ==========================================================================
  std::string yellow_detect_mode{"legacy_area"};
  double yellow_min_aspect_ratio{2.5};
  double yellow_min_width_ratio{0.20};
  double yellow_min_area_px{120.0};
  int yellow_ahead_strip_height_px{28};
  double yellow_dead_end_blue_ratio{0.35};
  double yellow_dead_end_dark_ratio{0.20};
  double yellow_dead_end_combined_ratio{0.50};
};

class ImageProcessor
{
public:
  ImageProcessor();
  explicit ImageProcessor(const ImageProcessorConfig & config);

  void set_config(const ImageProcessorConfig & config);
  const ImageProcessorConfig & config() const;

  // Backward-compatible single-argument entry point
  std::optional<SegmentedFrame> segment(const cv::Mat & frame_bgr) const;

  // Finals entry point supporting semantic hole-fill boxes and YOLO yellow boxes
  // (all cv::Rect coordinates are in full-image pixel space)
  std::optional<SegmentedFrame> segment(
    const cv::Mat & frame_bgr,
    const std::vector<cv::Rect> & fill_white_rects,
    const std::vector<cv::Rect> & yolo_yellow_rects = {}) const;

private:
  ImageProcessorConfig config_{};
  cv::Mat morph_element_;
};

}  // namespace line_follower
