#pragma once

#include <algorithm>
#include <string>

#if defined(__has_include)
#if __has_include(<opencv2/core.hpp>) && !defined(LINE_FOLLOWER_NO_OPENCV)
#define LINE_FOLLOWER_HAS_OPENCV 1
#endif
#elif !defined(LINE_FOLLOWER_NO_OPENCV)
#define LINE_FOLLOWER_HAS_OPENCV 1
#endif

#if defined(LINE_FOLLOWER_HAS_OPENCV)
#include <opencv2/core.hpp>
#else
namespace cv
{
struct Rect
{
  int x{0};
  int y{0};
  int width{0};
  int height{0};
  Rect() = default;
  Rect(int x_, int y_, int w_, int h_)
  : x(x_), y(y_), width(w_), height(h_) {}
};

struct Point2f
{
  float x{0.0F};
  float y{0.0F};
  Point2f() = default;
  Point2f(float x_, float y_)
  : x(x_), y(y_) {}
};

struct Point2d
{
  double x{0.0};
  double y{0.0};
  Point2d() = default;
  Point2d(double x_, double y_)
  : x(x_), y(y_) {}
};

struct Mat
{
  int rows{0};
  int cols{0};
  bool empty() const {return rows <= 0 || cols <= 0;}
};
}  // namespace cv
#endif

namespace line_follower
{

enum class TrackState
{
  LOST = 0,
  TRACKING,
  YELLOW_LINE_DETECTED
};

inline std::string track_state_to_string(TrackState state)
{
  switch (state) {
    case TrackState::TRACKING:
      return "tracking";
    case TrackState::YELLOW_LINE_DETECTED:
      return "yellow_line_detected";
    case TrackState::LOST:
    default:
      return "lost";
  }
}

inline double clamp_value(double value, double min_value, double max_value)
{
  return std::max(min_value, std::min(value, max_value));
}

struct BandObservation
{
  bool valid{false};
  double center_px{0.0};
  double raw_center_px{0.0};
  double center_norm{0.0};
  int left_px{0};
  int right_px{0};
  int width_px{0};
  double width_ratio{0.0};
  bool left_clipped{false};
  bool right_clipped{false};
  double reliability{1.0};

  // Explicit lane-edge geometry & multi-run fork detection (finals)
  double left_edge_px{0.0};
  double right_edge_px{0.0};
  bool left_valid{false};
  bool right_valid{false};

  int run_count{0};
  bool has_fork_split{false};
  double left_branch_center_px{0.0};
  double right_branch_center_px{0.0};
  bool left_branch_valid{false};
  bool right_branch_valid{false};
};

struct LaneObservation
{
  bool valid{false};
  bool yellow_detected{false};
  cv::Rect roi{};

  BandObservation near_band{};
  BandObservation far_band{};

  double near_center_px{0.0};
  double far_center_px{0.0};
  double near_center_norm{0.0};
  double far_center_norm{0.0};

  double near_width_ratio{0.0};
  double far_width_ratio{0.0};
  double lane_width_ratio{0.0};

  // Explicit near-band boundary coordinates (full-width pixel x)
  double near_left_edge_px{0.0};
  double near_right_edge_px{0.0};
  bool near_left_valid{false};
  bool near_right_valid{false};

  // Multi-run fork split detection & branch selection state
  bool has_fork_split{false};
  double left_branch_center_px{0.0};
  double right_branch_center_px{0.0};
  bool left_branch_valid{false};
  bool right_branch_valid{false};
  int active_branch_choice{0};

  double lateral_error{0.0};
  double heading_proxy{0.0};
  double curvature_proxy{0.0};
  double confidence{0.0};

  cv::Point2f target_centroid{0.0F, 0.0F};
};

struct ProcessingResult
{
  bool line_detected{false};
  bool yellow_line_detected{false};

  cv::Rect bounding_box{};
  cv::Point2f centroid{0.0F, 0.0F};

  double error{0.0};
  TrackState state{TrackState::LOST};

  cv::Mat roi_bgr;
  cv::Mat binary_mask;

  int roi_top{0};
  int roi_bottom{0};

  // Geometric yellow bar, YOLO-hybrid refinement & dead-end-ahead cues
  bool yellow_is_geometric{false};
  double yellow_bottom_y_full{0.0};
  double yellow_top_y_full{0.0};
  double yellow_bar_width_px{0.0};
  double yellow_center_x_full{0.0};
  cv::Rect yellow_bbox_full{};
  bool yellow_from_yolo_hybrid{false};
  double yellow_ahead_blue_ratio{0.0};
  double yellow_ahead_dark_ratio{0.0};
  bool yellow_has_dead_end_ahead{false};
  std::string active_lane_seg_mode{"relative_border"};

  LaneObservation observation{};
};

struct SegmentedFrame
{
  cv::Mat roi_bgr;
  cv::Mat binary_mask;

  int roi_top{0};
  int roi_bottom{0};
  int frame_width{0};
  int frame_height{0};

  bool yellow_line_detected{false};
  bool contour_valid{false};
  cv::Rect bounding_box{};
  cv::Point2f contour_centroid{0.0F, 0.0F};

  // Geometric yellow bar, YOLO-hybrid refinement & dead-end-ahead cues
  bool yellow_is_geometric{false};
  double yellow_bottom_y_full{0.0};
  double yellow_top_y_full{0.0};
  double yellow_bar_width_px{0.0};
  double yellow_center_x_full{0.0};
  cv::Rect yellow_bbox_full{};
  bool yellow_from_yolo_hybrid{false};
  double yellow_ahead_blue_ratio{0.0};
  double yellow_ahead_dark_ratio{0.0};
  bool yellow_has_dead_end_ahead{false};
  std::string active_lane_seg_mode{"relative_border"};
};

struct CurveAssistState
{
  bool reverse_curve_active{false};
  int reverse_confirm_count{0};
  int reverse_clear_count{0};
  bool near_valid{false};
  bool far_valid{false};
  double near_error{0.0};
  double far_error{0.0};
  double curve_level{0.0};

  bool filter_initialized{false};
  double filtered_error{0.0};
  double filtered_heading{0.0};
  double prev_far_center_px{0.0};
  double last_confidence{0.0};
  double last_heading_proxy{0.0};
  double last_curvature_proxy{0.0};
  double last_lane_width_ratio{0.0};
};

struct MotionState
{
  bool valid{false};
  double last_linear{0.08};
  double last_yaw{0.0};
};

struct MotionCommand
{
  double linear_speed{0.0};
  double yaw_rate{0.0};
  bool publish_stop{false};
};

}  // namespace line_follower
