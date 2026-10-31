#pragma once

#include <algorithm>
#include <opencv2/core.hpp>
#include <string>

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

struct RedAvoidState
{
  bool valid{false};

  bool red_line_detected{false};
  bool is_red_line{false};
  cv::Rect red_line_bbox_img{};

  cv::Rect bbox_img{};
  cv::Point2d center_img{};
  cv::Point2d bottom_img{};

  double aspect_ratio{0.0};
  double contour_area_ratio{0.0};
  double bottom_ratio{0.0};

  bool in_corridor{false};
  double risk{0.0};
  bool very_close{false};

  bool soft_red_logged{false};

  void clear_detection()
  {
    const bool prev_logged = soft_red_logged;
    *this = RedAvoidState{};
    soft_red_logged = prev_logged;
  }
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
