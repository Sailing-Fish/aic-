#include "line_follower/debug_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv2/imgproc.hpp>

namespace line_follower
{

cv::Mat DebugRenderer::render(
  const cv::Mat & frame_bgr,
  const ProcessingResult & result,
  const CurveAssistState & curve_state,
  const RedAvoidState & red_state,
  bool red_overlay_valid,
  double red_divider_angle_rad,
  bool enable_continuous_curve_assist) const
{
  if (frame_bgr.empty() || result.binary_mask.empty()) {
    return cv::Mat{};
  }

  cv::Mat overlay = frame_bgr.clone();
  cv::Mat mask_color;
  cv::cvtColor(result.binary_mask, mask_color, cv::COLOR_GRAY2BGR);

  cv::Mat blended;
  cv::addWeighted(result.roi_bgr, 0.6, mask_color, 0.4, 0.0, blended);

  blended.copyTo(
    overlay(cv::Rect(
      0,
      result.roi_top,
      result.roi_bgr.cols,
      result.roi_bgr.rows)));

  if (result.line_detected) {
    if (result.bounding_box.width > 0 && result.bounding_box.height > 0) {
      cv::rectangle(
        overlay,
        cv::Rect(
          result.bounding_box.x,
          result.bounding_box.y + result.roi_top,
          result.bounding_box.width,
          result.bounding_box.height),
        cv::Scalar(0, 255, 0),
        2);
    }

    const cv::Point centroid(
      static_cast<int>(result.centroid.x),
      static_cast<int>(result.centroid.y) + result.roi_top);

    cv::circle(overlay, centroid, 8, cv::Scalar(0, 0, 255), -1);

    cv::line(
      overlay,
      cv::Point(overlay.cols / 2, result.roi_top),
      cv::Point(overlay.cols / 2, result.roi_bottom),
      cv::Scalar(0, 255, 255),
      2);

    cv::Point2d base(
      static_cast<double>(frame_bgr.cols) * 0.5,
      static_cast<double>(result.roi_bottom));

    cv::Point2d v_line(
      static_cast<double>(result.centroid.x) - base.x,
      static_cast<double>(result.centroid.y + result.roi_top) - base.y);

    if (red_divider_angle_rad != 0.0) {
      const double angle =
        v_line.x >= 0.0 ? red_divider_angle_rad : -red_divider_angle_rad;
      const double c = std::cos(angle);
      const double s = std::sin(angle);
      v_line = cv::Point2d(
        v_line.x * c - v_line.y * s,
        v_line.x * s + v_line.y * c);
    }

    const cv::Point2d target = base + v_line;
    cv::line(overlay, base, target, cv::Scalar(0, 255, 255), 2);

    if (red_overlay_valid && red_state.valid) {
      cv::rectangle(
        overlay,
        red_state.bbox_img,
        cv::Scalar(0, 0, 255),
        2);

      cv::circle(
        overlay,
        red_state.center_img,
        6,
        cv::Scalar(0, 0, 255),
        -1);

      cv::circle(
        overlay,
        red_state.bottom_img,
        6,
        cv::Scalar(0, 255, 255),
        -1);

      char red_text[200];
      std::snprintf(
        red_text,
        sizeof(red_text),
        "%s bottom=%.2f risk=%.2f area=%.4f %s%s",
        red_state.is_red_line ? "RED_LINE" : "CONE",
        red_state.bottom_ratio,
        red_state.risk,
        red_state.contour_area_ratio,
        red_state.in_corridor ? "IN_PATH " : "OUT_PATH ",
        red_state.very_close ? "VERY_CLOSE" : "");

      cv::putText(
        overlay,
        red_text,
        cv::Point(10, 90),
        cv::FONT_HERSHEY_SIMPLEX,
        0.50,
        cv::Scalar(0, 0, 255),
        2);
    }

    if (red_state.red_line_detected) {
      cv::rectangle(
        overlay,
        red_state.red_line_bbox_img,
        cv::Scalar(255, 0, 255),
        2);

      cv::putText(
        overlay,
        "RED LINE AVOID",
        cv::Point(
          red_state.red_line_bbox_img.x,
          std::max(20, red_state.red_line_bbox_img.y - 5)),
        cv::FONT_HERSHEY_SIMPLEX,
        0.50,
        cv::Scalar(255, 0, 255),
        2);
    }
  }

  if (enable_continuous_curve_assist) {
    const char * curve_mode =
      curve_state.reverse_curve_active ? "REV_CURVE" : "NORMAL";
    char curve_text[220];
    std::snprintf(
      curve_text,
      sizeof(curve_text),
      "%s near=%+.2f far=%+.2f lvl=%.2f conf=%.2f hd=%+.2f w=%.2f",
      curve_mode,
      curve_state.near_error,
      curve_state.far_error,
      curve_state.curve_level,
      result.observation.confidence,
      result.observation.heading_proxy,
      result.observation.lane_width_ratio);
    cv::putText(
      overlay,
      curve_text,
      cv::Point(10, 60),
      cv::FONT_HERSHEY_SIMPLEX,
      0.50,
      cv::Scalar(255, 255, 0),
      2);
  }

  if (result.yellow_line_detected) {
    cv::putText(
      overlay,
      "YELLOW LINE DETECTED!",
      cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX,
      1.0,
      cv::Scalar(0, 255, 255),
      2);
  }

  return overlay;
}

}  // namespace line_follower
