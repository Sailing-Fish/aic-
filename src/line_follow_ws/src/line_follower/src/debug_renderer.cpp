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
  const FinalsDebugOverlay empty_overlay{};
  return render(
    frame_bgr,
    result,
    curve_state,
    red_state,
    red_overlay_valid,
    red_divider_angle_rad,
    enable_continuous_curve_assist,
    empty_overlay);
}

cv::Mat DebugRenderer::render(
  const cv::Mat & frame_bgr,
  const ProcessingResult & result,
  const CurveAssistState & curve_state,
  const RedAvoidState & red_state,
  bool red_overlay_valid,
  double red_divider_angle_rad,
  bool enable_continuous_curve_assist,
  const FinalsDebugOverlay & finals_overlay) const
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

  // ==========================================================================
  // Optional Finals Overlay (corridor, yellow distance, obstacles, fork/finish)
  // ==========================================================================
  if (finals_overlay.enabled) {
    // 1. Geometric yellow bar box & exact bottom edge
    if (result.yellow_is_geometric && result.yellow_bbox_full.width > 0) {
      cv::rectangle(overlay, result.yellow_bbox_full, cv::Scalar(0, 255, 255), 2);
      const int y_bot = std::clamp(
        static_cast<int>(std::round(result.yellow_bottom_y_full)),
        0,
        overlay.rows - 1);
      cv::line(
        overlay,
        cv::Point(result.yellow_bbox_full.x, y_bot),
        cv::Point(result.yellow_bbox_full.x + result.yellow_bbox_full.width, y_bot),
        cv::Scalar(0, 165, 255),
        2);
    }

    // 2. Explicit near-band lane edges & fork split branch markers
    const int near_draw_y = std::clamp(
      result.roi_top + static_cast<int>((result.roi_bottom - result.roi_top) * 0.72),
      0,
      overlay.rows - 1);
    if (result.observation.near_left_valid) {
      const int lx = std::clamp(
        static_cast<int>(std::round(result.observation.near_left_edge_px)),
        0,
        overlay.cols - 1);
      cv::line(
        overlay,
        cv::Point(lx, std::max(0, near_draw_y - 12)),
        cv::Point(lx, std::min(overlay.rows - 1, near_draw_y + 12)),
        cv::Scalar(255, 255, 0),
        2);
    }
    if (result.observation.near_right_valid) {
      const int rx = std::clamp(
        static_cast<int>(std::round(result.observation.near_right_edge_px)),
        0,
        overlay.cols - 1);
      cv::line(
        overlay,
        cv::Point(rx, std::max(0, near_draw_y - 12)),
        cv::Point(rx, std::min(overlay.rows - 1, near_draw_y + 12)),
        cv::Scalar(255, 255, 0),
        2);
    }

    if (result.observation.has_fork_split) {
      const int fork_y = std::clamp(
        result.roi_top + static_cast<int>((result.roi_bottom - result.roi_top) * 0.42),
        0,
        overlay.rows - 1);
      if (result.observation.left_branch_valid) {
        const int bx = std::clamp(
          static_cast<int>(std::round(result.observation.left_branch_center_px)),
          0,
          overlay.cols - 1);
        cv::circle(
          overlay,
          cv::Point(bx, fork_y),
          6,
          finals_overlay.preferred_branch < 0 ? cv::Scalar(0, 255, 0) : cv::Scalar(200, 200, 0),
          2);
      }
      if (result.observation.right_branch_valid) {
        const int bx = std::clamp(
          static_cast<int>(std::round(result.observation.right_branch_center_px)),
          0,
          overlay.cols - 1);
        cv::circle(
          overlay,
          cv::Point(bx, fork_y),
          6,
          finals_overlay.preferred_branch > 0 ? cv::Scalar(0, 255, 0) : cv::Scalar(200, 200, 0),
          2);
      }
    }

    // 3. Safe lateral corridor bar & target y_ref marker
    if (finals_overlay.corridor_valid &&
        finals_overlay.corridor_left_px >= 0 &&
        finals_overlay.corridor_right_px >= 0)
    {
      const int cy = (finals_overlay.corridor_row_y >= 0) ?
        std::clamp(finals_overlay.corridor_row_y, 0, overlay.rows - 1) :
        near_draw_y;
      const int x0 = std::clamp(finals_overlay.corridor_left_px, 0, overlay.cols - 1);
      const int x1 = std::clamp(finals_overlay.corridor_right_px, 0, overlay.cols - 1);
      cv::line(overlay, cv::Point(x0, cy), cv::Point(x1, cy), cv::Scalar(0, 255, 0), 4);
      if (finals_overlay.corridor_ref_px >= 0) {
        const int rx = std::clamp(finals_overlay.corridor_ref_px, 0, overlay.cols - 1);
        cv::circle(overlay, cv::Point(rx, cy), 6, cv::Scalar(255, 0, 255), -1);
      }
    }

    // 4. Detection boxes (obstacles, no-entry, digits, yellow)
    for (const auto & det : finals_overlay.detections) {
      if (det.bbox_img.width <= 0 || det.bbox_img.height <= 0) {
        continue;
      }
      cv::Scalar color(200, 200, 200);
      if (det.is_no_entry) {
        color = cv::Scalar(255, 0, 255);
      } else if (det.is_blocking) {
        color = cv::Scalar(0, 0, 255);
      } else if (det.is_yellow) {
        color = cv::Scalar(0, 255, 255);
      } else if (det.is_digit) {
        color = cv::Scalar(255, 200, 0);
      }

      cv::rectangle(overlay, det.bbox_img, color, 2);
      char det_txt[128];
      std::snprintf(
        det_txt,
        sizeof(det_txt),
        "%s %.2f d=%.2f y=%+.2f",
        det.label.c_str(),
        det.confidence,
        det.d_bumper_m,
        det.y_lat_m);
      cv::putText(
        overlay,
        det_txt,
        cv::Point(
          std::clamp(det.bbox_img.x, 4, std::max(4, overlay.cols - 180)),
          std::max(16, det.bbox_img.y - 4)),
        cv::FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1);
    }

    // 5. Compact finals status HUD at bottom of frame
    int hud_y = overlay.rows - 10;
    if (!finals_overlay.finish_guard_status.empty()) {
      cv::putText(
        overlay,
        finals_overlay.finish_guard_status,
        cv::Point(10, hud_y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.46,
        cv::Scalar(0, 220, 255),
        1);
      hud_y -= 18;
    }
    if (!finals_overlay.fork_vote_status.empty()) {
      cv::putText(
        overlay,
        finals_overlay.fork_vote_status,
        cv::Point(10, hud_y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.46,
        cv::Scalar(120, 255, 120),
        1);
      hud_y -= 18;
    }

    char hud_line[256];
    std::snprintf(
      hud_line,
      sizeof(hud_line),
      "seg=%s plan=%s[%s] vcap=%.2f br=%d yel=%s(%.2fm%s) cor=[%+.2f,%+.2f]->%+.2f",
      finals_overlay.lane_seg_mode.c_str(),
      finals_overlay.planner_mode.c_str(),
      finals_overlay.active_mode_tag.c_str(),
      finals_overlay.speed_cap_mps,
      finals_overlay.preferred_branch,
      finals_overlay.yellow_dist_valid ? "Y" : "-",
      finals_overlay.yellow_d_bumper_m,
      finals_overlay.yellow_dead_end_ahead ? ",DEAD" : "",
      finals_overlay.y_right_safe_m,
      finals_overlay.y_left_safe_m,
      finals_overlay.y_ref_m);
    cv::putText(
      overlay,
      hud_line,
      cv::Point(10, std::max(110, hud_y)),
      cv::FONT_HERSHEY_SIMPLEX,
      0.44,
      cv::Scalar(255, 255, 255),
      1);
  }

  return overlay;
}

}  // namespace line_follower
