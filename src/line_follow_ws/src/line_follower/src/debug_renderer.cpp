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
  // Finals Overlay (corridor, yellow distance, obstacle boxes, fork/finish HUD)
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
    if (finals_overlay.cbf_enabled) {
      char cbf_line[128];
      if (finals_overlay.cbf_intervened) {
        std::snprintf(
          cbf_line,
          sizeof(cbf_line),
          "CBF INTERVENE src=%s clr=%+.3fm",
          finals_overlay.cbf_limiting_source.c_str(),
          finals_overlay.cbf_min_clearance_m);
      } else {
        std::snprintf(
          cbf_line,
          sizeof(cbf_line),
          "CBF OK clr=%+.3fm",
          finals_overlay.cbf_min_clearance_m);
      }
      cv::putText(
        overlay,
        cbf_line,
        cv::Point(10, hud_y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.46,
        finals_overlay.cbf_intervened ? cv::Scalar(0, 80, 255) : cv::Scalar(160, 255, 160),
        finals_overlay.cbf_intervened ? 2 : 1);
      hud_y -= 18;
    }

    char hud_line[256];
    std::snprintf(
      hud_line,
      sizeof(hud_line),
      "seg=%s [%s] vcap=%.2f br=%d yel=%s(%.2fm%s) cor=[%+.2f,%+.2f]->%+.2f",
      finals_overlay.lane_seg_mode.c_str(),
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

    // 6. Live BEV DWA inset in top-right corner & pre-flight homography warning banner
    if (!finals_overlay.bev_inset_bgr.empty() &&
      finals_overlay.bev_inset_bgr.cols < overlay.cols &&
      finals_overlay.bev_inset_bgr.rows < overlay.rows)
    {
      const int ix = overlay.cols - finals_overlay.bev_inset_bgr.cols - 6;
      const int iy = 6;
      finals_overlay.bev_inset_bgr.copyTo(
        overlay(cv::Rect(
          ix, iy, finals_overlay.bev_inset_bgr.cols, finals_overlay.bev_inset_bgr.rows)));
    }
    if (!finals_overlay.preflight_warning.empty()) {
      cv::rectangle(
        overlay, cv::Point(8, 72), cv::Point(std::min(overlay.cols - 8, 400), 96),
        cv::Scalar(0, 0, 210), -1);
      cv::putText(
        overlay,
        finals_overlay.preflight_warning,
        cv::Point(12, 89),
        cv::FONT_HERSHEY_SIMPLEX,
        0.44,
        cv::Scalar(255, 255, 255),
        1);
    }
  }

  return overlay;
}

}  // namespace line_follower
