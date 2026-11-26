#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "line_follower/line_follower_types.hpp"

namespace line_follower
{

struct DebugDetectionBox
{
  cv::Rect bbox_img{};
  std::string label{};
  double confidence{0.0};
  double d_bumper_m{0.0};
  double y_lat_m{0.0};
  bool is_blocking{false};
  bool is_no_entry{false};
  bool is_digit{false};
  bool is_yellow{false};
};

struct FinalsDebugOverlay
{
  bool enabled{false};
  std::string lane_seg_mode{};
  std::string active_mode_tag{};

  bool yellow_dist_valid{false};
  double yellow_d_bumper_m{0.0};
  bool yellow_is_finish{false};
  bool yellow_dead_end_ahead{false};

  bool corridor_valid{false};
  double y_right_safe_m{0.0};
  double y_left_safe_m{0.0};
  double y_ref_m{0.0};
  int corridor_left_px{-1};
  int corridor_right_px{-1};
  int corridor_ref_px{-1};
  int corridor_row_y{-1};

  double speed_cap_mps{0.0};
  int preferred_branch{0};  // -1 left, 0 none, +1 right
  std::string fork_vote_status{};
  std::string finish_guard_status{};

  bool cbf_enabled{false};
  bool cbf_intervened{false};
  double cbf_min_clearance_m{0.0};
  std::string cbf_limiting_source{};

  std::vector<DebugDetectionBox> detections{};
};

class DebugRenderer
{
public:
  DebugRenderer() = default;

  cv::Mat render(
    const cv::Mat & frame_bgr,
    const ProcessingResult & result,
    const CurveAssistState & curve_state,
    bool enable_continuous_curve_assist,
    const FinalsDebugOverlay & finals_overlay = FinalsDebugOverlay{}) const;
};

}  // namespace line_follower
