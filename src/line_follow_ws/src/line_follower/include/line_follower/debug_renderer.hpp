#pragma once

#include <opencv2/core.hpp>

#include "line_follower/line_follower_types.hpp"

namespace line_follower
{

class DebugRenderer
{
public:
  DebugRenderer() = default;

  cv::Mat render(
    const cv::Mat & frame_bgr,
    const ProcessingResult & result,
    const CurveAssistState & curve_state,
    const RedAvoidState & red_state,
    bool red_overlay_valid,
    double red_divider_angle_rad,
    bool enable_continuous_curve_assist) const;
};

}  // namespace line_follower
