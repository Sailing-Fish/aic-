#include "line_follower/lane_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace line_follower
{

LaneGeometry::LaneGeometry()
{
  set_config(LaneGeometryConfig{});
}

LaneGeometry::LaneGeometry(const LaneGeometryConfig & config)
{
  set_config(config);
}

void LaneGeometry::set_config(const LaneGeometryConfig & config)
{
  config_ = config;

  config_.far_band_top_ratio = clamp_value(config_.far_band_top_ratio, 0.0, 1.0);
  config_.far_band_bottom_ratio = clamp_value(config_.far_band_bottom_ratio, 0.0, 1.0);
  if (config_.far_band_top_ratio > config_.far_band_bottom_ratio) {
    std::swap(config_.far_band_top_ratio, config_.far_band_bottom_ratio);
  }

  config_.near_band_top_ratio = clamp_value(config_.near_band_top_ratio, 0.0, 1.0);
  config_.near_band_bottom_ratio = clamp_value(config_.near_band_bottom_ratio, 0.0, 1.0);
  if (config_.near_band_top_ratio > config_.near_band_bottom_ratio) {
    std::swap(config_.near_band_top_ratio, config_.near_band_bottom_ratio);
  }

  config_.column_white_ratio = clamp_value(config_.column_white_ratio, 0.01, 1.0);
  config_.min_run_width = std::max(1, config_.min_run_width);

  config_.reverse_curve_min_error = std::max(0.0, config_.reverse_curve_min_error);
  config_.reverse_curve_threshold = std::max(0.0, config_.reverse_curve_threshold);
  config_.reverse_curve_confirm_frames = std::max(1, config_.reverse_curve_confirm_frames);
  config_.reverse_curve_clear_frames = std::max(1, config_.reverse_curve_clear_frames);

  config_.nominal_near_width_ratio = clamp_value(config_.nominal_near_width_ratio, 0.10, 1.0);
  config_.nominal_far_width_ratio = clamp_value(config_.nominal_far_width_ratio, 0.10, 1.0);
  config_.wide_road_ratio_threshold = clamp_value(config_.wide_road_ratio_threshold, 0.20, 1.0);
  config_.wide_band_min_reliability = clamp_value(config_.wide_band_min_reliability, 0.05, 1.0);
  config_.wide_lane_boundary_blend = clamp_value(config_.wide_lane_boundary_blend, 0.0, 1.0);
  config_.observation_filter_alpha = clamp_value(config_.observation_filter_alpha, 0.05, 1.0);
}

const LaneGeometryConfig & LaneGeometry::config() const
{
  return config_;
}

BandObservation LaneGeometry::observe_band(
  const cv::Mat & binary,
  int y0,
  int y1,
  double reference_x,
  double nominal_width_ratio,
  double dir_hint_norm) const
{
  BandObservation obs{};
  if (binary.empty() || binary.rows <= 0 || binary.cols <= 0) {
    return obs;
  }

  y0 = std::clamp(y0, 0, binary.rows - 1);
  y1 = std::clamp(y1, y0 + 1, binary.rows);

  const cv::Mat band = binary(cv::Rect(0, y0, binary.cols, y1 - y0));
  const int band_height = band.rows;
  if (band_height <= 0) {
    return obs;
  }

  const int min_white_pixels = std::max(
    1,
    static_cast<int>(static_cast<double>(band_height) * config_.column_white_ratio));

  std::vector<unsigned char> active(static_cast<size_t>(binary.cols), 0);
  for (int x = 0; x < binary.cols; ++x) {
    if (cv::countNonZero(band.col(x)) >= min_white_pixels) {
      active[static_cast<size_t>(x)] = 1;
    }
  }

  bool found = false;
  int best_start = 0;
  int best_end = 0;
  double best_distance = 1e9;
  int best_width = 0;
  int run_start = -1;

  for (int x = 0; x <= binary.cols; ++x) {
    const bool is_active =
      (x < binary.cols) ? (active[static_cast<size_t>(x)] != 0) : false;

    if (is_active && run_start < 0) {
      run_start = x;
    }

    if ((!is_active || x == binary.cols) && run_start >= 0) {
      const int run_end = x - 1;
      const int run_width = run_end - run_start + 1;

      if (run_width >= config_.min_run_width) {
        const double center = 0.5 * static_cast<double>(run_start + run_end);

        double distance;
        if (reference_x >= static_cast<double>(run_start) &&
            reference_x <= static_cast<double>(run_end))
        {
          distance = 0.0;
        } else {
          distance = std::abs(center - reference_x);
        }

        if (!found ||
            distance < best_distance ||
            (std::abs(distance - best_distance) < 1e-6 &&
             run_width > best_width))
        {
          found = true;
          best_start = run_start;
          best_end = run_end;
          best_distance = distance;
          best_width = run_width;
        }
      }

      run_start = -1;
    }
  }

  if (!found) {
    return obs;
  }

  const double cols_d = static_cast<double>(binary.cols);
  const double image_center_x = 0.5 * cols_d;

  obs.valid = true;
  obs.left_px = best_start;
  obs.right_px = best_end;
  obs.width_px = best_width;
  obs.width_ratio = clamp_value(static_cast<double>(best_width) / cols_d, 0.0, 1.0);
  obs.left_clipped = (best_start <= 1);
  obs.right_clipped = (best_end >= binary.cols - 2);
  obs.raw_center_px = 0.5 * static_cast<double>(best_start + best_end);

  if (obs.width_ratio <= config_.wide_road_ratio_threshold) {
    obs.reliability = 1.0;
  } else {
    const double excess =
      (obs.width_ratio - config_.wide_road_ratio_threshold) /
      std::max(1e-3, 1.0 - config_.wide_road_ratio_threshold);
    obs.reliability = clamp_value(
      1.0 - excess * (1.0 - config_.wide_band_min_reliability),
      config_.wide_band_min_reliability,
      1.0);
  }
  if (obs.left_clipped && obs.right_clipped) {
    obs.reliability = std::min(obs.reliability, config_.wide_band_min_reliability);
  }

  double effective_center = obs.raw_center_px;
  if (config_.wide_lane_boundary_blend > 0.0) {
    const double nominal_w = clamp_value(nominal_width_ratio, 0.15, 0.95) * cols_d;

    if (obs.left_clipped && !obs.right_clipped) {
      const double anchored = std::min(
        static_cast<double>(obs.right_px) - 0.5 * nominal_w,
        0.5 * nominal_w);
      effective_center =
        (1.0 - config_.wide_lane_boundary_blend) * obs.raw_center_px +
        config_.wide_lane_boundary_blend * anchored;
    } else if (obs.right_clipped && !obs.left_clipped) {
      const double anchored = std::max(
        static_cast<double>(obs.left_px) + 0.5 * nominal_w,
        cols_d - 0.5 * nominal_w);
      effective_center =
        (1.0 - config_.wide_lane_boundary_blend) * obs.raw_center_px +
        config_.wide_lane_boundary_blend * anchored;
    }

    effective_center = clamp_value(effective_center, 0.0, cols_d - 1.0);
  }

  obs.center_px = effective_center;
  obs.center_norm =
    clamp_value((obs.center_px - image_center_x) / image_center_x, -1.0, 1.0);

  return obs;
}

std::optional<double> LaneGeometry::find_band_center(
  const cv::Mat & binary,
  int y0,
  int y1,
  double reference_x,
  double column_white_ratio,
  int min_run_width)
{
  LaneGeometryConfig cfg{};
  cfg.column_white_ratio = column_white_ratio;
  cfg.min_run_width = min_run_width;
  cfg.wide_lane_boundary_blend = 0.0;
  LaneGeometry geo(cfg);
  const auto band_obs = geo.observe_band(binary, y0, y1, reference_x, 0.70, 0.0);
  if (!band_obs.valid) {
    return std::nullopt;
  }
  return band_obs.raw_center_px;
}

LaneObservation LaneGeometry::observe(
  const SegmentedFrame & seg,
  CurveAssistState & curve_state) const
{
  LaneObservation obs{};
  obs.valid = false;
  obs.yellow_detected = seg.yellow_line_detected;
  obs.roi = cv::Rect(0, seg.roi_top, seg.frame_width, seg.roi_bottom - seg.roi_top);

  if (seg.binary_mask.empty() || seg.frame_width <= 0) {
    curve_state.last_confidence = 0.0;
    return obs;
  }

  const int roi_height = seg.roi_bottom - seg.roi_top;
  const int width = seg.frame_width;
  const double image_center_x = static_cast<double>(width) * 0.5;

  const int far_y0 = static_cast<int>(static_cast<double>(roi_height) * config_.far_band_top_ratio);
  const int far_y1 = static_cast<int>(static_cast<double>(roi_height) * config_.far_band_bottom_ratio);
  const int near_y0 = static_cast<int>(static_cast<double>(roi_height) * config_.near_band_top_ratio);
  const int near_y1 = static_cast<int>(static_cast<double>(roi_height) * config_.near_band_bottom_ratio);

  const double prior_dir_hint =
    curve_state.filter_initialized ? curve_state.filtered_error : 0.0;

  BandObservation near_obs = observe_band(
    seg.binary_mask,
    near_y0,
    near_y1,
    image_center_x,
    config_.nominal_near_width_ratio,
    prior_dir_hint);

  double far_reference = image_center_x;
  if (near_obs.valid) {
    far_reference = near_obs.center_px;
  }

  BandObservation far_obs = observe_band(
    seg.binary_mask,
    far_y0,
    far_y1,
    far_reference,
    config_.nominal_far_width_ratio,
    prior_dir_hint);

  // 若远带有效且近带较宽，用远带方向线索精化近带边界中心
  if (config_.wide_lane_boundary_blend > 0.0 && near_obs.valid && far_obs.valid) {
    near_obs = observe_band(
      seg.binary_mask,
      near_y0,
      near_y1,
      image_center_x,
      config_.nominal_near_width_ratio,
      far_obs.center_norm);
  }

  const bool near_valid = near_obs.valid;
  const bool far_valid = far_obs.valid;

  const double near_error = near_valid ? near_obs.center_norm : 0.0;
  const double far_error = far_valid ? far_obs.center_norm : 0.0;

  curve_state.near_valid = near_valid;
  curve_state.far_valid = far_valid;
  curve_state.near_error = near_error;
  curve_state.far_error = far_error;
  curve_state.curve_level = 0.0;

  if (near_valid) {
    curve_state.curve_level = std::max(curve_state.curve_level, std::abs(near_error));
  }
  if (far_valid) {
    curve_state.curve_level = std::max(curve_state.curve_level, std::abs(far_error));
  }

  bool reverse_curve_candidate = false;
  if (config_.enable_continuous_curve_assist && near_valid && far_valid) {
    const bool opposite_direction = near_error * far_error < 0.0;
    const bool both_significant =
      std::abs(near_error) >= config_.reverse_curve_min_error &&
      std::abs(far_error) >= config_.reverse_curve_min_error;
    const bool large_difference =
      std::abs(far_error - near_error) >= config_.reverse_curve_threshold;

    reverse_curve_candidate =
      opposite_direction && both_significant && large_difference;
  }

  if (!config_.enable_continuous_curve_assist) {
    curve_state.reverse_curve_active = false;
    curve_state.reverse_confirm_count = 0;
    curve_state.reverse_clear_count = 0;
  } else if (reverse_curve_candidate) {
    curve_state.reverse_clear_count = 0;
    curve_state.reverse_confirm_count++;

    if (!curve_state.reverse_curve_active &&
        curve_state.reverse_confirm_count >= config_.reverse_curve_confirm_frames)
    {
      curve_state.reverse_curve_active = true;
      curve_state.reverse_confirm_count = config_.reverse_curve_confirm_frames;
    }
  } else {
    curve_state.reverse_confirm_count = 0;

    if (curve_state.reverse_curve_active) {
      curve_state.reverse_clear_count++;
      if (curve_state.reverse_clear_count >= config_.reverse_curve_clear_frames) {
        curve_state.reverse_curve_active = false;
        curve_state.reverse_clear_count = 0;
      }
    } else {
      curve_state.reverse_clear_count = 0;
    }
  }

  double target_x = static_cast<double>(seg.contour_centroid.x);
  double target_y = static_cast<double>(seg.contour_centroid.y);
  bool local_target_valid = false;

  const double near_y = 0.5 * static_cast<double>(near_y0 + near_y1);
  const double far_y = 0.5 * static_cast<double>(far_y0 + far_y1);

  if (near_valid && far_valid) {
    const bool use_reverse_near_only =
      config_.enable_continuous_curve_assist &&
      curve_state.reverse_curve_active &&
      (!config_.enable_width_adaptive_fusion || near_obs.reliability >= 0.5);

    if (use_reverse_near_only) {
      target_x = near_obs.center_px;
      target_y = near_y;
    } else {
      double w_near = std::max(0.0, config_.near_band_weight);
      double w_far = std::max(0.0, config_.far_band_weight);
      if (config_.enable_width_adaptive_fusion) {
        w_near *= near_obs.reliability;
        w_far *= far_obs.reliability;
      }
      const double w_sum = w_near + w_far;
      if (w_sum <= 1e-9) {
        w_near = 0.70;
        w_far = 0.30;
      } else if (std::abs(w_sum - 1.0) > 1e-9) {
        w_near /= w_sum;
        w_far /= w_sum;
      }

      target_x = w_near * near_obs.center_px + w_far * far_obs.center_px;
      target_y = w_near * near_y + w_far * far_y;
    }
    local_target_valid = true;
  } else if (near_valid) {
    target_x = near_obs.center_px;
    target_y = near_y;
    local_target_valid = true;
  } else if (far_valid) {
    target_x = far_obs.center_px;
    target_y = far_y;
    local_target_valid = true;
  }

  if (!local_target_valid && !seg.contour_valid) {
    curve_state.last_confidence = 0.0;
    return obs;
  }

  obs.valid = true;
  obs.near_band = near_obs;
  obs.far_band = far_obs;
  obs.near_center_px = near_obs.center_px;
  obs.far_center_px = far_obs.center_px;
  obs.near_center_norm = near_obs.center_norm;
  obs.far_center_norm = far_obs.center_norm;
  obs.near_width_ratio = near_obs.width_ratio;
  obs.far_width_ratio = far_obs.width_ratio;

  if (near_valid && far_valid) {
    obs.lane_width_ratio = 0.5 * (near_obs.width_ratio + far_obs.width_ratio);
  } else if (near_valid) {
    obs.lane_width_ratio = near_obs.width_ratio;
  } else if (far_valid) {
    obs.lane_width_ratio = far_obs.width_ratio;
  } else {
    obs.lane_width_ratio =
      clamp_value(static_cast<double>(seg.bounding_box.width) / static_cast<double>(width), 0.0, 1.0);
  }

  const float raw_centroid_x = static_cast<float>(target_x);
  const float raw_centroid_y = static_cast<float>(target_y);
  const double raw_error =
    clamp_value((static_cast<double>(raw_centroid_x) - image_center_x) / image_center_x, -1.0, 1.0);

  double raw_heading = 0.0;
  if (near_valid && far_valid) {
    if (config_.enable_width_adaptive_fusion &&
        far_obs.left_clipped && far_obs.right_clipped &&
        !(near_obs.left_clipped && near_obs.right_clipped))
    {
      raw_heading = near_obs.center_norm;
    } else {
      raw_heading = clamp_value(far_obs.center_norm - near_obs.center_norm, -1.0, 1.0);
    }
  } else if (far_valid) {
    raw_heading = far_obs.center_norm;
  } else if (near_valid) {
    raw_heading = curve_state.filter_initialized ? (0.5 * curve_state.filtered_heading) : 0.0;
  }

  // 置信度：由道路宽度可靠性、近远带一致性、连续帧稳定性、轮廓质量四部分组成
  double width_score = 0.30;
  if (near_valid && far_valid) {
    width_score = 0.5 * (near_obs.reliability + far_obs.reliability);
  } else if (near_valid) {
    width_score = 0.75 * near_obs.reliability;
  } else if (far_valid) {
    width_score = 0.75 * far_obs.reliability;
  }

  double consistency_score = 0.25;
  if (near_valid && far_valid) {
    consistency_score =
      clamp_value(1.0 - 0.5 * std::abs(far_obs.center_norm - near_obs.center_norm), 0.30, 1.0) *
      width_score;
  } else if (near_valid || far_valid) {
    consistency_score = 0.55 * width_score;
  }

  const double stability_score =
    (curve_state.filter_initialized ?
    clamp_value(1.0 - std::abs(raw_error - curve_state.filtered_error) / 0.50, 0.0, 1.0) :
    1.0) * std::sqrt(clamp_value(width_score, 0.0, 1.0));

  const double contour_score = seg.contour_valid ? 1.0 : 0.30;

  obs.confidence = clamp_value(
    0.45 * width_score +
      0.25 * consistency_score +
      0.15 * stability_score +
      0.15 * contour_score,
    0.0,
    1.0);

  const bool both_double_clipped =
    near_valid && far_valid &&
    near_obs.left_clipped && near_obs.right_clipped &&
    far_obs.left_clipped && far_obs.right_clipped;

  const double alpha = config_.observation_filter_alpha;
  if (!curve_state.filter_initialized || alpha >= 0.999) {
    obs.lateral_error = raw_error;
    obs.heading_proxy = raw_heading;
    obs.target_centroid = cv::Point2f(raw_centroid_x, raw_centroid_y);
  } else if (config_.enable_width_adaptive_fusion && both_double_clipped) {
    obs.lateral_error = raw_error;
    obs.heading_proxy = raw_heading;
    obs.target_centroid = cv::Point2f(raw_centroid_x, raw_centroid_y);
  } else {
    obs.lateral_error =
      clamp_value(alpha * raw_error + (1.0 - alpha) * curve_state.filtered_error, -1.0, 1.0);
    obs.heading_proxy =
      clamp_value(alpha * raw_heading + (1.0 - alpha) * curve_state.filtered_heading, -1.0, 1.0);
    const double filtered_target_x =
      clamp_value(image_center_x + obs.lateral_error * image_center_x, 0.0, static_cast<double>(width - 1));
    obs.target_centroid = cv::Point2f(
      static_cast<float>(filtered_target_x),
      raw_centroid_y);
  }

  double base_curve =
    (near_valid && far_valid) ?
    std::abs(far_obs.center_norm - near_obs.center_norm) :
    std::abs(obs.heading_proxy);
  if (config_.enable_width_adaptive_fusion) {
    const double max_w = std::max(obs.near_width_ratio, obs.far_width_ratio);
    if (max_w > config_.wide_road_ratio_threshold) {
      const double w_excess =
        (max_w - config_.wide_road_ratio_threshold) /
        std::max(1e-3, 1.0 - config_.wide_road_ratio_threshold);
      base_curve = std::max(base_curve, 0.35 * clamp_value(w_excess, 0.0, 1.0));
    }
  }
  obs.curvature_proxy = clamp_value(base_curve, 0.0, 1.0);

  curve_state.filter_initialized = true;
  if (config_.enable_width_adaptive_fusion && both_double_clipped) {
    curve_state.filtered_error *= 0.92;
    curve_state.filtered_heading *= 0.92;
  } else {
    curve_state.filtered_error = obs.lateral_error;
    curve_state.filtered_heading = obs.heading_proxy;
  }
  if (far_valid) {
    curve_state.prev_far_center_px = far_obs.center_px;
  }
  curve_state.last_confidence = obs.confidence;
  curve_state.last_heading_proxy = obs.heading_proxy;
  curve_state.last_curvature_proxy = obs.curvature_proxy;
  curve_state.last_lane_width_ratio = obs.lane_width_ratio;

  return obs;
}

ProcessingResult LaneGeometry::evaluate(
  const SegmentedFrame & seg,
  CurveAssistState & curve_state) const
{
  ProcessingResult result{};
  result.line_detected = false;
  result.yellow_line_detected = seg.yellow_line_detected;
  result.bounding_box = seg.bounding_box;
  result.roi_bgr = seg.roi_bgr;
  result.binary_mask = seg.binary_mask;
  result.roi_top = seg.roi_top;
  result.roi_bottom = seg.roi_bottom;

  const LaneObservation obs = observe(seg, curve_state);
  result.observation = obs;

  if (!obs.valid) {
    return result;
  }

  result.line_detected = true;
  result.centroid = obs.target_centroid;
  result.error = obs.lateral_error;
  result.state = TrackState::TRACKING;

  return result;
}

}  // namespace line_follower
