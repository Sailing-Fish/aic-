#include "line_follower/debug_renderer.hpp"
#include "line_follower/image_processor.hpp"
#include "line_follower/lane_geometry.hpp"
#include "line_follower/motion_controller.hpp"
#include "line_follower/track_state_machine.hpp"

#define DETECTION_TURN_CONTROLLER_PURE_LOGIC_ONLY
#include "../src/detection_turn_controller.cpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#define CHECK(condition)                                                                 \
  do {                                                                                   \
    if (!(condition)) {                                                                  \
      throw std::runtime_error(                                                          \
        std::string(__func__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
    }                                                                                    \
  } while (false)

namespace
{

using line_follower::CurveAssistState;
using line_follower::DebugRenderer;
using line_follower::ImageProcessor;
using line_follower::ImageProcessorConfig;
using line_follower::LaneGeometry;
using line_follower::LaneGeometryConfig;
using line_follower::MotionController;
using line_follower::MotionControllerConfig;
using line_follower::MotionState;
using line_follower::RedAvoidState;
using line_follower::TrackState;
using line_follower::TrackStateConfig;
using line_follower::TrackStateMachine;

cv::Mat make_slanted_lane_image(double near_x, double far_x, int lane_width = 80)
{
  cv::Mat img(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  // Default ROI: roi_top_ratio = 0.5 -> roi_top = 240, roi_bottom_ratio = 0.05 -> roi_bottom = 456
  // roi_height = 216
  // far band: [0.34, 0.56] of 216 -> y in [73, 120] -> center y_roi = 96.5 (img y = 336.5)
  // near band: [0.62, 0.88] of 216 -> y in [133, 190] -> center y_roi = 161.5 (img y = 401.5)
  const int roi_top = 240;
  const int roi_bottom = 456;
  const double y_far = roi_top + 96.5;
  const double y_near = roi_top + 161.5;

  for (int y = roi_top; y < roi_bottom; ++y) {
    const double t = (static_cast<double>(y) - y_far) / (y_near - y_far);
    const double center_x = far_x + t * (near_x - far_x);
    const int x0 = std::max(0, static_cast<int>( std::round(center_x - lane_width * 0.5)));
    const int x1 = std::min(img.cols - 1, static_cast<int>(std::round(center_x + lane_width * 0.5)));
    cv::line(img, cv::Point(x0, y), cv::Point(x1, y), cv::Scalar(255, 255, 255), 1);
  }
  return img;
}

void test_fixed_image_repeatability_and_70_30_formula()
{
  ImageProcessor processor;
  LaneGeometry geometry;

  const cv::Mat straight_img = make_slanted_lane_image(320.0, 320.0);
  CurveAssistState state0{};
  const auto seg0 = processor.segment(straight_img);
  CHECK(seg0.has_value());
  const auto res0 = geometry.evaluate(*seg0, state0);
  CHECK(res0.line_detected);
  CHECK(res0.state == TrackState::TRACKING);
  CHECK(std::abs(res0.error) < 1e-3);

  // Repeat 30 times on a slanted lane image and verify deterministic bit-exact repeatability
  const cv::Mat curved_img = make_slanted_lane_image(360.0, 440.0);
  double ref_x = 0.0;
  double ref_y = 0.0;
  double ref_err = 0.0;

  for (int i = 0; i < 30; ++i) {
    CurveAssistState state{};
    const auto seg = processor.segment(curved_img);
    CHECK(seg.has_value());
    const auto res = geometry.evaluate(*seg, state);
    CHECK(res.line_detected);
    if (i == 0) {
      ref_x = res.centroid.x;
      ref_y = res.centroid.y;
      ref_err = res.error;
      // Expected 0.70 * 360 + 0.30 * 440 = 252 + 132 = 384 (+/- 1 px quantization)
      CHECK(std::abs(ref_x - 384.0) <= 1.5);
    } else {
      CHECK(res.centroid.x == static_cast<float>(ref_x));
      CHECK(res.centroid.y == static_cast<float>(ref_y));
      CHECK(res.error == ref_err);
    }
  }
}

void test_parameterized_band_weights_and_ratios()
{
  ImageProcessor processor;
  const cv::Mat curved_img = make_slanted_lane_image(350.0, 450.0);
  const auto seg = processor.segment(curved_img);
  CHECK(seg.has_value());

  // 100% near weight
  LaneGeometryConfig near_only_cfg{};
  near_only_cfg.near_band_weight = 1.0;
  near_only_cfg.far_band_weight = 0.0;
  LaneGeometry near_geo(near_only_cfg);
  CurveAssistState s1{};
  const auto res_near = near_geo.evaluate(*seg, s1);

  // 100% far weight
  LaneGeometryConfig far_only_cfg{};
  far_only_cfg.near_band_weight = 0.0;
  far_only_cfg.far_band_weight = 1.0;
  LaneGeometry far_geo(far_only_cfg);
  CurveAssistState s2{};
  const auto res_far = far_geo.evaluate(*seg, s2);

  // Default 70/30 weight
  LaneGeometry default_geo;
  CurveAssistState s3{};
  const auto res_def = default_geo.evaluate(*seg, s3);

  CHECK(res_near.line_detected && res_far.line_detected && res_def.line_detected);
  CHECK(res_far.centroid.x > res_def.centroid.x);
  CHECK(res_def.centroid.x > res_near.centroid.x);

  const double expected_blend =
    0.70 * static_cast<double>(res_near.centroid.x) +
    0.30 * static_cast<double>(res_far.centroid.x);
  CHECK(std::abs(static_cast<double>(res_def.centroid.x) - expected_blend) < 1e-4);
}

void test_reverse_curve_state_isolation()
{
  ImageProcessor processor;
  LaneGeometryConfig cfg{};
  cfg.enable_continuous_curve_assist = true;
  cfg.reverse_curve_min_error = 0.12;
  cfg.reverse_curve_threshold = 0.35;
  cfg.reverse_curve_confirm_frames = 3;
  cfg.reverse_curve_clear_frames = 2;
  LaneGeometry geometry(cfg);

  // Near on left (240 -> err = -0.25), Far on right (420 -> err = +0.3125), diff = 0.5625
  const cv::Mat s_curve_img = make_slanted_lane_image(240.0, 420.0);
  const auto seg = processor.segment(s_curve_img);
  CHECK(seg.has_value());

  CurveAssistState state_a{};
  CurveAssistState state_b{};

  // Frame 1 & 2: not yet confirmed (confirm_frames = 3)
  geometry.evaluate(*seg, state_a);
  CHECK(!state_a.reverse_curve_active && state_a.reverse_confirm_count == 1);
  geometry.evaluate(*seg, state_a);
  CHECK(!state_a.reverse_curve_active && state_a.reverse_confirm_count == 2);

  // state_b is completely untouched (no global state sharing)
  CHECK(!state_b.reverse_curve_active && state_b.reverse_confirm_count == 0);

  // Frame 3: confirmed! Target switches to near band center (~240)
  const auto res_confirmed = geometry.evaluate(*seg, state_a);
  CHECK(state_a.reverse_curve_active);
  CHECK(std::abs(res_confirmed.centroid.x - 240.0F) <= 2.0F);
}

void test_yellow_line_and_lost_line_state_machine()
{
  TrackStateConfig cfg{};
  cfg.yellow_stop_duration = 1.0;
  cfg.yellow_stop_min_time = 0.35;
  cfg.odom_still_threshold = 0.02;
  cfg.odom_still_hold = 0.25;
  cfg.yellow_permanent_count = 2;
  cfg.yellow_clear_frames_required = 3;
  cfg.line_lost_stop_time = 0.8;

  TrackStateMachine sm(cfg);
  sm.reset(10.0);

  // Normal tracking at t=10.1
  auto ev = sm.step(false, true, 10.1);
  CHECK(ev.base_state == TrackState::TRACKING && !ev.yellow_hold_stop);
  sm.commit_state(ev.base_state);

  // First yellow line at t=11.0 -> latched, count=1, temp_stop
  ev = sm.step(true, true, 11.0);
  CHECK(ev.yellow_stop_begin && ev.yellow_temp_stop && !ev.yellow_final_stop);
  CHECK(ev.yellow_hold_stop && ev.base_state == TrackState::YELLOW_LINE_DETECTED);
  CHECK(sm.state().yellow_line_detection_count == 1);
  sm.commit_state(ev.base_state);

  // Still seeing same yellow line at t=11.1 -> no double count
  ev = sm.step(true, true, 11.1);
  CHECK(!ev.yellow_stop_begin && sm.state().yellow_line_detection_count == 1);
  sm.commit_state(ev.base_state);

  // Odom reports stillness starting at t=11.10; by t=11.40:
  // since_yellow = 0.40 >= 0.35 and odom_still = 0.30 >= 0.25 -> releases early (< 1.0s)!
  sm.update_odom(0.005, 11.10);
  ev = sm.step(false, true, 11.40);
  CHECK(ev.can_release && ev.yellow_stop_end && !ev.yellow_hold_stop);
  CHECK(ev.base_state == TrackState::TRACKING);
  sm.commit_state(ev.base_state);

  // Clear latch after 3 non-yellow frames (already had 1 at 11.40, need 2 more)
  ev = sm.step(false, true, 11.50);
  CHECK(!ev.yellow_cleared);
  ev = sm.step(false, true, 11.60);
  CHECK(ev.yellow_cleared && !sm.state().yellow_line_latched);
  sm.commit_state(ev.base_state);

  // Line lost progression: t=12.0 (< 0.8s since 11.60) keeps TRACKING; t=12.5 (>= 0.8s) -> LOST
  ev = sm.step(false, false, 12.00);
  CHECK(ev.base_state == TrackState::TRACKING);
  sm.commit_state(ev.base_state);
  ev = sm.step(false, false, 12.50);
  CHECK(ev.base_state == TrackState::LOST);
  sm.commit_state(ev.base_state);

  // Second yellow line at t=13.0 -> count=2 == yellow_permanent_count -> permanent_stop
  ev = sm.step(true, true, 13.0);
  CHECK(ev.yellow_final_stop && sm.state().permanent_stop && ev.yellow_hold_stop);
  sm.commit_state(ev.base_state);

  // Lap reset when start_cmd transitions false -> true after permanent_stop
  CHECK(sm.on_start_command(false, true, 20.0));
  CHECK(!sm.state().permanent_stop && sm.state().yellow_line_detection_count == 0);
}

void test_motion_controller_and_parameterized_lost_thresholds()
{
  MotionControllerConfig cfg{};
  cfg.target_speed = 0.5;
  cfg.max_speed = 0.6;
  cfg.min_speed = 0.0;
  cfg.max_yaw_rate = 1.5;
  cfg.kp = 2.0;
  cfg.speed_gain_damping = 0.5;
  cfg.line_lost_forward_speed = 0.30;
  cfg.line_lost_yaw_hold_factor = 0.80;
  cfg.line_lost_slow_time = 0.50;
  cfg.line_lost_slow_factor = 0.25;
  cfg.line_lost_full_stop_time = 1.50;

  MotionController controller(cfg);
  CurveAssistState curve{};
  const auto track_cmd = controller.compute_tracking_command(0.25, curve);
  CHECK(std::abs(track_cmd.yaw_rate - (-0.5)) < 1e-6);
  CHECK(track_cmd.linear_speed > 0.0 && track_cmd.linear_speed <= 0.5);

  MotionState motion{};
  MotionController::record_motion(false, track_cmd.linear_speed, track_cmd.yaw_rate, motion);
  CHECK(motion.valid);

  // Phase 1: since_detection <= line_lost_slow_time (0.50s)
  const auto lost1 = controller.compute_lost_line_command(0.30, motion);
  CHECK(std::abs(lost1.linear_speed - 0.30) < 1e-6);
  CHECK(std::abs(lost1.yaw_rate - (-0.40)) < 1e-6);

  // Phase 2: line_lost_slow_time < since_detection <= line_lost_full_stop_time
  const auto lost2 = controller.compute_lost_line_command(0.80, motion);
  CHECK(std::abs(lost2.linear_speed - (0.30 * 0.25)) < 1e-6);

  // Phase 3: since_detection > line_lost_full_stop_time (1.50s)
  const auto lost3 = controller.compute_lost_line_command(1.60, motion);
  CHECK(std::abs(lost3.linear_speed) < 1e-9);

  // Soft red avoidance fusion check
  RedAvoidState red{};
  red.valid = true;
  red.is_red_line = false;
  red.bottom_ratio = 0.72;
  double v = 0.45;
  double w = 0.30;  // turning left toward left cone
  controller.fuse_soft_red_avoidance(-0.20, curve, red, v, w);
  CHECK(red.risk > 0.99 && red.very_close);
  CHECK(w < 0.30);  // pushed right away from left cone
  CHECK(v >= cfg.red_min_forward_speed && v <= cfg.red_soft_speed);
}

void test_debug_renderer()
{
  ImageProcessor processor;
  LaneGeometry geometry;
  DebugRenderer renderer;

  const cv::Mat img = make_slanted_lane_image(320.0, 320.0);
  CurveAssistState curve{};
  const auto seg = processor.segment(img);
  CHECK(seg.has_value());
  auto res = geometry.evaluate(*seg, curve);
  res.yellow_line_detected = true;

  RedAvoidState red{};
  red.valid = true;
  red.bbox_img = cv::Rect(300, 320, 30, 40);
  red.center_img = cv::Point2d(315.0, 340.0);
  red.bottom_img = cv::Point2d(315.0, 360.0);
  red.bottom_ratio = 0.55;
  red.risk = 0.5;
  red.in_corridor = true;

  const cv::Mat out = renderer.render(img, res, curve, red, true, 0.1, true);
  CHECK(!out.empty());
  CHECK(out.rows == img.rows && out.cols == img.cols && out.type() == CV_8UC3);
}

cv::Mat make_wide_near_curved_far_image(
  int near_left,
  int near_right,
  int far_left,
  int far_right,
  double roi_top_ratio = 0.65,
  double roi_bottom_ratio = 0.20)
{
  cv::Mat img(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  const int roi_top = static_cast<int>(480 * (1.0 - roi_top_ratio));
  const int roi_bottom = static_cast<int>(480 * (1.0 - roi_bottom_ratio));
  const int roi_h = roi_bottom - roi_top;
  const int split_y = roi_top + static_cast<int>(roi_h * 0.46);

  for (int y = roi_top; y < split_y; ++y) {
    cv::line(
      img,
      cv::Point(std::max(0, far_left), y),
      cv::Point(std::min(639, far_right), y),
      cv::Scalar(255, 255, 255),
      1);
  }
  for (int y = split_y; y < roi_bottom; ++y) {
    cv::line(
      img,
      cv::Point(std::max(0, near_left), y),
      cv::Point(std::min(639, near_right), y),
      cv::Scalar(255, 255, 255),
      1);
  }
  return img;
}

void test_roundabout_wide_white_observation_and_turn_response()
{
  ImageProcessorConfig img_cfg{};
  img_cfg.roi_top_ratio = 0.65;
  img_cfg.roi_bottom_ratio = 0.20;
  img_cfg.use_white_filter = true;
  img_cfg.white_s_max = 45;
  img_cfg.white_v_min = 80;
  ImageProcessor processor(img_cfg);

  // Baseline configuration (legacy fork-like defaults)
  LaneGeometryConfig base_geo_cfg{};
  MotionControllerConfig base_ctrl_cfg{};
  base_ctrl_cfg.target_speed = 0.50;
  base_ctrl_cfg.max_speed = 0.50;
  base_ctrl_cfg.max_yaw_rate = 1.50;
  base_ctrl_cfg.kp = 3.30;
  base_ctrl_cfg.speed_gain_damping = 0.26;
  base_ctrl_cfg.enable_continuous_curve_assist = true;
  base_ctrl_cfg.reverse_curve_speed = 0.28;
  base_ctrl_cfg.sharp_curve_threshold = 0.42;
  base_ctrl_cfg.sharp_curve_speed = 0.33;

  // Roundabout configuration (separated from fork.yaml)
  LaneGeometryConfig rb_geo_cfg{};
  rb_geo_cfg.far_band_top_ratio = 0.22;
  rb_geo_cfg.far_band_bottom_ratio = 0.42;
  rb_geo_cfg.near_band_top_ratio = 0.50;
  rb_geo_cfg.near_band_bottom_ratio = 0.72;
  rb_geo_cfg.near_band_weight = 0.52;
  rb_geo_cfg.far_band_weight = 0.48;
  rb_geo_cfg.column_white_ratio = 0.50;
  rb_geo_cfg.min_run_width = 18;
  rb_geo_cfg.nominal_near_width_ratio = 0.68;
  rb_geo_cfg.nominal_far_width_ratio = 0.48;
  rb_geo_cfg.wide_road_ratio_threshold = 0.82;
  rb_geo_cfg.wide_band_min_reliability = 0.18;
  rb_geo_cfg.enable_width_adaptive_fusion = true;
  rb_geo_cfg.wide_lane_boundary_blend = 0.55;
  rb_geo_cfg.observation_filter_alpha = 0.85;
  rb_geo_cfg.enable_continuous_curve_assist = true;

  MotionControllerConfig rb_ctrl_cfg = base_ctrl_cfg;
  rb_ctrl_cfg.heading_gain = 1.15;
  rb_ctrl_cfg.low_confidence_threshold = 0.58;
  rb_ctrl_cfg.low_confidence_speed = 0.26;
  rb_ctrl_cfg.low_confidence_yaw_hold = 0.80;

  LaneGeometry base_geo(base_geo_cfg);
  MotionController base_ctrl(base_ctrl_cfg);
  LaneGeometry rb_geo(rb_geo_cfg);
  MotionController rb_ctrl(rb_ctrl_cfg);

  // Sequence of 5 frames entering roundabout wide-white zone then full-width throat:
  // Frame 0: straight wide road [0..639] in both bands
  // Frame 1: near band still saturated [0..639], far band starts curving left [0..460]
  // Frame 2: near band still saturated [0..639], far band curves further left [0..340]
  // Frame 3: near band left-clipped wide [0..520], far band [0..280]
  // Frame 4: temporary full-width white saturation in both bands [0..639] (low confidence)
  const int seq_near_l[5] = {0, 0, 0, 0, 0};
  const int seq_near_r[5] = {639, 639, 639, 520, 639};
  const int seq_far_l[5] = {0, 0, 0, 0, 0};
  const int seq_far_r[5] = {639, 460, 340, 280, 639};

  CurveAssistState base_curve{};
  MotionState base_motion{};
  CurveAssistState rb_curve{};
  MotionState rb_motion{};

  int base_turn_start_frame = -1;
  int rb_turn_start_frame = -1;

  for (int i = 0; i < 5; ++i) {
    const cv::Mat frame = make_wide_near_curved_far_image(
      seq_near_l[i], seq_near_r[i], seq_far_l[i], seq_far_r[i]);
    const auto seg = processor.segment(frame);
    CHECK(seg.has_value());

    const auto base_res = base_geo.evaluate(*seg, base_curve);
    const auto base_cmd =
      base_ctrl.compute_lane_command(base_res.observation, base_curve, base_motion);
    MotionController::record_motion(
      false, base_cmd.linear_speed, base_cmd.yaw_rate, base_motion);

    const auto rb_res = rb_geo.evaluate(*seg, rb_curve);
    const auto rb_cmd =
      rb_ctrl.compute_lane_command(rb_res.observation, rb_curve, rb_motion);
    MotionController::record_motion(
      false, rb_cmd.linear_speed, rb_cmd.yaw_rate, rb_motion);

    if (base_turn_start_frame < 0 && base_cmd.yaw_rate >= 0.35) {
      base_turn_start_frame = i;
    }
    if (rb_turn_start_frame < 0 && rb_cmd.yaw_rate >= 0.35) {
      rb_turn_start_frame = i;
    }

    if (i == 1) {
      // Frame 1: near is full width [0..639], far is [0..460].
      // Verify LaneObservation fields are populated and roundabout controller turns much stronger.
      CHECK(rb_res.observation.valid);
      CHECK(rb_res.observation.near_width_ratio > 0.95);
      CHECK(rb_res.observation.far_width_ratio < 0.75);
      CHECK(rb_res.observation.heading_proxy < -0.15);
      CHECK(rb_res.observation.curvature_proxy > 0.15);
      CHECK(rb_cmd.yaw_rate > base_cmd.yaw_rate + 0.35);
    }

    if (i == 4) {
      // Frame 4: both bands saturated [0..639] -> low confidence triggers speed cap & yaw hold!
      CHECK(rb_res.observation.confidence < rb_ctrl_cfg.low_confidence_threshold);
      CHECK(rb_cmd.linear_speed <= rb_ctrl_cfg.low_confidence_speed + 1e-6);
      CHECK(rb_cmd.yaw_rate > 0.40);  // holds left turn instead of dropping to ~0
      CHECK(std::abs(base_cmd.yaw_rate) < 0.01);  // baseline drops to ~0 yaw rate (0.5px quantization)
    }
  }

  // Roundabout config initiates strong turn at Frame 1, earlier than baseline (Frame 2+)
  CHECK(rb_turn_start_frame == 1);
  CHECK(base_turn_start_frame > rb_turn_start_frame);
}

void test_detection_turn_controller_scenarios()
{
  using line_follower::DetectionRoiInput;
  using line_follower::DetectionTurnConfig;
  using line_follower::DetectionTurnLogic;
  using line_follower::DigitTaskState;
  using line_follower::TurnEndReason;

  DetectionTurnConfig cfg{};
  cfg.confidence_threshold = 0.30;
  cfg.det_frame_width = 640;
  cfg.det_frame_height = 480;
  cfg.frame_window = 2;
  cfg.turn_cooldown = 0.80;
  cfg.turn_duration = 0.60;
  cfg.min_turn_duration = 0.12;
  cfg.recovery_center_tolerance = 0.10;
  cfg.recovery_confirm_frames = 2;
  cfg.dedup_clear_time = 2.0;
  cfg.max_detection_gap = 0.35;
  cfg.max_missed_frames = 1;
  cfg.max_position_jump = 0.15;
  cfg.min_position_consistency = 0.45;
  cfg.y_align_tolerance = 0.05;
  cfg.turn_linear = 0.20;
  cfg.yaw_gain = 1.6;
  cfg.max_yaw_rate = 1.3;
  cfg.turn_direction_sign = -1.0;
  cfg.require_start_cmd = true;

  // =========================================================================
  // 1. 未启动门控 (Unstarted gating): 未收到 start_cmd=true 时绝不触发或发布命令
  // =========================================================================
  DetectionTurnLogic logic(cfg);
  logic.update_track_state("tracking", 1.00);
  logic.update_centroid(0.50, 0.50, 1.00);

  DetectionRoiInput d2_high{"2", 0.92, 420, 220, 40, 40};  // center=(440,240) -> norm=(0.6875, 0.50)
  auto out = logic.on_detection({d2_high}, 1.00);
  CHECK(!logic.is_armed(1.00));
  CHECK(out.state == DigitTaskState::IDLE && !out.should_publish);
  out = logic.on_detection({d2_high}, 1.05);
  CHECK(out.state == DigitTaskState::IDLE && !out.should_publish);

  // 启动门控开启
  logic.update_start_cmd(true, 2.00);
  logic.update_track_state("tracking", 2.00);
  logic.update_centroid(0.50, 0.50, 2.00);
  CHECK(logic.is_armed(2.00));

  // =========================================================================
  // 2. 多数字与多置信度 (Multiple digits & confidences, NOT ordered by digit value)
  // =========================================================================
  // Frame 1: Digit 8 (low conf 0.35), Digit 2 (high conf 0.92), Digit 5 (below threshold 0.15)
  DetectionRoiInput d8_low{"8", 0.35, 420, 220, 40, 40};
  DetectionRoiInput d5_sub{"5", 0.15, 420, 220, 40, 40};
  out = logic.on_detection({d8_low, d2_high, d5_sub}, 2.05);
  CHECK(out.state == DigitTaskState::CANDIDATE);
  CHECK(!out.should_publish);
  CHECK(out.active_digit == 2);  // Digit 2 beats Digit 8 due to higher confidence!
  CHECK(!logic.track(5).active);  // Digit 5 rejected below threshold

  // Frame 2: Digit 8 jumps wildly (x_offset=50 -> norm_x=0.11, jump > 0.15), Digit 2 stays consistent
  DetectionRoiInput d8_jump{"8", 0.95, 50, 220, 40, 40};
  out = logic.on_detection({d8_jump, d2_high}, 2.10);
  CHECK(out.state == DigitTaskState::CONFIRMED);
  CHECK(out.just_confirmed);
  CHECK(out.should_publish);
  CHECK(out.active_digit == 2);
  CHECK(out.linear_x == 0.20 && out.angular_z < -0.10);

  // =========================================================================
  // 3. 状态流转与视觉恢复提前结束 (CONFIRMED -> EXECUTING -> RECOVERED -> COOLDOWN -> IDLE)
  // =========================================================================
  // Next control tick transitions CONFIRMED -> EXECUTING
  out = logic.step_control(2.15);
  CHECK(out.state == DigitTaskState::EXECUTING);
  CHECK(out.should_publish);

  // Before min_turn_duration (elapsed = 0.08s < 0.12s), even if centered, keeps executing
  logic.update_centroid(0.50, 0.50, 2.16);
  logic.update_centroid(0.50, 0.50, 2.18);
  out = logic.step_control(2.18);
  CHECK(out.state == DigitTaskState::EXECUTING && out.should_publish);

  // After min_turn_duration (t = 2.26, elapsed = 0.16s < max 0.60s), centroid is centered -> visual recovery!
  logic.update_centroid(0.49, 0.50, 2.24);
  logic.update_centroid(0.50, 0.50, 2.26);
  out = logic.step_control(2.26);
  CHECK(out.state == DigitTaskState::RECOVERED);
  CHECK(out.just_recovered);
  CHECK(out.end_reason == TurnEndReason::VISUAL_RECOVERY);
  CHECK(!out.should_publish && out.linear_x == 0.0 && out.angular_z == 0.0);

  // Next control tick transitions RECOVERED -> COOLDOWN
  out = logic.step_control(2.30);
  CHECK(out.state == DigitTaskState::COOLDOWN);
  CHECK(!out.should_publish);

  // =========================================================================
  // 4. 重复检测保护 (Duplicate detection prevention during & after cooldown)
  // =========================================================================
  // During cooldown (2.26 + 0.80 = 3.06), detections of Digit 2 or Digit 3 are ignored
  DetectionRoiInput d3_high{"3", 0.90, 200, 220, 40, 40};
  out = logic.on_detection({d2_high, d3_high}, 2.60);
  CHECK(out.state == DigitTaskState::COOLDOWN && !out.should_publish);

  // Cooldown expires at t = 3.10 -> IDLE
  logic.update_centroid(0.50, 0.50, 3.10);
  out = logic.step_control(3.10);
  CHECK(out.state == DigitTaskState::IDLE);

  // Repeated detection of same Digit 2 at t = 3.15, 3.20 (within dedup_clear_time = 2.0s) must NOT re-turn!
  out = logic.on_detection({d2_high}, 3.15);
  CHECK(out.state == DigitTaskState::IDLE && !out.should_publish);
  out = logic.on_detection({d2_high}, 3.20);
  CHECK(out.state == DigitTaskState::IDLE && !out.should_publish);

  // =========================================================================
  // 5. 稀疏帧场景 (Sparse frames: empty frames expire; stale gaps reset; 1-frame dropout tolerated)
  // =========================================================================
  // 5a: 1 detection of Digit 3 followed by 3 empty callbacks -> expires back to IDLE, never triggers
  logic.update_centroid(0.50, 0.50, 3.30);
  out = logic.on_detection({d3_high}, 3.30);
  CHECK(out.state == DigitTaskState::CANDIDATE && out.active_digit == 3);
  out = logic.on_detection({}, 3.35);  // 1st missed frame (tolerated, max_missed_frames=1)
  CHECK(out.state == DigitTaskState::CANDIDATE);
  out = logic.on_detection({}, 3.40);  // 2nd missed frame (> max_missed_frames) -> expires to IDLE!
  CHECK(out.state == DigitTaskState::IDLE && !out.should_publish);

  // 5b: Stale gap (> max_detection_gap = 0.35s) resets count to 1 instead of confirming
  logic.update_centroid(0.50, 0.50, 4.00);
  out = logic.on_detection({d3_high}, 4.00);
  CHECK(out.state == DigitTaskState::CANDIDATE && logic.track(3).consecutive_count == 1);
  logic.update_centroid(0.50, 0.50, 4.50);
  out = logic.on_detection({d3_high}, 4.50);  // dt = 0.50s > 0.35s -> reset to count=1!
  CHECK(out.state == DigitTaskState::CANDIDATE && logic.track(3).consecutive_count == 1);
  CHECK(!out.should_publish);

  // 5c: 1-frame dropout within max_detection_gap (seen at 4.50, missed at 4.55, seen at 4.60) -> CONFIRMED!
  logic.update_centroid(0.50, 0.50, 4.55);
  out = logic.on_detection({}, 4.55);
  CHECK(out.state == DigitTaskState::CANDIDATE);
  logic.update_centroid(0.50, 0.50, 4.60);
  out = logic.on_detection({d3_high}, 4.60);
  CHECK(out.state == DigitTaskState::CONFIRMED && out.active_digit == 3 && out.should_publish);

  // =========================================================================
  // 6. 超时场景与黄线停车中断 (Max duration timeout & Yellow line stop abort)
  // =========================================================================
  // Advance to EXECUTING; keep centroid off-center (x = 0.25) so visual recovery does NOT trigger
  out = logic.step_control(4.65);
  CHECK(out.state == DigitTaskState::EXECUTING && out.should_publish);
  logic.update_centroid(0.25, 0.50, 5.00);
  out = logic.step_control(5.00);  // elapsed = 0.40s < 0.60s -> still executing
  CHECK(out.state == DigitTaskState::EXECUTING && out.should_publish);

  // At t = 5.21s (elapsed = 0.61s >= turn_duration = 0.60s) -> ends via MAX_DURATION_TIMEOUT!
  logic.update_centroid(0.25, 0.50, 5.21);
  out = logic.step_control(5.21);
  CHECK(out.state == DigitTaskState::RECOVERED);
  CHECK(out.just_recovered);
  CHECK(out.end_reason == TurnEndReason::MAX_DURATION_TIMEOUT);
  CHECK(!out.should_publish);

  // Yellow line detected at t = 5.30s: clears cooldown & dedup lock, and disarms while yellow is active
  logic.update_track_state("yellow_line_detected", 5.30);
  CHECK(!logic.is_armed(5.30));
  CHECK(logic.last_executed_digit() == -1);
  out = logic.on_detection({d3_high}, 5.35);
  CHECK(out.state == DigitTaskState::IDLE && !out.should_publish);

  // Resume tracking after yellow line -> trigger Digit 4, then abort mid-turn via yellow line stop!
  logic.update_track_state("tracking", 6.00);
  logic.update_centroid(0.50, 0.50, 6.00);
  DetectionRoiInput d4_high{"4", 0.90, 420, 220, 40, 40};
  logic.on_detection({d4_high}, 6.05);
  out = logic.on_detection({d4_high}, 6.10);
  CHECK(out.state == DigitTaskState::CONFIRMED && out.should_publish);
  out = logic.step_control(6.15);
  CHECK(out.state == DigitTaskState::EXECUTING && out.should_publish);

  // Yellow line stop arrives during EXECUTING -> immediately aborts turn and zeroes output!
  logic.update_track_state("yellow_line_detected", 6.20);
  out = logic.step_control(6.20);
  CHECK(out.state == DigitTaskState::IDLE);
  CHECK(out.end_reason == TurnEndReason::ABORTED_BY_STOP_OR_DISARM);
  CHECK(!out.should_publish && out.linear_x == 0.0 && out.angular_z == 0.0);
}

void run_roundabout_sequence_replay(
  const std::string & dir,
  const std::string & prefix,
  int start_idx,
  int end_idx)
{
  ImageProcessorConfig img_cfg{};
  img_cfg.roi_top_ratio = 0.65;
  img_cfg.roi_bottom_ratio = 0.20;
  img_cfg.use_white_filter = true;
  img_cfg.white_s_max = 45;
  img_cfg.white_v_min = 80;
  ImageProcessor processor(img_cfg);

  LaneGeometryConfig base_geo_cfg{};
  MotionControllerConfig base_ctrl_cfg{};
  base_ctrl_cfg.target_speed = 0.50;
  base_ctrl_cfg.max_speed = 0.50;
  base_ctrl_cfg.max_yaw_rate = 1.50;
  base_ctrl_cfg.kp = 3.30;
  base_ctrl_cfg.speed_gain_damping = 0.22;
  base_ctrl_cfg.enable_continuous_curve_assist = true;
  base_ctrl_cfg.reverse_curve_speed = 0.34;
  base_ctrl_cfg.sharp_curve_threshold = 0.50;
  base_ctrl_cfg.sharp_curve_speed = 0.41;

  LaneGeometryConfig rb_geo_cfg{};
  rb_geo_cfg.far_band_top_ratio = 0.22;
  rb_geo_cfg.far_band_bottom_ratio = 0.42;
  rb_geo_cfg.near_band_top_ratio = 0.50;
  rb_geo_cfg.near_band_bottom_ratio = 0.72;
  rb_geo_cfg.near_band_weight = 0.52;
  rb_geo_cfg.far_band_weight = 0.48;
  rb_geo_cfg.column_white_ratio = 0.50;
  rb_geo_cfg.min_run_width = 18;
  rb_geo_cfg.nominal_near_width_ratio = 0.68;
  rb_geo_cfg.nominal_far_width_ratio = 0.48;
  rb_geo_cfg.wide_road_ratio_threshold = 0.82;
  rb_geo_cfg.wide_band_min_reliability = 0.18;
  rb_geo_cfg.enable_width_adaptive_fusion = true;
  rb_geo_cfg.wide_lane_boundary_blend = 0.55;
  rb_geo_cfg.observation_filter_alpha = 0.85;
  rb_geo_cfg.enable_continuous_curve_assist = true;

  MotionControllerConfig rb_ctrl_cfg{};
  rb_ctrl_cfg.target_speed = 0.50;
  rb_ctrl_cfg.max_speed = 0.50;
  rb_ctrl_cfg.max_yaw_rate = 1.50;
  rb_ctrl_cfg.kp = 3.30;
  rb_ctrl_cfg.speed_gain_damping = 0.26;
  rb_ctrl_cfg.enable_continuous_curve_assist = true;
  rb_ctrl_cfg.reverse_curve_speed = 0.28;
  rb_ctrl_cfg.sharp_curve_threshold = 0.42;
  rb_ctrl_cfg.sharp_curve_speed = 0.33;
  rb_ctrl_cfg.heading_gain = 1.15;
  rb_ctrl_cfg.low_confidence_threshold = 0.58;
  rb_ctrl_cfg.low_confidence_speed = 0.26;
  rb_ctrl_cfg.low_confidence_yaw_hold = 0.80;

  LaneGeometry base_geo(base_geo_cfg);
  MotionController base_ctrl(base_ctrl_cfg);
  LaneGeometry rb_geo(rb_geo_cfg);
  MotionController rb_ctrl(rb_ctrl_cfg);

  CurveAssistState base_curve{};
  MotionState base_motion{};
  CurveAssistState rb_curve{};
  MotionState rb_motion{};

  int base_turn_start = -1;
  int rb_turn_start = -1;
  double base_max_err = 0.0;
  double rb_max_err = 0.0;
  double base_min_speed = 1.0;
  double rb_min_speed = 1.0;
  int rb_low_conf_frames = 0;
  double rb_low_conf_max_speed = 0.0;

  std::printf("=== Replay %s [%d..%d] ===\n", prefix.c_str(), start_idx, end_idx);
  for (int idx = start_idx; idx <= end_idx; ++idx) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/%s_%03d.png", dir.c_str(), prefix.c_str(), idx);
    cv::Mat frame = cv::imread(buf, cv::IMREAD_COLOR);
    if (frame.empty()) {
      continue;
    }

    const auto seg = processor.segment(frame);
    if (!seg) {
      continue;
    }

    const auto base_res = base_geo.evaluate(*seg, base_curve);
    const auto base_cmd =
      base_ctrl.compute_lane_command(base_res.observation, base_curve, base_motion);
    MotionController::record_motion(
      false, base_cmd.linear_speed, base_cmd.yaw_rate, base_motion);

    const auto rb_res = rb_geo.evaluate(*seg, rb_curve);
    const auto rb_cmd =
      rb_ctrl.compute_lane_command(rb_res.observation, rb_curve, rb_motion);
    MotionController::record_motion(
      false, rb_cmd.linear_speed, rb_cmd.yaw_rate, rb_motion);

    if (base_turn_start < 0 && std::abs(base_cmd.yaw_rate) >= 0.15) {
      base_turn_start = idx;
    }
    if (rb_turn_start < 0 && std::abs(rb_cmd.yaw_rate) >= 0.15) {
      rb_turn_start = idx;
    }

    base_max_err = std::max(base_max_err, std::abs(base_res.error));
    rb_max_err = std::max(rb_max_err, std::abs(rb_res.error));
    base_min_speed = std::min(base_min_speed, base_cmd.linear_speed);
    rb_min_speed = std::min(rb_min_speed, rb_cmd.linear_speed);

    if (rb_res.observation.confidence < rb_ctrl_cfg.low_confidence_threshold) {
      rb_low_conf_frames++;
      rb_low_conf_max_speed = std::max(rb_low_conf_max_speed, rb_cmd.linear_speed);
    }

    std::printf(
      "idx=%03d s=%.2fm | BASE err=%+.3f yaw=%+.3f v=%.3f w=%.2f | "
      "RB err=%+.3f hd=%+.3f curv=%.3f conf=%.2f yaw=%+.3f v=%.3f w=(%.2f,%.2f)\n",
      idx,
      idx * 0.02,
      base_res.error,
      base_cmd.yaw_rate,
      base_cmd.linear_speed,
      base_res.observation.lane_width_ratio,
      rb_res.error,
      rb_res.observation.heading_proxy,
      rb_res.observation.curvature_proxy,
      rb_res.observation.confidence,
      rb_cmd.yaw_rate,
      rb_cmd.linear_speed,
      rb_res.observation.near_width_ratio,
      rb_res.observation.far_width_ratio);
  }

  std::printf(
    "SUMMARY %s: turn_start_idx BASE=%d (s=%.2fm) vs RB=%d (s=%.2fm) | "
    "max_err BASE=%.3f vs RB=%.3f | min_speed BASE=%.3f vs RB=%.3f | "
    "low_conf_frames=%d low_conf_speed_cap=%.3f\n\n",
    prefix.c_str(),
    base_turn_start,
    base_turn_start * 0.02,
    rb_turn_start,
    rb_turn_start * 0.02,
    base_max_err,
    rb_max_err,
    base_min_speed,
    rb_min_speed,
    rb_low_conf_frames,
    rb_low_conf_max_speed);
}

void run_closed_loop_curve_sim(const std::string & dir, const std::string & map_png)
{
  cv::Mat map_bgr = cv::imread(map_png, cv::IMREAD_COLOR);
  if (map_bgr.empty()) {
    return;
  }

  struct Wp { double x; double y; double s; };
  std::vector<Wp> wps;
  std::ifstream in(dir + "/waypoints.txt");
  Wp w{};
  while (in >> w.x >> w.y >> w.s) {
    wps.push_back(w);
  }
  if (wps.empty()) {
    return;
  }

  // Precompute ground (d, w) grid matching tools/sim/render.py
  const double h = 110.0;
  const double alpha = 12.5 * 3.14159265358979323846 / 180.0;
  const double tan_a = std::tan(alpha);
  const double f = 325.0;
  const double cx = 320.0;
  const double cy = 240.0;
  const double mpp = 3.527778;
  const double A = cy - f * tan_a;
  const double B = h * (cy * tan_a + f);

  cv::Mat grid_d(480, 640, CV_32FC1, cv::Scalar(-1.0F));
  cv::Mat grid_w(480, 640, CV_32FC1, cv::Scalar(0.0F));
  for (int v = 0; v < 480; ++v) {
    const double denom = static_cast<double>(v) - A;
    if (denom <= 1e-3) {
      continue;
    }
    const double d = (B - static_cast<double>(v) * h * tan_a) / denom;
    if (d < 0.0 || d > 5000.0) {
      continue;
    }
    for (int u = 0; u < 640; ++u) {
      grid_d.at<float>(v, u) = static_cast<float>(d);
      grid_w.at<float>(v, u) =
        static_cast<float>((static_cast<double>(u) - cx) * (d + h * tan_a) / f);
    }
  }

  auto render_cam = [&](double x_mm, double y_mm, double yaw) {
    const float c = static_cast<float>(std::cos(yaw));
    const float s = static_cast<float>(std::sin(yaw));
    cv::Mat mx(480, 640, CV_32FC1, cv::Scalar(-1e6F));
    cv::Mat my(480, 640, CV_32FC1, cv::Scalar(-1e6F));
    for (int v = 0; v < 480; ++v) {
      const float * dp = grid_d.ptr<float>(v);
      const float * wp = grid_w.ptr<float>(v);
      float * mxp = mx.ptr<float>(v);
      float * myp = my.ptr<float>(v);
      for (int u = 0; u < 640; ++u) {
        if (dp[u] < 0.0F) {
          continue;
        }
        const float gx = static_cast<float>(x_mm) + dp[u] * c + wp[u] * s;
        const float gy = static_cast<float>(y_mm) + dp[u] * s - wp[u] * c;
        mxp[u] = gx / static_cast<float>(mpp);
        myp[u] = static_cast<float>(map_bgr.rows - 1) - gy / static_cast<float>(mpp);
      }
    }
    cv::Mat out;
    cv::remap(
      map_bgr, out, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(166, 85, 37));
    return out;
  };

  ImageProcessorConfig img_cfg{};
  img_cfg.roi_top_ratio = 0.65;
  img_cfg.roi_bottom_ratio = 0.20;
  img_cfg.use_white_filter = true;
  img_cfg.white_s_max = 45;
  img_cfg.white_v_min = 80;
  ImageProcessor processor(img_cfg);

  auto run_mode = [&](bool use_rb, const char * label) {
    LaneGeometryConfig geo_cfg{};
    MotionControllerConfig ctrl_cfg{};
    ctrl_cfg.target_speed = 0.50;
    ctrl_cfg.max_speed = 0.50;
    ctrl_cfg.max_yaw_rate = 1.50;
    ctrl_cfg.kp = 3.30;
    ctrl_cfg.speed_gain_damping = use_rb ? 0.26 : 0.22;
    ctrl_cfg.enable_continuous_curve_assist = true;
    ctrl_cfg.reverse_curve_speed = use_rb ? 0.28 : 0.34;
    ctrl_cfg.sharp_curve_threshold = use_rb ? 0.42 : 0.50;
    ctrl_cfg.sharp_curve_speed = use_rb ? 0.33 : 0.41;

    if (use_rb) {
      geo_cfg.far_band_top_ratio = 0.22;
      geo_cfg.far_band_bottom_ratio = 0.42;
      geo_cfg.near_band_top_ratio = 0.50;
      geo_cfg.near_band_bottom_ratio = 0.72;
      geo_cfg.near_band_weight = 0.52;
      geo_cfg.far_band_weight = 0.48;
      geo_cfg.column_white_ratio = 0.50;
      geo_cfg.min_run_width = 18;
      geo_cfg.nominal_near_width_ratio = 0.68;
      geo_cfg.nominal_far_width_ratio = 0.48;
      geo_cfg.wide_road_ratio_threshold = 0.82;
      geo_cfg.wide_band_min_reliability = 0.18;
      geo_cfg.enable_width_adaptive_fusion = true;
      geo_cfg.wide_lane_boundary_blend = 0.45;
      geo_cfg.observation_filter_alpha = 0.85;
      geo_cfg.enable_continuous_curve_assist = true;

      ctrl_cfg.heading_gain = 0.85;
      ctrl_cfg.low_confidence_threshold = 0.58;
      ctrl_cfg.low_confidence_speed = 0.26;
      ctrl_cfg.low_confidence_yaw_hold = 0.80;
    }

    LaneGeometry geo(geo_cfg);
    MotionController ctrl(ctrl_cfg);
    CurveAssistState curve{};
    MotionState motion{};

    double x_mm = wps[195].x;
    double y_mm = wps[195].y;
    double yaw = std::atan2(wps[196].y - wps[195].y, wps[196].x - wps[195].x);
    const double dt = 1.0 / 30.0;

    int turn_step = -1;
    double turn_s = -1.0;
    double max_cte_mm = 0.0;
    double sum_cte_mm = 0.0;
    double max_visual_err = 0.0;
    double min_speed = 1.0;

    for (int step = 0; step < 90; ++step) {
      cv::Mat frame = render_cam(x_mm, y_mm, yaw);
      const auto seg = processor.segment(frame);
      const auto res = geo.evaluate(*seg, curve);
      const auto cmd = ctrl.compute_lane_command(res.observation, curve, motion);
      MotionController::record_motion(false, cmd.linear_speed, cmd.yaw_rate, motion);

      double best_d = 1e9;
      double cur_s = 0.0;
      for (const auto & wp : wps) {
        const double d = std::hypot(wp.x - x_mm, wp.y - y_mm);
        if (d < best_d) {
          best_d = d;
          cur_s = wp.s;
        }
      }

      if (turn_step < 0 && std::abs(cmd.yaw_rate) >= 0.20) {
        turn_step = step;
        turn_s = cur_s;
      }
      max_cte_mm = std::max(max_cte_mm, best_d);
      sum_cte_mm += best_d;
      max_visual_err = std::max(max_visual_err, std::abs(res.error));
      min_speed = std::min(min_speed, cmd.linear_speed);

      x_mm += cmd.linear_speed * 1000.0 * std::cos(yaw) * dt;
      y_mm += cmd.linear_speed * 1000.0 * std::sin(yaw) * dt;
      yaw += cmd.yaw_rate * dt;
    }

    std::printf(
      "CLOSED-LOOP [%s]: turn_start_step=%d (s=%.2fm) | max_CTE=%.1fmm mean_CTE=%.1fmm | "
      "max_visual_err=%.3f | min_speed=%.3fm/s\n",
      label,
      turn_step,
      turn_s,
      max_cte_mm,
      sum_cte_mm / 90.0,
      max_visual_err,
      min_speed);
  };

  run_mode(false, "BASE (fork.yaml)");
  run_mode(true, "RB (roundabout.yaml)");
}

}  // namespace

int main(int argc, char ** argv)
{
  try {
    if (argc >= 3 && std::string(argv[1]) == "--replay") {
      run_roundabout_sequence_replay(argv[2], "seqA", 25, 54);
      run_roundabout_sequence_replay(argv[2], "seqB", 198, 227);
      if (argc >= 4) {
        run_closed_loop_curve_sim(argv[2], argv[3]);
      }
      return 0;
    }
    test_fixed_image_repeatability_and_70_30_formula();
    test_parameterized_band_weights_and_ratios();
    test_reverse_curve_state_isolation();
    test_yellow_line_and_lost_line_state_machine();
    test_motion_controller_and_parameterized_lost_thresholds();
    test_debug_renderer();
    test_roundabout_wide_white_observation_and_turn_response();
    test_detection_turn_controller_scenarios();
    std::cout << "8 line_follower, roundabout & detection_turn_controller test groups passed\n";
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Test failure: " << e.what() << '\n';
    return 1;
  }
}
