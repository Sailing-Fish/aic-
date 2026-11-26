#include "line_follower/line_follower_node.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <std_msgs/msg/bool.hpp>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxBrakeSpeed = 1.5;
}  // namespace

namespace line_follower
{

LineFollowerNode::LineFollowerNode()
: Node("line_follower_node")
{
  declare_parameters();
  sync_runtime_parameters();
  log_effective_parameters();

  param_cb_handle_ = this->add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & /*params*/) {
      params_dirty_.store(true, std::memory_order_release);
      rcl_interfaces::msg::SetParametersResult res;
      res.successful = true;
      return res;
    });

  setup_interfaces();

  const double now_sec = this->now().seconds();
  track_state_machine_.reset(now_sec);
  curve_assist_ = CurveAssistState{};
  motion_state_ = MotionState{};
  obstacle_memory_.reset();
  corridor_planner_.reset();
  finish_guard_.reset();

  start_enabled_ = false;
  waiting_log_initialized_ = true;
  last_waiting_log_time_ = this->now();

  last_odom_time_ = this->now();
  last_ekf_odom_time_ = this->now();
  last_segment_pub_time_ = this->now();
  yellow_brake_start_ = this->now();
  last_brake_cmd_time_ = this->now();

  RCLCPP_INFO(
    this->get_logger(),
    "Line follower (finals) subscribing '%s', ekf '%s', det '%s', cmd '%s', mode=%s/%s/%s",
    image_topic_.c_str(), ekf_odom_topic_.c_str(), detection_topic_.c_str(),
    cmd_vel_topic_.c_str(), track_mode_.c_str(), lane_seg_mode_.c_str(),
    yellow_detect_mode_.c_str());
}

double LineFollowerNode::quat_to_yaw(double qx, double qy, double qz, double qw)
{
  const double siny_cosp = 2.0 * (qw * qz + qx * qy);
  const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
  return std::atan2(siny_cosp, cosy_cosp);
}

double LineFollowerNode::wrap_angle_rad(double angle)
{
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

SemanticObjectType LineFollowerNode::classify_detection_label(
  const std::string & label,
  int & digit_out)
{
  digit_out = 0;
  if (label.empty()) {
    return SemanticObjectType::UNKNOWN;
  }

  std::string lower;
  lower.reserve(label.size());
  for (unsigned char ch : label) {
    lower.push_back(static_cast<char>(std::tolower(ch)));
  }

  if (lower == "red_cone" || lower == "cone") {
    return SemanticObjectType::RED_CONE;
  }
  if (lower == "red_no_entry" || lower == "no_entry" || lower == "forbid" || lower == "forbidden") {
    return SemanticObjectType::RED_NO_ENTRY;
  }
  if (lower == "yellow_stop_line" || lower == "yellow_line" || lower == "stop_line") {
    return SemanticObjectType::YELLOW_STOP_LINE;
  }

  // Parse Bonus digits "1".."8" or "d1".."d8" or "digit_1".."digit_8"
  std::string num_part = lower;
  if (num_part.rfind("digit_", 0) == 0) {
    num_part = num_part.substr(6);
  } else if (num_part.size() >= 2 && num_part[0] == 'd') {
    num_part = num_part.substr(1);
  }
  if (num_part.size() == 1 && num_part[0] >= '1' && num_part[0] <= '8') {
    digit_out = num_part[0] - '0';
    return SemanticObjectType::BONUS_DIGIT;
  }

  return SemanticObjectType::UNKNOWN;
}

void LineFollowerNode::declare_parameters()
{
  this->declare_parameter("image_topic", "/ascamera/camera_publisher/rgb/image0");
  this->declare_parameter("cmd_vel_topic", "/cmd_vel_follower");
  this->declare_parameter("debug_image_topic", "line_follower/debug_image");
  this->declare_parameter("line_state_topic", "line_follower/track_state");
  this->declare_parameter("start_topic", "line_follower/start_cmd");
  this->declare_parameter("stop_cmd_topic", "/cmd_vel_stop");
  this->declare_parameter("odom_topic", "/odom");
  this->declare_parameter("ekf_odom_topic", "/odom_combined");
  this->declare_parameter("detection_topic", "hobot_dnn_detection");
  this->declare_parameter("detection_timeout_sec", 0.35);
  this->declare_parameter("segment_topic", "line_follower/segment");
  this->declare_parameter("voltage_topic", "/PowerVoltage");

  // ============================================================
  // Finals mode switches (lane_seg_mode supports relative_border | hybrid | legacy_hsv)
  // ============================================================
  this->declare_parameter("track_mode", "fork");
  this->declare_parameter("lane_seg_mode", "relative_border");
  this->declare_parameter("yellow_detect_mode", "geometric_cv");
  this->declare_parameter("enable_semantic_hole_fill", true);
  this->declare_parameter("enable_obstacle_memory", true);
  this->declare_parameter("enable_cv_red_fallback", true);
  this->declare_parameter("allow_dual_digit_cover", false);
  this->declare_parameter("enable_cbf_shield", false);
  this->declare_parameter("enable_risk_aware_decision", true);
  this->declare_parameter("enable_brake_adaptation", true);

  // ============================================================
  // Orthogonal motion & ROI parameters
  // ============================================================
  this->declare_parameter("target_speed", 0.44);
  this->declare_parameter("max_speed", 0.85);
  this->declare_parameter("min_speed", 0.0);
  this->declare_parameter("max_yaw_rate", 1.6);
  this->declare_parameter("speed_gain_damping", 0.45);
  this->declare_parameter("roi_top_ratio", 0.68);
  this->declare_parameter("roi_bottom_ratio", 0.18);
  this->declare_parameter("yellow_roi_bottom_ratio", 0.0);
  this->declare_parameter("publish_debug_image", true);

  // ============================================================
  // Image processing (relative_border / hybrid / legacy_hsv fallback)
  // ============================================================
  this->declare_parameter("gaussian_kernel", 5);
  this->declare_parameter("binary_threshold", 160.0);
  this->declare_parameter("morph_kernel", 3);
  this->declare_parameter("morph_iterations", 2);
  this->declare_parameter("min_contour_area", 2500.0);
  this->declare_parameter("invert_binary", false);
  this->declare_parameter("use_white_filter", true);
  this->declare_parameter("white_h_low", 0);
  this->declare_parameter("white_h_high", 180);
  this->declare_parameter("white_s_max", 45);
  this->declare_parameter("white_v_min", 80);

  this->declare_parameter("blue_margin", 22);
  this->declare_parameter("lab_blue_b_max", 116);
  this->declare_parameter("black_block_size", 31);
  this->declare_parameter("black_c_offset", 14.0);
  this->declare_parameter("black_v_max", 115);
  this->declare_parameter("lab_chroma_radius", 24.0);
  this->declare_parameter("lab_l_min_offset", 38.0);
  this->declare_parameter("hybrid_white_s_max", 75);
  this->declare_parameter("hybrid_white_v_min", 65);
  this->declare_parameter("semantic_hole_fill_dilate_px", 2);

  // ============================================================
  // Lane band geometry & fork split detection
  // ============================================================
  this->declare_parameter("far_band_top_ratio", 0.28);
  this->declare_parameter("far_band_bottom_ratio", 0.48);
  this->declare_parameter("near_band_top_ratio", 0.55);
  this->declare_parameter("near_band_bottom_ratio", 0.80);
  this->declare_parameter("near_band_weight", 0.60);
  this->declare_parameter("far_band_weight", 0.40);
  this->declare_parameter("band_column_white_ratio", 0.40);
  this->declare_parameter("band_min_run_width", 18);
  this->declare_parameter("nominal_near_width_ratio", 0.68);
  this->declare_parameter("nominal_far_width_ratio", 0.48);
  this->declare_parameter("wide_road_ratio_threshold", 0.84);
  this->declare_parameter("wide_band_min_reliability", 0.18);
  this->declare_parameter("enable_width_adaptive_fusion", true);
  this->declare_parameter("wide_lane_boundary_blend", 0.45);
  this->declare_parameter("observation_filter_alpha", 0.85);
  this->declare_parameter("fork_min_gap_px", 14);
  this->declare_parameter("fork_min_branch_width_px", 18);

  this->declare_parameter("enable_continuous_curve_assist", true);
  this->declare_parameter("reverse_curve_min_error", 0.12);
  this->declare_parameter("reverse_curve_threshold", 0.35);
  this->declare_parameter("reverse_curve_confirm_frames", 3);
  this->declare_parameter("reverse_curve_clear_frames", 4);
  this->declare_parameter("low_confidence_threshold", 0.55);
  this->declare_parameter("low_confidence_speed", 0.24);
  this->declare_parameter("low_confidence_yaw_hold", 0.80);

  // ============================================================
  // Geometric yellow stop line & dead-end-ahead cues
  // ============================================================
  this->declare_parameter("use_yellow_filter", true);
  this->declare_parameter("yellow_h_low", 20);
  this->declare_parameter("yellow_h_high", 40);
  this->declare_parameter("yellow_s_min", 80);
  this->declare_parameter("yellow_v_min", 120);
  this->declare_parameter("yellow_min_aspect_ratio", 2.5);
  this->declare_parameter("yellow_min_width_ratio", 0.20);
  this->declare_parameter("yellow_min_area_px", 120.0);
  this->declare_parameter("yellow_ahead_strip_height_px", 28);
  this->declare_parameter("yellow_dead_end_blue_ratio", 0.35);
  this->declare_parameter("yellow_dead_end_dark_ratio", 0.20);
  this->declare_parameter("yellow_dead_end_combined_ratio", 0.50);

  this->declare_parameter("yellow_stop_duration", 1.0);
  this->declare_parameter("yellow_clear_frames", 10);
  this->declare_parameter("yellow_stop_min_time", 0.35);
  this->declare_parameter("odom_still_threshold", 0.02);
  this->declare_parameter("odom_still_hold", 0.18);
  this->declare_parameter("odom_max_age_sec", 0.20);
  this->declare_parameter("yellow_require_odom_still", true);
  this->declare_parameter("yellow_max_hold_sec", 3.0);

  // Reverse braking pulse
  this->declare_parameter("yellow_brake_speed", 0.0);
  this->declare_parameter("yellow_brake_period_s", 0.2);
  this->declare_parameter("yellow_brake_pulse_s", 0.0);
  this->declare_parameter("yellow_brake_release_speed", 0.08);
  this->declare_parameter("yellow_brake_max_time_s", 0.8);

  // ============================================================
  // B10: CV Red Obstacle Fallback parameters (under enable_cv_red_fallback)
  // ============================================================
  this->declare_parameter("red_h_low_1", 0);
  this->declare_parameter("red_h_high_1", 10);
  this->declare_parameter("red_h_low_2", 160);
  this->declare_parameter("red_h_high_2", 180);
  this->declare_parameter("red_s_min", 90);
  this->declare_parameter("red_v_min", 70);
  this->declare_parameter("red_cone_area_ratio_threshold", 0.0006);

  // ============================================================
  // Lost-line safety control
  // ============================================================
  this->declare_parameter("line_lost_stop_time", 0.6);
  this->declare_parameter("line_lost_forward_speed", 0.15);
  this->declare_parameter("line_lost_yaw_hold_factor", 0.85);
  this->declare_parameter("line_lost_slow_time", 0.4);
  this->declare_parameter("line_lost_slow_factor", 0.4);
  this->declare_parameter("line_lost_full_stop_time", 1.2);
  this->declare_parameter("lost_line_safe_mode", true);
  this->declare_parameter("lost_line_safe_speed", 0.12);
  this->declare_parameter("lost_line_safe_stop_time", 0.35);

  // ============================================================
  // GroundProjector / Homography parameters
  // ============================================================
  this->declare_parameter("use_homography", false);
  this->declare_parameter(
    "homography_matrix",
    std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
  this->declare_parameter("cam_height_m", 0.11);
  this->declare_parameter("cam_pitch_deg", 12.5);
  this->declare_parameter("focal_px", 325.0);
  this->declare_parameter("cam_cx", 320.0);
  this->declare_parameter("cam_cy", 240.0);
  this->declare_parameter("front_overhang_m", 0.06);

  // ============================================================
  // ObstacleMemory parameters (including B3 rear_axle_to_bumper_m)
  // ============================================================
  this->declare_parameter("obstacle_merge_radius_m", 0.16);
  this->declare_parameter("obstacle_memory_ttl_sec", 1.8);
  this->declare_parameter("car_length_m", 0.280);
  this->declare_parameter("rear_axle_to_bumper_m", 0.18);
  this->declare_parameter("obstacle_rear_keep_margin_m", 0.15);

  // ============================================================
  // CorridorPlanner parameters (including B5 two-zone & stagger params)
  // ============================================================
  this->declare_parameter("car_width_m", 0.191);
  this->declare_parameter("default_lane_width_m", 0.430);
  this->declare_parameter("border_margin_m", 0.025);
  this->declare_parameter("cone_margin_m", 0.025);
  this->declare_parameter("no_entry_margin_m", 0.030);
  this->declare_parameter("obstacle_lookahead_m", 1.10);
  this->declare_parameter("obstacle_steer_horizon_m", 0.65);
  this->declare_parameter("obstacle_curve_min_horizon_m", 0.45);
  this->declare_parameter("obstacle_stagger_gap_m", 0.25);
  this->declare_parameter("obstacle_rear_keep_m", 0.28);
  this->declare_parameter("obstacle_speed_mps", 0.26);
  this->declare_parameter("no_entry_speed_mps", 0.22);
  this->declare_parameter("blocked_creep_speed_mps", 0.12);
  this->declare_parameter("fork_outer_branch", -1);
  this->declare_parameter("fork_outer_time_penalty_s", 3.0);
  this->declare_parameter("fork_min_votes", 3);
  this->declare_parameter("fork_commit_dist_m", 1.10);
  this->declare_parameter("digit_min_conf", 0.55);
  this->declare_parameter("digit_min_frames", 2);
  this->declare_parameter("digit_lookahead_max_m", 1.15);
  this->declare_parameter("max_digit_lateral_shift_m", 0.12);
  this->declare_parameter("digit_approach_speed_mps", 0.30);
  this->declare_parameter("corridor_kp_y", 3.8);
  this->declare_parameter("corridor_heading_gain", 0.85);

  // ============================================================
  // Tier 1: Ackermann 3-disc DCBF shield & CRC bottom-edge padding
  // ============================================================
  this->declare_parameter("cbf_wheelbase_m", 0.142);
  this->declare_parameter("cbf_max_steer_deg", 35.0);
  this->declare_parameter("cbf_horizon_steps", 6);
  this->declare_parameter("cbf_step_dt_sec", 0.06);
  this->declare_parameter("cbf_gamma", 0.60);
  this->declare_parameter("crc_no_entry_pad_y_px", 0.0);
  this->declare_parameter("crc_cone_pad_y_px", 0.0);
  this->declare_parameter("crc_yellow_pad_y_px", 0.0);

  // ============================================================
  // FinishGuard & two-stage yellow stop parameters
  // ============================================================
  this->declare_parameter("odom_scale_factor", 1.0);
  this->declare_parameter("finish_min_dist_m", 8.5);
  this->declare_parameter("finish_force_dist_m", 11.5);
  this->declare_parameter("finish_min_cum_yaw_rad", 0.0);
  this->declare_parameter("yellow_pre_brake_dist_m", 0.70);
  this->declare_parameter("yellow_creep_speed_mps", 0.24);
  this->declare_parameter("yellow_final_ramp_dist_m", 0.35);
  this->declare_parameter("yellow_final_creep_speed_mps", 0.18);
  this->declare_parameter("yellow_brake_decel_mps2", 1.8);
  this->declare_parameter("voltage_decel_slope", 0.0);
  this->declare_parameter("yellow_control_latency_s", 0.10);
  this->declare_parameter("yellow_bumper_margin_m", 0.08);
  this->declare_parameter("yellow_rearm_dist_m", 0.35);

  image_topic_ = this->get_parameter("image_topic").as_string();
  cmd_vel_topic_ = this->get_parameter("cmd_vel_topic").as_string();
  debug_image_topic_ = this->get_parameter("debug_image_topic").as_string();
  line_state_topic_ = this->get_parameter("line_state_topic").as_string();
  start_topic_ = this->get_parameter("start_topic").as_string();
  stop_cmd_topic_ = this->get_parameter("stop_cmd_topic").as_string();
  odom_topic_ = this->get_parameter("odom_topic").as_string();
  ekf_odom_topic_ = this->get_parameter("ekf_odom_topic").as_string();
  detection_topic_ = this->get_parameter("detection_topic").as_string();
  segment_topic_ = this->get_parameter("segment_topic").as_string();
  voltage_topic_ = this->get_parameter("voltage_topic").as_string();
}

void LineFollowerNode::sync_runtime_parameters()
{
  track_mode_ = this->get_parameter("track_mode").as_string();
  if (track_mode_ != "fork" && track_mode_ != "roundabout") {
    RCLCPP_WARN(
      this->get_logger(),
      "[DEC] param track_mode='%s' unknown (expected fork/roundabout), using fork",
      track_mode_.c_str());
    track_mode_ = "fork";
  }

  lane_seg_mode_ = this->get_parameter("lane_seg_mode").as_string();
  yellow_detect_mode_ = this->get_parameter("yellow_detect_mode").as_string();
  enable_semantic_hole_fill_ = this->get_parameter("enable_semantic_hole_fill").as_bool();
  enable_obstacle_memory_ = this->get_parameter("enable_obstacle_memory").as_bool();
  enable_cv_red_fallback_ = this->get_parameter("enable_cv_red_fallback").as_bool();
  allow_dual_digit_cover_ = this->get_parameter("allow_dual_digit_cover").as_bool();
  enable_cbf_shield_ = this->get_parameter("enable_cbf_shield").as_bool();
  enable_risk_aware_decision_ = this->get_parameter("enable_risk_aware_decision").as_bool();
  crc_no_entry_pad_y_px_ = std::max(0.0, this->get_parameter("crc_no_entry_pad_y_px").as_double());
  crc_cone_pad_y_px_ = std::max(0.0, this->get_parameter("crc_cone_pad_y_px").as_double());
  crc_yellow_pad_y_px_ = std::max(0.0, this->get_parameter("crc_yellow_pad_y_px").as_double());
  detection_timeout_sec_ = std::max(0.05, this->get_parameter("detection_timeout_sec").as_double());

  target_speed_ = std::max(0.0, this->get_parameter("target_speed").as_double());
  max_speed_ = std::max(0.0, this->get_parameter("max_speed").as_double());
  min_speed_ = std::max(0.0, this->get_parameter("min_speed").as_double());
  max_yaw_rate_ = std::max(0.0, this->get_parameter("max_yaw_rate").as_double());
  speed_gain_damping_ = std::max(0.0, this->get_parameter("speed_gain_damping").as_double());

  roi_top_ratio_ = clamp(this->get_parameter("roi_top_ratio").as_double(), 0.0, 1.0);
  roi_bottom_ratio_ = clamp(this->get_parameter("roi_bottom_ratio").as_double(), 0.0, 0.5);
  yellow_roi_bottom_ratio_ =
    clamp(this->get_parameter("yellow_roi_bottom_ratio").as_double(), 0.0, 0.5);
  publish_debug_image_ = this->get_parameter("publish_debug_image").as_bool();

  gaussian_kernel_ = static_cast<int>(this->get_parameter("gaussian_kernel").as_int());
  if (gaussian_kernel_ <= 0) {
    gaussian_kernel_ = 1;
  }
  if (gaussian_kernel_ % 2 == 0) {
    ++gaussian_kernel_;
  }
  binary_threshold_ = this->get_parameter("binary_threshold").as_double();
  morph_kernel_ =
    std::max<int>(1, static_cast<int>(this->get_parameter("morph_kernel").as_int()));
  morph_iterations_ =
    std::max<int>(1, static_cast<int>(this->get_parameter("morph_iterations").as_int()));
  min_contour_area_ = this->get_parameter("min_contour_area").as_double();
  invert_binary_ = this->get_parameter("invert_binary").as_bool();

  use_white_filter_ = this->get_parameter("use_white_filter").as_bool();
  white_h_low_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_h_low").as_int()), 0, 180);
  white_h_high_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_h_high").as_int()), 0, 180);
  if (white_h_low_ > white_h_high_) {
    std::swap(white_h_low_, white_h_high_);
  }
  white_s_max_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_s_max").as_int()), 0, 255);
  white_v_min_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_v_min").as_int()), 0, 255);

  use_yellow_filter_ = this->get_parameter("use_yellow_filter").as_bool();
  yellow_h_low_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_h_low").as_int()), 0, 180);
  yellow_h_high_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_h_high").as_int()), 0, 180);
  if (yellow_h_low_ > yellow_h_high_) {
    std::swap(yellow_h_low_, yellow_h_high_);
  }
  yellow_s_min_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_s_min").as_int()), 0, 255);
  yellow_v_min_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_v_min").as_int()), 0, 255);

  yellow_stop_duration_ = this->get_parameter("yellow_stop_duration").as_double();
  yellow_clear_frames_required_ = std::max<int>(
    1, static_cast<int>(this->get_parameter("yellow_clear_frames").as_int()));
  yellow_stop_min_time_ = this->get_parameter("yellow_stop_min_time").as_double();
  odom_still_threshold_ = this->get_parameter("odom_still_threshold").as_double();
  odom_still_hold_ = this->get_parameter("odom_still_hold").as_double();
  odom_max_age_sec_ = this->get_parameter("odom_max_age_sec").as_double();
  yellow_require_odom_still_ = this->get_parameter("yellow_require_odom_still").as_bool();
  yellow_max_hold_sec_ = this->get_parameter("yellow_max_hold_sec").as_double();

  const double brake_raw = std::max(0.0, this->get_parameter("yellow_brake_speed").as_double());
  yellow_brake_speed_ = std::min(brake_raw, kMaxBrakeSpeed);
  if (brake_raw > kMaxBrakeSpeed + 1e-6) {
    RCLCPP_WARN(
      this->get_logger(),
      "[DEC] param yellow_brake_speed=%.2f 超过上限 %.1f，已截断",
      brake_raw, kMaxBrakeSpeed);
  }
  yellow_brake_period_s_ =
    std::max(0.0, this->get_parameter("yellow_brake_period_s").as_double());
  yellow_brake_pulse_s_ =
    std::max(0.0, this->get_parameter("yellow_brake_pulse_s").as_double());
  yellow_brake_release_speed_ =
    std::max(0.0, this->get_parameter("yellow_brake_release_speed").as_double());
  yellow_brake_max_time_s_ =
    std::max(0.0, this->get_parameter("yellow_brake_max_time_s").as_double());

  // B10: Sync & cache CV red obstacle fallback parameters
  red_h_low_1_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("red_h_low_1").as_int()), 0, 180);
  red_h_high_1_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("red_h_high_1").as_int()), 0, 180);
  if (red_h_low_1_ > red_h_high_1_) {
    std::swap(red_h_low_1_, red_h_high_1_);
  }
  red_h_low_2_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("red_h_low_2").as_int()), 0, 180);
  red_h_high_2_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("red_h_high_2").as_int()), 0, 180);
  if (red_h_low_2_ > red_h_high_2_) {
    std::swap(red_h_low_2_, red_h_high_2_);
  }
  red_s_min_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("red_s_min").as_int()), 0, 255);
  red_v_min_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("red_v_min").as_int()), 0, 255);
  red_cone_area_ratio_threshold_ =
    std::max(0.0, this->get_parameter("red_cone_area_ratio_threshold").as_double());

  ImageProcessorConfig img_cfg{};
  img_cfg.roi_top_ratio = roi_top_ratio_;
  img_cfg.roi_bottom_ratio = roi_bottom_ratio_;
  img_cfg.yellow_roi_bottom_ratio = yellow_roi_bottom_ratio_;
  img_cfg.gaussian_kernel = gaussian_kernel_;
  img_cfg.binary_threshold = binary_threshold_;
  img_cfg.morph_kernel = morph_kernel_;
  img_cfg.morph_iterations = morph_iterations_;
  img_cfg.min_contour_area = min_contour_area_;
  img_cfg.invert_binary = invert_binary_;
  img_cfg.use_white_filter = use_white_filter_;
  img_cfg.white_h_low = white_h_low_;
  img_cfg.white_h_high = white_h_high_;
  img_cfg.white_s_max = white_s_max_;
  img_cfg.white_v_min = white_v_min_;
  img_cfg.use_yellow_filter = use_yellow_filter_;
  img_cfg.yellow_h_low = yellow_h_low_;
  img_cfg.yellow_h_high = yellow_h_high_;
  img_cfg.yellow_s_min = yellow_s_min_;
  img_cfg.yellow_v_min = yellow_v_min_;
  img_cfg.lane_seg_mode = lane_seg_mode_;
  img_cfg.blue_margin = static_cast<int>(this->get_parameter("blue_margin").as_int());
  img_cfg.lab_blue_b_max = static_cast<int>(this->get_parameter("lab_blue_b_max").as_int());
  img_cfg.black_block_size = static_cast<int>(this->get_parameter("black_block_size").as_int());
  img_cfg.black_c_offset = this->get_parameter("black_c_offset").as_double();
  img_cfg.black_v_max = static_cast<int>(this->get_parameter("black_v_max").as_int());
  img_cfg.lab_chroma_radius = this->get_parameter("lab_chroma_radius").as_double();
  img_cfg.lab_l_min_offset = this->get_parameter("lab_l_min_offset").as_double();
  img_cfg.hybrid_white_s_max =
    static_cast<int>(this->get_parameter("hybrid_white_s_max").as_int());
  img_cfg.hybrid_white_v_min =
    static_cast<int>(this->get_parameter("hybrid_white_v_min").as_int());
  img_cfg.enable_semantic_hole_fill = enable_semantic_hole_fill_;
  img_cfg.semantic_hole_fill_dilate_px =
    static_cast<int>(this->get_parameter("semantic_hole_fill_dilate_px").as_int());
  img_cfg.yellow_detect_mode = yellow_detect_mode_;
  img_cfg.yellow_min_aspect_ratio = this->get_parameter("yellow_min_aspect_ratio").as_double();
  img_cfg.yellow_min_width_ratio = this->get_parameter("yellow_min_width_ratio").as_double();
  img_cfg.yellow_min_area_px = this->get_parameter("yellow_min_area_px").as_double();
  img_cfg.yellow_ahead_strip_height_px =
    static_cast<int>(this->get_parameter("yellow_ahead_strip_height_px").as_int());
  img_cfg.yellow_dead_end_blue_ratio =
    this->get_parameter("yellow_dead_end_blue_ratio").as_double();
  img_cfg.yellow_dead_end_dark_ratio =
    this->get_parameter("yellow_dead_end_dark_ratio").as_double();
  img_cfg.yellow_dead_end_combined_ratio =
    this->get_parameter("yellow_dead_end_combined_ratio").as_double();
  image_processor_.set_config(img_cfg);

  // Sync GroundProjector (enable homography only if vector is valid & invertible)
  use_homography_ = this->get_parameter("use_homography").as_bool();
  GroundProjectorConfig proj_cfg{};
  proj_cfg.use_homography = false;
  proj_cfg.cam_height_m = this->get_parameter("cam_height_m").as_double();
  proj_cfg.cam_pitch_deg = this->get_parameter("cam_pitch_deg").as_double();
  proj_cfg.focal_px = this->get_parameter("focal_px").as_double();
  proj_cfg.cx = this->get_parameter("cam_cx").as_double();
  proj_cfg.cy = this->get_parameter("cam_cy").as_double();
  proj_cfg.front_overhang_m = this->get_parameter("front_overhang_m").as_double();
  ground_projector_.set_config(proj_cfg);
  if (use_homography_) {
    const auto h_vec = this->get_parameter("homography_matrix").as_double_array();
    if (!ground_projector_.set_homography_from_vector(h_vec, true)) {
      use_homography_ = false;
    }
  }

  far_band_top_ratio_ = this->get_parameter("far_band_top_ratio").as_double();
  far_band_bottom_ratio_ = this->get_parameter("far_band_bottom_ratio").as_double();
  near_band_top_ratio_ = this->get_parameter("near_band_top_ratio").as_double();
  near_band_bottom_ratio_ = this->get_parameter("near_band_bottom_ratio").as_double();
  near_band_weight_ = this->get_parameter("near_band_weight").as_double();
  far_band_weight_ = this->get_parameter("far_band_weight").as_double();
  enable_curve_assist_ = this->get_parameter("enable_continuous_curve_assist").as_bool();

  LaneGeometryConfig geo_cfg{};
  geo_cfg.far_band_top_ratio = far_band_top_ratio_;
  geo_cfg.far_band_bottom_ratio = far_band_bottom_ratio_;
  geo_cfg.near_band_top_ratio = near_band_top_ratio_;
  geo_cfg.near_band_bottom_ratio = near_band_bottom_ratio_;
  geo_cfg.near_band_weight = near_band_weight_;
  geo_cfg.far_band_weight = far_band_weight_;
  geo_cfg.column_white_ratio =
    this->get_parameter("band_column_white_ratio").as_double();
  geo_cfg.min_run_width =
    static_cast<int>(this->get_parameter("band_min_run_width").as_int());
  geo_cfg.enable_continuous_curve_assist = enable_curve_assist_;
  geo_cfg.reverse_curve_min_error =
    this->get_parameter("reverse_curve_min_error").as_double();
  geo_cfg.reverse_curve_threshold =
    this->get_parameter("reverse_curve_threshold").as_double();
  geo_cfg.reverse_curve_confirm_frames =
    static_cast<int>(this->get_parameter("reverse_curve_confirm_frames").as_int());
  geo_cfg.reverse_curve_clear_frames =
    static_cast<int>(this->get_parameter("reverse_curve_clear_frames").as_int());
  geo_cfg.nominal_near_width_ratio =
    this->get_parameter("nominal_near_width_ratio").as_double();
  geo_cfg.nominal_far_width_ratio =
    this->get_parameter("nominal_far_width_ratio").as_double();
  geo_cfg.wide_road_ratio_threshold =
    this->get_parameter("wide_road_ratio_threshold").as_double();
  geo_cfg.wide_band_min_reliability =
    this->get_parameter("wide_band_min_reliability").as_double();
  geo_cfg.enable_width_adaptive_fusion =
    this->get_parameter("enable_width_adaptive_fusion").as_bool();
  geo_cfg.wide_lane_boundary_blend =
    this->get_parameter("wide_lane_boundary_blend").as_double();
  geo_cfg.observation_filter_alpha =
    this->get_parameter("observation_filter_alpha").as_double();
  geo_cfg.fork_min_gap_px =
    static_cast<int>(this->get_parameter("fork_min_gap_px").as_int());
  geo_cfg.fork_min_branch_width_px =
    static_cast<int>(this->get_parameter("fork_min_branch_width_px").as_int());
  lane_geometry_.set_config(geo_cfg);

  line_lost_stop_time_ = this->get_parameter("line_lost_stop_time").as_double();
  line_lost_forward_speed_ = this->get_parameter("line_lost_forward_speed").as_double();
  line_lost_yaw_hold_factor_ = this->get_parameter("line_lost_yaw_hold_factor").as_double();
  line_lost_slow_time_ = this->get_parameter("line_lost_slow_time").as_double();
  line_lost_slow_factor_ = this->get_parameter("line_lost_slow_factor").as_double();
  line_lost_full_stop_time_ = this->get_parameter("line_lost_full_stop_time").as_double();

  TrackStateConfig state_cfg{};
  state_cfg.yellow_stop_duration = yellow_stop_duration_;
  state_cfg.yellow_clear_frames_required = yellow_clear_frames_required_;
  state_cfg.yellow_stop_min_time = yellow_stop_min_time_;
  state_cfg.odom_still_threshold = odom_still_threshold_;
  state_cfg.odom_still_hold = odom_still_hold_;
  state_cfg.odom_max_age_sec = odom_max_age_sec_;
  state_cfg.yellow_require_odom_still = yellow_require_odom_still_;
  state_cfg.yellow_max_hold_sec = yellow_max_hold_sec_;
  state_cfg.line_lost_stop_time = line_lost_stop_time_;
  track_state_machine_.set_config(state_cfg);

  MotionControllerConfig motion_cfg{};
  motion_cfg.target_speed = target_speed_;
  motion_cfg.max_speed = max_speed_;
  motion_cfg.min_speed = min_speed_;
  motion_cfg.max_yaw_rate = max_yaw_rate_;
  motion_cfg.speed_gain_damping = speed_gain_damping_;
  motion_cfg.low_confidence_threshold =
    this->get_parameter("low_confidence_threshold").as_double();
  motion_cfg.low_confidence_speed =
    this->get_parameter("low_confidence_speed").as_double();
  motion_cfg.low_confidence_yaw_hold =
    this->get_parameter("low_confidence_yaw_hold").as_double();
  motion_cfg.line_lost_forward_speed = line_lost_forward_speed_;
  motion_cfg.line_lost_yaw_hold_factor = line_lost_yaw_hold_factor_;
  motion_cfg.line_lost_slow_time = line_lost_slow_time_;
  motion_cfg.line_lost_slow_factor = line_lost_slow_factor_;
  motion_cfg.line_lost_full_stop_time = line_lost_full_stop_time_;
  motion_cfg.lost_line_safe_mode =
    this->get_parameter("lost_line_safe_mode").as_bool();
  motion_cfg.lost_line_safe_speed =
    this->get_parameter("lost_line_safe_speed").as_double();
  motion_cfg.lost_line_safe_stop_time =
    this->get_parameter("lost_line_safe_stop_time").as_double();
  motion_cfg.corridor_kp_y = this->get_parameter("corridor_kp_y").as_double();
  motion_cfg.corridor_heading_gain = this->get_parameter("corridor_heading_gain").as_double();
  motion_controller_.set_config(motion_cfg);

  // Sync ObstacleMemory (including B3 rear_axle_to_bumper_m)
  ObstacleMemoryConfig mem_cfg{};
  mem_cfg.merge_radius_m = this->get_parameter("obstacle_merge_radius_m").as_double();
  mem_cfg.memory_ttl_sec = this->get_parameter("obstacle_memory_ttl_sec").as_double();
  mem_cfg.car_length_m = this->get_parameter("car_length_m").as_double();
  mem_cfg.rear_axle_to_bumper_m = this->get_parameter("rear_axle_to_bumper_m").as_double();
  mem_cfg.rear_keep_margin_m = this->get_parameter("obstacle_rear_keep_margin_m").as_double();
  obstacle_memory_.set_config(mem_cfg);

  // Sync CorridorPlanner (including B5 horizon & stagger params)
  CorridorPlannerConfig cor_cfg{};
  cor_cfg.car_width_m = this->get_parameter("car_width_m").as_double();
  cor_cfg.car_length_m = mem_cfg.car_length_m;
  cor_cfg.default_lane_width_m = this->get_parameter("default_lane_width_m").as_double();
  cor_cfg.border_margin_m = this->get_parameter("border_margin_m").as_double();
  cor_cfg.cone_margin_m = this->get_parameter("cone_margin_m").as_double();
  cor_cfg.no_entry_margin_m = this->get_parameter("no_entry_margin_m").as_double();
  cor_cfg.obstacle_lookahead_m = this->get_parameter("obstacle_lookahead_m").as_double();
  cor_cfg.obstacle_steer_horizon_m =
    this->get_parameter("obstacle_steer_horizon_m").as_double();
  cor_cfg.obstacle_curve_min_horizon_m =
    this->get_parameter("obstacle_curve_min_horizon_m").as_double();
  cor_cfg.obstacle_stagger_gap_m = this->get_parameter("obstacle_stagger_gap_m").as_double();
  cor_cfg.obstacle_rear_keep_m = this->get_parameter("obstacle_rear_keep_m").as_double();
  cor_cfg.cruise_speed_mps = target_speed_;
  cor_cfg.obstacle_speed_mps = this->get_parameter("obstacle_speed_mps").as_double();
  cor_cfg.no_entry_speed_mps = this->get_parameter("no_entry_speed_mps").as_double();
  cor_cfg.blocked_creep_speed_mps = this->get_parameter("blocked_creep_speed_mps").as_double();
  cor_cfg.track_mode = track_mode_;
  cor_cfg.fork_outer_branch = static_cast<int>(this->get_parameter("fork_outer_branch").as_int());
  cor_cfg.fork_outer_time_penalty_s =
    this->get_parameter("fork_outer_time_penalty_s").as_double();
  cor_cfg.fork_min_votes = static_cast<int>(this->get_parameter("fork_min_votes").as_int());
  cor_cfg.fork_commit_dist_m = this->get_parameter("fork_commit_dist_m").as_double();
  cor_cfg.digit_min_conf = this->get_parameter("digit_min_conf").as_double();
  cor_cfg.digit_min_frames = static_cast<int>(this->get_parameter("digit_min_frames").as_int());
  cor_cfg.digit_lookahead_max_m = this->get_parameter("digit_lookahead_max_m").as_double();
  cor_cfg.max_digit_lateral_shift_m =
    this->get_parameter("max_digit_lateral_shift_m").as_double();
  cor_cfg.digit_approach_speed_mps =
    this->get_parameter("digit_approach_speed_mps").as_double();
  cor_cfg.allow_dual_digit_cover = allow_dual_digit_cover_;
  cor_cfg.enable_risk_aware_decision = enable_risk_aware_decision_;
  corridor_planner_.set_config(cor_cfg);

  // Sync AckermannCbfShield (Tier 1 3-disc multi-step DCBF safety shield)
  AckermannCbfConfig cbf_cfg{};
  cbf_cfg.car_length_m = mem_cfg.car_length_m;
  cbf_cfg.car_width_m = cor_cfg.car_width_m;
  cbf_cfg.rear_axle_to_bumper_m = mem_cfg.rear_axle_to_bumper_m;
  cbf_cfg.wheelbase_m = this->get_parameter("cbf_wheelbase_m").as_double();
  cbf_cfg.max_steer_rad = this->get_parameter("cbf_max_steer_deg").as_double() * (kPi / 180.0);
  cbf_cfg.max_yaw_rate = max_yaw_rate_;
  cbf_cfg.default_lane_width_m = cor_cfg.default_lane_width_m;
  cbf_cfg.border_margin_m = cor_cfg.border_margin_m;
  cbf_cfg.cone_margin_m = cor_cfg.cone_margin_m;
  cbf_cfg.no_entry_margin_m = cor_cfg.no_entry_margin_m;
  cbf_cfg.horizon_steps = static_cast<int>(this->get_parameter("cbf_horizon_steps").as_int());
  cbf_cfg.step_dt_sec = this->get_parameter("cbf_step_dt_sec").as_double();
  cbf_cfg.cbf_gamma = this->get_parameter("cbf_gamma").as_double();
  cbf_shield_.set_config(cbf_cfg);

  // Sync FinishGuard
  FinishGuardConfig fg_cfg{};
  fg_cfg.odom_scale_factor = this->get_parameter("odom_scale_factor").as_double();
  fg_cfg.finish_min_dist_m = this->get_parameter("finish_min_dist_m").as_double();
  fg_cfg.finish_force_dist_m = this->get_parameter("finish_force_dist_m").as_double();
  fg_cfg.min_cumulative_abs_yaw_rad = this->get_parameter("finish_min_cum_yaw_rad").as_double();
  fg_cfg.pre_brake_dist_m = this->get_parameter("yellow_pre_brake_dist_m").as_double();
  fg_cfg.creep_speed_mps = this->get_parameter("yellow_creep_speed_mps").as_double();
  fg_cfg.yellow_final_ramp_dist_m = this->get_parameter("yellow_final_ramp_dist_m").as_double();
  fg_cfg.yellow_final_creep_speed_mps =
    this->get_parameter("yellow_final_creep_speed_mps").as_double();
  fg_cfg.brake_decel_mps2 = this->get_parameter("yellow_brake_decel_mps2").as_double();
  fg_cfg.voltage_decel_slope = this->get_parameter("voltage_decel_slope").as_double();
  fg_cfg.enable_brake_adaptation = this->get_parameter("enable_brake_adaptation").as_bool();
  fg_cfg.control_latency_s = this->get_parameter("yellow_control_latency_s").as_double();
  fg_cfg.bumper_margin_m = this->get_parameter("yellow_bumper_margin_m").as_double();
  fg_cfg.rearm_travel_dist_m = this->get_parameter("yellow_rearm_dist_m").as_double();
  finish_guard_.set_config(fg_cfg);
}

void LineFollowerNode::log_effective_parameters() const
{
  const auto & mcfg = motion_controller_.config();
  const auto & ccfg = corridor_planner_.config();
  const auto & fcfg = finish_guard_.config();
  const auto & ocfg = obstacle_memory_.config();
  const auto & bcfg = cbf_shield_.config();

  RCLCPP_INFO(
    this->get_logger(),
    "[DEC] param effective #0 mode=%s lane_seg_mode=%s yellow_detect_mode=%s "
    "enable_semantic_hole_fill=%d enable_obstacle_memory=%d allow_dual_digit_cover=%d "
    "target_speed=%.3f max_speed=%.2f min_speed=%.2f "
    "roi_top_ratio=%.3f roi_bottom_ratio=%.3f yellow_roi_bottom_ratio=%.3f "
    "yellow_stop_dur=%.2f yellow_brake_speed=%.3f yellow_brake_period_s=%.3f "
    "yellow_brake_pulse_s=%.3f yellow_brake_release_speed=%.3f yellow_brake_max_time_s=%.3f",
    track_mode_.c_str(), lane_seg_mode_.c_str(), yellow_detect_mode_.c_str(),
    enable_semantic_hole_fill_ ? 1 : 0,
    enable_obstacle_memory_ ? 1 : 0,
    allow_dual_digit_cover_ ? 1 : 0,
    target_speed_, max_speed_, min_speed_,
    roi_top_ratio_, roi_bottom_ratio_, yellow_roi_bottom_ratio_,
    yellow_stop_duration_,
    yellow_brake_speed_, yellow_brake_period_s_, yellow_brake_pulse_s_,
    yellow_brake_release_speed_, yellow_brake_max_time_s_);

  RCLCPP_INFO(
    this->get_logger(),
    "[DEC] param effective #finals corridor_kp_y=%.3f corridor_heading_gain=%.3f "
    "speed_gain_damping=%.3f max_yaw_rate=%.3f "
    "yellow_roi_bottom_ratio=%.3f rear_axle_to_bumper_m=%.3f "
    "obstacle_steer_horizon_m=%.3f obstacle_curve_min_horizon_m=%.3f obstacle_stagger_gap_m=%.3f "
    "yellow_bumper_margin_m=%.3f yellow_pre_brake_dist_m=%.3f yellow_creep_speed_mps=%.3f "
    "yellow_stop_min_time=%.3f yellow_brake_decel_mps2=%.3f yellow_control_latency_s=%.3f "
    "obstacle_speed_mps=%.3f no_entry_speed_mps=%.3f digit_approach_speed_mps=%.3f "
    "border_margin_m=%.3f cone_margin_m=%.3f no_entry_margin_m=%.3f "
    "digit_min_conf=%.3f max_digit_lateral_shift_m=%.3f "
    "finish_min_dist_m=%.3f finish_force_dist_m=%.3f odom_scale_factor=%.3f "
    "use_homography=%d yellow_require_odom_still=%d lost_line_safe_mode=%d "
    "enable_cv_red_fallback=%d red_cone_area_ratio_threshold=%.5f red_hsv=[%d..%d,%d..%d,s>=%d,v>=%d] "
    "enable_cbf_shield=%d cbf_wb=%.3f cbf_h=%d cbf_dt=%.3f cbf_gamma=%.2f "
    "yellow_final_ramp_dist_m=%.3f yellow_final_creep_speed_mps=%.3f "
    "voltage_decel_slope=%.3f enable_brake_adaptation=%d enable_risk_aware_decision=%d "
    "crc_pad_px=[no_entry=%.1f,cone=%.1f,yellow=%.1f]",
    mcfg.corridor_kp_y, mcfg.corridor_heading_gain,
    mcfg.speed_gain_damping, mcfg.max_yaw_rate,
    yellow_roi_bottom_ratio_, ocfg.rear_axle_to_bumper_m,
    ccfg.obstacle_steer_horizon_m, ccfg.obstacle_curve_min_horizon_m, ccfg.obstacle_stagger_gap_m,
    fcfg.bumper_margin_m, fcfg.pre_brake_dist_m, fcfg.creep_speed_mps,
    yellow_stop_min_time_, fcfg.brake_decel_mps2, fcfg.control_latency_s,
    ccfg.obstacle_speed_mps, ccfg.no_entry_speed_mps, ccfg.digit_approach_speed_mps,
    ccfg.border_margin_m, ccfg.cone_margin_m, ccfg.no_entry_margin_m,
    ccfg.digit_min_conf, ccfg.max_digit_lateral_shift_m,
    fcfg.finish_min_dist_m, fcfg.finish_force_dist_m, fcfg.odom_scale_factor,
    use_homography_ ? 1 : 0,
    yellow_require_odom_still_ ? 1 : 0,
    mcfg.lost_line_safe_mode ? 1 : 0,
    enable_cv_red_fallback_ ? 1 : 0,
    red_cone_area_ratio_threshold_,
    red_h_low_1_, red_h_high_1_, red_h_low_2_, red_h_high_2_, red_s_min_, red_v_min_,
    enable_cbf_shield_ ? 1 : 0,
    bcfg.wheelbase_m, bcfg.horizon_steps, bcfg.step_dt_sec, bcfg.cbf_gamma,
    fcfg.yellow_final_ramp_dist_m, fcfg.yellow_final_creep_speed_mps,
    fcfg.voltage_decel_slope,
    fcfg.enable_brake_adaptation ? 1 : 0,
    ccfg.enable_risk_aware_decision ? 1 : 0,
    crc_no_entry_pad_y_px_, crc_cone_pad_y_px_, crc_yellow_pad_y_px_);
}

bool LineFollowerNode::brake_pulse_on(double elapsed_s) const
{
  if (yellow_brake_period_s_ <= 1e-3 ||
    yellow_brake_pulse_s_ <= 0.0 ||
    yellow_brake_pulse_s_ >= yellow_brake_period_s_)
  {
    return true;
  }
  return std::fmod(elapsed_s, yellow_brake_period_s_) < yellow_brake_pulse_s_;
}

void LineFollowerNode::publish_segment_info(bool force)
{
  if (!segment_pub_) {
    return;
  }
  const auto now = this->now();
  if (!force && (now - last_segment_pub_time_).seconds() < 1.0) {
    return;
  }
  last_segment_pub_time_ = now;

  char buf[160];
  std::snprintf(
    buf, sizeof(buf), "mode=finals s=%.2f count=%d branch=%d final=%d",
    finish_guard_.total_arc_m(),
    track_state_machine_.state().yellow_line_detection_count,
    corridor_planner_.committed_branch(finish_guard_.total_arc_m()),
    track_state_machine_.state().permanent_stop ? 1 : 0);

  std_msgs::msg::String msg;
  msg.data = buf;
  segment_pub_->publish(msg);
}

void LineFollowerNode::setup_interfaces()
{
  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
  stop_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(stop_cmd_topic_, 10);
  state_pub_ = this->create_publisher<std_msgs::msg::String>(line_state_topic_, 10);
  segment_pub_ = this->create_publisher<std_msgs::msg::String>(segment_topic_, 10);
  segment_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(500),
    [this]() {publish_segment_info(false);});

  image_sub_ = image_transport::create_subscription(
    this,
    image_topic_,
    std::bind(&LineFollowerNode::image_callback, this, std::placeholders::_1),
    "raw");

  start_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    start_topic_, 10, std::bind(&LineFollowerNode::start_callback, this, std::placeholders::_1));

  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_, 10, std::bind(&LineFollowerNode::odom_callback, this, std::placeholders::_1));

  if (!ekf_odom_topic_.empty() && ekf_odom_topic_ != odom_topic_) {
    ekf_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      ekf_odom_topic_, 10,
      std::bind(&LineFollowerNode::ekf_odom_callback, this, std::placeholders::_1));
  }

  if (!detection_topic_.empty()) {
    detection_sub_ = this->create_subscription<ai_msgs::msg::PerceptionTargets>(
      detection_topic_, rclcpp::QoS(10).best_effort(),
      std::bind(&LineFollowerNode::detection_callback, this, std::placeholders::_1));
  }

  if (!voltage_topic_.empty()) {
    voltage_sub_ = this->create_subscription<std_msgs::msg::Float32>(
      voltage_topic_, 10,
      std::bind(&LineFollowerNode::voltage_callback, this, std::placeholders::_1));
  }

  if (publish_debug_image_) {
    debug_pub_ = image_transport::create_publisher(this, debug_image_topic_);
    debug_enabled_ = true;
  }
}

void LineFollowerNode::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  const auto now = this->now();
  const double now_sec = now.seconds();
  track_state_machine_.update_odom(msg->twist.twist.linear.x, now_sec);

  const double yaw = quat_to_yaw(
    msg->pose.pose.orientation.x,
    msg->pose.pose.orientation.y,
    msg->pose.pose.orientation.z,
    msg->pose.pose.orientation.w);

  odom_pose_.valid = true;
  odom_pose_.x = msg->pose.pose.position.x;
  odom_pose_.y = msg->pose.pose.position.y;
  odom_pose_.yaw = yaw;
  odom_pose_.linear_v = msg->twist.twist.linear.x;
  odom_pose_.angular_w = msg->twist.twist.angular.z;
  odom_pose_.stamp_sec = now_sec;

  double dyaw = 0.0;
  if (last_odom_yaw_valid_) {
    dyaw = wrap_angle_rad(yaw - last_odom_yaw_);
  }
  last_odom_yaw_ = yaw;
  last_odom_yaw_valid_ = true;

  if (odom_arc_valid_ && start_enabled_ && std::abs(msg->twist.twist.linear.x) >= 0.015) {
    const double dt = (now - last_odom_time_).seconds();
    if (dt > 0.0 && dt < 0.2) {
      // Update FinishGuard from /odom only when /odom_combined is not actively updating
      const bool ekf_active = ekf_pose_seen_ && ((now - last_ekf_odom_time_).seconds() < 0.5);
      if (!ekf_active) {
        const double ds_fwd = std::max(0.0, msg->twist.twist.linear.x) * dt;
        finish_guard_.update_motion(ds_fwd, dyaw);
      }
    }
  }
  last_odom_time_ = now;
  odom_arc_valid_ = true;
}

void LineFollowerNode::ekf_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  const auto now = this->now();
  const double now_sec = now.seconds();

  const double yaw = quat_to_yaw(
    msg->pose.pose.orientation.x,
    msg->pose.pose.orientation.y,
    msg->pose.pose.orientation.z,
    msg->pose.pose.orientation.w);

  const double prev_x = ekf_pose_.x;
  const double prev_y = ekf_pose_.y;
  const bool had_prev = ekf_pose_seen_;

  ekf_pose_.valid = true;
  ekf_pose_.x = msg->pose.pose.position.x;
  ekf_pose_.y = msg->pose.pose.position.y;
  ekf_pose_.yaw = yaw;
  ekf_pose_.linear_v = msg->twist.twist.linear.x;
  ekf_pose_.angular_w = msg->twist.twist.angular.z;
  ekf_pose_.stamp_sec = now_sec;

  double dyaw = 0.0;
  if (last_ekf_yaw_valid_) {
    dyaw = wrap_angle_rad(yaw - last_ekf_yaw_);
  }
  last_ekf_yaw_ = yaw;
  last_ekf_yaw_valid_ = true;

  // B7: Gate on start_enabled_ & non-trivial speed, and forward-project displacement onto heading
  if (had_prev && start_enabled_ && std::abs(ekf_pose_.linear_v) >= 0.015) {
    const double dt = (now - last_ekf_odom_time_).seconds();
    if (dt > 0.0 && dt < 0.25) {
      const double dx = ekf_pose_.x - prev_x;
      const double dy = ekf_pose_.y - prev_y;
      const double ds_pos = std::hypot(dx, dy);
      const double ds_fwd = (ds_pos > 1e-4 && ds_pos < 0.5) ?
        (dx * std::cos(yaw) + dy * std::sin(yaw)) :
        (std::max(0.0, ekf_pose_.linear_v) * dt);
      finish_guard_.update_motion(ds_fwd, dyaw);
    }
  }

  // Fallback standstill update if raw /odom hasn't been heard recently
  if (!odom_arc_valid_ || (now - last_odom_time_).seconds() > 0.35) {
    track_state_machine_.update_odom(msg->twist.twist.linear.x, now_sec);
  }

  ekf_pose_seen_ = true;
  last_ekf_odom_time_ = now;
}

void LineFollowerNode::detection_callback(
  const ai_msgs::msg::PerceptionTargets::SharedPtr msg)
{
  if (!msg) {
    return;
  }

  std::vector<RawYoloBox> boxes;
  for (const auto & target : msg->targets) {
    const int tid = static_cast<int>(target.track_id);
    for (const auto & roi : target.rois) {
      if (roi.rect.width == 0U || roi.rect.height == 0U) {
        continue;
      }
      int digit = 0;
      const SemanticObjectType type = classify_detection_label(roi.type, digit);
      if (type == SemanticObjectType::UNKNOWN) {
        continue;
      }
      RawYoloBox box{};
      box.type = type;
      box.digit = digit;
      box.label = roi.type;
      box.confidence = static_cast<double>(roi.confidence);
      box.bbox_img = cv::Rect(
        static_cast<int>(roi.rect.x_offset),
        static_cast<int>(roi.rect.y_offset),
        static_cast<int>(roi.rect.width),
        static_cast<int>(roi.rect.height));
      box.track_id = tid;
      boxes.push_back(box);
    }
  }

  latest_raw_boxes_ = std::move(boxes);
  latest_det_stamp_sec_ = this->now().seconds();
}

void LineFollowerNode::voltage_callback(const std_msgs::msg::Float32::SharedPtr msg)
{
  if (!msg) {
    return;
  }
  finish_guard_.update_battery_voltage(
    static_cast<double>(msg->data), this->now().seconds());
}

void LineFollowerNode::start_callback(const std_msgs::msg::Bool::SharedPtr msg)
{
  const bool new_state = msg->data;

  if (new_state == start_enabled_) {
    return;
  }

  const bool old_state = start_enabled_;
  const auto now = this->now();

  if (!old_state && new_state) {
    track_state_machine_.reset(now.seconds());
    curve_assist_ = CurveAssistState{};
    motion_state_ = MotionState{};
    yellow_brake_active_ = false;

    obstacle_memory_.reset();
    corridor_planner_.reset();
    finish_guard_.reset();
    last_red_src_state_ = RedDetectionSource::NONE;
    last_red_src_log_sec_ = -1e9;
    red_src_yolo_only_frames_ = 0;
    red_src_both_frames_ = 0;
    red_src_cv_only_frames_ = 0;
    last_ctrl_mode_tag_.clear();
    last_ctrl_branch_ = 0;
    last_ctrl_log_sec_ = -1e9;
    last_cbf_log_sec_ = -1e9;
    yellow_brake_measure_pending_ = false;
    yellow_brake_entry_speed_mps_ = 0.0;
    yellow_brake_start_arc_m_ = 0.0;

    RCLCPP_INFO(
      this->get_logger(),
      "New lap reset: yellow count=0, permanent stop cleared, finals guard/memory reset.");

    publish_segment_info(true);
  }

  start_enabled_ = new_state;

  if (!start_enabled_) {
    curve_assist_ = CurveAssistState{};
    yellow_brake_active_ = false;
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Start signal changed: %s -> %s",
    old_state ? "true" : "false",
    start_enabled_ ? "true" : "false");
}

void LineFollowerNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  cv::Mat frame_bgr;
  try {
    frame_bgr = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  // B4: Re-sync parameters only when updated dynamically via ROS2 parameter service
  if (params_dirty_.exchange(false, std::memory_order_acq_rel)) {
    sync_runtime_parameters();
  }

  const auto now = this->now();
  const double now_sec = now.seconds();

  // Collect fresh YOLO boxes for semantic hole-filling, yellow hybrid, and ground projection
  std::vector<cv::Rect> fill_white_rects;
  std::vector<cv::Rect> cv_red_exclude_rects;
  std::vector<cv::Rect> yolo_yellow_rects;
  std::vector<GroundDetection> ground_detections;
  std::vector<GroundDetection> yolo_red_dets;
  std::vector<DebugDetectionBox> debug_boxes;
  std::vector<DebugDetectionBox> yolo_red_dboxes;

  const bool dets_fresh = (now_sec - latest_det_stamp_sec_) <= detection_timeout_sec_;
  if (dets_fresh) {
    for (const auto & box : latest_raw_boxes_) {
      if (box.type == SemanticObjectType::BONUS_DIGIT ||
        box.type == SemanticObjectType::YELLOW_STOP_LINE)
      {
        cv_red_exclude_rects.push_back(box.bbox_img);
        if (enable_semantic_hole_fill_) {
          fill_white_rects.push_back(box.bbox_img);
        }
      }
      if (box.type == SemanticObjectType::YELLOW_STOP_LINE) {
        yolo_yellow_rects.push_back(box.bbox_img);
      }

      const double u_bot =
        static_cast<double>(box.bbox_img.x) + 0.5 * static_cast<double>(box.bbox_img.width);
      double crc_pad_y = 0.0;
      if (box.type == SemanticObjectType::RED_NO_ENTRY) {
        crc_pad_y = crc_no_entry_pad_y_px_;
      } else if (box.type == SemanticObjectType::RED_CONE) {
        crc_pad_y = crc_cone_pad_y_px_;
      } else if (box.type == SemanticObjectType::YELLOW_STOP_LINE) {
        crc_pad_y = crc_yellow_pad_y_px_;
      }
      const double max_v_px = static_cast<double>(std::max(0, frame_bgr.rows - 1));
      const double v_bot = std::clamp(
        static_cast<double>(box.bbox_img.y + box.bbox_img.height) + crc_pad_y,
        0.0,
        max_v_px);
      double d_m = 0.0;
      double y_m = 0.0;
      if (ground_projector_.pixel_to_ground(u_bot, v_bot, d_m, y_m)) {
        GroundDetection gd{};
        gd.type = box.type;
        gd.digit = box.digit;
        gd.confidence = box.confidence;
        gd.d_bumper_m = d_m;
        gd.y_lat_m = y_m;
        gd.track_id = box.track_id;
        gd.stamp_sec = latest_det_stamp_sec_;

        if (box.type == SemanticObjectType::RED_CONE) {
          gd.half_len_m = 0.0675;
          gd.half_wid_m = 0.0675;
        } else if (box.type == SemanticObjectType::RED_NO_ENTRY) {
          double dl = 0.0, yl = 0.0, dr = 0.0, yr = 0.0;
          double est_hw = 0.065;
          if (ground_projector_.pixel_to_ground(
              static_cast<double>(box.bbox_img.x), v_bot, dl, yl) &&
            ground_projector_.pixel_to_ground(
              static_cast<double>(box.bbox_img.x + box.bbox_img.width), v_bot, dr, yr))
          {
            est_hw = std::clamp(0.5 * std::abs(yl - yr), 0.065, 0.13);
          }
          gd.half_wid_m = est_hw;
          gd.half_len_m = (est_hw > 0.095) ? 0.065 : 0.12;
        } else {
          gd.half_len_m = 0.05;
          gd.half_wid_m = 0.05;
        }

        DebugDetectionBox dbox{};
        dbox.bbox_img = box.bbox_img;
        dbox.label = box.label;
        dbox.confidence = box.confidence;
        dbox.d_bumper_m = d_m;
        dbox.y_lat_m = y_m;
        dbox.is_no_entry = (box.type == SemanticObjectType::RED_NO_ENTRY);
        dbox.is_digit = (box.type == SemanticObjectType::BONUS_DIGIT);
        dbox.is_yellow = (box.type == SemanticObjectType::YELLOW_STOP_LINE);

        if (box.type == SemanticObjectType::RED_CONE ||
          box.type == SemanticObjectType::RED_NO_ENTRY)
        {
          yolo_red_dets.push_back(gd);
          yolo_red_dboxes.push_back(dbox);
        } else {
          ground_detections.push_back(gd);
          debug_boxes.push_back(dbox);
        }
      }
    }
  }

  const int active_branch = corridor_planner_.committed_branch(finish_guard_.total_arc_m());

  auto result_opt = process_image(frame_bgr, fill_white_rects, yolo_yellow_rects, active_branch);

  // ============================================================
  // Finals Red Obstacle OR-Fusion (YOLO || White-Lane CV) + Source Attribution Log
  // ============================================================
  {
    std::vector<GroundDetection> cv_red_dets;
    std::vector<cv::Rect> cv_red_bboxes;
    if (enable_cv_red_fallback_ && result_opt && result_opt->line_detected) {
      detect_cv_red_in_lane(
        frame_bgr, *result_opt, cv_red_exclude_rects, now_sec, cv_red_dets, cv_red_bboxes);
    }

    const RedFusionSummary red_fusion = fuse_red_obstacles_or(yolo_red_dets, cv_red_dets, 0.24);
    for (const auto & gd : red_fusion.fused_red_detections) {
      ground_detections.push_back(gd);
    }

    for (size_t i = 0; i < yolo_red_dboxes.size(); ++i) {
      DebugDetectionBox dbox = yolo_red_dboxes[i];
      const bool matched = (i < red_fusion.fused_sources.size() &&
        red_fusion.fused_sources[i] == RedDetectionSource::BOTH);
      dbox.label = std::string(matched ? "[Y+C]" : "[Y]") + dbox.label;
      debug_boxes.push_back(dbox);
    }
    for (size_t j = 0; j < cv_red_dets.size(); ++j) {
      if (j < red_fusion.cv_matched_flags.size() && red_fusion.cv_matched_flags[j]) {
        continue;
      }
      DebugDetectionBox dbox{};
      dbox.bbox_img = (j < cv_red_bboxes.size()) ? cv_red_bboxes[j] : cv::Rect{};
      dbox.label = (cv_red_dets[j].type == SemanticObjectType::RED_NO_ENTRY) ?
        "[CV]no_entry" : "[CV]cone";
      dbox.confidence = cv_red_dets[j].confidence;
      dbox.d_bumper_m = cv_red_dets[j].d_bumper_m;
      dbox.y_lat_m = cv_red_dets[j].y_lat_m;
      dbox.is_no_entry = (cv_red_dets[j].type == SemanticObjectType::RED_NO_ENTRY);
      debug_boxes.push_back(dbox);
    }

    if (start_enabled_) {
      if (red_fusion.overall_source == RedDetectionSource::YOLO_ONLY) {
        red_src_yolo_only_frames_++;
      } else if (red_fusion.overall_source == RedDetectionSource::BOTH) {
        red_src_both_frames_++;
      } else if (red_fusion.overall_source == RedDetectionSource::CV_ONLY) {
        red_src_cv_only_frames_++;
      }
    }

    const bool src_changed = (red_fusion.overall_source != last_red_src_state_);
    if (red_fusion.overall_source != RedDetectionSource::NONE) {
      if (src_changed || (now_sec - last_red_src_log_sec_) >= 0.50) {
        RCLCPP_INFO(
          this->get_logger(),
          "[DEC] red_src detect src=%s yolo=%d cv=%d both=%d yolo_only=%d cv_only=%d "
          "d=%.2f y=%+.2f type=%s totals=Y:%d/B:%d/C:%d",
          red_detection_source_to_string(red_fusion.overall_source),
          red_fusion.yolo_red_count, red_fusion.cv_red_count,
          red_fusion.matched_both_count, red_fusion.yolo_only_count, red_fusion.cv_only_count,
          red_fusion.nearest_d_bumper_m, red_fusion.nearest_y_lat_m,
          semantic_object_type_to_string(red_fusion.nearest_type),
          red_src_yolo_only_frames_, red_src_both_frames_, red_src_cv_only_frames_);
        last_red_src_log_sec_ = now_sec;
      }
    } else if (src_changed && last_red_src_state_ != RedDetectionSource::NONE) {
      RCLCPP_INFO(
        this->get_logger(),
        "[DEC] red_src clear prev=%s totals=Y:%d/B:%d/C:%d",
        red_detection_source_to_string(last_red_src_state_),
        red_src_yolo_only_frames_, red_src_both_frames_, red_src_cv_only_frames_);
      last_red_src_log_sec_ = now_sec;
    }
    last_red_src_state_ = red_fusion.overall_source;
  }

  FinalsDebugOverlay finals_overlay{};
  finals_overlay.enabled = true;
  finals_overlay.lane_seg_mode = lane_seg_mode_;
  finals_overlay.detections = debug_boxes;

  if (!start_enabled_) {
    curve_assist_ = CurveAssistState{};

    if (!waiting_log_initialized_) {
      waiting_log_initialized_ = true;
      last_waiting_log_time_ = now;
    } else if ((now - last_waiting_log_time_).seconds() > 2.0) {
      RCLCPP_INFO(this->get_logger(), "Line follower is not started, waiting for start signal...");
      last_waiting_log_time_ = now;
    }

    if (result_opt && debug_enabled_) {
      publish_debug_image(msg, frame_bgr, *result_opt, finals_overlay);
    }

    publish_cmd_vel(0.0, 0.0, true);
    return;
  }

  double linear_speed = 0.0;
  double yaw_rate = 0.0;
  TrackState current_state = TrackState::LOST;
  bool publish_stop_cmd = false;

  // ============================================================
  // Active pose & ObstacleMemory update
  // ============================================================
  const bool ekf_fresh = ekf_pose_seen_ && ((now - last_ekf_odom_time_).seconds() < 0.5);
  const VehiclePose2D & active_pose = ekf_fresh ? ekf_pose_ : odom_pose_;

  std::vector<LocalObstacle> local_obstacles;
  if (enable_obstacle_memory_) {
    obstacle_memory_.update(ground_detections, active_pose, now_sec);
    local_obstacles = obstacle_memory_.get_local_obstacles(active_pose, now_sec);
  } else {
    for (const auto & gd : ground_detections) {
      if (gd.type == SemanticObjectType::RED_CONE ||
        gd.type == SemanticObjectType::RED_NO_ENTRY)
      {
        LocalObstacle loc{};
        loc.type = gd.type;
        loc.d_bumper_m = gd.d_bumper_m;
        loc.d_min_m = gd.d_bumper_m - gd.half_len_m;
        loc.d_max_m = gd.d_bumper_m + gd.half_len_m;
        loc.y_lat_m = gd.y_lat_m;
        loc.y_min_m = gd.y_lat_m - gd.half_wid_m;
        loc.y_max_m = gd.y_lat_m + gd.half_wid_m;
        loc.confidence = gd.confidence;
        local_obstacles.push_back(loc);
      }
    }
  }

  // ============================================================
  // Two-stage zero-touch yellow line stop & FinishGuard gating
  // ============================================================
  const bool raw_yellow_detected = result_opt && result_opt->yellow_line_detected;
  const bool line_detected = result_opt && result_opt->line_detected;
  const TrackState prev_state = track_state_machine_.state().last_state;

  YellowStopAdvice yellow_advice{};
  bool sm_yellow_trigger = false;
  bool external_is_finish = false;
  {
    bool has_metric_dist = false;
    double yellow_d_bumper_m = 0.0;
    double yellow_y_lat_m = 0.0;

    if (raw_yellow_detected && result_opt && result_opt->yellow_is_geometric) {
      const double max_v_px = static_cast<double>(std::max(0, frame_bgr.rows - 1));
      const double v_yellow_bot = std::clamp(
        result_opt->yellow_bottom_y_full + crc_yellow_pad_y_px_,
        0.0,
        max_v_px);
      has_metric_dist = ground_projector_.pixel_to_ground(
        result_opt->yellow_center_x_full,
        v_yellow_bot,
        yellow_d_bumper_m,
        yellow_y_lat_m);
    }

    const double cur_v = track_state_machine_.state().odom_seen ?
      track_state_machine_.state().odom_speed : motion_state_.last_linear;
    const bool dead_end = result_opt && result_opt->yellow_has_dead_end_ahead;

    yellow_advice = finish_guard_.evaluate_yellow_stop(
      raw_yellow_detected,
      has_metric_dist,
      yellow_d_bumper_m,
      cur_v,
      dead_end,
      now_sec);

    finals_overlay.yellow_dist_valid = has_metric_dist;
    finals_overlay.yellow_d_bumper_m = yellow_d_bumper_m;
    finals_overlay.yellow_is_finish = yellow_advice.is_finish_line;
    finals_overlay.yellow_dead_end_ahead = dead_end;

    const bool already_stopping =
      track_state_machine_.state().yellow_line_stop_active ||
      track_state_machine_.state().permanent_stop;
    sm_yellow_trigger = already_stopping ?
      raw_yellow_detected :
      (finish_guard_.is_yellow_rearmed() && yellow_advice.trigger_stop);
    external_is_finish = yellow_advice.is_finish_line;
  }

  const TrackStepEvents step_events =
    track_state_machine_.step(sm_yellow_trigger, line_detected, now_sec, external_is_finish);
  const auto & sm_state = track_state_machine_.state();

  if (step_events.yellow_stop_begin) {
    finish_guard_.mark_yellow_stop_triggered();
    corridor_planner_.reset_fork_votes();

    yellow_brake_entry_speed_mps_ = sm_state.odom_seen ?
      sm_state.odom_speed : std::abs(motion_state_.last_linear);
    yellow_brake_start_arc_m_ = finish_guard_.total_arc_m();
    yellow_brake_measure_pending_ = true;

    yellow_brake_active_ = yellow_brake_speed_ > 0.0;
    yellow_brake_start_ = now;
    last_brake_cmd_time_ = now;
    if (yellow_brake_active_) {
      RCLCPP_INFO(
        this->get_logger(),
        "[DEC] yellow brake_begin v=%.2f period=%.2f pulse=%.2f rel=%.2f max=%.2f",
        yellow_brake_speed_, yellow_brake_period_s_, yellow_brake_pulse_s_,
        yellow_brake_release_speed_, yellow_brake_max_time_s_);
    }
  } else if (yellow_brake_measure_pending_) {
    if (step_events.odom_still_ok) {
      yellow_brake_measure_pending_ = false;
      const double stop_travel_m =
        finish_guard_.total_arc_m() - yellow_brake_start_arc_m_;
      const double prev_scale = finish_guard_.friction_scale();
      if (finish_guard_.update_measured_yellow_stop(
          yellow_brake_entry_speed_mps_, stop_travel_m, now_sec))
      {
        RCLCPP_INFO(
          this->get_logger(),
          "[DEC] brake_adapt v_in=%.3f travel=%.4f scale=%.3f->%.3f a_eff=%.3f vbat=%.2f",
          yellow_brake_entry_speed_mps_,
          stop_travel_m,
          prev_scale,
          finish_guard_.friction_scale(),
          finish_guard_.effective_brake_decel_mps2(now_sec),
          finish_guard_.active_battery_voltage(now_sec));
      }
    } else if (!step_events.yellow_hold_stop) {
      yellow_brake_measure_pending_ = false;
    }
  }

  if (step_events.yellow_stop_end) {
    yellow_brake_active_ = false;
    const int chosen = corridor_planner_.finalize_fork_decision(finish_guard_.total_arc_m());
    const auto summary = corridor_planner_.fork_summary();
    if (chosen != 0) {
      finish_guard_.mark_fork_passed();
      RCLCPP_INFO(
        this->get_logger(),
        "[DEC] fork_decision branch=%d dL=%d(%d) dR=%d(%d) noEntry=(%d,%d) s=%.2f",
        chosen, summary.left_digit, summary.left_votes,
        summary.right_digit, summary.right_votes,
        summary.left_has_no_entry ? 1 : 0, summary.right_has_no_entry ? 1 : 0,
        finish_guard_.total_arc_m());
    }
  }

  // B6: Immediately publish final=1 on yellow_final_stop for deterministic panel timer stop
  if (step_events.yellow_final_stop) {
    publish_segment_info(true);
  } else {
    publish_segment_info(false);
  }

  if (step_events.yellow_stop_begin) {
    RCLCPP_INFO(
      this->get_logger(),
      "[DEC] yellow stop_begin count=%d s=%.2f finish=%d",
      sm_state.yellow_line_detection_count,
      finish_guard_.total_arc_m(),
      external_is_finish ? 1 : 0);
  }
  if (step_events.yellow_final_stop) {
    RCLCPP_INFO(
      this->get_logger(),
      "[DEC] yellow final_stop count=%d s=%.2f",
      sm_state.yellow_line_detection_count,
      finish_guard_.total_arc_m());
  }
  if (step_events.yellow_temp_stop) {
    RCLCPP_INFO(
      this->get_logger(),
      "[DEC] yellow temp_stop count=%d dur=%.2f",
      sm_state.yellow_line_detection_count,
      yellow_stop_duration_);
  }
  if (step_events.yellow_cleared) {
    RCLCPP_INFO(
      this->get_logger(),
      "Yellow line cleared. Ready for next yellow line.");
  }
  if (step_events.yellow_stop_end) {
    RCLCPP_INFO(
      this->get_logger(),
      "[DEC] yellow stop_end dur=%.2f odom_v=%.3f fresh=%d still=%d",
      step_events.since_yellow_detection,
      sm_state.odom_speed,
      step_events.odom_fresh ? 1 : 0,
      step_events.odom_still_ok ? 1 : 0);
  }

  // While stationary at a temporary yellow line, accumulate fork static votes
  if (step_events.yellow_hold_stop && !sm_state.permanent_stop) {
    corridor_planner_.update_fork_static_votes(ground_detections, now_sec);
  }

  // ============================================================
  // Control: Yellow Stop > Finals Corridor > Lost-Line Safety
  // ============================================================
  if (step_events.yellow_hold_stop) {
    current_state = TrackState::YELLOW_LINE_DETECTED;
    linear_speed = 0.0;
    yaw_rate = 0.0;
    publish_stop_cmd = true;

    if (yellow_brake_active_) {
      const double brake_elapsed = (now - yellow_brake_start_).seconds();
      const bool slow_enough =
        yellow_brake_release_speed_ > 0.0 &&
        brake_elapsed >= 0.1 &&
        sm_state.odom_speed <= yellow_brake_release_speed_;
      const bool timeout =
        yellow_brake_max_time_s_ > 0.0 && brake_elapsed >= yellow_brake_max_time_s_;

      if (slow_enough || timeout) {
        yellow_brake_active_ = false;
        RCLCPP_INFO(
          this->get_logger(),
          "[DEC] yellow brake_end dur=%.2f v=%.3f why=%s",
          brake_elapsed, sm_state.odom_speed, timeout ? "timeout" : "slow");
      } else {
        publish_stop_cmd = false;
        linear_speed = brake_pulse_on(brake_elapsed) ? -yellow_brake_speed_ : 0.0;
      }
    }
  } else if (line_detected) {
    current_state = TrackState::TRACKING;

    const auto & obs = result_opt->observation;
    if (obs.has_fork_split) {
      finish_guard_.mark_fork_passed();
    }

    // Project near-band lane edges into vehicle bumper frame (+y left)
    const double roi_h = static_cast<double>(result_opt->roi_bottom - result_opt->roi_top);
    const double v_near =
      static_cast<double>(result_opt->roi_top) +
      0.5 * (near_band_top_ratio_ + near_band_bottom_ratio_) * roi_h;
    const double half_w_def = 0.5 * corridor_planner_.config().default_lane_width_m;

    double d_tmp = 0.0;
    double y_center_m = 0.0;
    const bool center_ok = ground_projector_.pixel_to_ground(
      static_cast<double>(result_opt->centroid.x), v_near, d_tmp, y_center_m);

    double y_lane_left_m = center_ok ? (y_center_m + half_w_def) : half_w_def;
    double y_lane_right_m = center_ok ? (y_center_m - half_w_def) : -half_w_def;

    double d_l = 0.0, y_l = 0.0;
    double d_r = 0.0, y_r = 0.0;
    const bool left_proj_ok = obs.near_left_valid &&
      ground_projector_.pixel_to_ground(obs.near_left_edge_px, v_near, d_l, y_l);
    const bool right_proj_ok = obs.near_right_valid &&
      ground_projector_.pixel_to_ground(obs.near_right_edge_px, v_near, d_r, y_r);

    // B8: Only recover measured [y_l, y_r] on clean unsplit lanes of physical width [0.30, 0.52]m
    // (and during active fork steering, only when near-band center still agrees within 4cm);
    // otherwise keep anchored to y_center_m +/- half_w_def so far_band fork steering is preserved.
    // When exactly one near-band edge is clipped off-screen (non-fork), enforce the visible edge
    // strictly inward if it is more restrictive than the far-blended estimate so inner corners
    // cannot mask an imminent lane departure while preserving full far-band curve anticipation.
    if (!obs.has_fork_split && left_proj_ok && right_proj_ok &&
      (y_l - y_r) >= 0.30 && (y_l - y_r) <= 0.52 &&
      (active_branch == 0 ||
        (center_ok && std::abs(y_center_m - 0.5 * (y_l + y_r)) <= 0.04)))
    {
      y_lane_left_m = y_l;
      y_lane_right_m = y_r;
    } else if (!obs.has_fork_split && active_branch == 0 && obs.near_band.valid) {
      const double def_w = corridor_planner_.config().default_lane_width_m;
      if (obs.near_band.left_clipped && !obs.near_band.right_clipped &&
        right_proj_ok && y_r >= -0.45 && y_r <= 0.25 && y_r > y_lane_right_m)
      {
        y_lane_right_m = y_r;
        y_lane_left_m = y_r + def_w;
      } else if (obs.near_band.right_clipped && !obs.near_band.left_clipped &&
        left_proj_ok && y_l >= -0.25 && y_l <= 0.45 && y_l < y_lane_left_m)
      {
        y_lane_left_m = y_l;
        y_lane_right_m = y_l - def_w;
      }
    }

    const double eff_curve_lvl = std::max(obs.curvature_proxy, curve_assist_.curve_level);
    const CorridorPlan plan = corridor_planner_.plan(
      y_lane_left_m,
      y_lane_right_m,
      0.0,
      local_obstacles,
      ground_detections,
      false,
      finish_guard_.total_arc_m(),
      now_sec,
      eff_curve_lvl);

    double eff_speed_cap = plan.speed_cap_mps;
    if (finish_guard_.is_yellow_rearmed() &&
      yellow_advice.pre_brake &&
      !yellow_advice.trigger_stop)
    {
      eff_speed_cap = std::min(eff_speed_cap, yellow_advice.speed_cap_mps);
    }

    const MotionCommand cmd = motion_controller_.compute_corridor_command(
      plan.target_y_m,
      0.0,
      obs.heading_proxy,
      obs.curvature_proxy,
      obs.confidence,
      eff_speed_cap,
      curve_assist_,
      motion_state_);

    linear_speed = cmd.linear_speed;
    yaw_rate = cmd.yaw_rate;
    publish_stop_cmd = cmd.publish_stop;

    if (enable_cbf_shield_) {
      const CbfShieldResult cbf = cbf_shield_.filter(
        linear_speed,
        yaw_rate,
        y_lane_left_m,
        y_lane_right_m,
        local_obstacles);
      finals_overlay.cbf_enabled = true;
      finals_overlay.cbf_intervened = cbf.intervened;
      finals_overlay.cbf_min_clearance_m = cbf.min_clearance_m;
      finals_overlay.cbf_limiting_source = cbf.limiting_source;
      if (cbf.intervened) {
        if ((now_sec - last_cbf_log_sec_) >= 0.25) {
          RCLCPP_INFO(
            this->get_logger(),
            "[DEC] cbf intervene src=%s v=%.2f->%.2f yaw=%+.2f->%+.2f clr=%.3f",
            cbf.limiting_source.c_str(),
            linear_speed, cbf.linear_speed,
            yaw_rate, cbf.yaw_rate,
            cbf.min_clearance_m);
          last_cbf_log_sec_ = now_sec;
        }
        linear_speed = cbf.linear_speed;
        yaw_rate = cbf.yaw_rate;
        if (linear_speed <= 1e-6 && std::abs(yaw_rate) <= 1e-6) {
          publish_stop_cmd = true;
        }
      }
    }

    const bool ctrl_changed =
      (plan.active_mode_tag != last_ctrl_mode_tag_) ||
      (plan.preferred_branch != last_ctrl_branch_);
    if (ctrl_changed || (now_sec - last_ctrl_log_sec_) >= 0.20) {
      RCLCPP_INFO(
        this->get_logger(),
        "[DEC] ctrl mode=%s branch=%d y_ref=%+.3f lane=[%+.2f,%+.2f] "
        "hdg=%+.2f rev=%d v=%.2f yaw=%+.2f obs=%d/%d s=%.2f",
        plan.active_mode_tag.c_str(),
        plan.preferred_branch,
        plan.target_y_m,
        y_lane_right_m,
        y_lane_left_m,
        obs.heading_proxy,
        curve_assist_.reverse_curve_active ? 1 : 0,
        linear_speed,
        yaw_rate,
        plan.near_obstacle_count,
        plan.far_obstacle_count,
        finish_guard_.total_arc_m());
      last_ctrl_mode_tag_ = plan.active_mode_tag;
      last_ctrl_branch_ = plan.preferred_branch;
      last_ctrl_log_sec_ = now_sec;
    }

    finals_overlay.active_mode_tag = plan.active_mode_tag;
    finals_overlay.corridor_valid = true;
    finals_overlay.y_right_safe_m = plan.safe_y_min_m;
    finals_overlay.y_left_safe_m = plan.safe_y_max_m;
    finals_overlay.y_ref_m = plan.target_y_m;
    finals_overlay.speed_cap_mps = eff_speed_cap;
    finals_overlay.preferred_branch = plan.preferred_branch;
    finals_overlay.corridor_row_y = static_cast<int>(std::round(v_near));

    const double d_ref = center_ok ? std::max(0.15, d_tmp) : 0.35;
    double u_l = 0.0, v_l = 0.0, u_r = 0.0, v_r = 0.0, u_ref = 0.0, v_ref = 0.0;
    if (ground_projector_.ground_to_pixel(d_ref, plan.safe_y_max_m, u_l, v_l)) {
      finals_overlay.corridor_left_px = static_cast<int>(std::round(u_l));
    }
    if (ground_projector_.ground_to_pixel(d_ref, plan.safe_y_min_m, u_r, v_r)) {
      finals_overlay.corridor_right_px = static_cast<int>(std::round(u_r));
    }
    if (ground_projector_.ground_to_pixel(d_ref, plan.target_y_m, u_ref, v_ref)) {
      finals_overlay.corridor_ref_px = static_cast<int>(std::round(u_ref));
    }
  } else {
    current_state = step_events.base_state;
    const MotionCommand cmd =
      motion_controller_.compute_lost_line_command(step_events.since_line_detection, motion_state_);
    linear_speed = cmd.linear_speed;
    yaw_rate = cmd.yaw_rate;
    publish_stop_cmd = cmd.publish_stop;
  }

  const bool yellow_stop_now = step_events.yellow_stop_now;

  // Final safety lock: yellow stop holds highest priority while preserving reverse-brake pulse
  if (step_events.yellow_hold_stop) {
    current_state = TrackState::YELLOW_LINE_DETECTED;
    yaw_rate = 0.0;
    if (yellow_brake_active_) {
      const double brake_elapsed = (now - yellow_brake_start_).seconds();
      linear_speed = brake_pulse_on(brake_elapsed) ? -yellow_brake_speed_ : 0.0;
      publish_stop_cmd = false;
    } else {
      linear_speed = 0.0;
      publish_stop_cmd = true;
    }
  }

  MotionController::record_motion(yellow_stop_now, linear_speed, yaw_rate, motion_state_);

  {
    const auto fsum = corridor_planner_.fork_summary();
    char fbuf[96];
    std::snprintf(
      fbuf, sizeof(fbuf), "L:%d(%d) R:%d(%d) br=%d",
      fsum.left_digit, fsum.left_votes, fsum.right_digit, fsum.right_votes,
      finals_overlay.preferred_branch);
    finals_overlay.fork_vote_status = fbuf;

    char gbuf[96];
    std::snprintf(
      gbuf, sizeof(gbuf), "s=%.2fm arm=%d obs=%zu",
      finish_guard_.total_arc_m(),
      finish_guard_.is_finish_armed() ? 1 : 0,
      local_obstacles.size());
    finals_overlay.finish_guard_status = gbuf;
  }

  if (result_opt && debug_enabled_) {
    publish_debug_image(msg, frame_bgr, *result_opt, finals_overlay);
  }

  publish_cmd_vel(linear_speed, yaw_rate, publish_stop_cmd);

  if (current_state != prev_state) {
    RCLCPP_INFO(
      this->get_logger(),
      "[DEC] state change from=%s to=%s",
      state_to_string(prev_state).c_str(),
      state_to_string(current_state).c_str());
  }

  if (current_state != prev_state ||
    current_state == TrackState::LOST ||
    current_state == TrackState::YELLOW_LINE_DETECTED)
  {
    publish_state(current_state);
  }

  track_state_machine_.commit_state(current_state);
}

std::optional<ProcessingResult> LineFollowerNode::process_image(
  const cv::Mat & frame_bgr,
  const std::vector<cv::Rect> & fill_white_rects,
  const std::vector<cv::Rect> & yolo_yellow_rects,
  int preferred_branch)
{
  const auto segmented_opt =
    image_processor_.segment(frame_bgr, fill_white_rects, yolo_yellow_rects);
  if (!segmented_opt) {
    return std::nullopt;
  }

  return lane_geometry_.evaluate(*segmented_opt, curve_assist_, preferred_branch);
}

void LineFollowerNode::detect_cv_red_in_lane(
  const cv::Mat & frame_bgr,
  const ProcessingResult & result,
  const std::vector<cv::Rect> & exclude_rects,
  double now_sec,
  std::vector<GroundDetection> & out_detections,
  std::vector<cv::Rect> & out_bboxes_img) const
{
  out_detections.clear();
  out_bboxes_img.clear();

  if (frame_bgr.empty() || result.binary_mask.empty()) {
    return;
  }

  const int roi_top = std::clamp(result.roi_top, 0, frame_bgr.rows - 1);
  const int roi_bottom = std::clamp(result.roi_bottom, roi_top + 1, frame_bgr.rows);
  const int roi_h = roi_bottom - roi_top;
  const int roi_w = frame_bgr.cols;
  if (roi_h <= 4 || roi_w <= 4 ||
    result.binary_mask.rows != roi_h || result.binary_mask.cols != roi_w)
  {
    return;
  }

  // 1) Row-span fill of white lane interior (+4px border pad)
  cv::Mat lane_span_mask = cv::Mat::zeros(roi_h, roi_w, CV_8UC1);
  for (int r = 0; r < roi_h; ++r) {
    const uint8_t * row_ptr = result.binary_mask.ptr<uint8_t>(r);
    int x_left = -1;
    int x_right = -1;
    for (int c = 0; c < roi_w; ++c) {
      if (row_ptr[c] > 0) {
        x_left = c;
        break;
      }
    }
    if (x_left < 0) {
      continue;
    }
    for (int c = roi_w - 1; c >= x_left; --c) {
      if (row_ptr[c] > 0) {
        x_right = c;
        break;
      }
    }
    if (x_right - x_left >= 16) {
      const int p0 = std::max(0, x_left - 4);
      const int p1 = std::min(roi_w - 1, x_right + 4);
      uint8_t * dst_ptr = lane_span_mask.ptr<uint8_t>(r);
      std::fill(dst_ptr + p0, dst_ptr + p1 + 1, static_cast<uint8_t>(255));
    }
  }

  // 2) HSV red threshold inside ROI gated by lane_span_mask
  const cv::Mat roi_bgr = frame_bgr(cv::Rect(0, roi_top, roi_w, roi_h));
  cv::Mat hsv;
  cv::cvtColor(roi_bgr, hsv, cv::COLOR_BGR2HSV);
  cv::Mat mask1, mask2, red_mask;
  cv::inRange(hsv, cv::Scalar(red_h_low_1_, red_s_min_, red_v_min_),
    cv::Scalar(red_h_high_1_, 255, 255), mask1);
  cv::inRange(hsv, cv::Scalar(red_h_low_2_, red_s_min_, red_v_min_),
    cv::Scalar(red_h_high_2_, 255, 255), mask2);
  cv::bitwise_or(mask1, mask2, red_mask);
  cv::bitwise_and(red_mask, lane_span_mask, red_mask);

  // 3) Mask out YOLO digit/yellow boxes
  for (const auto & ex : exclude_rects) {
    const int x1 = std::clamp(ex.x - 4, 0, roi_w);
    const int x2 = std::clamp(ex.x + ex.width + 4, 0, roi_w);
    const int y1 = std::clamp(ex.y - roi_top - 4, 0, roi_h);
    const int y2 = std::clamp(ex.y + ex.height - roi_top + 4, 0, roi_h);
    if (x2 > x1 && y2 > y1) {
      red_mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).setTo(0);
    }
  }

  // 4) Open + Close morphology
  const cv::Mat k_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  const cv::Mat k_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 9));
  cv::Mat raw_red_clean;
  cv::morphologyEx(red_mask, raw_red_clean, cv::MORPH_OPEN, k_open);
  cv::Mat grouped_red;
  cv::morphologyEx(raw_red_clean, grouped_red, cv::MORPH_CLOSE, k_close);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(grouped_red, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  const double roi_area = static_cast<double>(roi_w) * static_cast<double>(roi_h);
  const double min_area_px = std::max(70.0, red_cone_area_ratio_threshold_ * roi_area);

  // Merge fragmented contours from the same hollow red no-entry rectangle before projection
  std::vector<cv::Rect> merged_rois;
  for (const auto & cnt : contours) {
    const double area = cv::contourArea(cnt);
    if (area < min_area_px) {
      continue;
    }
    const cv::Rect r_roi = cv::boundingRect(cnt);
    if (r_roi.width < 6 || r_roi.height < 6) {
      continue;
    }
    // Skip tiny clipped slivers grazing the left/right image border (already tracked in memory)
    const bool touches_lr_border = (r_roi.x <= 1) || (r_roi.x + r_roi.width >= roi_w - 2);
    if (touches_lr_border && r_roi.width < 16 && r_roi.height < 16) {
      continue;
    }
    merged_rois.push_back(r_roi);
  }

  bool did_merge = true;
  const int max_merged_w = static_cast<int>(0.55 * static_cast<double>(roi_w));
  while (did_merge && merged_rois.size() >= 2) {
    did_merge = false;
    for (size_t i = 0; i < merged_rois.size() && !did_merge; ++i) {
      for (size_t j = i + 1; j < merged_rois.size(); ++j) {
        const cv::Rect & a = merged_rois[i];
        const cv::Rect & b = merged_rois[j];
        const int dx = std::max(0, std::max(a.x - (b.x + b.width), b.x - (a.x + a.width)));
        const int dy = std::max(0, std::max(a.y - (b.y + b.height), b.y - (a.y + a.height)));
        if (dx <= 24 && dy <= 16) {
          const cv::Rect u = a | b;
          if (u.width <= max_merged_w) {
            merged_rois[i] = u;
            merged_rois.erase(merged_rois.begin() + static_cast<std::ptrdiff_t>(j));
            did_merge = true;
            break;
          }
        }
      }
    }
  }

  for (const auto & r_roi : merged_rois) {
    // Reject off-track red objects (e.g. cones on outer blue mat) whose base center lies outside lane span
    const int u_base_roi = std::clamp(r_roi.x + r_roi.width / 2, 0, roi_w - 1);
    const int v_base_roi = std::clamp(r_roi.y + r_roi.height - 1, 0, roi_h - 1);
    if (lane_span_mask.at<uint8_t>(v_base_roi, u_base_roi) == 0) {
      continue;
    }

    const cv::Rect r_img(r_roi.x, r_roi.y + roi_top, r_roi.width, r_roi.height);
    const double u_bot = static_cast<double>(r_img.x) + 0.5 * static_cast<double>(r_img.width);
    const double max_v_px = static_cast<double>(std::max(0, frame_bgr.rows - 1));
    const double v_raw = std::clamp(
      static_cast<double>(r_img.y + r_img.height), 0.0, max_v_px);
    double d_m = 0.0;
    double y_m = 0.0;
    if (!ground_projector_.pixel_to_ground(u_bot, v_raw, d_m, y_m)) {
      continue;
    }

    double dl = 0.0, yl = 0.0, dr = 0.0, yr = 0.0;
    double est_width_m = 0.135;
    if (ground_projector_.pixel_to_ground(static_cast<double>(r_img.x), v_raw, dl, yl) &&
      ground_projector_.pixel_to_ground(
        static_cast<double>(r_img.x + r_img.width), v_raw, dr, yr))
    {
      est_width_m = std::clamp(std::abs(yl - yr), 0.10, 0.26);
    }

    const int raw_red_px = cv::countNonZero(raw_red_clean(r_roi));
    const double fill_ratio =
      static_cast<double>(raw_red_px) / std::max(1.0, static_cast<double>(r_roi.area()));
    const bool looks_no_entry = (fill_ratio < 0.42 && r_roi.width >= 18) || (est_width_m > 0.155);

    const double crc_pad_y = looks_no_entry ? crc_no_entry_pad_y_px_ : crc_cone_pad_y_px_;
    const double v_bot = std::clamp(
      static_cast<double>(r_img.y + r_img.height) + crc_pad_y, 0.0, max_v_px);
    if (std::abs(v_bot - v_raw) > 1e-6) {
      if (!ground_projector_.pixel_to_ground(u_bot, v_bot, d_m, y_m)) {
        continue;
      }
      if (ground_projector_.pixel_to_ground(static_cast<double>(r_img.x), v_bot, dl, yl) &&
        ground_projector_.pixel_to_ground(
          static_cast<double>(r_img.x + r_img.width), v_bot, dr, yr))
      {
        est_width_m = std::clamp(std::abs(yl - yr), 0.10, 0.26);
      }
    }
    if (d_m < 0.02 || d_m > 1.45 || std::abs(y_m) > 0.35) {
      continue;
    }

    GroundDetection gd{};
    gd.type = looks_no_entry ? SemanticObjectType::RED_NO_ENTRY : SemanticObjectType::RED_CONE;
    gd.confidence = 0.60;
    gd.d_bumper_m = d_m;
    gd.y_lat_m = y_m;
    gd.stamp_sec = now_sec;
    if (looks_no_entry) {
      gd.half_wid_m = std::clamp(0.5 * est_width_m, 0.065, 0.13);
      gd.half_len_m = (gd.half_wid_m > 0.095) ? 0.065 : 0.12;
    } else {
      gd.half_wid_m = std::clamp(0.5 * est_width_m, 0.0675, 0.10);
      gd.half_len_m = 0.0675;
    }

    out_detections.push_back(gd);
    out_bboxes_img.push_back(r_img);
  }
}

std::string LineFollowerNode::state_to_string(TrackState state) const
{
  return track_state_to_string(state);
}

void LineFollowerNode::publish_cmd_vel(double linear_speed, double yaw_rate, bool publish_stop)
{
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = linear_speed;
  cmd.angular.z = yaw_rate;
  cmd_vel_pub_->publish(cmd);

  if (stop_pub_ && publish_stop && (linear_speed == 0.0 && yaw_rate == 0.0)) {
    stop_pub_->publish(cmd);
  }
}

void LineFollowerNode::publish_state(TrackState state)
{
  std_msgs::msg::String msg;
  msg.data = state_to_string(state);
  state_pub_->publish(msg);
}

void LineFollowerNode::publish_debug_image(
  const sensor_msgs::msg::Image::ConstSharedPtr & source,
  const cv::Mat & frame_bgr,
  const ProcessingResult & result,
  const FinalsDebugOverlay & finals_overlay) const
{
  if (!debug_enabled_ || result.binary_mask.empty()) {
    return;
  }

  const cv::Mat overlay = debug_renderer_.render(
    frame_bgr,
    result,
    curve_assist_,
    enable_curve_assist_,
    finals_overlay);

  if (overlay.empty()) {
    return;
  }

  cv_bridge::CvImage debug_image;
  debug_image.header = source->header;
  debug_image.encoding = "bgr8";
  debug_image.image = overlay;

  debug_pub_.publish(debug_image.toImageMsg());
}

double LineFollowerNode::clamp(double value, double min_value, double max_value)
{
  return clamp_value(value, min_value, max_value);
}

}  // namespace line_follower
