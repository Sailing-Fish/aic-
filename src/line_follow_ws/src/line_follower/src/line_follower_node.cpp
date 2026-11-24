#include "line_follower/line_follower_node.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point.hpp>
#include <opencv2/imgproc.hpp>
#include <std_msgs/msg/bool.hpp>

namespace
{
constexpr double kPi = 3.14159265358979323846;
}  // namespace

namespace line_follower
{

LineFollowerNode::LineFollowerNode()
: Node("line_follower_node")
{
  declare_parameters();
  setup_interfaces();

  const double now_sec = this->now().seconds();
  track_state_machine_.reset(now_sec);
  curve_assist_ = CurveAssistState{};
  red_avoid_ = RedAvoidState{};
  motion_state_ = MotionState{};
  obstacle_memory_.reset();
  corridor_planner_.reset();
  finish_guard_.reset();

  start_enabled_ = false;
  waiting_log_initialized_ = true;
  last_waiting_log_time_ = this->now();

  red_left_votes_ = 0;
  red_right_votes_ = 0;
  red_turning_ = false;
  red_turn_end_time_ = this->now();
  last_red_overlay_valid_ = false;

  intersection_active_ = false;
  intersection_end_time_ = this->now();
  intersection_cooldown_until_ = this->now();

  last_odom_time_ = this->now();
  last_ekf_odom_time_ = this->now();
  last_segment_pub_time_ = this->now();
  last_seg_diag_time_ = this->now();
  yellow_brake_start_ = this->now();
  last_brake_cmd_time_ = this->now();

  RCLCPP_INFO(
    this->get_logger(),
    "Line follower subscribing to '%s', ekf '%s', det '%s', publishing cmd_vel on '%s', mode=%s/%s/%s",
    image_topic_.c_str(), ekf_odom_topic_.c_str(), detection_topic_.c_str(),
    cmd_vel_topic_.c_str(), planner_mode_.c_str(), lane_seg_mode_.c_str(),
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
  this->declare_parameter("cmd_vel_topic", "/cmd_vel");
  this->declare_parameter("debug_image_topic", "line_follower/debug_image");
  this->declare_parameter("line_state_topic", "line_follower/track_state");
  this->declare_parameter("start_topic", "line_follower/start_cmd");
  this->declare_parameter("stop_cmd_topic", "/cmd_vel_stop");
  this->declare_parameter("line_centroid_topic", "line_follower/centroid");
  this->declare_parameter("odom_topic", "/odom");
  this->declare_parameter("ekf_odom_topic", "/odom_combined");
  this->declare_parameter("detection_topic", "hobot_dnn_detection");
  this->declare_parameter("detection_timeout_sec", 0.35);

  // ============================================================
  // Finals decoupled mode switches (defaults preserve legacy_segment + legacy_hsv)
  // ============================================================
  this->declare_parameter("planner_mode", "legacy_segment");
  this->declare_parameter("lane_seg_mode", "legacy_hsv");
  this->declare_parameter("yellow_detect_mode", "legacy_area");
  this->declare_parameter("enable_semantic_hole_fill", false);
  this->declare_parameter("enable_obstacle_memory", true);
  this->declare_parameter("allow_dual_digit_cover", false);

  this->declare_parameter("target_speed", 0.6);
  this->declare_parameter("max_speed", 1.0);
  this->declare_parameter("min_speed", 0.0);
  this->declare_parameter("max_yaw_rate", 1.2);
  this->declare_parameter("kp", 1.8);

  this->declare_parameter("roi_top_ratio", 0.5);
  this->declare_parameter("roi_bottom_ratio", 0.05);
  this->declare_parameter("gaussian_kernel", 5);
  this->declare_parameter("binary_threshold", 160.0);
  this->declare_parameter("morph_kernel", 5);
  this->declare_parameter("morph_iterations", 2);
  this->declare_parameter("min_contour_area", 2000.0);
  this->declare_parameter("invert_binary", false);
  this->declare_parameter("use_white_filter", false);
  this->declare_parameter("white_h_low", 0);
  this->declare_parameter("white_h_high", 180);
  this->declare_parameter("white_s_max", 60);
  this->declare_parameter("white_v_min", 180);

  // Finals relative_border / hybrid / geometric yellow ImageProcessor parameters
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
  this->declare_parameter("yellow_min_aspect_ratio", 2.5);
  this->declare_parameter("yellow_min_width_ratio", 0.20);
  this->declare_parameter("yellow_min_area_px", 120.0);
  this->declare_parameter("yellow_ahead_strip_height_px", 28);
  this->declare_parameter("yellow_dead_end_blue_ratio", 0.35);
  this->declare_parameter("yellow_dead_end_dark_ratio", 0.20);
  this->declare_parameter("yellow_dead_end_combined_ratio", 0.50);

  // Lane band geometry & fork detection parameters
  this->declare_parameter("far_band_top_ratio", 0.34);
  this->declare_parameter("far_band_bottom_ratio", 0.56);
  this->declare_parameter("near_band_top_ratio", 0.62);
  this->declare_parameter("near_band_bottom_ratio", 0.88);
  this->declare_parameter("near_band_weight", 0.70);
  this->declare_parameter("far_band_weight", 0.30);
  this->declare_parameter("band_column_white_ratio", 0.30);
  this->declare_parameter("band_min_run_width", 20);
  this->declare_parameter("nominal_near_width_ratio", 0.72);
  this->declare_parameter("nominal_far_width_ratio", 0.52);
  this->declare_parameter("wide_road_ratio_threshold", 0.85);
  this->declare_parameter("wide_band_min_reliability", 0.15);
  this->declare_parameter("enable_width_adaptive_fusion", false);
  this->declare_parameter("wide_lane_boundary_blend", 0.0);
  this->declare_parameter("observation_filter_alpha", 1.0);
  this->declare_parameter("fork_min_gap_px", 14);
  this->declare_parameter("fork_min_branch_width_px", 18);

  this->declare_parameter("use_yellow_filter", false);
  this->declare_parameter("yellow_h_low", 20);
  this->declare_parameter("yellow_h_high", 40);
  this->declare_parameter("yellow_s_min", 80);
  this->declare_parameter("yellow_v_min", 120);
  this->declare_parameter("yellow_area_ratio_threshold", 0.1);
  this->declare_parameter("yellow_stop_duration", 1.0);
  this->declare_parameter("yellow_permanent_count", 5);
  this->declare_parameter("yellow_clear_frames", 10);

  this->declare_parameter("enable_intersection_probe", false);
  this->declare_parameter("intersection_band_height", 50);
  this->declare_parameter("intersection_dark_ratio_threshold", 0.9);
  this->declare_parameter("intersection_turn_speed", 0.2);
  this->declare_parameter("intersection_turn_yaw_rate", 0.8);
  this->declare_parameter("intersection_turn_duration", 1.0);
  this->declare_parameter("intersection_cooldown", 2.0);

  // 红色避障（legacy HSV）
  this->declare_parameter("enable_red_filter", false);
  this->declare_parameter("red_h_low_1", 0);
  this->declare_parameter("red_h_high_1", 10);
  this->declare_parameter("red_h_low_2", 160);
  this->declare_parameter("red_h_high_2", 180);
  this->declare_parameter("red_s_min", 100);
  this->declare_parameter("red_v_min", 80);
  this->declare_parameter("red_area_ratio_threshold", 0.005);
  this->declare_parameter("red_turn_yaw", 0.6);
  this->declare_parameter("red_turn_linear", 0.0);
  this->declare_parameter("red_turn_duration", 1.0);
  this->declare_parameter("red_vote_frames", 8);
  this->declare_parameter("red_divider_angle_deg", 0.0);

  // 连续弯辅助与航向/置信度控制
  this->declare_parameter("enable_continuous_curve_assist", false);
  this->declare_parameter("reverse_curve_min_error", 0.12);
  this->declare_parameter("reverse_curve_threshold", 0.35);
  this->declare_parameter("reverse_curve_confirm_frames", 3);
  this->declare_parameter("reverse_curve_clear_frames", 4);
  this->declare_parameter("reverse_curve_speed", 0.16);
  this->declare_parameter("sharp_curve_threshold", 0.50);
  this->declare_parameter("sharp_curve_speed", 0.20);
  this->declare_parameter("heading_gain", 0.0);
  this->declare_parameter("low_confidence_threshold", 0.0);
  this->declare_parameter("low_confidence_speed", 0.26);
  this->declare_parameter("low_confidence_yaw_hold", 0.75);

  // 柔性红锥避让（legacy）
  this->declare_parameter("enable_soft_red_avoidance", false);
  this->declare_parameter("red_soft_yaw", 0.30);
  this->declare_parameter("red_near_ratio", 0.35);
  this->declare_parameter("red_very_close_ratio", 0.72);
  this->declare_parameter("red_corridor_half_width_ratio", 0.42);
  this->declare_parameter("red_soft_speed", 0.15);
  this->declare_parameter("red_soft_curve_yaw_limit", 0.15);

  this->declare_parameter("red_line_aspect_ratio_min", 3.0);
  this->declare_parameter("red_line_min_width_ratio", 0.30);
  this->declare_parameter("red_cone_area_ratio_threshold", 0.0006);
  this->declare_parameter("red_cone_min_bbox_width", 4);
  this->declare_parameter("red_cone_min_bbox_height", 4);
  this->declare_parameter("red_repulsion_gain", 1.5);
  this->declare_parameter("red_lane_keep_min_factor", 0.40);
  this->declare_parameter("red_min_forward_speed", 0.06);
  this->declare_parameter("red_line_fallback_direction", -1);
  this->declare_parameter("red_line_soft_yaw", 0.35);
  this->declare_parameter("red_line_speed", 0.12);

  // 丢线阈值与分段参数化
  this->declare_parameter("line_lost_stop_time", 0.8);
  this->declare_parameter("line_lost_forward_speed", 0.06);
  this->declare_parameter("line_lost_yaw_hold_factor", 0.85);
  this->declare_parameter("line_lost_slow_time", 0.4);
  this->declare_parameter("line_lost_slow_factor", 0.4);
  this->declare_parameter("line_lost_full_stop_time", 1.2);
  this->declare_parameter("lost_line_safe_mode", false);
  this->declare_parameter("lost_line_safe_speed", 0.12);
  this->declare_parameter("lost_line_safe_stop_time", 0.35);
  this->declare_parameter("speed_gain_damping", 0.5);
  this->declare_parameter("steer_ff_gain", 0.0);
  this->declare_parameter("steer_ff_use_v", false);
  this->declare_parameter("publish_debug_image", true);

  // 停稳即走 & odom freshness
  this->declare_parameter("yellow_stop_min_time", 0.35);
  this->declare_parameter("odom_still_threshold", 0.02);
  this->declare_parameter("odom_still_hold", 0.25);
  this->declare_parameter("odom_max_age_sec", 0.20);
  this->declare_parameter("yellow_require_odom_still", false);
  this->declare_parameter("yellow_max_hold_sec", 3.0);

  // 黄线刹车（反拖）
  this->declare_parameter("yellow_brake_speed", 0.0);
  this->declare_parameter("yellow_brake_period_s", 0.2);
  this->declare_parameter("yellow_brake_pulse_s", 0.0);
  this->declare_parameter("yellow_brake_release_speed", 0.08);
  this->declare_parameter("yellow_brake_max_time_s", 0.8);

  // GroundProjector / Homography parameters
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

  // ObstacleMemory parameters
  this->declare_parameter("obstacle_merge_radius_m", 0.16);
  this->declare_parameter("obstacle_memory_ttl_sec", 1.8);
  this->declare_parameter("car_length_m", 0.280);
  this->declare_parameter("obstacle_rear_keep_margin_m", 0.15);

  // CorridorPlanner parameters
  this->declare_parameter("car_width_m", 0.191);
  this->declare_parameter("default_lane_width_m", 0.430);
  this->declare_parameter("border_margin_m", 0.025);
  this->declare_parameter("cone_margin_m", 0.025);
  this->declare_parameter("no_entry_margin_m", 0.030);
  this->declare_parameter("obstacle_lookahead_m", 1.10);
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

  // FinishGuard & two-stage yellow stop parameters
  this->declare_parameter("odom_scale_factor", 1.0);
  this->declare_parameter("finish_min_dist_m", 8.5);
  this->declare_parameter("finish_force_dist_m", 11.5);
  this->declare_parameter("finish_min_cum_yaw_rad", 0.0);
  this->declare_parameter("yellow_pre_brake_dist_m", 0.70);
  this->declare_parameter("yellow_creep_speed_mps", 0.24);
  this->declare_parameter("yellow_brake_decel_mps2", 1.8);
  this->declare_parameter("yellow_control_latency_s", 0.10);
  this->declare_parameter("yellow_bumper_margin_m", 0.08);
  this->declare_parameter("yellow_rearm_dist_m", 0.35);

  // 分段参数（legacy_segment 模式）
  this->declare_parameter("track_mode", "fork");
  this->declare_parameter("seg2_start_count", 2);
  this->declare_parameter("seg3_start_count", 0);
  this->declare_parameter("seg3_start_dist_m", 0.0);
  this->declare_parameter("seg4_start_dist_m", 0.0);
  this->declare_parameter("segment_topic", "line_follower/segment");

  {
    for (int i = 1; i <= kMaxSegments; ++i) {
      char key[64];
      const auto decl_d = [&](const char * name, double def) {
          std::snprintf(key, sizeof(key), "seg%d_%s", i, name);
          this->declare_parameter(key, def);
        };
      const auto decl_b = [&](const char * name, bool def) {
          std::snprintf(key, sizeof(key), "seg%d_%s", i, name);
          this->declare_parameter(key, def);
        };
      decl_d("target_speed", 0.50);
      decl_d("kp", 3.00);
      decl_d("steer_bias", 0.0);
      decl_d("steer_bias_dist_m", 0.0);
      decl_d("steer_bias_gap_m", 0.0);
      decl_d("steer_bias2", 0.0);
      decl_d("steer_bias2_dist_m", 0.0);
      decl_d("max_yaw_rate", 1.50);
      decl_d("speed_gain_damping", 0.50);
      decl_d("roi_top_ratio", 0.65);
      decl_d("roi_bottom_ratio", 0.20);
      decl_d("reverse_curve_speed", 0.34);
      decl_d("reverse_curve_threshold", 0.35);
      decl_d("sharp_curve_threshold", 0.50);
      decl_d("sharp_curve_speed", 0.36);
      decl_d("steer_ff_gain", 0.0);
      decl_b("enable_continuous_curve_assist", true);
      decl_b("steer_ff_use_v", false);
    }
  }

  image_topic_ = this->get_parameter("image_topic").as_string();
  cmd_vel_topic_ = this->get_parameter("cmd_vel_topic").as_string();
  debug_image_topic_ = this->get_parameter("debug_image_topic").as_string();
  line_state_topic_ = this->get_parameter("line_state_topic").as_string();
  start_topic_ = this->get_parameter("start_topic").as_string();
  stop_cmd_topic_ = this->get_parameter("stop_cmd_topic").as_string();
  line_centroid_topic_ = this->get_parameter("line_centroid_topic").as_string();
  odom_topic_ = this->get_parameter("odom_topic").as_string();
  ekf_odom_topic_ = this->get_parameter("ekf_odom_topic").as_string();
  detection_topic_ = this->get_parameter("detection_topic").as_string();
  segment_topic_ = this->get_parameter("segment_topic").as_string();

  planner_mode_ = this->get_parameter("planner_mode").as_string();
  lane_seg_mode_ = this->get_parameter("lane_seg_mode").as_string();
  yellow_detect_mode_ = this->get_parameter("yellow_detect_mode").as_string();
  enable_semantic_hole_fill_ = this->get_parameter("enable_semantic_hole_fill").as_bool();
  enable_obstacle_memory_ = this->get_parameter("enable_obstacle_memory").as_bool();
  allow_dual_digit_cover_ = this->get_parameter("allow_dual_digit_cover").as_bool();
  detection_timeout_sec_ = std::max(0.05, this->get_parameter("detection_timeout_sec").as_double());

  target_speed_ = this->get_parameter("target_speed").as_double();
  max_speed_ = this->get_parameter("max_speed").as_double();
  min_speed_ = this->get_parameter("min_speed").as_double();
  max_yaw_rate_ = this->get_parameter("max_yaw_rate").as_double();
  kp_ = this->get_parameter("kp").as_double();

  roi_top_ratio_ = this->get_parameter("roi_top_ratio").as_double();
  roi_bottom_ratio_ = this->get_parameter("roi_bottom_ratio").as_double();

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
  yellow_area_ratio_threshold_ = this->get_parameter("yellow_area_ratio_threshold").as_double();
  yellow_stop_duration_ = this->get_parameter("yellow_stop_duration").as_double();
  yellow_permanent_count_ = std::max<int>(
    1, static_cast<int>(this->get_parameter("yellow_permanent_count").as_int()));
  yellow_clear_frames_required_ = std::max<int>(
    1, static_cast<int>(this->get_parameter("yellow_clear_frames").as_int()));

  constexpr double kMaxBrakeSpeed = 1.5;
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

  far_band_top_ratio_ = this->get_parameter("far_band_top_ratio").as_double();
  far_band_bottom_ratio_ = this->get_parameter("far_band_bottom_ratio").as_double();
  near_band_top_ratio_ = this->get_parameter("near_band_top_ratio").as_double();
  near_band_bottom_ratio_ = this->get_parameter("near_band_bottom_ratio").as_double();
  near_band_weight_ = this->get_parameter("near_band_weight").as_double();
  far_band_weight_ = this->get_parameter("far_band_weight").as_double();

  enable_intersection_probe_ = this->get_parameter("enable_intersection_probe").as_bool();
  intersection_band_height_ = std::max<int>(
    1, static_cast<int>(this->get_parameter("intersection_band_height").as_int()));
  intersection_dark_ratio_threshold_ =
    this->get_parameter("intersection_dark_ratio_threshold").as_double();
  intersection_turn_speed_ = this->get_parameter("intersection_turn_speed").as_double();
  intersection_turn_yaw_rate_ = this->get_parameter("intersection_turn_yaw_rate").as_double();
  intersection_turn_duration_ = this->get_parameter("intersection_turn_duration").as_double();
  intersection_cooldown_ = this->get_parameter("intersection_cooldown").as_double();

  enable_red_filter_ = this->get_parameter("enable_red_filter").as_bool();
  red_h_low_1_ = static_cast<int>(this->get_parameter("red_h_low_1").as_int());
  red_h_high_1_ = static_cast<int>(this->get_parameter("red_h_high_1").as_int());
  red_h_low_2_ = static_cast<int>(this->get_parameter("red_h_low_2").as_int());
  red_h_high_2_ = static_cast<int>(this->get_parameter("red_h_high_2").as_int());
  red_s_min_ = static_cast<int>(this->get_parameter("red_s_min").as_int());
  red_v_min_ = static_cast<int>(this->get_parameter("red_v_min").as_int());
  red_area_ratio_threshold_ = this->get_parameter("red_area_ratio_threshold").as_double();
  red_turn_yaw_ = this->get_parameter("red_turn_yaw").as_double();
  red_turn_linear_ = this->get_parameter("red_turn_linear").as_double();
  red_turn_duration_ = this->get_parameter("red_turn_duration").as_double();
  red_vote_frames_ = std::max<int>(
    1, static_cast<int>(this->get_parameter("red_vote_frames").as_int()));
  red_divider_angle_deg_ = this->get_parameter("red_divider_angle_deg").as_double();
  red_divider_angle_rad_ = red_divider_angle_deg_ * (kPi / 180.0);

  line_lost_stop_time_ = this->get_parameter("line_lost_stop_time").as_double();
  line_lost_forward_speed_ = this->get_parameter("line_lost_forward_speed").as_double();
  line_lost_yaw_hold_factor_ = this->get_parameter("line_lost_yaw_hold_factor").as_double();
  line_lost_slow_time_ = this->get_parameter("line_lost_slow_time").as_double();
  line_lost_slow_factor_ = this->get_parameter("line_lost_slow_factor").as_double();
  line_lost_full_stop_time_ = this->get_parameter("line_lost_full_stop_time").as_double();
  speed_gain_damping_ = this->get_parameter("speed_gain_damping").as_double();
  steer_ff_gain_ = std::max(0.0, this->get_parameter("steer_ff_gain").as_double());
  steer_ff_use_v_ = this->get_parameter("steer_ff_use_v").as_bool();
  publish_debug_image_ = this->get_parameter("publish_debug_image").as_bool();

  enable_curve_assist_ = this->get_parameter("enable_continuous_curve_assist").as_bool();
  reverse_curve_speed_ = std::max(0.0, this->get_parameter("reverse_curve_speed").as_double());
  reverse_curve_threshold_ = this->get_parameter("reverse_curve_threshold").as_double();
  sharp_curve_threshold_ = this->get_parameter("sharp_curve_threshold").as_double();
  sharp_curve_speed_ = std::max(0.0, this->get_parameter("sharp_curve_speed").as_double());

  yellow_stop_min_time_ = this->get_parameter("yellow_stop_min_time").as_double();
  odom_still_threshold_ = this->get_parameter("odom_still_threshold").as_double();
  odom_still_hold_ = this->get_parameter("odom_still_hold").as_double();
  odom_max_age_sec_ = this->get_parameter("odom_max_age_sec").as_double();
  yellow_require_odom_still_ = this->get_parameter("yellow_require_odom_still").as_bool();
  yellow_max_hold_sec_ = this->get_parameter("yellow_max_hold_sec").as_double();

  // 分段参数必须在最后加载：apply_segment(1) 会覆盖上面这些成员并重建所有 cfg（包括 ImageProcessor）。
  load_segment_parameters();
}

void LineFollowerNode::sync_runtime_parameters()
{
  planner_mode_ = this->get_parameter("planner_mode").as_string();
  lane_seg_mode_ = this->get_parameter("lane_seg_mode").as_string();
  yellow_detect_mode_ = this->get_parameter("yellow_detect_mode").as_string();
  enable_semantic_hole_fill_ = this->get_parameter("enable_semantic_hole_fill").as_bool();
  enable_obstacle_memory_ = this->get_parameter("enable_obstacle_memory").as_bool();
  allow_dual_digit_cover_ = this->get_parameter("allow_dual_digit_cover").as_bool();
  detection_timeout_sec_ = std::max(0.05, this->get_parameter("detection_timeout_sec").as_double());

  // In finals_corridor mode, top-level YAML/panel params are authoritative (never clobbered by seg1_*)
  if (planner_mode_ == "finals_corridor") {
    target_speed_ = std::max(0.0, this->get_parameter("target_speed").as_double());
    max_speed_ = std::max(0.0, this->get_parameter("max_speed").as_double());
    min_speed_ = std::max(0.0, this->get_parameter("min_speed").as_double());
    kp_ = std::max(0.0, this->get_parameter("kp").as_double());
    max_yaw_rate_ = std::max(0.0, this->get_parameter("max_yaw_rate").as_double());
    speed_gain_damping_ = std::max(0.0, this->get_parameter("speed_gain_damping").as_double());
    roi_top_ratio_ = clamp(this->get_parameter("roi_top_ratio").as_double(), 0.0, 1.0);
    roi_bottom_ratio_ = clamp(this->get_parameter("roi_bottom_ratio").as_double(), 0.0, 0.5);
    enable_curve_assist_ = this->get_parameter("enable_continuous_curve_assist").as_bool();
    reverse_curve_speed_ = std::max(0.0, this->get_parameter("reverse_curve_speed").as_double());
    reverse_curve_threshold_ = this->get_parameter("reverse_curve_threshold").as_double();
    sharp_curve_threshold_ = this->get_parameter("sharp_curve_threshold").as_double();
    sharp_curve_speed_ = std::max(0.0, this->get_parameter("sharp_curve_speed").as_double());
    steer_ff_gain_ = std::max(0.0, this->get_parameter("steer_ff_gain").as_double());
    steer_ff_use_v_ = this->get_parameter("steer_ff_use_v").as_bool();
    steer_bias_ = 0.0;
    steer_bias2_ = 0.0;
    bias_phase1_dist_m_ = 0.0;
    bias_gap_m_ = 0.0;
    bias_phase2_dist_m_ = 0.0;
  }

  // ============================================================
  // P0 FIX: Always rebuild ImageProcessorConfig here so segN_roi_*
  // and all live lane_seg_mode / yellow_detect_mode / HSV switches take effect!
  // ============================================================
  white_h_low_ = static_cast<int>(this->get_parameter("white_h_low").as_int());
  white_h_high_ = static_cast<int>(this->get_parameter("white_h_high").as_int());
  white_s_max_ = static_cast<int>(this->get_parameter("white_s_max").as_int());
  white_v_min_ = static_cast<int>(this->get_parameter("white_v_min").as_int());
  yellow_h_low_ = static_cast<int>(this->get_parameter("yellow_h_low").as_int());
  yellow_h_high_ = static_cast<int>(this->get_parameter("yellow_h_high").as_int());
  yellow_s_min_ = static_cast<int>(this->get_parameter("yellow_s_min").as_int());
  yellow_v_min_ = static_cast<int>(this->get_parameter("yellow_v_min").as_int());
  yellow_area_ratio_threshold_ = this->get_parameter("yellow_area_ratio_threshold").as_double();

  ImageProcessorConfig img_cfg{};
  img_cfg.roi_top_ratio = roi_top_ratio_;
  img_cfg.roi_bottom_ratio = roi_bottom_ratio_;
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
  img_cfg.yellow_area_ratio_threshold = yellow_area_ratio_threshold_;
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
  const bool want_homography = this->get_parameter("use_homography").as_bool();
  GroundProjectorConfig proj_cfg{};
  proj_cfg.use_homography = false;
  proj_cfg.cam_height_m = this->get_parameter("cam_height_m").as_double();
  proj_cfg.cam_pitch_deg = this->get_parameter("cam_pitch_deg").as_double();
  proj_cfg.focal_px = this->get_parameter("focal_px").as_double();
  proj_cfg.cx = this->get_parameter("cam_cx").as_double();
  proj_cfg.cy = this->get_parameter("cam_cy").as_double();
  proj_cfg.front_overhang_m = this->get_parameter("front_overhang_m").as_double();
  ground_projector_.set_config(proj_cfg);
  if (want_homography) {
    const auto h_vec = this->get_parameter("homography_matrix").as_double_array();
    ground_projector_.set_homography_from_vector(h_vec, true);
  }

  far_band_top_ratio_ = this->get_parameter("far_band_top_ratio").as_double();
  far_band_bottom_ratio_ = this->get_parameter("far_band_bottom_ratio").as_double();
  near_band_top_ratio_ = this->get_parameter("near_band_top_ratio").as_double();
  near_band_bottom_ratio_ = this->get_parameter("near_band_bottom_ratio").as_double();
  near_band_weight_ = this->get_parameter("near_band_weight").as_double();
  far_band_weight_ = this->get_parameter("far_band_weight").as_double();

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
  geo_cfg.reverse_curve_threshold = reverse_curve_threshold_;
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

  odom_max_age_sec_ = this->get_parameter("odom_max_age_sec").as_double();
  yellow_require_odom_still_ = this->get_parameter("yellow_require_odom_still").as_bool();
  yellow_max_hold_sec_ = this->get_parameter("yellow_max_hold_sec").as_double();

  const bool finals_mode = (planner_mode_ == "finals_corridor");

  TrackStateConfig state_cfg{};
  state_cfg.yellow_stop_duration = yellow_stop_duration_;
  state_cfg.yellow_permanent_count = yellow_permanent_count_;
  state_cfg.yellow_clear_frames_required = yellow_clear_frames_required_;
  state_cfg.yellow_stop_min_time = yellow_stop_min_time_;
  state_cfg.odom_still_threshold = odom_still_threshold_;
  state_cfg.odom_still_hold = odom_still_hold_;
  state_cfg.odom_max_age_sec = odom_max_age_sec_;
  state_cfg.yellow_require_odom_still = yellow_require_odom_still_;
  state_cfg.yellow_max_hold_sec = yellow_max_hold_sec_;
  state_cfg.use_external_finish_decision = finals_mode;
  state_cfg.line_lost_stop_time = line_lost_stop_time_;
  track_state_machine_.set_config(state_cfg);

  MotionControllerConfig motion_cfg{};
  motion_cfg.target_speed = target_speed_;
  motion_cfg.max_speed = max_speed_;
  motion_cfg.min_speed = min_speed_;
  motion_cfg.max_yaw_rate = max_yaw_rate_;
  motion_cfg.kp = kp_;
  motion_cfg.speed_gain_damping = speed_gain_damping_;
  motion_cfg.enable_continuous_curve_assist = enable_curve_assist_;
  motion_cfg.reverse_curve_speed = reverse_curve_speed_;
  motion_cfg.sharp_curve_threshold = sharp_curve_threshold_;
  motion_cfg.sharp_curve_speed = sharp_curve_speed_;
  motion_cfg.heading_gain =
    this->get_parameter("heading_gain").as_double();
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
    this->get_parameter("lost_line_safe_mode").as_bool() || finals_mode;
  motion_cfg.lost_line_safe_speed =
    this->get_parameter("lost_line_safe_speed").as_double();
  motion_cfg.lost_line_safe_stop_time =
    this->get_parameter("lost_line_safe_stop_time").as_double();
  motion_cfg.corridor_kp_y = this->get_parameter("corridor_kp_y").as_double();
  motion_cfg.corridor_heading_gain = this->get_parameter("corridor_heading_gain").as_double();
  motion_cfg.red_near_ratio =
    this->get_parameter("red_near_ratio").as_double();
  motion_cfg.red_very_close_ratio =
    this->get_parameter("red_very_close_ratio").as_double();
  motion_cfg.red_soft_speed =
    this->get_parameter("red_soft_speed").as_double();
  motion_cfg.red_line_speed =
    this->get_parameter("red_line_speed").as_double();
  motion_cfg.red_min_forward_speed =
    this->get_parameter("red_min_forward_speed").as_double();
  motion_cfg.red_soft_curve_yaw_limit =
    this->get_parameter("red_soft_curve_yaw_limit").as_double();
  motion_cfg.red_repulsion_gain =
    this->get_parameter("red_repulsion_gain").as_double();
  motion_cfg.red_lane_keep_min_factor =
    this->get_parameter("red_lane_keep_min_factor").as_double();
  motion_controller_.set_config(motion_cfg);

  // Sync ObstacleMemory
  ObstacleMemoryConfig mem_cfg{};
  mem_cfg.merge_radius_m = this->get_parameter("obstacle_merge_radius_m").as_double();
  mem_cfg.memory_ttl_sec = this->get_parameter("obstacle_memory_ttl_sec").as_double();
  mem_cfg.car_length_m = this->get_parameter("car_length_m").as_double();
  mem_cfg.rear_keep_margin_m = this->get_parameter("obstacle_rear_keep_margin_m").as_double();
  obstacle_memory_.set_config(mem_cfg);

  // Sync CorridorPlanner
  CorridorPlannerConfig cor_cfg{};
  cor_cfg.car_width_m = this->get_parameter("car_width_m").as_double();
  cor_cfg.car_length_m = mem_cfg.car_length_m;
  cor_cfg.default_lane_width_m = this->get_parameter("default_lane_width_m").as_double();
  cor_cfg.border_margin_m = this->get_parameter("border_margin_m").as_double();
  cor_cfg.cone_margin_m = this->get_parameter("cone_margin_m").as_double();
  cor_cfg.no_entry_margin_m = this->get_parameter("no_entry_margin_m").as_double();
  cor_cfg.obstacle_lookahead_m = this->get_parameter("obstacle_lookahead_m").as_double();
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
  corridor_planner_.set_config(cor_cfg);

  // Sync FinishGuard
  FinishGuardConfig fg_cfg{};
  fg_cfg.odom_scale_factor = this->get_parameter("odom_scale_factor").as_double();
  fg_cfg.finish_min_dist_m = this->get_parameter("finish_min_dist_m").as_double();
  fg_cfg.finish_force_dist_m = this->get_parameter("finish_force_dist_m").as_double();
  fg_cfg.min_cumulative_abs_yaw_rad = this->get_parameter("finish_min_cum_yaw_rad").as_double();
  fg_cfg.pre_brake_dist_m = this->get_parameter("yellow_pre_brake_dist_m").as_double();
  fg_cfg.creep_speed_mps = this->get_parameter("yellow_creep_speed_mps").as_double();
  fg_cfg.brake_decel_mps2 = this->get_parameter("yellow_brake_decel_mps2").as_double();
  fg_cfg.control_latency_s = this->get_parameter("yellow_control_latency_s").as_double();
  fg_cfg.bumper_margin_m = this->get_parameter("yellow_bumper_margin_m").as_double();
  fg_cfg.rearm_travel_dist_m = this->get_parameter("yellow_rearm_dist_m").as_double();
  finish_guard_.set_config(fg_cfg);
}

void LineFollowerNode::load_segment_parameters()
{
  track_mode_ = this->get_parameter("track_mode").as_string();

  SegmentPlan plan;
  if (track_mode_ == "roundabout") {
    plan.segment_count = 4;
  } else if (track_mode_ == "fork") {
    plan.segment_count = 3;
  } else {
    RCLCPP_WARN(
      this->get_logger(),
      "[DEC] param track_mode='%s' 不认识（可选 fork / roundabout），按 fork 3 段处理",
      track_mode_.c_str());
    track_mode_ = "fork";
    plan.segment_count = 3;
  }
  plan.seg2_start_count = this->get_parameter("seg2_start_count").as_int();
  plan.seg3_start_count = this->get_parameter("seg3_start_count").as_int();
  plan.seg3_start_dist_m = this->get_parameter("seg3_start_dist_m").as_double();
  plan.seg4_start_dist_m = this->get_parameter("seg4_start_dist_m").as_double();
  planner_.configure(plan);

  for (int i = 1; i <= kMaxSegments; ++i) {
    SegmentParams & p = seg_params_[i - 1];
    char key[64];
    const auto d = [&](const char * name) {
        std::snprintf(key, sizeof(key), "seg%d_%s", i, name);
        return std::max(0.0, this->get_parameter(key).as_double());
      };
    const auto b = [&](const char * name) {
        std::snprintf(key, sizeof(key), "seg%d_%s", i, name);
        return this->get_parameter(key).as_bool();
      };
    const auto sd = [&](const char * name) {
        std::snprintf(key, sizeof(key), "seg%d_%s", i, name);
        return this->get_parameter(key).as_double();
      };
    p.target_speed = d("target_speed");
    p.kp = d("kp");
    p.max_yaw_rate = d("max_yaw_rate");
    const double bias_raw = sd("steer_bias");
    p.steer_bias = std::clamp(bias_raw, -p.max_yaw_rate, p.max_yaw_rate);
    if (std::abs(bias_raw) > p.max_yaw_rate + 1e-6) {
      RCLCPP_WARN(
        this->get_logger(),
        "[DEC] param seg%d_steer_bias=%.2f 超过 seg%d_max_yaw_rate=%.2f，已截断",
        i, bias_raw, i, p.max_yaw_rate);
    }
    p.bias_phase1_dist_m = d("steer_bias_dist_m");
    p.bias_gap_m = d("steer_bias_gap_m");
    p.steer_bias2 = std::clamp(sd("steer_bias2"), -p.max_yaw_rate, p.max_yaw_rate);
    p.bias_phase2_dist_m = d("steer_bias2_dist_m");
    p.speed_gain_damping = d("speed_gain_damping");
    p.roi_top_ratio = clamp(d("roi_top_ratio"), 0.0, 1.0);
    p.roi_bottom_ratio = clamp(d("roi_bottom_ratio"), 0.0, 0.5);
    p.reverse_curve_speed = d("reverse_curve_speed");
    p.reverse_curve_threshold = d("reverse_curve_threshold");
    p.sharp_curve_threshold = d("sharp_curve_threshold");
    p.sharp_curve_speed = d("sharp_curve_speed");
    p.steer_ff_gain = d("steer_ff_gain");
    p.continuous_curve_assist = b("enable_continuous_curve_assist");
    p.steer_ff_use_v = b("steer_ff_use_v");
  }

  for (int i = 1; i <= planner_.count(); ++i) {
    const SegmentParams & p = seg_params_[i - 1];
    if (p.target_speed > max_speed_ + 1e-6) {
      RCLCPP_WARN(
        this->get_logger(),
        "[DEC] param seg%d_target_speed=%.2f 超过全局 max_speed=%.2f，实际被限幅到 %.2f",
        i, p.target_speed, max_speed_, max_speed_);
    }
    if (p.kp <= 0.0) {
      RCLCPP_WARN(
        this->get_logger(), "[DEC] param seg%d_kp=%.2f 非法（必须 >0），该段不会转向", i, p.kp);
    }
  }

  if (planner_mode_ != "finals_corridor") {
    const auto overrides =
      this->get_node_parameters_interface()->get_parameter_overrides();
    static const char * kLegacyKeys[] = {
      "target_speed", "kp", "max_yaw_rate", "speed_gain_damping", "roi_top_ratio",
      "roi_bottom_ratio", "enable_continuous_curve_assist", "reverse_curve_speed",
      "reverse_curve_threshold", "sharp_curve_threshold", "sharp_curve_speed",
      "steer_ff_gain", "steer_ff_use_v"};
    std::string stale;
    for (const char * k : kLegacyKeys) {
      if (overrides.find(k) != overrides.end()) {
        stale += k;
        stale += ' ';
      }
    }
    if (!stale.empty()) {
      RCLCPP_WARN(
        this->get_logger(),
        "[DEC] param stale legacy keys ignored: %s（已改为逐段 segN_<key>）", stale.c_str());
    }
  }

  active_seg_ = 1;
  if (planner_mode_ == "finals_corridor") {
    sync_runtime_parameters();
  } else {
    apply_segment(1);
  }

  RCLCPP_INFO(
    this->get_logger(),
    "[DEC] param effective #0 mode=%s planner_mode=%s lane_seg_mode=%s yellow_detect_mode=%s "
    "enable_semantic_hole_fill=%d enable_obstacle_memory=%d allow_dual_digit_cover=%d "
    "target_speed=%.3f segs=%d max_speed=%.2f min_speed=%.2f "
    "seg2_start_count=%d seg3_start_count=%d seg3_start_dist_m=%.3f seg4_start_dist_m=%.3f "
    "yellow_stop_dur=%.2f yellow_count=%d "
    "yellow_brake_speed=%.3f yellow_brake_period_s=%.3f yellow_brake_pulse_s=%.3f "
    "yellow_brake_release_speed=%.3f yellow_brake_max_time_s=%.3f",
    track_mode_.c_str(), planner_mode_.c_str(), lane_seg_mode_.c_str(),
    yellow_detect_mode_.c_str(), enable_semantic_hole_fill_ ? 1 : 0,
    enable_obstacle_memory_ ? 1 : 0, allow_dual_digit_cover_ ? 1 : 0,
    target_speed_, planner_.count(), max_speed_, min_speed_,
    planner_.plan().seg2_start_count, planner_.plan().seg3_start_count,
    planner_.plan().seg3_start_dist_m, planner_.plan().seg4_start_dist_m,
    yellow_stop_duration_, yellow_permanent_count_,
    yellow_brake_speed_, yellow_brake_period_s_, yellow_brake_pulse_s_,
    yellow_brake_release_speed_, yellow_brake_max_time_s_);

  const auto & mcfg = motion_controller_.config();
  const auto & ccfg = corridor_planner_.config();
  const auto & fcfg = finish_guard_.config();
  RCLCPP_INFO(
    this->get_logger(),
    "[DEC] param effective #finals corridor_kp_y=%.3f corridor_heading_gain=%.3f "
    "speed_gain_damping=%.3f max_yaw_rate=%.3f "
    "yellow_bumper_margin_m=%.3f yellow_pre_brake_dist_m=%.3f yellow_creep_speed_mps=%.3f "
    "yellow_stop_min_time=%.3f yellow_brake_decel_mps2=%.3f yellow_control_latency_s=%.3f "
    "obstacle_speed_mps=%.3f no_entry_speed_mps=%.3f digit_approach_speed_mps=%.3f "
    "border_margin_m=%.3f cone_margin_m=%.3f no_entry_margin_m=%.3f "
    "digit_min_conf=%.3f max_digit_lateral_shift_m=%.3f "
    "finish_min_dist_m=%.3f finish_force_dist_m=%.3f odom_scale_factor=%.3f "
    "use_homography=%d yellow_require_odom_still=%d lost_line_safe_mode=%d",
    mcfg.corridor_kp_y, mcfg.corridor_heading_gain,
    mcfg.speed_gain_damping, mcfg.max_yaw_rate,
    fcfg.bumper_margin_m, fcfg.pre_brake_dist_m, fcfg.creep_speed_mps,
    yellow_stop_min_time_, fcfg.brake_decel_mps2, fcfg.control_latency_s,
    ccfg.obstacle_speed_mps, ccfg.no_entry_speed_mps, ccfg.digit_approach_speed_mps,
    ccfg.border_margin_m, ccfg.cone_margin_m, ccfg.no_entry_margin_m,
    ccfg.digit_min_conf, ccfg.max_digit_lateral_shift_m,
    fcfg.finish_min_dist_m, fcfg.finish_force_dist_m, fcfg.odom_scale_factor,
    this->get_parameter("use_homography").as_bool() ? 1 : 0,
    this->get_parameter("yellow_require_odom_still").as_bool() ? 1 : 0,
    mcfg.lost_line_safe_mode ? 1 : 0);

  for (int i = 1; i <= planner_.count(); ++i) {
    const SegmentParams & p = seg_params_[i - 1];
    std::string line = "[DEC] param effective #" + std::to_string(i);
    char kv[96];
    const auto add_d = [&](const char * key, double v) {
        std::snprintf(kv, sizeof(kv), " seg%d_%s=%.3f", i, key, v);
        line += kv;
      };
    add_d("target_speed", p.target_speed);
    add_d("kp", p.kp);
    add_d("steer_bias", p.steer_bias);
    add_d("steer_bias_dist_m", p.bias_phase1_dist_m);
    add_d("steer_bias_gap_m", p.bias_gap_m);
    add_d("steer_bias2", p.steer_bias2);
    add_d("steer_bias2_dist_m", p.bias_phase2_dist_m);
    add_d("max_yaw_rate", p.max_yaw_rate);
    add_d("speed_gain_damping", p.speed_gain_damping);
    add_d("roi_top_ratio", p.roi_top_ratio);
    add_d("roi_bottom_ratio", p.roi_bottom_ratio);
    add_d("reverse_curve_speed", p.reverse_curve_speed);
    add_d("reverse_curve_threshold", p.reverse_curve_threshold);
    add_d("sharp_curve_threshold", p.sharp_curve_threshold);
    add_d("sharp_curve_speed", p.sharp_curve_speed);
    add_d("steer_ff_gain", p.steer_ff_gain);
    std::snprintf(
      kv, sizeof(kv), " seg%d_enable_continuous_curve_assist=%d seg%d_steer_ff_use_v=%d",
      i, p.continuous_curve_assist ? 1 : 0, i, p.steer_ff_use_v ? 1 : 0);
    line += kv;
    RCLCPP_INFO(this->get_logger(), "%s", line.c_str());
  }
}

void LineFollowerNode::apply_segment(int segment)
{
  if (planner_mode_ == "finals_corridor") {
    sync_runtime_parameters();
    curve_assist_ = CurveAssistState{};
    ff_state_ = 0.0;
    return;
  }

  const int idx = std::clamp(segment, 1, kMaxSegments) - 1;
  const SegmentParams & p = seg_params_[idx];

  target_speed_ = p.target_speed;
  kp_ = p.kp;
  steer_bias_ = p.steer_bias;
  steer_bias2_ = p.steer_bias2;
  bias_phase1_dist_m_ = p.bias_phase1_dist_m;
  bias_gap_m_ = p.bias_gap_m;
  bias_phase2_dist_m_ = p.bias_phase2_dist_m;
  seg_dist_m_ = 0.0;
  max_yaw_rate_ = p.max_yaw_rate;
  speed_gain_damping_ = p.speed_gain_damping;
  roi_top_ratio_ = p.roi_top_ratio;
  roi_bottom_ratio_ = p.roi_bottom_ratio;
  enable_curve_assist_ = p.continuous_curve_assist;
  reverse_curve_speed_ = p.reverse_curve_speed;
  reverse_curve_threshold_ = p.reverse_curve_threshold;
  sharp_curve_threshold_ = p.sharp_curve_threshold;
  sharp_curve_speed_ = p.sharp_curve_speed;
  steer_ff_gain_ = std::max(0.0, p.steer_ff_gain);
  steer_ff_use_v_ = p.steer_ff_use_v;

  // Rebuild ImageProcessor, LaneGeometry, TrackStateMachine, MotionController, Finals configs
  sync_runtime_parameters();

  curve_assist_ = CurveAssistState{};
  ff_state_ = 0.0;
}

bool LineFollowerNode::awaiting_dist_boundary() const
{
  if (planner_mode_ == "finals_corridor") {
    return false;
  }
  const SegmentPlan & plan = planner_.plan();
  if (plan.segment_count >= 3 && plan.seg3_start_dist_m > 0.0 && active_seg_ == 2) {
    return true;
  }
  if (plan.segment_count >= 4 && plan.seg4_start_dist_m > 0.0 && active_seg_ == 3) {
    return true;
  }
  return false;
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

void LineFollowerNode::update_segment()
{
  if (planner_mode_ == "finals_corridor") {
    publish_segment_info(false);
    return;
  }

  const int yellow_count = track_state_machine_.state().yellow_line_detection_count;
  const int seg = planner_.update(yellow_count, dist_since_yellow_);
  if (seg != active_seg_) {
    const int prev = active_seg_;
    active_seg_ = seg;
    apply_segment(active_seg_);
    RCLCPP_INFO(
      this->get_logger(),
      "[DEC] seg change from=%d to=%d count=%d/%d dist=%.2f",
      prev, active_seg_, yellow_count, yellow_permanent_count_,
      dist_since_yellow_);
    publish_segment_info(true);
  } else if (awaiting_dist_boundary()) {
    const auto now = this->now();
    if ((now - last_seg_diag_time_).seconds() >= 0.5) {
      last_seg_diag_time_ = now;
      const double thr = (active_seg_ == 2) ?
        planner_.plan().seg3_start_dist_m : planner_.plan().seg4_start_dist_m;
      RCLCPP_INFO(
        this->get_logger(), "[DEC] seg d=%.2f cnt=%d thr=%.2f",
        dist_since_yellow_, yellow_count, thr);
    }
  }

  publish_segment_info(false);
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
  if (planner_mode_ == "finals_corridor") {
    std::snprintf(
      buf, sizeof(buf), "mode=finals s=%.2f count=%d branch=%d",
      finish_guard_.total_arc_m(),
      track_state_machine_.state().yellow_line_detection_count,
      corridor_planner_.committed_branch(finish_guard_.total_arc_m()));
  } else {
    std::snprintf(
      buf, sizeof(buf), "seg=%d n=%d count=%d dist=%.2f",
      active_seg_, planner_.count(),
      track_state_machine_.state().yellow_line_detection_count, dist_since_yellow_);
  }

  std_msgs::msg::String msg;
  msg.data = buf;
  segment_pub_->publish(msg);
}

void LineFollowerNode::setup_interfaces()
{
  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
  stop_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(stop_cmd_topic_, 10);
  state_pub_ = this->create_publisher<std_msgs::msg::String>(line_state_topic_, 10);
  centroid_pub_ = this->create_publisher<geometry_msgs::msg::Point>(line_centroid_topic_, 10);
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

  const double v = std::abs(msg->twist.twist.linear.x);
  if (odom_arc_valid_) {
    const double dt = (now - last_odom_time_).seconds();
    if (dt > 0.0 && dt < 0.2) {
      const double ds = v * dt;
      odom_arc_s_ += ds;
      dist_since_yellow_ += ds;
      seg_dist_m_ += ds;

      // Update FinishGuard from /odom only when /odom_combined is not actively updating
      const bool ekf_active = ekf_pose_seen_ && ((now - last_ekf_odom_time_).seconds() < 0.5);
      if (!ekf_active) {
        finish_guard_.update_motion(ds, dyaw);
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

  if (had_prev) {
    const double dt = (now - last_ekf_odom_time_).seconds();
    if (dt > 0.0 && dt < 0.25) {
      const double ds_pos = std::hypot(ekf_pose_.x - prev_x, ekf_pose_.y - prev_y);
      const double ds_vel = std::abs(ekf_pose_.linear_v) * dt;
      const double ds = (ds_pos > 1e-4 && ds_pos < 0.5) ? ds_pos : ds_vel;
      finish_guard_.update_motion(ds, dyaw);
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

void LineFollowerNode::start_callback(const std_msgs::msg::Bool::SharedPtr msg)
{
  const bool new_state = msg->data;

  if (new_state == start_enabled_) {
    return;
  }

  const bool old_state = start_enabled_;
  const auto now = this->now();

  const bool reset_lap =
    track_state_machine_.on_start_command(old_state, new_state, now.seconds()) ||
    (!old_state && new_state && planner_mode_ == "finals_corridor");

  if (reset_lap) {
    if (planner_mode_ == "finals_corridor") {
      track_state_machine_.reset(now.seconds());
    }
    red_left_votes_ = 0;
    red_right_votes_ = 0;
    red_turning_ = false;
    red_turn_end_time_ = now;

    intersection_active_ = false;
    intersection_end_time_ = now;
    intersection_cooldown_until_ = now;

    yellow_brake_active_ = false;

    obstacle_memory_.reset();
    corridor_planner_.reset();
    finish_guard_.reset();

    RCLCPP_INFO(
      this->get_logger(),
      "New lap reset: yellow count=0, permanent stop cleared, finals guard/memory reset.");

    planner_.reset();
    odom_arc_s_ = 0.0;
    dist_since_yellow_ = 0.0;
    active_seg_ = 1;
    apply_segment(1);
    publish_segment_info(true);
  }

  start_enabled_ = new_state;

  if (!start_enabled_) {
    curve_assist_ = CurveAssistState{};
    ff_state_ = 0.0;
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

  sync_runtime_parameters();

  const auto now = this->now();
  const double now_sec = now.seconds();
  const bool finals_mode = (planner_mode_ == "finals_corridor");

  // Collect fresh YOLO boxes for semantic hole-filling, yellow hybrid, and ground projection
  std::vector<cv::Rect> fill_white_rects;
  std::vector<cv::Rect> yolo_yellow_rects;
  std::vector<GroundDetection> ground_detections;
  std::vector<DebugDetectionBox> debug_boxes;

  const bool dets_fresh = (now_sec - latest_det_stamp_sec_) <= detection_timeout_sec_;
  if (dets_fresh) {
    for (const auto & box : latest_raw_boxes_) {
      if (enable_semantic_hole_fill_ &&
        (box.type == SemanticObjectType::BONUS_DIGIT ||
        box.type == SemanticObjectType::YELLOW_STOP_LINE))
      {
        fill_white_rects.push_back(box.bbox_img);
      }
      if (box.type == SemanticObjectType::YELLOW_STOP_LINE) {
        yolo_yellow_rects.push_back(box.bbox_img);
      }

      const double u_bot =
        static_cast<double>(box.bbox_img.x) + 0.5 * static_cast<double>(box.bbox_img.width);
      const double v_bot =
        static_cast<double>(box.bbox_img.y + box.bbox_img.height);
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
          // Adapt half-width to projected bbox width (covers 13cm or 24cm orientation)
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
        ground_detections.push_back(gd);

        DebugDetectionBox dbox{};
        dbox.bbox_img = box.bbox_img;
        dbox.label = box.label;
        dbox.confidence = box.confidence;
        dbox.d_bumper_m = d_m;
        dbox.y_lat_m = y_m;
        dbox.is_no_entry = (box.type == SemanticObjectType::RED_NO_ENTRY);
        dbox.is_digit = (box.type == SemanticObjectType::BONUS_DIGIT);
        dbox.is_yellow = (box.type == SemanticObjectType::YELLOW_STOP_LINE);
        debug_boxes.push_back(dbox);
      }
    }
  }

  const int active_branch = finals_mode ?
    corridor_planner_.committed_branch(finish_guard_.total_arc_m()) : 0;

  auto result_opt = process_image(frame_bgr, fill_white_rects, yolo_yellow_rects, active_branch);

  if (result_opt && result_opt->line_detected) {
    double norm_x = 0.5;
    double norm_y = 0.5;
    if (frame_bgr.cols > 0 && frame_bgr.rows > 0) {
      norm_x = result_opt->centroid.x / static_cast<double>(frame_bgr.cols);
      norm_y = (static_cast<double>(result_opt->roi_top) + result_opt->centroid.y) /
        static_cast<double>(frame_bgr.rows);
      norm_x = clamp(norm_x, 0.0, 1.0);
      norm_y = clamp(norm_y, 0.0, 1.0);
    }
    publish_centroid(norm_x, norm_y);
  }

  FinalsDebugOverlay finals_overlay{};
  finals_overlay.enabled = finals_mode || (lane_seg_mode_ != "legacy_hsv") ||
    (yellow_detect_mode_ != "legacy_area");
  finals_overlay.lane_seg_mode = lane_seg_mode_;
  finals_overlay.planner_mode = planner_mode_;
  finals_overlay.detections = debug_boxes;

  if (!start_enabled_) {
    curve_assist_ = CurveAssistState{};
    ff_state_ = 0.0;

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
  // Active pose & ObstacleMemory update (finals_corridor mode)
  // ============================================================
  const bool ekf_fresh = ekf_pose_seen_ && ((now - last_ekf_odom_time_).seconds() < 0.5);
  const VehiclePose2D & active_pose = ekf_fresh ? ekf_pose_ : odom_pose_;

  std::vector<LocalObstacle> local_obstacles;
  if (finals_mode) {
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
  }

  // ============================================================
  // Legacy HSV red avoidance (active only in legacy_segment mode)
  // ============================================================
  const bool soft_red_enabled =
    this->get_parameter("enable_soft_red_avoidance").as_bool();
  const double red_soft_yaw =
    std::max(0.0, this->get_parameter("red_soft_yaw").as_double());
  const double red_near_ratio = clamp(
    this->get_parameter("red_near_ratio").as_double(), 0.0, 1.0);
  const double red_very_close_ratio = clamp(
    this->get_parameter("red_very_close_ratio").as_double(), 0.0, 1.0);
  const double red_corridor_half_width_ratio = clamp(
    this->get_parameter("red_corridor_half_width_ratio").as_double(), 0.0, 1.0);
  const double red_min_forward_speed =
    std::max(0.01, this->get_parameter("red_min_forward_speed").as_double());

  bool soft_red_active = false;
  double soft_red_yaw_bias = 0.0;

  red_avoid_.in_corridor = false;
  red_avoid_.risk = 0.0;
  red_avoid_.very_close = false;

  if (!finals_mode && enable_red_filter_ && result_opt && result_opt->line_detected) {
    int roi_top = result_opt->roi_top;
    int roi_bottom = result_opt->roi_bottom;
    cv::Point2d line_centroid_roi(result_opt->centroid.x, result_opt->centroid.y);

    double side_sign = 0.0;
    cv::Point2d red_center_img, div_base_img, div_target_img;
    bool red_seen = detect_red(
      frame_bgr, roi_top, roi_bottom, line_centroid_roi, side_sign,
      red_center_img, div_base_img, div_target_img);

    last_red_overlay_valid_ = red_seen;

    if (red_seen) {
      last_red_center_img_ = red_center_img;
      last_div_base_img_ = div_base_img;
      last_div_target_img_ = div_target_img;

      if (side_sign < 0.0) {
        red_left_votes_++;
        red_right_votes_ = 0;
      } else {
        red_right_votes_++;
        red_left_votes_ = 0;
      }

      if (soft_red_enabled) {
        red_turning_ = false;

        const double red_bottom_ratio = clamp(red_avoid_.bottom_ratio, 0.0, 1.0);
        const bool vote_confirmed =
          red_left_votes_ >= red_vote_frames_ ||
          red_right_votes_ >= red_vote_frames_;

        const cv::Point2d corridor_v = div_target_img - div_base_img;
        const cv::Point2d red_v = red_avoid_.bottom_img - div_base_img;
        const double corridor_norm = std::hypot(corridor_v.x, corridor_v.y);

        double lateral_distance = 1e9;
        if (corridor_norm > 1e-6) {
          lateral_distance = std::abs(
            corridor_v.x * red_v.y - corridor_v.y * red_v.x) / corridor_norm;
        }

        const double corridor_limit =
          red_corridor_half_width_ratio * static_cast<double>(frame_bgr.cols);
        const bool red_in_corridor = lateral_distance <= corridor_limit;
        red_avoid_.in_corridor = red_in_corridor;

        const bool red_close = red_bottom_ratio >= red_near_ratio;
        const bool red_very_close = red_bottom_ratio >= red_very_close_ratio;

        if (vote_confirmed && red_close && (red_in_corridor || red_very_close)) {
          soft_red_active = true;
          const double base_avoid_yaw =
            red_avoid_.is_red_line ?
            std::max(0.0, this->get_parameter("red_line_soft_yaw").as_double()) :
            red_soft_yaw;

          soft_red_yaw_bias = (side_sign < 0.0) ? -base_avoid_yaw : base_avoid_yaw;
        }
      } else {
        if (red_left_votes_ >= red_vote_frames_) {
          const int dir = -1;
          red_turn_yaw_cmd_ = clamp(dir * red_turn_yaw_, -max_yaw_rate_, max_yaw_rate_);
          red_turning_ = true;
          red_turn_end_time_ = now + rclcpp::Duration::from_seconds(red_turn_duration_);
          red_left_votes_ = 0;
          red_right_votes_ = 0;
        } else if (red_right_votes_ >= red_vote_frames_) {
          const int dir = 1;
          red_turn_yaw_cmd_ = clamp(dir * red_turn_yaw_, -max_yaw_rate_, max_yaw_rate_);
          red_turning_ = true;
          red_turn_end_time_ = now + rclcpp::Duration::from_seconds(red_turn_duration_);
          red_left_votes_ = 0;
          red_right_votes_ = 0;
        }
      }
    } else {
      red_left_votes_ = 0;
      red_right_votes_ = 0;

      if (soft_red_enabled) {
        red_turning_ = false;
      } else if (now >= red_turn_end_time_) {
        red_turning_ = false;
      }
    }
  } else {
    last_red_overlay_valid_ = false;
    red_avoid_.valid = false;
    red_avoid_.red_line_detected = false;
    red_avoid_.is_red_line = false;

    if (soft_red_enabled) {
      red_left_votes_ = 0;
      red_right_votes_ = 0;
      red_turning_ = false;
    }
  }

  // ============================================================
  // 黄线检测、两段式零碰线停车与 FinishGuard 判定
  // ============================================================
  const bool raw_yellow_detected = result_opt && result_opt->yellow_line_detected;
  const bool line_detected = result_opt && result_opt->line_detected;
  const TrackState prev_state = track_state_machine_.state().last_state;

  YellowStopAdvice yellow_advice{};
  bool sm_yellow_trigger = raw_yellow_detected;
  bool external_is_finish = false;

  if (finals_mode || yellow_detect_mode_ != "legacy_area") {
    bool has_metric_dist = false;
    double yellow_d_bumper_m = 0.0;
    double yellow_y_lat_m = 0.0;

    if (raw_yellow_detected && result_opt && result_opt->yellow_is_geometric) {
      has_metric_dist = ground_projector_.pixel_to_ground(
        result_opt->yellow_center_x_full,
        result_opt->yellow_bottom_y_full,
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
      dead_end);

    finals_overlay.yellow_dist_valid = has_metric_dist;
    finals_overlay.yellow_d_bumper_m = yellow_d_bumper_m;
    finals_overlay.yellow_is_finish = yellow_advice.is_finish_line;
    finals_overlay.yellow_dead_end_ahead = dead_end;

    if (finals_mode) {
      const bool already_stopping =
        track_state_machine_.state().yellow_line_stop_active ||
        track_state_machine_.state().permanent_stop;
      sm_yellow_trigger = already_stopping ?
        raw_yellow_detected :
        (finish_guard_.is_yellow_rearmed() && yellow_advice.trigger_stop);
      external_is_finish = yellow_advice.is_finish_line;
    }
  }

  const TrackStepEvents step_events =
    track_state_machine_.step(sm_yellow_trigger, line_detected, now_sec, external_is_finish);
  const auto & sm_state = track_state_machine_.state();

  if (step_events.yellow_stop_begin) {
    dist_since_yellow_ = 0.0;
    finish_guard_.mark_yellow_stop_triggered();
    if (finals_mode) {
      corridor_planner_.reset_fork_votes();
    }

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
  }

  if (step_events.yellow_stop_end) {
    yellow_brake_active_ = false;
  }

  if (step_events.yellow_stop_end && finals_mode) {
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

  update_segment();

  if (step_events.yellow_stop_begin) {
    RCLCPP_INFO(
      this->get_logger(),
      "[DEC] yellow stop_begin count=%d/%d s=%.2f finish=%d",
      sm_state.yellow_line_detection_count,
      yellow_permanent_count_,
      finish_guard_.total_arc_m(),
      external_is_finish ? 1 : 0);
  }
  if (step_events.yellow_final_stop) {
    RCLCPP_INFO(
      this->get_logger(),
      "[DEC] yellow final_stop count=%d/%d s=%.2f",
      sm_state.yellow_line_detection_count,
      yellow_permanent_count_,
      finish_guard_.total_arc_m());
  }
  if (step_events.yellow_temp_stop) {
    RCLCPP_INFO(
      this->get_logger(),
      "[DEC] yellow temp_stop count=%d/%d dur=%.2f",
      sm_state.yellow_line_detection_count,
      yellow_permanent_count_,
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

  // While stationary at a yellow line in finals mode, continuously accumulate fork static votes
  if (finals_mode && step_events.yellow_hold_stop && !sm_state.permanent_stop) {
    corridor_planner_.update_fork_static_votes(ground_detections, now_sec);
  }

  // ============================================================
  // 交叉口探针（仅 legacy 模式）
  // ============================================================
  if (!finals_mode && enable_intersection_probe_ && line_detected &&
    !intersection_active_ && now >= intersection_cooldown_until_ &&
    !sm_state.yellow_line_stop_active && !sm_state.permanent_stop)
  {
    const cv::Mat & mask = result_opt->binary_mask;
    if (!mask.empty()) {
      int band_h = std::min(intersection_band_height_, mask.rows);
      if (band_h > 0) {
        cv::Rect band_rect(0, 0, mask.cols, band_h);
        cv::Mat band = mask(band_rect);
        int nonzero = cv::countNonZero(band);
        double total = static_cast<double>(band.total());
        double black_ratio =
          total > 0.0 ? 1.0 - static_cast<double>(nonzero) / total : 0.0;

        if (black_ratio >= intersection_dark_ratio_threshold_) {
          intersection_active_ = true;
          intersection_end_time_ =
            now + rclcpp::Duration::from_seconds(intersection_turn_duration_);
          intersection_cooldown_until_ =
            now + rclcpp::Duration::from_seconds(intersection_cooldown_);

          RCLCPP_INFO(
            this->get_logger(),
            "Intersection probe triggered: black_ratio=%.2f band_h=%d",
            black_ratio, band_h);
        }
      }
    }
  }

  // ============================================================
  // 基础控制：黄色停车 > Finals Corridor / Legacy 巡线 > 丢线处理
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

    if (finals_mode) {
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

      if (left_proj_ok && right_proj_ok && (y_l - y_r) >= 0.25 && (y_l - y_r) <= 0.70) {
        y_lane_left_m = y_l;
        y_lane_right_m = y_r;
      } else if (left_proj_ok && !right_proj_ok) {
        y_lane_left_m = y_l;
        y_lane_right_m = y_l - 2.0 * half_w_def;
      } else if (right_proj_ok && !left_proj_ok) {
        y_lane_right_m = y_r;
        y_lane_left_m = y_r + 2.0 * half_w_def;
      }

      const CorridorPlan plan = corridor_planner_.plan(
        y_lane_left_m,
        y_lane_right_m,
        0.0,  // vehicle center is y=0 in vehicle bumper frame
        local_obstacles,
        ground_detections,
        false,
        finish_guard_.total_arc_m(),
        now_sec);

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

      // Populate FinalsDebugOverlay for live inspection
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
      // Exact legacy_segment tracking controller
      const MotionCommand cmd =
        motion_controller_.compute_lane_command(
          result_opt->observation,
          curve_assist_,
          motion_state_);
      linear_speed = cmd.linear_speed;
      yaw_rate = cmd.yaw_rate;
      const double ff_raw = curve_assist_.far_error - curve_assist_.near_error;
      ff_state_ = 0.25 * ff_raw + 0.75 * ff_state_;
      double ff = ff_state_;
      if (steer_ff_use_v_) {
        ff *= clamp(sm_state.odom_speed / 0.45, 0.0, 1.5);
      }
      yaw_rate = clamp(
        yaw_rate - steer_ff_gain_ * ff + current_bias(), -max_yaw_rate_, max_yaw_rate_);
      if (finish_guard_.is_yellow_rearmed() &&
        yellow_advice.pre_brake &&
        !yellow_advice.trigger_stop)
      {
        linear_speed = std::min(linear_speed, yellow_advice.speed_cap_mps);
      }
      publish_stop_cmd = cmd.publish_stop;
    }
  } else {
    current_state = step_events.base_state;
    const MotionCommand cmd =
      motion_controller_.compute_lost_line_command(step_events.since_line_detection, motion_state_);
    linear_speed = cmd.linear_speed;
    yaw_rate = cmd.yaw_rate;
    publish_stop_cmd = cmd.publish_stop;
  }

  // ============================================================
  // 交叉口/红色控制只有在 legacy 模式且黄色停车未激活时才能覆盖巡线
  // ============================================================
  const bool yellow_stop_now = step_events.yellow_stop_now;

  if (!finals_mode && !yellow_stop_now && intersection_active_) {
    if (now <= intersection_end_time_) {
      linear_speed = intersection_turn_speed_;
      yaw_rate = clamp(intersection_turn_yaw_rate_, -max_yaw_rate_, max_yaw_rate_);
      current_state = TrackState::TRACKING;
    } else {
      intersection_active_ = false;
    }
  }

  if (!finals_mode && !yellow_stop_now && soft_red_enabled) {
    if (soft_red_active) {
      if (!red_avoid_.soft_red_logged) {
        red_avoid_.soft_red_logged = true;
        RCLCPP_INFO(
          this->get_logger(),
          "[DEC] red avoid_on side=%s kind=%s bottom=%.2f area=%.4f in_path=%d",
          soft_red_yaw_bias < 0.0 ? "L" : "R",
          red_avoid_.is_red_line ? "line" : "cone",
          red_avoid_.bottom_ratio,
          red_avoid_.contour_area_ratio,
          red_avoid_.in_corridor ? 1 : 0);
      }

      motion_controller_.fuse_soft_red_avoidance(
        soft_red_yaw_bias,
        curve_assist_,
        red_avoid_,
        linear_speed,
        yaw_rate);

      current_state = TrackState::TRACKING;
    } else {
      red_avoid_.soft_red_logged = false;
    }
  } else if (!finals_mode && !yellow_stop_now && red_turning_ && now < red_turn_end_time_) {
    linear_speed = std::max(red_min_forward_speed, red_turn_linear_);
    yaw_rate = red_turn_yaw_cmd_;
    current_state = TrackState::TRACKING;
  } else if (red_turning_ && now >= red_turn_end_time_) {
    red_turning_ = false;
  }

  // ============================================================
  // 最终安全锁：黄色停车拥有最高优先级
  // P0 FIX: Preserve active reverse-braking pulse until release_speed or timeout,
  // THEN enforce zero velocity + /cmd_vel_stop!
  // ============================================================
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

  if (finals_overlay.enabled) {
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

bool LineFollowerNode::detect_red(
  const cv::Mat & frame_bgr,
  int roi_top,
  int roi_bottom,
  const cv::Point2d & line_centroid_roi,
  double & side_sign,
  cv::Point2d & red_centroid_img,
  cv::Point2d & div_base_img,
  cv::Point2d & div_target_img)
{
  side_sign = 0.0;
  red_avoid_.clear_detection();

  if (!enable_red_filter_ || frame_bgr.empty()) {
    return false;
  }

  roi_top = std::clamp(roi_top, 0, frame_bgr.rows - 1);
  roi_bottom = std::clamp(roi_bottom, roi_top + 1, frame_bgr.rows);

  const cv::Rect roi_rect(0, roi_top, frame_bgr.cols, roi_bottom - roi_top);
  const cv::Mat roi = frame_bgr(roi_rect);

  cv::Mat hsv;
  cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

  cv::Mat mask1;
  cv::Mat mask2;
  cv::Mat mask;

  cv::inRange(
    hsv,
    cv::Scalar(
      static_cast<double>(red_h_low_1_),
      static_cast<double>(red_s_min_),
      static_cast<double>(red_v_min_)),
    cv::Scalar(static_cast<double>(red_h_high_1_), 255.0, 255.0),
    mask1);

  cv::inRange(
    hsv,
    cv::Scalar(
      static_cast<double>(red_h_low_2_),
      static_cast<double>(red_s_min_),
      static_cast<double>(red_v_min_)),
    cv::Scalar(static_cast<double>(red_h_high_2_), 255.0, 255.0),
    mask2);

  cv::bitwise_or(mask1, mask2, mask);

  const cv::Mat red_kernel =
    cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, red_kernel, cv::Point(-1, -1), 1);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, red_kernel, cv::Point(-1, -1), 1);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  if (contours.empty()) {
    return false;
  }

  const double line_aspect_min =
    std::max(1.0, this->get_parameter("red_line_aspect_ratio_min").as_double());
  const double line_width_ratio_min =
    clamp(this->get_parameter("red_line_min_width_ratio").as_double(), 0.0, 1.0);
  const double cone_area_ratio_min =
    std::max(0.0, this->get_parameter("red_cone_area_ratio_threshold").as_double());
  const int cone_min_bbox_width =
    std::max(1, static_cast<int>(this->get_parameter("red_cone_min_bbox_width").as_int()));
  const int cone_min_bbox_height =
    std::max(1, static_cast<int>(this->get_parameter("red_cone_min_bbox_height").as_int()));

  int best_cone_index = -1;
  double best_cone_area = 0.0;

  int best_line_index = -1;
  double best_line_area = 0.0;

  const double roi_area = std::max(1.0, static_cast<double>(mask.total()));

  for (size_t i = 0; i < contours.size(); ++i) {
    const double area = cv::contourArea(contours[i]);
    if (area <= 0.0) {
      continue;
    }

    const cv::Rect box = cv::boundingRect(contours[i]);
    if (box.width <= 0 || box.height <= 0) {
      continue;
    }

    const double aspect_ratio =
      static_cast<double>(box.width) / static_cast<double>(box.height);

    const bool looks_like_red_line =
      aspect_ratio >= line_aspect_min &&
      static_cast<double>(box.width) >=
      static_cast<double>(frame_bgr.cols) * line_width_ratio_min;

    if (looks_like_red_line) {
      if (area > best_line_area) {
        best_line_area = area;
        best_line_index = static_cast<int>(i);
      }
      continue;
    }

    if (box.width < cone_min_bbox_width || box.height < cone_min_bbox_height) {
      continue;
    }

    const double contour_area_ratio = area / roi_area;
    if (contour_area_ratio < cone_area_ratio_min) {
      continue;
    }

    if (area > best_cone_area) {
      best_cone_area = area;
      best_cone_index = static_cast<int>(i);
    }
  }

  if (best_line_index >= 0) {
    const cv::Rect line_box_roi =
      cv::boundingRect(contours[static_cast<size_t>(best_line_index)]);

    red_avoid_.red_line_detected = true;
    red_avoid_.red_line_bbox_img = cv::Rect(
      line_box_roi.x,
      line_box_roi.y + roi_top,
      line_box_roi.width,
      line_box_roi.height);
  }

  const bool use_red_line = best_cone_index < 0 && best_line_index >= 0;
  if (best_cone_index < 0 && best_line_index < 0) {
    return false;
  }

  const int selected_index = use_red_line ? best_line_index : best_cone_index;
  const double selected_area = use_red_line ? best_line_area : best_cone_area;
  const cv::Rect box = cv::boundingRect(contours[static_cast<size_t>(selected_index)]);

  const double aspect_ratio =
    static_cast<double>(box.width) / static_cast<double>(std::max(1, box.height));
  const double contour_area_ratio = selected_area / roi_area;

  const cv::Point2d red_centroid_roi(
    static_cast<double>(box.x) + static_cast<double>(box.width) * 0.5,
    static_cast<double>(box.y) + static_cast<double>(box.height) * 0.5);

  const cv::Point2d red_bottom_roi(
    static_cast<double>(box.x) + static_cast<double>(box.width) * 0.5,
    static_cast<double>(box.y + box.height));

  cv::Point2d base(
    static_cast<double>(mask.cols) * 0.5,
    static_cast<double>(mask.rows));

  cv::Point2d v_line = line_centroid_roi - base;

  if (red_divider_angle_rad_ != 0.0) {
    const double angle =
      v_line.x >= 0.0 ? red_divider_angle_rad_ : -red_divider_angle_rad_;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    v_line = cv::Point2d(
      v_line.x * c - v_line.y * s,
      v_line.x * s + v_line.y * c);
  }

  const cv::Point2d v_red = red_centroid_roi - base;

  if (use_red_line) {
    const double left_space = static_cast<double>(box.x);
    const double right_space = static_cast<double>(mask.cols - (box.x + box.width));

    if (right_space > left_space + 10.0) {
      side_sign = -1.0;
    } else if (left_space > right_space + 10.0) {
      side_sign = 1.0;
    } else {
      const int fallback_direction =
        static_cast<int>(this->get_parameter("red_line_fallback_direction").as_int());
      side_sign = fallback_direction >= 0 ? 1.0 : -1.0;
    }
  } else {
    const double cross = v_line.x * v_red.y - v_line.y * v_red.x;
    side_sign = cross >= 0.0 ? 1.0 : -1.0;
  }

  red_centroid_img = cv::Point2d(
    red_centroid_roi.x,
    red_centroid_roi.y + static_cast<double>(roi_top));

  const cv::Point2d red_bottom_img(
    red_bottom_roi.x,
    red_bottom_roi.y + static_cast<double>(roi_top));

  div_base_img = cv::Point2d(
    base.x,
    base.y + static_cast<double>(roi_top));

  const cv::Point2d rotated_target = base + v_line;
  div_target_img = cv::Point2d(
    rotated_target.x,
    rotated_target.y + static_cast<double>(roi_top));

  red_avoid_.valid = true;
  red_avoid_.is_red_line = use_red_line;
  red_avoid_.bbox_img = cv::Rect(box.x, box.y + roi_top, box.width, box.height);
  red_avoid_.center_img = red_centroid_img;
  red_avoid_.bottom_img = red_bottom_img;
  red_avoid_.aspect_ratio = aspect_ratio;
  red_avoid_.contour_area_ratio = contour_area_ratio;

  const double roi_h = std::max(1, roi_bottom - roi_top);
  red_avoid_.bottom_ratio = clamp(
    (red_bottom_img.y - static_cast<double>(roi_top)) / static_cast<double>(roi_h),
    0.0,
    1.0);

  return true;
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

void LineFollowerNode::publish_centroid(double norm_x, double norm_y)
{
  if (!centroid_pub_) {
    return;
  }
  geometry_msgs::msg::Point msg;
  msg.x = norm_x;
  msg.y = norm_y;
  centroid_pub_->publish(msg);
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

  const cv::Mat overlay = finals_overlay.enabled ?
    debug_renderer_.render(
      frame_bgr,
      result,
      curve_assist_,
      red_avoid_,
      last_red_overlay_valid_,
      red_divider_angle_rad_,
      enable_curve_assist_,
      finals_overlay) :
    debug_renderer_.render(
      frame_bgr,
      result,
      curve_assist_,
      red_avoid_,
      last_red_overlay_valid_,
      red_divider_angle_rad_,
      enable_curve_assist_);

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
