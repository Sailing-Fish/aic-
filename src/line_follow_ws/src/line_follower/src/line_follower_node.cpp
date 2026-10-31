#include "line_follower/line_follower_node.hpp"

#include <algorithm>
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

  RCLCPP_INFO(
    this->get_logger(),
    "[DEC] param effective target_speed=%.2f max_speed=%.2f kp=%.2f max_yaw_rate=%.2f "
    "roi_top=%.2f roi_bottom=%.2f yellow_stop_dur=%.2f yellow_count=%d "
    "red_near=%.2f red_close=%.2f lost_speed=%.2f",
    target_speed_, max_speed_, kp_, max_yaw_rate_,
    roi_top_ratio_, roi_bottom_ratio_, yellow_stop_duration_, yellow_permanent_count_,
    this->get_parameter("red_near_ratio").as_double(),
    this->get_parameter("red_very_close_ratio").as_double(),
    this->get_parameter("line_lost_forward_speed").as_double());

  const double now_sec = this->now().seconds();
  track_state_machine_.reset(now_sec);
  curve_assist_ = CurveAssistState{};
  red_avoid_ = RedAvoidState{};
  motion_state_ = MotionState{};

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

  RCLCPP_INFO(
    this->get_logger(),
    "Line follower subscribing to '%s', publishing cmd_vel on '%s', start_topic: '%s'",
    image_topic_.c_str(), cmd_vel_topic_.c_str(), start_topic_.c_str());
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

  // 参数化近带比例、远带比例与近远权重（默认值与原写死常量完全一致）
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

  // 红色避障
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

  // 柔性红锥避让
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

  // 丢线阈值与分段参数化（默认值与原写死常量 0.4、1.2 完全一致）
  this->declare_parameter("line_lost_stop_time", 0.8);
  this->declare_parameter("line_lost_forward_speed", 0.06);
  this->declare_parameter("line_lost_yaw_hold_factor", 0.85);
  this->declare_parameter("line_lost_slow_time", 0.4);
  this->declare_parameter("line_lost_slow_factor", 0.4);
  this->declare_parameter("line_lost_full_stop_time", 1.2);
  this->declare_parameter("speed_gain_damping", 0.5);
  this->declare_parameter("publish_debug_image", true);

  // 停稳即走
  this->declare_parameter("odom_topic", "/odom");
  this->declare_parameter("yellow_stop_min_time", 0.35);
  this->declare_parameter("odom_still_threshold", 0.02);
  this->declare_parameter("odom_still_hold", 0.25);

  image_topic_ = this->get_parameter("image_topic").as_string();
  cmd_vel_topic_ = this->get_parameter("cmd_vel_topic").as_string();
  debug_image_topic_ = this->get_parameter("debug_image_topic").as_string();
  line_state_topic_ = this->get_parameter("line_state_topic").as_string();
  start_topic_ = this->get_parameter("start_topic").as_string();
  stop_cmd_topic_ = this->get_parameter("stop_cmd_topic").as_string();
  line_centroid_topic_ = this->get_parameter("line_centroid_topic").as_string();
  odom_topic_ = this->get_parameter("odom_topic").as_string();

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
  publish_debug_image_ = this->get_parameter("publish_debug_image").as_bool();

  yellow_stop_min_time_ = this->get_parameter("yellow_stop_min_time").as_double();
  odom_still_threshold_ = this->get_parameter("odom_still_threshold").as_double();
  odom_still_hold_ = this->get_parameter("odom_still_hold").as_double();

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
  image_processor_.set_config(img_cfg);

  sync_runtime_parameters();
}

void LineFollowerNode::sync_runtime_parameters()
{
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
  geo_cfg.enable_continuous_curve_assist =
    this->get_parameter("enable_continuous_curve_assist").as_bool();
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
  lane_geometry_.set_config(geo_cfg);

  line_lost_stop_time_ = this->get_parameter("line_lost_stop_time").as_double();
  line_lost_forward_speed_ = this->get_parameter("line_lost_forward_speed").as_double();
  line_lost_yaw_hold_factor_ = this->get_parameter("line_lost_yaw_hold_factor").as_double();
  line_lost_slow_time_ = this->get_parameter("line_lost_slow_time").as_double();
  line_lost_slow_factor_ = this->get_parameter("line_lost_slow_factor").as_double();
  line_lost_full_stop_time_ = this->get_parameter("line_lost_full_stop_time").as_double();

  TrackStateConfig state_cfg{};
  state_cfg.yellow_stop_duration = yellow_stop_duration_;
  state_cfg.yellow_permanent_count = yellow_permanent_count_;
  state_cfg.yellow_clear_frames_required = yellow_clear_frames_required_;
  state_cfg.yellow_stop_min_time = yellow_stop_min_time_;
  state_cfg.odom_still_threshold = odom_still_threshold_;
  state_cfg.odom_still_hold = odom_still_hold_;
  state_cfg.line_lost_stop_time = line_lost_stop_time_;
  track_state_machine_.set_config(state_cfg);

  MotionControllerConfig motion_cfg{};
  motion_cfg.target_speed = target_speed_;
  motion_cfg.max_speed = max_speed_;
  motion_cfg.min_speed = min_speed_;
  motion_cfg.max_yaw_rate = max_yaw_rate_;
  motion_cfg.kp = kp_;
  motion_cfg.speed_gain_damping = speed_gain_damping_;
  motion_cfg.enable_continuous_curve_assist = geo_cfg.enable_continuous_curve_assist;
  motion_cfg.reverse_curve_speed =
    this->get_parameter("reverse_curve_speed").as_double();
  motion_cfg.sharp_curve_threshold =
    this->get_parameter("sharp_curve_threshold").as_double();
  motion_cfg.sharp_curve_speed =
    this->get_parameter("sharp_curve_speed").as_double();
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
}

void LineFollowerNode::setup_interfaces()
{
  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
  stop_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(stop_cmd_topic_, 10);
  state_pub_ = this->create_publisher<std_msgs::msg::String>(line_state_topic_, 10);
  centroid_pub_ = this->create_publisher<geometry_msgs::msg::Point>(line_centroid_topic_, 10);

  image_sub_ = image_transport::create_subscription(
    this,
    image_topic_,
    std::bind(&LineFollowerNode::image_callback, this, std::placeholders::_1),
    "raw");

  start_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    start_topic_, 10, std::bind(&LineFollowerNode::start_callback, this, std::placeholders::_1));

  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_, 10, std::bind(&LineFollowerNode::odom_callback, this, std::placeholders::_1));

  if (publish_debug_image_) {
    debug_pub_ = image_transport::create_publisher(this, debug_image_topic_);
    debug_enabled_ = true;
  }
}

void LineFollowerNode::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  track_state_machine_.update_odom(msg->twist.twist.linear.x, this->now().seconds());
}

void LineFollowerNode::start_callback(const std_msgs::msg::Bool::SharedPtr msg)
{
  const bool new_state = msg->data;

  if (new_state == start_enabled_) {
    return;
  }

  const bool old_state = start_enabled_;
  const auto now = this->now();

  if (track_state_machine_.on_start_command(old_state, new_state, now.seconds())) {
    red_left_votes_ = 0;
    red_right_votes_ = 0;
    red_turning_ = false;
    red_turn_end_time_ = now;

    intersection_active_ = false;
    intersection_end_time_ = now;
    intersection_cooldown_until_ = now;

    RCLCPP_INFO(
      this->get_logger(),
      "Previous lap finished. New lap reset: yellow count=0, permanent stop cleared.");
  }

  start_enabled_ = new_state;

  if (!start_enabled_) {
    curve_assist_ = CurveAssistState{};
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
  auto result_opt = process_image(frame_bgr);

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

  if (!start_enabled_) {
    curve_assist_ = CurveAssistState{};

    const rclcpp::Time current_time = this->now();
    if (!waiting_log_initialized_) {
      waiting_log_initialized_ = true;
      last_waiting_log_time_ = current_time;
    } else if ((current_time - last_waiting_log_time_).seconds() > 2.0) {
      RCLCPP_INFO(this->get_logger(), "Line follower is not started, waiting for start signal...");
      last_waiting_log_time_ = current_time;
    }

    if (result_opt && debug_enabled_) {
      publish_debug_image(msg, frame_bgr, *result_opt);
    }

    publish_cmd_vel(0.0, 0.0, true);
    return;
  }

  const auto now = this->now();

  double linear_speed = 0.0;
  double yaw_rate = 0.0;
  TrackState current_state = TrackState::LOST;
  bool publish_stop_cmd = false;

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

  if (enable_red_filter_ && result_opt && result_opt->line_detected) {
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
  // 黄线与丢线状态机更新
  // ============================================================
  const bool yellow_detected = result_opt && result_opt->yellow_line_detected;
  const bool line_detected = result_opt && result_opt->line_detected;
  const TrackState prev_state = track_state_machine_.state().last_state;

  const TrackStepEvents step_events =
    track_state_machine_.step(yellow_detected, line_detected, now.seconds());
  const auto & sm_state = track_state_machine_.state();

  if (step_events.yellow_stop_begin) {
    RCLCPP_INFO(
      this->get_logger(),
      "[DEC] yellow stop_begin count=%d/%d",
      sm_state.yellow_line_detection_count,
      yellow_permanent_count_);
  }
  if (step_events.yellow_final_stop) {
    RCLCPP_INFO(
      this->get_logger(),
      "[DEC] yellow final_stop count=%d/%d",
      sm_state.yellow_line_detection_count,
      yellow_permanent_count_);
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
      "[DEC] yellow stop_end dur=%.2f odom_v=%.3f still=%d",
      step_events.since_yellow_detection,
      sm_state.odom_speed,
      step_events.odom_still_ok ? 1 : 0);
  }

  // ============================================================
  // 交叉口探针
  // ============================================================
  if (enable_intersection_probe_ && line_detected &&
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
  // 基础控制：黄色停车 > 普通巡线 > 丢线低速继续
  // ============================================================
  if (step_events.yellow_hold_stop) {
    current_state = TrackState::YELLOW_LINE_DETECTED;
    linear_speed = 0.0;
    yaw_rate = 0.0;
    publish_stop_cmd = true;
  } else if (line_detected) {
    current_state = TrackState::TRACKING;
    const MotionCommand cmd =
      motion_controller_.compute_lane_command(
        result_opt->observation,
        curve_assist_,
        motion_state_);
    linear_speed = cmd.linear_speed;
    yaw_rate = cmd.yaw_rate;
    publish_stop_cmd = cmd.publish_stop;
  } else {
    current_state = step_events.base_state;
    const MotionCommand cmd =
      motion_controller_.compute_lost_line_command(step_events.since_line_detection, motion_state_);
    linear_speed = cmd.linear_speed;
    yaw_rate = cmd.yaw_rate;
    publish_stop_cmd = cmd.publish_stop;
  }

  // ============================================================
  // 交叉口/红色控制只有在黄色停车未激活时才能覆盖巡线
  // ============================================================
  const bool yellow_stop_now = step_events.yellow_stop_now;

  if (!yellow_stop_now && intersection_active_) {
    if (now <= intersection_end_time_) {
      linear_speed = intersection_turn_speed_;
      yaw_rate = clamp(intersection_turn_yaw_rate_, -max_yaw_rate_, max_yaw_rate_);
      current_state = TrackState::TRACKING;
    } else {
      intersection_active_ = false;
    }
  }

  if (!yellow_stop_now && soft_red_enabled) {
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
  } else if (!yellow_stop_now && red_turning_ && now < red_turn_end_time_) {
    linear_speed = std::max(red_min_forward_speed, red_turn_linear_);
    yaw_rate = red_turn_yaw_cmd_;
    current_state = TrackState::TRACKING;
  } else if (red_turning_ && now >= red_turn_end_time_) {
    red_turning_ = false;
  }

  // ============================================================
  // 最终安全锁：黄色停车拥有最高优先级
  // ============================================================
  if (step_events.yellow_hold_stop) {
    current_state = TrackState::YELLOW_LINE_DETECTED;
    linear_speed = 0.0;
    yaw_rate = 0.0;
    publish_stop_cmd = true;
  }

  MotionController::record_motion(yellow_stop_now, linear_speed, yaw_rate, motion_state_);

  if (result_opt && debug_enabled_) {
    publish_debug_image(msg, frame_bgr, *result_opt);
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

std::optional<ProcessingResult> LineFollowerNode::process_image(const cv::Mat & frame_bgr)
{
  const auto segmented_opt = image_processor_.segment(frame_bgr);
  if (!segmented_opt) {
    return std::nullopt;
  }

  return lane_geometry_.evaluate(*segmented_opt, curve_assist_);
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
  const ProcessingResult & result) const
{
  if (!debug_enabled_ || result.binary_mask.empty()) {
    return;
  }

  const bool curve_assist_enabled =
    this->get_parameter("enable_continuous_curve_assist").as_bool();

  const cv::Mat overlay = debug_renderer_.render(
    frame_bgr,
    result,
    curve_assist_,
    red_avoid_,
    last_red_overlay_valid_,
    red_divider_angle_rad_,
    curve_assist_enabled);

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
