#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ai_msgs/msg/perception_targets.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <image_transport/image_transport.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include "line_follower/corridor_planner.hpp"
#include "line_follower/debug_renderer.hpp"
#include "line_follower/finals_types.hpp"
#include "line_follower/finish_guard.hpp"
#include "line_follower/ground_projector.hpp"
#include "line_follower/image_processor.hpp"
#include "line_follower/lane_geometry.hpp"
#include "line_follower/line_follower_types.hpp"
#include "line_follower/motion_controller.hpp"
#include "line_follower/obstacle_memory.hpp"
#include "line_follower/segment_planner.hpp"
#include "line_follower/track_state_machine.hpp"

namespace line_follower
{

class LineFollowerNode : public rclcpp::Node
{
public:
  LineFollowerNode();

private:
  struct RawYoloBox
  {
    SemanticObjectType type{SemanticObjectType::UNKNOWN};
    int digit{0};
    std::string label{};
    double confidence{0.0};
    cv::Rect bbox_img{};
    int track_id{-1};
  };

  void declare_parameters();
  void sync_runtime_parameters();
  void setup_interfaces();

  void start_callback(const std_msgs::msg::Bool::SharedPtr msg);
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void ekf_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void detection_callback(const ai_msgs::msg::PerceptionTargets::SharedPtr msg);

  static SemanticObjectType classify_detection_label(const std::string & label, int & digit_out);
  static double quat_to_yaw(double qx, double qy, double qz, double qw);
  static double wrap_angle_rad(double angle);

  std::optional<ProcessingResult> process_image(
    const cv::Mat & frame_bgr,
    const std::vector<cv::Rect> & fill_white_rects,
    const std::vector<cv::Rect> & yolo_yellow_rects,
    int preferred_branch);

  // 分段：读参数 -> 判定当前段 -> 把该段参数镜像到热循环使用的成员变量
  void load_segment_parameters();
  void apply_segment(int segment);
  void update_segment();
  void publish_segment_info(bool force);
  // 当前段是否在等一个"里程"边界（岔路段2→3 / 环岛段3→4），用于 2Hz 打 d= 方便校准
  bool awaiting_dist_boundary() const;

  // 反拖脉冲相位：true = 这一帧该发 -v（周期<=0 或脉冲>=周期 时永远是 true，即连续反拖）
  bool brake_pulse_on(double elapsed_s) const;

  // 此刻实际施加的转向偏置（段内两段时序：B1 走 D1 米 → B2 走 D2 米 → 0）
  double current_bias() const
  {
    return steer_bias_at(
      seg_dist_m_, steer_bias_, bias_phase1_dist_m_, bias_gap_m_, steer_bias2_,
      bias_phase2_dist_m_);
  }

  bool detect_red(
    const cv::Mat & frame_bgr,
    int roi_top,
    int roi_bottom,
    const cv::Point2d & line_centroid_roi,
    double & side_sign,
    cv::Point2d & red_centroid_img,
    cv::Point2d & div_base_img,
    cv::Point2d & div_target_img);

  // White-lane-gated CV red obstacle fallback for finals_corridor OR-fusion
  void detect_cv_red_in_lane(
    const cv::Mat & frame_bgr,
    const ProcessingResult & result,
    const std::vector<cv::Rect> & exclude_rects,
    double now_sec,
    std::vector<GroundDetection> & out_detections,
    std::vector<cv::Rect> & out_bboxes_img) const;

  std::string state_to_string(TrackState state) const;

  void publish_cmd_vel(
    double linear_speed,
    double yaw_rate,
    bool publish_stop);

  void publish_state(TrackState state);
  void publish_centroid(double norm_x, double norm_y);

  void publish_debug_image(
    const sensor_msgs::msg::Image::ConstSharedPtr & source,
    const cv::Mat & frame_bgr,
    const ProcessingResult & result,
    const FinalsDebugOverlay & finals_overlay) const;

  static double clamp(double value, double min_value, double max_value);

  // ============================================================
  // Topics
  // ============================================================
  std::string image_topic_;
  std::string cmd_vel_topic_;
  std::string debug_image_topic_;
  std::string line_state_topic_;
  std::string start_topic_;
  std::string stop_cmd_topic_;
  std::string line_centroid_topic_;
  std::string odom_topic_;
  std::string ekf_odom_topic_;
  std::string detection_topic_;

  // ============================================================
  // Decoupled Finals Mode Switches
  // ============================================================
  std::string planner_mode_{"legacy_segment"};      // "legacy_segment" | "finals_corridor"
  std::string lane_seg_mode_{"legacy_hsv"};         // "legacy_hsv" | "relative_border" | "hybrid"
  std::string yellow_detect_mode_{"legacy_area"};   // "legacy_area" | "geometric_cv" | "yolo_cv_hybrid"
  bool enable_semantic_hole_fill_{false};
  bool enable_obstacle_memory_{true};
  bool enable_cv_red_fallback_{false};
  bool allow_dual_digit_cover_{false};
  double detection_timeout_sec_{0.35};

  // ============================================================
  // Basic motion & ROI parameters
  // ============================================================
  double target_speed_{0.6};
  double max_speed_{1.0};
  double min_speed_{0.0};
  double max_yaw_rate_{1.2};
  double kp_{1.8};
  double steer_bias_{0.0};
  double steer_bias2_{0.0};
  double bias_phase1_dist_m_{0.0};
  double bias_gap_m_{0.0};
  double bias_phase2_dist_m_{0.0};
  double seg_dist_m_{0.0};

  double roi_top_ratio_{0.5};
  double roi_bottom_ratio_{0.05};

  int gaussian_kernel_{5};
  double binary_threshold_{160.0};
  int morph_kernel_{5};
  int morph_iterations_{2};
  double min_contour_area_{2000.0};
  bool invert_binary_{false};

  // ============================================================
  // White road & yellow line filter parameters
  // ============================================================
  bool use_white_filter_{false};
  int white_h_low_{0};
  int white_h_high_{180};
  int white_s_max_{60};
  int white_v_min_{180};

  bool use_yellow_filter_{false};
  int yellow_h_low_{20};
  int yellow_h_high_{40};
  int yellow_s_min_{80};
  int yellow_v_min_{120};
  double yellow_area_ratio_threshold_{0.1};
  double yellow_stop_duration_{1.0};
  int yellow_permanent_count_{5};
  int yellow_clear_frames_required_{10};
  double yellow_stop_min_time_{0.35};
  double odom_still_threshold_{0.02};
  double odom_still_hold_{0.25};
  double odom_max_age_sec_{0.20};
  bool yellow_require_odom_still_{false};
  double yellow_max_hold_sec_{3.0};

  // ============================================================
  // 黄线刹车（反拖）
  // ============================================================
  double yellow_brake_speed_{0.0};
  double yellow_brake_period_s_{0.2};
  double yellow_brake_pulse_s_{0.0};
  double yellow_brake_release_speed_{0.08};
  double yellow_brake_max_time_s_{0.8};

  bool yellow_brake_active_{false};
  rclcpp::Time yellow_brake_start_;
  rclcpp::Time last_brake_cmd_time_;

  // ============================================================
  // Parameterized lane band geometry & lost-line thresholds
  // ============================================================
  double far_band_top_ratio_{0.34};
  double far_band_bottom_ratio_{0.56};
  double near_band_top_ratio_{0.62};
  double near_band_bottom_ratio_{0.88};
  double near_band_weight_{0.70};
  double far_band_weight_{0.30};

  double line_lost_stop_time_{0.8};
  double line_lost_forward_speed_{0.06};
  double line_lost_yaw_hold_factor_{0.85};
  double line_lost_slow_time_{0.4};
  double line_lost_slow_factor_{0.4};
  double line_lost_full_stop_time_{1.2};

  double speed_gain_damping_{0.5};
  double steer_ff_gain_{0.0};
  bool steer_ff_use_v_{false};
  double ff_state_{0.0};
  bool enable_curve_assist_{true};
  double reverse_curve_speed_{0.34};
  double reverse_curve_threshold_{0.35};
  double sharp_curve_threshold_{0.50};
  double sharp_curve_speed_{0.36};
  bool publish_debug_image_{true};

  // ============================================================
  // Intersection probe
  // ============================================================
  bool enable_intersection_probe_{false};
  int intersection_band_height_{50};
  double intersection_dark_ratio_threshold_{0.9};
  double intersection_turn_speed_{0.2};
  double intersection_turn_yaw_rate_{0.8};
  double intersection_turn_duration_{1.0};
  double intersection_cooldown_{2.0};

  bool intersection_active_{false};
  rclcpp::Time intersection_end_time_;
  rclcpp::Time intersection_cooldown_until_;

  // ============================================================
  // Red obstacle avoidance (legacy HSV path)
  // ============================================================
  bool enable_red_filter_{false};
  int red_h_low_1_{0};
  int red_h_high_1_{10};
  int red_h_low_2_{160};
  int red_h_high_2_{180};
  int red_s_min_{100};
  int red_v_min_{80};
  double red_area_ratio_threshold_{0.005};
  double red_turn_yaw_{0.6};
  double red_turn_linear_{0.0};
  double red_turn_duration_{1.0};
  int red_vote_frames_{8};
  double red_divider_angle_deg_{0.0};
  double red_divider_angle_rad_{0.0};

  int red_left_votes_{0};
  int red_right_votes_{0};
  bool red_turning_{false};
  double red_turn_yaw_cmd_{0.0};
  rclcpp::Time red_turn_end_time_;

  bool last_red_overlay_valid_{false};
  cv::Point2d last_red_center_img_{0.0, 0.0};
  cv::Point2d last_div_base_img_{0.0, 0.0};
  cv::Point2d last_div_target_img_{0.0, 0.0};

  // ============================================================
  // Extracted domain components & explicit runtime state objects
  // ============================================================
  ImageProcessor image_processor_;
  LaneGeometry lane_geometry_;
  TrackStateMachine track_state_machine_;
  MotionController motion_controller_;
  DebugRenderer debug_renderer_;

  // Finals decoupled domain components
  GroundProjector ground_projector_;
  ObstacleMemory obstacle_memory_;
  CorridorPlanner corridor_planner_;
  FinishGuard finish_guard_;

  CurveAssistState curve_assist_{};
  RedAvoidState red_avoid_{};
  MotionState motion_state_{};

  // Latest EKF/odom poses & YOLO detections
  VehiclePose2D odom_pose_{};
  VehiclePose2D ekf_pose_{};
  bool ekf_pose_seen_{false};
  double last_odom_yaw_{0.0};
  bool last_odom_yaw_valid_{false};
  double last_ekf_yaw_{0.0};
  bool last_ekf_yaw_valid_{false};
  rclcpp::Time last_ekf_odom_time_;

  std::vector<RawYoloBox> latest_raw_boxes_{};
  double latest_det_stamp_sec_{-1e9};

  // Dedicated red obstacle source attribution log state (YOLO vs CV vs BOTH)
  RedDetectionSource last_red_src_state_{RedDetectionSource::NONE};
  double last_red_src_log_sec_{-1e9};
  int red_src_yolo_only_frames_{0};
  int red_src_both_frames_{0};
  int red_src_cv_only_frames_{0};

  // Throttled [DEC] ctrl decision trace state for synchronized blackbox replay
  std::string last_ctrl_mode_tag_{};
  int last_ctrl_branch_{0};
  double last_ctrl_log_sec_{-1e9};

  bool start_enabled_{false};
  bool waiting_log_initialized_{false};
  rclcpp::Time last_waiting_log_time_;

  // ============================================================
  // 分段参数（legacy_segment 模式）
  // ============================================================
  SegmentPlanner planner_;
  SegmentParams seg_params_[kMaxSegments]{};
  int active_seg_{1};
  std::string track_mode_{"fork"};
  std::string segment_topic_;

  double odom_arc_s_{0.0};
  double dist_since_yellow_{0.0};
  bool odom_arc_valid_{false};
  rclcpp::Time last_odom_time_;
  rclcpp::Time last_segment_pub_time_;
  rclcpp::Time last_seg_diag_time_;

  // ============================================================
  // ROS interfaces
  // ============================================================
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr stop_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr centroid_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr segment_pub_;

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ekf_odom_sub_;
  rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr detection_sub_;

  image_transport::Publisher debug_pub_;
  bool debug_enabled_{false};

  rclcpp::TimerBase::SharedPtr segment_timer_;
};

}  // namespace line_follower
