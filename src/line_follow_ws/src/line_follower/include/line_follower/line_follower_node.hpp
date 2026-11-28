#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ai_msgs/msg/perception_targets.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <image_transport/image_transport.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/core.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>

#include "line_follower/ackermann_cbf_shield.hpp"
#include "line_follower/bev_dwa_planner.hpp"
#include "line_follower/bev_mapper.hpp"
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
  void log_effective_parameters() const;
  void setup_interfaces();

  void start_callback(const std_msgs::msg::Bool::SharedPtr msg);
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void ekf_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void detection_callback(const ai_msgs::msg::PerceptionTargets::SharedPtr msg);
  void voltage_callback(const std_msgs::msg::Float32::SharedPtr msg);

  static SemanticObjectType classify_detection_label(const std::string & label, int & digit_out);
  static double quat_to_yaw(double qx, double qy, double qz, double qw);
  static double wrap_angle_rad(double angle);

  std::optional<ProcessingResult> process_image(
    const cv::Mat & frame_bgr,
    const std::vector<cv::Rect> & fill_white_rects,
    const std::vector<cv::Rect> & yolo_yellow_rects,
    int preferred_branch);

  void publish_segment_info(bool force);
  bool brake_pulse_on(double elapsed_s) const;

  // White-lane-gated CV red obstacle fallback for finals OR-fusion
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

  void publish_debug_image(
    const sensor_msgs::msg::Image::ConstSharedPtr & source,
    const cv::Mat & frame_bgr,
    const ProcessingResult & result,
    const FinalsDebugOverlay & finals_overlay);

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
  std::string odom_topic_;
  std::string ekf_odom_topic_;
  std::string detection_topic_;
  std::string segment_topic_;
  std::string voltage_topic_;

  // ============================================================
  // Finals Mode Switches
  // ============================================================
  std::string planner_mode_{"bev_dwa"};            // "bev_dwa" (first-principles) | "legacy"
  std::string track_mode_{"fork"};                 // "fork" | "roundabout"
  std::string lane_seg_mode_{"relative_border"};   // "relative_border" | "hybrid" | "legacy_hsv"
  std::string yellow_detect_mode_{"geometric_cv"}; // "geometric_cv" | "yolo_cv_hybrid"
  bool enable_semantic_hole_fill_{true};
  bool enable_obstacle_memory_{true};
  bool enable_cv_red_fallback_{true};
  bool allow_dual_digit_cover_{false};
  bool enable_cbf_shield_{false};
  bool enable_risk_aware_decision_{true};
  double crc_no_entry_pad_y_px_{0.0};
  double crc_cone_pad_y_px_{0.0};
  double crc_yellow_pad_y_px_{0.0};
  double detection_timeout_sec_{0.35};

  // ============================================================
  // Basic motion & ROI parameters
  // ============================================================
  double target_speed_{0.44};
  double max_speed_{0.85};
  double min_speed_{0.0};
  double max_yaw_rate_{1.6};
  double speed_gain_damping_{0.45};

  double roi_top_ratio_{0.68};
  double roi_bottom_ratio_{0.18};
  double yellow_roi_bottom_ratio_{0.0};

  int gaussian_kernel_{5};
  double binary_threshold_{160.0};
  int morph_kernel_{3};
  int morph_iterations_{2};
  double min_contour_area_{2500.0};
  bool invert_binary_{false};

  // ============================================================
  // White road (legacy_hsv fallback) & yellow line parameters
  // ============================================================
  bool use_white_filter_{true};
  int white_h_low_{0};
  int white_h_high_{180};
  int white_s_max_{45};
  int white_v_min_{80};

  bool use_yellow_filter_{true};
  int yellow_h_low_{20};
  int yellow_h_high_{40};
  int yellow_s_min_{80};
  int yellow_v_min_{120};
  double yellow_stop_duration_{1.0};
  int yellow_clear_frames_required_{10};
  double yellow_stop_min_time_{0.35};
  double odom_still_threshold_{0.02};
  double odom_still_hold_{0.18};
  double odom_max_age_sec_{0.20};
  bool yellow_require_odom_still_{true};
  double yellow_max_hold_sec_{3.0};

  // ============================================================
  // Reverse-brake pulse parameters & state
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
  // Lane band geometry & lost-line thresholds
  // ============================================================
  double far_band_top_ratio_{0.28};
  double far_band_bottom_ratio_{0.48};
  double near_band_top_ratio_{0.55};
  double near_band_bottom_ratio_{0.80};
  double near_band_weight_{0.60};
  double far_band_weight_{0.40};

  double line_lost_stop_time_{0.6};
  double line_lost_forward_speed_{0.15};
  double line_lost_yaw_hold_factor_{0.85};
  double line_lost_slow_time_{0.4};
  double line_lost_slow_factor_{0.4};
  double line_lost_full_stop_time_{1.2};

  bool enable_curve_assist_{true};
  bool publish_debug_image_{true};
  double debug_image_max_fps_{15.0};
  double last_debug_pub_sec_{-1e9};
  bool use_homography_{false};

  // ============================================================
  // CV Red Obstacle Fallback parameters (enable_cv_red_fallback)
  // ============================================================
  int red_h_low_1_{0};
  int red_h_high_1_{10};
  int red_h_low_2_{160};
  int red_h_high_2_{180};
  int red_s_min_{90};
  int red_v_min_{70};
  double red_cone_area_ratio_threshold_{0.0006};

  // ============================================================
  // Domain components & runtime state objects
  // ============================================================
  ImageProcessor image_processor_;
  LaneGeometry lane_geometry_;
  TrackStateMachine track_state_machine_;
  MotionController motion_controller_;
  DebugRenderer debug_renderer_;

  GroundProjector ground_projector_;
  ObstacleMemory obstacle_memory_;
  AckermannCbfShield cbf_shield_;
  CorridorPlanner corridor_planner_;
  FinishGuard finish_guard_;
  BevDwaPlanner bev_dwa_planner_;
  BevMapper bev_mapper_;
  bool preflight_logged_{false};

  CurveAssistState curve_assist_{};
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

  // Red obstacle source attribution log state (YOLO vs CV vs BOTH)
  RedDetectionSource last_red_src_state_{RedDetectionSource::NONE};
  double last_red_src_log_sec_{-1e9};
  int red_src_yolo_only_frames_{0};
  int red_src_both_frames_{0};
  int red_src_cv_only_frames_{0};

  // Throttled [DEC] ctrl decision trace state
  std::string last_ctrl_mode_tag_{};
  int last_ctrl_branch_{0};
  double last_ctrl_log_sec_{-1e9};
  double last_bev_summary_sec_{-1e9};  // 1s 低频 BEV 白盒聚合（面板专用，避免 5Hz 全字段刷屏）
  double last_cbf_log_sec_{-1e9};

  // Tier 1 yellow-stop measured braking state for 1-param friction adaptation
  bool yellow_brake_measure_pending_{false};
  double yellow_brake_entry_speed_mps_{0.0};
  double yellow_brake_start_arc_m_{0.0};

  bool start_enabled_{false};
  bool waiting_log_initialized_{false};
  rclcpp::Time last_waiting_log_time_;

  bool odom_arc_valid_{false};
  rclcpp::Time last_odom_time_;
  rclcpp::Time last_segment_pub_time_;

  // B4: On-set parameter dirty flag (avoid per-frame get_parameter overhead)
  std::atomic<bool> params_dirty_{false};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // ============================================================
  // ROS interfaces
  // ============================================================
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr stop_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr segment_pub_;

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ekf_odom_sub_;
  rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr detection_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr voltage_sub_;

  image_transport::Publisher debug_pub_;
  bool debug_enabled_{false};

  rclcpp::TimerBase::SharedPtr segment_timer_;
};

}  // namespace line_follower
