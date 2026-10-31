#pragma once

#include <memory>
#include <optional>
#include <string>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <image_transport/image_transport.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include "line_follower/debug_renderer.hpp"
#include "line_follower/image_processor.hpp"
#include "line_follower/lane_geometry.hpp"
#include "line_follower/line_follower_types.hpp"
#include "line_follower/motion_controller.hpp"
#include "line_follower/track_state_machine.hpp"

namespace line_follower
{

class LineFollowerNode : public rclcpp::Node
{
public:
  LineFollowerNode();

private:
  void declare_parameters();
  void sync_runtime_parameters();
  void setup_interfaces();

  void start_callback(const std_msgs::msg::Bool::SharedPtr msg);
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);

  std::optional<ProcessingResult> process_image(const cv::Mat & frame_bgr);

  bool detect_red(
    const cv::Mat & frame_bgr,
    int roi_top,
    int roi_bottom,
    const cv::Point2d & line_centroid_roi,
    double & side_sign,
    cv::Point2d & red_centroid_img,
    cv::Point2d & div_base_img,
    cv::Point2d & div_target_img);

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
    const ProcessingResult & result) const;

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

  // ============================================================
  // Basic motion & ROI parameters
  // ============================================================
  double target_speed_{0.6};
  double max_speed_{1.0};
  double min_speed_{0.0};
  double max_yaw_rate_{1.2};
  double kp_{1.8};

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
  // Red obstacle avoidance
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

  CurveAssistState curve_assist_{};
  RedAvoidState red_avoid_{};
  MotionState motion_state_{};

  bool start_enabled_{false};
  bool waiting_log_initialized_{false};
  rclcpp::Time last_waiting_log_time_;

  // ============================================================
  // ROS interfaces
  // ============================================================
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr stop_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr centroid_pub_;

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  image_transport::Publisher debug_pub_;
  bool debug_enabled_{false};
};

}  // namespace line_follower
