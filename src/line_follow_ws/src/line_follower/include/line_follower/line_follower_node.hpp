#pragma once

#include <memory>
#include <optional>
#include <string>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

namespace line_follower
{

enum class TrackState
{
  LOST = 0,
  TRACKING,
  YELLOW_LINE_DETECTED
};

struct ProcessingResult
{
  bool line_detected{false};
  bool yellow_line_detected{false};

  cv::Rect bounding_box{};
  cv::Point2f centroid{0.0F, 0.0F};

  double error{0.0};
  TrackState state{TrackState::LOST};

  cv::Mat roi_bgr;
  cv::Mat binary_mask;

  int roi_top{0};
  int roi_bottom{0};
};

class LineFollowerNode : public rclcpp::Node
{
public:
  LineFollowerNode();

private:
  void declare_parameters();
  void setup_interfaces();

  void start_callback(const std_msgs::msg::Bool::SharedPtr msg);
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  std::optional<ProcessingResult> process_image(const cv::Mat & frame_bgr) const;

  bool detect_red(
    const cv::Mat & frame_bgr,
    int roi_top,
    int roi_bottom,
    const cv::Point2d & line_centroid_roi,
    double & side_sign,
    cv::Point2d & red_centroid_img,
    cv::Point2d & div_base_img,
    cv::Point2d & div_target_img) const;

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

  // ============================================================
  // Basic motion
  // ============================================================
  double target_speed_{0.6};
  double max_speed_{1.0};
  double min_speed_{0.0};
  double max_yaw_rate_{1.2};
  double kp_{1.8};

  // ============================================================
  // ROI / image processing
  // ============================================================
  double roi_top_ratio_{0.5};
  double roi_bottom_ratio_{0.05};

  int gaussian_kernel_{5};
  double binary_threshold_{160.0};
  int morph_kernel_{5};
  int morph_iterations_{2};
  double min_contour_area_{2000.0};
  bool invert_binary_{false};

  cv::Mat morph_element_;

  // ============================================================
  // White road filter
  // ============================================================
  bool use_white_filter_{false};
  int white_h_low_{0};
  int white_h_high_{180};
  int white_s_max_{60};
  int white_v_min_{180};

  // ============================================================
  // Yellow line
  // ============================================================
  bool use_yellow_filter_{false};
  int yellow_h_low_{20};
  int yellow_h_high_{40};
  int yellow_s_min_{80};
  int yellow_v_min_{120};
  double yellow_area_ratio_threshold_{0.1};
  double yellow_stop_duration_{1.0};
  int yellow_permanent_count_{5};

  // 新增：同一条实体黄线只计数一次
  bool yellow_line_latched_{false};
  int yellow_clear_frames_{0};
  int yellow_clear_frames_required_{10};

  bool yellow_line_stop_active_{false};
  int yellow_line_detection_count_{0};
  bool permanent_stop_{false};
  rclcpp::Time yellow_line_detection_time_;

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
  // Line state / runtime state
  // ============================================================
  double line_lost_stop_time_{0.8};
  double speed_gain_damping_{0.5};
  bool publish_debug_image_{true};

  bool start_enabled_{false};
  TrackState last_state_{TrackState::LOST};
  rclcpp::Time last_detection_time_;

  // ============================================================
  // ROS interfaces
  // ============================================================
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr stop_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr centroid_pub_;

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub_;

  image_transport::Publisher debug_pub_;
  bool debug_enabled_{false};
};

}  // namespace line_follower
