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
#include "line_follower/segment_planner.hpp"
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
  // 主动转向偏置（rad/s，负=向右）。稳态平衡点 error = -steer_bias_ / kp_，
  // 用来故意把车偏到线的一侧（岔路选支）。默认 0 = 完全不影响原行为。
  // 段内可分两段施加：前 bias_phase1_dist_m_ 米用 steer_bias_，
  // 之后 bias_phase2_dist_m_ 米用 steer_bias2_（拐进去再拐出来），再之后归零。
  double steer_bias_{0.0};
  double steer_bias2_{0.0};
  double bias_phase1_dist_m_{0.0};
  double bias_gap_m_{0.0};
  double bias_phase2_dist_m_{0.0};
  // 进入本段后走过的里程（分段偏置时序用；与 dist_since_yellow_ 分开，
  // 免得中途压到黄线把时序重置）
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

  // ============================================================
  // 黄线刹车（反拖）
  //
  // 检测到黄线后不再直接发 0，而是发一段**负的 linear.x** 让底盘反拖，
  // 速度降到阈值就撤回（避免顶成倒车）。yellow_brake_speed=0 = 关闭，行为与原来一致。
  //
  // 前提：必须先用 tools/brake_probe.py --probe-only 确认固件对负速度是
  // "接受并反转"（A），而不是"当 0"（B，无效果）或"符号取反"（C，会变成前进）。
  // ============================================================
  double yellow_brake_speed_{0.0};           // 反拖速度大小 m/s，0 = 关闭
  double yellow_brake_period_s_{0.2};        // 反拖指令的发送周期
  double yellow_brake_pulse_s_{0.0};         // 每个周期里反拖多久；0 或 >=周期 = 连续
  double yellow_brake_release_speed_{0.08};  // odom |v| 降到它以下就停止反拖（0=不用）
  double yellow_brake_max_time_s_{0.8};      // 兜底：最多反拖多久（0=不限，不建议）

  bool yellow_brake_active_{false};          // 本次黄线停车是否处于反拖阶段
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
  // 视觉前馈：用 far-near 的带符号差提前打方向。默认 0 = 关闭，行为与原来完全一致。
  double steer_ff_gain_{0.0};
  bool steer_ff_use_v_{false};
  double ff_state_{0.0};
  // 连续弯辅助：默认由全局参数初始化，apply_segment() 之后按段生效，
  // sync_runtime_parameters() 每帧从这几个成员重建 cfg，所以必须留在成员里。
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
  // 分段参数（track_mode 决定 2 段还是 4 段）
  //
  // 注意：上面 target_speed_ / kp_ / max_yaw_rate_ / roi_*_ / speed_gain_damping_ /
  // steer_ff_* 都是"当前段的生效值"，由 apply_segment() 从 seg_params_ 镜像过来。
  // 热循环、红锥避让、限幅一律读这些镜像，算法层不知道"段"的存在。
  // ============================================================
  SegmentPlanner planner_;
  SegmentParams seg_params_[kMaxSegments]{};
  int active_seg_{1};
  std::string track_mode_{"fork"};
  std::string segment_topic_;

  // odom 里程积分：只服务分段切换（横向控制仍然只用视觉）。
  // 每根黄线把 dist_since_yellow_ 归零，黄线在这里起"里程锚点"作用。
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
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr segment_pub_;   // 当前段号，面板用

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  image_transport::Publisher debug_pub_;
  bool debug_enabled_{false};

  // 段号心跳：不依赖图像与 odom，相机没起来时面板也要能显示当前段
  rclcpp::TimerBase::SharedPtr segment_timer_;
};

}  // namespace line_follower
