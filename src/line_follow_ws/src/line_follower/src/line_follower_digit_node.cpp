#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <ai_msgs/msg/perception_targets.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/twist.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/imgproc.hpp>
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
  double error{0.0};
  cv::Point2f centroid{0.0F, 0.0F};
  cv::Rect bounding_box{0, 0, 0, 0};
  cv::Mat binary_mask;
  cv::Mat roi_bgr;
  int roi_top{0};
  int roi_bottom{0};
  TrackState state{TrackState::LOST};
};

class LineFollowerDigitNode : public rclcpp::Node
{
public:
  LineFollowerDigitNode()
  : Node("line_follower_digit_node")
  {
    declare_parameters();
    setup_interfaces();

    const int kernel_size = std::max<int>(1, morph_kernel_);
    morph_element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    last_detection_time_ = this->now();
    yellow_line_detection_time_ = this->now();
    last_state_ = TrackState::LOST;
    yellow_line_stop_active_ = false;
    yellow_line_detection_count_ = 0;
    permanent_stop_ = false;
    start_enabled_ = false;
    post_yellow_active_until_ = this->now();
    intersection_phase_until_ = this->now();
    intersection_turn_until_ = this->now();
    intersection_cooldown_until_ = this->now();
    red_active_until_ = this->now();
    last_speed_update_time_ = this->now();

    RCLCPP_INFO(
      this->get_logger(),
      "Line follower+digit subscribing '%s' and '%s', publishing cmd_vel on '%s'",
      image_topic_.c_str(), detection_topic_.c_str(), cmd_vel_topic_.c_str());
  }

private:
  void declare_parameters()
  {
    this->declare_parameter("image_topic", "/ascamera/camera_publisher/rgb/image0");
    this->declare_parameter("detection_topic", "hobot_dnn_detection");
    this->declare_parameter("cmd_vel_topic", "/cmd_vel");
    this->declare_parameter("start_topic", "line_follower/start_cmd");
    this->declare_parameter("line_state_topic", "line_follower/track_state");

    this->declare_parameter("target_speed", 0.6);
    this->declare_parameter("max_speed", 1.0);
    this->declare_parameter("min_speed", 0.0);
    this->declare_parameter("max_yaw_rate", 1.0);
    this->declare_parameter("kp", 1.8);
    this->declare_parameter("speed_gain_damping", 0.5);

    this->declare_parameter("roi_top_ratio", 0.8);
    this->declare_parameter("roi_bottom_ratio", 0.3);
    this->declare_parameter("gaussian_kernel", 5);
    this->declare_parameter("binary_threshold", 160.0);
    this->declare_parameter("morph_kernel", 3);
    this->declare_parameter("morph_iterations", 2);
    this->declare_parameter("min_contour_area", 2500.0);
    this->declare_parameter("invert_binary", false);
    this->declare_parameter("use_white_filter", true);
    this->declare_parameter("white_h_low", 0);
    this->declare_parameter("white_h_high", 180);
    this->declare_parameter("white_s_max", 60);
    this->declare_parameter("white_v_min", 100);
    this->declare_parameter("use_bgr_white_filter", false);
    this->declare_parameter("white_b_min", 170);
    this->declare_parameter("white_g_min", 170);
    this->declare_parameter("white_r_min", 170);
    this->declare_parameter("white_b_max", 255);
    this->declare_parameter("white_g_max", 255);
    this->declare_parameter("white_r_max", 255);

    this->declare_parameter("use_yellow_filter", true);
    this->declare_parameter("yellow_h_low", 20);
    this->declare_parameter("yellow_h_high", 40);
    this->declare_parameter("yellow_s_min", 80);
    this->declare_parameter("yellow_v_min", 120);
    this->declare_parameter("yellow_area_ratio_threshold", 0.05);
    this->declare_parameter("yellow_stop_duration", 0.7);
    this->declare_parameter("line_lost_stop_time", 0.8);
    this->declare_parameter("yellow_permanent_count", 5);
    this->declare_parameter("post_yellow_duration", 1.5);
    this->declare_parameter("post_yellow_speed_cap", 0.4);
    this->declare_parameter("speed_accel_limit", 0.8);  // m/s per second

    // Intersection鎺㈤拡
    this->declare_parameter("enable_intersection_probe", true);
    this->declare_parameter("intersection_top_ratio", 0.25);  // 椤堕儴甯︾姸楂樺害鍗犳瘮
    this->declare_parameter("intersection_bottom_ratio", 0.02);  // 淇濈暀鍙傛暟锛屽綋鍓嶆湭浣跨敤
    this->declare_parameter("intersection_area_ratio_threshold", 0.25);
    this->declare_parameter("intersection_dark_area_threshold", 0.02);  // 椤堕儴甯﹁繎涔庡叏榛戞椂涔熻Е鍙?    this->declare_parameter("intersection_width_ratio_threshold", 0.6);
    this->declare_parameter("intersection_slow_speed", 0.2);
    this->declare_parameter("intersection_turn_yaw", 0.8);
    this->declare_parameter("intersection_slow_duration", 0.6);
    this->declare_parameter("intersection_turn_duration", 0.8);
    this->declare_parameter("intersection_cooldown", 2.0);

    // 绾㈣壊绂佽妫€娴?    this->declare_parameter("enable_red_filter", false);
    this->declare_parameter("red_h_low_1", 0);
    this->declare_parameter("red_h_high_1", 10);
    this->declare_parameter("red_h_low_2", 160);
    this->declare_parameter("red_h_high_2", 180);
    this->declare_parameter("red_s_min", 100);
    this->declare_parameter("red_v_min", 80);
    this->declare_parameter("red_area_ratio_threshold", 0.01);
    this->declare_parameter("red_push_gain", 0.8);
    this->declare_parameter("red_active_timeout", 1.0);

    // Digit detection parameters
    this->declare_parameter("min_digit_confidence", 0.4);
    this->declare_parameter("digit_stable_frames", 2);
    this->declare_parameter("digit_stable_timeout", 1.0);
    this->declare_parameter("digit_goal_timeout", 2.0);
    this->declare_parameter("digit_yaw_gain", 0.75);
    this->declare_parameter("digit_y_align_tolerance", 0.05);
    this->declare_parameter("det_frame_width", 640);
    this->declare_parameter("det_frame_height", 480);

    image_topic_ = this->get_parameter("image_topic").as_string();
    detection_topic_ = this->get_parameter("detection_topic").as_string();
    cmd_vel_topic_ = this->get_parameter("cmd_vel_topic").as_string();
    start_topic_ = this->get_parameter("start_topic").as_string();
    line_state_topic_ = this->get_parameter("line_state_topic").as_string();

    target_speed_ = this->get_parameter("target_speed").as_double();
    max_speed_ = this->get_parameter("max_speed").as_double();
    min_speed_ = this->get_parameter("min_speed").as_double();
    max_yaw_rate_ = this->get_parameter("max_yaw_rate").as_double();
    kp_ = this->get_parameter("kp").as_double();
    speed_gain_damping_ = this->get_parameter("speed_gain_damping").as_double();

    roi_top_ratio_ = this->get_parameter("roi_top_ratio").as_double();
    roi_bottom_ratio_ = this->get_parameter("roi_bottom_ratio").as_double();
    gaussian_kernel_ = this->get_parameter("gaussian_kernel").as_int();
    if (gaussian_kernel_ <= 0) {
      gaussian_kernel_ = 1;
    }
    if (gaussian_kernel_ % 2 == 0) {
      ++gaussian_kernel_;
    }

    binary_threshold_ = this->get_parameter("binary_threshold").as_double();
    morph_kernel_ = std::max<int>(1, this->get_parameter("morph_kernel").as_int());
    morph_iterations_ = std::max<int>(1, this->get_parameter("morph_iterations").as_int());
    min_contour_area_ = this->get_parameter("min_contour_area").as_double();
    invert_binary_ = this->get_parameter("invert_binary").as_bool();
    use_white_filter_ = this->get_parameter("use_white_filter").as_bool();
    use_bgr_white_filter_ = this->get_parameter("use_bgr_white_filter").as_bool();
    white_h_low_ = this->get_parameter("white_h_low").as_int();
    white_h_high_ = this->get_parameter("white_h_high").as_int();
    white_s_max_ = this->get_parameter("white_s_max").as_int();
    white_v_min_ = this->get_parameter("white_v_min").as_int();
    white_b_min_ = this->get_parameter("white_b_min").as_int();
    white_g_min_ = this->get_parameter("white_g_min").as_int();
    white_r_min_ = this->get_parameter("white_r_min").as_int();
    white_b_max_ = this->get_parameter("white_b_max").as_int();
    white_g_max_ = this->get_parameter("white_g_max").as_int();
    white_r_max_ = this->get_parameter("white_r_max").as_int();

    use_yellow_filter_ = this->get_parameter("use_yellow_filter").as_bool();
    yellow_h_low_ = this->get_parameter("yellow_h_low").as_int();
    yellow_h_high_ = this->get_parameter("yellow_h_high").as_int();
    yellow_s_min_ = this->get_parameter("yellow_s_min").as_int();
    yellow_v_min_ = this->get_parameter("yellow_v_min").as_int();
    yellow_area_ratio_threshold_ = this->get_parameter("yellow_area_ratio_threshold").as_double();
    yellow_stop_duration_ = this->get_parameter("yellow_stop_duration").as_double();
    line_lost_stop_time_ = this->get_parameter("line_lost_stop_time").as_double();
    yellow_permanent_count_ = std::max<int>(1, this->get_parameter("yellow_permanent_count").as_int());
    post_yellow_duration_ = this->get_parameter("post_yellow_duration").as_double();
    post_yellow_speed_cap_ = this->get_parameter("post_yellow_speed_cap").as_double();
    speed_accel_limit_ = this->get_parameter("speed_accel_limit").as_double();

    enable_intersection_probe_ = this->get_parameter("enable_intersection_probe").as_bool();
    intersection_top_ratio_ = this->get_parameter("intersection_top_ratio").as_double();
    intersection_bottom_ratio_ = this->get_parameter("intersection_bottom_ratio").as_double();
    intersection_area_ratio_threshold_ =
      this->get_parameter("intersection_area_ratio_threshold").as_double();
    intersection_dark_area_threshold_ =
      this->get_parameter("intersection_dark_area_threshold").as_double();
    intersection_width_ratio_threshold_ =
      this->get_parameter("intersection_width_ratio_threshold").as_double();
    intersection_slow_speed_ = this->get_parameter("intersection_slow_speed").as_double();
    intersection_turn_yaw_ = this->get_parameter("intersection_turn_yaw").as_double();
    intersection_slow_duration_ = this->get_parameter("intersection_slow_duration").as_double();
    intersection_turn_duration_ = this->get_parameter("intersection_turn_duration").as_double();
    intersection_cooldown_ = this->get_parameter("intersection_cooldown").as_double();

    enable_red_filter_ = this->get_parameter("enable_red_filter").as_bool();
    red_h_low_1_ = this->get_parameter("red_h_low_1").as_int();
    red_h_high_1_ = this->get_parameter("red_h_high_1").as_int();
    red_h_low_2_ = this->get_parameter("red_h_low_2").as_int();
    red_h_high_2_ = this->get_parameter("red_h_high_2").as_int();
    red_s_min_ = this->get_parameter("red_s_min").as_int();
    red_v_min_ = this->get_parameter("red_v_min").as_int();
    red_area_ratio_threshold_ = this->get_parameter("red_area_ratio_threshold").as_double();
    red_push_gain_ = this->get_parameter("red_push_gain").as_double();
    red_active_timeout_ = this->get_parameter("red_active_timeout").as_double();

    min_digit_confidence_ = this->get_parameter("min_digit_confidence").as_double();
    digit_stable_frames_ = std::max<int>(1, this->get_parameter("digit_stable_frames").as_int());
    digit_stable_timeout_ = this->get_parameter("digit_stable_timeout").as_double();
    digit_goal_timeout_ = this->get_parameter("digit_goal_timeout").as_double();
    digit_yaw_gain_ = this->get_parameter("digit_yaw_gain").as_double();
    digit_y_align_tolerance_ = this->get_parameter("digit_y_align_tolerance").as_double();
    det_frame_width_ = this->get_parameter("det_frame_width").as_int();
    det_frame_height_ = this->get_parameter("det_frame_height").as_int();
  }

  void setup_interfaces()
  {
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
    state_pub_ = this->create_publisher<std_msgs::msg::String>(line_state_topic_, 10);

    image_sub_ = image_transport::create_subscription(
      this, image_topic_,
      std::bind(&LineFollowerDigitNode::image_callback, this, std::placeholders::_1),
      "raw");

    start_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      start_topic_, 10, std::bind(&LineFollowerDigitNode::start_callback, this, std::placeholders::_1));

    detection_sub_ = this->create_subscription<ai_msgs::msg::PerceptionTargets>(
      detection_topic_, rclcpp::QoS(10).best_effort(),
      std::bind(&LineFollowerDigitNode::detection_callback, this, std::placeholders::_1));
  }

  void start_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    bool new_state = msg->data;
    if (new_state == start_enabled_) {
      return;
    }
    start_enabled_ = new_state;
    RCLCPP_INFO(this->get_logger(), "Start signal changed to: %s", start_enabled_ ? "true" : "false");
  }

  void detection_callback(const ai_msgs::msg::PerceptionTargets::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    struct Candidate
    {
      int digit;
      double conf;
      double center_x;
      double center_y;
    };
    std::vector<Candidate> candidates;
    double frame_w = det_frame_width_ > 0 ? static_cast<double>(det_frame_width_) : 1.0;
    double frame_h = det_frame_height_ > 0 ? static_cast<double>(det_frame_height_) : 1.0;

    for (const auto & target : msg->targets) {
      for (const auto & roi : target.rois) {
        if (roi.rect.width <= 0 || roi.confidence < min_digit_confidence_) {
          continue;
        }
        int digit = parse_digit_label(roi.type);
        if (digit < 0) {
          continue;
        }
        double cx = static_cast<double>(roi.rect.x_offset) + static_cast<double>(roi.rect.width) * 0.5;
        double cy = static_cast<double>(roi.rect.y_offset) + static_cast<double>(roi.rect.height) * 0.5;
        candidates.push_back(Candidate{digit, roi.confidence, cx, cy});
      }
    }

    if (candidates.size() < 2) {
      reset_digit_state();
      return;
    }

    const auto winner = *std::max_element(
      candidates.begin(), candidates.end(),
      [](const Candidate & a, const Candidate & b) {
        if (a.digit != b.digit) {
          return a.digit < b.digit;
        }
        return a.conf < b.conf;
      });

    rclcpp::Time now = this->now();
    double half_w = frame_w * 0.5;
    double half_h = frame_h * 0.5;
    double cx_norm = half_w > 0.0 ? (winner.center_x - half_w) / half_w : 0.0;
    double cy_norm = half_h > 0.0 ? (winner.center_y - half_h) / half_h : 0.0;
    cx_norm = clamp(cx_norm, -1.0, 1.0);
    cy_norm = clamp(cy_norm, -1.0, 1.0);

    double since_start = (now - pending_digit_start_time_).seconds();
    if (pending_digit_value_ != winner.digit || since_start > digit_stable_timeout_) {
      pending_digit_value_ = winner.digit;
      pending_digit_count_ = 1;
      pending_digit_start_time_ = now;
    } else {
      pending_digit_count_++;
    }

    since_start = (now - pending_digit_start_time_).seconds();
    if (pending_digit_count_ < digit_stable_frames_ || since_start > digit_stable_timeout_) {
      return;
    }

    last_digit_value_ = winner.digit;
    last_digit_center_norm_x_ = cx_norm;
    last_digit_center_norm_y_ = cy_norm;
    last_digit_time_ = now;
    digit_active_ = true;
  }

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    cv::Mat frame_bgr;
    try {
      frame_bgr = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    auto result_opt = process_image(frame_bgr);

    if (!start_enabled_) {
      publish_cmd_vel(0.0, 0.0);
      return;
    }

    auto now = this->now();
    double linear_speed = 0.0;
    double yaw_rate = 0.0;
    TrackState current_state = TrackState::LOST;
    bool publish_stop = false;
    double line_center_norm_x = 0.0;
    double line_center_norm_y = 0.0;
    bool line_center_valid = false;
    if (result_opt && result_opt->line_detected) {
      const auto & result = *result_opt;
      double half_w = frame_bgr.cols > 0 ? static_cast<double>(frame_bgr.cols) * 0.5 : 1.0;
      double half_h = frame_bgr.rows > 0 ? static_cast<double>(frame_bgr.rows) * 0.5 : 1.0;
      double cx = static_cast<double>(result.centroid.x);
      double cy = static_cast<double>(result.roi_top) + static_cast<double>(result.centroid.y);
      line_center_norm_x = half_w > 0.0 ? (cx - half_w) / half_w : 0.0;
      line_center_norm_y = half_h > 0.0 ? (cy - half_h) / half_h : 0.0;
      line_center_norm_x = clamp(line_center_norm_x, -1.0, 1.0);
      line_center_norm_y = clamp(line_center_norm_y, -1.0, 1.0);
      line_center_valid = true;
    }

    // 绾㈣壊绂佽妫€娴?    if (enable_red_filter_) {
      double offset_norm = 0.0;
      if (detect_red_zone(frame_bgr, offset_norm)) {
        red_active_until_ = now + rclcpp::Duration::from_seconds(red_active_timeout_);
        red_offset_norm_ = offset_norm;
      }
    }
    bool red_active = enable_red_filter_ && (now <= red_active_until_);

    // 浜ゅ弶鍙ｆ帰閽?    if (enable_intersection_probe_ &&
      result_opt && result_opt->line_detected &&
      now >= intersection_cooldown_until_ &&
      !yellow_line_stop_active_ && !permanent_stop_)
    {
      double band_area_ratio = 0.0;
      double band_width_ratio = 0.0;
      if (detect_intersection(result_opt->binary_mask, band_area_ratio, band_width_ratio)) {
        intersection_phase_ = 1;
        intersection_phase_until_ = now + rclcpp::Duration::from_seconds(intersection_slow_duration_);
        intersection_turn_until_ = intersection_phase_until_ +
          rclcpp::Duration::from_seconds(intersection_turn_duration_);
        intersection_cooldown_until_ = now + rclcpp::Duration::from_seconds(intersection_cooldown_);
        intersection_turn_sign_ = 1.0;  // 鍥哄畾宸﹁浆
        RCLCPP_INFO(this->get_logger(), "Intersection triggered: area %.2f width %.2f",
          band_area_ratio, band_width_ratio);
      }
    }

    // 姝ｅ父宸＄嚎鎺у埗
    if (current_state != TrackState::YELLOW_LINE_DETECTED) {
      if (result_opt && result_opt->line_detected) {
        const auto & result = *result_opt;
        last_detection_time_ = now;
        current_state = TrackState::TRACKING;

        double yaw_cmd = clamp(-(kp_ * result.error), -max_yaw_rate_, max_yaw_rate_);
        double composite_err = std::abs(result.error);
        double speed_scale = 1.0 - speed_gain_damping_ * std::min(1.0, composite_err);
        double speed_cmd = clamp(target_speed_ * speed_scale, min_speed_, max_speed_);

        linear_speed = speed_cmd;
        yaw_rate = yaw_cmd;
      } else {
        double since_detection = (now - last_detection_time_).seconds();
        if (since_detection >= line_lost_stop_time_) {
          current_state = TrackState::LOST;
          linear_speed = 0.0;
          yaw_rate = 0.0;
          publish_stop = true;
        } else {
          current_state = last_state_;
          linear_speed = 0.0;
          yaw_rate = 0.0;
        }
      }
    }

    // 浜ゅ弶鍙ｈ鐩栨帶鍒?    if (intersection_phase_ > 0 && now <= intersection_cooldown_until_) {
      if (now <= intersection_phase_until_) {
        linear_speed = intersection_slow_speed_;
        yaw_rate = 0.0;
      } else if (now <= intersection_turn_until_) {
        linear_speed = intersection_slow_speed_;
        yaw_rate = clamp(intersection_turn_sign_ * intersection_turn_yaw_, -max_yaw_rate_, max_yaw_rate_);
      } else {
        intersection_phase_ = 0;
      }
    }

    // 绾㈣壊绂佽鍋忚浆/鍑忛€?    if (red_active) {
      double bias = clamp(red_push_gain_ * red_offset_norm_, -max_yaw_rate_, max_yaw_rate_);
      yaw_rate = clamp(yaw_rate + bias, -max_yaw_rate_, max_yaw_rate_);
      linear_speed = std::min(linear_speed, target_speed_ * 0.3);
    }

    // 鏁板瓧鍋忚埅鍙犲姞锛堝鏋滄湁鏁堜笖鏈繃鏈燂級
    if (digit_active_ && line_center_valid) {
      double age = (now - last_digit_time_).seconds();
      if (age > digit_goal_timeout_) {
        reset_digit_state();
      } else {
        double y_diff = last_digit_center_norm_y_ - line_center_norm_y;
        if (std::abs(y_diff) <= digit_y_align_tolerance_) {
          double x_diff = last_digit_center_norm_x_ - line_center_norm_x;
          double bias = clamp(digit_yaw_gain_ * x_diff, -max_yaw_rate_, max_yaw_rate_);
          yaw_rate = clamp(yaw_rate + bias, -max_yaw_rate_, max_yaw_rate_);
        }
      }
    }


    // 鍔犻€熷害闄愬埗
    double dt = (now - last_speed_update_time_).seconds();
    if (dt > 0.0) {
      double max_delta = speed_accel_limit_ * dt;
      double delta = clamp(linear_speed - last_speed_cmd_, -max_delta, max_delta);
      linear_speed = last_speed_cmd_ + delta;
    }
    last_speed_cmd_ = linear_speed;
    last_speed_update_time_ = now;

    publish_cmd_vel(linear_speed, yaw_rate, publish_stop);
    if (current_state != last_state_ || current_state == TrackState::LOST || current_state == TrackState::YELLOW_LINE_DETECTED) {
      publish_state(current_state);
    }
    last_state_ = current_state;
  }

  std::optional<ProcessingResult> process_image(const cv::Mat & frame_bgr) const
  {
    if (frame_bgr.empty()) {
      return std::nullopt;
    }

    const int height = frame_bgr.rows;
    const int width = frame_bgr.cols;

    int roi_top = std::clamp(
      static_cast<int>(height * (1.0 - roi_top_ratio_)), 0, height - 1);
    int roi_bottom = std::clamp(
      height - static_cast<int>(height * roi_bottom_ratio_), roi_top + 1, height);
    const int roi_height = roi_bottom - roi_top;

    cv::Rect roi_rect(0, roi_top, width, roi_height);
    cv::Mat roi = frame_bgr(roi_rect);

    cv::Mat blurred;
    cv::GaussianBlur(roi, blurred, cv::Size(gaussian_kernel_, gaussian_kernel_), 0);

    cv::Mat gray;
    cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);

    cv::Mat hsv;
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    cv::Mat white_mask;
    if (use_white_filter_) {
      cv::Scalar lower(
        static_cast<double>(white_h_low_), 0.0, static_cast<double>(white_v_min_));
      cv::Scalar upper(
        static_cast<double>(white_h_high_), static_cast<double>(white_s_max_), 255.0);
      cv::inRange(hsv, lower, upper, white_mask);
    }

    cv::Mat yellow_mask;
    bool yellow_line_detected = false;
    if (use_yellow_filter_) {
      cv::Scalar lower(
        static_cast<double>(yellow_h_low_), static_cast<double>(yellow_s_min_), static_cast<double>(yellow_v_min_));
      cv::Scalar upper(
        static_cast<double>(yellow_h_high_), 255.0, 255.0);
      cv::inRange(hsv, lower, upper, yellow_mask);

      int yellow_pixels = cv::countNonZero(yellow_mask);
      double yellow_area_ratio = static_cast<double>(yellow_pixels) / static_cast<double>(yellow_mask.total());
      if (yellow_area_ratio >= yellow_area_ratio_threshold_) {
        yellow_line_detected = true;
      }
    }

    int threshold_type = invert_binary_ ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
    cv::Mat binary;
    cv::threshold(gray, binary, binary_threshold_, 255, threshold_type);

    if (use_white_filter_ && !white_mask.empty() && !invert_binary_) {
      cv::bitwise_or(binary, white_mask, binary);
    }
    if (use_bgr_white_filter_) {
      cv::Mat white_bgr_mask;
      cv::Scalar lower_bgr(
        static_cast<double>(white_b_min_),
        static_cast<double>(white_g_min_),
        static_cast<double>(white_r_min_));
      cv::Scalar upper_bgr(
        static_cast<double>(white_b_max_),
        static_cast<double>(white_g_max_),
        static_cast<double>(white_r_max_));
      cv::inRange(roi, lower_bgr, upper_bgr, white_bgr_mask);
      if (!white_bgr_mask.empty() && !invert_binary_) {
        cv::morphologyEx(
          white_bgr_mask, white_bgr_mask, cv::MORPH_CLOSE, morph_element_, cv::Point(-1, -1), 1);
        cv::bitwise_or(binary, white_bgr_mask, binary);
      }
    }

    cv::morphologyEx(
      binary, binary, cv::MORPH_CLOSE, morph_element_, cv::Point(-1, -1), morph_iterations_);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
      return ProcessingResult{};
    }

    double max_area = 0.0;
    size_t best_index = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
      double area = cv::contourArea(contours[i]);
      if (area > max_area) {
        max_area = area;
        best_index = i;
      }
    }

    if (max_area < min_contour_area_) {
      return ProcessingResult{};
    }

    ProcessingResult result;
    result.line_detected = true;
    result.yellow_line_detected = yellow_line_detected;
    result.bounding_box = cv::boundingRect(contours[best_index]);
    result.roi_bgr = roi.clone();
    result.binary_mask = binary.clone();
    result.roi_top = roi_rect.y;
    result.roi_bottom = roi_rect.y + roi_rect.height;

    const auto moments = cv::moments(contours[best_index]);
    if (moments.m00 != 0.0) {
      result.centroid.x = static_cast<float>(moments.m10 / moments.m00);
      result.centroid.y = static_cast<float>(moments.m01 / moments.m00);
    } else {
      result.centroid = cv::Point2f(
        static_cast<float>(result.bounding_box.x + result.bounding_box.width / 2.0),
        static_cast<float>(result.bounding_box.y + result.bounding_box.height / 2.0));
    }
    result.error = (static_cast<double>(result.centroid.x) - width / 2.0) / (width / 2.0);
    result.error = clamp(result.error, -1.0, 1.0);
    result.state = TrackState::TRACKING;
    return result;
  }

  std::string state_to_string(TrackState state) const
  {
    switch (state) {
      case TrackState::TRACKING:
        return "tracking";
      case TrackState::YELLOW_LINE_DETECTED:
        return "yellow_line_detected";
      case TrackState::LOST:
      default:
        return "lost";
    }
  }

  void publish_cmd_vel(double linear_speed, double yaw_rate, bool publish_stop = false)
  {
    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = linear_speed;
    cmd.angular.z = yaw_rate;
    cmd_vel_pub_->publish(cmd);
    // no stop topic here; upper-level mux handles priority
    (void)publish_stop;
  }

  void publish_state(TrackState state)
  {
    std_msgs::msg::String msg;
    msg.data = state_to_string(state);
    state_pub_->publish(msg);
  }

  static double clamp(double value, double min_value, double max_value)
  {
    return std::max(min_value, std::min(value, max_value));
  }

  bool detect_intersection(const cv::Mat & binary, double & area_ratio, double & width_ratio) const
  {
    if (binary.empty()) {
      return false;
    }
    int h = binary.rows;
    int w = binary.cols;
    // 椤堕儴甯︾姸鍖哄煙
    int band_top = 0;
    int band_bottom = std::clamp(static_cast<int>(h * intersection_top_ratio_), 1, h);
    cv::Rect band_rect(0, band_top, w, band_bottom - band_top);
    cv::Mat band = binary(band_rect);
    int white_pixels = cv::countNonZero(band);
    area_ratio = static_cast<double>(white_pixels) / static_cast<double>(band.total());
    // 鏉′欢1锛氶《閮ㄥ甫杩戜箮鍏ㄩ粦涔熷垽瀹氳Е鍙?    if (area_ratio <= intersection_dark_area_threshold_) {
      width_ratio = 0.0;
      return true;
    }
    if (area_ratio < intersection_area_ratio_threshold_) {
      width_ratio = 0.0;
      return false;
    }
    cv::Mat proj;
    cv::reduce(band, proj, 0, cv::REDUCE_MAX);
    int nonzero_cols = cv::countNonZero(proj);
    width_ratio = static_cast<double>(nonzero_cols) / static_cast<double>(w);
    return width_ratio >= intersection_width_ratio_threshold_;
  }

  bool detect_red_zone(const cv::Mat & frame_bgr, double & offset_norm) const
  {
    if (!enable_red_filter_ || frame_bgr.empty()) {
      return false;
    }
    cv::Mat hsv;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask1, mask2, mask;
    cv::inRange(
      hsv,
      cv::Scalar(static_cast<double>(red_h_low_1_), static_cast<double>(red_s_min_), static_cast<double>(red_v_min_)),
      cv::Scalar(static_cast<double>(red_h_high_1_), 255.0, 255.0),
      mask1);
    cv::inRange(
      hsv,
      cv::Scalar(static_cast<double>(red_h_low_2_), static_cast<double>(red_s_min_), static_cast<double>(red_v_min_)),
      cv::Scalar(static_cast<double>(red_h_high_2_), 255.0, 255.0),
      mask2);
    cv::bitwise_or(mask1, mask2, mask);
    int red_pixels = cv::countNonZero(mask);
    double area_ratio = static_cast<double>(red_pixels) / static_cast<double>(mask.total());
    if (area_ratio < red_area_ratio_threshold_) {
      return false;
    }
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
      return false;
    }
    double max_area = 0.0;
    size_t best = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
      double a = cv::contourArea(contours[i]);
      if (a > max_area) {
        max_area = a;
        best = i;
      }
    }
    cv::Rect box = cv::boundingRect(contours[best]);
    double cx = static_cast<double>(box.x) + static_cast<double>(box.width) * 0.5;
    double half_w = mask.cols * 0.5;
    offset_norm = half_w > 0.0 ? (cx - half_w) / half_w : 0.0;
    offset_norm = clamp(offset_norm, -1.0, 1.0);
    return true;
  }

  int parse_digit_label(const std::string & label) const
  {
    if (label.empty()) {
      return -1;
    }
    size_t start_idx = 0;
    if (label[0] == 'd' || label[0] == 'D') {
      start_idx = 1;
    }
    if (start_idx >= label.size()) {
      return -1;
    }
    for (size_t i = start_idx; i < label.size(); ++i) {
      if (!std::isdigit(static_cast<unsigned char>(label[i]))) {
        return -1;
      }
    }
    int value = -1;
    try {
      value = std::stoi(label.substr(start_idx));
    } catch (const std::exception &) {
      return -1;
    }
    if (value >= 1 && value <= 8) {
      return value;
    }
    return -1;
  }

  void reset_digit_state()
  {
    digit_active_ = false;
    pending_digit_value_ = -1;
    pending_digit_count_ = 0;
    last_digit_value_ = -1;
    last_digit_center_norm_x_ = 0.0;
    last_digit_center_norm_y_ = 0.0;
    last_digit_time_ = this->now();
    pending_digit_start_time_ = this->now();
  }

  // Parameters
  std::string image_topic_;
  std::string detection_topic_;
  std::string cmd_vel_topic_;
  std::string start_topic_;
  std::string line_state_topic_;

  double target_speed_{0.6};
  double max_speed_{1.0};
  double min_speed_{0.0};
  double max_yaw_rate_{1.0};
  double kp_{1.8};
  double speed_gain_damping_{0.5};

  double roi_top_ratio_{0.8};
  double roi_bottom_ratio_{0.3};
  int gaussian_kernel_{5};
  double binary_threshold_{160.0};
  int morph_kernel_{3};
  int morph_iterations_{2};
  double min_contour_area_{2500.0};
  bool invert_binary_{false};
  bool use_white_filter_{true};
  bool use_bgr_white_filter_{false};
  int white_h_low_{0};
  int white_h_high_{180};
  int white_s_max_{60};
  int white_v_min_{100};
  int white_b_min_{170};
  int white_g_min_{170};
  int white_r_min_{170};
  int white_b_max_{255};
  int white_g_max_{255};
  int white_r_max_{255};

  bool use_yellow_filter_{true};
  int yellow_h_low_{20};
  int yellow_h_high_{40};
  int yellow_s_min_{80};
  int yellow_v_min_{120};
  double yellow_area_ratio_threshold_{0.05};
  double yellow_stop_duration_{0.7};
  double line_lost_stop_time_{0.8};
  int yellow_permanent_count_{5};
  double post_yellow_duration_{1.5};
  double post_yellow_speed_cap_{0.4};
  double speed_accel_limit_{0.8};

  bool enable_intersection_probe_{true};
  double intersection_top_ratio_{0.1};
  double intersection_bottom_ratio_{0.02};
  double intersection_area_ratio_threshold_{0.25};
  double intersection_dark_area_threshold_{0.02};
  double intersection_width_ratio_threshold_{0.6};
  double intersection_slow_speed_{0.2};
  double intersection_turn_yaw_{0.8};
  double intersection_slow_duration_{0.6};
  double intersection_turn_duration_{0.8};
  double intersection_cooldown_{2.0};

  bool enable_red_filter_{false};
  int red_h_low_1_{0};
  int red_h_high_1_{10};
  int red_h_low_2_{160};
  int red_h_high_2_{180};
  int red_s_min_{100};
  int red_v_min_{80};
  double red_area_ratio_threshold_{0.01};
  double red_push_gain_{0.8};
  double red_active_timeout_{1.0};

  double min_digit_confidence_{0.4};
  int digit_stable_frames_{2};
  double digit_stable_timeout_{1.0};
  double digit_goal_timeout_{2.0};
  double digit_yaw_gain_{0.75};
  double digit_y_align_tolerance_{0.05};
  int det_frame_width_{640};
  int det_frame_height_{480};

  // Interfaces
  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr detection_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  image_transport::Publisher debug_pub_;
  bool debug_enabled_{false};

  // State
  mutable cv::Mat morph_element_;
  rclcpp::Time last_detection_time_;
  rclcpp::Time yellow_line_detection_time_;
  TrackState last_state_{TrackState::LOST};
  bool yellow_line_stop_active_{false};
  int yellow_line_detection_count_{0};
  bool permanent_stop_{false};
  bool start_enabled_{false};
  rclcpp::Time post_yellow_active_until_;

  int intersection_phase_{0};  // 0: none, 1: slow, then turn
  rclcpp::Time intersection_phase_until_;
  rclcpp::Time intersection_turn_until_;
  rclcpp::Time intersection_cooldown_until_;
  double intersection_turn_sign_{1.0};

  rclcpp::Time red_active_until_;
  double red_offset_norm_{0.0};

  double last_speed_cmd_{0.0};
  rclcpp::Time last_speed_update_time_;

  bool digit_active_{false};
  int pending_digit_value_{-1};
  int pending_digit_count_{0};
  rclcpp::Time pending_digit_start_time_;
  int last_digit_value_{-1};
  double last_digit_center_norm_x_{0.0};
  double last_digit_center_norm_y_{0.0};
  rclcpp::Time last_digit_time_;
};

}  // namespace line_follower

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<line_follower::LineFollowerDigitNode>());
  rclcpp::shutdown();
  return 0;
}


