#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <ai_msgs/msg/perception_targets.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace
{
constexpr double kDefaultConfThreshold = 0.6;
constexpr int kDefaultDetFrameWidth = 640;
constexpr int kDefaultDetFrameHeight = 480;
constexpr int kDefaultFrameWindow = 5;
constexpr double kDefaultTurnCooldown = 0.3;
constexpr double kDefaultTurnDuration = 0.6;
constexpr double kDefaultTurnLinear = 0.3;
constexpr double kDefaultYawGain = 1.0;
constexpr double kDefaultMaxYaw = 1.2;
constexpr double kDefaultYAlignTolerance = 0.05;
constexpr double kDefaultTurnDirectionSign = -1.0;  // 正值时 digit 在右转右，负值则相反
constexpr double kDefaultCentroidTimeout = 0.5;
constexpr const char * kDefaultLineCentroidTopic = "line_follower/centroid";
constexpr const char * kDefaultStateTopic = "line_follower/track_state";
}  // namespace

class DetectionTurnController : public rclcpp::Node
{
public:
  DetectionTurnController()
  : Node("detection_turn_controller")
  {
    detection_topic_ = this->declare_parameter<std::string>("detection_topic", "hobot_dnn_detection");
    state_topic_ = this->declare_parameter<std::string>("state_topic", kDefaultStateTopic);
    cmd_vel_topic_ = this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel_detection");

    conf_threshold_ = this->declare_parameter<double>("confidence_threshold", kDefaultConfThreshold);
    det_frame_width_ = this->declare_parameter<int>("det_frame_width", kDefaultDetFrameWidth);
    det_frame_height_ = this->declare_parameter<int>("det_frame_height", kDefaultDetFrameHeight);
    frame_window_ = this->declare_parameter<int>("frame_window", kDefaultFrameWindow);
    turn_cooldown_ = this->declare_parameter<double>("turn_cooldown", kDefaultTurnCooldown);
    turn_duration_ = this->declare_parameter<double>("turn_duration", kDefaultTurnDuration);
    turn_linear_ = this->declare_parameter<double>("turn_linear", kDefaultTurnLinear);
    yaw_gain_ = this->declare_parameter<double>("yaw_gain", kDefaultYawGain);
    max_yaw_rate_ = this->declare_parameter<double>("max_yaw_rate", kDefaultMaxYaw);
    y_align_tolerance_ = this->declare_parameter<double>("y_align_tolerance", kDefaultYAlignTolerance);
    turn_direction_sign_ = this->declare_parameter<double>("turn_direction_sign", kDefaultTurnDirectionSign);

    line_centroid_topic_ = this->declare_parameter<std::string>(
      "line_centroid_topic", kDefaultLineCentroidTopic);
    centroid_timeout_ = this->declare_parameter<double>("centroid_timeout", kDefaultCentroidTimeout);

    detection_sub_ = this->create_subscription<ai_msgs::msg::PerceptionTargets>(
      detection_topic_, rclcpp::QoS(10).best_effort(),
      std::bind(&DetectionTurnController::detection_callback, this, std::placeholders::_1));

    centroid_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
      line_centroid_topic_, rclcpp::QoS(10),
      std::bind(&DetectionTurnController::centroid_callback, this, std::placeholders::_1));

    state_sub_ = this->create_subscription<std_msgs::msg::String>(
      state_topic_, rclcpp::QoS(10),
      std::bind(&DetectionTurnController::state_callback, this, std::placeholders::_1));

    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

    publish_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50), std::bind(&DetectionTurnController::on_timer, this));

    RCLCPP_INFO(
      get_logger(),
      "Detection turn controller listening detections '%s', centroid '%s', publishing '%s'",
      detection_topic_.c_str(), line_centroid_topic_.c_str(), cmd_vel_topic_.c_str());
  }

private:
  struct Candidate
  {
    int digit;
    double conf;
    double center_x;
    double center_y;
  };

  static int parse_digit_label(const std::string & label)
  {
    if (label.empty()) {
      return -1;
    }
    size_t start = (label[0] == 'd' || label[0] == 'D') ? 1 : 0;
    if (start >= label.size()) {
      return -1;
    }
    for (size_t i = start; i < label.size(); ++i) {
      if (!std::isdigit(static_cast<unsigned char>(label[i]))) {
        return -1;
      }
    }
    int value = -1;
    try {
      value = std::stoi(label.substr(start));
    } catch (...) {
      return -1;
    }
    if (value >= 1 && value <= 8) {
      return value;
    }
    return -1;
  }

  void centroid_callback(const geometry_msgs::msg::Point::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    last_centroid_norm_x_ = std::clamp(static_cast<double>(msg->x), 0.0, 1.0);
    last_centroid_norm_y_ = std::clamp(static_cast<double>(msg->y), 0.0, 1.0);
    last_centroid_time_ = this->now();
  }

  void state_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    const auto & s = msg->data;
    if (s == "yellow_line_detected") {
      // 黄线触发：清空当前数字会话并解除冷却，便于黄线后立即开始新一次数字识别
      session_active_ = false;
      session_frames_ = 0;
      best_digit_ = -1;
      best_conf_ = 0.0;
      best_x_diff_ = 0.0;
      best_y_diff_ = 0.0;
      turning_ = false;
      turn_cooldown_until_ = this->now();
    }
  }

  void detection_callback(const ai_msgs::msg::PerceptionTargets::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    rclcpp::Time now = this->now();

    // 正在转弯或冷却中，忽略
    if (turning_ || now < turn_cooldown_until_) {
      return;
    }

    const bool centroid_valid =
      (now - last_centroid_time_).seconds() <= centroid_timeout_;
    if (!centroid_valid) {
      return;
    }

    // 启动新一轮
    if (!session_active_) {
      session_active_ = true;
      session_frames_ = 0;
      best_digit_ = -1;
      best_conf_ = 0.0;
      best_x_diff_ = 0.0;
      best_y_diff_ = 0.0;
    }

    // 收集本帧候选
    const double half_w = det_frame_width_ > 0 ? static_cast<double>(det_frame_width_) * 0.5 : 1.0;
    const double half_h = det_frame_height_ > 0 ? static_cast<double>(det_frame_height_) * 0.5 : 1.0;
    std::vector<Candidate> candidates;
    for (const auto & target : msg->targets) {
      for (const auto & roi : target.rois) {
        if (roi.rect.width <= 0 || roi.rect.height <= 0 || roi.confidence < conf_threshold_) {
          continue;
        }
        int digit = parse_digit_label(roi.type);
        if (digit < 0) {
          continue;
        }
        double cx = static_cast<double>(roi.rect.x_offset) +
          static_cast<double>(roi.rect.width) * 0.5;
        double cy = static_cast<double>(roi.rect.y_offset) +
          static_cast<double>(roi.rect.height) * 0.5;
        candidates.push_back(Candidate{digit, static_cast<double>(roi.confidence), cx, cy});
      }
    }

    if (!candidates.empty()) {
      const auto chosen = *std::max_element(
        candidates.begin(), candidates.end(),
        [](const Candidate & a, const Candidate & b) {
          if (a.digit != b.digit) {
            return a.digit < b.digit;
          }
          return a.conf < b.conf;
        });

      double det_x_norm = (half_w > 0.0) ? (chosen.center_x / (half_w * 2.0)) : 0.5;
      double det_y_norm = (half_h > 0.0) ? (chosen.center_y / (half_h * 2.0)) : 0.5;
      det_x_norm = std::clamp(det_x_norm, 0.0, 1.0);
      det_y_norm = std::clamp(det_y_norm, 0.0, 1.0);

      double y_diff = det_y_norm - last_centroid_norm_y_;
      if (std::abs(y_diff) <= y_align_tolerance_) {
        double x_diff = det_x_norm - last_centroid_norm_x_;
        if (chosen.digit > best_digit_ || (chosen.digit == best_digit_ && chosen.conf > best_conf_)) {
          best_digit_ = chosen.digit;
          best_conf_ = chosen.conf;
          best_x_diff_ = x_diff;
          best_y_diff_ = y_diff;
        }
      }
    }

    session_frames_++;

    if (session_frames_ >= frame_window_) {
      trigger_turn(now);
      session_active_ = false;
    }
  }

  void trigger_turn(const rclcpp::Time & now)
  {
    if (best_digit_ < 0) {
      return;
    }

    double yaw = std::clamp(turn_direction_sign_ * best_x_diff_ * yaw_gain_, -max_yaw_rate_, max_yaw_rate_);
    if (yaw == 0.0) {
      return;
    }
    double lin = turn_linear_;
    double dur = turn_duration_;

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = lin;
    cmd.angular.z = yaw;

    turning_cmd_ = cmd;
    turning_end_time_ = now + rclcpp::Duration::from_seconds(dur);
    turn_cooldown_until_ = turning_end_time_ + rclcpp::Duration::from_seconds(turn_cooldown_);
    turning_ = true;

    RCLCPP_INFO(
      get_logger(),
      "[digit turn] best:%d x_diff:%.2f y_diff:%.2f dur:%.2fs lin:%.2f ang:%.2f (frames:%d)",
      best_digit_, best_x_diff_, best_y_diff_, dur, cmd.linear.x, cmd.angular.z, session_frames_);
  }

  void on_timer()
  {
    if (!turning_) {
      return;
    }
    if (this->now() >= turning_end_time_) {
      turning_ = false;
      return;
    }
    cmd_pub_->publish(turning_cmd_);
  }

  // Params
  std::string detection_topic_;
  std::string state_topic_;
  std::string cmd_vel_topic_;
  std::string line_centroid_topic_;
  double conf_threshold_{kDefaultConfThreshold};
  int det_frame_width_{kDefaultDetFrameWidth};
  int det_frame_height_{kDefaultDetFrameHeight};
  int frame_window_{kDefaultFrameWindow};
  double turn_cooldown_{kDefaultTurnCooldown};
  double turn_duration_{kDefaultTurnDuration};
  double turn_linear_{kDefaultTurnLinear};
  double yaw_gain_{kDefaultYawGain};
  double max_yaw_rate_{kDefaultMaxYaw};
  double y_align_tolerance_{kDefaultYAlignTolerance};
  double turn_direction_sign_{kDefaultTurnDirectionSign};
  double centroid_timeout_{kDefaultCentroidTimeout};

  // State
  bool turning_{false};
  geometry_msgs::msg::Twist turning_cmd_;
  rclcpp::Time turning_end_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time turn_cooldown_until_{0, 0, RCL_ROS_TIME};

  bool session_active_{false};
  int session_frames_{0};
  int best_digit_{-1};
  double best_conf_{0.0};
  double best_x_diff_{0.0};
  double best_y_diff_{0.0};

  double last_centroid_norm_x_{0.0};
  double last_centroid_norm_y_{0.0};
  rclcpp::Time last_centroid_time_{0, 0, RCL_ROS_TIME};

  // ROS interfaces
  rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr detection_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr centroid_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectionTurnController>());
  rclcpp::shutdown();
  return 0;
}
