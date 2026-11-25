#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

class CmdMuxNode : public rclcpp::Node
{
public:
  CmdMuxNode()
  : Node("cmd_mux_node")
  {
    stop_topic_ =
      this->declare_parameter<std::string>("stop_topic", "/cmd_vel_stop");

    low_topic_ =
      this->declare_parameter<std::string>("low_topic", "/cmd_vel_follower");

    output_topic_ =
      this->declare_parameter<std::string>("output_topic", "/cmd_vel");

    heartbeat_topic_ =
      this->declare_parameter<std::string>("heartbeat_topic", "/cmd_mux/heartbeat");

    stop_timeout_ =
      std::max(0.0, this->declare_parameter<double>("stop_timeout", 0.5));

    low_timeout_ =
      std::max(0.0, this->declare_parameter<double>("low_timeout", 0.35));

    watchdog_rate_hz_ =
      std::max(1.0, this->declare_parameter<double>("watchdog_rate_hz", 20.0));

    const auto now = this->now();
    last_stop_time_ = now;
    last_low_time_ = now;

    stop_sub_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        stop_topic_,
        rclcpp::QoS(10),
        std::bind(
          &CmdMuxNode::stop_callback,
          this,
          std::placeholders::_1));

    low_sub_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        low_topic_,
        rclcpp::QoS(10),
        std::bind(
          &CmdMuxNode::low_callback,
          this,
          std::placeholders::_1));

    pub_ =
      this->create_publisher<geometry_msgs::msg::Twist>(
        output_topic_,
        10);

    heartbeat_pub_ =
      this->create_publisher<std_msgs::msg::Bool>(
        heartbeat_topic_,
        10);

    const auto period_ms = static_cast<int64_t>(
      std::max(1.0, 1000.0 / watchdog_rate_hz_));
    watchdog_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(period_ms),
      std::bind(&CmdMuxNode::watchdog_callback, this));

    RCLCPP_INFO(
      get_logger(),
      "Cmd mux (2-ch): stop='%s'(%.2fs) > low='%s'(%.2fs) -> output='%s' "
      "heartbeat='%s' watchdog=%.1fHz",
      stop_topic_.c_str(),
      stop_timeout_,
      low_topic_.c_str(),
      low_timeout_,
      output_topic_.c_str(),
      heartbeat_topic_.c_str(),
      watchdog_rate_hz_);
  }

private:
  bool stop_is_active(const rclcpp::Time & now) const
  {
    return has_stop_ && (now - last_stop_time_).seconds() <= stop_timeout_;
  }

  bool low_is_active(const rclcpp::Time & now) const
  {
    return has_low_ && (now - last_low_time_).seconds() <= low_timeout_;
  }

  // ============================================================
  // 最高优先级：停车 (/cmd_vel_stop)
  // ============================================================
  void stop_callback(
    const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    const auto now = this->now();
    last_stop_msg_ = *msg;
    last_stop_time_ = now;
    has_stop_ = true;

    note_source("stop");
    publish_twist(last_stop_msg_);
  }

  // ============================================================
  // 常规优先级：统一走廊巡线 (/cmd_vel_follower)
  // ============================================================
  void low_callback(
    const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    const auto now = this->now();
    last_low_msg_ = *msg;
    last_low_time_ = now;
    has_low_ = true;

    if (stop_is_active(now)) {
      note_source("stop");
      publish_twist(last_stop_msg_);
      return;
    }

    note_source("follower");
    publish_twist(*msg);
  }

  // ============================================================
  // 周期看门狗 + 心跳（20Hz）：
  // - 始终发布 /cmd_mux/heartbeat 供 safety_node 监测 mux 存活
  // - 当输入源超时时主动发零速，防止 STM32 锁存末速度
  // ============================================================
  void watchdog_callback()
  {
    if (heartbeat_pub_) {
      std_msgs::msg::Bool hb;
      hb.data = true;
      heartbeat_pub_->publish(hb);
    }

    const auto now = this->now();

    if (stop_is_active(now)) {
      note_source("stop");
      publish_twist(last_stop_msg_);
      return;
    }

    if (low_is_active(now)) {
      return;
    }

    note_source("watchdog_zero");
    publish_zero();
  }

  void note_source(const char * src)
  {
    if (last_source_ == src) {
      return;
    }
    last_source_ = src;
    RCLCPP_INFO(get_logger(), "[DEC] mux change active=%s", src);
  }

  void publish_twist(const geometry_msgs::msg::Twist & msg)
  {
    if (pub_) {
      pub_->publish(msg);
    }
  }

  void publish_zero()
  {
    geometry_msgs::msg::Twist zero;
    publish_twist(zero);
  }

  std::string stop_topic_;
  std::string low_topic_;
  std::string output_topic_;
  std::string heartbeat_topic_;

  double stop_timeout_{0.5};
  double low_timeout_{0.35};
  double watchdog_rate_hz_{20.0};

  rclcpp::Time last_stop_time_;
  rclcpp::Time last_low_time_;

  bool has_stop_{false};
  bool has_low_{false};

  std::string last_source_;

  geometry_msgs::msg::Twist last_stop_msg_;
  geometry_msgs::msg::Twist last_low_msg_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr stop_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr low_sub_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr heartbeat_pub_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdMuxNode>());
  rclcpp::shutdown();
  return 0;
}
