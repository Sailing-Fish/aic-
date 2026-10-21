#include <memory>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

class CmdMuxNode : public rclcpp::Node
{
public:
  CmdMuxNode()
  : Node("cmd_mux_node")
  {
    high_topic_ =
      this->declare_parameter<std::string>("high_topic", "/cmd_vel_detection");

    low_topic_ =
      this->declare_parameter<std::string>("low_topic", "/cmd_vel_follower");

    output_topic_ =
      this->declare_parameter<std::string>("output_topic", "/cmd_vel");

    high_timeout_ =
      this->declare_parameter<double>("high_timeout", 0.5);

    stop_topic_ =
      this->declare_parameter<std::string>("stop_topic", "/cmd_vel_stop");

    stop_timeout_ =
      this->declare_parameter<double>("stop_timeout", 0.5);

    // ------------------------------------------------
    // 关键修改：
    // 使用当前节点的 clock 初始化时间
    // ------------------------------------------------
    last_high_time_ = this->now();
    last_stop_time_ = this->now();

    has_high_ = false;
    has_stop_ = false;

    high_sub_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        high_topic_,
        rclcpp::QoS(10),
        std::bind(
          &CmdMuxNode::high_callback,
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

    stop_sub_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        stop_topic_,
        rclcpp::QoS(10),
        std::bind(
          &CmdMuxNode::stop_callback,
          this,
          std::placeholders::_1));

    pub_ =
      this->create_publisher<geometry_msgs::msg::Twist>(
        output_topic_,
        10);

    RCLCPP_INFO(
      get_logger(),
      "Cmd mux: stop='%s' timeout=%.2f, high='%s' timeout=%.2f, "
      "low='%s', output='%s'",
      stop_topic_.c_str(),
      stop_timeout_,
      high_topic_.c_str(),
      high_timeout_,
      low_topic_.c_str(),
      output_topic_.c_str());
  }

private:

  // ============================================================
  // 高优先级：YOLO 转弯控制
  // ============================================================
  void high_callback(
    const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    const auto now = this->now();

    // 如果最近确实收到过 stop，则 stop 优先
    if (
      has_stop_ &&
      (now - last_stop_time_).seconds() <= stop_timeout_)
    {
      note_source("stop");
      publish(
        std::make_shared<geometry_msgs::msg::Twist>(
          last_stop_msg_));
      return;
    }

    // 保存最新 detection 指令
    last_high_msg_ = *msg;
    last_high_time_ = now;
    has_high_ = true;

    note_source("detection");
    publish(msg);
  }


  // ============================================================
  // 最高优先级：停车
  // ============================================================
  void stop_callback(
    const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    last_stop_msg_ = *msg;
    last_stop_time_ = this->now();
    has_stop_ = true;

    note_source("stop");
    publish(msg);
  }


  // ============================================================
  // 低优先级：普通巡线
  // ============================================================
  void low_callback(
    const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    const auto now = this->now();

    // ------------------------------------------------
    // 第一优先级：STOP
    // ------------------------------------------------
    if (
      has_stop_ &&
      (now - last_stop_time_).seconds() <= stop_timeout_)
    {
      note_source("stop");
      publish(
        std::make_shared<geometry_msgs::msg::Twist>(
          last_stop_msg_));
      return;
    }

    // ------------------------------------------------
    // 第二优先级：YOLO detection
    // ------------------------------------------------
    if (
      has_high_ &&
      (now - last_high_time_).seconds() <= high_timeout_)
    {
      note_source("detection");
      publish(
        std::make_shared<geometry_msgs::msg::Twist>(
          last_high_msg_));
      return;
    }

    // ------------------------------------------------
    // 第三优先级：普通巡线
    // ------------------------------------------------
    note_source("follower");
    publish(msg);
  }


  // [DEC] 仲裁源变化时才打一条，避免每帧刷屏。
  void note_source(const char * src)
  {
    if (last_source_ == src) {
      return;
    }
    last_source_ = src;
    RCLCPP_INFO(get_logger(), "[DEC] mux change active=%s", src);
  }

  void publish(
    const geometry_msgs::msg::Twist::SharedPtr & msg)
  {
    if (pub_) {
      pub_->publish(*msg);
    }
  }


  std::string high_topic_;
  std::string low_topic_;
  std::string output_topic_;
  std::string stop_topic_;

  double high_timeout_{0.5};
  double stop_timeout_{0.5};

  // 时间全部使用节点自身 clock
  rclcpp::Time last_high_time_;
  rclcpp::Time last_stop_time_;

  bool has_high_{false};
  bool has_stop_{false};

  std::string last_source_;   // [DEC] 当前仲裁源，用于只记录切换

  geometry_msgs::msg::Twist last_high_msg_;
  geometry_msgs::msg::Twist last_stop_msg_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr high_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr low_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr stop_sub_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<CmdMuxNode>());

  rclcpp::shutdown();

  return 0;
}
