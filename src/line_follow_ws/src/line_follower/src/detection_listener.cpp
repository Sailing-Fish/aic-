#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <ai_msgs/msg/perception_targets.hpp>

namespace
{
constexpr char kTopicName[] = "hobot_dnn_detection";
}

class DetectionListener : public rclcpp::Node
{
public:
  DetectionListener()
  : Node("detection_listener")
  {
    sub_ = this->create_subscription<ai_msgs::msg::PerceptionTargets>(
      kTopicName,
      rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(),
      std::bind(&DetectionListener::callback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Subscribed to '%s' for YOLO detections", kTopicName);
  }

private:
  void callback(const ai_msgs::msg::PerceptionTargets::SharedPtr msg)
  {
    // 目前仅打印简要信息，为后续处理做准备
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Detections: %zu targets, fps: %d", msg->targets.size(), msg->fps);
  }

  rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr sub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectionListener>());
  rclcpp::shutdown();
  return 0;
}
