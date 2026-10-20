#include <iostream>
#include <memory>
#include <string>
#include <optional>

#include <termios.h>
#include <unistd.h>
#include <sys/select.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

class StartToggleNode : public rclcpp::Node
{
public:
  StartToggleNode()
  : Node("start_toggle_node")
  , stdin_raw_(setup_stdin_raw())
  {
    topic_ = this->declare_parameter<std::string>("start_topic", "line_follower/start_cmd");
    pub_ = this->create_publisher<std_msgs::msg::Bool>(topic_, rclcpp::QoS(10));
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100), std::bind(&StartToggleNode::on_timer, this));

    RCLCPP_INFO(
      get_logger(),
      "Publishing start signal on '%s' at 10Hz. Press Space to toggle (false<->true).",
      topic_.c_str());
  }

  ~StartToggleNode()
  {
    if (stdin_raw_.active) {
      tcsetattr(STDIN_FILENO, TCSANOW, &stdin_raw_.oldt);
    }
  }

private:
  void on_timer()
  {
    std_msgs::msg::Bool msg;
    msg.data = started_;
    pub_->publish(msg);

    // Non-blocking check: if stdin有空格就切换
    if (_kbhit()) {
      auto c_opt = _read_char();
      if (!c_opt.has_value()) {
        return;
      }
      int c = c_opt.value();
      if (c == ' ') {
        started_ = !started_;
        RCLCPP_INFO(get_logger(), "Start signal toggled to %s", started_ ? "true" : "false");
      } else {
        // 清空其他输入
      }
    }
  }

  // Simple cross-platform kbhit using std::cin.rdbuf()->in_avail()
  bool _kbhit()
  {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    int ret = select(STDIN_FILENO + 1, &rfds, nullptr, nullptr, &tv);
    return ret > 0 && FD_ISSET(STDIN_FILENO, &rfds);
  }

  std::optional<int> _read_char()
  {
    unsigned char c;
    ssize_t n = ::read(STDIN_FILENO, &c, 1);
    if (n == 1) {
      return static_cast<int>(c);
    }
    return std::nullopt;
  }

  struct StdinGuard {
    termios oldt{};
    bool active{false};
  };

  static StdinGuard setup_stdin_raw()
  {
    StdinGuard g;
    if (tcgetattr(STDIN_FILENO, &g.oldt) == 0) {
      termios newt = g.oldt;
      newt.c_lflag &= ~(ICANON | ECHO);
      newt.c_cc[VMIN] = 0;
      newt.c_cc[VTIME] = 0;
      if (tcsetattr(STDIN_FILENO, TCSANOW, &newt) == 0) {
        g.active = true;
      }
    }
    return g;
  }

  std::string topic_;
  bool started_{false};
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  StdinGuard stdin_raw_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StartToggleNode>());
  rclcpp::shutdown();
  return 0;
}
