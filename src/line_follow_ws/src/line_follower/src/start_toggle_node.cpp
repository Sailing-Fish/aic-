#include <cstdio>
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
    set_topic_ = this->declare_parameter<std::string>("start_set_topic", "line_follower/start_set");
    pub_ = this->create_publisher<std_msgs::msg::Bool>(topic_, rclcpp::QoS(10));
    set_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      set_topic_, rclcpp::QoS(10),
      std::bind(&StartToggleNode::on_set_cmd, this, std::placeholders::_1));
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100), std::bind(&StartToggleNode::on_timer, this));

    RCLCPP_INFO(
      get_logger(),
      "Publishing start signal on '%s' at 10Hz (tty_raw=%d, remote_set='%s'). Press Space to toggle (false<->true).",
      topic_.c_str(),
      stdin_raw_.active ? 1 : 0,
      set_topic_.c_str());
    std::fflush(stdout);
  }

  ~StartToggleNode()
  {
    if (stdin_raw_.active) {
      tcsetattr(STDIN_FILENO, TCSANOW, &stdin_raw_.oldt);
    }
  }

private:
  void on_set_cmd(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    const bool desired = msg->data;
    if (desired != started_) {
      started_ = desired;
      RCLCPP_INFO(
        get_logger(),
        "[DEC] start_toggle set by=panel started=%s",
        started_ ? "true" : "false");
      std::fflush(stdout);
    }
    std_msgs::msg::Bool out;
    out.data = started_;
    pub_->publish(out);
  }

  void on_timer()
  {
    std_msgs::msg::Bool msg;
    msg.data = started_;
    pub_->publish(msg);

    // Non-blocking check: drain buffered stdin, toggle on Space, exit cleanly on EOF (window closed)
    while (_kbhit()) {
      auto c_opt = _read_char();
      if (!c_opt.has_value()) {
        if (stdin_eof_) {
          RCLCPP_WARN(get_logger(), "stdin EOF detected (terminal closed); shutting down start_toggle_node.");
          rclcpp::shutdown();
        }
        return;
      }
      int c = c_opt.value();
      if (c == ' ') {
        started_ = !started_;
        RCLCPP_INFO(get_logger(), "Start signal toggled to %s", started_ ? "true" : "false");
        std::fflush(stdout);
      } else if (c == '\r' || c == '\n') {
        RCLCPP_INFO(
          get_logger(),
          "Received Enter (not Space); current start=%s. Switch Windows IME to ENG and press Space.",
          started_ ? "true" : "false");
        std::fflush(stdout);
      }
    }
  }

  // Simple cross-platform kbhit using select()
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
    if (n == 0) {
      stdin_eof_ = true;
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
  std::string set_topic_;
  bool started_{false};
  bool stdin_eof_{false};
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr set_sub_;
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
