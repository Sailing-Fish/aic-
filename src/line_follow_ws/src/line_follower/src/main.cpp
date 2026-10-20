#include "line_follower/line_follower_node.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<line_follower::LineFollowerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
