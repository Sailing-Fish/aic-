#include <memory>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace
{
constexpr int kDefaultWidth = 320;
constexpr int kDefaultHeight = 240;
}

class ImageResizerNode : public rclcpp::Node
{
public:
  ImageResizerNode()
  : Node("image_resizer_node")
  {
    input_topic_ = this->declare_parameter<std::string>("input_topic", "/ascamera/camera_publisher/rgb/image0");
    output_topic_ = this->declare_parameter<std::string>("output_topic", "image_resized");
    target_width_ = this->declare_parameter<int>("target_width", kDefaultWidth);
    target_height_ = this->declare_parameter<int>("target_height", kDefaultHeight);

    image_sub_ = image_transport::create_subscription(
      this,
      input_topic_,
      std::bind(&ImageResizerNode::image_callback, this, std::placeholders::_1),
      "raw");

    image_pub_ = image_transport::create_publisher(this, output_topic_);

    RCLCPP_INFO(
      get_logger(),
      "Subscribing to '%s', publishing resized images (%dx%d) on '%s'",
      input_topic_.c_str(), target_width_, target_height_, output_topic_.c_str());
  }

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    cv::Mat input_bgr;
    try {
      input_bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge exception: %s", e.what());
      return;
    }

    if (input_bgr.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Empty image received, skip");
      return;
    }

    cv::Mat resized;
    cv::resize(input_bgr, resized, cv::Size(target_width_, target_height_));

    cv_bridge::CvImage out_msg;
    out_msg.header = msg->header;
    out_msg.encoding = "bgr8";
    out_msg.image = resized;

    image_pub_.publish(out_msg.toImageMsg());
  }

  std::string input_topic_;
  std::string output_topic_;
  int target_width_{kDefaultWidth};
  int target_height_{kDefaultHeight};

  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageResizerNode>());
  rclcpp::shutdown();
  return 0;
}
