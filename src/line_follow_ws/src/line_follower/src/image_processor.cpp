#include "line_follower/image_processor.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace line_follower
{

ImageProcessor::ImageProcessor()
{
  set_config(ImageProcessorConfig{});
}

ImageProcessor::ImageProcessor(const ImageProcessorConfig & config)
{
  set_config(config);
}

void ImageProcessor::set_config(const ImageProcessorConfig & config)
{
  config_ = config;

  if (config_.gaussian_kernel <= 0) {
    config_.gaussian_kernel = 1;
  }
  if (config_.gaussian_kernel % 2 == 0) {
    ++config_.gaussian_kernel;
  }

  config_.morph_kernel = std::max(1, config_.morph_kernel);
  config_.morph_iterations = std::max(1, config_.morph_iterations);

  config_.white_h_low = std::clamp(config_.white_h_low, 0, 180);
  config_.white_h_high = std::clamp(config_.white_h_high, 0, 180);
  if (config_.white_h_low > config_.white_h_high) {
    std::swap(config_.white_h_low, config_.white_h_high);
  }
  config_.white_s_max = std::clamp(config_.white_s_max, 0, 255);
  config_.white_v_min = std::clamp(config_.white_v_min, 0, 255);

  config_.yellow_h_low = std::clamp(config_.yellow_h_low, 0, 180);
  config_.yellow_h_high = std::clamp(config_.yellow_h_high, 0, 180);
  if (config_.yellow_h_low > config_.yellow_h_high) {
    std::swap(config_.yellow_h_low, config_.yellow_h_high);
  }
  config_.yellow_s_min = std::clamp(config_.yellow_s_min, 0, 255);
  config_.yellow_v_min = std::clamp(config_.yellow_v_min, 0, 255);

  morph_element_ = cv::getStructuringElement(
    cv::MORPH_RECT,
    cv::Size(config_.morph_kernel, config_.morph_kernel));
}

const ImageProcessorConfig & ImageProcessor::config() const
{
  return config_;
}

std::optional<SegmentedFrame> ImageProcessor::segment(const cv::Mat & frame_bgr) const
{
  if (frame_bgr.empty() || frame_bgr.type() != CV_8UC3) {
    return std::nullopt;
  }

  const int height = frame_bgr.rows;
  const int width = frame_bgr.cols;

  const int roi_top = std::clamp(
    static_cast<int>(height * (1.0 - config_.roi_top_ratio)),
    0,
    height - 1);

  const int roi_bottom = std::clamp(
    height - static_cast<int>(height * config_.roi_bottom_ratio),
    roi_top + 1,
    height);

  const int roi_height = roi_bottom - roi_top;
  const cv::Rect roi_rect(0, roi_top, width, roi_height);
  const cv::Mat roi = frame_bgr(roi_rect);

  cv::Mat blurred;
  cv::GaussianBlur(
    roi,
    blurred,
    cv::Size(config_.gaussian_kernel, config_.gaussian_kernel),
    0);

  cv::Mat hsv;
  cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

  bool yellow_line_detected = false;
  if (config_.use_yellow_filter) {
    cv::Mat yellow_mask;
    const cv::Scalar lower_yellow(
      static_cast<double>(config_.yellow_h_low),
      static_cast<double>(config_.yellow_s_min),
      static_cast<double>(config_.yellow_v_min));
    const cv::Scalar upper_yellow(
      static_cast<double>(config_.yellow_h_high),
      255.0,
      255.0);

    cv::inRange(hsv, lower_yellow, upper_yellow, yellow_mask);

    const int yellow_pixels = cv::countNonZero(yellow_mask);
    const double yellow_area_ratio =
      yellow_mask.total() > 0 ?
      static_cast<double>(yellow_pixels) / static_cast<double>(yellow_mask.total()) :
      0.0;

    yellow_line_detected = yellow_area_ratio >= config_.yellow_area_ratio_threshold;
  }

  cv::Mat binary;
  if (config_.use_white_filter) {
    const cv::Scalar lower_white(
      static_cast<double>(config_.white_h_low),
      0.0,
      static_cast<double>(config_.white_v_min));
    const cv::Scalar upper_white(
      static_cast<double>(config_.white_h_high),
      static_cast<double>(config_.white_s_max),
      255.0);

    cv::inRange(hsv, lower_white, upper_white, binary);
  } else {
    cv::Mat gray;
    cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);

    const int threshold_type =
      config_.invert_binary ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;

    cv::threshold(
      gray,
      binary,
      config_.binary_threshold,
      255,
      threshold_type);
  }

  cv::morphologyEx(
    binary,
    binary,
    cv::MORPH_OPEN,
    morph_element_,
    cv::Point(-1, -1),
    1);

  cv::morphologyEx(
    binary,
    binary,
    cv::MORPH_CLOSE,
    morph_element_,
    cv::Point(-1, -1),
    config_.morph_iterations);

  SegmentedFrame out{};
  out.roi_bgr = roi.clone();
  out.binary_mask = binary.clone();
  out.roi_top = roi_rect.y;
  out.roi_bottom = roi_rect.y + roi_rect.height;
  out.frame_width = width;
  out.frame_height = height;
  out.yellow_line_detected = yellow_line_detected;
  out.contour_valid = false;
  out.contour_centroid = cv::Point2f(
    static_cast<float>(width) * 0.5F,
    static_cast<float>(roi_height) * 0.70F);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(
    binary,
    contours,
    cv::RETR_EXTERNAL,
    cv::CHAIN_APPROX_SIMPLE);

  if (!contours.empty()) {
    double max_area = 0.0;
    size_t best_index = 0;

    for (size_t i = 0; i < contours.size(); ++i) {
      const double area = cv::contourArea(contours[i]);
      if (area > max_area) {
        max_area = area;
        best_index = i;
      }
    }

    if (max_area >= config_.min_contour_area) {
      out.contour_valid = true;
      out.bounding_box = cv::boundingRect(contours[best_index]);

      const auto moments = cv::moments(contours[best_index]);
      if (moments.m00 != 0.0) {
        out.contour_centroid.x = static_cast<float>(moments.m10 / moments.m00);
        out.contour_centroid.y = static_cast<float>(moments.m01 / moments.m00);
      } else {
        out.contour_centroid.x =
          static_cast<float>(out.bounding_box.x + out.bounding_box.width / 2.0);
        out.contour_centroid.y =
          static_cast<float>(out.bounding_box.y + out.bounding_box.height / 2.0);
      }
    }
  }

  return out;
}

}  // namespace line_follower
