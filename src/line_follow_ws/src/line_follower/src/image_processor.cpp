#include "line_follower/image_processor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace line_follower
{

namespace
{

double compute_median(std::vector<double> & values, double fallback)
{
  if (values.empty()) {
    return fallback;
  }
  const size_t mid = values.size() / 2;
  std::nth_element(
    values.begin(),
    values.begin() + static_cast<std::ptrdiff_t>(mid),
    values.end());
  return values[mid];
}

int find_lowest_yellow_row(const cv::Mat & yellow_mask, const cv::Rect & box)
{
  const cv::Rect clipped = box & cv::Rect(0, 0, yellow_mask.cols, yellow_mask.rows);
  if (clipped.width <= 0 || clipped.height <= 0) {
    return box.y + box.height - 1;
  }
  const int min_row_pixels = std::max(3, clipped.width / 4);
  for (int y = clipped.y + clipped.height - 1; y >= clipped.y; --y) {
    const cv::Mat row_slice = yellow_mask(cv::Rect(clipped.x, y, clipped.width, 1));
    if (cv::countNonZero(row_slice) >= min_row_pixels) {
      return y;
    }
  }
  return clipped.y + clipped.height - 1;
}

int find_highest_yellow_row(const cv::Mat & yellow_mask, const cv::Rect & box)
{
  const cv::Rect clipped = box & cv::Rect(0, 0, yellow_mask.cols, yellow_mask.rows);
  if (clipped.width <= 0 || clipped.height <= 0) {
    return box.y;
  }
  const int min_row_pixels = std::max(3, clipped.width / 4);
  for (int y = clipped.y; y < clipped.y + clipped.height; ++y) {
    const cv::Mat row_slice = yellow_mask(cv::Rect(clipped.x, y, clipped.width, 1));
    if (cv::countNonZero(row_slice) >= min_row_pixels) {
      return y;
    }
  }
  return clipped.y;
}

cv::Mat build_relative_border_mask(
  const cv::Mat & blurred_bgr,
  const ImageProcessorConfig & cfg)
{
  const int rows = blurred_bgr.rows;
  const int cols = blurred_bgr.cols;

  cv::Mat lab;
  cv::cvtColor(blurred_bgr, lab, cv::COLOR_BGR2Lab);

  std::vector<cv::Mat> lab_channels;
  cv::split(lab, lab_channels);
  const cv::Mat & l_chan = lab_channels[0];

  cv::Mat local_dark;
  cv::adaptiveThreshold(
    l_chan,
    local_dark,
    255,
    cv::ADAPTIVE_THRESH_MEAN_C,
    cv::THRESH_BINARY_INV,
    cfg.black_block_size,
    cfg.black_c_offset);

  cv::Mat non_blue_non_dark(rows, cols, CV_8UC1, cv::Scalar(0));

  for (int y = 0; y < rows; ++y) {
    const cv::Vec3b * bgr_row = blurred_bgr.ptr<cv::Vec3b>(y);
    const cv::Vec3b * lab_row = lab.ptr<cv::Vec3b>(y);
    const unsigned char * dark_row = local_dark.ptr<unsigned char>(y);
    unsigned char * out_row = non_blue_non_dark.ptr<unsigned char>(y);

    for (int x = 0; x < cols; ++x) {
      const int b = static_cast<int>(bgr_row[x][0]);
      const int g = static_cast<int>(bgr_row[x][1]);
      const int r = static_cast<int>(bgr_row[x][2]);
      const int l_val = static_cast<int>(lab_row[x][0]);
      const int b_lab = static_cast<int>(lab_row[x][2]);

      const int max_rg = std::max(g, r);
      const bool blue_by_bgr = (b - max_rg) > cfg.blue_margin && b_lab < 126;
      const bool blue_by_lab = b_lab < cfg.lab_blue_b_max && b > max_rg;
      const bool is_blue = blue_by_bgr || blue_by_lab;

      const bool is_dark =
        (l_val < 45) || (dark_row[x] > 0 && l_val <= cfg.black_v_max);

      if (!is_blue && !is_dark) {
        out_row[x] = 255;
      }
    }
  }

  // Dynamic intra-frame Lab white reference from non-blue, non-dark bottom-center zone
  const int sample_y0 = std::clamp(static_cast<int>(rows * 0.55), 0, rows - 1);
  const int sample_y1 = std::clamp(static_cast<int>(rows * 0.95), sample_y0 + 1, rows);
  const int sample_x0 = std::clamp(static_cast<int>(cols * 0.25), 0, cols - 1);
  const int sample_x1 = std::clamp(static_cast<int>(cols * 0.75), sample_x0 + 1, cols);

  std::vector<double> l_samples;
  std::vector<double> a_samples;
  std::vector<double> b_samples;
  const size_t approx_n =
    static_cast<size_t>(((sample_y1 - sample_y0 + 1) / 2) * ((sample_x1 - sample_x0 + 1) / 2));
  l_samples.reserve(approx_n);
  a_samples.reserve(approx_n);
  b_samples.reserve(approx_n);

  for (int y = sample_y0; y < sample_y1; y += 2) {
    const unsigned char * ok_row = non_blue_non_dark.ptr<unsigned char>(y);
    const cv::Vec3b * lab_row = lab.ptr<cv::Vec3b>(y);
    for (int x = sample_x0; x < sample_x1; x += 2) {
      if (ok_row[x] == 0) {
        continue;
      }
      const int l_val = static_cast<int>(lab_row[x][0]);
      const int a_val = static_cast<int>(lab_row[x][1]);
      const int b_val = static_cast<int>(lab_row[x][2]);
      // Exclude highly saturated red/yellow props from poisoning white reference
      if (std::abs(a_val - 128) > 30 || (b_val - 128) > 40) {
        continue;
      }
      l_samples.push_back(static_cast<double>(l_val));
      a_samples.push_back(static_cast<double>(a_val));
      b_samples.push_back(static_cast<double>(b_val));
    }
  }

  double med_l = 175.0;
  double med_a = 128.0;
  double med_b = 128.0;
  if (l_samples.size() >= 40) {
    med_l = std::clamp(compute_median(l_samples, 175.0), 85.0, 245.0);
    med_a = std::clamp(compute_median(a_samples, 128.0), 112.0, 144.0);
    med_b = std::clamp(compute_median(b_samples, 128.0), 114.0, 152.0);
  }

  const double chroma_r2 = cfg.lab_chroma_radius * cfg.lab_chroma_radius;
  const double min_l = std::max(55.0, med_l - cfg.lab_l_min_offset);

  cv::Mat binary(rows, cols, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < rows; ++y) {
    const unsigned char * ok_row = non_blue_non_dark.ptr<unsigned char>(y);
    const cv::Vec3b * lab_row = lab.ptr<cv::Vec3b>(y);
    unsigned char * bin_row = binary.ptr<unsigned char>(y);

    for (int x = 0; x < cols; ++x) {
      if (ok_row[x] == 0) {
        continue;
      }
      const double l_val = static_cast<double>(lab_row[x][0]);
      if (l_val < min_l) {
        continue;
      }
      const double da = static_cast<double>(lab_row[x][1]) - med_a;
      const double db = static_cast<double>(lab_row[x][2]) - med_b;
      if (da * da + db * db <= chroma_r2) {
        bin_row[x] = 255;
      }
    }
  }

  return binary;
}

}  // namespace

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
  config_.roi_top_ratio = std::clamp(config_.roi_top_ratio, 0.0, 1.0);
  config_.roi_bottom_ratio = std::clamp(config_.roi_bottom_ratio, 0.0, 0.5);
  config_.yellow_roi_bottom_ratio = std::clamp(config_.yellow_roi_bottom_ratio, 0.0, 0.5);

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

  if (config_.lane_seg_mode != "legacy_hsv" &&
      config_.lane_seg_mode != "relative_border" &&
      config_.lane_seg_mode != "hybrid")
  {
    config_.lane_seg_mode = "relative_border";
  }

  config_.blue_margin = std::clamp(config_.blue_margin, 0, 255);
  config_.lab_blue_b_max = std::clamp(config_.lab_blue_b_max, 0, 255);
  config_.black_block_size = std::max(3, config_.black_block_size);
  if (config_.black_block_size % 2 == 0) {
    ++config_.black_block_size;
  }
  config_.black_v_max = std::clamp(config_.black_v_max, 0, 255);
  config_.lab_chroma_radius = std::max(5.0, config_.lab_chroma_radius);
  config_.lab_l_min_offset = std::clamp(config_.lab_l_min_offset, 5.0, 200.0);
  config_.hybrid_white_s_max = std::clamp(config_.hybrid_white_s_max, 0, 255);
  config_.hybrid_white_v_min = std::clamp(config_.hybrid_white_v_min, 0, 255);
  config_.semantic_hole_fill_dilate_px = std::max(0, config_.semantic_hole_fill_dilate_px);

  if (config_.yellow_detect_mode != "geometric_cv" &&
      config_.yellow_detect_mode != "yolo_cv_hybrid")
  {
    config_.yellow_detect_mode = "geometric_cv";
  }
  config_.yellow_min_aspect_ratio = std::max(1.0, config_.yellow_min_aspect_ratio);
  config_.yellow_min_width_ratio = std::clamp(config_.yellow_min_width_ratio, 0.01, 1.0);
  config_.yellow_min_area_px = std::max(1.0, config_.yellow_min_area_px);
  config_.yellow_ahead_strip_height_px = std::max(4, config_.yellow_ahead_strip_height_px);
  config_.yellow_dead_end_blue_ratio = std::clamp(config_.yellow_dead_end_blue_ratio, 0.0, 1.0);
  config_.yellow_dead_end_dark_ratio = std::clamp(config_.yellow_dead_end_dark_ratio, 0.0, 1.0);
  config_.yellow_dead_end_combined_ratio =
    std::clamp(config_.yellow_dead_end_combined_ratio, 0.0, 1.0);

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
  static const std::vector<cv::Rect> kEmptyRects{};
  return segment(frame_bgr, kEmptyRects, kEmptyRects);
}

std::optional<SegmentedFrame> ImageProcessor::segment(
  const cv::Mat & frame_bgr,
  const std::vector<cv::Rect> & fill_white_rects,
  const std::vector<cv::Rect> & yolo_yellow_rects) const
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

  // ==========================================================================
  // 1. Yellow stop line detection on independent ROI [roi_top, yellow_roi_bottom]
  //    Decoupled from lane roi_bottom_ratio so close-range yellow bars &
  //    dead-end-ahead strip stay strictly aligned in full-image pixel space.
  // ==========================================================================
  const int yellow_roi_bottom = std::clamp(
    height - static_cast<int>(height * config_.yellow_roi_bottom_ratio),
    roi_top + 1,
    height);
  const int yellow_roi_height = yellow_roi_bottom - roi_top;
  const cv::Rect yellow_roi_rect(0, roi_top, width, yellow_roi_height);

  bool yellow_is_geometric = false;
  bool yellow_from_yolo_hybrid = false;
  double yellow_bottom_y_full = 0.0;
  double yellow_top_y_full = 0.0;
  double yellow_bar_width_px = 0.0;
  double yellow_center_x_full = 0.0;
  cv::Rect yellow_bbox_full{};

  if (config_.use_yellow_filter) {
    cv::Mat yellow_bgr_blur;
    cv::Mat yellow_hsv;
    if (yellow_roi_bottom == roi_bottom) {
      yellow_bgr_blur = blurred;
      yellow_hsv = hsv;
    } else {
      const cv::Mat yellow_roi = frame_bgr(yellow_roi_rect);
      cv::GaussianBlur(
        yellow_roi,
        yellow_bgr_blur,
        cv::Size(config_.gaussian_kernel, config_.gaussian_kernel),
        0);
      cv::cvtColor(yellow_bgr_blur, yellow_hsv, cv::COLOR_BGR2HSV);
    }

    cv::Mat yellow_mask;
    const cv::Scalar lower_yellow(
      static_cast<double>(config_.yellow_h_low),
      static_cast<double>(config_.yellow_s_min),
      static_cast<double>(config_.yellow_v_min));
    const cv::Scalar upper_yellow(
      static_cast<double>(config_.yellow_h_high),
      255.0,
      255.0);

    cv::inRange(yellow_hsv, lower_yellow, upper_yellow, yellow_mask);

    // Anti-shadow chroma dominance gate (R-B >= 18 && G-B >= 12):
    // Allows yellow_v_min down to ~45-55 so yellow tape inside the car's own shadow (V=55..75)
    // is still captured, while strictly rejecting neutral gray/dark shadows & dim white paper!
    for (int y = 0; y < yellow_mask.rows; ++y) {
      uint8_t * m_row = yellow_mask.ptr<uint8_t>(y);
      const cv::Vec3b * bgr_row = yellow_bgr_blur.ptr<cv::Vec3b>(y);
      for (int x = 0; x < yellow_mask.cols; ++x) {
        if (m_row[x] == 0) {
          continue;
        }
        const int b = static_cast<int>(bgr_row[x][0]);
        const int g = static_cast<int>(bgr_row[x][1]);
        const int r = static_cast<int>(bgr_row[x][2]);
        if ((r - b) < 18 || (g - b) < 12) {
          m_row[x] = 0;
        }
      }
    }

    const int yellow_pixels = cv::countNonZero(yellow_mask);

    // Clean mask for geometric bar extraction: wide horizontal close (15x3) bridges small
    // lighting/shadow gaps across the horizontal yellow stop strip without inflating height.
    cv::Mat yellow_clean;
    const cv::Mat open_k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    const cv::Mat close_k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3));
    cv::morphologyEx(yellow_mask, yellow_clean, cv::MORPH_OPEN, open_k, cv::Point(-1, -1), 1);
    cv::morphologyEx(yellow_clean, yellow_clean, cv::MORPH_CLOSE, close_k, cv::Point(-1, -1), 1);

    // YOLO-guided CV refinement (YOLO selectivity + CV sub-row precision)
    if (config_.yellow_detect_mode == "yolo_cv_hybrid" && !yolo_yellow_rects.empty()) {
      int best_yolo_idx = -1;
      int best_yolo_bottom = -1;
      for (size_t i = 0; i < yolo_yellow_rects.size(); ++i) {
        const cv::Rect & r = yolo_yellow_rects[i];
        if (r.width <= 0 || r.height <= 0) {
          continue;
        }
        const cv::Rect inter = r & yellow_roi_rect;
        if (inter.width <= 0 || inter.height <= 0) {
          continue;
        }
        const int bot = r.y + r.height;
        if (bot > best_yolo_bottom) {
          best_yolo_bottom = bot;
          best_yolo_idx = static_cast<int>(i);
        }
      }

      if (best_yolo_idx >= 0) {
        const cv::Rect & yr = yolo_yellow_rects[static_cast<size_t>(best_yolo_idx)];
        const int px0 = std::clamp(yr.x - 4, 0, width - 1);
        const int py0 = std::clamp(yr.y - 6, roi_top, yellow_roi_bottom - 1);
        const int px1 = std::clamp(yr.x + yr.width + 4, px0 + 1, width);
        const int py1 = std::clamp(yr.y + yr.height + 6, py0 + 1, yellow_roi_bottom);
        const cv::Rect padded_full(px0, py0, px1 - px0, py1 - py0);

        const cv::Rect local_box(
          padded_full.x,
          padded_full.y - roi_top,
          padded_full.width,
          padded_full.height);
        const cv::Rect local_clipped = local_box & cv::Rect(0, 0, width, yellow_roi_height);

        yellow_from_yolo_hybrid = true;
        yellow_is_geometric = true;

        if (local_clipped.width > 0 && local_clipped.height > 0) {
          const int local_nz = cv::countNonZero(yellow_clean(local_clipped));
          const int min_local = std::max(20, static_cast<int>(0.04 * local_clipped.area()));
          if (local_nz >= min_local) {
            const int bot_roi = find_lowest_yellow_row(yellow_clean, local_clipped);
            const int top_roi = find_highest_yellow_row(yellow_clean, local_clipped);
            yellow_bottom_y_full = static_cast<double>(roi_top + bot_roi);
            yellow_top_y_full = static_cast<double>(roi_top + top_roi);
            yellow_bar_width_px = static_cast<double>(yr.width);
            yellow_center_x_full = static_cast<double>(yr.x) + 0.5 * static_cast<double>(yr.width);
            yellow_bbox_full = cv::Rect(
              yr.x,
              roi_top + top_roi,
              yr.width,
              std::max(1, bot_roi - top_roi + 1));
          } else {
            yellow_bottom_y_full = static_cast<double>(yr.y + yr.height - 1);
            yellow_top_y_full = static_cast<double>(yr.y);
            yellow_bar_width_px = static_cast<double>(yr.width);
            yellow_center_x_full = static_cast<double>(yr.x) + 0.5 * static_cast<double>(yr.width);
            yellow_bbox_full = yr;
          }
        }
      }
    }

    // Pure CV geometric horizontal yellow bar detection
    if (!yellow_from_yolo_hybrid && yellow_pixels > 0) {
      std::vector<std::vector<cv::Point>> y_contours;
      cv::findContours(yellow_clean, y_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

      int best_bar_idx = -1;
      int best_bar_bottom = -1;
      double best_bar_area = 0.0;
      const double min_w = config_.yellow_min_width_ratio * static_cast<double>(width);

      for (size_t i = 0; i < y_contours.size(); ++i) {
        const double area = cv::contourArea(y_contours[i]);
        if (area < config_.yellow_min_area_px) {
          continue;
        }
        const cv::Rect box = cv::boundingRect(y_contours[i]);
        if (box.width <= 0 || box.height <= 0) {
          continue;
        }
        const double aspect =
          static_cast<double>(box.width) / static_cast<double>(box.height);
        if (aspect < config_.yellow_min_aspect_ratio ||
            static_cast<double>(box.width) < min_w)
        {
          continue;
        }
        const int bot = box.y + box.height;
        if (bot > best_bar_bottom || (bot == best_bar_bottom && area > best_bar_area)) {
          best_bar_bottom = bot;
          best_bar_area = area;
          best_bar_idx = static_cast<int>(i);
        }
      }

      if (best_bar_idx >= 0) {
        const cv::Rect box = cv::boundingRect(y_contours[static_cast<size_t>(best_bar_idx)]);
        const int bot_roi = find_lowest_yellow_row(yellow_clean, box);
        const int top_roi = find_highest_yellow_row(yellow_clean, box);
        yellow_is_geometric = true;
        yellow_bottom_y_full = static_cast<double>(roi_top + bot_roi);
        yellow_top_y_full = static_cast<double>(roi_top + top_roi);
        yellow_bar_width_px = static_cast<double>(box.width);
        yellow_center_x_full = static_cast<double>(box.x) + 0.5 * static_cast<double>(box.width);
        yellow_bbox_full = cv::Rect(
          box.x,
          roi_top + top_roi,
          box.width,
          std::max(1, bot_roi - top_roi + 1));
      }
    }
  }

  const bool yellow_line_detected = (config_.yellow_detect_mode == "yolo_cv_hybrid") ?
    (yellow_from_yolo_hybrid || yellow_is_geometric) :
    yellow_is_geometric;

  // ==========================================================================
  // 2. Dead-end-ahead check above yellow bar (distinguishes END vs random straight yellow)
  // ==========================================================================
  double yellow_ahead_blue_ratio = 0.0;
  double yellow_ahead_dark_ratio = 0.0;
  bool yellow_has_dead_end_ahead = false;

  if (yellow_is_geometric && yellow_bbox_full.width > 0) {
    const int strip_bot = std::clamp(yellow_bbox_full.y - 2, 0, height);
    const int strip_top = std::clamp(
      strip_bot - config_.yellow_ahead_strip_height_px,
      0,
      strip_bot);
    const int strip_x0 = std::clamp(yellow_bbox_full.x, 0, width - 1);
    const int strip_x1 = std::clamp(
      yellow_bbox_full.x + yellow_bbox_full.width,
      strip_x0 + 1,
      width);

    if (strip_bot > strip_top && strip_x1 > strip_x0) {
      const cv::Rect strip_rect(strip_x0, strip_top, strip_x1 - strip_x0, strip_bot - strip_top);
      const cv::Mat strip_bgr = frame_bgr(strip_rect);
      cv::Mat strip_lab;
      cv::cvtColor(strip_bgr, strip_lab, cv::COLOR_BGR2Lab);

      int blue_cnt = 0;
      int dark_cnt = 0;
      const int total_px = strip_bgr.rows * strip_bgr.cols;

      for (int y = 0; y < strip_bgr.rows; ++y) {
        const cv::Vec3b * bgr_row = strip_bgr.ptr<cv::Vec3b>(y);
        const cv::Vec3b * lab_row = strip_lab.ptr<cv::Vec3b>(y);
        for (int x = 0; x < strip_bgr.cols; ++x) {
          const int b = static_cast<int>(bgr_row[x][0]);
          const int g = static_cast<int>(bgr_row[x][1]);
          const int r = static_cast<int>(bgr_row[x][2]);
          const int l_val = static_cast<int>(lab_row[x][0]);
          const int b_lab = static_cast<int>(lab_row[x][2]);

          const int max_rg = std::max(g, r);
          const bool is_blue =
            ((b - max_rg) > config_.blue_margin && b_lab < 126) ||
            (b_lab < config_.lab_blue_b_max && b > max_rg);
          if (is_blue) {
            ++blue_cnt;
          } else if (l_val <= std::min(90, config_.black_v_max)) {
            ++dark_cnt;
          }
        }
      }

      if (total_px > 0) {
        yellow_ahead_blue_ratio = static_cast<double>(blue_cnt) / static_cast<double>(total_px);
        yellow_ahead_dark_ratio = static_cast<double>(dark_cnt) / static_cast<double>(total_px);
        const double combined = yellow_ahead_blue_ratio + yellow_ahead_dark_ratio;
        yellow_has_dead_end_ahead =
          (yellow_ahead_blue_ratio >= config_.yellow_dead_end_blue_ratio) ||
          (combined >= config_.yellow_dead_end_combined_ratio &&
           yellow_ahead_dark_ratio >= config_.yellow_dead_end_dark_ratio);
      }
    }
  }

  // ==========================================================================
  // 3. Lane binary mask segmentation (pluggable: relative_border | hybrid | legacy_hsv)
  // ==========================================================================
  cv::Mat binary;
  if (config_.lane_seg_mode == "relative_border") {
    binary = build_relative_border_mask(blurred, config_);
  } else if (config_.lane_seg_mode == "hybrid") {
    cv::Mat rel_mask = build_relative_border_mask(blurred, config_);
    cv::Mat relaxed_hsv;
    const cv::Scalar lower_relaxed(
      static_cast<double>(config_.white_h_low),
      0.0,
      static_cast<double>(config_.hybrid_white_v_min));
    const cv::Scalar upper_relaxed(
      static_cast<double>(config_.white_h_high),
      static_cast<double>(config_.hybrid_white_s_max),
      255.0);
    cv::inRange(hsv, lower_relaxed, upper_relaxed, relaxed_hsv);
    cv::bitwise_and(rel_mask, relaxed_hsv, binary);
  } else {
    // Legacy HSV emergency fallback mode
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
  }

  cv::morphologyEx(
    binary,
    binary,
    cv::MORPH_OPEN,
    morph_element_,
    cv::Point(-1, -1),
    1);

  // ==========================================================================
  // 4. Semantic hole-filling (Bonus digit cards & yellow bars painted white)
  // ==========================================================================
  if (config_.enable_semantic_hole_fill) {
    const int pad = config_.semantic_hole_fill_dilate_px;
    auto paint_full_rect = [&](const cv::Rect & r_full) {
        if (r_full.width <= 0 || r_full.height <= 0) {
          return;
        }
        const cv::Rect padded(
          r_full.x - pad,
          r_full.y - pad,
          r_full.width + 2 * pad,
          r_full.height + 2 * pad);
        const cv::Rect inter = padded & roi_rect;
        if (inter.width <= 0 || inter.height <= 0) {
          return;
        }
        const cv::Rect local_rect(
          inter.x - roi_rect.x,
          inter.y - roi_rect.y,
          inter.width,
          inter.height);
        cv::rectangle(binary, local_rect, cv::Scalar(255), cv::FILLED);
      };

    for (const auto & rect : fill_white_rects) {
      paint_full_rect(rect);
    }
    if (yellow_is_geometric && yellow_bbox_full.width > 0) {
      paint_full_rect(yellow_bbox_full);
    }

    // Fill small closed interior holes completely surrounded by white lane
    std::vector<std::vector<cv::Point>> hole_contours;
    std::vector<cv::Vec4i> hole_hierarchy;
    cv::findContours(binary, hole_contours, hole_hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    const double max_hole_area = 0.06 * static_cast<double>(binary.cols * binary.rows);
    for (size_t i = 0; i < hole_contours.size(); ++i) {
      if (hole_hierarchy[i][3] < 0) {
        continue;
      }
      const double a = cv::contourArea(hole_contours[i]);
      if (a <= 4.0 || a > max_hole_area) {
        continue;
      }
      const cv::Rect hb = cv::boundingRect(hole_contours[i]);
      if (hb.x <= 1 || hb.y <= 1 ||
        (hb.x + hb.width) >= (binary.cols - 1) ||
        (hb.y + hb.height) >= (binary.rows - 1))
      {
        continue;
      }
      cv::drawContours(binary, hole_contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
    }
  }

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
  out.yellow_is_geometric = yellow_is_geometric;
  out.yellow_bottom_y_full = yellow_bottom_y_full;
  out.yellow_top_y_full = yellow_top_y_full;
  out.yellow_bar_width_px = yellow_bar_width_px;
  out.yellow_center_x_full = yellow_center_x_full;
  out.yellow_bbox_full = yellow_bbox_full;
  out.yellow_from_yolo_hybrid = yellow_from_yolo_hybrid;
  out.yellow_ahead_blue_ratio = yellow_ahead_blue_ratio;
  out.yellow_ahead_dark_ratio = yellow_ahead_dark_ratio;
  out.yellow_has_dead_end_ahead = yellow_has_dead_end_ahead;
  out.active_lane_seg_mode = config_.lane_seg_mode;

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
