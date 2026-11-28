#include "line_follower/bev_mapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv2/imgproc.hpp>

namespace line_follower
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

void stamp_bev_circle(
  std::array<uint8_t, kBevCells> & grid,
  double d_center,
  double y_center,
  double radius_m,
  uint8_t cell_val)
{
  const int r_cells = static_cast<int>(std::ceil(radius_m / kBevRes));
  int cr = 0, cc = 0;
  if (!BevDwaPlanner::metric_to_cell(d_center, y_center, cr, cc)) {
    // Allow partial overlap even if center is slightly outside
    cr = static_cast<int>(std::floor((d_center - kBevDMin) / kBevRes));
    cc = static_cast<int>(std::floor((kBevYMax - y_center) / kBevRes));
  }
  const double r2 = radius_m * radius_m;
  for (int dr = -r_cells; dr <= r_cells; ++dr) {
    const int r = cr + dr;
    if (r < 0 || r >= kBevRows) {
      continue;
    }
    const double dm = (kBevDMin + (static_cast<double>(r) + 0.5) * kBevRes) - d_center;
    for (int dc = -r_cells; dc <= r_cells; ++dc) {
      const int c = cc + dc;
      if (c < 0 || c >= kBevCols) {
        continue;
      }
      const double ym = (kBevYMax - (static_cast<double>(c) + 0.5) * kBevRes) - y_center;
      if (dm * dm + ym * ym <= r2) {
        const size_t idx = static_cast<size_t>(r * kBevCols + c);
        // Never downgrade +30s no_entry to +10s cone
        if (grid[idx] != BEV_OBS_NO_ENTRY) {
          grid[idx] = cell_val;
        }
      }
    }
  }
}

}  // namespace

BevMapper::BevMapper(const BevMapperConfig & config)
{
  set_config(config);
}

void BevMapper::set_config(const BevMapperConfig & config)
{
  config_ = config;
}

void BevMapper::ensure_lut(const GroundProjector & projector, int img_w, int img_h)
{
  const auto & pcfg = projector.config();
  const bool same_h = (pcfg.homography_matrix == lut_h_);
  if (lut_ready_ && img_w == lut_img_w_ && img_h == lut_img_h_ &&
    pcfg.use_homography == lut_use_homography_ && same_h)
  {
    return;
  }

  lut_img_w_ = img_w;
  lut_img_h_ = img_h;
  lut_use_homography_ = pcfg.use_homography;
  lut_h_ = pcfg.homography_matrix;

  for (int r = 0; r < kBevRows; ++r) {
    for (int c = 0; c < kBevCols; ++c) {
      const size_t idx = static_cast<size_t>(r * kBevCols + c);
      double d_m = 0.0, y_m = 0.0;
      BevDwaPlanner::cell_to_metric(r, c, d_m, y_m);
      double u = -1.0, v = -1.0;
      if (d_m >= 0.01 && projector.ground_to_pixel(d_m, y_m, u, v)) {
        const int ui = static_cast<int>(std::round(u));
        const int vi = static_cast<int>(std::round(v));
        if (ui >= 0 && ui < img_w && vi >= 0 && vi < img_h) {
          lut_u_[idx] = ui;
          lut_v_[idx] = vi;
          continue;
        }
      }
      lut_u_[idx] = -1;
      lut_v_[idx] = -1;
    }
  }
  lut_ready_ = true;
}

void BevMapper::build_live_grid(
  const cv::Mat & frame_bgr,
  const cv::Mat & roi_white_mask,
  int roi_top,
  int roi_bottom,
  const GroundProjector & projector,
  std::array<uint8_t, kBevCells> & out_grid,
  std::vector<BevRedOverlayBox> & out_red_boxes)
{
  out_grid.fill(BEV_UNKNOWN);
  out_red_boxes.clear();

  if (frame_bgr.empty() || frame_bgr.cols < 16 || frame_bgr.rows < 16) {
    return;
  }

  const int img_w = frame_bgr.cols;
  const int img_h = frame_bgr.rows;
  ensure_lut(projector, img_w, img_h);

  const int r_top = std::clamp(roi_top, 0, img_h - 1);
  const int r_bot = std::clamp(roi_bottom, r_top + 1, img_h);
  const int roi_h = r_bot - r_top;

  // 1) Classify blue mat & build drivable road mask (white + non-blue dark digit ink inside road!)
  //    Using full lower image from r_top to img_h so BEV sees all the way down to d=0.03m!
  const int full_roi_h = img_h - r_top;
  const cv::Mat lower_bgr = frame_bgr(cv::Rect(0, r_top, img_w, full_roi_h));
  cv::Mat lower_lab;
  cv::cvtColor(lower_bgr, lower_lab, cv::COLOR_BGR2Lab);
  cv::Mat lower_hsv;
  cv::cvtColor(lower_bgr, lower_hsv, cv::COLOR_BGR2HSV);

  cv::Mat blue_mask(full_roi_h, img_w, CV_8UC1, cv::Scalar(0));
  cv::Mat white_full(full_roi_h, img_w, CV_8UC1, cv::Scalar(0));

  for (int y = 0; y < full_roi_h; ++y) {
    const cv::Vec3b * bgr_row = lower_bgr.ptr<cv::Vec3b>(y);
    const cv::Vec3b * lab_row = lower_lab.ptr<cv::Vec3b>(y);
    uint8_t * blue_row = blue_mask.ptr<uint8_t>(y);
    uint8_t * wf_row = white_full.ptr<uint8_t>(y);
    const uint8_t * seg_row =
      (!roi_white_mask.empty() && y < roi_h && roi_white_mask.cols == img_w) ?
      roi_white_mask.ptr<uint8_t>(y) : nullptr;

    for (int x = 0; x < img_w; ++x) {
      const int b = static_cast<int>(bgr_row[x][0]);
      const int g = static_cast<int>(bgr_row[x][1]);
      const int r = static_cast<int>(bgr_row[x][2]);
      const int b_lab = static_cast<int>(lab_row[x][2]);
      const int max_rg = std::max(g, r);
      const bool is_blue =
        ((b - max_rg) > config_.blue_margin && b_lab < 126) ||
        (b_lab < config_.lab_blue_b_max && b > max_rg);
      if (is_blue) {
        blue_row[x] = 255;
      } else if (seg_row != nullptr) {
        if (seg_row[x] > 0) {
          wf_row[x] = 255;
        }
      } else {
        // Below roi_bottom (down to v=479, d=0.03m): non-blue bright/neutral pixels on mat
        const int l_val = static_cast<int>(lab_row[x][0]);
        if (l_val >= 75) {
          wf_row[x] = 255;
        }
      }
    }
  }

  // Morphological open removes isolated 1-2px blue chromatic noise specks inside white road
  const cv::Mat k_blue_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_OPEN, k_blue_open);

  // Fill non-blue dark digit cards inside horizontal white road spans (including bottom-edge digits!)
  cv::Mat road_span(full_roi_h, img_w, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < full_roi_h; ++y) {
    const uint8_t * wf_row = white_full.ptr<uint8_t>(y);
    const uint8_t * bl_row = blue_mask.ptr<uint8_t>(y);
    uint8_t * rs_row = road_span.ptr<uint8_t>(y);

    int x_left = -1;
    int x_right = -1;
    for (int x = 0; x < img_w; ++x) {
      if (wf_row[x] > 0) {
        if (x_left < 0) {
          x_left = x;
        }
        x_right = x;
      }
    }
    if (x_left >= 0 && (x_right - x_left) >= 14) {
      // Near the bottom of the image (narrow +-11cm FOV wedge), also allow non-blue dark
      // digits touching the left/right image border to remain drivable instead of fake walls.
      const bool near_bottom = (y >= full_roi_h - 48);
      const int span_l = near_bottom ? 0 : x_left;
      const int span_r = near_bottom ? (img_w - 1) : x_right;
      for (int x = span_l; x <= span_r; ++x) {
        if (bl_row[x] == 0) {
          rs_row[x] = 255;
        }
      }
    }
  }

  // 2) Fast LUT projection of FREE (road_span) and WALL_BLUE (blue_mask + outer black border) into 160x160 BEV!
  for (int i = 0; i < kBevCells; ++i) {
    const size_t idx = static_cast<size_t>(i);
    const int u = lut_u_[idx];
    const int v = lut_v_[idx];
    if (u < 0 || v < r_top || v >= img_h) {
      continue;
    }
    const int vy = v - r_top;
    if (blue_mask.at<uint8_t>(vy, u) > 0) {
      out_grid[idx] = BEV_WALL_BLUE;
    } else if (road_span.at<uint8_t>(vy, u) > 0) {
      out_grid[idx] = BEV_FREE;
    } else {
      // Dark pixels strictly OUTSIDE road_span are the black track border line (never interior digits!)
      const int l_val = static_cast<int>(lower_lab.at<cv::Vec3b>(vy, u)[0]);
      out_grid[idx] = (l_val < 75) ? BEV_WALL_BLUE : BEV_UNKNOWN;
    }
  }

  // 3) Red obstacle extraction (NEVER masked by YOLO!) + 3D cone ground-base projection!
  cv::Mat rmask1, rmask2, red_mask;
  cv::inRange(
    lower_hsv,
    cv::Scalar(config_.red_h_low_1, config_.red_s_min, config_.red_v_min),
    cv::Scalar(config_.red_h_high_1, 255, 255),
    rmask1);
  cv::inRange(
    lower_hsv,
    cv::Scalar(config_.red_h_low_2, config_.red_s_min, config_.red_v_min),
    cv::Scalar(config_.red_h_high_2, 255, 255),
    rmask2);
  cv::bitwise_or(rmask1, rmask2, red_mask);

  // Gate red pixels to road_span (dilated slightly to keep cones grazing the edge)
  cv::Mat road_gate;
  const cv::Mat k_gate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
  cv::dilate(road_span, road_gate, k_gate);
  cv::bitwise_and(red_mask, road_gate, red_mask);

  const cv::Mat k_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  const cv::Mat k_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 9));
  cv::Mat red_clean, red_grouped;
  cv::morphologyEx(red_mask, red_clean, cv::MORPH_OPEN, k_open);
  cv::morphologyEx(red_clean, red_grouped, cv::MORPH_CLOSE, k_close);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(red_grouped, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::Rect> blobs;
  for (const auto & cnt : contours) {
    if (cv::contourArea(cnt) < config_.min_red_area_px) {
      continue;
    }
    const cv::Rect r = cv::boundingRect(cnt);
    if (r.width < 6 || r.height < 6) {
      continue;
    }
    blobs.push_back(r);
  }

  // Merge fragmented strips belonging to the same hollow red no-entry patch
  bool merged = true;
  while (merged && blobs.size() >= 2) {
    merged = false;
    for (size_t i = 0; i < blobs.size() && !merged; ++i) {
      for (size_t j = i + 1; j < blobs.size(); ++j) {
        const cv::Rect & a = blobs[i];
        const cv::Rect & b = blobs[j];
        const int dx = std::max(0, std::max(a.x - (b.x + b.width), b.x - (a.x + a.width)));
        const int dy = std::max(0, std::max(a.y - (b.y + b.height), b.y - (a.y + a.height)));
        if (dx <= 22 && dy <= 16) {
          const cv::Rect u = a | b;
          if (u.width <= static_cast<int>(0.55 * img_w)) {
            blobs[i] = u;
            blobs.erase(blobs.begin() + static_cast<std::ptrdiff_t>(j));
            merged = true;
            break;
          }
        }
      }
    }
  }

  struct ClassifiedRedBlob
  {
    cv::Rect r_roi;
    bool is_3d_cone{false};
  };
  std::vector<ClassifiedRedBlob> classified_blobs;
  classified_blobs.reserve(blobs.size());

  for (const auto & r_roi : blobs) {
    const int u_mid = std::clamp(r_roi.x + r_roi.width / 2, 0, img_w - 1);
    const int v_bot_roi = std::clamp(r_roi.y + r_roi.height - 1, 0, full_roi_h - 1);
    if (road_gate.at<uint8_t>(v_bot_roi, u_mid) == 0) {
      continue;
    }

    const int raw_red_px = cv::countNonZero(red_clean(r_roi));
    const double fill_ratio =
      static_cast<double>(raw_red_px) / std::max(1.0, static_cast<double>(r_roi.area()));
    const double aspect_hw =
      static_cast<double>(r_roi.height) / std::max(1.0, static_cast<double>(r_roi.width));

    // Check if the red object sticks up ABOVE r_top (3D tall cone!)
    bool red_above_roi = false;
    if (r_roi.y <= 4 && r_top >= 12) {
      const int check_h = std::min(24, r_top);
      const cv::Rect above_rect(
        std::clamp(r_roi.x, 0, img_w - 1),
        r_top - check_h,
        std::clamp(r_roi.width, 1, img_w - r_roi.x),
        check_h);
      cv::Mat above_hsv, am1, am2, am;
      cv::cvtColor(frame_bgr(above_rect), above_hsv, cv::COLOR_BGR2HSV);
      cv::inRange(
        above_hsv,
        cv::Scalar(config_.red_h_low_1, config_.red_s_min, config_.red_v_min),
        cv::Scalar(config_.red_h_high_1, 255, 255),
        am1);
      cv::inRange(
        above_hsv,
        cv::Scalar(config_.red_h_low_2, config_.red_s_min, config_.red_v_min),
        cv::Scalar(config_.red_h_high_2, 255, 255),
        am2);
      cv::bitwise_or(am1, am2, am);
      red_above_roi = (cv::countNonZero(am) >= std::max(12, above_rect.area() / 10));
    }

    // 3D cone vs flat ground no-entry:
    //   - Tall solid object or extends above ROI top -> 3D cone (project ONLY ground base!)
    //   - Hollow striped / low flat patch -> flat no-entry (project exact ground footprint!)
    const bool is_3d_cone = red_above_roi || (aspect_hw > 0.68 && fill_ratio > 0.30);
    classified_blobs.push_back(ClassifiedRedBlob{r_roi, is_3d_cone});
  }

  // Close-range cone-base fragment guard:
  // When a 3D cone gets within d < 0.25m, its shadowed base flange can detach from the upper body
  // into a low-aspect-ratio strip sitting directly beneath the cone. Merge it back into the cone
  // instead of falsely spawning a phantom +30s flat no-entry box!
  for (size_t i = 0; i < classified_blobs.size(); ++i) {
    if (classified_blobs[i].is_3d_cone) {
      continue;
    }
    const cv::Rect & sub = classified_blobs[i].r_roi;
    const double sub_u = static_cast<double>(sub.x) + 0.5 * static_cast<double>(sub.width);
    for (size_t j = 0; j < classified_blobs.size(); ++j) {
      if (i == j || !classified_blobs[j].is_3d_cone) {
        continue;
      }
      const cv::Rect & cone = classified_blobs[j].r_roi;
      const double cone_u = static_cast<double>(cone.x) + 0.5 * static_cast<double>(cone.width);
      const int cone_bot = cone.y + cone.height;
      const bool horiz_aligned = std::abs(sub_u - cone_u) <= std::max(28.0, 0.70 * cone.width);
      const bool directly_below = (sub.y >= cone.y) && (sub.y <= cone_bot + 30);
      if (horiz_aligned && directly_below) {
        classified_blobs[j].r_roi = cone | sub;
        classified_blobs.erase(classified_blobs.begin() + static_cast<std::ptrdiff_t>(i));
        if (i < j) {
          --j;
        }
        --i;
        break;
      }
    }
  }

  for (const auto & cb : classified_blobs) {
    const cv::Rect & r_roi = cb.r_roi;
    const bool is_3d_cone = cb.is_3d_cone;

    const double v_bot_full = static_cast<double>(r_top + r_roi.y + r_roi.height - 1);
    const double u_center_full = static_cast<double>(r_roi.x) + 0.5 * static_cast<double>(r_roi.width);
    double d_base = 0.0, y_base = 0.0;
    if (!projector.pixel_to_ground(u_center_full, v_bot_full, d_base, y_base)) {
      continue;
    }
    if (d_base < 0.02 || d_base > 1.25 || std::abs(y_base) > 0.72) {
      continue;
    }

    if (is_3d_cone) {
      // Ground-contact base projection -> ZERO 1-meter comet tail!
      double dl = 0.0, yl = 0.0, dr = 0.0, yr = 0.0;
      double radius = 0.070;
      if (projector.pixel_to_ground(static_cast<double>(r_roi.x), v_bot_full, dl, yl) &&
        projector.pixel_to_ground(static_cast<double>(r_roi.x + r_roi.width - 1), v_bot_full, dr, yr))
      {
        radius = std::clamp(0.5 * std::abs(yl - yr), 0.060, 0.085);
      }
      // Shift center backward by +0.5*radius (bottom edge is the front face of the cone base)
      stamp_bev_circle(out_grid, d_base + 0.5 * radius, y_base, radius, BEV_OBS_CONE);
    } else {
      // Flat red no-entry zone on z=0: project exact ground pixels into BEV!
      int stamped = 0;
      const int y_end = r_roi.y + r_roi.height;
      const int x_end = r_roi.x + r_roi.width;
      for (int py = r_roi.y; py < y_end; py += 2) {
        const uint8_t * row_ptr = red_grouped.ptr<uint8_t>(py);
        const double vf = static_cast<double>(r_top + py);
        for (int px = r_roi.x; px < x_end; px += 2) {
          if (row_ptr[px] == 0) {
            continue;
          }
          double dm = 0.0, ym = 0.0;
          if (projector.pixel_to_ground(static_cast<double>(px), vf, dm, ym) &&
            dm >= 0.02 && dm <= 1.22 && std::abs(ym) <= 0.75)
          {
            stamp_bev_circle(out_grid, dm, ym, 0.016, BEV_OBS_NO_ENTRY);
            stamped++;
          }
        }
      }
      if (stamped == 0) {
        stamp_bev_circle(out_grid, d_base + 0.05, y_base, 0.075, BEV_OBS_NO_ENTRY);
      }
    }

    BevRedOverlayBox ob{};
    ob.bbox_img = cv::Rect(r_roi.x, r_top + r_roi.y, r_roi.width, r_roi.height);
    ob.is_no_entry = !is_3d_cone;
    ob.d_bumper_m = d_base;
    ob.y_lat_m = y_base;
    out_red_boxes.push_back(ob);
  }
}

bool BevMapper::validate_yolo_digit(
  const cv::Mat & frame_bgr,
  const cv::Rect & bbox_img,
  const GroundProjector & projector,
  double & out_d_m,
  double & out_y_m) const
{
  out_d_m = 0.0;
  out_y_m = 0.0;
  if (frame_bgr.empty()) {
    return false;
  }

  const cv::Rect clipped = bbox_img & cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
  if (clipped.width < 8 || clipped.height < 8) {
    return false;
  }

  // 1) Pixel color sanity gate: reject red-striped no-entry (false '1') & blue border/shadow (false '6'/'2')
  const cv::Mat patch = frame_bgr(clipped);
  cv::Mat patch_hsv, patch_lab;
  cv::cvtColor(patch, patch_hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(patch, patch_lab, cv::COLOR_BGR2Lab);

  int red_px = 0;
  int blue_px = 0;
  const int total_px = clipped.width * clipped.height;

  for (int y = 0; y < clipped.height; ++y) {
    const cv::Vec3b * bgr_r = patch.ptr<cv::Vec3b>(y);
    const cv::Vec3b * hsv_r = patch_hsv.ptr<cv::Vec3b>(y);
    const cv::Vec3b * lab_r = patch_lab.ptr<cv::Vec3b>(y);
    for (int x = 0; x < clipped.width; ++x) {
      const int h = hsv_r[x][0];
      const int s = hsv_r[x][1];
      const int v = hsv_r[x][2];
      if (s >= config_.red_s_min && v >= config_.red_v_min &&
        ((h >= config_.red_h_low_1 && h <= config_.red_h_high_1) ||
        (h >= config_.red_h_low_2 && h <= config_.red_h_high_2)))
      {
        red_px++;
      }
      const int b = bgr_r[x][0];
      const int max_rg = std::max(bgr_r[x][1], bgr_r[x][2]);
      const int b_lab = lab_r[x][2];
      if (((b - max_rg) > config_.blue_margin && b_lab < 126) ||
        (b_lab < config_.lab_blue_b_max && b > max_rg))
      {
        blue_px++;
      }
    }
  }

  const double red_ratio = static_cast<double>(red_px) / std::max(1, total_px);
  const double blue_ratio = static_cast<double>(blue_px) / std::max(1, total_px);
  if (red_ratio > config_.digit_max_red_ratio || blue_ratio > config_.digit_max_blue_ratio) {
    return false;
  }

  // 2) Ground projection & physical size check
  const double u_c = static_cast<double>(clipped.x) + 0.5 * static_cast<double>(clipped.width);
  const double v_b = static_cast<double>(clipped.y + clipped.height - 1);
  if (!projector.pixel_to_ground(u_c, v_b, out_d_m, out_y_m)) {
    return false;
  }
  if (out_d_m < 0.04 || out_d_m > 1.25 || std::abs(out_y_m) > 0.65) {
    return false;
  }

  double dl = 0.0, yl = 0.0, dr = 0.0, yr = 0.0;
  if (projector.pixel_to_ground(static_cast<double>(clipped.x), v_b, dl, yl) &&
    projector.pixel_to_ground(static_cast<double>(clipped.x + clipped.width - 1), v_b, dr, yr))
  {
    const double gw = std::abs(yl - yr);
    if (gw < config_.digit_min_ground_w_m || gw > config_.digit_max_ground_w_m) {
      return false;
    }
  }

  return true;
}

cv::Mat BevMapper::render_bev_inset(
  const std::array<uint8_t, kBevCells> & fused_grid,
  const BevDwaPlanResult & plan,
  const PreFlightStatus & preflight,
  bool yellow_visible,
  double yellow_d_m,
  int inset_size)
{
  cv::Mat raw(kBevRows, kBevCols, CV_8UC3, cv::Scalar(35, 35, 42));

  // Flip vertically when drawing: top of inset = forward (+d = +1.25m), bottom = rear (-0.35m)
  for (int r = 0; r < kBevRows; ++r) {
    const int draw_y = (kBevRows - 1) - r;
    cv::Vec3b * dst = raw.ptr<cv::Vec3b>(draw_y);
    for (int c = 0; c < kBevCols; ++c) {
      const uint8_t cell = fused_grid[static_cast<size_t>(r * kBevCols + c)];
      switch (cell) {
        case BEV_FREE:
          dst[c] = cv::Vec3b(230, 232, 235);
          break;
        case BEV_WALL_BLUE:
          dst[c] = cv::Vec3b(150, 70, 30);
          break;
        case BEV_OBS_CONE:
          dst[c] = cv::Vec3b(0, 120, 255);
          break;
        case BEV_OBS_NO_ENTRY:
          dst[c] = cv::Vec3b(200, 0, 255);
          break;
        default:
          dst[c] = cv::Vec3b(32, 32, 38);
          break;
      }
    }
  }

  cv::Mat vis;
  cv::resize(raw, vis, cv::Size(inset_size, inset_size), 0.0, 0.0, cv::INTER_NEAREST);
  const double scale = static_cast<double>(inset_size) / static_cast<double>(kBevRows);

  auto to_px = [scale, inset_size](double d_m, double y_m) -> cv::Point {
    const double col_f = (kBevYMax - y_m) / kBevRes;
    const double row_f = (d_m - kBevDMin) / kBevRes;
    const int px = static_cast<int>(std::round(col_f * scale));
    const int py = static_cast<int>(std::round((static_cast<double>(kBevRows - 1) - row_f) * scale));
    return cv::Point(std::clamp(px, 0, inset_size - 1), std::clamp(py, 0, inset_size - 1));
  };

  // Yellow stop line marker
  if (yellow_visible && yellow_d_m >= 0.0 && yellow_d_m <= kBevDMax) {
    cv::line(vis, to_px(yellow_d_m, 0.30), to_px(yellow_d_m, -0.30), cv::Scalar(0, 255, 255), 2);
  }

  // Draw 1D Safe Corridor bounds (cyan) and smoothed centerline target (bright green)
  if (!plan.corridor_slices.empty()) {
    std::vector<cv::Point> left_pts;
    std::vector<cv::Point> right_pts;
    std::vector<cv::Point> mid_pts;
    left_pts.reserve(plan.corridor_slices.size());
    right_pts.reserve(plan.corridor_slices.size());
    mid_pts.reserve(plan.corridor_slices.size());
    for (const auto & sl : plan.corridor_slices) {
      if (!sl.valid) {
        continue;
      }
      left_pts.push_back(to_px(sl.d_m, sl.y_safe_left));
      right_pts.push_back(to_px(sl.d_m, sl.y_safe_right));
      mid_pts.push_back(to_px(sl.d_m, sl.y_target));
    }
    if (left_pts.size() >= 2U) {
      cv::polylines(vis, left_pts, false, cv::Scalar(235, 210, 40), 1, cv::LINE_AA);
      cv::polylines(vis, right_pts, false, cv::Scalar(235, 210, 40), 1, cv::LINE_AA);
      cv::polylines(vis, mid_pts, false, cv::Scalar(40, 230, 80), 2, cv::LINE_AA);
    }
  }

  // Draw chosen Ackermann predicted arc (yellow) + Pure Pursuit lookahead target (magenta dot)
  auto draw_arc = [&](double kappa, double len_m, const cv::Scalar & color, int thick) {
    if (len_m < 0.03) {
      return;
    }
    double x = 0.0, y = 0.0, th = 0.0;
    const double ds = 0.04;
    cv::Point prev = to_px(x, y);
    for (double s = ds; s <= len_m + 1e-4; s += ds) {
      const double dth = kappa * ds;
      x += ds * std::cos(th + 0.5 * dth);
      y += ds * std::sin(th + 0.5 * dth);
      th += dth;
      const cv::Point cur = to_px(x, y);
      cv::line(vis, prev, cur, color, thick, cv::LINE_AA);
      prev = cur;
    }
  };

  draw_arc(plan.chosen_kappa, std::max(0.25, std::min(0.65, plan.lookahead_m + 0.10)), cv::Scalar(0, 220, 255), 2);
  if (plan.valid_slice_count >= 3) {
    const cv::Point la_pt = to_px(plan.lookahead_x_m, plan.lookahead_y_m);
    cv::circle(vis, la_pt, 4, cv::Scalar(255, 60, 255), -1, cv::LINE_AA);
    cv::circle(vis, la_pt, 5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  }

  // Draw exact physical car rectangle (0.191m x 0.280m, bumper at d=0, tail at d=-0.28m)
  const cv::Point car_tl = to_px(0.0, 0.0955);
  const cv::Point car_br = to_px(-0.280, -0.0955);
  cv::rectangle(vis, car_tl, car_br, cv::Scalar(0, 230, 255), 2);

  // Topology compass arrow in top-left of BEV inset
  const cv::Point comp_c(24, 24);
  cv::circle(vis, comp_c, 16, cv::Scalar(40, 40, 48), -1);
  cv::circle(vis, comp_c, 16, cv::Scalar(180, 180, 190), 1);
  if (plan.goal_yaw_active) {
    // Arrow shows goal heading relative to current car heading (up = straight ahead)
    const double rel_goal = BevDwaPlanner::wrap_angle_rad(plan.goal_yaw_rad - plan.rel_yaw_rad);
    const cv::Point tip(
      comp_c.x - static_cast<int>(std::round(13.0 * std::sin(rel_goal))),
      comp_c.y - static_cast<int>(std::round(13.0 * std::cos(rel_goal))));
    cv::arrowedLine(vis, comp_c, tip, cv::Scalar(0, 255, 255), 2, cv::LINE_AA, 0, 0.35);
  } else {
    cv::arrowedLine(vis, comp_c, cv::Point(comp_c.x, comp_c.y - 12), cv::Scalar(120, 255, 120), 1);
  }

  // Compact HUD text inside BEV inset
  char txt1[64];
  std::snprintf(
    txt1, sizeof(txt1), "BEV %s k=%+.2f",
    topology_phase_to_string(plan.phase), plan.chosen_kappa);
  cv::putText(vis, txt1, cv::Point(44, 15), cv::FONT_HERSHEY_SIMPLEX, 0.34, cv::Scalar(255, 255, 255), 1);

  char txt2[64];
  std::snprintf(
    txt2, sizeof(txt2), "kr=%+.2f ey=%+.2f Ld=%.2f",
    plan.road_kappa_fit, plan.lat_err_m, plan.lookahead_m);
  cv::putText(vis, txt2, cv::Point(44, 29), cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(200, 245, 255), 1);

  if (preflight.checked && !preflight.homography_ok) {
    cv::rectangle(vis, cv::Point(0, inset_size - 18), cv::Point(inset_size, inset_size), cv::Scalar(0, 0, 200), -1);
    cv::putText(
      vis, "WARN: CAM/H TILT!", cv::Point(8, inset_size - 5),
      cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(255, 255, 255), 1);
  }

  cv::rectangle(vis, cv::Point(0, 0), cv::Point(inset_size - 1, inset_size - 1), cv::Scalar(90, 220, 255), 1);
  return vis;
}

}  // namespace line_follower
