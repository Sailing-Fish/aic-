#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "line_follower/ackermann_cbf_shield.hpp"
#include "line_follower/corridor_planner.hpp"
#include "line_follower/finals_types.hpp"
#include "line_follower/finish_guard.hpp"
#include "line_follower/ground_projector.hpp"
#include "line_follower/line_follower_types.hpp"
#include "line_follower/motion_controller.hpp"
#include "line_follower/obstacle_memory.hpp"
#include "line_follower/track_state_machine.hpp"

#if defined(_WIN32)
#define BRAIN_API __declspec(dllexport)
#else
#define BRAIN_API __attribute__((visibility("default")))
#endif

extern "C" {

struct BrainDetectionC
{
  int32_t obj_type;   // 0=UNKNOWN, 1=RED_CONE, 2=RED_NO_ENTRY, 3=BONUS_DIGIT, 4=YELLOW_STOP_LINE
  int32_t digit;      // 1..8 when BONUS_DIGIT, else 0
  double confidence;
  double x;           // pixel bbox top-left x in full image
  double y;           // pixel bbox top-left y in full image
  double w;           // pixel bbox width
  double h;           // pixel bbox height
  int32_t track_id;   // -1 if unavailable
  int32_t _pad;
};

struct BrainCvRedBoxC
{
  double x;           // pixel bbox top-left x in full image
  double y;           // pixel bbox top-left y in full image
  double w;           // pixel bbox width
  double h;           // pixel bbox height
  double confidence;  // default 0.60
  double fill_ratio;  // < 0 if not provided; >= 0 uses exact C++ formula
  int32_t is_no_entry;
  int32_t _pad;
};

struct BrainStepInput
{
  int32_t image_width;
  int32_t image_height;

  int32_t lane_valid;
  int32_t near_valid;        // -1 = infer from lane_valid, 0/1 explicit
  int32_t far_valid;         // -1 = infer from lane_valid, 0/1 explicit
  int32_t has_lateral_error; // 1 if lateral_error explicitly supplied

  double lateral_error;      // [-1, 1] fused & filtered lateral error
  double near_error;         // [-1, 1]
  double far_error;          // [-1, 1]
  double heading_error;      // [-1, 1] (obs.heading_proxy)
  double curvature_proxy;    // [0, 1], < 0 means compute from near/far/heading
  double confidence;         // [0, 1]
  double near_width_ratio;   // [0, 1]
  double far_width_ratio;    // [0, 1]
  double lane_width_ratio;   // [0, 1], < 0 means compute from near/far width

  int32_t has_left_edge;
  int32_t has_right_edge;
  int32_t edges_are_px;      // 0 = normalized [-1, 1], 1 = pixel x
  int32_t _pad0;
  double left_edge;
  double right_edge;

  int32_t fork_detected;
  int32_t use_external_reverse_curve;
  int32_t reverse_curve_active;
  int32_t _pad1;
  double left_branch_error;
  double right_branch_error;

  int32_t yellow_detected;
  int32_t yellow_is_geometric;
  int32_t yellow_dead_end;
  int32_t _pad2;
  double yellow_bottom_y;    // pixels [0, H] (if in (0, 1], auto-scaled by H)
  double yellow_center_x;    // pixels [0, W] (< 0 defaults to W * 0.5)
  double yellow_width_ratio;

  int32_t has_odom;
  int32_t _pad3;
  double odom_x;
  double odom_y;
  double odom_yaw;
  double odom_v;
  double odom_w;

  double voltage;            // <= 0 or NaN means unused
  double timestamp_sec;

  const BrainDetectionC * detections;
  int32_t num_detections;
  int32_t _pad4;

  const BrainCvRedBoxC * cv_red_boxes;
  int32_t num_cv_red_boxes;
  int32_t _pad5;
};

struct BrainStepOutput
{
  double v_cmd;
  double w_cmd;
  int32_t state;                  // 0=LOST, 1=TRACKING, 2=YELLOW_LINE_DETECTED
  int32_t publish_stop;

  double corridor_left_m;         // safe_y_max_m
  double corridor_right_m;        // safe_y_min_m
  double corridor_target_y_m;     // target_y_m
  double lane_left_m;
  double lane_right_m;
  double speed_cap_mps;

  int32_t active_digit;           // 1..8 when digit_target_active, else 0
  int32_t digit_target_active;
  int32_t obstacle_count;
  int32_t near_obstacle_count;
  int32_t far_obstacle_count;
  int32_t has_blocking_obstacle;
  int32_t preferred_branch;       // -1, 0, +1

  int32_t cbf_active;
  double cbf_min_clearance_m;

  int32_t finish_armed;
  int32_t permanent_stop;
  int32_t yellow_stop_active;
  int32_t yellow_brake_active;
  int32_t yellow_count;
  int32_t yellow_dist_valid;
  int32_t yellow_is_finish;
  int32_t yellow_rearmed;
  double yellow_d_bumper_m;

  double total_arc_m;
  double friction_scale;
  double effective_brake_decel;
  double v_safe_cap_mps;
  double clear_path_score;
  double brake_cmd_mps;
  double effective_brake_decel_mps2;

  int32_t corridor_left_px;
  int32_t corridor_right_px;
  int32_t corridor_ref_px;
  int32_t corridor_row_y;

  char state_name[32];
  char active_mode_tag[32];
  char yellow_phase[32];
  char cbf_limiting_source[32];
};

}  // extern "C"

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxBrakeSpeed = 1.5;

double wrap_angle_rad(double angle)
{
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

void copy_str(char * dst, std::size_t dst_size, const std::string & src)
{
  if (dst == nullptr || dst_size == 0) {
    return;
  }
  const std::size_t n = std::min(dst_size - 1, src.size());
  if (n > 0) {
    std::memcpy(dst, src.data(), n);
  }
  dst[n] = '\0';
}

line_follower::SemanticObjectType decode_obj_type(int32_t code)
{
  switch (code) {
    case 1:
      return line_follower::SemanticObjectType::RED_CONE;
    case 2:
      return line_follower::SemanticObjectType::RED_NO_ENTRY;
    case 3:
      return line_follower::SemanticObjectType::BONUS_DIGIT;
    case 4:
      return line_follower::SemanticObjectType::YELLOW_STOP_LINE;
    default:
      return line_follower::SemanticObjectType::UNKNOWN;
  }
}

struct BrainInstance
{
  // Mode switches
  std::string track_mode{"fork"};
  std::string lane_seg_mode{"relative_border"};
  std::string yellow_detect_mode{"geometric_cv"};
  bool enable_obstacle_memory{true};
  bool enable_cv_red_fallback{true};
  bool allow_dual_digit_cover{false};
  bool enable_cbf_shield{false};
  bool enable_risk_aware_decision{true};
  bool enable_brake_adaptation{true};

  // CRC padding & timeouts
  double crc_no_entry_pad_y_px{0.0};
  double crc_cone_pad_y_px{0.0};
  double crc_yellow_pad_y_px{0.0};
  double detection_timeout_sec{0.35};

  // Motion & ROI
  double target_speed{0.44};
  double max_speed{0.85};
  double min_speed{0.0};
  double max_yaw_rate{1.6};
  double speed_gain_damping{0.45};
  bool enable_clear_path_speed{true};
  double clear_straight_speed{0.62};
  double clear_preview_dist_m{0.95};
  double max_accel_mps2{0.90};
  double max_decel_mps2{2.20};
  double active_decel_kp{0.45};
  double roi_top_ratio{0.68};
  double roi_bottom_ratio{0.18};
  double yellow_roi_bottom_ratio{0.0};

  // Lane geometry band & curve assist params
  double far_band_top_ratio{0.28};
  double far_band_bottom_ratio{0.48};
  double near_band_top_ratio{0.55};
  double near_band_bottom_ratio{0.80};
  double near_band_weight{0.60};
  double far_band_weight{0.40};
  double wide_road_ratio_threshold{0.84};
  bool enable_width_adaptive_fusion{true};
  bool enable_continuous_curve_assist{true};
  double reverse_curve_min_error{0.12};
  double reverse_curve_threshold{0.35};
  int reverse_curve_confirm_frames{3};
  int reverse_curve_clear_frames{4};
  double low_confidence_threshold{0.55};
  double low_confidence_speed{0.24};
  double low_confidence_yaw_hold{0.80};

  // Yellow stop & reverse-brake params
  double yellow_stop_duration{1.0};
  int yellow_clear_frames_required{10};
  double yellow_stop_min_time{0.35};
  double odom_still_threshold{0.02};
  double odom_still_hold{0.18};
  double odom_max_age_sec{0.20};
  bool yellow_require_odom_still{true};
  double yellow_max_hold_sec{3.0};
  double yellow_brake_speed{0.0};
  double yellow_brake_period_s{0.2};
  double yellow_brake_pulse_s{0.0};
  double yellow_brake_release_speed{0.08};
  double yellow_brake_max_time_s{0.8};

  // Lost-line params
  double line_lost_stop_time{0.6};
  double line_lost_forward_speed{0.15};
  double line_lost_yaw_hold_factor{0.85};
  double line_lost_slow_time{0.4};
  double line_lost_slow_factor{0.4};
  double line_lost_full_stop_time{1.2};
  bool lost_line_safe_mode{true};
  double lost_line_safe_speed{0.12};
  double lost_line_safe_stop_time{0.35};

  // GroundProjector params
  bool use_homography{false};
  std::vector<double> homography_matrix{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  double cam_height_m{0.11};
  double cam_pitch_deg{12.5};
  double focal_px{325.0};
  double cam_cx{320.0};
  double cam_cy{240.0};
  double front_overhang_m{0.06};
  int image_width{640};
  int image_height{480};

  // ObstacleMemory params
  double obstacle_merge_radius_m{0.16};
  double obstacle_memory_ttl_sec{1.8};
  double car_length_m{0.280};
  double rear_axle_to_bumper_m{0.18};
  double obstacle_rear_keep_margin_m{0.15};

  // CorridorPlanner params
  double car_width_m{0.191};
  double default_lane_width_m{0.430};
  double border_margin_m{0.025};
  double cone_margin_m{0.025};
  double no_entry_margin_m{0.030};
  double obstacle_lookahead_m{1.10};
  double obstacle_steer_horizon_m{0.65};
  double obstacle_curve_min_horizon_m{0.45};
  double obstacle_stagger_gap_m{0.25};
  double obstacle_rear_keep_m{0.28};
  double obstacle_speed_mps{0.26};
  double no_entry_speed_mps{0.22};
  double blocked_creep_speed_mps{0.12};
  int fork_outer_branch{-1};
  double fork_outer_time_penalty_s{3.0};
  int fork_min_votes{3};
  double fork_commit_dist_m{1.10};
  double digit_min_conf{0.55};
  int digit_min_frames{2};
  double digit_lookahead_max_m{1.15};
  double max_digit_lateral_shift_m{0.12};
  double digit_approach_speed_mps{0.30};
  double corridor_kp_y{3.8};
  double corridor_heading_gain{0.85};

  // CBF Shield params
  double cbf_wheelbase_m{0.142};
  double cbf_max_steer_deg{35.0};
  int cbf_horizon_steps{6};
  double cbf_step_dt_sec{0.06};
  double cbf_gamma{0.60};

  // FinishGuard params
  double odom_scale_factor{1.0};
  double finish_min_dist_m{8.5};
  double finish_force_dist_m{11.5};
  double finish_min_cum_yaw_rad{0.0};
  double yellow_pre_brake_dist_m{0.70};
  double yellow_creep_speed_mps{0.24};
  double yellow_final_ramp_dist_m{0.35};
  double yellow_final_creep_speed_mps{0.18};
  double yellow_brake_decel_mps2{1.8};
  double voltage_decel_slope{0.0};
  double yellow_control_latency_s{0.10};
  double yellow_bumper_margin_m{0.08};
  double yellow_rearm_dist_m{0.35};
  bool enable_sqrt_brake_envelope{true};
  bool enable_closed_loop_brake{true};
  double brake_damping_gain{0.85};
  double brake_zero_cutoff_mps{0.025};

  // The 7 pure C++17 modules
  line_follower::GroundProjector ground_projector{};
  line_follower::ObstacleMemory obstacle_memory{};
  line_follower::CorridorPlanner corridor_planner{};
  line_follower::FinishGuard finish_guard{};
  line_follower::TrackStateMachine track_state_machine{};
  line_follower::MotionController motion_controller{};
  line_follower::AckermannCbfShield cbf_shield{};

  // Runtime states matching LineFollowerNode
  line_follower::CurveAssistState curve_assist{};
  line_follower::MotionState motion_state{};
  line_follower::VehiclePose2D odom_pose{};
  bool odom_pose_seen{false};
  double last_odom_yaw{0.0};
  bool last_odom_yaw_valid{false};
  double last_odom_time_sec{0.0};

  bool yellow_brake_active{false};
  double yellow_brake_start_sec{0.0};
  bool yellow_brake_measure_pending{false};
  double yellow_brake_entry_speed_mps{0.0};
  double yellow_brake_start_arc_m{0.0};

  bool has_stepped{false};
  bool params_dirty{true};

  BrainInstance()
  {
    sync_configs();
    reset(0.0);
  }

  void sync_configs()
  {
    using namespace line_follower;

    if (track_mode != "fork" && track_mode != "roundabout") {
      track_mode = "fork";
    }
    crc_no_entry_pad_y_px = std::max(0.0, crc_no_entry_pad_y_px);
    crc_cone_pad_y_px = std::max(0.0, crc_cone_pad_y_px);
    crc_yellow_pad_y_px = std::max(0.0, crc_yellow_pad_y_px);
    detection_timeout_sec = std::max(0.05, detection_timeout_sec);

    target_speed = std::max(0.0, target_speed);
    max_speed = std::max(0.0, max_speed);
    min_speed = std::max(0.0, min_speed);
    max_yaw_rate = std::max(0.0, max_yaw_rate);
    speed_gain_damping = std::max(0.0, speed_gain_damping);
    roi_top_ratio = clamp_value(roi_top_ratio, 0.0, 1.0);
    roi_bottom_ratio = clamp_value(roi_bottom_ratio, 0.0, 0.5);
    yellow_roi_bottom_ratio = clamp_value(yellow_roi_bottom_ratio, 0.0, 0.5);

    yellow_clear_frames_required = std::max(1, yellow_clear_frames_required);
    yellow_brake_speed = std::clamp(yellow_brake_speed, 0.0, kMaxBrakeSpeed);
    yellow_brake_period_s = std::max(0.0, yellow_brake_period_s);
    yellow_brake_pulse_s = std::max(0.0, yellow_brake_pulse_s);
    yellow_brake_release_speed = std::max(0.0, yellow_brake_release_speed);
    yellow_brake_max_time_s = std::max(0.0, yellow_brake_max_time_s);

    // 1. GroundProjector
    GroundProjectorConfig proj_cfg{};
    proj_cfg.use_homography = false;
    proj_cfg.cam_height_m = cam_height_m;
    proj_cfg.cam_pitch_deg = cam_pitch_deg;
    proj_cfg.focal_px = focal_px;
    proj_cfg.cx = cam_cx;
    proj_cfg.cy = cam_cy;
    proj_cfg.front_overhang_m = front_overhang_m;
    proj_cfg.image_width = std::max(1, image_width);
    proj_cfg.image_height = std::max(1, image_height);
    ground_projector.set_config(proj_cfg);
    if (use_homography) {
      if (!ground_projector.set_homography_from_vector(homography_matrix, true)) {
        use_homography = false;
      }
    }

    // 2. TrackStateMachine
    TrackStateConfig state_cfg{};
    state_cfg.yellow_stop_duration = yellow_stop_duration;
    state_cfg.yellow_clear_frames_required = yellow_clear_frames_required;
    state_cfg.yellow_stop_min_time = yellow_stop_min_time;
    state_cfg.odom_still_threshold = odom_still_threshold;
    state_cfg.odom_still_hold = odom_still_hold;
    state_cfg.odom_max_age_sec = odom_max_age_sec;
    state_cfg.yellow_require_odom_still = yellow_require_odom_still;
    state_cfg.yellow_max_hold_sec = yellow_max_hold_sec;
    state_cfg.line_lost_stop_time = line_lost_stop_time;
    track_state_machine.set_config(state_cfg);

    // 3. MotionController
    MotionControllerConfig motion_cfg{};
    motion_cfg.target_speed = target_speed;
    motion_cfg.max_speed = max_speed;
    motion_cfg.min_speed = min_speed;
    motion_cfg.max_yaw_rate = max_yaw_rate;
    motion_cfg.speed_gain_damping = speed_gain_damping;
    motion_cfg.low_confidence_threshold = low_confidence_threshold;
    motion_cfg.low_confidence_speed = low_confidence_speed;
    motion_cfg.low_confidence_yaw_hold = low_confidence_yaw_hold;
    motion_cfg.line_lost_forward_speed = line_lost_forward_speed;
    motion_cfg.line_lost_yaw_hold_factor = line_lost_yaw_hold_factor;
    motion_cfg.line_lost_slow_time = line_lost_slow_time;
    motion_cfg.line_lost_slow_factor = line_lost_slow_factor;
    motion_cfg.line_lost_full_stop_time = line_lost_full_stop_time;
    motion_cfg.lost_line_safe_mode = lost_line_safe_mode;
    motion_cfg.lost_line_safe_speed = lost_line_safe_speed;
    motion_cfg.lost_line_safe_stop_time = lost_line_safe_stop_time;
    motion_cfg.corridor_kp_y = corridor_kp_y;
    motion_cfg.corridor_heading_gain = corridor_heading_gain;
    motion_cfg.enable_clear_path_speed = enable_clear_path_speed;
    motion_cfg.clear_straight_speed = clear_straight_speed;
    motion_cfg.clear_preview_dist_m = clear_preview_dist_m;
    motion_cfg.max_accel_mps2 = max_accel_mps2;
    motion_cfg.max_decel_mps2 = max_decel_mps2;
    motion_cfg.active_decel_kp = active_decel_kp;
    motion_controller.set_config(motion_cfg);

    // 4. ObstacleMemory
    ObstacleMemoryConfig mem_cfg{};
    mem_cfg.merge_radius_m = obstacle_merge_radius_m;
    mem_cfg.memory_ttl_sec = obstacle_memory_ttl_sec;
    mem_cfg.car_length_m = car_length_m;
    mem_cfg.rear_axle_to_bumper_m = rear_axle_to_bumper_m;
    mem_cfg.rear_keep_margin_m = obstacle_rear_keep_margin_m;
    obstacle_memory.set_config(mem_cfg);

    // 5. CorridorPlanner
    CorridorPlannerConfig cor_cfg{};
    cor_cfg.car_width_m = car_width_m;
    cor_cfg.car_length_m = mem_cfg.car_length_m;
    cor_cfg.default_lane_width_m = default_lane_width_m;
    cor_cfg.border_margin_m = border_margin_m;
    cor_cfg.cone_margin_m = cone_margin_m;
    cor_cfg.no_entry_margin_m = no_entry_margin_m;
    cor_cfg.obstacle_lookahead_m = obstacle_lookahead_m;
    cor_cfg.obstacle_steer_horizon_m = obstacle_steer_horizon_m;
    cor_cfg.obstacle_curve_min_horizon_m = obstacle_curve_min_horizon_m;
    cor_cfg.obstacle_stagger_gap_m = obstacle_stagger_gap_m;
    cor_cfg.obstacle_rear_keep_m = obstacle_rear_keep_m;
    cor_cfg.cruise_speed_mps = target_speed;
    cor_cfg.obstacle_speed_mps = obstacle_speed_mps;
    cor_cfg.no_entry_speed_mps = no_entry_speed_mps;
    cor_cfg.blocked_creep_speed_mps = blocked_creep_speed_mps;
    cor_cfg.track_mode = track_mode;
    cor_cfg.fork_outer_branch = fork_outer_branch;
    cor_cfg.fork_outer_time_penalty_s = fork_outer_time_penalty_s;
    cor_cfg.fork_min_votes = fork_min_votes;
    cor_cfg.fork_commit_dist_m = fork_commit_dist_m;
    cor_cfg.digit_min_conf = digit_min_conf;
    cor_cfg.digit_min_frames = digit_min_frames;
    cor_cfg.digit_lookahead_max_m = digit_lookahead_max_m;
    cor_cfg.max_digit_lateral_shift_m = max_digit_lateral_shift_m;
    cor_cfg.digit_approach_speed_mps = digit_approach_speed_mps;
    cor_cfg.allow_dual_digit_cover = allow_dual_digit_cover;
    cor_cfg.enable_risk_aware_decision = enable_risk_aware_decision;
    corridor_planner.set_config(cor_cfg);

    // 6. AckermannCbfShield
    AckermannCbfConfig cbf_cfg{};
    cbf_cfg.car_length_m = mem_cfg.car_length_m;
    cbf_cfg.car_width_m = cor_cfg.car_width_m;
    cbf_cfg.rear_axle_to_bumper_m = mem_cfg.rear_axle_to_bumper_m;
    cbf_cfg.wheelbase_m = cbf_wheelbase_m;
    cbf_cfg.max_steer_rad = cbf_max_steer_deg * (kPi / 180.0);
    cbf_cfg.max_yaw_rate = max_yaw_rate;
    cbf_cfg.default_lane_width_m = cor_cfg.default_lane_width_m;
    cbf_cfg.border_margin_m = cor_cfg.border_margin_m;
    cbf_cfg.cone_margin_m = cor_cfg.cone_margin_m;
    cbf_cfg.no_entry_margin_m = cor_cfg.no_entry_margin_m;
    cbf_cfg.horizon_steps = cbf_horizon_steps;
    cbf_cfg.step_dt_sec = cbf_step_dt_sec;
    cbf_cfg.cbf_gamma = cbf_gamma;
    cbf_shield.set_config(cbf_cfg);

    // 7. FinishGuard
    FinishGuardConfig fg_cfg{};
    fg_cfg.odom_scale_factor = odom_scale_factor;
    fg_cfg.finish_min_dist_m = finish_min_dist_m;
    fg_cfg.finish_force_dist_m = finish_force_dist_m;
    fg_cfg.min_cumulative_abs_yaw_rad = finish_min_cum_yaw_rad;
    fg_cfg.pre_brake_dist_m = yellow_pre_brake_dist_m;
    fg_cfg.creep_speed_mps = yellow_creep_speed_mps;
    fg_cfg.yellow_final_ramp_dist_m = yellow_final_ramp_dist_m;
    fg_cfg.yellow_final_creep_speed_mps = yellow_final_creep_speed_mps;
    fg_cfg.brake_decel_mps2 = yellow_brake_decel_mps2;
    fg_cfg.voltage_decel_slope = voltage_decel_slope;
    fg_cfg.enable_brake_adaptation = enable_brake_adaptation;
    fg_cfg.control_latency_s = yellow_control_latency_s;
    fg_cfg.bumper_margin_m = yellow_bumper_margin_m;
    fg_cfg.rearm_travel_dist_m = yellow_rearm_dist_m;
    fg_cfg.enable_sqrt_brake_envelope = enable_sqrt_brake_envelope;
    fg_cfg.enable_closed_loop_brake = enable_closed_loop_brake;
    fg_cfg.brake_damping_gain = brake_damping_gain;
    fg_cfg.brake_zero_cutoff_mps = brake_zero_cutoff_mps;
    finish_guard.set_config(fg_cfg);

    params_dirty = false;
  }

  void reset(double now_sec)
  {
    if (params_dirty) {
      sync_configs();
    }
    track_state_machine.mutable_state() = line_follower::TrackRuntimeState{};
    track_state_machine.reset(now_sec);
    curve_assist = line_follower::CurveAssistState{};
    motion_state = line_follower::MotionState{};
    motion_controller.reset();
    obstacle_memory.reset();
    corridor_planner.reset();
    finish_guard.reset();

    odom_pose = line_follower::VehiclePose2D{};
    odom_pose_seen = false;
    last_odom_yaw = 0.0;
    last_odom_yaw_valid = false;
    last_odom_time_sec = now_sec;

    yellow_brake_active = false;
    yellow_brake_start_sec = now_sec;
    yellow_brake_measure_pending = false;
    yellow_brake_entry_speed_mps = 0.0;
    yellow_brake_start_arc_m = 0.0;
    has_stepped = false;
  }

  bool brake_pulse_on(double elapsed_s) const
  {
    if (yellow_brake_period_s <= 1e-3 ||
      yellow_brake_pulse_s <= 0.0 ||
      yellow_brake_pulse_s >= yellow_brake_period_s)
    {
      return true;
    }
    return std::fmod(elapsed_s, yellow_brake_period_s) < yellow_brake_pulse_s;
  }

  void update_odom(
    double x,
    double y,
    double yaw,
    double linear_v,
    double angular_w,
    double now_sec)
  {
    track_state_machine.update_odom(linear_v, now_sec);

    const double prev_x = odom_pose.x;
    const double prev_y = odom_pose.y;
    const bool had_prev = odom_pose_seen;

    odom_pose.valid = true;
    odom_pose.x = x;
    odom_pose.y = y;
    odom_pose.yaw = yaw;
    odom_pose.linear_v = linear_v;
    odom_pose.angular_w = angular_w;
    odom_pose.stamp_sec = now_sec;

    double dyaw = 0.0;
    if (last_odom_yaw_valid) {
      dyaw = wrap_angle_rad(yaw - last_odom_yaw);
    }
    last_odom_yaw = yaw;
    last_odom_yaw_valid = true;

    // Identical to LineFollowerNode::ekf_odom_callback (B7 forward projection)
    if (had_prev && std::abs(linear_v) >= 0.015) {
      const double dt = now_sec - last_odom_time_sec;
      if (dt > 0.0 && dt < 0.25) {
        const double dx = x - prev_x;
        const double dy = y - prev_y;
        const double ds_pos = std::hypot(dx, dy);
        const double ds_fwd = (ds_pos > 1e-4 && ds_pos < 0.5) ?
          (dx * std::cos(yaw) + dy * std::sin(yaw)) :
          (std::max(0.0, linear_v) * dt);
        finish_guard.update_motion(ds_fwd, dyaw);
      }
    }

    odom_pose_seen = true;
    last_odom_time_sec = now_sec;
  }

  void step(const BrainStepInput & in, BrainStepOutput & out)
  {
    using namespace line_follower;

    std::memset(&out, 0, sizeof(out));
    out.yellow_d_bumper_m = -1.0;
    out.corridor_left_px = -1;
    out.corridor_right_px = -1;
    out.corridor_ref_px = -1;
    out.corridor_row_y = -1;

    if (in.image_width > 0 && in.image_height > 0 &&
      (in.image_width != image_width || in.image_height != image_height))
    {
      image_width = in.image_width;
      image_height = in.image_height;
      params_dirty = true;
    }

    if (params_dirty) {
      sync_configs();
    }

    const double now_sec = in.timestamp_sec;
    if (!has_stepped) {
      track_state_machine.mutable_state().yellow_line_detection_time_sec = now_sec;
      track_state_machine.mutable_state().last_detection_time_sec = now_sec;
      has_stepped = true;
    }

    // 1. Update battery voltage & odom (matching ROS2 callbacks prior to image_callback)
    if (std::isfinite(in.voltage) && in.voltage > 0.0) {
      finish_guard.update_battery_voltage(in.voltage, now_sec);
    }
    if (in.has_odom != 0) {
      update_odom(in.odom_x, in.odom_y, in.odom_yaw, in.odom_v, in.odom_w, now_sec);
    }

    const int img_w = std::max(1, image_width);
    const int img_h = std::max(1, image_height);
    const double max_v_px = static_cast<double>(std::max(0, img_h - 1));
    const double image_center_x = 0.5 * static_cast<double>(img_w);

    // 2. Project YOLO / mock detections into bumper ground frame (matching lines 1131-1198)
    std::vector<GroundDetection> ground_detections;
    std::vector<GroundDetection> yolo_red_dets;

    if (in.detections != nullptr && in.num_detections > 0) {
      for (int32_t i = 0; i < in.num_detections; ++i) {
        const BrainDetectionC & box = in.detections[i];
        const SemanticObjectType obj_type = decode_obj_type(box.obj_type);
        if (obj_type == SemanticObjectType::UNKNOWN) {
          continue;
        }
        if (box.w <= 0.0 || box.h <= 0.0) {
          continue;
        }

        const double u_bot = box.x + 0.5 * box.w;
        double crc_pad_y = 0.0;
        if (obj_type == SemanticObjectType::RED_NO_ENTRY) {
          crc_pad_y = crc_no_entry_pad_y_px;
        } else if (obj_type == SemanticObjectType::RED_CONE) {
          crc_pad_y = crc_cone_pad_y_px;
        } else if (obj_type == SemanticObjectType::YELLOW_STOP_LINE) {
          crc_pad_y = crc_yellow_pad_y_px;
        }
        const double v_bot = std::clamp(box.y + box.h + crc_pad_y, 0.0, max_v_px);

        double d_m = 0.0;
        double y_m = 0.0;
        if (ground_projector.pixel_to_ground(u_bot, v_bot, d_m, y_m)) {
          GroundDetection gd{};
          gd.type = obj_type;
          gd.digit = box.digit;
          gd.confidence = box.confidence;
          gd.d_bumper_m = d_m;
          gd.y_lat_m = y_m;
          gd.track_id = box.track_id;
          gd.stamp_sec = now_sec;

          if (obj_type == SemanticObjectType::RED_CONE) {
            gd.half_len_m = 0.0675;
            gd.half_wid_m = 0.0675;
          } else if (obj_type == SemanticObjectType::RED_NO_ENTRY) {
            double dl = 0.0, yl = 0.0, dr = 0.0, yr = 0.0;
            double est_hw = 0.065;
            if (ground_projector.pixel_to_ground(box.x, v_bot, dl, yl) &&
              ground_projector.pixel_to_ground(box.x + box.w, v_bot, dr, yr))
            {
              est_hw = std::clamp(0.5 * std::abs(yl - yr), 0.065, 0.13);
            }
            gd.half_wid_m = est_hw;
            gd.half_len_m = (est_hw > 0.095) ? 0.065 : 0.12;
          } else {
            gd.half_len_m = 0.05;
            gd.half_wid_m = 0.05;
          }

          if (obj_type == SemanticObjectType::RED_CONE ||
            obj_type == SemanticObjectType::RED_NO_ENTRY)
          {
            yolo_red_dets.push_back(gd);
          } else {
            ground_detections.push_back(gd);
          }
        }
      }
    }

    const int active_branch = corridor_planner.committed_branch(finish_guard.total_arc_m());
    const bool line_detected = (in.lane_valid != 0);

    // 3. Project CV red boxes & run exact OR-fusion (matching lines 1208-1220 & 1940-1998)
    {
      std::vector<GroundDetection> cv_red_dets;
      if (enable_cv_red_fallback && line_detected &&
        in.cv_red_boxes != nullptr && in.num_cv_red_boxes > 0)
      {
        for (int32_t i = 0; i < in.num_cv_red_boxes; ++i) {
          const BrainCvRedBoxC & r_img = in.cv_red_boxes[i];
          if (r_img.w <= 0.0 || r_img.h <= 0.0) {
            continue;
          }
          const double u_bot = r_img.x + 0.5 * r_img.w;
          const double v_raw = std::clamp(r_img.y + r_img.h, 0.0, max_v_px);
          double d_m = 0.0;
          double y_m = 0.0;
          if (!ground_projector.pixel_to_ground(u_bot, v_raw, d_m, y_m)) {
            continue;
          }

          double dl = 0.0, yl = 0.0, dr = 0.0, yr = 0.0;
          double est_width_m = 0.135;
          if (ground_projector.pixel_to_ground(r_img.x, v_raw, dl, yl) &&
            ground_projector.pixel_to_ground(r_img.x + r_img.w, v_raw, dr, yr))
          {
            est_width_m = std::clamp(std::abs(yl - yr), 0.10, 0.26);
          }

          bool looks_no_entry = false;
          if (r_img.fill_ratio >= 0.0) {
            looks_no_entry = (r_img.fill_ratio < 0.42 && r_img.w >= 18.0) || (est_width_m > 0.155);
          } else {
            looks_no_entry = (r_img.is_no_entry != 0) || (est_width_m > 0.155);
          }

          const double crc_pad_y = looks_no_entry ? crc_no_entry_pad_y_px : crc_cone_pad_y_px;
          const double v_bot = std::clamp(r_img.y + r_img.h + crc_pad_y, 0.0, max_v_px);
          if (std::abs(v_bot - v_raw) > 1e-6) {
            if (!ground_projector.pixel_to_ground(u_bot, v_bot, d_m, y_m)) {
              continue;
            }
            if (ground_projector.pixel_to_ground(r_img.x, v_bot, dl, yl) &&
              ground_projector.pixel_to_ground(r_img.x + r_img.w, v_bot, dr, yr))
            {
              est_width_m = std::clamp(std::abs(yl - yr), 0.10, 0.26);
            }
          }

          if (d_m < 0.02 || d_m > 1.45 || std::abs(y_m) > 0.35) {
            continue;
          }

          GroundDetection gd{};
          gd.type = looks_no_entry ?
            SemanticObjectType::RED_NO_ENTRY : SemanticObjectType::RED_CONE;
          gd.confidence = (r_img.confidence > 0.0) ? r_img.confidence : 0.60;
          gd.d_bumper_m = d_m;
          gd.y_lat_m = y_m;
          gd.stamp_sec = now_sec;
          if (looks_no_entry) {
            gd.half_wid_m = std::clamp(0.5 * est_width_m, 0.065, 0.13);
            gd.half_len_m = (gd.half_wid_m > 0.095) ? 0.065 : 0.12;
          } else {
            gd.half_wid_m = std::clamp(0.5 * est_width_m, 0.0675, 0.10);
            gd.half_len_m = 0.0675;
          }
          cv_red_dets.push_back(gd);
        }
      }

      const RedFusionSummary red_fusion = fuse_red_obstacles_or(yolo_red_dets, cv_red_dets, 0.24);
      for (const auto & gd : red_fusion.fused_red_detections) {
        ground_detections.push_back(gd);
      }
    }

    // 4. ObstacleMemory update (matching lines 1314-1335)
    std::vector<LocalObstacle> local_obstacles;
    if (enable_obstacle_memory) {
      obstacle_memory.update(ground_detections, odom_pose, now_sec);
      local_obstacles = obstacle_memory.get_local_obstacles(odom_pose, now_sec);
    } else {
      for (const auto & gd : ground_detections) {
        if (gd.type == SemanticObjectType::RED_CONE ||
          gd.type == SemanticObjectType::RED_NO_ENTRY)
        {
          LocalObstacle loc{};
          loc.type = gd.type;
          loc.d_bumper_m = gd.d_bumper_m;
          loc.d_min_m = gd.d_bumper_m - gd.half_len_m;
          loc.d_max_m = gd.d_bumper_m + gd.half_len_m;
          loc.y_lat_m = gd.y_lat_m;
          loc.y_min_m = gd.y_lat_m - gd.half_wid_m;
          loc.y_max_m = gd.y_lat_m + gd.half_wid_m;
          loc.confidence = gd.confidence;
          local_obstacles.push_back(loc);
        }
      }
    }

    // 5. Two-stage zero-touch yellow line stop & FinishGuard gating (matching lines 1340-1390)
    const bool raw_yellow_detected = (in.yellow_detected != 0);
    const bool dead_end = (in.yellow_dead_end != 0);

    bool has_metric_dist = false;
    double yellow_d_bumper_m = 0.0;
    double yellow_y_lat_m = 0.0;

    if (raw_yellow_detected && (in.yellow_is_geometric != 0)) {
      double y_bot_full = in.yellow_bottom_y;
      if (y_bot_full > 0.0 && y_bot_full <= 1.0) {
        y_bot_full *= static_cast<double>(img_h);
      }
      const double y_cx_full = (in.yellow_center_x >= 0.0) ? in.yellow_center_x : image_center_x;
      const double v_yellow_bot = std::clamp(y_bot_full + crc_yellow_pad_y_px, 0.0, max_v_px);
      has_metric_dist = ground_projector.pixel_to_ground(
        y_cx_full, v_yellow_bot, yellow_d_bumper_m, yellow_y_lat_m);
    }

    const double cur_v = track_state_machine.state().odom_seen ?
      track_state_machine.state().odom_speed : motion_state.last_linear;

    const YellowStopAdvice yellow_advice = finish_guard.evaluate_yellow_stop(
      raw_yellow_detected,
      has_metric_dist,
      yellow_d_bumper_m,
      cur_v,
      dead_end,
      now_sec);

    const bool already_stopping =
      track_state_machine.state().yellow_line_stop_active ||
      track_state_machine.state().permanent_stop;
    const bool sm_yellow_trigger = already_stopping ?
      raw_yellow_detected :
      (finish_guard.is_yellow_rearmed() && yellow_advice.trigger_stop);
    const bool external_is_finish = yellow_advice.is_finish_line;

    // 6. TrackStateMachine step & yellow brake / friction adaptation (matching lines 1391-1504)
    const TrackStepEvents step_events =
      track_state_machine.step(sm_yellow_trigger, line_detected, now_sec, external_is_finish);
    const auto & sm_state = track_state_machine.state();

    if (step_events.yellow_stop_begin) {
      finish_guard.mark_yellow_stop_triggered();
      corridor_planner.reset_fork_votes();

      yellow_brake_entry_speed_mps = sm_state.odom_seen ?
        sm_state.odom_speed : std::abs(motion_state.last_linear);
      yellow_brake_start_arc_m = finish_guard.total_arc_m();
      yellow_brake_measure_pending = true;

      const bool entry_fast_enough_for_brake =
        finish_guard.config().enable_closed_loop_brake ?
        (!sm_state.odom_seen || (sm_state.odom_linear_x > finish_guard.config().brake_zero_cutoff_mps)) :
        (!sm_state.odom_seen || (sm_state.odom_linear_x > yellow_brake_release_speed));
      yellow_brake_active = (yellow_brake_speed > 0.0) && entry_fast_enough_for_brake;
      yellow_brake_start_sec = now_sec;
    } else if (yellow_brake_measure_pending) {
      if (step_events.odom_still_ok) {
        yellow_brake_measure_pending = false;
        const double stop_travel_m =
          finish_guard.total_arc_m() - yellow_brake_start_arc_m;
        finish_guard.update_measured_yellow_stop(
          yellow_brake_entry_speed_mps, stop_travel_m, now_sec);
      } else if (!step_events.yellow_hold_stop) {
        yellow_brake_measure_pending = false;
      }
    }

    if (step_events.yellow_stop_end) {
      yellow_brake_active = false;
      const int chosen = corridor_planner.finalize_fork_decision(finish_guard.total_arc_m());
      if (chosen != 0) {
        finish_guard.mark_fork_passed();
      }
    }

    if (step_events.yellow_hold_stop && !sm_state.permanent_stop) {
      corridor_planner.update_fork_static_votes(ground_detections, now_sec);
    }

    // 7. Control hierarchy: Yellow Stop > Finals Corridor > Lost-Line Safety
    double linear_speed = 0.0;
    double yaw_rate = 0.0;
    TrackState current_state = TrackState::LOST;
    bool publish_stop_cmd = false;

    double y_lane_left_m = 0.5 * corridor_planner.config().default_lane_width_m;
    double y_lane_right_m = -y_lane_left_m;
    CorridorPlan plan{};
    plan.safe_y_max_m = y_lane_left_m;
    plan.safe_y_min_m = y_lane_right_m;
    plan.target_y_m = 0.0;
    plan.speed_cap_mps = target_speed;
    plan.preferred_branch = corridor_planner.committed_branch(finish_guard.total_arc_m());
    plan.active_mode_tag = "lost";

    bool cbf_intervened = false;
    double cbf_min_clearance_m = 0.0;
    std::string cbf_limiting_source = "none";

    const double a_eff_now = finish_guard.effective_brake_decel_mps2(now_sec);
    double v_safe_cap_mps = 0.0;
    double clear_path_score = 0.0;
    double brake_cmd_mps = 0.0;

    if (step_events.yellow_hold_stop) {
      current_state = TrackState::YELLOW_LINE_DETECTED;
      linear_speed = 0.0;
      yaw_rate = 0.0;
      publish_stop_cmd = true;
      plan.active_mode_tag = sm_state.permanent_stop ? "finish_stop" : "yellow_stop";
      plan.speed_cap_mps = 0.0;

      if (yellow_brake_active) {
        const double brake_elapsed = now_sec - yellow_brake_start_sec;
        if (finish_guard.config().enable_closed_loop_brake && sm_state.odom_seen) {
          const double cl_cmd = finish_guard.compute_closed_loop_brake_cmd(
            sm_state.odom_linear_x,
            brake_elapsed,
            yellow_brake_speed,
            yellow_brake_max_time_s);
          if (cl_cmd < -1e-9) {
            publish_stop_cmd = false;
            linear_speed = cl_cmd;
            brake_cmd_mps = cl_cmd;
          } else {
            yellow_brake_active = false;
          }
        } else {
          // Signed velocity zero-reverse guard (matching line_follower_node.cpp)
          const bool reverse_guard =
            sm_state.odom_seen && sm_state.odom_linear_x <= 0.01;
          const bool slow_enough =
            yellow_brake_release_speed > 0.0 &&
            brake_elapsed >= 0.04 &&
            sm_state.odom_seen &&
            sm_state.odom_linear_x <= yellow_brake_release_speed;
          const bool timeout =
            yellow_brake_max_time_s > 0.0 && brake_elapsed >= yellow_brake_max_time_s;

          if (reverse_guard || slow_enough || timeout) {
            yellow_brake_active = false;
          } else {
            publish_stop_cmd = false;
            linear_speed = brake_pulse_on(brake_elapsed) ? -yellow_brake_speed : 0.0;
            brake_cmd_mps = linear_speed;
          }
        }
      }
      motion_controller.notify_external_speed(0.0, now_sec);
    } else if (line_detected) {
      current_state = TrackState::TRACKING;

      const bool near_valid = (in.near_valid < 0) ? true : (in.near_valid != 0);
      const bool far_valid = (in.far_valid < 0) ? true : (in.far_valid != 0);
      const double near_err = near_valid ? in.near_error : 0.0;
      const double far_err = far_valid ? in.far_error : 0.0;

      // Update CurveAssistState (matching lane_geometry.cpp lines 332-384)
      curve_assist.near_valid = near_valid;
      curve_assist.far_valid = far_valid;
      curve_assist.near_error = near_err;
      curve_assist.far_error = far_err;
      curve_assist.curve_level = 0.0;
      if (near_valid) {
        curve_assist.curve_level = std::max(curve_assist.curve_level, std::abs(near_err));
      }
      if (far_valid) {
        curve_assist.curve_level = std::max(curve_assist.curve_level, std::abs(far_err));
      }

      if (in.use_external_reverse_curve != 0) {
        curve_assist.reverse_curve_active = (in.reverse_curve_active != 0);
      } else {
        bool reverse_curve_candidate = false;
        if (enable_continuous_curve_assist && near_valid && far_valid) {
          const bool opposite_direction = (near_err * far_err) < 0.0;
          const bool both_significant =
            std::abs(near_err) >= reverse_curve_min_error &&
            std::abs(far_err) >= reverse_curve_min_error;
          const bool large_difference =
            std::abs(far_err - near_err) >= reverse_curve_threshold;
          reverse_curve_candidate = opposite_direction && both_significant && large_difference;
        }

        if (!enable_continuous_curve_assist) {
          curve_assist.reverse_curve_active = false;
          curve_assist.reverse_confirm_count = 0;
          curve_assist.reverse_clear_count = 0;
        } else if (reverse_curve_candidate) {
          curve_assist.reverse_clear_count = 0;
          curve_assist.reverse_confirm_count++;
          if (!curve_assist.reverse_curve_active &&
            curve_assist.reverse_confirm_count >= reverse_curve_confirm_frames)
          {
            curve_assist.reverse_curve_active = true;
            curve_assist.reverse_confirm_count = reverse_curve_confirm_frames;
          }
        } else {
          curve_assist.reverse_confirm_count = 0;
          if (curve_assist.reverse_curve_active) {
            curve_assist.reverse_clear_count++;
            if (curve_assist.reverse_clear_count >= reverse_curve_clear_frames) {
              curve_assist.reverse_curve_active = false;
              curve_assist.reverse_clear_count = 0;
            }
          } else {
            curve_assist.reverse_clear_count = 0;
          }
        }
      }

      double lane_width_ratio = in.lane_width_ratio;
      if (lane_width_ratio < 0.0) {
        if (near_valid && far_valid) {
          lane_width_ratio = 0.5 * (in.near_width_ratio + in.far_width_ratio);
        } else if (near_valid) {
          lane_width_ratio = in.near_width_ratio;
        } else if (far_valid) {
          lane_width_ratio = in.far_width_ratio;
        } else {
          lane_width_ratio = 0.0;
        }
      }
      curve_assist.last_lane_width_ratio = lane_width_ratio;
      curve_assist.last_confidence = in.confidence;
      curve_assist.last_heading_proxy = in.heading_error;

      double curvature_proxy = in.curvature_proxy;
      if (curvature_proxy < 0.0) {
        double base_curve = (near_valid && far_valid) ?
          std::abs(far_err - near_err) : std::abs(in.heading_error);
        if (enable_width_adaptive_fusion) {
          const double max_w = std::max(in.near_width_ratio, in.far_width_ratio);
          if (max_w > wide_road_ratio_threshold) {
            const double w_excess =
              (max_w - wide_road_ratio_threshold) /
              std::max(1e-3, 1.0 - wide_road_ratio_threshold);
            base_curve = std::max(base_curve, 0.35 * clamp_value(w_excess, 0.0, 1.0));
          }
        }
        curvature_proxy = clamp_value(base_curve, 0.0, 1.0);
      }
      curve_assist.last_curvature_proxy = curvature_proxy;

      const bool has_fork_split = (in.fork_detected != 0);
      if (has_fork_split) {
        finish_guard.mark_fork_passed();
      }

      // Compute centroid_x in pixels
      double lateral_err = in.lateral_error;
      if (in.has_lateral_error == 0) {
        if (near_valid && far_valid) {
          if (enable_continuous_curve_assist && curve_assist.reverse_curve_active) {
            lateral_err = near_err;
          } else {
            double wn = std::max(0.0, near_band_weight);
            double wf = std::max(0.0, far_band_weight);
            const double wsum = wn + wf;
            if (wsum <= 1e-9) {
              wn = 0.60;
              wf = 0.40;
            } else {
              wn /= wsum;
              wf /= wsum;
            }
            lateral_err = wn * near_err + wf * far_err;
          }
        } else if (near_valid) {
          lateral_err = near_err;
        } else if (far_valid) {
          lateral_err = far_err;
        } else {
          lateral_err = 0.0;
        }
      }
      lateral_err = clamp_value(lateral_err, -1.0, 1.0);
      const double centroid_x_px = clamp_value(
        image_center_x + lateral_err * image_center_x,
        0.0,
        static_cast<double>(img_w - 1));

      // Project near-band lane edges into vehicle bumper frame (matching lines 1548-1595)
      const int roi_top = std::clamp(
        static_cast<int>(static_cast<double>(img_h) * (1.0 - roi_top_ratio)),
        0,
        img_h - 1);
      const int roi_bottom = std::clamp(
        img_h - static_cast<int>(static_cast<double>(img_h) * roi_bottom_ratio),
        roi_top + 1,
        img_h);
      const double roi_h = static_cast<double>(roi_bottom - roi_top);
      const double v_near =
        static_cast<double>(roi_top) +
        0.5 * (near_band_top_ratio + near_band_bottom_ratio) * roi_h;
      const double half_w_def = 0.5 * corridor_planner.config().default_lane_width_m;

      double d_tmp = 0.0;
      double y_center_m = 0.0;
      const bool center_ok = ground_projector.pixel_to_ground(centroid_x_px, v_near, d_tmp, y_center_m);

      const double left_edge_px = (in.edges_are_px != 0) ?
        in.left_edge :
        clamp_value((in.left_edge + 1.0) * image_center_x, 0.0, static_cast<double>(img_w - 1));
      const double right_edge_px = (in.edges_are_px != 0) ?
        in.right_edge :
        clamp_value((in.right_edge + 1.0) * image_center_x, 0.0, static_cast<double>(img_w - 1));

      const bool near_left_valid = near_valid && (in.has_left_edge != 0);
      const bool near_right_valid = near_valid && (in.has_right_edge != 0);
      const bool left_clipped = near_valid && (in.has_left_edge == 0);
      const bool right_clipped = near_valid && (in.has_right_edge == 0);

      // Prevent far-band lookahead from shifting near corridor center onto the inside apex
      // when both near edges clip on a tight curve (matching line_follower_node.cpp)
      if (center_ok && !has_fork_split && active_branch == 0 &&
        near_valid && left_clipped && right_clipped &&
        lane_width_ratio < wide_road_ratio_threshold &&
        std::abs(far_err) > std::abs(near_err) + 0.15)
      {
        const double near_cx_px = clamp_value(
          image_center_x + near_err * image_center_x,
          0.0,
          static_cast<double>(img_w - 1));
        double d_near_tmp = 0.0;
        double y_near_m = 0.0;
        if (ground_projector.pixel_to_ground(near_cx_px, v_near, d_near_tmp, y_near_m)) {
          y_center_m = y_near_m;
        }
      }

      y_lane_left_m = center_ok ? (y_center_m + half_w_def) : half_w_def;
      y_lane_right_m = center_ok ? (y_center_m - half_w_def) : -half_w_def;

      double d_l = 0.0, y_l = 0.0;
      double d_r = 0.0, y_r = 0.0;
      const bool left_proj_ok = near_left_valid &&
        ground_projector.pixel_to_ground(left_edge_px, v_near, d_l, y_l);
      const bool right_proj_ok = near_right_valid &&
        ground_projector.pixel_to_ground(right_edge_px, v_near, d_r, y_r);

      const double measured_span = y_l - y_r;
      if (!has_fork_split && left_proj_ok && right_proj_ok &&
        measured_span >= 0.30 && measured_span <= 0.52 &&
        (active_branch == 0 ||
          (center_ok && std::abs(y_center_m - 0.5 * (y_l + y_r)) <= 0.04)))
      {
        y_lane_left_m = y_l;
        y_lane_right_m = y_r;
      } else if (!has_fork_split && active_branch == 0 && left_proj_ok && right_proj_ok &&
        measured_span > 0.52 && measured_span <= 0.68 &&
        lane_width_ratio < wide_road_ratio_threshold)
      {
        const double y_mid = 0.5 * (y_l + y_r);
        y_lane_left_m = y_mid + half_w_def;
        y_lane_right_m = y_mid - half_w_def;
      } else if (!has_fork_split && active_branch == 0 && near_valid) {
        const double def_w = corridor_planner.config().default_lane_width_m;
        if (left_clipped && !right_clipped && right_proj_ok) {
          double d_clip = 0.0;
          double y_l_clip = 0.0;
          const bool clip_ok = ground_projector.pixel_to_ground(0.0, v_near, d_clip, y_l_clip);
          const double vis_span = y_l_clip - y_r;
          if (clip_ok && lane_width_ratio < wide_road_ratio_threshold &&
            vis_span > def_w && vis_span <= 0.68)
          {
            const double y_mid = 0.5 * (y_l_clip + y_r);
            y_lane_left_m = y_mid + half_w_def;
            y_lane_right_m = y_mid - half_w_def;
          } else if (y_r >= -0.45 && y_r <= 0.25 && y_r > y_lane_right_m) {
            y_lane_right_m = y_r;
            y_lane_left_m = y_r + def_w;
          }
        } else if (right_clipped && !left_clipped && left_proj_ok) {
          double d_clip = 0.0;
          double y_r_clip = 0.0;
          const double max_u_px = static_cast<double>(std::max(1, img_w - 1));
          const bool clip_ok = ground_projector.pixel_to_ground(max_u_px, v_near, d_clip, y_r_clip);
          const double vis_span = y_l - y_r_clip;
          if (clip_ok && lane_width_ratio < wide_road_ratio_threshold &&
            vis_span > def_w && vis_span <= 0.68)
          {
            const double y_mid = 0.5 * (y_l + y_r_clip);
            y_lane_left_m = y_mid + half_w_def;
            y_lane_right_m = y_mid - half_w_def;
          } else if (y_l >= -0.25 && y_l <= 0.45 && y_l < y_lane_left_m) {
            y_lane_left_m = y_l;
            y_lane_right_m = y_l - def_w;
          }
        }
      }

      double eff_curve_lvl = std::max(curvature_proxy, curve_assist.curve_level);
      if (lane_width_ratio >= 0.85 ||
        (motion_controller.config().low_confidence_threshold > 0.0 &&
        in.confidence < motion_controller.config().low_confidence_threshold))
      {
        eff_curve_lvl = std::max(eff_curve_lvl, 0.85);
      }

      const double curve_hdg_for_apex =
        (!has_fork_split && active_branch == 0 &&
        lane_width_ratio < wide_road_ratio_threshold &&
        std::abs(far_err) >= std::abs(near_err)) ?
        in.heading_error : 0.0;

      plan = corridor_planner.plan(
        y_lane_left_m,
        y_lane_right_m,
        0.0,
        local_obstacles,
        ground_detections,
        false,
        finish_guard.total_arc_m(),
        now_sec,
        eff_curve_lvl,
        curve_hdg_for_apex);

      const bool corridor_event_active =
        plan.has_blocking_obstacle ||
        plan.near_obstacle_count > 0 ||
        plan.far_obstacle_count > 0 ||
        plan.digit_target_active ||
        plan.preferred_branch != 0 ||
        plan.active_mode_tag != "cruise";

      double eff_speed_cap = corridor_event_active ?
        plan.speed_cap_mps :
        (motion_controller.config().enable_clear_path_speed ?
          std::max(target_speed, motion_controller.config().clear_straight_speed) :
          plan.speed_cap_mps);
      if (finish_guard.is_yellow_rearmed() &&
        yellow_advice.pre_brake &&
        !yellow_advice.trigger_stop)
      {
        eff_speed_cap = std::min(eff_speed_cap, yellow_advice.speed_cap_mps);
      }

      double eff_preview_m = motion_controller.config().clear_preview_dist_m;
      if (!curve_assist.far_valid) {
        eff_preview_m = std::min(eff_preview_m, 0.50 * motion_controller.config().clear_preview_dist_m);
      }
      if (in.confidence < 0.70) {
        eff_preview_m *= clamp_value(in.confidence / 0.70, 0.35, 1.0);
      }
      if (has_metric_dist && yellow_d_bumper_m > 0.0 && finish_guard.is_yellow_rearmed()) {
        eff_preview_m = std::min(eff_preview_m, yellow_d_bumper_m);
      }
      v_safe_cap_mps = finish_guard.compute_safe_speed_ceiling(eff_preview_m, 0.0, now_sec);
      bool has_no_entry_ahead = false;
      for (const auto & loc : local_obstacles) {
        if (loc.type == SemanticObjectType::RED_NO_ENTRY &&
          loc.d_max_m >= -corridor_planner.config().obstacle_rear_keep_m &&
          loc.d_min_m <= corridor_planner.config().obstacle_lookahead_m)
        {
          has_no_entry_ahead = true;
        }
        if (loc.d_min_m > 0.0 && loc.d_min_m <= eff_preview_m) {
          const double obs_end_v = (loc.type == SemanticObjectType::RED_NO_ENTRY) ?
            corridor_planner.config().no_entry_speed_mps :
            corridor_planner.config().obstacle_speed_mps;
          v_safe_cap_mps = std::min(
            v_safe_cap_mps,
            finish_guard.compute_safe_speed_ceiling(loc.d_min_m, obs_end_v, now_sec));
        }
      }

      ClearPathInput cp_in{};
      cp_in.has_active_obstacle =
        plan.has_blocking_obstacle ||
        plan.near_obstacle_count > 0 ||
        plan.far_obstacle_count > 0;
      cp_in.has_no_entry_ahead = has_no_entry_ahead;
      cp_in.has_yellow_ahead =
        finish_guard.is_yellow_rearmed() &&
        (raw_yellow_detected || yellow_advice.pre_brake || yellow_advice.trigger_stop);
      cp_in.has_digit_target = plan.digit_target_active;
      cp_in.has_fork_context = (plan.preferred_branch != 0) || has_fork_split;
      cp_in.confidence = in.confidence;
      cp_in.curvature_proxy = curvature_proxy;
      cp_in.heading_proxy = in.heading_error;
      cp_in.lateral_err_m = plan.target_y_m;
      cp_in.lane_width_ratio = lane_width_ratio;
      cp_in.reverse_curve_active = curve_assist.reverse_curve_active;
      cp_in.v_safe_cap_mps = v_safe_cap_mps;
      cp_in.odom_linear_x = sm_state.odom_linear_x;
      cp_in.odom_valid = sm_state.odom_seen;

      const AdaptiveSpeedResult adaptive_res = motion_controller.step_adaptive_corridor_command(
        plan.target_y_m,
        0.0,
        in.heading_error,
        curvature_proxy,
        in.confidence,
        eff_speed_cap,
        curve_assist,
        motion_state,
        cp_in,
        now_sec);

      const MotionCommand & cmd = adaptive_res.command;
      clear_path_score = adaptive_res.clear_path_score;
      plan.speed_cap_mps = adaptive_res.effective_speed_cap_mps;

      linear_speed = cmd.linear_speed;
      yaw_rate = cmd.yaw_rate;
      publish_stop_cmd = cmd.publish_stop;

      if (enable_cbf_shield) {
        const CbfShieldResult cbf = cbf_shield.filter(
          linear_speed,
          yaw_rate,
          y_lane_left_m,
          y_lane_right_m,
          local_obstacles);
        cbf_intervened = cbf.intervened;
        cbf_min_clearance_m = cbf.min_clearance_m;
        cbf_limiting_source = cbf.limiting_source;
        if (cbf.intervened) {
          linear_speed = cbf.linear_speed;
          yaw_rate = cbf.yaw_rate;
          if (linear_speed <= 1e-6 && std::abs(yaw_rate) <= 1e-6) {
            publish_stop_cmd = true;
          }
        }
      }

      out.corridor_row_y = static_cast<int32_t>(std::round(v_near));
      const double d_ref = center_ok ? std::max(0.15, d_tmp) : 0.35;
      double u_l = 0.0, v_l = 0.0, u_r = 0.0, v_r = 0.0, u_ref = 0.0, v_ref = 0.0;
      if (ground_projector.ground_to_pixel(d_ref, plan.safe_y_max_m, u_l, v_l)) {
        out.corridor_left_px = static_cast<int32_t>(std::round(u_l));
      }
      if (ground_projector.ground_to_pixel(d_ref, plan.safe_y_min_m, u_r, v_r)) {
        out.corridor_right_px = static_cast<int32_t>(std::round(u_r));
      }
      if (ground_projector.ground_to_pixel(d_ref, plan.target_y_m, u_ref, v_ref)) {
        out.corridor_ref_px = static_cast<int32_t>(std::round(u_ref));
      }
    } else {
      current_state = step_events.base_state;
      const MotionCommand cmd =
        motion_controller.compute_lost_line_command(step_events.since_line_detection, motion_state);
      linear_speed = cmd.linear_speed;
      yaw_rate = cmd.yaw_rate;
      publish_stop_cmd = cmd.publish_stop;
      plan.active_mode_tag = "lost";
      plan.speed_cap_mps = linear_speed;
      v_safe_cap_mps = linear_speed;
      motion_controller.notify_external_speed(linear_speed, now_sec);
    }

    const bool yellow_stop_now = step_events.yellow_stop_now;

    // Final safety lock (matching line_follower_node.cpp)
    if (step_events.yellow_hold_stop) {
      current_state = TrackState::YELLOW_LINE_DETECTED;
      yaw_rate = 0.0;
      if (yellow_brake_active) {
        const double brake_elapsed = now_sec - yellow_brake_start_sec;
        if (finish_guard.config().enable_closed_loop_brake && sm_state.odom_seen) {
          linear_speed = finish_guard.compute_closed_loop_brake_cmd(
            sm_state.odom_linear_x,
            brake_elapsed,
            yellow_brake_speed,
            yellow_brake_max_time_s);
        } else {
          linear_speed = brake_pulse_on(brake_elapsed) ? -yellow_brake_speed : 0.0;
        }
        brake_cmd_mps = linear_speed;
        publish_stop_cmd = (linear_speed >= -1e-9);
      } else {
        linear_speed = 0.0;
        brake_cmd_mps = 0.0;
        publish_stop_cmd = true;
      }
    }

    MotionController::record_motion(yellow_stop_now, linear_speed, yaw_rate, motion_state);
    track_state_machine.commit_state(current_state);

    // Populate BrainStepOutput
    out.v_cmd = linear_speed;
    out.w_cmd = yaw_rate;
    out.state = static_cast<int32_t>(current_state);
    out.publish_stop = publish_stop_cmd ? 1 : 0;

    out.corridor_left_m = plan.safe_y_max_m;
    out.corridor_right_m = plan.safe_y_min_m;
    out.corridor_target_y_m = plan.target_y_m;
    out.lane_left_m = y_lane_left_m;
    out.lane_right_m = y_lane_right_m;
    out.speed_cap_mps = plan.speed_cap_mps;

    out.active_digit = plan.digit_target_active ? plan.target_digit : 0;
    out.digit_target_active = plan.digit_target_active ? 1 : 0;
    out.obstacle_count = static_cast<int32_t>(local_obstacles.size());
    out.near_obstacle_count = plan.near_obstacle_count;
    out.far_obstacle_count = plan.far_obstacle_count;
    out.has_blocking_obstacle = plan.has_blocking_obstacle ? 1 : 0;
    out.preferred_branch = corridor_planner.committed_branch(finish_guard.total_arc_m());

    out.cbf_active = cbf_intervened ? 1 : 0;
    out.cbf_min_clearance_m = cbf_min_clearance_m;

    out.finish_armed = finish_guard.is_finish_armed() ? 1 : 0;
    out.permanent_stop = sm_state.permanent_stop ? 1 : 0;
    out.yellow_stop_active = sm_state.yellow_line_stop_active ? 1 : 0;
    out.yellow_brake_active = yellow_brake_active ? 1 : 0;
    out.yellow_count = sm_state.yellow_line_detection_count;
    out.yellow_dist_valid = has_metric_dist ? 1 : 0;
    out.yellow_is_finish = yellow_advice.is_finish_line ? 1 : 0;
    out.yellow_rearmed = finish_guard.is_yellow_rearmed() ? 1 : 0;
    out.yellow_d_bumper_m = has_metric_dist ? yellow_d_bumper_m : -1.0;

    out.total_arc_m = finish_guard.total_arc_m();
    out.friction_scale = finish_guard.friction_scale();
    out.effective_brake_decel = a_eff_now;
    out.v_safe_cap_mps = v_safe_cap_mps;
    out.clear_path_score = clear_path_score;
    out.brake_cmd_mps = brake_cmd_mps;
    out.effective_brake_decel_mps2 = a_eff_now;

    std::string yphase = "none";
    if (step_events.yellow_hold_stop) {
      if (yellow_brake_active) {
        yphase = "braking";
      } else if (sm_state.permanent_stop) {
        yphase = "finished";
      } else {
        yphase = "stopped";
      }
    } else if (finish_guard.is_yellow_rearmed() && yellow_advice.pre_brake) {
      yphase = "pre_brake";
    }

    copy_str(out.state_name, sizeof(out.state_name), track_state_to_string(current_state));
    copy_str(out.active_mode_tag, sizeof(out.active_mode_tag), plan.active_mode_tag);
    copy_str(out.yellow_phase, sizeof(out.yellow_phase), yphase);
    copy_str(out.cbf_limiting_source, sizeof(out.cbf_limiting_source), cbf_limiting_source);
  }
};

}  // namespace

extern "C" {

BRAIN_API void * brain_create()
{
  return new BrainInstance();
}

BRAIN_API void brain_destroy(void * handle)
{
  auto * inst = static_cast<BrainInstance *>(handle);
  delete inst;
}

BRAIN_API void brain_reset(void * handle, double now_sec)
{
  auto * inst = static_cast<BrainInstance *>(handle);
  if (inst == nullptr) {
    return;
  }
  inst->reset(now_sec);
}

BRAIN_API int32_t brain_set_param_double(void * handle, const char * name, double value)
{
  auto * inst = static_cast<BrainInstance *>(handle);
  if (inst == nullptr || name == nullptr || !std::isfinite(value)) {
    return -1;
  }
  const std::string k(name);

  if (k == "target_speed") {
    inst->target_speed = value;
  } else if (k == "max_speed") {
    inst->max_speed = value;
  } else if (k == "min_speed") {
    inst->min_speed = value;
  } else if (k == "max_yaw_rate") {
    inst->max_yaw_rate = value;
  } else if (k == "speed_gain_damping") {
    inst->speed_gain_damping = value;
  } else if (k == "roi_top_ratio") {
    inst->roi_top_ratio = value;
  } else if (k == "roi_bottom_ratio") {
    inst->roi_bottom_ratio = value;
  } else if (k == "yellow_roi_bottom_ratio") {
    inst->yellow_roi_bottom_ratio = value;
  } else if (k == "far_band_top_ratio") {
    inst->far_band_top_ratio = value;
  } else if (k == "far_band_bottom_ratio") {
    inst->far_band_bottom_ratio = value;
  } else if (k == "near_band_top_ratio") {
    inst->near_band_top_ratio = value;
  } else if (k == "near_band_bottom_ratio") {
    inst->near_band_bottom_ratio = value;
  } else if (k == "near_band_weight") {
    inst->near_band_weight = value;
  } else if (k == "far_band_weight") {
    inst->far_band_weight = value;
  } else if (k == "wide_road_ratio_threshold") {
    inst->wide_road_ratio_threshold = value;
  } else if (k == "reverse_curve_min_error") {
    inst->reverse_curve_min_error = value;
  } else if (k == "reverse_curve_threshold") {
    inst->reverse_curve_threshold = value;
  } else if (k == "low_confidence_threshold") {
    inst->low_confidence_threshold = value;
  } else if (k == "low_confidence_speed") {
    inst->low_confidence_speed = value;
  } else if (k == "low_confidence_yaw_hold") {
    inst->low_confidence_yaw_hold = value;
  } else if (k == "yellow_stop_duration") {
    inst->yellow_stop_duration = value;
  } else if (k == "yellow_stop_min_time") {
    inst->yellow_stop_min_time = value;
  } else if (k == "odom_still_threshold") {
    inst->odom_still_threshold = value;
  } else if (k == "odom_still_hold") {
    inst->odom_still_hold = value;
  } else if (k == "odom_max_age_sec") {
    inst->odom_max_age_sec = value;
  } else if (k == "yellow_max_hold_sec") {
    inst->yellow_max_hold_sec = value;
  } else if (k == "yellow_brake_speed") {
    inst->yellow_brake_speed = value;
  } else if (k == "yellow_brake_period_s") {
    inst->yellow_brake_period_s = value;
  } else if (k == "yellow_brake_pulse_s") {
    inst->yellow_brake_pulse_s = value;
  } else if (k == "yellow_brake_release_speed") {
    inst->yellow_brake_release_speed = value;
  } else if (k == "yellow_brake_max_time_s") {
    inst->yellow_brake_max_time_s = value;
  } else if (k == "line_lost_stop_time") {
    inst->line_lost_stop_time = value;
  } else if (k == "line_lost_forward_speed") {
    inst->line_lost_forward_speed = value;
  } else if (k == "line_lost_yaw_hold_factor") {
    inst->line_lost_yaw_hold_factor = value;
  } else if (k == "line_lost_slow_time") {
    inst->line_lost_slow_time = value;
  } else if (k == "line_lost_slow_factor") {
    inst->line_lost_slow_factor = value;
  } else if (k == "line_lost_full_stop_time") {
    inst->line_lost_full_stop_time = value;
  } else if (k == "lost_line_safe_speed") {
    inst->lost_line_safe_speed = value;
  } else if (k == "lost_line_safe_stop_time") {
    inst->lost_line_safe_stop_time = value;
  } else if (k == "cam_height_m") {
    inst->cam_height_m = value;
  } else if (k == "cam_pitch_deg") {
    inst->cam_pitch_deg = value;
  } else if (k == "focal_px") {
    inst->focal_px = value;
  } else if (k == "cam_cx") {
    inst->cam_cx = value;
  } else if (k == "cam_cy") {
    inst->cam_cy = value;
  } else if (k == "front_overhang_m") {
    inst->front_overhang_m = value;
  } else if (k == "obstacle_merge_radius_m") {
    inst->obstacle_merge_radius_m = value;
  } else if (k == "obstacle_memory_ttl_sec") {
    inst->obstacle_memory_ttl_sec = value;
  } else if (k == "car_length_m") {
    inst->car_length_m = value;
  } else if (k == "rear_axle_to_bumper_m") {
    inst->rear_axle_to_bumper_m = value;
  } else if (k == "obstacle_rear_keep_margin_m") {
    inst->obstacle_rear_keep_margin_m = value;
  } else if (k == "car_width_m") {
    inst->car_width_m = value;
  } else if (k == "default_lane_width_m") {
    inst->default_lane_width_m = value;
  } else if (k == "border_margin_m") {
    inst->border_margin_m = value;
  } else if (k == "cone_margin_m") {
    inst->cone_margin_m = value;
  } else if (k == "no_entry_margin_m") {
    inst->no_entry_margin_m = value;
  } else if (k == "obstacle_lookahead_m") {
    inst->obstacle_lookahead_m = value;
  } else if (k == "obstacle_steer_horizon_m") {
    inst->obstacle_steer_horizon_m = value;
  } else if (k == "obstacle_curve_min_horizon_m") {
    inst->obstacle_curve_min_horizon_m = value;
  } else if (k == "obstacle_stagger_gap_m") {
    inst->obstacle_stagger_gap_m = value;
  } else if (k == "obstacle_rear_keep_m") {
    inst->obstacle_rear_keep_m = value;
  } else if (k == "obstacle_speed_mps") {
    inst->obstacle_speed_mps = value;
  } else if (k == "no_entry_speed_mps") {
    inst->no_entry_speed_mps = value;
  } else if (k == "blocked_creep_speed_mps") {
    inst->blocked_creep_speed_mps = value;
  } else if (k == "fork_outer_time_penalty_s") {
    inst->fork_outer_time_penalty_s = value;
  } else if (k == "fork_commit_dist_m") {
    inst->fork_commit_dist_m = value;
  } else if (k == "digit_min_conf") {
    inst->digit_min_conf = value;
  } else if (k == "digit_lookahead_max_m") {
    inst->digit_lookahead_max_m = value;
  } else if (k == "max_digit_lateral_shift_m") {
    inst->max_digit_lateral_shift_m = value;
  } else if (k == "digit_approach_speed_mps") {
    inst->digit_approach_speed_mps = value;
  } else if (k == "corridor_kp_y") {
    inst->corridor_kp_y = value;
  } else if (k == "corridor_heading_gain") {
    inst->corridor_heading_gain = value;
  } else if (k == "cbf_wheelbase_m") {
    inst->cbf_wheelbase_m = value;
  } else if (k == "cbf_max_steer_deg") {
    inst->cbf_max_steer_deg = value;
  } else if (k == "cbf_step_dt_sec") {
    inst->cbf_step_dt_sec = value;
  } else if (k == "cbf_gamma") {
    inst->cbf_gamma = value;
  } else if (k == "crc_no_entry_pad_y_px") {
    inst->crc_no_entry_pad_y_px = value;
  } else if (k == "crc_cone_pad_y_px") {
    inst->crc_cone_pad_y_px = value;
  } else if (k == "crc_yellow_pad_y_px") {
    inst->crc_yellow_pad_y_px = value;
  } else if (k == "detection_timeout_sec") {
    inst->detection_timeout_sec = value;
  } else if (k == "odom_scale_factor") {
    inst->odom_scale_factor = value;
  } else if (k == "finish_min_dist_m") {
    inst->finish_min_dist_m = value;
  } else if (k == "finish_force_dist_m") {
    inst->finish_force_dist_m = value;
  } else if (k == "finish_min_cum_yaw_rad") {
    inst->finish_min_cum_yaw_rad = value;
  } else if (k == "yellow_pre_brake_dist_m") {
    inst->yellow_pre_brake_dist_m = value;
  } else if (k == "yellow_creep_speed_mps") {
    inst->yellow_creep_speed_mps = value;
  } else if (k == "yellow_final_ramp_dist_m") {
    inst->yellow_final_ramp_dist_m = value;
  } else if (k == "yellow_final_creep_speed_mps") {
    inst->yellow_final_creep_speed_mps = value;
  } else if (k == "yellow_brake_decel_mps2") {
    inst->yellow_brake_decel_mps2 = value;
  } else if (k == "voltage_decel_slope") {
    inst->voltage_decel_slope = value;
  } else if (k == "yellow_control_latency_s") {
    inst->yellow_control_latency_s = value;
  } else if (k == "yellow_bumper_margin_m") {
    inst->yellow_bumper_margin_m = value;
  } else if (k == "yellow_rearm_dist_m") {
    inst->yellow_rearm_dist_m = value;
  } else if (k == "brake_damping_gain") {
    inst->brake_damping_gain = value;
  } else if (k == "brake_zero_cutoff_mps") {
    inst->brake_zero_cutoff_mps = value;
  } else if (k == "clear_straight_speed") {
    inst->clear_straight_speed = value;
  } else if (k == "clear_preview_dist_m") {
    inst->clear_preview_dist_m = value;
  } else if (k == "max_accel_mps2") {
    inst->max_accel_mps2 = value;
  } else if (k == "max_decel_mps2") {
    inst->max_decel_mps2 = value;
  } else if (k == "active_decel_kp") {
    inst->active_decel_kp = value;
  } else {
    return 0;
  }

  inst->params_dirty = true;
  return 1;
}

BRAIN_API int32_t brain_set_param_int(void * handle, const char * name, int32_t value)
{
  auto * inst = static_cast<BrainInstance *>(handle);
  if (inst == nullptr || name == nullptr) {
    return -1;
  }
  const std::string k(name);

  if (k == "yellow_clear_frames") {
    inst->yellow_clear_frames_required = value;
  } else if (k == "fork_outer_branch") {
    inst->fork_outer_branch = value;
  } else if (k == "fork_min_votes") {
    inst->fork_min_votes = value;
  } else if (k == "digit_min_frames") {
    inst->digit_min_frames = value;
  } else if (k == "cbf_horizon_steps") {
    inst->cbf_horizon_steps = value;
  } else if (k == "reverse_curve_confirm_frames") {
    inst->reverse_curve_confirm_frames = std::max(1, value);
  } else if (k == "reverse_curve_clear_frames") {
    inst->reverse_curve_clear_frames = std::max(1, value);
  } else if (k == "image_width") {
    inst->image_width = std::max(1, value);
  } else if (k == "image_height") {
    inst->image_height = std::max(1, value);
  } else {
    // Allow integer YAML values for double parameters (e.g. focal_px: 325)
    return brain_set_param_double(handle, name, static_cast<double>(value));
  }

  inst->params_dirty = true;
  return 1;
}

BRAIN_API int32_t brain_set_param_bool(void * handle, const char * name, int32_t value)
{
  auto * inst = static_cast<BrainInstance *>(handle);
  if (inst == nullptr || name == nullptr) {
    return -1;
  }
  const bool b = (value != 0);
  const std::string k(name);

  if (k == "enable_obstacle_memory") {
    inst->enable_obstacle_memory = b;
  } else if (k == "enable_cv_red_fallback") {
    inst->enable_cv_red_fallback = b;
  } else if (k == "allow_dual_digit_cover") {
    inst->allow_dual_digit_cover = b;
  } else if (k == "enable_cbf_shield") {
    inst->enable_cbf_shield = b;
  } else if (k == "enable_risk_aware_decision") {
    inst->enable_risk_aware_decision = b;
  } else if (k == "enable_brake_adaptation") {
    inst->enable_brake_adaptation = b;
  } else if (k == "yellow_require_odom_still") {
    inst->yellow_require_odom_still = b;
  } else if (k == "use_homography") {
    inst->use_homography = b;
  } else if (k == "lost_line_safe_mode") {
    inst->lost_line_safe_mode = b;
  } else if (k == "enable_continuous_curve_assist") {
    inst->enable_continuous_curve_assist = b;
  } else if (k == "enable_width_adaptive_fusion") {
    inst->enable_width_adaptive_fusion = b;
  } else if (k == "enable_sqrt_brake_envelope") {
    inst->enable_sqrt_brake_envelope = b;
  } else if (k == "enable_closed_loop_brake") {
    inst->enable_closed_loop_brake = b;
  } else if (k == "enable_clear_path_speed") {
    inst->enable_clear_path_speed = b;
  } else {
    return 0;
  }

  inst->params_dirty = true;
  return 1;
}

BRAIN_API int32_t brain_set_param_string(void * handle, const char * name, const char * value)
{
  auto * inst = static_cast<BrainInstance *>(handle);
  if (inst == nullptr || name == nullptr || value == nullptr) {
    return -1;
  }
  const std::string k(name);
  const std::string v(value);

  if (k == "track_mode") {
    inst->track_mode = v;
  } else if (k == "lane_seg_mode") {
    inst->lane_seg_mode = v;
  } else if (k == "yellow_detect_mode") {
    inst->yellow_detect_mode = v;
  } else {
    return 0;
  }

  inst->params_dirty = true;
  return 1;
}

BRAIN_API int32_t brain_set_homography(void * handle, const double * h9, int32_t enable)
{
  auto * inst = static_cast<BrainInstance *>(handle);
  if (inst == nullptr || h9 == nullptr) {
    return -1;
  }
  inst->homography_matrix.assign(h9, h9 + 9);
  inst->use_homography = (enable != 0);
  inst->params_dirty = true;
  inst->sync_configs();
  return inst->ground_projector.using_homography() ? 1 : 0;
}

BRAIN_API int32_t brain_step(
  void * handle,
  const BrainStepInput * in,
  BrainStepOutput * out)
{
  auto * inst = static_cast<BrainInstance *>(handle);
  if (inst == nullptr || in == nullptr || out == nullptr) {
    return -1;
  }
  inst->step(*in, *out);
  return 0;
}

BRAIN_API int32_t brain_get_preferred_branch(void * handle)
{
  auto * inst = static_cast<BrainInstance *>(handle);
  if (inst == nullptr) {
    return 0;
  }
  if (inst->params_dirty) {
    inst->sync_configs();
  }
  return inst->corridor_planner.committed_branch(inst->finish_guard.total_arc_m());
}

BRAIN_API int32_t brain_pixel_to_ground(
  void * handle,
  double u,
  double v,
  double * d_bumper_m_out,
  double * y_lat_m_out)
{
  auto * inst = static_cast<BrainInstance *>(handle);
  if (inst == nullptr || d_bumper_m_out == nullptr || y_lat_m_out == nullptr) {
    return 0;
  }
  if (inst->params_dirty) {
    inst->sync_configs();
  }
  double d = 0.0;
  double y = 0.0;
  if (!inst->ground_projector.pixel_to_ground(u, v, d, y)) {
    return 0;
  }
  *d_bumper_m_out = d;
  *y_lat_m_out = y;
  return 1;
}

}  // extern "C"
