#include "line_follower/ackermann_cbf_shield.hpp"
#include "line_follower/bev_dwa_planner.hpp"
#include "line_follower/corridor_planner.hpp"
#include "line_follower/finals_types.hpp"
#include "line_follower/finish_guard.hpp"
#include "line_follower/ground_projector.hpp"
#include "line_follower/motion_controller.hpp"
#include "line_follower/obstacle_memory.hpp"
#include "line_follower/track_state_machine.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition)                                                                 \
  do {                                                                                   \
    if (!(condition)) {                                                                  \
      throw std::runtime_error(                                                          \
        std::string(__func__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
    }                                                                                    \
  } while (false)

namespace
{

using line_follower::AckermannCbfConfig;
using line_follower::AckermannCbfShield;
using line_follower::AdaptiveSpeedResult;
using line_follower::CbfShieldResult;
using line_follower::ClearPathInput;
using line_follower::CorridorPlan;
using line_follower::CorridorPlanner;
using line_follower::CorridorPlannerConfig;
using line_follower::CurveAssistState;
using line_follower::FinishGuard;
using line_follower::FinishGuardConfig;
using line_follower::GroundDetection;
using line_follower::GroundProjector;
using line_follower::GroundProjectorConfig;
using line_follower::LocalObstacle;
using line_follower::MotionCommand;
using line_follower::MotionController;
using line_follower::MotionControllerConfig;
using line_follower::MotionState;
using line_follower::ObstacleMemory;
using line_follower::ObstacleMemoryConfig;
using line_follower::RedDetectionSource;
using line_follower::SemanticObjectType;
using line_follower::TrackState;
using line_follower::TrackStateConfig;
using line_follower::fuse_red_obstacles_or;
using line_follower::TrackStateMachine;
using line_follower::VehiclePose2D;
using line_follower::YellowStopAdvice;

void test_ground_projector_pinhole_and_homography()
{
  GroundProjectorConfig cfg{};
  cfg.use_homography = false;
  cfg.cam_height_m = 0.11;
  cfg.cam_pitch_deg = 12.5;
  cfg.focal_px = 325.0;
  cfg.cx = 320.0;
  cfg.cy = 240.0;
  cfg.front_overhang_m = 0.06;
  GroundProjector proj(cfg);

  CHECK(!proj.using_homography());
  const double horizon = proj.pinhole_horizon_v();
  CHECK(horizon > 150.0 && horizon < 185.0);

  // Above horizon fails
  double d_bad = 0.0, y_bad = 0.0;
  CHECK(!proj.pixel_to_ground(320.0, horizon - 5.0, d_bad, y_bad));

  // Center column near bottom projects forward with y=0
  double d_near = 0.0, y_near = 0.0;
  double d_far = 0.0, y_far = 0.0;
  CHECK(proj.pixel_to_ground(320.0, 440.0, d_near, y_near));
  CHECK(proj.pixel_to_ground(320.0, 280.0, d_far, y_far));
  CHECK(std::abs(y_near) < 1e-6);
  CHECK(d_far > d_near && d_near > 0.0);

  // ROS REP-103 convention: u < cx (left in image) -> +y_lat_m (LEFT)
  double d_l = 0.0, y_l = 0.0, d_r = 0.0, y_r = 0.0;
  CHECK(proj.pixel_to_ground(220.0, 360.0, d_l, y_l));
  CHECK(proj.pixel_to_ground(420.0, 360.0, d_r, y_r));
  CHECK(y_l > 0.05 && y_r < -0.05 && std::abs(y_l + y_r) < 1e-6);

  // Exact round-trip pixel -> ground -> pixel
  double u_rt = 0.0, v_rt = 0.0;
  CHECK(proj.ground_to_pixel(d_l, y_l, u_rt, v_rt));
  CHECK(std::abs(u_rt - 220.0) < 1e-4 && std::abs(v_rt - 360.0) < 1e-4);

  // Lane width in pixels increases monotonically toward bottom of image
  const double w_far = proj.lane_width_px_at_row(280.0, 0.43);
  const double w_near = proj.lane_width_px_at_row(420.0, 0.43);
  CHECK(w_far > 20.0 && w_near > w_far);

  // Homography mode: simple scaling/translation matrix + singular fallback
  const std::vector<double> h_valid{
    0.0, -0.002, 1.0,
    -0.001, 0.0, 0.32,
    0.0, 0.0, 1.0};
  CHECK(proj.set_homography_from_vector(h_valid, true));
  CHECK(proj.using_homography() && proj.homography_valid());
  double dh = 0.0, yh = 0.0;
  CHECK(proj.pixel_to_ground(220.0, 300.0, dh, yh));
  CHECK(std::abs(dh - 0.40) < 1e-6 && std::abs(yh - 0.10) < 1e-6);
  CHECK(proj.ground_to_pixel(dh, yh, u_rt, v_rt));
  CHECK(std::abs(u_rt - 220.0) < 1e-5 && std::abs(v_rt - 300.0) < 1e-5);

  // Atomic rejection of singular homography preserves prior valid state
  const std::vector<double> h_singular{1, 0, 0, 2, 0, 0, 0, 0, 0};
  CHECK(!proj.set_homography_from_vector(h_singular, true));
  CHECK(proj.using_homography());

  // Fresh projector with singular homography in config falls back cleanly to pinhole
  GroundProjectorConfig bad_cfg = cfg;
  bad_cfg.use_homography = true;
  bad_cfg.homography_matrix = {1, 0, 0, 2, 0, 0, 0, 0, 0};
  GroundProjector fallback_proj(bad_cfg);
  CHECK(!fallback_proj.using_homography());
  CHECK(fallback_proj.pixel_to_ground(320.0, 440.0, d_near, y_near));
  CHECK(d_near > 0.0);
}

void test_obstacle_memory_rear_wheel_retention_and_merge()
{
  ObstacleMemoryConfig cfg{};
  cfg.merge_radius_m = 0.16;
  cfg.memory_ttl_sec = 1.8;
  cfg.car_length_m = 0.28;
  cfg.rear_keep_margin_m = 0.15;  // keep until rear edge < -0.43m
  ObstacleMemory mem(cfg);

  // Car at world (0, 0, yaw=0), sees a red no-entry 0.35m ahead on the right (y=-0.14m)
  VehiclePose2D pose0{true, 0.0, 0.0, 0.0, 0.3, 0.0, 10.0};
  GroundDetection det1{};
  det1.type = SemanticObjectType::RED_NO_ENTRY;
  det1.confidence = 0.90;
  det1.d_bumper_m = 0.35;
  det1.y_lat_m = -0.14;
  det1.half_len_m = 0.12;
  det1.half_wid_m = 0.065;

  mem.update({det1}, pose0, 10.0);
  CHECK(mem.size() == 1);

  // Second observation slightly shifted merges into same tracked obstacle
  GroundDetection det2 = det1;
  det2.d_bumper_m = 0.36;
  det2.y_lat_m = -0.13;
  mem.update({det2}, pose0, 10.05);
  CHECK(mem.size() == 1);

  // Car moves forward 0.55m -> obstacle is now in camera blind zone beside chassis/rear wheels!
  // d_bumper ~ 0.355 - 0.55 = -0.195m (in [-0.30m, 0.0m]) -> MUST still be retained!
  VehiclePose2D pose1{true, 0.55, 0.0, 0.0, 0.3, 0.0, 10.8};
  mem.update({}, pose1, 10.8);
  const auto local1 = mem.get_local_obstacles(pose1, 10.8);
  CHECK(local1.size() == 1);
  CHECK(local1[0].type == SemanticObjectType::RED_NO_ENTRY);
  CHECK(local1[0].d_bumper_m < 0.0 && local1[0].d_bumper_m > -0.25);
  CHECK(local1[0].y_lat_m < -0.10);

  // Car moves far past obstacle (x = 1.2m -> d_max = 0.355 + 0.12 - 1.2 = -0.725 < -0.43) -> pruned!
  VehiclePose2D pose2{true, 1.20, 0.0, 0.0, 0.3, 0.0, 11.2};
  mem.update({}, pose2, 11.2);
  CHECK(mem.size() == 0);

  // TTL expiration check
  mem.update({det1}, pose0, 20.0);
  CHECK(mem.size() == 1);
  mem.update({}, pose0, 22.0);  // dt = 2.0s > 1.8s
  CHECK(mem.size() == 0);
}

void test_corridor_planner_tight_obstacles_and_rear_cutin()
{
  CorridorPlannerConfig cfg{};
  cfg.car_width_m = 0.191;
  cfg.car_length_m = 0.280;
  cfg.default_lane_width_m = 0.430;
  cfg.border_margin_m = 0.025;
  cfg.cone_margin_m = 0.025;
  cfg.no_entry_margin_m = 0.030;
  cfg.obstacle_lookahead_m = 1.10;
  cfg.obstacle_rear_keep_m = 0.28;
  cfg.cruise_speed_mps = 0.50;
  cfg.obstacle_speed_mps = 0.26;
  cfg.no_entry_speed_mps = 0.22;
  CorridorPlanner planner(cfg);

  // 1. Clean straight lane (±0.215m), car drifted slightly right (lane_mid = +0.04m in bumper frame)
  const CorridorPlan cruise = planner.plan(+0.255, -0.175, 0.0, {}, {}, false, 1.0, 1.0);
  CHECK(!cruise.has_blocking_obstacle);
  CHECK(std::abs(cruise.target_y_m - 0.04) < 1e-4);
  CHECK(std::abs(cruise.speed_cap_mps - 0.50) < 1e-6);

  // 2. Red cone (13.5cm base) on right side of 43cm lane: y in [-0.215, -0.080]
  LocalObstacle cone{};
  cone.type = SemanticObjectType::RED_CONE;
  cone.d_bumper_m = 0.55;
  cone.d_min_m = 0.48;
  cone.d_max_m = 0.62;
  cone.y_lat_m = -0.1475;
  cone.y_min_m = -0.215;
  cone.y_max_m = -0.080;

  const CorridorPlan avoid_cone = planner.plan(+0.215, -0.215, 0.0, {cone}, {}, false, 2.0, 2.0);
  CHECK(!avoid_cone.has_blocking_obstacle);
  CHECK(avoid_cone.active_mode_tag == "avoid_cone");
  CHECK(avoid_cone.speed_cap_mps <= cfg.obstacle_speed_mps + 1e-6);
  // Car center must stay strictly clear of cone right edge AND left black border
  CHECK(avoid_cone.target_y_m - 0.5 * cfg.car_width_m > cone.y_max_m);
  CHECK(avoid_cone.target_y_m + 0.5 * cfg.car_width_m < +0.215);

  // 3. Red no-entry (24x13cm) beside rear wheels (d_bumper = -0.18m, d_max = -0.06m) on LEFT side!
  LocalObstacle no_entry_rear{};
  no_entry_rear.type = SemanticObjectType::RED_NO_ENTRY;
  no_entry_rear.d_bumper_m = -0.18;
  no_entry_rear.d_min_m = -0.30;
  no_entry_rear.d_max_m = -0.06;
  no_entry_rear.y_lat_m = +0.150;
  no_entry_rear.y_min_m = +0.085;
  no_entry_rear.y_max_m = +0.215;

  const CorridorPlan avoid_rear =
    planner.plan(+0.215, -0.215, 0.0, {no_entry_rear}, {}, false, 2.5, 2.5);
  CHECK(!avoid_rear.has_blocking_obstacle);
  CHECK(avoid_rear.active_mode_tag == "avoid_no_entry");
  CHECK(avoid_rear.speed_cap_mps <= cfg.no_entry_speed_mps + 1e-6);
  // Must keep car shifted right so rear wheel doesn't clip +30s no-entry corner!
  CHECK(avoid_rear.target_y_m + 0.5 * cfg.car_width_m < no_entry_rear.y_min_m);
  CHECK(avoid_rear.target_y_m - 0.5 * cfg.car_width_m > -0.215);
}

void test_corridor_planner_fork_voting_and_straight_digits()
{
  CorridorPlannerConfig cfg{};
  cfg.track_mode = "fork";
  cfg.fork_outer_branch = -1;       // left is outer (+3s time penalty)
  cfg.fork_outer_time_penalty_s = 3.0;
  cfg.bonus_score_weight_s = 5.0;
  cfg.fork_min_votes = 3;
  cfg.fork_commit_dist_m = 1.10;
  cfg.digit_min_conf = 0.55;
  cfg.digit_min_frames = 2;
  cfg.allow_dual_digit_cover = false;
  CorridorPlanner planner(cfg);

  // Scenario A: Fork outer digit=6 vs inner digit=5 -> J_L = 3 - 30 = -27 < J_R = 0 - 25 = -25 -> LEFT (-1)
  GroundDetection dl{SemanticObjectType::BONUS_DIGIT, 6, 0.88, 0.80, +0.10, 0.05, 0.05, 1, 1.0};
  GroundDetection dr{SemanticObjectType::BONUS_DIGIT, 5, 0.90, 0.80, -0.10, 0.05, 0.05, 2, 1.0};
  for (int i = 0; i < 4; ++i) {
    planner.update_fork_static_votes({dl, dr}, 1.0 + i * 0.05);
  }
  CHECK(planner.finalize_fork_decision(3.0) == -1);
  CHECK(planner.committed_branch(3.5) == -1);
  CHECK(planner.committed_branch(4.2) == 0);  // expired past 1.10m commit dist

  // Scenario B: Fork outer digit=5 vs inner digit=5 -> J_L = -22 > J_R = -25 -> RIGHT (+1, inner)
  planner.reset_fork_votes();
  dl.digit = 5;
  dr.digit = 5;
  for (int i = 0; i < 4; ++i) {
    planner.update_fork_static_votes({dl, dr}, 2.0 + i * 0.05);
  }
  CHECK(planner.finalize_fork_decision(5.0) == +1);

  // Scenario C: Outer has digit=8 (-40s) BUT ALSO RED_NO_ENTRY (+30s hard ban) -> MUST pick RIGHT (+1)!
  planner.reset_fork_votes();
  dl.digit = 8;
  dr.digit = 1;
  GroundDetection no_entry_l{SemanticObjectType::RED_NO_ENTRY, 0, 0.85, 1.10, +0.12, 0.12, 0.065, 3, 3.0};
  for (int i = 0; i < 4; ++i) {
    planner.update_fork_static_votes({dl, dr, no_entry_l}, 3.0 + i * 0.05);
  }
  CHECK(planner.finalize_fork_decision(6.5) == +1);
  planner.clear_fork_commit();

  // Scenario D: Straight dual handwritten digits (10cm apart: digit 2 @ +0.05, digit 7 @ -0.05)
  // Must cleanly target LARGER digit 7 and stay inside safe corridor
  GroundDetection sd2{SemanticObjectType::BONUS_DIGIT, 2, 0.85, 0.65, +0.05, 0.04, 0.04, 10, 10.0};
  GroundDetection sd7{SemanticObjectType::BONUS_DIGIT, 7, 0.82, 0.65, -0.05, 0.04, 0.04, 11, 10.0};
  planner.plan(+0.215, -0.215, 0.0, {}, {sd2, sd7}, false, 2.0, 10.0);
  const CorridorPlan dplan = planner.plan(+0.215, -0.215, 0.0, {}, {sd2, sd7}, false, 2.0, 10.05);
  CHECK(dplan.digit_target_active && dplan.target_digit == 7);
  CHECK(dplan.target_y_m <= -0.05);  // biased cleanly onto/away from smaller sibling
  CHECK(dplan.target_y_m >= dplan.safe_y_min_m && dplan.target_y_m <= dplan.safe_y_max_m);
  CHECK(dplan.speed_cap_mps <= cfg.digit_approach_speed_mps + 1e-6);

  // Scenario E: Obstacle blocking digit 7 -> must abort digit 7 and safely fall back to digit 2!
  LocalObstacle cone_on_7{};
  cone_on_7.type = SemanticObjectType::RED_CONE;
  cone_on_7.d_bumper_m = 0.70;
  cone_on_7.d_min_m = 0.63;
  cone_on_7.d_max_m = 0.77;
  cone_on_7.y_lat_m = -0.145;
  cone_on_7.y_min_m = -0.212;
  cone_on_7.y_max_m = -0.078;

  const CorridorPlan dplan_obs =
    planner.plan(+0.215, -0.215, 0.0, {cone_on_7}, {sd2, sd7}, false, 2.1, 10.10);
  CHECK(!dplan_obs.has_blocking_obstacle);
  CHECK(dplan_obs.target_y_m > 0.03);  // strictly clear of cone on right
  CHECK(dplan_obs.target_digit != 7);  // never steers into cone for digit 7!
}

void test_finish_guard_random_yellow_rejection_and_two_stage_stop()
{
  FinishGuardConfig cfg{};
  cfg.odom_scale_factor = 1.05;
  cfg.finish_min_dist_m = 8.5;
  cfg.finish_force_dist_m = 11.5;
  cfg.pre_brake_dist_m = 0.70;
  cfg.creep_speed_mps = 0.24;
  cfg.brake_decel_mps2 = 1.8;
  cfg.control_latency_s = 0.10;
  cfg.bumper_margin_m = 0.08;
  cfg.rearm_travel_dist_m = 0.35;
  FinishGuard guard(cfg);

  // At s = 4.2m (4.0 * 1.05): random straight yellow line MUST NOT trigger finish even if dead_end=true
  guard.update_motion(4.0, 0.5);
  CHECK(!guard.is_finish_armed());
  CHECK(!guard.evaluate_is_finish(true));

  // Two-stage yellow stop law at v = 0.24 m/s:
  // stop_trigger_dist = 0.24^2/(2*1.8) + 0.24*0.10 + 0.08 = 0.016 + 0.024 + 0.08 = 0.120m
  const YellowStopAdvice far_adv = guard.evaluate_yellow_stop(true, true, 0.90, 0.50, false);
  CHECK(!far_adv.pre_brake && !far_adv.trigger_stop);

  const YellowStopAdvice pre_adv = guard.evaluate_yellow_stop(true, true, 0.55, 0.50, false);
  CHECK(pre_adv.pre_brake && !pre_adv.trigger_stop);
  CHECK(std::abs(pre_adv.speed_cap_mps - 0.24) < 1e-6);

  const YellowStopAdvice stop_adv = guard.evaluate_yellow_stop(true, true, 0.11, 0.24, false);
  CHECK(stop_adv.pre_brake && stop_adv.trigger_stop && !stop_adv.is_finish_line);
  CHECK(std::abs(stop_adv.speed_cap_mps) < 1e-9);

  // Rearm travel guard prevents double-triggering AND prevents 0.0 m/s resume deadlock
  // while still sitting right in front of the yellow line just stopped for!
  guard.mark_yellow_stop_triggered();
  CHECK(!guard.is_yellow_rearmed());
  const YellowStopAdvice resume_adv = guard.evaluate_yellow_stop(true, true, 0.08, 0.0, false);
  CHECK(!resume_adv.pre_brake && !resume_adv.trigger_stop && resume_adv.speed_cap_mps > 1.0);
  guard.update_motion(0.20, 0.0);  // 0.21m < 0.35m
  CHECK(!guard.is_yellow_rearmed());
  const YellowStopAdvice crossing_adv = guard.evaluate_yellow_stop(true, true, 0.05, 0.30, false);
  CHECK(!crossing_adv.pre_brake && !crossing_adv.trigger_stop && crossing_adv.speed_cap_mps > 1.0);
  guard.update_motion(0.20, 0.0);  // 0.42m >= 0.35m
  CHECK(guard.is_yellow_rearmed());
  const YellowStopAdvice next_adv = guard.evaluate_yellow_stop(true, true, 0.55, 0.45, false);
  CHECK(next_adv.pre_brake && !next_adv.trigger_stop);

  // Drive past finish_min_dist_m (4.2 + 0.42 + 4.5*1.05 = 9.345m >= 8.5m)
  guard.update_motion(4.5, 2.0);
  CHECK(guard.is_finish_armed());
  CHECK(!guard.evaluate_is_finish(false));  // still < 11.5m without dead-end cue
  CHECK(guard.evaluate_is_finish(true));    // dead-end cue confirms real END line!
}

void test_track_state_machine_odom_freshness_and_finals_mode()
{
  TrackStateConfig cfg{};
  cfg.yellow_stop_duration = 1.0;
  cfg.yellow_stop_min_time = 0.30;
  cfg.odom_still_threshold = 0.02;
  cfg.odom_still_hold = 0.25;
  cfg.odom_max_age_sec = 0.20;
  cfg.yellow_require_odom_still = true;
  cfg.yellow_max_hold_sec = 3.0;
  cfg.yellow_clear_frames_required = 2;

  TrackStateMachine sm(cfg);
  sm.reset(10.0);

  // Trigger yellow stop at t=11.0 (NOT finish)
  auto ev = sm.step(true, true, 11.0, false);
  CHECK(ev.yellow_stop_begin && ev.yellow_temp_stop && ev.yellow_hold_stop);

  // Car is STILL MOVING (v=0.12) at t=12.2 (> legacy 1.0s duration): MUST NOT release in finals mode!
  sm.update_odom(0.12, 12.15);
  ev = sm.step(true, true, 12.20, false);
  CHECK(ev.odom_fresh && !ev.odom_still_ok && !ev.can_release && ev.yellow_hold_stop);

  // Car stops at t=12.30, then odom goes STALE at t=12.70 (age 0.40 > 0.20): MUST NOT count as still!
  sm.update_odom(0.005, 12.30);
  ev = sm.step(true, true, 12.70, false);
  CHECK(!ev.odom_fresh && !ev.odom_still_ok && !ev.can_release && ev.yellow_hold_stop);

  // Fresh odom confirms stillness at t=12.75 (held since 12.30 -> 0.45s >= 0.25s) -> releases!
  sm.update_odom(0.005, 12.72);
  ev = sm.step(false, true, 12.75, false);
  CHECK(ev.odom_fresh && ev.odom_still_ok && ev.can_release && ev.yellow_stop_end);

  // Clear latch and trigger 5 more random intermediate yellow lines (count=6) -> never permanent_stop!
  double t = 13.0;
  for (int k = 0; k < 5; ++k) {
    sm.step(false, true, t);
    sm.step(false, true, t + 0.05);
    ev = sm.step(true, true, t + 0.10, false);
    CHECK(!ev.yellow_final_stop && !sm.state().permanent_stop);
    sm.update_odom(0.001, t + 0.15);
    sm.update_odom(0.001, t + 0.50);
    ev = sm.step(false, true, t + 0.55, false);
    CHECK(ev.yellow_stop_end);
    t += 1.0;
  }

  // Final yellow line with external_is_finish=true -> permanent_stop!
  sm.step(false, true, t);
  sm.step(false, true, t + 0.05);
  ev = sm.step(true, true, t + 0.10, true);
  CHECK(ev.yellow_final_stop && sm.state().permanent_stop && ev.yellow_hold_stop);
}

void test_motion_controller_corridor_and_safe_lost()
{
  MotionControllerConfig cfg{};
  cfg.target_speed = 0.55;
  cfg.max_speed = 0.70;
  cfg.min_speed = 0.0;
  cfg.max_yaw_rate = 1.5;
  cfg.corridor_kp_y = 4.0;
  cfg.corridor_heading_gain = 0.8;
  cfg.line_lost_forward_speed = 0.30;
  cfg.line_lost_full_stop_time = 1.20;
  cfg.lost_line_safe_mode = true;
  cfg.lost_line_safe_speed = 0.12;
  cfg.lost_line_safe_stop_time = 0.35;

  MotionController ctrl(cfg);
  CurveAssistState curve{};
  MotionState motion{};

  // +y target (left) -> +yaw_rate (CCW / left), speed capped cleanly by speed_cap_mps
  ctrl.set_config(cfg);
  const MotionCommand cmd_l = ctrl.compute_corridor_command(
    +0.08, 0.0, 0.0, 0.30, 1.0, 0.25, curve, motion);
  CHECK(cmd_l.yaw_rate > 0.25);
  CHECK(std::abs(cmd_l.linear_speed - 0.25) < 1e-6);

  // Without event cap, curve speed scales purely with target_speed * (1 - damping * slow_factor)
  cfg.speed_gain_damping = 0.40;
  ctrl.set_config(cfg);
  const MotionCommand cmd_curve = ctrl.compute_corridor_command(
    0.0, 0.0, 0.0, 0.50, 1.0, 1.0, curve, motion);
  CHECK(cmd_curve.linear_speed > 0.25 && cmd_curve.linear_speed < cfg.target_speed);

  // Safe lost-line mode clamps blind speed to <= 0.12 and stops at > 0.35s
  MotionController::record_motion(false, cmd_l.linear_speed, cmd_l.yaw_rate, motion);
  const MotionCommand lost_early = ctrl.compute_lost_line_command(0.10, motion);
  CHECK(lost_early.linear_speed <= 0.12 + 1e-6);
  const MotionCommand lost_stop = ctrl.compute_lost_line_command(0.40, motion);
  CHECK(std::abs(lost_stop.linear_speed) < 1e-9);
}

void test_motion_controller_edge_heading_guard_and_curve_preservation()
{
  MotionControllerConfig cfg{};
  cfg.target_speed = 0.42;
  cfg.max_speed = 0.80;
  cfg.min_speed = 0.0;
  cfg.max_yaw_rate = 2.0;
  cfg.speed_gain_damping = 0.55;
  cfg.corridor_kp_y = 3.8;
  cfg.corridor_heading_gain = 0.85;
  MotionController ctrl(cfg);
  MotionState motion{};

  // 1. Exact roundabout crash frame #141: car hugging right border (near=-0.90, far=-0.18 -> hdg=+0.72),
  //    corridor wants LEFT recovery (y_ref=+0.082). Guard MUST suppress edge-pushing hdg and steer LEFT!
  {
    CurveAssistState c{};
    c.near_valid = true;
    c.far_valid = true;
    c.near_error = -0.90;
    c.far_error = -0.18;
    c.reverse_curve_active = false;
    const MotionCommand cmd = ctrl.compute_corridor_command(
      +0.082, 0.0, +0.72, 0.72, 0.90, 0.22, c, motion);
    CHECK(cmd.yaw_rate > 0.25);  // Previously -0.32 (drove off right edge!)
  }

  // 2. Exact crash onset frame #136: car near right edge (near=-0.53) while far just crosses right (+0.04),
  //    corridor wants gentle LEFT recovery (y_ref=+0.019). Guard MUST NOT flip recovery into diving right!
  {
    CurveAssistState c{};
    c.near_valid = true;
    c.far_valid = true;
    c.near_error = -0.53;
    c.far_error = +0.04;
    c.reverse_curve_active = false;
    const MotionCommand cmd = ctrl.compute_corridor_command(
      +0.019, 0.0, +0.57, 0.57, 0.92, 0.22, c, motion);
    CHECK(cmd.yaw_rate > 0.0);  // Previously -0.38!
  }

  // 3. Mirror case: car hugging LEFT border (near=+0.80, far=+0.15, hdg=-0.65), corridor wants RIGHT (-0.08)
  {
    CurveAssistState c{};
    c.near_valid = true;
    c.far_valid = true;
    c.near_error = +0.80;
    c.far_error = +0.15;
    c.reverse_curve_active = false;
    const MotionCommand cmd = ctrl.compute_corridor_command(
      -0.080, 0.0, -0.65, 0.65, 0.90, 0.22, c, motion);
    CHECK(cmd.yaw_rate < -0.25);
  }

  // 4. Anti-regression: normal centered sharp right curve (#69: near=0.0, far=+0.60, hdg=+0.60, y_ref=-0.152)
  //    MUST keep full aggressive right steering (~ -1.09 rad/s)!
  {
    CurveAssistState c{};
    c.near_valid = true;
    c.far_valid = true;
    c.near_error = 0.0;
    c.far_error = +0.60;
    c.reverse_curve_active = false;
    const MotionCommand cmd = ctrl.compute_corridor_command(
      -0.152, 0.0, +0.60, 0.60, 0.63, 0.42, c, motion);
    CHECK(cmd.yaw_rate < -1.05);
  }

  // 5. Anti-regression: active right dodge around left obstacle (#125: near=-0.28, far=-0.23, y_ref=-0.097)
  //    corridor actively asks to go RIGHT -> guard MUST NOT fight obstacle dodge!
  {
    CurveAssistState c{};
    c.near_valid = true;
    c.far_valid = true;
    c.near_error = -0.28;
    c.far_error = -0.23;
    c.reverse_curve_active = false;
    const MotionCommand cmd = ctrl.compute_corridor_command(
      -0.097, 0.0, +0.05, 0.05, 0.97, 0.22, c, motion);
    CHECK(cmd.yaw_rate < -0.40);
  }

  // 6. Fork Y-merge wide junction (#153: w=0.89, near=-0.26, far=0.0, decaying right hdg=+0.35, y_ref=+0.056)
  //    MUST preserve right-turn exit continuity (yaw < 0) instead of reversing left into outer branch!
  {
    CurveAssistState c{};
    c.near_valid = true;
    c.far_valid = true;
    c.near_error = -0.26;
    c.far_error = 0.0;
    c.last_lane_width_ratio = 0.89;
    c.reverse_curve_active = false;
    const MotionCommand cmd = ctrl.compute_corridor_command(
      +0.056, 0.0, +0.35, 0.35, 0.66, 0.44, c, motion);
    CHECK(cmd.yaw_rate < -0.05);
  }
}

void test_fork_signed_odom_and_dual_digit_no_late_swerve()
{
  // 1. TrackStateMachine preserves signed odom_linear_x for reverse-brake zero-crossing guard
  {
    TrackStateConfig tcfg{};
    tcfg.odom_still_threshold = 0.02;
    TrackStateMachine sm(tcfg);
    sm.update_odom(-0.15, 1.0);
    CHECK(sm.state().odom_linear_x < -0.14);
    CHECK(std::abs(sm.state().odom_speed - 0.15) < 1e-9);
    CHECK(!sm.state().odom_still_valid);
    sm.update_odom(+0.005, 1.1);
    CHECK(std::abs(sm.state().odom_linear_x - 0.005) < 1e-9);
    CHECK(sm.state().odom_still_valid);
  }

  // 2. Exact fork #86-#91 dual-digit wobble reproduction (5 on left, 2 on right):
  //    Once car commits to 5 and 5 exits FOV, planner MUST NOT swerve right into smaller sibling 2 (or 3)!
  {
    CorridorPlannerConfig cfg{};
    cfg.digit_min_conf = 0.55;
    cfg.digit_min_frames = 2;
    cfg.max_digit_lateral_shift_m = 0.12;
    CorridorPlanner planner(cfg);

    GroundDetection d5{SemanticObjectType::BONUS_DIGIT, 5, 0.75, 0.22, +0.07, 0.04, 0.04, 1, 10.0};
    GroundDetection d2{SemanticObjectType::BONUS_DIGIT, 2, 0.79, 0.21, -0.12, 0.04, 0.04, 2, 10.0};
    planner.plan(+0.215, -0.215, 0.0, {}, {d5, d2}, false, 1.80, 10.00);
    const CorridorPlan p_pick5 = planner.plan(+0.215, -0.215, 0.0, {}, {d5, d2}, false, 1.86, 10.07);
    CHECK(p_pick5.digit_target_active && p_pick5.target_digit == 5);
    CHECK(p_pick5.target_y_m > +0.05);

    // 5 leaves FOV at t=10.14 while 2 stays visible at d=0.14m:
    //   - At t=10.14 (5 still within 0.25s TTL): keeps holding 5, never jumps to 2
    //   - At t=10.35 (5 TTL expired, 2 still alive from t=10.14): sibling lock MUST reject 2!
    d2.d_bumper_m = 0.14;
    const CorridorPlan p_hold5 = planner.plan(+0.215, -0.215, 0.0, {}, {d2}, false, 1.94, 10.14);
    CHECK(p_hold5.digit_target_active && p_hold5.target_digit == 5);
    CHECK(p_hold5.target_y_m > +0.05);

    const CorridorPlan p_after = planner.plan(+0.215, -0.215, 0.0, {}, {}, false, 2.00, 10.35);
    CHECK(!p_after.digit_target_active);
    CHECK(std::abs(p_after.target_y_m) < 0.01);

    GroundDetection d3_close{SemanticObjectType::BONUS_DIGIT, 3, 0.81, 0.08, -0.10, 0.04, 0.04, 3, 10.36};
    const CorridorPlan p_close = planner.plan(+0.215, -0.215, 0.0, {}, {d3_close}, false, 2.01, 10.36);
    CHECK(!p_close.digit_target_active);
    CHECK(std::abs(p_close.target_y_m) < 0.01);

    // Anti-regression: upgrading UP to a strictly larger digit (e.g. 1 -> 7 at d=0.28m) MUST still work!
    planner.reset();
    GroundDetection d1{SemanticObjectType::BONUS_DIGIT, 1, 0.78, 0.32, -0.05, 0.04, 0.04, 10, 20.0};
    GroundDetection d7{SemanticObjectType::BONUS_DIGIT, 7, 0.88, 0.28, +0.08, 0.04, 0.04, 11, 20.1};
    planner.plan(+0.215, -0.215, 0.0, {}, {d1}, false, 2.00, 20.00);
    const CorridorPlan p_pick1 = planner.plan(+0.215, -0.215, 0.0, {}, {d1}, false, 2.05, 20.07);
    CHECK(p_pick1.digit_target_active && p_pick1.target_digit == 1);
    planner.plan(+0.215, -0.215, 0.0, {}, {d1, d7}, false, 2.10, 20.14);
    const CorridorPlan p_upgrade7 = planner.plan(+0.215, -0.215, 0.0, {}, {d1, d7}, false, 2.15, 20.21);
    CHECK(p_upgrade7.digit_target_active && p_upgrade7.target_digit == 7);
    CHECK(p_upgrade7.target_y_m > +0.05);
  }
}

void test_red_obstacle_or_fusion_and_attribution()
{
  GroundDetection y_cone{};
  y_cone.type = SemanticObjectType::RED_CONE;
  y_cone.confidence = 0.88;
  y_cone.d_bumper_m = 0.70;
  y_cone.y_lat_m = 0.08;

  GroundDetection c_near{};
  c_near.type = SemanticObjectType::RED_NO_ENTRY;  // CV misclassified cone as no_entry
  c_near.confidence = 0.60;
  c_near.d_bumper_m = 0.74;
  c_near.y_lat_m = 0.06;

  GroundDetection c_extra{};
  c_extra.type = SemanticObjectType::RED_NO_ENTRY;
  c_extra.confidence = 0.60;
  c_extra.d_bumper_m = 0.55;
  c_extra.y_lat_m = -0.10;

  // Case 1: YOLO only (ideal when CV fails under glare)
  {
    const auto r = fuse_red_obstacles_or({y_cone}, {});
    CHECK(r.overall_source == RedDetectionSource::YOLO_ONLY);
    CHECK(r.fused_red_detections.size() == 1);
    CHECK(r.yolo_only_count == 1 && r.matched_both_count == 0 && r.cv_only_count == 0);
    CHECK(r.fused_sources[0] == RedDetectionSource::YOLO_ONLY);
  }

  // Case 2: CV only (pre-YOLO fallback or YOLO miss)
  {
    const auto r = fuse_red_obstacles_or({}, {c_extra});
    CHECK(r.overall_source == RedDetectionSource::CV_ONLY);
    CHECK(r.fused_red_detections.size() == 1);
    CHECK(r.cv_only_count == 1 && r.yolo_only_count == 0 && r.matched_both_count == 0);
    CHECK(r.fused_sources[0] == RedDetectionSource::CV_ONLY);
    CHECK(std::abs(r.nearest_d_bumper_m - 0.55) < 1e-6);
  }

  // Case 3: Both see same obstacle + CV sees an extra missed obstacle -> OR union, YOLO wins matched class
  {
    const auto r = fuse_red_obstacles_or({y_cone}, {c_near, c_extra}, 0.24);
    CHECK(r.overall_source == RedDetectionSource::BOTH);
    CHECK(r.fused_red_detections.size() == 2);
    CHECK(r.matched_both_count == 1 && r.cv_only_count == 1 && r.yolo_only_count == 0);
    // Matched obstacle preserves YOLO's RED_CONE type & confidence
    CHECK(r.fused_red_detections[0].type == SemanticObjectType::RED_CONE);
    CHECK(r.fused_sources[0] == RedDetectionSource::BOTH);
    // Unmatched CV obstacle is injected as CV_ONLY and is the nearest one (0.55m)
    CHECK(r.fused_red_detections[1].type == SemanticObjectType::RED_NO_ENTRY);
    CHECK(r.fused_sources[1] == RedDetectionSource::CV_ONLY);
    CHECK(std::abs(r.nearest_d_bumper_m - 0.55) < 1e-6);
  }

  // Case 4: Neither sees red
  {
    const auto r = fuse_red_obstacles_or({}, {});
    CHECK(r.overall_source == RedDetectionSource::NONE);
    CHECK(r.fused_red_detections.empty());
  }
}

void test_corridor_two_zone_staggered_and_curve_horizon()
{
  CorridorPlannerConfig cfg{};
  cfg.car_width_m = 0.191;
  cfg.default_lane_width_m = 0.430;
  cfg.obstacle_lookahead_m = 1.10;
  cfg.obstacle_steer_horizon_m = 0.65;
  cfg.obstacle_curve_min_horizon_m = 0.45;
  cfg.obstacle_stagger_gap_m = 0.25;
  cfg.cruise_speed_mps = 0.45;
  cfg.obstacle_speed_mps = 0.26;
  cfg.no_entry_speed_mps = 0.22;

  CorridorPlanner planner(cfg);

  // Staggered S-slalom: no-entry on LEFT at 0.28..0.40m, cone on RIGHT at 0.85..0.98m
  // Must steer RIGHT around near no-entry first (NOT deadlocked as blocked!)
  LocalObstacle near_left{};
  near_left.type = SemanticObjectType::RED_NO_ENTRY;
  near_left.d_bumper_m = 0.34;
  near_left.d_min_m = 0.28;
  near_left.d_max_m = 0.40;
  near_left.y_lat_m = +0.13;
  near_left.y_min_m = +0.065;
  near_left.y_max_m = +0.195;

  LocalObstacle far_right{};
  far_right.type = SemanticObjectType::RED_CONE;
  far_right.d_bumper_m = 0.90;
  far_right.d_min_m = 0.84;
  far_right.d_max_m = 0.96;
  far_right.y_lat_m = -0.13;
  far_right.y_min_m = -0.195;
  far_right.y_max_m = -0.065;

  const CorridorPlan slalom = planner.plan(
    +0.215, -0.215, 0.0, {near_left, far_right}, {}, false, 1.0, 10.0, 0.0);
  CHECK(!slalom.has_blocking_obstacle);
  CHECK(slalom.target_y_m < -0.04);  // Steers right around near_left
  CHECK(slalom.near_obstacle_count == 1 && slalom.far_obstacle_count == 1);
  CHECK(slalom.speed_cap_mps <= cfg.no_entry_speed_mps + 1e-6);

  // Curve far-view interference: cone at 0.58m on a sharp curve (curve_level=0.8 -> horizon=0.49m)
  // Must slow down (far_obs_slow) WITHOUT steering sideways prematurely!
  LocalObstacle curve_far{};
  curve_far.type = SemanticObjectType::RED_CONE;
  curve_far.d_bumper_m = 0.64;
  curve_far.d_min_m = 0.58;
  curve_far.d_max_m = 0.70;
  curve_far.y_lat_m = +0.12;
  curve_far.y_min_m = +0.055;
  curve_far.y_max_m = +0.185;

  const CorridorPlan curve_plan = planner.plan(
    +0.215, -0.215, 0.0, {curve_far}, {}, false, 2.0, 11.0, 0.8);
  CHECK(!curve_plan.has_blocking_obstacle);
  CHECK(curve_plan.near_obstacle_count == 0 && curve_plan.far_obstacle_count == 1);
  CHECK(curve_plan.active_mode_tag == "far_obs_slow");
  CHECK(std::abs(curve_plan.target_y_m) < 0.01);  // stays on lane center
  CHECK(curve_plan.speed_cap_mps <= cfg.obstacle_speed_mps + 1e-6);
}

void test_b3_rear_axle_rotation_pivot()
{
  ObstacleMemoryConfig cfg{};
  cfg.rear_axle_to_bumper_m = 0.18;
  cfg.car_length_m = 0.28;
  cfg.rear_keep_margin_m = 0.15;
  ObstacleMemory mem(cfg);

  // Car rear axle at world (0, 0), facing +X (yaw = 0).
  // Obstacle at d_bumper = -0.08m (0.10m ahead of rear axle), y_lat = +0.16m (left of rear axle).
  const VehiclePose2D pose0{true, 0.0, 0.0, 0.0, 0.3, 0.0, 1.0};
  double wx = 0.0, wy = 0.0;
  ObstacleMemory::bumper_to_world(pose0, -0.08, +0.16, wx, wy, cfg.rear_axle_to_bumper_m);
  CHECK(std::abs(wx - 0.10) < 1e-9);
  CHECK(std::abs(wy - 0.16) < 1e-9);

  // Car pivots 90 deg CCW around rear axle (0, 0) -> now facing +Y world.
  constexpr double kHalfPi = 1.5707963267948966;
  const VehiclePose2D pose_rot{true, 0.0, 0.0, kHalfPi, 0.3, 0.5, 1.2};
  double d_rot = 0.0, y_rot = 0.0;
  ObstacleMemory::world_to_bumper(pose_rot, wx, wy, d_rot, y_rot, cfg.rear_axle_to_bumper_m);
  // World (0.10, 0.16) relative to +Y heading: +0.16m ahead of rear axle (d_bumper = -0.02),
  // and +0.10m to the RIGHT (y_lat = -0.10).
  CHECK(std::abs(d_rot - (-0.02)) < 1e-6);
  CHECK(std::abs(y_rot - (-0.10)) < 1e-6);

  // Full ObstacleMemory update & query preserves exact rear-axle transform
  GroundDetection det{};
  det.type = SemanticObjectType::RED_NO_ENTRY;
  det.confidence = 0.9;
  det.d_bumper_m = -0.08;
  det.y_lat_m = +0.16;
  det.half_len_m = 0.065;
  det.half_wid_m = 0.065;
  mem.update({det}, pose0, 1.0);
  const auto locs = mem.get_local_obstacles(pose_rot, 1.2);
  CHECK(locs.size() == 1);
  CHECK(std::abs(locs[0].d_bumper_m - (-0.02)) < 1e-6);
  CHECK(std::abs(locs[0].y_lat_m - (-0.10)) < 1e-6);
}

void test_b7_reverse_motion_never_adds_arc_length()
{
  FinishGuardConfig cfg{};
  cfg.odom_scale_factor = 1.0;
  cfg.finish_min_dist_m = 8.5;
  FinishGuard guard(cfg);

  guard.update_motion(2.5, 0.2);
  CHECK(std::abs(guard.total_arc_m() - 2.5) < 1e-9);

  guard.mark_yellow_stop_triggered();
  CHECK(std::abs(guard.dist_since_last_yellow_m()) < 1e-9);

  // Reverse braking / pushback (ds_fwd_m < 0) MUST NOT accumulate forward mileage or rearm!
  guard.update_motion(-0.40, 0.01);
  guard.update_motion(-0.10, 0.01);
  CHECK(std::abs(guard.total_arc_m() - 2.5) < 1e-9);
  CHECK(std::abs(guard.dist_since_last_yellow_m()) < 1e-9);
  CHECK(!guard.is_yellow_rearmed());
}

void test_b9_same_digit_vector_tracks_and_horizon_clamping()
{
  CorridorPlannerConfig cfg{};
  cfg.obstacle_lookahead_m = 1.00;
  cfg.obstacle_steer_horizon_m = 5.0;      // invalid > lookahead -> clamped to 1.00
  cfg.obstacle_curve_min_horizon_m = 2.0;  // invalid > steer_horizon -> clamped to 1.00
  cfg.obstacle_stagger_gap_m = 0.0;        // invalid < 0.05 -> clamped to 0.05
  cfg.digit_min_conf = 0.55;
  cfg.digit_min_frames = 2;
  cfg.allow_dual_digit_cover = true;
  cfg.max_digit_lateral_shift_m = 0.15;

  CorridorPlanner planner(cfg);
  CHECK(std::abs(planner.config().obstacle_steer_horizon_m - 1.00) < 1e-9);
  CHECK(std::abs(planner.config().obstacle_curve_min_horizon_m - 1.00) < 1e-9);
  CHECK(std::abs(planner.config().obstacle_stagger_gap_m - 0.05) < 1e-9);

  // Guard: obstacle_lookahead_m < 0.25 must clamp up to 0.25 before std::clamp(steer_horizon, 0.25, lookahead)
  CorridorPlannerConfig small_cfg = cfg;
  small_cfg.obstacle_lookahead_m = 0.10;
  small_cfg.obstacle_steer_horizon_m = 0.20;
  small_cfg.obstacle_curve_min_horizon_m = 0.10;
  CorridorPlanner small_planner(small_cfg);
  CHECK(std::abs(small_planner.config().obstacle_lookahead_m - 0.25) < 1e-9);
  CHECK(std::abs(small_planner.config().obstacle_steer_horizon_m - 0.25) < 1e-9);
  CHECK(std::abs(small_planner.config().obstacle_curve_min_horizon_m - 0.20) < 1e-9);

  // Two IDENTICAL digits (both digit=5) 10cm apart on a straight, with distinct track_ids
  GroundDetection d5_left{SemanticObjectType::BONUS_DIGIT, 5, 0.90, 0.65, +0.05, 0.04, 0.04, 21, 10.0};
  GroundDetection d5_right{SemanticObjectType::BONUS_DIGIT, 5, 0.88, 0.65, -0.05, 0.04, 0.04, 22, 10.0};
  planner.plan(+0.215, -0.215, 0.0, {}, {d5_left, d5_right}, false, 2.0, 10.0);
  const CorridorPlan dual_plan =
    planner.plan(+0.215, -0.215, 0.0, {}, {d5_left, d5_right}, false, 2.0, 10.05);
  CHECK(dual_plan.digit_target_active && dual_plan.target_digit == 5);
  // Both tracks preserved -> dual_digit_cover centers midpoint at ~0.0 (instead of clobbering to +0.05 or -0.05)
  CHECK(std::abs(dual_plan.target_y_m) < 0.015);

  // Same test WITHOUT track_ids (track_id = -1), spatial nearest-neighbor must also preserve both tracks!
  planner.reset();
  GroundDetection d6_l{SemanticObjectType::BONUS_DIGIT, 6, 0.90, 0.65, +0.06, 0.04, 0.04, -1, 12.0};
  GroundDetection d6_r{SemanticObjectType::BONUS_DIGIT, 6, 0.85, 0.65, -0.06, 0.04, 0.04, -1, 12.0};
  planner.plan(+0.215, -0.215, 0.0, {}, {d6_l, d6_r}, false, 2.5, 12.0);
  const CorridorPlan spatial_plan =
    planner.plan(+0.215, -0.215, 0.0, {}, {d6_l, d6_r}, false, 2.5, 12.05);
  CHECK(spatial_plan.digit_target_active && spatial_plan.target_digit == 6);
  CHECK(std::abs(spatial_plan.target_y_m) < 0.015);
}

void test_tier1_obb_yaw_and_ackermann_cbf_shield()
{
  constexpr double kHalfPi = 1.5707963267948966;

  // 1. ObstacleMemory OBB yaw recording & relative rotation
  {
    ObstacleMemoryConfig mcfg{};
    mcfg.rear_axle_to_bumper_m = 0.18;
    mcfg.car_length_m = 0.28;
    mcfg.rear_keep_margin_m = 0.15;
    ObstacleMemory mem(mcfg);

    const VehiclePose2D pose0{true, 0.0, 0.0, 0.25, 0.3, 0.0, 1.0};
    GroundDetection det{};
    det.type = SemanticObjectType::RED_NO_ENTRY;
    det.confidence = 0.92;
    det.d_bumper_m = 0.40;
    det.y_lat_m = 0.10;
    det.half_len_m = 0.12;
    det.half_wid_m = 0.065;
    mem.update({det}, pose0, 1.0);
    CHECK(mem.size() == 1);
    CHECK(std::abs(mem.entries()[0].obs_yaw_rad - 0.25) < 1e-6);

    // Car turns +90 deg CCW -> relative OBB yaw must be -pi/2
    const VehiclePose2D pose1{true, 0.0, 0.0, 0.25 + kHalfPi, 0.3, 0.5, 1.2};
    const auto locs = mem.get_local_obstacles(pose1, 1.2);
    CHECK(locs.size() == 1);
    CHECK(std::abs(locs[0].rel_yaw_rad - (-kHalfPi)) < 1e-6);

    // Re-observing the same obstacle at rotated pose preserves original obs_yaw_rad
    GroundDetection det_rot = det;
    det_rot.d_bumper_m = locs[0].d_bumper_m;
    det_rot.y_lat_m = locs[0].y_lat_m;
    mem.update({det_rot}, pose1, 1.25);
    CHECK(mem.size() == 1);
    CHECK(std::abs(mem.entries()[0].obs_yaw_rad - 0.25) < 1e-6);
  }

  // 2. Analytical OBB SDF & 3-disc geometry check
  {
    // Unrotated 24x13cm box: hl=0.12, hw=0.065
    CHECK(std::abs(AckermannCbfShield::point_to_obb_sdf(0.22, 0.0, 0.0, 0.0, 0.12, 0.065, 0.0) - 0.10) < 1e-6);
    CHECK(std::abs(AckermannCbfShield::point_to_obb_sdf(0.0, 0.165, 0.0, 0.0, 0.12, 0.065, 0.0) - 0.10) < 1e-6);
    // Rotated 90 deg: hl=0.12 now aligns with Y, hw=0.065 aligns with X
    CHECK(std::abs(AckermannCbfShield::point_to_obb_sdf(0.165, 0.0, 0.0, 0.0, 0.12, 0.065, kHalfPi) - 0.10) < 1e-6);
    CHECK(std::abs(AckermannCbfShield::point_to_obb_sdf(0.0, 0.22, 0.0, 0.0, 0.12, 0.065, kHalfPi) - 0.10) < 1e-6);
    // Inside center
    CHECK(std::abs(AckermannCbfShield::point_to_obb_sdf(0.0, 0.0, 0.0, 0.0, 0.12, 0.065, kHalfPi) - (-0.065)) < 1e-6);

    AckermannCbfConfig cfg{};
    const auto offsets = AckermannCbfShield::compute_disc_offsets(cfg);
    const double r = 0.5 * cfg.car_width_m;
    // Rear disc rear edge = rear_axle_to_bumper - car_length = -0.10; front disc front edge = +0.18
    CHECK(std::abs((offsets[0] - r) - (cfg.rear_axle_to_bumper_m - cfg.car_length_m)) < 1e-6);
    CHECK(std::abs((offsets[2] + r) - cfg.rear_axle_to_bumper_m) < 1e-6);
  }

  // 3. 3-disc DCBF: clean cruise, rear-wheel cut-in protection, lane border hard limit, speed ladder
  {
    AckermannCbfConfig cfg{};
    cfg.car_length_m = 0.280;
    cfg.car_width_m = 0.191;
    cfg.rear_axle_to_bumper_m = 0.180;
    cfg.wheelbase_m = 0.142;
    cfg.max_steer_rad = 35.0 * (3.14159265358979323846 / 180.0);
    cfg.max_yaw_rate = 1.6;
    cfg.border_margin_m = 0.025;
    cfg.cone_margin_m = 0.025;
    cfg.no_entry_margin_m = 0.030;
    cfg.horizon_steps = 6;
    cfg.step_dt_sec = 0.06;
    cfg.cbf_gamma = 0.60;
    AckermannCbfShield shield(cfg);

    // 3a) Clean lane, straight nominal -> no intervention
    const CbfShieldResult clean = shield.filter(0.44, 0.0, +0.215, -0.215, {});
    CHECK(!clean.intervened);
    CHECK(std::abs(clean.linear_speed - 0.44) < 1e-6);
    CHECK(std::abs(clean.yaw_rate) < 1e-6);
    CHECK(clean.min_clearance_m > 0.08);

    // 3b) Red no-entry (24x13cm) beside LEFT rear/mid chassis; nominal turns LEFT (+1.0) into it!
    LocalObstacle rear_left_ne{};
    rear_left_ne.type = SemanticObjectType::RED_NO_ENTRY;
    rear_left_ne.d_bumper_m = -0.05;
    rear_left_ne.d_min_m = -0.17;
    rear_left_ne.d_max_m = +0.07;
    rear_left_ne.y_lat_m = +0.205;
    rear_left_ne.y_min_m = +0.140;
    rear_left_ne.y_max_m = +0.270;
    rear_left_ne.rel_yaw_rad = 0.0;

    const CbfShieldResult guard_rear =
      shield.filter(0.24, +1.0, +0.215, -0.215, {rear_left_ne});
    CHECK(guard_rear.intervened);
    CHECK(guard_rear.yaw_rate < +0.6);  // stops cutting left into rear-side no-entry
    CHECK(guard_rear.min_clearance_m >= -1e-6);

    // 3c) Lane boundary hard constraint: aggressive left turn near left white border must NOT push car out
    const CbfShieldResult guard_lane =
      shield.filter(0.30, +1.4, +0.150, -0.280, {});
    CHECK(guard_lane.intervened);
    CHECK(guard_lane.limiting_source == "lane_left");
    CHECK(guard_lane.yaw_rate < 1.0);
    CHECK(guard_lane.min_clearance_m >= -1e-6);

    // 3d) Rotated OBB (90 deg): 24cm long axis protrudes laterally into path ahead-left
    LocalObstacle rot_ne{};
    rot_ne.type = SemanticObjectType::RED_NO_ENTRY;
    rot_ne.d_bumper_m = 0.14;
    rot_ne.d_min_m = 0.02;   // half_len = 0.12
    rot_ne.d_max_m = 0.26;
    rot_ne.y_lat_m = +0.19;
    rot_ne.y_min_m = +0.125; // half_wid = 0.065
    rot_ne.y_max_m = +0.255;
    rot_ne.rel_yaw_rad = 0.0;
    const CbfShieldResult unrot_res = shield.filter(0.26, 0.0, +0.215, -0.215, {rot_ne});
    rot_ne.rel_yaw_rad = kHalfPi;  // now extends laterally down to y = +0.19 - 0.12 = +0.07!
    const CbfShieldResult rot_res = shield.filter(0.26, 0.0, +0.215, -0.215, {rot_ne});
    // Unrotated is clear at y=0 (gap = 0.125 - 0.0955 - 0.030 ~ 0); rotated protrudes 5.5cm deeper into lane!
    CHECK(rot_res.intervened);
    CHECK(rot_res.limiting_source == "red_no_entry");
    CHECK(rot_res.yaw_rate < unrot_res.yaw_rate - 0.05 || rot_res.linear_speed < unrot_res.linear_speed);

    // 3e) Completely blocked corridor dead ahead -> speed ladder drops safely to 0.0
    LocalObstacle wall_ne{};
    wall_ne.type = SemanticObjectType::RED_NO_ENTRY;
    wall_ne.d_bumper_m = 0.07;
    wall_ne.d_min_m = 0.01;
    wall_ne.d_max_m = 0.13;
    wall_ne.y_lat_m = 0.0;
    wall_ne.y_min_m = -0.22;
    wall_ne.y_max_m = +0.22;
    wall_ne.rel_yaw_rad = 0.0;

    const CbfShieldResult blocked = shield.filter(0.30, 0.0, +0.215, -0.215, {wall_ne});
    CHECK(blocked.intervened);
    CHECK(std::abs(blocked.linear_speed) < 1e-6);
    CHECK(std::abs(blocked.yaw_rate) < 1e-6);
    CHECK(blocked.limiting_source == "red_no_entry");

    // 3f) Non-finite (NaN) command and lane bounds fail-safe check
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const CbfShieldResult nan_cmd = shield.filter(qnan, 0.2, qnan, qnan, {});
    CHECK(nan_cmd.intervened);
    CHECK(std::abs(nan_cmd.linear_speed) < 1e-9 && std::abs(nan_cmd.yaw_rate) < 1e-9);
    CHECK(nan_cmd.limiting_source == "invalid_cmd");
    const CbfShieldResult nan_lane = shield.filter(0.30, 0.0, qnan, qnan, {});
    CHECK(!nan_lane.intervened && nan_lane.min_clearance_m > 0.08);
  }
}

void test_tier1_finish_guard_ramp_voltage_and_adaptation()
{
  FinishGuardConfig cfg{};
  cfg.pre_brake_dist_m = 0.70;
  cfg.creep_speed_mps = 0.24;
  cfg.yellow_final_ramp_dist_m = 0.35;
  cfg.yellow_final_creep_speed_mps = 0.18;
  cfg.brake_decel_mps2 = 1.8;
  cfg.control_latency_s = 0.10;
  cfg.bumper_margin_m = 0.08;
  cfg.voltage_decel_slope = 0.25;
  cfg.voltage_timeout_sec = 3.0;
  cfg.enable_brake_adaptation = true;
  cfg.brake_adapt_ema_alpha = 0.35;
  cfg.friction_scale_min = 0.65;
  cfg.friction_scale_max = 1.25;
  FinishGuard guard(cfg);

  // 1. Near-line ramp pre-deceleration: 0.24 down to 0.18 inside 0.35m
  const YellowStopAdvice adv_outer = guard.evaluate_yellow_stop(true, true, 0.50, 0.24, false, 1.0);
  const YellowStopAdvice adv_ramp1 = guard.evaluate_yellow_stop(true, true, 0.28, 0.24, false, 1.0);
  const YellowStopAdvice adv_ramp2 = guard.evaluate_yellow_stop(true, true, 0.18, 0.20, false, 1.0);
  CHECK(adv_outer.pre_brake && !adv_outer.trigger_stop);
  CHECK(std::abs(adv_outer.speed_cap_mps - 0.24) < 1e-6);
  CHECK(adv_ramp1.pre_brake && !adv_ramp1.trigger_stop);
  CHECK(adv_ramp1.speed_cap_mps < 0.24 && adv_ramp1.speed_cap_mps > 0.18);
  CHECK(adv_ramp2.pre_brake && !adv_ramp2.trigger_stop);
  CHECK(adv_ramp2.speed_cap_mps < adv_ramp1.speed_cap_mps && adv_ramp2.speed_cap_mps >= 0.18);

  // 2. Battery voltage feedforward + [9.0, 15.0]V rejection + 3.0s timeout fallback to 12.0V
  guard.update_battery_voltage(5.0, 2.0);   // invalid low -> ignored
  guard.update_battery_voltage(18.5, 2.1);  // invalid high -> ignored
  CHECK(!guard.battery_voltage_valid());
  CHECK(std::abs(guard.effective_brake_decel_mps2(2.2) - 1.8) < 1e-6);

  guard.update_battery_voltage(10.8, 10.0);  // -1.2V -> 1.8 - 0.3 = 1.5 m/s^2
  CHECK(guard.battery_voltage_valid());
  CHECK(std::abs(guard.effective_brake_decel_mps2(10.5) - 1.5) < 1e-6);
  const double d_low_v = guard.compute_stop_trigger_dist(0.30, 10.5);

  guard.update_battery_voltage(12.8, 11.0);  // +0.8V -> 1.8 + 0.2 = 2.0 m/s^2
  CHECK(std::abs(guard.effective_brake_decel_mps2(11.5) - 2.0) < 1e-6);
  const double d_high_v = guard.compute_stop_trigger_dist(0.30, 11.5);
  CHECK(d_low_v > d_high_v + 0.005);  // lower battery -> earlier stop trigger!

  // Timeout (> 3.0s since t=11.0 -> at t=14.5 falls back to 12.0V -> 1.8 m/s^2)
  CHECK(std::abs(guard.active_battery_voltage(14.5) - 12.0) < 1e-6);
  CHECK(std::abs(guard.effective_brake_decel_mps2(14.5) - 1.8) < 1e-6);

  // 3. 1-param clamped friction EMA adaptation (never diverges)
  CHECK(!guard.update_measured_yellow_stop(0.08, 0.02, 15.0));   // v_entry < 0.12 rejected
  CHECK(!guard.update_measured_yellow_stop(0.24, 0.002, 15.0));  // stop_travel <= 0.005 rejected
  CHECK(std::abs(guard.friction_scale() - 1.0) < 1e-9);

  // Slippery floor (measured decel ~ 0.45 m/s^2) -> converges smoothly & clamps at 0.65
  for (int i = 0; i < 40; ++i) {
    CHECK(guard.update_measured_yellow_stop(0.24, 0.064, 15.0 + i));
    CHECK(guard.friction_scale() >= 0.65 - 1e-9 && guard.friction_scale() <= 1.25 + 1e-9);
  }
  CHECK(std::abs(guard.friction_scale() - 0.65) < 1e-4);
  CHECK(guard.effective_brake_decel_mps2(60.0) < 1.20);

  // Ultra-grippy floor (measured decel ~ 5.0 m/s^2) -> converges smoothly & clamps at 1.25
  for (int i = 0; i < 40; ++i) {
    CHECK(guard.update_measured_yellow_stop(0.30, 0.009, 65.0 + i));
    CHECK(guard.friction_scale() >= 0.65 - 1e-9 && guard.friction_scale() <= 1.25 + 1e-9);
  }
  CHECK(std::abs(guard.friction_scale() - 1.25) < 1e-4);
  CHECK(std::abs(guard.effective_brake_decel_mps2(120.0) - 2.25) < 1e-3);

  // Disabling adaptation locks effective decel back to nominal
  cfg.enable_brake_adaptation = false;
  guard.set_config(cfg);
  CHECK(!guard.update_measured_yellow_stop(0.24, 0.05, 130.0));
  CHECK(std::abs(guard.effective_brake_decel_mps2(130.0) - 1.8) < 1e-6);
}

void test_tier1_corridor_risk_aware_fork_and_straight_digits()
{
  // 1. Fork voting: noisy flickering high digit discounted by confidence & consistency
  CorridorPlannerConfig cfg{};
  cfg.track_mode = "fork";
  cfg.fork_outer_branch = -1;
  cfg.fork_outer_time_penalty_s = 3.0;
  cfg.bonus_score_weight_s = 5.0;
  cfg.fork_min_votes = 3;
  cfg.digit_min_conf = 0.55;
  cfg.digit_min_frames = 2;
  cfg.max_digit_lateral_shift_m = 0.12;
  cfg.enable_risk_aware_decision = true;

  CorridorPlanner risk_planner(cfg);

  // Left outer flickers between 6 (3 votes @ 0.60) and 3 (3 votes @ 0.57) -> p_conf_eff = 0.30
  // Right inner is rock-solid 5 (6 votes @ 0.92) -> p_conf_eff = 0.92
  GroundDetection l6{SemanticObjectType::BONUS_DIGIT, 6, 0.60, 0.80, +0.10, 0.05, 0.05, 1, 1.0};
  GroundDetection l3{SemanticObjectType::BONUS_DIGIT, 3, 0.57, 0.80, +0.10, 0.05, 0.05, 1, 1.0};
  GroundDetection r5{SemanticObjectType::BONUS_DIGIT, 5, 0.92, 0.80, -0.10, 0.05, 0.05, 2, 1.0};
  for (int i = 0; i < 3; ++i) {
    risk_planner.update_fork_static_votes({l6, r5}, 1.0 + i * 0.05);
    risk_planner.update_fork_static_votes({l3, r5}, 1.2 + i * 0.05);
  }
  const auto fsum = risk_planner.fork_summary();
  CHECK(fsum.left_digit == 6 && fsum.right_digit == 5);
  CHECK(fsum.left_conf_eff < 0.35 && fsum.right_conf_eff > 0.85);
  CHECK(risk_planner.finalize_fork_decision(2.0) == +1);  // Picks confident inner right (+1)

  // With legacy switch disabled, mechanical -5*d picks noisy outer left (-1)
  cfg.enable_risk_aware_decision = false;
  CorridorPlanner legacy_planner(cfg);
  for (int i = 0; i < 3; ++i) {
    legacy_planner.update_fork_static_votes({l6, r5}, 1.0 + i * 0.05);
    legacy_planner.update_fork_static_votes({l3, r5}, 1.2 + i * 0.05);
  }
  CHECK(legacy_planner.finalize_fork_decision(2.0) == -1);

  // 2. Straight multi-digit scoring: safe centered 5 beats edge-hugging low-conf 7
  cfg.enable_risk_aware_decision = true;
  CorridorPlanner straight_risk(cfg);
  // Safe corridor for lane [-0.215, +0.215] is [-0.0945, +0.0945].
  // Digit 7 sits at +0.090 (edge_clear = 0.0045m < 0.04m, large shift, conf=0.58)
  // Digit 5 sits at -0.010 (edge_clear = 0.0845m >= 0.04m, centered, conf=0.94)
  GroundDetection edge7{SemanticObjectType::BONUS_DIGIT, 7, 0.58, 0.65, +0.090, 0.04, 0.04, 31, 5.0};
  GroundDetection safe5{SemanticObjectType::BONUS_DIGIT, 5, 0.94, 0.65, -0.010, 0.04, 0.04, 32, 5.0};
  straight_risk.plan(+0.215, -0.215, 0.0, {}, {edge7, safe5}, false, 1.0, 5.0);
  const CorridorPlan rplan =
    straight_risk.plan(+0.215, -0.215, 0.0, {}, {edge7, safe5}, false, 1.0, 5.05);
  CHECK(rplan.digit_target_active && rplan.target_digit == 5);
  CHECK(rplan.target_y_m >= rplan.safe_y_min_m && rplan.target_y_m <= rplan.safe_y_max_m);

  // Legacy planner mechanically chases digit 7
  cfg.enable_risk_aware_decision = false;
  CorridorPlanner straight_legacy(cfg);
  straight_legacy.plan(+0.215, -0.215, 0.0, {}, {edge7, safe5}, false, 1.0, 5.0);
  const CorridorPlan lplan =
    straight_legacy.plan(+0.215, -0.215, 0.0, {}, {edge7, safe5}, false, 1.0, 5.05);
  CHECK(lplan.digit_target_active && lplan.target_digit == 7);

  // Hard corridor rule still rejects any digit outside safe corridor
  GroundDetection out8{SemanticObjectType::BONUS_DIGIT, 8, 0.99, 0.65, +0.160, 0.04, 0.04, 33, 6.0};
  straight_risk.reset();
  straight_risk.plan(+0.215, -0.215, 0.0, {}, {out8}, false, 1.5, 6.0);
  const CorridorPlan oplan =
    straight_risk.plan(+0.215, -0.215, 0.0, {}, {out8}, false, 1.5, 6.05);
  CHECK(!oplan.digit_target_active);
}

void test_dual_layer_brake_and_safe_speed_ceiling()
{
  FinishGuardConfig cfg{};
  cfg.pre_brake_dist_m = 0.70;
  cfg.creep_speed_mps = 0.55;
  cfg.yellow_final_ramp_dist_m = 0.35;
  cfg.yellow_final_creep_speed_mps = 0.14;
  cfg.brake_decel_mps2 = 1.8;
  cfg.control_latency_s = 0.10;
  cfg.bumper_margin_m = 0.08;
  cfg.voltage_decel_slope = 0.25;
  cfg.enable_brake_adaptation = true;
  cfg.enable_sqrt_brake_envelope = true;
  cfg.enable_closed_loop_brake = true;
  cfg.brake_damping_gain = 0.85;
  cfg.brake_zero_cutoff_mps = 0.025;
  FinishGuard guard(cfg);

  // 1. Sqrt brake envelope: drops smoothly below creep_speed_mps as d -> stop_trigger_dist,
  //    and tightens when effective decel drops via lower battery voltage!
  guard.update_battery_voltage(12.0, 1.0);
  const double d_trig_nom = guard.compute_stop_trigger_dist(0.50, 1.0);
  const double d_test = d_trig_nom + 0.06;  // inside pre-brake zone -> smoothly below creep_speed_mps
  const YellowStopAdvice adv_sqrt = guard.evaluate_yellow_stop(true, true, d_test, 0.50, false, 1.0);
  CHECK(adv_sqrt.pre_brake && !adv_sqrt.trigger_stop);
  CHECK(adv_sqrt.speed_cap_mps >= cfg.yellow_final_creep_speed_mps - 1e-9);
  CHECK(adv_sqrt.speed_cap_mps < cfg.creep_speed_mps);

  // Low battery voltage lowers a_eff -> lowers speed cap at the same physical bumper distance
  guard.update_battery_voltage(10.0, 1.2);  // a_eff = 1.8 - 0.5 = 1.3
  const YellowStopAdvice adv_low_v = guard.evaluate_yellow_stop(
    true, true, d_test, 0.50, false, 1.2);
  CHECK(adv_low_v.speed_cap_mps < adv_sqrt.speed_cap_mps);

  // Legacy fallback when enable_sqrt_brake_envelope = false vs exact sqrt envelope at fixed d_rem
  FinishGuardConfig cfg_lin = cfg;
  cfg_lin.yellow_final_ramp_dist_m = 0.0;  // pure flat creep vs sqrt envelope
  cfg_lin.enable_sqrt_brake_envelope = false;
  FinishGuard guard_lin(cfg_lin);
  const double d_lin = guard_lin.compute_stop_trigger_dist(0.50, 1.0) + 0.04;
  const YellowStopAdvice adv_lin = guard_lin.evaluate_yellow_stop(true, true, d_lin, 0.50, false, 1.0);
  CHECK(std::abs(adv_lin.speed_cap_mps - cfg_lin.creep_speed_mps) < 1e-6);

  cfg_lin.enable_sqrt_brake_envelope = true;
  guard_lin.set_config(cfg_lin);
  const YellowStopAdvice adv_sqrt_pure =
    guard_lin.evaluate_yellow_stop(true, true, d_lin, 0.50, false, 1.0);
  const double expected_v_env = std::sqrt(2.0 * 1.8 * 0.04);
  CHECK(std::abs(adv_sqrt_pure.speed_cap_mps - expected_v_env) < 1e-6);

  // Pure sqrt envelope at fixed remaining distance strictly lowers with lower battery voltage
  guard_lin.update_battery_voltage(10.0, 1.2);  // a_eff = 1.3
  const double d_lin_low = guard_lin.compute_stop_trigger_dist(0.50, 1.2) + 0.04;
  const YellowStopAdvice adv_sqrt_low =
    guard_lin.evaluate_yellow_stop(true, true, d_lin_low, 0.50, false, 1.2);
  const double expected_v_env_low = std::sqrt(2.0 * 1.3 * 0.04);
  CHECK(std::abs(adv_sqrt_low.speed_cap_mps - expected_v_env_low) < 1e-6);
  CHECK(adv_sqrt_low.speed_cap_mps < adv_sqrt_pure.speed_cap_mps);

  // 2. Closed-loop proportional damping brake & zero-reverse cutoff
  const double b_high = guard.compute_closed_loop_brake_cmd(0.40, 0.05, 0.25, 0.35);
  CHECK(std::abs(b_high - (-0.25)) < 1e-9);  // 0.85 * 0.40 = 0.34 -> clamped to -0.25
  const double b_mid = guard.compute_closed_loop_brake_cmd(0.20, 0.10, 0.25, 0.35);
  CHECK(std::abs(b_mid - (-0.17)) < 1e-9);   // -0.85 * 0.20 = -0.17
  const double b_cut = guard.compute_closed_loop_brake_cmd(0.025, 0.12, 0.25, 0.35);
  CHECK(std::abs(b_cut) < 1e-12);            // snaps cleanly to 0.0 at <= cutoff
  const double b_neg = guard.compute_closed_loop_brake_cmd(-0.01, 0.12, 0.25, 0.35);
  CHECK(std::abs(b_neg) < 1e-12);            // never commands reverse when already <= 0
  const double b_to = guard.compute_closed_loop_brake_cmd(0.20, 0.35, 0.25, 0.35);
  CHECK(std::abs(b_to) < 1e-12);             // timeout cuts off cleanly

  // 3. Exact kinematic safe speed ceiling: v^2/(2*a_eff) + v*t_lat == max(0, d - margin) + v_end^2/(2*a_eff)
  guard.update_battery_voltage(12.0, 2.0);
  const double a_eff = guard.effective_brake_decel_mps2(2.0);
  const double preview = 0.95;
  const double v_end = 0.22;
  const double v_safe = guard.compute_safe_speed_ceiling(preview, v_end, 2.0);
  CHECK(v_safe >= v_end);
  const double lhs = (v_safe * v_safe) / (2.0 * a_eff) + v_safe * cfg.control_latency_s;
  const double rhs = std::max(0.0, preview - cfg.bumper_margin_m) + (v_end * v_end) / (2.0 * a_eff);
  CHECK(std::abs(lhs - rhs) < 1e-9);

  // Shorter preview or lower friction strictly lowers v_safe
  const double v_safe_short = guard.compute_safe_speed_ceiling(0.30, 0.0, 2.0);
  const double v_safe_long = guard.compute_safe_speed_ceiling(0.95, 0.0, 2.0);
  CHECK(v_safe_short > 0.0 && v_safe_short < v_safe_long);

  // When preview <= margin and v_end > 0, still satisfies v_safe >= v_end
  CHECK(std::abs(guard.compute_safe_speed_ceiling(0.05, 0.20, 2.0) - 0.20) < 1e-9);
  CHECK(std::abs(guard.compute_safe_speed_ceiling(0.05, 0.0, 2.0)) < 1e-9);
}

void test_clear_path_adaptive_speed_and_asymmetric_slew()
{
  MotionControllerConfig cfg{};
  cfg.target_speed = 0.44;
  cfg.max_speed = 0.85;
  cfg.min_speed = 0.0;
  cfg.max_yaw_rate = 1.6;
  cfg.speed_gain_damping = 0.45;
  cfg.low_confidence_threshold = 0.55;
  cfg.low_confidence_speed = 0.24;
  cfg.enable_clear_path_speed = true;
  cfg.clear_straight_speed = 0.62;
  cfg.clear_preview_dist_m = 0.95;
  cfg.max_accel_mps2 = 0.90;
  cfg.max_decel_mps2 = 2.20;
  cfg.active_decel_kp = 0.45;

  MotionController ctrl(cfg);
  CurveAssistState curve{};
  curve.near_valid = true;
  curve.far_valid = true;
  curve.last_lane_width_ratio = 0.58;
  MotionState motion{};

  // 1. Clean centered straight -> clear_path_score == 1.0, smoothly ramps from 0.44 -> 0.62
  ClearPathInput cp{};
  cp.confidence = 0.95;
  cp.curvature_proxy = 0.02;
  cp.heading_proxy = 0.01;
  cp.lateral_err_m = 0.01;
  cp.lane_width_ratio = 0.58;
  cp.v_safe_cap_mps = 1.50;
  cp.odom_linear_x = 0.44;
  cp.odom_valid = true;
  CHECK(std::abs(ctrl.compute_clear_path_score(cp) - 1.0) < 1e-9);

  // First frame starts anchored at base cruise speed (0.44), then ramps at 0.90 m/s^2
  const AdaptiveSpeedResult r0 = ctrl.step_adaptive_corridor_command(
    0.0, 0.0, 0.01, 0.02, 0.95, 0.85, curve, motion, cp, 0.0);
  CHECK(std::abs(r0.clear_path_score - 1.0) < 1e-9);
  CHECK(r0.command.linear_speed <= 0.45);

  double last_v = r0.command.linear_speed;
  for (int i = 1; i <= 10; ++i) {
    cp.odom_linear_x = last_v;
    const AdaptiveSpeedResult ri = ctrl.step_adaptive_corridor_command(
      0.0, 0.0, 0.01, 0.02, 0.95, 0.85, curve, motion, cp, i * 0.05);
    CHECK(ri.command.linear_speed >= last_v - 1e-9);
    CHECK(ri.command.linear_speed <= last_v + cfg.max_accel_mps2 * 0.05 + 1e-6);
    last_v = ri.command.linear_speed;
  }
  CHECK(last_v > 0.58 && last_v <= 0.62 + 1e-6);

  // 2. Immediate drop to clear_path_score == 0.0 on obstacle, no-entry, yellow, or sharp curve,
  //    plus asymmetric fast decel (2.20 m/s^2) and non-negative active decel!
  ClearPathInput cp_obs = cp;
  cp_obs.has_active_obstacle = true;
  cp_obs.odom_linear_x = 0.62;
  CHECK(std::abs(ctrl.compute_clear_path_score(cp_obs)) < 1e-9);

  const AdaptiveSpeedResult r_obs = ctrl.step_adaptive_corridor_command(
    0.0, 0.0, 0.01, 0.02, 0.95, 0.22, curve, motion, cp_obs, 0.55);
  CHECK(std::abs(r_obs.clear_path_score) < 1e-9);
  CHECK(r_obs.command.linear_speed <= 0.22 + 1e-6);
  CHECK(r_obs.command.linear_speed >= 0.0);  // active decel NEVER commands < 0.0 during tracking!

  // Active decel pulls command below speed_cap when odom overspeeds sharp curve / obstacle
  ClearPathInput cp_over = cp_obs;
  cp_over.dt_sec = 0.10;
  cp_over.prev_speed_cmd_mps = 0.22;
  cp_over.odom_linear_x = 0.60;
  const AdaptiveSpeedResult r_over = ctrl.compute_adaptive_corridor_command(
    0.0, 0.0, 0.0, 0.0, 0.95, 0.22, curve, motion, cp_over);
  CHECK(r_over.command.linear_speed < 0.22 - 0.05);
  CHECK(r_over.command.linear_speed >= 0.0);

  // Wide junction & sharp curve drop clear_path_score to 0.0
  ClearPathInput cp_wide = cp;
  cp_wide.lane_width_ratio = 0.88;
  CHECK(std::abs(ctrl.compute_clear_path_score(cp_wide)) < 1e-9);
  ClearPathInput cp_curve = cp;
  cp_curve.curvature_proxy = 0.30;
  CHECK(std::abs(ctrl.compute_clear_path_score(cp_curve)) < 1e-9);
  ClearPathInput cp_yellow = cp;
  cp_yellow.has_yellow_ahead = true;
  CHECK(std::abs(ctrl.compute_clear_path_score(cp_yellow)) < 1e-9);

  // 3. Physics safe ceiling caps clear-path speed below clear_straight_speed
  ClearPathInput cp_cap = cp;
  cp_cap.v_safe_cap_mps = 0.48;
  cp_cap.dt_sec = 0.50;
  cp_cap.prev_speed_cmd_mps = 0.44;
  cp_cap.odom_linear_x = 0.44;
  const AdaptiveSpeedResult r_cap = ctrl.compute_adaptive_corridor_command(
    0.0, 0.0, 0.0, 0.0, 0.95, 0.85, curve, motion, cp_cap);
  CHECK(std::abs(r_cap.command.linear_speed - 0.48) < 1e-6);

  // 4. Asymmetric slew check: decel step > accel step for same dt
  ClearPathInput cp_acc = cp;
  cp_acc.dt_sec = 0.05;
  cp_acc.prev_speed_cmd_mps = 0.44;
  cp_acc.odom_valid = false;
  const AdaptiveSpeedResult r_acc = ctrl.compute_adaptive_corridor_command(
    0.0, 0.0, 0.0, 0.0, 0.95, 0.85, curve, motion, cp_acc);
  const double delta_up = r_acc.command.linear_speed - 0.44;

  ClearPathInput cp_dec = cp;
  cp_dec.has_active_obstacle = true;
  cp_dec.dt_sec = 0.05;
  cp_dec.prev_speed_cmd_mps = 0.62;
  cp_dec.odom_valid = false;
  const AdaptiveSpeedResult r_dec = ctrl.compute_adaptive_corridor_command(
    0.0, 0.0, 0.0, 0.0, 0.95, 0.22, curve, motion, cp_dec);
  const double delta_dn = 0.62 - r_dec.command.linear_speed;
  CHECK(std::abs(delta_up - cfg.max_accel_mps2 * 0.05) < 1e-6);
  CHECK(delta_dn > delta_up * 2.0);

  // 5. Backward compatibility when enable_clear_path_speed = false
  MotionControllerConfig cfg_off = cfg;
  cfg_off.enable_clear_path_speed = false;
  MotionController ctrl_off(cfg_off);
  const AdaptiveSpeedResult r_off = ctrl_off.compute_adaptive_corridor_command(
    0.0, 0.0, 0.0, 0.0, 0.95, 0.85, curve, motion, cp_acc);
  CHECK(std::abs(r_off.clear_path_score) < 1e-9);
  CHECK(std::abs(r_off.command.linear_speed - cfg_off.target_speed) < 1e-6);
}

void test_corridor_ackermann_curve_offtracking_compensation()
{
  CorridorPlannerConfig cfg{};
  cfg.car_width_m = 0.191;
  cfg.default_lane_width_m = 0.430;
  cfg.border_margin_m = 0.025;
  CorridorPlanner planner(cfg);

  // Deadband on straights (|hdg| <= 0.15) -> zero shift
  CHECK(std::abs(CorridorPlanner::compute_curve_offtracking_offset(0.0)) < 1e-12);
  CHECK(std::abs(CorridorPlanner::compute_curve_offtracking_offset(+0.15)) < 1e-12);
  CHECK(std::abs(CorridorPlanner::compute_curve_offtracking_offset(-0.15)) < 1e-12);

  // Sharp right curve (+0.55) shifts target OUTSIDE (LEFT, +y) inside safe corridor,
  // bounded by Ackermann wheelbase off-tracking (<= 0.05m) so yaw authority is preserved.
  const CorridorPlan right_turn = planner.plan(
    +0.215, -0.215, 0.0, {}, {}, false, 1.0, 1.0, 0.55, +0.55);
  CHECK(right_turn.target_y_m > 0.030 && right_turn.target_y_m <= 0.050);
  CHECK(right_turn.target_y_m <= right_turn.safe_y_max_m);

  // Mirror: sharp left curve (-0.55) shifts target OUTSIDE (RIGHT, -y) inside safe corridor
  const CorridorPlan left_turn = planner.plan(
    +0.215, -0.215, 0.0, {}, {}, false, 1.0, 1.1, 0.55, -0.55);
  CHECK(left_turn.target_y_m < -0.030 && left_turn.target_y_m >= -0.050);
  CHECK(left_turn.target_y_m >= left_turn.safe_y_min_m);
  CHECK(std::abs(right_turn.target_y_m + left_turn.target_y_m) < 1e-9);

  // Never pushes into an obstacle on the outside of the curve!
  LocalObstacle outer_cone{};
  outer_cone.type = SemanticObjectType::RED_CONE;
  outer_cone.d_bumper_m = 0.45;
  outer_cone.d_min_m = 0.38;
  outer_cone.d_max_m = 0.52;
  outer_cone.y_lat_m = +0.14;
  outer_cone.y_min_m = +0.0725;
  outer_cone.y_max_m = +0.2075;
  const CorridorPlan right_with_obs = planner.plan(
    +0.215, -0.215, 0.0, {outer_cone}, {}, false, 1.0, 1.2, 0.55, +0.55);
  CHECK(right_with_obs.target_y_m + 0.5 * cfg.car_width_m < outer_cone.y_min_m);
}

void test_bev_dwa_first_principles_and_execution()
{
  using line_follower::BEV_FREE;
  using line_follower::BEV_OBS_CONE;
  using line_follower::BEV_OBS_NO_ENTRY;
  using line_follower::BEV_WALL_BLUE;
  using line_follower::BevDwaConfig;
  using line_follower::BevDwaPlanner;
  using line_follower::BevDwaPlanResult;
  using line_follower::TopologyPhase;
  using line_follower::kBevCells;
  using line_follower::kBevCols;
  using line_follower::kBevRows;

  auto make_straight_grid = [](double half_w_m) {
    std::array<uint8_t, kBevCells> g{};
    g.fill(BEV_WALL_BLUE);
    for (int r = 0; r < kBevRows; ++r) {
      for (int c = 0; c < kBevCols; ++c) {
        double d = 0.0, y = 0.0;
        BevDwaPlanner::cell_to_metric(r, c, d, y);
        if (d >= -0.30 && d <= 1.20 && std::abs(y) <= half_w_m) {
          g[static_cast<size_t>(r * kBevCols + c)] = BEV_FREE;
        }
      }
    }
    return g;
  };

  // 1) Pre-flight self-check: gyro bias calibration + homography width alarm
  {
    BevDwaConfig cfg{};
    BevDwaPlanner planner(cfg);
    auto good_grid = make_straight_grid(0.215);  // 430mm exact
    VehiclePose2D idle_pose{true, 0.0, 0.0, 0.0, 0.0, +0.025, 0.0};
    for (int i = 0; i < 20; ++i) {
      planner.update_standstill_check(good_grid, idle_pose);
    }
    CHECK(planner.preflight_status().checked);
    CHECK(planner.preflight_status().homography_ok);
    CHECK(planner.preflight_status().gyro_calibrated);
    CHECK(std::abs(planner.preflight_status().gyro_bias_rps - 0.025) < 1e-4);

    // Tilted camera producing narrow 0.30m apparent width must trigger warning!
    auto bad_grid = make_straight_grid(0.15);
    planner.update_standstill_check(bad_grid, idle_pose);
    CHECK(!planner.preflight_status().homography_ok);
    CHECK(!planner.preflight_status().warning_msg.empty());
  }

  // 2) Fork S T-junction anti-wrong-way test:
  //    Arriving North (-90 deg) up the right shortcut, even with a red no-entry strip
  //    on the top-right shoulder (exact fork F195-F203 scenario!), MUST turn RIGHT (kappa < 0)
  //    into East-bound S-curve and NEVER turn LEFT backwards!
  {
    BevDwaConfig cfg{};
    cfg.track_mode = "fork";
    cfg.kappa_max_left = 2.38;
    cfg.kappa_max_right = 2.80;
    BevDwaPlanner planner(cfg);
    VehiclePose2D p0{true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    planner.reset(0.0, p0);

    // Vote single-side right=8 (left missing -> E=4.5 fallback -> chooses RIGHT +1!)
    GroundDetection d8{SemanticObjectType::BONUS_DIGIT, 8, 0.90, 0.55, -0.10, 0.04, 0.04, 1, 1.0};
    planner.update_fork_votes({d8}, true);
    planner.update_fork_votes({d8}, true);
    const int chosen = planner.finalize_fork_decision();
    CHECK(chosen == +1);
    CHECK(planner.phase() == TopologyPhase::FORK_RIGHT_ENTER);

    // Rotate cleanly to North (-pi/2) along straight vertical shortcut
    auto straight = make_straight_grid(0.215);
    constexpr double kHalfPi = 1.5707963267948966;
    for (int k = 1; k <= 10; ++k) {
      VehiclePose2D pk{
        true, 0.05 * k, -0.05 * k, -kHalfPi * (static_cast<double>(k) / 10.0), 0.25, -0.5, 0.1 * k};
      planner.step(straight, pk, {}, false, false, 0.0, 1e9, false, 0.1 * k);
    }
    CHECK(planner.phase() == TopologyPhase::FORK_RIGHT_STEM);

    // Build symmetric T-junction grid: vertical stem d in [-0.30, 0.45], horizontal bar d in [0.15, 0.58], y in [-0.78, +0.78]
    // PLUS red no-entry patch on top-right corner (d in [0.46, 0.56], y in [-0.35, -0.18])!
    std::array<uint8_t, kBevCells> t_grid{};
    t_grid.fill(BEV_WALL_BLUE);
    for (int r = 0; r < kBevRows; ++r) {
      for (int c = 0; c < kBevCols; ++c) {
        double d = 0.0, y = 0.0;
        BevDwaPlanner::cell_to_metric(r, c, d, y);
        const bool in_stem = (d >= -0.30 && d <= 0.45 && std::abs(y) <= 0.215);
        const bool in_bar = (d >= 0.15 && d <= 0.58 && std::abs(y) <= 0.78);
        if (in_stem || in_bar) {
          t_grid[static_cast<size_t>(r * kBevCols + c)] = BEV_FREE;
        }
        if (d >= 0.48 && d <= 0.57 && y >= -0.36 && y <= -0.18) {
          t_grid[static_cast<size_t>(r * kBevCols + c)] = BEV_OBS_NO_ENTRY;
        }
      }
    }

    VehiclePose2D p_t{true, 0.55, -0.80, -kHalfPi, 0.22, 0.0, 1.5};
    const BevDwaPlanResult t_res = planner.step(t_grid, p_t, {}, false, false, 0.0, 1e9, false, 1.5);
    CHECK(t_res.phase == TopologyPhase::FORK_T_RIGHT_TURN);
    CHECK(t_res.chosen_kappa < -0.8);   // Strongly turns RIGHT (East), NEVER turns LEFT backwards!
    CHECK(t_res.yaw_rate_rps < -0.15);
  }

  // 3) Roundabout cross-island rejection: disconnected white road across blue island is wiped!
  {
    BevDwaConfig cfg{};
    cfg.track_mode = "roundabout";
    BevDwaPlanner planner(cfg);
    planner.reset(0.0, VehiclePose2D{true, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0});

    std::array<uint8_t, kBevCells> rb_grid{};
    rb_grid.fill(BEV_WALL_BLUE);
    for (int r = 0; r < kBevRows; ++r) {
      for (int c = 0; c < kBevCols; ++c) {
        double d = 0.0, y = 0.0;
        BevDwaPlanner::cell_to_metric(r, c, d, y);
        // Real road curves LEFT: y_center = +0.5 * d^2
        const double y_road = 0.55 * std::max(0.0, d) * std::max(0.0, d);
        if (d >= -0.30 && d <= 0.90 && std::abs(y - y_road) <= 0.215) {
          rb_grid[static_cast<size_t>(r * kBevCols + c)] = BEV_FREE;
        }
        // Disconnected opposite side of roundabout across blue island on the RIGHT (y in [-0.75, -0.45])
        if (d >= 0.45 && d <= 1.00 && y >= -0.75 && y <= -0.45) {
          rb_grid[static_cast<size_t>(r * kBevCols + c)] = BEV_FREE;
        }
      }
    }
    const BevDwaPlanResult res = planner.step(
      rb_grid, VehiclePose2D{true, 0.05, 0.0, 0.0, 0.25, 0.0, 0.1}, {}, false, false, 0.0, 1e9, false, 0.1);
    // Disconnected right island blob MUST be converted to WALL_BLUE and car must turn LEFT!
    int rr = 0, rc = 0;
    CHECK(BevDwaPlanner::metric_to_cell(0.70, -0.60, rr, rc));
    CHECK(planner.fused_grid()[static_cast<size_t>(rr * kBevCols + rc)] == BEV_WALL_BLUE);
    CHECK(res.chosen_kappa > 0.2);
  }

  // 4) Asymmetric penalty economics (+30s no_entry > +10s cone) + non-freezing tight gap
  {
    BevDwaConfig cfg{};
    BevDwaPlanner planner(cfg);
    planner.reset(0.0, VehiclePose2D{true, 0.0, 0.0, 0.0, 0.22, 0.0, 0.0});
    auto g = make_straight_grid(0.215);
    // Put +30s no-entry on left (y in [+0.10, +0.21]) and +10s cone on right (y in [-0.21, -0.10]) at d=0.45m
    for (int r = 0; r < kBevRows; ++r) {
      for (int c = 0; c < kBevCols; ++c) {
        double d = 0.0, y = 0.0;
        BevDwaPlanner::cell_to_metric(r, c, d, y);
        if (d >= 0.38 && d <= 0.52 && y >= 0.105 && y <= 0.21) {
          g[static_cast<size_t>(r * kBevCols + c)] = BEV_OBS_NO_ENTRY;
        }
        if (d >= 0.38 && d <= 0.52 && y >= -0.21 && y <= -0.105) {
          g[static_cast<size_t>(r * kBevCols + c)] = BEV_OBS_CONE;
        }
      }
    }
    const BevDwaPlanResult res = planner.step(
      g, VehiclePose2D{true, 0.0, 0.0, 0.0, 0.22, 0.0, 0.1}, {}, false, false, 0.0, 1e9, false, 0.1);
    CHECK(res.linear_speed_mps > 0.05);  // Never freezes!
    CHECK(res.chosen_kappa <= 0.05);     // Prefers shifting away from +30s no-entry (rightward)!
  }

  // 5) Zero-yellow-count finish immunity + dual visual/distance yellow rearm + deadzone brake floor
  {
    BevDwaConfig cfg{};
    cfg.finish_min_cum_yaw_rad = 2.2;
    cfg.yellow_rearm_dist_m = 0.25;
    BevDwaPlanner planner(cfg);
    planner.reset(0.0, VehiclePose2D{true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    auto g = make_straight_grid(0.215);

    // Trigger yellow stop -> must NOT be rearmed immediately even after seeing yellow disappear once
    planner.notify_yellow_stop_triggered();
    CHECK(!planner.is_yellow_rearmed());
    // Still seeing yellow at d=0.08m after moving 0.30m -> still NOT rearmed until yellow exits!
    planner.step(g, VehiclePose2D{true, 0.30, 0.0, 0.0, 0.20, 0.0, 1.0}, {}, true, true, 0.08, 1e9, false, 1.0);
    CHECK(!planner.is_yellow_rearmed());
    // After 3 frames with yellow gone + dist >= 0.25m -> cleanly rearmed!
    for (int i = 1; i <= 3; ++i) {
      planner.step(g, VehiclePose2D{true, 0.30 + 0.02 * i, 0.0, 0.0, 0.20, 0.0, 1.0 + 0.05 * i},
        {}, false, false, 0.0, 1e9, false, 1.0 + 0.05 * i);
    }
    CHECK(planner.is_yellow_rearmed());

    // Deadzone-clearing FinishGuard closed-loop brake floor (brake_min_cmd_mps = 0.20)
    FinishGuardConfig fg{};
    fg.brake_damping_gain = 0.85;
    fg.brake_min_cmd_mps = 0.20;
    fg.brake_zero_cutoff_mps = 0.04;
    FinishGuard guard(fg);
    // At v=0.10 m/s, proportional alone would be -0.085 (deadzone!); floor lifts it to -0.20!
    CHECK(std::abs(guard.compute_closed_loop_brake_cmd(0.10, 0.05, 0.24, 0.28) - (-0.20)) < 1e-6);
    // At v=0.03 <= 0.04 cutoff -> immediately cuts to 0.0 (zero rollback)!
    CHECK(std::abs(guard.compute_closed_loop_brake_cmd(0.03, 0.08, 0.24, 0.28)) < 1e-9);

    // Tiny enclosed blue-tinted digit speck (6x6cm = 36 cells) inside white road MUST be healed to FREE!
    auto g_speck = make_straight_grid(0.215);
    for (int r = 0; r < kBevRows; ++r) {
      for (int c = 0; c < kBevCols; ++c) {
        double d = 0.0, y = 0.0;
        BevDwaPlanner::cell_to_metric(r, c, d, y);
        if (d >= 0.30 && d <= 0.36 && y >= -0.03 && y <= 0.03) {
          g_speck[static_cast<size_t>(r * kBevCols + c)] = BEV_WALL_BLUE;
        }
      }
    }
    const BevDwaPlanResult speck_res = planner.step(
      g_speck, VehiclePose2D{true, 0.40, 0.0, 0.0, 0.30, 0.0, 2.0}, {}, false, false, 0.0, 1e9, false, 2.0);
    CHECK(speck_res.chosen_safe_len_m >= 0.70);
    int sr = 0, sc = 0;
    CHECK(BevDwaPlanner::metric_to_cell(0.33, 0.0, sr, sc));
    CHECK(planner.fused_grid()[static_cast<size_t>(sr * kBevCols + sc)] == BEV_FREE);
  }

  // 6) Real-car regression tests (from 2026-11-28 fork_151853 & rb_152314):
  //    a) Half-car deep in right blue wall MUST steer LEFT back onto white road (NEVER tie-lock k=-2.80!)
  //    b) Single-frame YOLO digit flicker (6+5 -> 5 -> 6+5) MUST NOT cause 60ms left-right twitching!
  //    c) Fork right branch (+1) with car angled +14 deg left MUST still steer RIGHT (not overruled by soft wall)!
  //    d) Early wall-graze 33cm past fork yellow MUST NOT prematurely jump to FORK_T_RIGHT_TURN!
  {
    BevDwaConfig cfg{};
    cfg.track_mode = "roundabout";
    BevDwaPlanner planner(cfg);
    planner.reset(0.0, VehiclePose2D{true, 0.0, 0.0, 0.0, 0.20, 0.0, 0.0});

    // (a) White road is shifted to the LEFT (y in [+0.05, +0.48]), car is deep in right blue wall (y=0)!
    std::array<uint8_t, kBevCells> off_right{};
    off_right.fill(BEV_WALL_BLUE);
    for (int r = 0; r < kBevRows; ++r) {
      for (int c = 0; c < kBevCols; ++c) {
        double d = 0.0, y = 0.0;
        BevDwaPlanner::cell_to_metric(r, c, d, y);
        if (d >= -0.30 && d <= 1.20 && y >= 0.05 && y <= 0.48) {
          off_right[static_cast<size_t>(r * kBevCols + c)] = BEV_FREE;
        }
      }
    }
    const BevDwaPlanResult rec = planner.step(
      off_right, VehiclePose2D{true, 0.02, 0.0, 0.0, 0.14, 0.0, 0.1}, {}, false, false, 0.0, 1e9, false, 0.1);
    CHECK(rec.chosen_kappa > 0.5);  // Strongly steers LEFT back to white road! NEVER -2.80!

    // (b) Digit flicker immunity on straight road
    planner.reset(0.0, VehiclePose2D{true, 0.0, 0.0, 0.0, 0.40, 0.0, 0.0});
    auto straight = make_straight_grid(0.215);
    GroundDetection d6{SemanticObjectType::BONUS_DIGIT, 6, 0.90, 0.55, +0.09, 0.04, 0.04, 1, 0.1};
    GroundDetection d5{SemanticObjectType::BONUS_DIGIT, 5, 0.90, 0.55, -0.09, 0.04, 0.04, 2, 0.1};
    // Lock onto digit 6 over 4 frames
    for (int f = 1; f <= 4; ++f) {
      planner.step(straight, VehiclePose2D{true, 0.01 * f, 0.0, 0.0, 0.40, 0.0, 0.03 * f},
        {d6, d5}, false, false, 0.0, 1e9, false, 0.03 * f);
    }
    // Flicker frame: digit 6 drops out for 2 frames, only right digit 5 remains -> MUST NOT swing right!
    const BevDwaPlanResult flk = planner.step(
      straight, VehiclePose2D{true, 0.05, 0.0, 0.0, 0.40, 0.0, 0.15},
      {d5}, false, false, 0.0, 1e9, false, 0.15);
    CHECK(flk.chosen_kappa >= -0.05);  // Holds leftward/straight lock, ZERO 60ms right snap!

    // (d) Fork premature T-junction anti-jump guard: 0.35m past fork with L=0 graze MUST stay in FORK_RIGHT_ENTER
    BevDwaConfig fcfg{};
    fcfg.track_mode = "fork";
    BevDwaPlanner fplanner(fcfg);
    fplanner.reset(0.0, VehiclePose2D{true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    GroundDetection fr5{SemanticObjectType::BONUS_DIGIT, 5, 0.90, 0.55, -0.10, 0.04, 0.04, 1, 1.0};
    fplanner.update_fork_votes({fr5}, true);
    fplanner.update_fork_votes({fr5}, true);
    CHECK(fplanner.finalize_fork_decision() == +1);
    CHECK(fplanner.phase() == TopologyPhase::FORK_RIGHT_ENTER);
    // Move only 0.33m turning to -53 deg while grazing wall -> must NOT jump to FORK_T_RIGHT_TURN!
    for (int step_i = 1; step_i <= 6; ++step_i) {
      VehiclePose2D pp{true, 0.055 * step_i, -0.02 * step_i, -0.155 * step_i, 0.14, -0.5, 0.1 * step_i};
      fplanner.step(off_right, pp, {}, false, false, 0.0, 1e9, false, 0.1 * step_i);
    }
    CHECK(fplanner.phase() != TopologyPhase::FORK_T_RIGHT_TURN);
  }
}

}  // namespace

int main()
{
  test_ground_projector_pinhole_and_homography();
  test_obstacle_memory_rear_wheel_retention_and_merge();
  test_corridor_planner_tight_obstacles_and_rear_cutin();
  test_corridor_planner_fork_voting_and_straight_digits();
  test_finish_guard_random_yellow_rejection_and_two_stage_stop();
  test_track_state_machine_odom_freshness_and_finals_mode();
  test_motion_controller_corridor_and_safe_lost();
  test_motion_controller_edge_heading_guard_and_curve_preservation();
  test_fork_signed_odom_and_dual_digit_no_late_swerve();
  test_red_obstacle_or_fusion_and_attribution();
  test_corridor_two_zone_staggered_and_curve_horizon();
  test_b3_rear_axle_rotation_pivot();
  test_b7_reverse_motion_never_adds_arc_length();
  test_b9_same_digit_vector_tracks_and_horizon_clamping();
  test_tier1_obb_yaw_and_ackermann_cbf_shield();
  test_tier1_finish_guard_ramp_voltage_and_adaptation();
  test_tier1_corridor_risk_aware_fork_and_straight_digits();
  test_dual_layer_brake_and_safe_speed_ceiling();
  test_clear_path_adaptive_speed_and_asymmetric_slew();
  test_corridor_ackermann_curve_offtracking_compensation();
  test_bev_dwa_first_principles_and_execution();
  std::cout << "ALL_FINALS_CPP_TESTS_PASSED" << std::endl;
  return 0;
}
