#include "line_follower/corridor_planner.hpp"
#include "line_follower/finals_types.hpp"
#include "line_follower/finish_guard.hpp"
#include "line_follower/ground_projector.hpp"
#include "line_follower/motion_controller.hpp"
#include "line_follower/obstacle_memory.hpp"
#include "line_follower/track_state_machine.hpp"

#include <cmath>
#include <iostream>
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
  test_red_obstacle_or_fusion_and_attribution();
  test_corridor_two_zone_staggered_and_curve_horizon();
  test_b3_rear_axle_rotation_pivot();
  test_b7_reverse_motion_never_adds_arc_length();
  test_b9_same_digit_vector_tracks_and_horizon_clamping();
  std::cout << "ALL_FINALS_CPP_TESTS_PASSED" << std::endl;
  return 0;
}
