#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive verification suite for the Windows PC Simulator stack.

Covers:
  1. C++17 Brain DLL build & BrainBridge parity (cruise, curve, anti-reverse yellow
     brake, CV red obstacle avoidance + CBF shield, fork voting, dual-digit lock,
     finish guard).
  2. VisionFrontend 1:1 OpenCV mirror (lane seg modes on fork_s & roundabout,
     geometric yellow & dead-end, CV red cone vs hollow no-entry & off-track
     rejection) + MockNoisyDetector + RealYoloDetector (RTX 5060 CUDA / CPU).
  3. PCSimWorld & AICReferee (3D standing cone rendering + white collar, reverse-brake
     overshoot vs clean stop, latency & slippery scaling, 1-2 wheel boundary press
     +15s/s, >=3 wheel DNF, no-entry +30s, cone +10s, yellow clean stop vs +15s
     violation, bonus digit -5s*v, net_score = lap + penalty - bonus, top-down render).
  4. End-to-end closed-loop runner (pc_sim.py), dual-pane 1280x480 composition,
     regression scenario definitions, and auto_tune_pc safety floors & DR suite.

Runnable via:
    python -m unittest tools/tests/test_pc_sim.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Guard against tools/sim/profile.py shadowing Python stdlib `profile` (needed for torch/ultralytics)
REPO_ROOT = Path(__file__).resolve().parents[2]
_SIM_DIR = REPO_ROOT / "tools" / "sim"
_sim_dir_norm = os.path.normcase(str(_SIM_DIR))
sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or ".")) != _sim_dir_norm]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tools.sim import auto_tune_pc  # noqa: E402
from tools.sim import build_brain_dll  # noqa: E402
from tools.sim import pc_sim  # noqa: E402
from tools.sim import regression_suite  # noqa: E402
from tools.sim.brain_bridge import BrainBridge  # noqa: E402
from tools.sim.sim_world_pc import (  # noqa: E402
    AICReferee,
    FinalsTrackRenderer,
    PCSimWorld,
    VehicleDynamics,
    resolve_track_json,
)
from tools.sim.vision_mirror import (  # noqa: E402
    DEFAULT_YOLO_WEIGHTS,
    MockNoisyDetector,
    RealYoloDetector,
    VisionFrontend,
)

FORK_YAML = REPO_ROOT / "tools" / "config" / "finals_fork.yaml"
ROUNDABOUT_YAML = REPO_ROOT / "tools" / "config" / "finals_roundabout.yaml"


def _base_obs(t_s: float = 0.0, **overrides) -> dict:
    obs = {
        "image_width": 640,
        "image_height": 480,
        "lane_valid": True,
        "near_valid": True,
        "far_valid": True,
        "near_error": 0.0,
        "far_error": 0.0,
        "heading_error": 0.0,
        "confidence": 0.95,
        "near_width_ratio": 0.65,
        "far_width_ratio": 0.45,
        "left_edge_norm": -0.90,
        "right_edge_norm": 0.90,
        "has_left_edge": True,
        "has_right_edge": True,
        "fork_detected": False,
        "left_branch_error": 0.0,
        "right_branch_error": 0.0,
        "yellow_detected": False,
        "yellow_bottom_y": 0.0,
        "yellow_width_ratio": 0.0,
        "yellow_dead_end": False,
        "cv_red_boxes": [],
        "detections": [],
        "odom_x": 0.0,
        "odom_y": 0.0,
        "odom_yaw": 0.0,
        "odom_v": 0.40,
        "odom_w": 0.0,
        "voltage": 12.0,
        "timestamp_sec": t_s,
    }
    obs.update(overrides)
    return obs


class TestCppBrainBridge(unittest.TestCase):
    """Verify build_brain_dll.py and C++17 BrainBridge edge-case parity."""

    def test_dll_sources_and_load_both_configs(self):
        sources = build_brain_dll.get_source_files()
        names = {p.name for p in sources}
        expected = {
            "cpp_brain_bridge.cpp",
            "ground_projector.cpp",
            "obstacle_memory.cpp",
            "ackermann_cbf_shield.cpp",
            "corridor_planner.cpp",
            "finish_guard.cpp",
            "track_state_machine.cpp",
            "motion_controller.cpp",
        }
        self.assertEqual(names, expected)
        dll_path = build_brain_dll.build_dll(force=False, verbose=False)
        self.assertTrue(dll_path.is_file())

        for cfg in (FORK_YAML, ROUNDABOUT_YAML):
            with BrainBridge(cfg, param_overrides={"target_speed": 0.42}) as brain:
                out = brain.step(_base_obs(t_s=0.1))
                self.assertEqual(out["state_name"], "tracking")
                self.assertAlmostEqual(out["v_cmd"], 0.42, places=2)
                self.assertAlmostEqual(out["w_cmd"], 0.0, places=2)
                self.assertIn("corridor_left_m", out)
                self.assertIn("corridor_right_m", out)
                self.assertIn("yellow_phase", out)

    def test_yellow_stop_anti_reverse_guard(self):
        with BrainBridge(
            FORK_YAML,
            param_overrides={
                "yellow_brake_speed": 0.25,
                "yellow_brake_release_speed": 0.08,
                "yellow_bumper_margin_m": 0.08,
            },
        ) as brain:
            # Warmup cruise
            brain.step(_base_obs(t_s=0.0, odom_x=0.0, odom_v=0.35))
            brain.step(_base_obs(t_s=0.05, odom_x=0.0175, odom_v=0.35))

            # Close yellow line right in front of bumper (v=440 px -> very small d_bumper_m)
            out_brake = brain.step(
                _base_obs(
                    t_s=0.10,
                    odom_x=0.035,
                    odom_v=0.35,
                    yellow_detected=True,
                    yellow_bottom_y=440.0,
                    yellow_width_ratio=0.70,
                )
            )
            self.assertEqual(out_brake["state_name"], "yellow_line_detected")
            self.assertEqual(out_brake["yellow_phase"], "braking")
            self.assertLess(out_brake["v_cmd"], 0.0)

            # Signed velocity drops to <= 0.01 m/s -> anti-reverse guard MUST cut brake to 0.0!
            out_guard = brain.step(
                _base_obs(
                    t_s=0.15,
                    odom_x=0.040,
                    odom_v=0.005,
                    yellow_detected=True,
                    yellow_bottom_y=445.0,
                    yellow_width_ratio=0.70,
                )
            )
            self.assertEqual(out_guard["yellow_phase"], "stopped")
            self.assertEqual(out_guard["v_cmd"], 0.0)
            self.assertTrue(out_guard["publish_stop"])

            # Negative odom velocity also stays locked at 0.0 (never drives further backward)
            out_neg = brain.step(
                _base_obs(
                    t_s=0.20,
                    odom_x=0.039,
                    odom_v=-0.02,
                    yellow_detected=True,
                    yellow_bottom_y=445.0,
                    yellow_width_ratio=0.70,
                )
            )
            self.assertEqual(out_neg["v_cmd"], 0.0)
            self.assertTrue(out_neg["publish_stop"])

    def test_obstacle_avoidance_and_cbf_shield(self):
        with BrainBridge(
            FORK_YAML,
            param_overrides={
                "enable_cbf_shield": True,
                "enable_cv_red_fallback": True,
                "target_speed": 0.44,
            },
        ) as brain:
            # Place a red cone on the right side of the lane (u ~ 390, v_bot ~ 360 -> d ~ 0.4m, y < 0)
            cv_box = {
                "x": 365.0,
                "y": 300.0,
                "w": 55.0,
                "h": 65.0,
                "is_no_entry": False,
                "confidence": 0.85,
            }
            out = brain.step(
                _base_obs(
                    t_s=0.10,
                    odom_x=0.05,
                    odom_v=0.40,
                    cv_red_boxes=[cv_box],
                )
            )
            self.assertGreaterEqual(out["obstacle_count"], 1)
            # Obstacle on right (y < 0) must push corridor target to left (y > 0)
            self.assertGreater(out["corridor_target_y_m"], 0.01)
            # Speed capped for obstacle avoidance
            self.assertLess(out["speed_cap_mps"], 0.44)

            # Very close obstacle directly ahead triggers CBF intervention
            close_box = {
                "x": 295.0,
                "y": 325.0,
                "w": 65.0,
                "h": 85.0,
                "is_no_entry": False,
                "confidence": 0.90,
            }
            out_cbf = brain.step(
                _base_obs(
                    t_s=0.15,
                    odom_x=0.07,
                    odom_v=0.40,
                    cv_red_boxes=[close_box],
                )
            )
            self.assertTrue(out_cbf["cbf_active"])
            self.assertNotEqual(out_cbf["cbf_limiting_source"], "none")

    def test_fork_static_vote_and_dual_digit_lock(self):
        # 1) Fork choice: static votes during yellow stop -> digit 7 left vs digit 2 right -> branch = -1
        with BrainBridge(
            FORK_YAML,
            param_overrides={
                "yellow_stop_duration": 0.25,
                "yellow_stop_min_time": 0.10,
                "yellow_require_odom_still": False,
                "yellow_clear_frames": 2,
                "fork_min_votes": 2,
                "fork_outer_branch": -1,
                "fork_outer_time_penalty_s": 3.0,
            },
        ) as brain:
            t = 0.0
            brain.step(_base_obs(t_s=t, odom_x=0.0, odom_v=0.30))
            dets_fork = [
                {"class_name": "7", "conf": 0.92, "x": 160.0, "y": 240.0, "w": 36.0, "h": 42.0, "track_id": 1},
                {"class_name": "2", "conf": 0.90, "x": 450.0, "y": 240.0, "w": 36.0, "h": 42.0, "track_id": 2},
            ]
            # Trigger yellow stop and accumulate votes while stopped
            for _ in range(10):
                t += 0.05
                brain.step(
                    _base_obs(
                        t_s=t,
                        odom_x=0.05,
                        odom_v=0.0,
                        yellow_detected=True,
                        yellow_bottom_y=440.0,
                        yellow_width_ratio=0.70,
                        detections=dets_fork,
                    )
                )
            # Clear yellow line to end stop & finalize fork decision
            out = {}
            for _ in range(5):
                t += 0.05
                out = brain.step(
                    _base_obs(
                        t_s=t,
                        odom_x=0.06,
                        odom_v=0.0,
                        yellow_detected=False,
                    )
                )
            # 7 * 5s - 3s outer penalty = 32s > 2 * 5s = 10s -> prefers left branch (-1)
            self.assertEqual(out["preferred_branch"], -1)
            self.assertEqual(brain.preferred_branch, -1)

        # 2) Straight dual digit lock: digits 5 and 2 (at ~0.45m ahead, well inside lookahead) -> locks onto 5
        with BrainBridge(FORK_YAML, param_overrides={"digit_min_frames": 2}) as brain:
            dets_dual = [
                {"class_name": "5", "conf": 0.94, "x": 245.0, "y": 235.0, "w": 34.0, "h": 34.0, "track_id": 10},
                {"class_name": "2", "conf": 0.92, "x": 365.0, "y": 235.0, "w": 34.0, "h": 34.0, "track_id": 11},
            ]
            out = {}
            for i in range(4):
                out = brain.step(_base_obs(t_s=i * 0.05, odom_x=i * 0.02, odom_v=0.35, detections=dets_dual))
            self.assertTrue(out["digit_target_active"])
            self.assertEqual(out["active_digit"], 5)

    def test_finish_guard_permanent_stop(self):
        with BrainBridge(
            FORK_YAML,
            param_overrides={
                "finish_min_dist_m": 2.0,
                "finish_force_dist_m": 3.0,
                "finish_min_cum_yaw_rad": 0.0,
            },
        ) as brain:
            # Advance forward odom past finish_min_dist_m (2.0 m)
            t = 0.0
            x = 0.0
            out = {}
            for _ in range(50):
                out = brain.step(_base_obs(t_s=t, odom_x=x, odom_v=0.50))
                t += 0.10
                x += 0.05
            self.assertTrue(out["finish_armed"])
            self.assertGreaterEqual(out["total_arc_m"], 2.0)

            # Encounter finish yellow line with dead_end=True close to bumper
            out_fin = brain.step(
                _base_obs(
                    t_s=t,
                    odom_x=x,
                    odom_v=0.0,
                    yellow_detected=True,
                    yellow_bottom_y=440.0,
                    yellow_width_ratio=0.75,
                    yellow_dead_end=True,
                )
            )
            self.assertTrue(out_fin["permanent_stop"])
            self.assertEqual(out_fin["yellow_phase"], "finished")
            self.assertEqual(out_fin["v_cmd"], 0.0)


class TestVisionMirrorAndDetectors(unittest.TestCase):
    """Verify VisionFrontend 1:1 mirror, MockNoisyDetector, and RealYoloDetector."""

    def test_vision_frontend_lane_all_seg_modes_both_tracks(self):
        for track, cfg in (("fork_s", FORK_YAML), ("roundabout", ROUNDABOUT_YAML)):
            world = PCSimWorld(track=track, include_base_yellow=True)
            info = world.reset(pose={"s_m": 0.15, "lat_mm": 0.0})
            for mode in ("relative_border", "hybrid", "legacy_hsv"):
                vf = VisionFrontend(cfg)
                vf.update_params({"lane_seg_mode": mode})
                obs, dbg = vf.process(info.bgr_img)
                self.assertTrue(obs["lane_valid"], f"{track}/{mode} lane_valid=False")
                self.assertGreater(obs["confidence"], 0.5)
                self.assertLess(abs(obs["near_error"]), 0.35)
                self.assertGreater(obs["y_lane_left_m"], obs["y_lane_right_m"])
                self.assertIn("binary_mask", dbg)

    def test_vision_frontend_yellow_and_cv_red_classification(self):
        # 1) Yellow stop line ahead on fork_s (base yellow #1 at s ~ 0.774m)
        world = PCSimWorld(track="fork_s", include_base_yellow=True)
        info = world.reset(pose={"s_m": 0.25, "lat_mm": 0.0})
        vf = VisionFrontend(FORK_YAML)
        obs_y, _ = vf.process(info.bgr_img)
        self.assertTrue(obs_y["yellow_detected"])
        self.assertGreater(obs_y["yellow_bottom_y"], 150.0)
        self.assertGreater(obs_y["yellow_width_ratio"], 0.25)

        # 2) CV red cone vs hollow red no-entry + off-track red rejection
        world_red = PCSimWorld(track="fork_s", include_base_yellow=False)
        info_cone = world_red.reset(
            pose={"s_m": 1.00, "lat_mm": 0.0},
            props=[
                {"type": "red_cone", "s_m": 1.60, "lat_mm": 70.0, "id": "c1"},
                {"type": "red_cone", "s_m": 1.60, "lat_mm": 420.0, "id": "c_offtrack"},
            ],
        )
        vf.reset()
        obs_c, _ = vf.process(info_cone.bgr_img)
        self.assertGreaterEqual(len(obs_c["cv_red_boxes"]), 1)
        # Only the in-lane cone should be accepted; off-track red cone is rejected
        self.assertEqual(len(obs_c["cv_red_boxes"]), 1)
        self.assertFalse(obs_c["cv_red_boxes"][0]["is_no_entry"])

        info_ne = world_red.reset(
            pose={"s_m": 1.00, "lat_mm": 0.0},
            props=[{"type": "red_no_entry", "s_m": 1.65, "lat_mm": -40.0, "id": "ne1"}],
        )
        vf.reset()
        obs_ne, _ = vf.process(info_ne.bgr_img)
        self.assertGreaterEqual(len(obs_ne["cv_red_boxes"]), 1)
        self.assertTrue(obs_ne["cv_red_boxes"][0]["is_no_entry"])

    def test_mock_noisy_detector(self):
        world = PCSimWorld(track="fork_s", include_base_yellow=False)
        info = world.reset(
            pose={"s_m": 1.00, "lat_mm": 0.0},
            props=[
                {"type": "digit", "s_m": 1.46, "lat_mm": 75.0, "value": 5, "id": "d5"},
                {"type": "digit", "s_m": 1.48, "lat_mm": -75.0, "value": 2, "id": "d2"},
            ],
        )
        det = MockNoisyDetector(seed=123, flicker_prob=0.0, base_recall=1.0)
        vf = VisionFrontend(FORK_YAML, detector=det)
        gt = info.gt_pose
        obs, _ = vf.process(
            info.bgr_img,
            world_context={
                "pose": (gt["cam_x_mm"], gt["cam_y_mm"], gt["cam_yaw"]),
                "world_props": info.world_props,
                "cam": world.cam,
            },
        )
        classes = {d["class_name"] for d in obs["detections"]}
        self.assertIn("5", classes)
        self.assertIn("2", classes)
        for d in obs["detections"]:
            self.assertGreater(d["w"], 5.0)
            self.assertGreater(d["h"], 5.0)
            self.assertGreaterEqual(d["conf"], 0.25)

    def test_real_yolo_detector_on_rendered_digits(self):
        self.assertTrue(
            DEFAULT_YOLO_WEIGHTS.is_file(),
            f"Expected trained YOLO weights at {DEFAULT_YOLO_WEIGHTS}",
        )
        yolo = RealYoloDetector(weights_path=DEFAULT_YOLO_WEIGHTS, conf_thresh=0.25)
        world = PCSimWorld(track="fork_s", include_base_yellow=False)
        for digit_val in (5, 7):
            info = world.reset(
                pose={"s_m": 1.00, "lat_mm": 0.0},
                props=[{"type": "digit", "s_m": 1.62, "lat_mm": 0.0, "value": digit_val, "id": f"yd{digit_val}"}],
            )
            dets = yolo.detect(info.bgr_img)
            matched = [d for d in dets if d["class_name"] == str(digit_val)]
            self.assertGreaterEqual(
                len(matched),
                1,
                f"RealYoloDetector missed rendered digit {digit_val}; got {dets}",
            )
            self.assertGreater(matched[0]["conf"], 0.45)


class TestWorldPhysicsAndAICReferee(unittest.TestCase):
    """Verify 3D cone rendering, vehicle reverse-brake physics, and official AIC scoring."""

    def test_3d_cone_rendering_height_and_white_collar(self):
        world = PCSimWorld(track="fork_s", include_base_yellow=False)
        info_empty = world.reset(pose={"s_m": 1.00, "lat_mm": 0.0}, props=[])
        info_cone = world.reset(
            pose={"s_m": 1.00, "lat_mm": 0.0},
            props=[{"type": "red_cone", "s_m": 1.60, "lat_mm": 0.0, "id": "cone3d"}],
        )
        diff = cv2.absdiff(info_cone.bgr_img, info_empty.bgr_img)
        changed = np.any(diff > 18, axis=2)
        ys, xs = np.where(changed)
        self.assertGreater(len(ys), 800, "3D cone should occupy substantial screen area")
        cone_h_px = int(ys.max() - ys.min())
        cone_w_px = int(xs.max() - xs.min())
        # Standing 150mm tall cone at ~480mm distance projects taller than a flat ground mark
        self.assertGreater(cone_h_px, 45)
        self.assertGreater(cone_w_px, 40)

        # Verify red cone body + white reflective collar inside changed bbox
        crop = info_cone.bgr_img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red_mask = ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168)) & (hsv[:, :, 1] > 100)
        # Upper-middle third of cone should contain white collar pixels (low saturation, high value)
        mid_crop = hsv[cone_h_px // 4: (3 * cone_h_px) // 4, :]
        white_collar = (mid_crop[:, :, 1] < 45) & (mid_crop[:, :, 2] > 175)
        self.assertGreater(int(red_mask.sum()), 400)
        self.assertGreater(int(white_collar.sum()), 60)

    def test_reverse_brake_overshoot_vs_clean_stop_and_latency(self):
        # 1) Holding negative v_cmd past zero reproduces reverse-drag bug
        dyn_rev = VehicleDynamics(a_eff=1.44, latency_ms=0.0)
        dyn_rev.reset(0.0, 0.0, 0.0, v=0.35)
        for _ in range(30):  # 1.0s at 30Hz holding v_cmd = -0.25
            dyn_rev.step(-0.25, 0.0, 1.0 / 30.0)
        self.assertLess(dyn_rev.min_v_seen, -0.05, "Sustained negative v_cmd must cause reverse overshoot")

        # 2) Releasing negative v_cmd at v <= 0.08 stops cleanly with zero reverse
        dyn_ok = VehicleDynamics(a_eff=1.44, latency_ms=0.0)
        dyn_ok.reset(0.0, 0.0, 0.0, v=0.35)
        for _ in range(30):
            cmd = -0.25 if dyn_ok.v > 0.08 else 0.0
            dyn_ok.step(cmd, 0.0, 1.0 / 30.0)
        self.assertGreaterEqual(dyn_ok.min_v_seen, -0.005)
        self.assertAlmostEqual(dyn_ok.v, 0.0, places=2)

        # 3) Latency queue delays command effect
        dyn_lat = VehicleDynamics(latency_ms=100.0)
        dyn_lat.reset(0.0, 0.0, 0.0, v=0.0)
        dyn_lat.step(0.50, 0.0, 1.0 / 30.0)  # t=0.033s < 0.100s -> still 0
        self.assertAlmostEqual(dyn_lat.v, 0.0, places=4)
        for _ in range(6):
            dyn_lat.step(0.50, 0.0, 1.0 / 30.0)
        self.assertGreater(dyn_lat.v, 0.05)

        # 4) Slippery floor scaling lowers effective_brake_decel
        dyn_nom = VehicleDynamics(a_eff=1.44, a_eff_scale=1.0)
        dyn_slip = VehicleDynamics(a_eff=1.44, a_eff_scale=0.75)
        self.assertAlmostEqual(dyn_slip.effective_brake_decel, 0.75 * dyn_nom.effective_brake_decel, places=4)

    def test_referee_boundary_press_and_3wheel_dnf(self):
        # 1) 1-2 wheels outside -> +15s/s accumulated, NO DNF
        world = PCSimWorld(track="fork_s", rate_hz=30.0, include_base_yellow=False)
        # Lane half-width ~ 215mm, car half-width ~ 95.5mm -> lat_mm=160mm puts 2 left wheels outside, 2 inside
        info = world.reset(pose={"s_m": 1.20, "lat_mm": 165.0}, props=[])
        for _ in range(15):  # 0.5s
            info = world.step(0.0, 0.0)
        ref = info.referee_status
        self.assertIn(ref["wheels_out"], (1, 2))
        self.assertFalse(ref["dnf"])
        self.assertAlmostEqual(ref["boundary_press_time_s"], 0.5, places=2)
        self.assertAlmostEqual(ref["boundary_penalty_s"], 15.0 * ref["boundary_press_time_s"], places=2)

        # 2) >= 3 wheels outside -> immediate DNF
        world_dnf = PCSimWorld(track="fork_s", rate_hz=30.0, include_base_yellow=False)
        world_dnf.reset(pose={"s_m": 1.20, "lat_mm": 380.0}, props=[])
        info_dnf = world_dnf.step(0.0, 0.0)
        self.assertTrue(info_dnf.referee_status["dnf"])
        self.assertGreaterEqual(info_dnf.referee_status["wheels_out"], 3)
        self.assertTrue(info_dnf.done)
        self.assertIn("dnf", info_dnf.done_reason)

    def test_referee_cone_no_entry_bonus_and_net_score(self):
        world = PCSimWorld(track="fork_s", rate_hz=30.0, include_base_yellow=False)
        world.reset(
            pose={"s_m": 1.00, "lat_mm": 0.0},
            props=[
                {"type": "red_cone", "s_m": 1.32, "lat_mm": 0.0, "id": "c_hit"},
                {"type": "red_no_entry", "s_m": 1.62, "lat_mm": 0.0, "id": "ne_hit"},
                {"type": "digit", "s_m": 1.92, "lat_mm": 0.0, "value": 6, "id": "dig6"},
            ],
        )
        info = None
        for _ in range(130):
            info = world.step(0.40, 0.0)
            if info.referee_status["s_m"] >= 2.15:
                break
        assert info is not None
        ref = info.referee_status
        self.assertEqual(ref["cone_hits"], 1)
        self.assertAlmostEqual(ref["cone_penalty_s"], 10.0, places=3)
        self.assertEqual(ref["no_entry_hits"], 1)
        self.assertAlmostEqual(ref["no_entry_penalty_s"], 30.0, places=3)
        self.assertEqual(ref["bonus_digits_collected"], [6])
        self.assertAlmostEqual(ref["total_bonus_s"], 30.0, places=3)
        expected_net = ref["lap_time_s"] + ref["total_penalty_s"] - ref["total_bonus_s"]
        self.assertAlmostEqual(ref["net_score_s"], expected_net, places=4)

        # Verify top-down rendering output shape
        td = world.render_topdown(width=640, height=480)
        self.assertEqual(td.shape, (480, 640, 3))
        self.assertEqual(td.dtype, np.uint8)

    def test_referee_yellow_clean_stop_vs_violation(self):
        # 1) Blow right through yellow line without stopping -> +15s violation (once)
        world_viol = PCSimWorld(track="fork_s", rate_hz=30.0, include_base_yellow=False)
        world_viol.reset(
            pose={"s_m": 1.10, "lat_mm": 0.0},
            props=[{"type": "yellow_stop_line", "s_m": 1.55, "lat_mm": 0.0, "id": "y_viol"}],
        )
        info = None
        for _ in range(70):
            info = world_viol.step(0.45, 0.0)
        assert info is not None
        self.assertEqual(info.referee_status["yellow_violations"], 1)
        self.assertAlmostEqual(info.referee_status["yellow_penalty_s"], 15.0, places=3)

        # 2) Drive toward yellow line, stop cleanly before bumper touches yellow (|v|<=0.02 for >=0.15s), then cross -> 0 penalty
        world_ok = PCSimWorld(track="fork_s", rate_hz=30.0, include_base_yellow=False)
        world_ok.reset(
            pose={"s_m": 1.00, "lat_mm": 0.0},
            props=[{"type": "yellow_stop_line", "s_m": 1.65, "lat_mm": 0.0, "id": "y_ok"}],
        )
        # Approach yellow line so vehicle has genuinely moved (_has_moved=True), then brake to stop before bumper touches
        for _ in range(18):
            info = world_ok.step(0.30, 0.0)
        for _ in range(45):  # brake & hold still >= 0.15s before yellow line
            cmd_v = -0.25 if world_ok.dynamics.v > 0.08 else 0.0
            info = world_ok.step(cmd_v, 0.0)
        # Then drive across yellow line
        for _ in range(70):
            info = world_ok.step(0.40, 0.0)
        assert info is not None
        self.assertEqual(info.referee_status["yellow_violations"], 0)
        self.assertGreaterEqual(info.referee_status["yellow_stops_ok"], 1)
        self.assertAlmostEqual(info.referee_status["yellow_penalty_s"], 0.0, places=3)


class TestEndToEndSuiteAndAutoTune(unittest.TestCase):
    """Verify pc_sim closed-loop runner, dual-pane composition, regression specs, and auto_tune_pc."""

    def test_closed_loop_runner_and_dual_pane_snapshot(self):
        cfg = pc_sim.SimConfig(
            track="fork_s",
            detector="mock",
            rate_hz=15.0,
            max_time_s=3.5,
            cones=1,
            digits=1,
            seed=42,
            prop_seed=42,
        )
        with tempfile.TemporaryDirectory() as tmpd:
            snap_path = Path(tmpd) / "test_dual_pane.png"
            res = pc_sim.run_episode(cfg, snapshot_path=snap_path)
            self.assertGreater(res.steps, 20)
            self.assertFalse(res.dnf)
            self.assertTrue(snap_path.is_file())
            img = cv2.imdecode(np.frombuffer(snap_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(img)
            self.assertEqual(img.shape, (480, 1280, 3))

    def test_regression_scenario_builder_has_all_4_scenarios(self):
        specs = regression_suite.build_regression_scenarios(detector="mock", rate_hz=30.0)
        names = [s.name for s in specs]
        self.assertEqual(
            names,
            [
                "roundabout_red_to_sharp_right",
                "fork_y_merge_anti_uturn",
                "straight_dual_digit_5_2",
                "yellow_stop_no_reverse",
            ],
        )

    def test_auto_tune_safety_floors_and_dr_suite(self):
        # Safety floors strictly locked
        clamped = auto_tune_pc.enforce_safety_floors(
            {
                "border_margin_m": 0.005,
                "no_entry_margin_m": 0.010,
                "yellow_bumper_margin_m": 0.030,
                "target_speed": 0.45,
            }
        )
        self.assertGreaterEqual(clamped["border_margin_m"], 0.020)
        self.assertGreaterEqual(clamped["no_entry_margin_m"], 0.025)
        self.assertGreaterEqual(clamped["yellow_bumper_margin_m"], 0.070)
        auto_tune_pc.validate_safety_floors(clamped)

        with self.assertRaises(ValueError):
            auto_tune_pc.validate_safety_floors({"border_margin_m": 0.019})
        with self.assertRaises(ValueError):
            auto_tune_pc.validate_safety_floors({"no_entry_margin_m": 0.024})
        with self.assertRaises(ValueError):
            auto_tune_pc.validate_safety_floors({"yellow_bumper_margin_m": 0.069})

        # DR suite includes nominal + slippery (0.75) + +40ms latency + +-15% steer/speed mismatch
        suite = auto_tune_pc.build_dr_suite(seed=42)
        self.assertEqual(len(suite), 5)
        tags = {s.tag for s in suite}
        self.assertEqual(
            tags,
            {
                "nominal",
                "slippery_0.75",
                "latency_+40ms",
                "mismatch_s0.85_v1.15",
                "mismatch_s1.15_v0.85",
            },
        )
        by_tag = {s.tag: s for s in suite}
        self.assertAlmostEqual(by_tag["slippery_0.75"].a_eff_scale, 0.75, places=3)
        self.assertAlmostEqual(by_tag["latency_+40ms"].latency_ms, 40.0, places=3)
        self.assertAlmostEqual(by_tag["mismatch_s0.85_v1.15"].steer_scale, 0.85, places=3)
        self.assertAlmostEqual(by_tag["mismatch_s0.85_v1.15"].speed_scale, 1.15, places=3)


if __name__ == "__main__":
    unittest.main()
