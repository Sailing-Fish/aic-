#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone Windows/Linux unit test suite for tools/config, tools/finals, tools/panel, tools/sim."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "finals"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "panel"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "sim"))

import calibrate_homography as calib_mod  # noqa: E402
import extract_keyframes as kf_mod  # noqa: E402
import params_schema  # noqa: E402
import record_and_compare_cv as cv_cmp  # noqa: E402
import render as sim_render  # noqa: E402
import server as panel_server  # noqa: E402
import synth_finals_dataset as synth_mod  # noqa: E402
import train_digits8_yolo as train8_mod  # noqa: E402
import tune as sim_tune  # noqa: E402
import world_node as sim_world  # noqa: E402


class TestYamlAndSchemaParity(unittest.TestCase):
    YAML_FILES = [
        REPO_ROOT / "tools" / "config" / "finals_fork.yaml",
        REPO_ROOT / "tools" / "config" / "finals_roundabout.yaml",
        REPO_ROOT / "src" / "line_follow_ws" / "src" / "line_follower" / "config" / "line_follower_params.yaml",
    ]

    def test_legacy_yamls_removed_and_finals_yamls_parse_cleanly(self):
        for removed in ("fork.yaml", "roundabout.yaml"):
            self.assertFalse(
                (REPO_ROOT / "tools" / "config" / removed).exists(),
                f"Legacy config should be deleted: {removed}",
            )
        for p in self.YAML_FILES:
            self.assertTrue(p.is_file(), f"Missing YAML: {p}")
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            self.assertIsInstance(data, dict, f"YAML root not dict: {p}")
            self.assertIn("line_follower_node", data, f"Missing line_follower_node in {p}")
            self.assertIn("cmd_mux_node", data, f"Missing cmd_mux_node in {p}")
            self.assertNotIn("detection_turn_controller", data, f"Legacy section left in {p}")

    def test_finals_yamls_b1_b2_b3_b5_b10_and_cleanup(self):
        removed_lf = {
            "planner_mode", "enable_red_filter", "enable_soft_red_avoidance",
            "seg1_target_speed", "seg2_start_count", "seg3_start_count",
            "seg3_start_dist_m", "seg4_start_dist_m", "yellow_permanent_count",
            "yellow_area_ratio_threshold", "red_soft_yaw",
        }
        removed_mux = {"high_topic", "high_timeout", "enable_detection_override", "session_cooldown"}

        for name in ("finals_fork.yaml", "finals_roundabout.yaml"):
            data = yaml.safe_load((REPO_ROOT / "tools" / "config" / name).read_text(encoding="utf-8"))
            self.assertIn("camera_publisher_node", data)
            self.assertIn("bpu_yolov8_digit_node", data)

            lf = data["line_follower_node"]["ros__parameters"]
            mux = data["cmd_mux_node"]["ros__parameters"]
            cam = data["camera_publisher_node"]["ros__parameters"]
            yolo = data["bpu_yolov8_digit_node"]["ros__parameters"]

            self.assertEqual(lf["lane_seg_mode"], "relative_border")
            self.assertEqual(lf["yellow_detect_mode"], "geometric_cv")
            self.assertTrue(lf["enable_semantic_hole_fill"])
            self.assertTrue(lf["enable_obstacle_memory"])
            self.assertTrue(lf["enable_cv_red_fallback"])
            self.assertTrue(lf["yellow_require_odom_still"])
            self.assertTrue(lf["lost_line_safe_mode"])
            self.assertEqual(len(lf["homography_matrix"]), 9)

            # B1 / B3 / B5 parameters
            self.assertAlmostEqual(lf["yellow_roi_bottom_ratio"], 0.0)
            self.assertAlmostEqual(lf["rear_axle_to_bumper_m"], 0.18)
            self.assertAlmostEqual(lf["obstacle_steer_horizon_m"], 0.65)
            self.assertAlmostEqual(lf["obstacle_curve_min_horizon_m"], 0.45)
            self.assertAlmostEqual(lf["obstacle_stagger_gap_m"], 0.25)

            # B10 red fallback HSV parameters
            for k in ("red_h_low_1", "red_h_high_1", "red_h_low_2", "red_h_high_2",
                      "red_s_min", "red_v_min", "red_cone_area_ratio_threshold"):
                self.assertIn(k, lf)

            # B2 camera + 8-class YOLO params
            self.assertEqual(cam["ae_awb_mode"], "lock_on_start")
            self.assertEqual(cam["power_line_frequency"], 1)
            self.assertEqual(cam["backlight_compensation"], 0)
            self.assertTrue(cam["unlock_on_stop"])
            self.assertEqual(yolo["cls_list_path"], "")
            self.assertAlmostEqual(yolo["conf_thresh"], 0.25)
            self.assertEqual(yolo["class_conf_overrides"], "{}")

            self.assertFalse(removed_lf & set(lf.keys()), f"Legacy lf keys in {name}: {removed_lf & set(lf.keys())}")
            self.assertFalse(removed_mux & set(mux.keys()), f"Legacy mux keys in {name}: {removed_mux & set(mux.keys())}")

    def test_yaml_and_schema_keys_exist_in_cpp_and_python_declarations(self):
        lf_cpp = (
            REPO_ROOT / "src" / "line_follow_ws" / "src" / "line_follower" / "src" / "line_follower_node.cpp"
        ).read_text(encoding="utf-8")
        mux_cpp = (
            REPO_ROOT / "src" / "line_follow_ws" / "src" / "line_follower" / "src" / "cmd_mux_node.cpp"
        ).read_text(encoding="utf-8")
        cam_py = (REPO_ROOT / "src" / "camera" / "camera.py").read_text(encoding="utf-8")
        yolo_py = (REPO_ROOT / "src" / "yolov8" / "bpu_yolov8_node.py").read_text(encoding="utf-8")

        lf_declared = set(re.findall(r'declare_parameter(?:<[^>]+>)?\(\s*"([^"]+)"', lf_cpp))
        mux_declared = set(re.findall(r'declare_parameter(?:<[^>]+>)?\(\s*"([^"]+)"', mux_cpp))
        cam_declared = set(re.findall(r"declare_parameter\(\s*'([^']+)'", cam_py))
        yolo_declared = set(re.findall(r"declare_parameter\(\s*'([^']+)'", yolo_py))

        for grp in params_schema.SCHEMA:
            sec = grp["section"]
            if sec == "line_follower_node":
                for item in grp["items"]:
                    self.assertIn(item[0], lf_declared, f"Schema key '{item[0]}' not declared in line_follower_node.cpp")
            elif sec == "cmd_mux_node":
                for item in grp["items"]:
                    self.assertIn(item[0], mux_declared, f"Schema key '{item[0]}' not declared in cmd_mux_node.cpp")

        for name in ("finals_fork.yaml", "finals_roundabout.yaml"):
            data = yaml.safe_load((REPO_ROOT / "tools" / "config" / name).read_text(encoding="utf-8"))
            for key in data["line_follower_node"]["ros__parameters"]:
                self.assertIn(key, lf_declared, f"YAML key '{key}' in {name} not declared in line_follower_node.cpp")
            for key in data["cmd_mux_node"]["ros__parameters"]:
                self.assertIn(key, mux_declared, f"YAML key '{key}' in {name} not declared in cmd_mux_node.cpp")
            for key in data["camera_publisher_node"]["ros__parameters"]:
                self.assertIn(key, cam_declared, f"YAML key '{key}' in {name} not declared in camera.py")
            for key in data["bpu_yolov8_digit_node"]["ros__parameters"]:
                self.assertIn(key, yolo_declared, f"YAML key '{key}' in {name} not declared in bpu_yolov8_node.py")


class TestHomographyCalibration(unittest.TestCase):
    def test_exact_pinhole_roundtrip_and_yaml_output(self):
        # Generate synthetic ground -> pixel points from CameraModel (ROS +y left => w_right = -y_lat)
        cam = sim_render.CameraModel(h_mm=110.0, alpha_deg=12.5, f_px=325.0, width=640, height=480)
        overhang_m = 0.06
        ground_pts = [
            (0.30, 0.215),
            (0.30, -0.215),
            (0.60, 0.215),
            (0.60, -0.215),
            (1.00, 0.215),
            (1.00, -0.215),
        ]
        img_pts = []
        for d_m, y_m in ground_pts:
            u, v = cam.wd_to_uv(w_mm=-y_m * 1000.0, d_mm=(d_m + overhang_m) * 1000.0)
            img_pts.append((u, v))

        res = calib_mod.solve_homography(img_pts, ground_pts)
        self.assertLess(res["mean_err_mm"], 0.5)
        self.assertLess(res["max_err_mm"], 1.0)
        self.assertLess(res["max_px_err"], 0.1)
        self.assertGreater(abs(res["det"]), 1e-12)

        # Test projection on an unseen midpoint (0.80m, 0.0m)
        u_mid, v_mid = cam.wd_to_uv(w_mm=0.0, d_mm=(0.80 + overhang_m) * 1000.0)
        d_pred, y_pred = calib_mod.pixel_to_ground(res["H"], u_mid, v_mid)
        self.assertAlmostEqual(d_pred, 0.80, places=3)
        self.assertAlmostEqual(y_pred, 0.0, places=3)

        # Test YAML snippet parsing & strict ROS2 double_array float serialization
        snippet = calib_mod.format_yaml_snippet(res["H"])
        parsed = yaml.safe_load(snippet)
        self.assertTrue(parsed["use_homography"])
        self.assertEqual(len(parsed["homography_matrix"]), 9)
        self.assertTrue(
            all(isinstance(x, float) for x in parsed["homography_matrix"]),
            f"All homography_matrix entries must parse as float for ROS2 double_array: {parsed['homography_matrix']}",
        )
        # Also test exact identity matrix (0.0 and 1.0) serializes strictly as YAML floats
        id_parsed = yaml.safe_load(calib_mod.format_yaml_snippet(np.eye(3)))
        self.assertTrue(all(isinstance(x, float) for x in id_parsed["homography_matrix"]))

    def test_load_points_json_and_csv(self):
        pts = [
            {"u": 180, "v": 420, "d_bumper_m": 0.30, "y_lat_m": 0.215},
            {"u": 460, "v": 420, "d_bumper_m": 0.30, "y_lat_m": -0.215},
            {"u": 250, "v": 270, "d_bumper_m": 1.00, "y_lat_m": 0.215},
            {"u": 390, "v": 270, "d_bumper_m": 1.00, "y_lat_m": -0.215},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            jp = Path(tmp) / "pts.json"
            jp.write_text(json.dumps(pts), encoding="utf-8")
            ip, gp = calib_mod.load_points_file(jp)
            self.assertEqual(len(ip), 4)
            self.assertEqual(gp[0], (0.30, 0.215))

            cp = Path(tmp) / "pts.csv"
            with open(cp, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["u", "v", "d_bumper_m", "y_lat_m"])
                w.writeheader()
                w.writerows(pts)
            ip2, gp2 = calib_mod.load_points_file(cp)
            self.assertEqual(ip, ip2)
            self.assertEqual(gp, gp2)

        with self.assertRaises(ValueError):
            calib_mod.solve_homography([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])


class TestSyntheticDatasetGenerator(unittest.TestCase):
    def test_flip_strictly_forbidden_and_rotation_capped(self):
        self.assertFalse(synth_mod.ALLOW_HFLIP)
        self.assertFalse(synth_mod.ALLOW_VFLIP)
        rng = __import__("random").Random(1)
        card = synth_mod.make_digit_card_bgra("5", rng)
        with self.assertRaises(ValueError):
            synth_mod.warp_prop_bgra(card, rng, allow_flip=True)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                synth_mod.generate_dataset(tmp, num_train=1, num_val=1, max_rot_deg=25.0)

    def test_generate_11class_dataset_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = synth_mod.generate_dataset(
                out_dir=tmp, num_train=8, num_val=3, seed=123, max_rot_deg=15.0, calib_count=6
            )
            self.assertEqual(info["num_train"], 8)
            self.assertEqual(info["num_val"], 3)
            self.assertEqual(info["calib_count"], 6)
            self.assertGreater(info["total_boxes"], 0)

            ds_yaml = yaml.safe_load(Path(info["dataset_yaml"]).read_text(encoding="utf-8"))
            self.assertEqual(ds_yaml["nc"], 11)
            self.assertEqual(ds_yaml["names"][0], "1")
            self.assertEqual(ds_yaml["names"][7], "8")
            self.assertEqual(ds_yaml["names"][8], "red_cone")
            self.assertEqual(ds_yaml["names"][9], "red_no_entry")
            self.assertEqual(ds_yaml["names"][10], "yellow_stop_line")

            calib_lines = [
                ln.strip() for ln in Path(info["calib_list"]).read_text(encoding="utf-8").splitlines() if ln.strip()
            ]
            self.assertEqual(len(calib_lines), 6)
            for p in calib_lines:
                self.assertTrue(Path(p).is_file(), f"Missing calib image: {p}")

            # Verify every generated YOLO label line
            for split in ("train", "val"):
                lbl_files = list((Path(tmp) / "labels" / split).glob("*.txt"))
                self.assertGreater(len(lbl_files), 0)
                for lf in lbl_files:
                    for ln in lf.read_text(encoding="utf-8").splitlines():
                        parts = ln.split()
                        self.assertEqual(len(parts), 5)
                        cid = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        self.assertTrue(0 <= cid < 11)
                        self.assertTrue(0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0)
                        self.assertTrue(0.0 < bw <= 1.0 and 0.0 < bh <= 1.0)

    def test_generate_8class_dataset_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                synth_mod.generate_dataset(tmp, num_train=1, num_val=1, num_classes=9)

            info = synth_mod.generate_dataset(
                out_dir=tmp,
                num_train=10,
                num_val=4,
                seed=456,
                max_rot_deg=15.0,
                calib_count=5,
                num_classes=8,
                mnist_dir=None,
            )
            self.assertEqual(info["num_classes"], 8)
            self.assertGreater(info["total_boxes"], 0)
            ds_yaml = yaml.safe_load(Path(info["dataset_yaml"]).read_text(encoding="utf-8"))
            self.assertEqual(ds_yaml["nc"], 8)
            self.assertEqual([ds_yaml["names"][i] for i in range(8)], ["1", "2", "3", "4", "5", "6", "7", "8"])

            for split in ("train", "val"):
                lbl_files = list((Path(tmp) / "labels" / split).glob("*.txt"))
                self.assertEqual(len(lbl_files), 10 if split == "train" else 4)
                for lf in lbl_files:
                    for ln in lf.read_text(encoding="utf-8").splitlines():
                        parts = ln.split()
                        self.assertEqual(len(parts), 5)
                        cid = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        self.assertTrue(0 <= cid < 8, f"8-class mode leaked cid={cid}")
                        self.assertTrue(0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0)
                        self.assertTrue(0.0 < bw <= 1.0 and 0.0 < bh <= 1.0)

    def test_3d_ground_homography_ellipse_aspect_and_guards(self):
        cam = sim_render.CameraModel(h_mm=110.0, alpha_deg=12.5, f_px=325.0, width=640, height=480)
        for w_mm, d_mm in ((0.0, 450.0), (-120.0, 800.0), (150.0, 320.0)):
            u1, v1 = synth_mod.project_3d_to_uv(w_mm, d_mm, z_mm=0.0)
            u2, v2 = cam.wd_to_uv(w_mm, d_mm)
            self.assertAlmostEqual(u1, u2, places=5)
            self.assertAlmostEqual(v1, v2, places=5)

        rng = __import__("random").Random(77)
        sticker = synth_mod.make_official_sticker_bgra("6", rng, handwritten=False, glossy_pvc=True, size=160)
        canvas = np.full((480, 640, 3), 240, dtype=np.uint8)
        box = synth_mod.project_ground_prop_bgra(
            canvas, sticker, d_m=0.45, y_m=0.0, phys_w_m=0.13, phys_h_m=0.13, rot_deg=0.0
        )
        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        self.assertGreater(bw, 40)
        self.assertGreater(bh, 8)
        aspect = bw / float(bh)
        self.assertGreater(aspect, 2.8, f"Expected flattened 3.5:1~5:1 ellipse, got aspect={aspect:.2f}")

        with self.assertRaises(ValueError):
            synth_mod.project_ground_prop_bgra(canvas, sticker, d_m=0.45, y_m=0.0, phys_w_m=0.13, phys_h_m=0.13, allow_flip=True)
        with self.assertRaises(ValueError):
            synth_mod.project_ground_prop_bgra(canvas, sticker, d_m=0.45, y_m=0.0, phys_w_m=0.13, phys_h_m=0.13, rot_deg=20.0)

    def test_mnist_cache_load_and_offline_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_bank = synth_mod.load_mnist_digits(tmp, allow_download=False)
            self.assertEqual(sum(len(v) for v in empty_bank.values()), 0)

            # Create fake cached mnist_1_8.npz and verify fast load + dataset generation
            fake_glyph = np.zeros((28, 28), dtype=np.uint8)
            cv2.putText(fake_glyph, "3", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
            np.savez_compressed(
                Path(tmp) / "mnist_1_8.npz",
                **{f"d{d}": np.repeat(fake_glyph[None, :, :], 3, axis=0) for d in synth_mod.DIGITS8_CLASSES},
            )
            loaded = synth_mod.load_mnist_digits(tmp, allow_download=False)
            for d in synth_mod.DIGITS8_CLASSES:
                self.assertEqual(loaded[d].shape, (3, 28, 28))

            out_ds = Path(tmp) / "ds"
            info = synth_mod.generate_dataset(
                out_dir=out_ds,
                num_train=4,
                num_val=2,
                seed=99,
                num_classes=8,
                mnist_dir=tmp,
                allow_mnist_download=False,
            )
            self.assertEqual(info["mnist_samples"], 24)
            self.assertEqual(info["num_classes"], 8)

    def test_train_digits8_hardcoded_no_flip_params(self):
        self.assertEqual(train8_mod.FORCED_TRAIN_AUG_KWARGS["fliplr"], 0.0)
        self.assertEqual(train8_mod.FORCED_TRAIN_AUG_KWARGS["flipud"], 0.0)
        self.assertLessEqual(train8_mod.FORCED_TRAIN_AUG_KWARGS["degrees"], 15.0)


class TestCVComparisonTool(unittest.TestCase):
    def test_compare_three_modes_on_aged_warm_track(self):
        rng = __import__("random").Random(7)
        frames = []
        for _ in range(8):
            frm = synth_mod.make_synthetic_track_bg(rng, 640, 480)
            # Simulate warm/aged yellowed mat (pushes white S > 55, challenging legacy_hsv)
            arr = frm.astype(np.float32)
            arr[:, :, 0] *= 0.72
            arr[:, :, 1] *= 0.96
            arr[:, :, 2] *= 1.04
            frames.append(np.clip(arr, 0, 255).astype(np.uint8))

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "metrics.csv"
            summary = cv_cmp.compare_frames(frames, out_csv=csv_path, out_video=None)
            self.assertTrue(csv_path.is_file())
            for mode in cv_cmp.MODES:
                self.assertIn(mode, summary)
                self.assertEqual(summary[mode]["frames"], 8.0)
            # relative_border should detect the lane reliably even on yellowed mat
            self.assertLessEqual(summary["relative_border"]["dropout_rate"], 0.25)


class TestPanelAndTelemetry(unittest.TestCase):
    def test_load_params_finals_maps_only(self):
        self.assertEqual(panel_server.MAP_FILES, ("finals_fork", "finals_roundabout"))
        self.assertEqual(panel_server.DEFAULT_MAP, "finals_fork")
        with self.assertRaises(FileNotFoundError):
            panel_server.load_params("fork")

        required_keys = {
            "lane_seg_mode", "yellow_detect_mode",
            "enable_semantic_hole_fill", "enable_obstacle_memory",
            "enable_cv_red_fallback", "allow_dual_digit_cover",
            "yellow_roi_bottom_ratio", "rear_axle_to_bumper_m",
            "obstacle_steer_horizon_m", "obstacle_curve_min_horizon_m",
            "obstacle_stagger_gap_m",
            "yellow_h_low", "yellow_h_high", "yellow_s_min", "yellow_v_min",
            "red_h_low_1", "red_h_high_1", "red_h_low_2", "red_h_high_2",
            "red_s_min", "red_v_min", "red_cone_area_ratio_threshold",
            "stop_timeout", "low_timeout", "watchdog_rate_hz",
        }
        for map_name in ("finals_fork", "finals_roundabout"):
            payload = panel_server.load_params(map_name)
            self.assertEqual(payload["map"], map_name)
            found = {it["key"] for g in payload["groups"] for it in g["items"]}
            missing = required_keys - found
            self.assertFalse(missing, f"Map {map_name} missing panel keys: {missing}")

    def test_save_params_enum_validation_in_temp_dir(self):
        orig_dir = panel_server.CONFIG_DIR
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(REPO_ROOT / "tools" / "config" / "finals_fork.yaml", Path(tmp) / "finals_fork.yaml")
            panel_server.CONFIG_DIR = tmp
            try:
                ok, _msg = panel_server.save_params("finals_fork", {
                    "line_follower_node": {"lane_seg_mode": "invalid_mode_xyz"}
                })
                self.assertFalse(ok)

                ok2, msg2 = panel_server.save_params("finals_fork", {
                    "line_follower_node": {"lane_seg_mode": "hybrid", "allow_dual_digit_cover": True}
                })
                self.assertTrue(ok2, msg2)
                reloaded = panel_server.load_params("finals_fork")
                kv = {it["key"]: it["value"] for g in reloaded["groups"] for it in g["items"]}
                self.assertEqual(kv["lane_seg_mode"], "hybrid")
                self.assertTrue(kv["allow_dual_digit_cover"])
            finally:
                panel_server.CONFIG_DIR = orig_dir

    def test_panel_html_has_voltage_and_finals_only_ui(self):
        html = (REPO_ROOT / "tools" / "panel" / "index.html").read_text(encoding="utf-8")
        self.assertIn("voltBadge", html)
        self.assertIn("11.6", html)
        self.assertIn("finals_fork", html)
        self.assertIn("finals_roundabout", html)
        self.assertIn("it.type==='enum'", html)
        self.assertIn('data-tier="finals_core"', html)
        self.assertIn('data-tier="calibration"', html)
        self.assertIn("已停 0 次", html)
        self.assertIn("v=- kp=-", html)
        self.assertNotIn('data-tier="legacy"', html)
        self.assertNotIn("segSel", html)
        self.assertNotIn("0 / 4", html)
        self.assertNotIn('<option value="fork">', html)

    def test_panel_two_tiers_and_b10_grouping(self):
        payload = panel_server.load_params("finals_fork")
        self.assertEqual([t["id"] for t in payload["tiers"]], ["finals_core", "calibration"])

        by_tier = {"finals_core": set(), "calibration": set()}
        for g in payload["groups"]:
            self.assertIn(g["tier"], by_tier)
            for it in g["items"]:
                by_tier[g["tier"]].add(it["key"])

        self.assertLessEqual(len(by_tier["finals_core"]), 20)
        for core_key in ("target_speed", "speed_gain_damping", "corridor_kp_y",
                         "yellow_bumper_margin_m", "yellow_creep_speed_mps",
                         "obstacle_speed_mps", "no_entry_speed_mps",
                         "cone_margin_m", "no_entry_margin_m"):
            self.assertIn(core_key, by_tier["finals_core"])

        for b_key in ("yellow_roi_bottom_ratio", "rear_axle_to_bumper_m",
                      "obstacle_steer_horizon_m", "obstacle_curve_min_horizon_m",
                      "obstacle_stagger_gap_m"):
            self.assertIn(b_key, by_tier["calibration"])

        lane_seg_spec = params_schema.find("line_follower_node", "lane_seg_mode")
        self.assertEqual(lane_seg_spec[3], ["relative_border", "hybrid", "legacy_hsv"])

        # B10: 6 red HSV params + red_cone_area_ratio_threshold directly follow enable_cv_red_fallback
        switch_grp = next(
            g for g in payload["groups"]
            if any(it["key"] == "enable_cv_red_fallback" for it in g["items"])
        )
        keys = [it["key"] for it in switch_grp["items"]]
        idx = keys.index("enable_cv_red_fallback")
        self.assertEqual(
            keys[idx + 1:idx + 8],
            ["red_h_low_1", "red_h_high_1", "red_h_low_2", "red_h_high_2",
             "red_s_min", "red_v_min", "red_cone_area_ratio_threshold"],
        )

        # No deleted legacy keys remain in schema
        all_keys = by_tier["finals_core"] | by_tier["calibration"]
        for removed in ("seg1_target_speed", "seg2_start_count", "planner_mode",
                        "enable_red_filter", "enable_soft_red_avoidance",
                        "enable_detection_override", "turn_duration",
                        "yellow_area_ratio_threshold", "yellow_permanent_count"):
            self.assertNotIn(removed, all_keys)

    def test_b6_explicit_final_stop_ignores_long_temp_yellow_stop(self):
        self.assertFalse(hasattr(panel_server.PanelNode, "_tick"))
        with panel_server.SH.lock:
            panel_server.reset_round_locked()

        # Pre-start final=1 must be ignored
        panel_server.apply_segment_message("mode=finals s=0.00 count=0 branch=0 final=1", now=90.0)
        with panel_server.SH.lock:
            self.assertFalse(panel_server.SH.final_stopped)
            self.assertIsNone(panel_server.SH.timer_stop)

            # Start run at t=100.0
            panel_server.SH.timer_start = 100.0
            panel_server.SH.started = True
            panel_server.SH.track_state = "yellow_line_detected"
            panel_server.SH.yellow_count = 1
            panel_server.SH.last_yellow_ts = 101.0

        # Even if temporary yellow stop lasts > 2.5s (t=104.0), final=0 must NOT stop timer
        panel_server.apply_segment_message("mode=finals s=3.20 count=1 branch=0 final=0", now=104.0)
        with panel_server.SH.lock:
            self.assertFalse(panel_server.SH.final_stopped)
            self.assertIsNone(panel_server.SH.timer_stop)

        # Explicit final=1 stops the timer cleanly
        panel_server.apply_segment_message("mode=finals s=10.80 count=3 branch=0 final=1", now=122.5)
        with panel_server.SH.lock:
            self.assertTrue(panel_server.SH.final_stopped)
            self.assertEqual(panel_server.SH.timer_stop, 122.5)
            panel_server.reset_round_locked()

    def test_raw_video_recorder_and_keyframe_extractor_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap_dir = Path(tmp) / "raw_videos"
            rec = panel_server.RawVideoRecorder(capture_dir=str(cap_dir))
            ok, fname = rec.start(scene="hard_neg!@#", lighting="dim_shadow", fps=10.0)
            self.assertTrue(ok, fname)
            self.assertIn("hard_neg", fname)
            self.assertIn("dim_shadow", fname)

            # Push 20 frames across 2.0 seconds (moving square -> distinct frames, plus static duplicates)
            t0 = 1000.0
            for i in range(20):
                img = np.full((240, 320, 3), 210, dtype=np.uint8)
                # First 10 frames move a textured box; last 10 stay static to test dedup
                x_off = (i * 22) if i < 10 else (9 * 22)
                cv2.rectangle(img, (10 + x_off, 40), (70 + x_off, 180), (20, 20, 220), -1)
                cv2.putText(img, str(i), (15 + x_off, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                rec.push_frame(img, now=t0 + i * 0.11)

            ok_stop, info = rec.stop()
            self.assertTrue(ok_stop, info)
            self.assertGreaterEqual(info["frames"], 10)
            files = rec.list_files()
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0]["name"], info["filename"])

            # Path traversal must be rejected
            self.assertIsNone(rec.resolve_safe_path("../secret.mp4"))
            self.assertIsNotNone(rec.resolve_safe_path(info["filename"]))

            # Extract keyframes: should deduplicate static tail and auto-create empty labels for hard_neg
            out_dir = Path(tmp) / "keyframes"
            summary = kf_mod.extract_directory(
                cap_dir, out_dir, interval_sec=0.1, min_diff=4.0, min_sharpness=10.0
            )
            self.assertEqual(summary["video_count"], 1)
            self.assertGreaterEqual(summary["total_saved"], 3)
            self.assertGreater(summary["videos"][0]["skipped_dup"], 0)
            self.assertTrue(summary["videos"][0]["is_hard_neg"])
            imgs = list((out_dir / "images").glob("*.jpg"))
            lbls = list((out_dir / "labels").glob("*.txt"))
            self.assertEqual(len(imgs), summary["total_saved"])
            self.assertEqual(len(lbls), summary["total_saved"])
            self.assertTrue((out_dir / "manifest.json").is_file())

            # Delete file via recorder API
            ok_del, _ = rec.delete_file(info["filename"])
            self.assertTrue(ok_del)
            self.assertEqual(len(rec.list_files()), 0)

            # Test synchronized debug blackbox bundle (.zip with mp4 + follow_slice + events_slice + timestamps)
            dbg_dir = Path(tmp) / "debug_runs"
            fake_log_dir = Path(tmp) / "run_log"
            fake_log_dir.mkdir(parents=True, exist_ok=True)
            (fake_log_dir / "follow.log").write_text("old line\n", encoding="utf-8")
            (fake_log_dir / "events.csv").write_text("ts,t,src,area,event,kv\n", encoding="utf-8")
            (fake_log_dir / "params_snapshot.yaml").write_text("line_follower_node: {}\n", encoding="utf-8")

            rec2 = panel_server.RawVideoRecorder(capture_dir=str(cap_dir), debug_dir=str(dbg_dir))
            ok_d, dfn = rec2.start(scene="fork_test", fps=10.0, source="debug", log_dir=str(fake_log_dir))
            self.assertTrue(ok_d, dfn)
            with open(fake_log_dir / "follow.log", "a", encoding="utf-8") as fl:
                fl.write("[DEC] ctrl mode=fork_right branch=1 v=0.35 yaw=-0.80\n")
            with open(fake_log_dir / "events.csv", "a", encoding="utf-8") as ef:
                ef.write("100,1.2,follow,ctrl,log,mode=fork_right\n")
            for i in range(6):
                img = np.full((240, 320, 3), 180, dtype=np.uint8)
                rec2.push_frame(img, now=2000.0 + i * 0.12)
            ok_ds, dinfo = rec2.stop()
            self.assertTrue(ok_ds, dinfo)
            self.assertTrue(dinfo["filename"].endswith(".zip"))
            zip_fp = rec2.resolve_safe_path(dinfo["filename"])
            self.assertIsNotNone(zip_fp)
            import zipfile as _zf
            with _zf.ZipFile(zip_fp, "r") as zf:
                names = set(zf.namelist())
                self.assertIn("frame_timestamps.csv", names)
                self.assertIn("follow_slice.log", names)
                self.assertIn("events_slice.csv", names)
                self.assertIn("params_snapshot.yaml", names)
                self.assertIn("meta.json", names)
                self.assertIn("fork_right", zf.read("follow_slice.log").decode("utf-8"))

        # Panel HTML includes recorder drawer hooks
        html = (REPO_ROOT / "tools" / "panel" / "index.html").read_text(encoding="utf-8")
        self.assertIn("btnRecToggle", html)
        self.assertIn("recMode", html)
        self.assertIn("/api/record/start", html)
        self.assertIn("/api/record/stop", html)
        self.assertIn("/api/record/download/", html)


class TestSimulatorFixesAndPerturbations(unittest.TestCase):
    def test_off_track_distance_and_accumulation(self):
        centerline = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0]], dtype=np.float64)
        self.assertAlmostEqual(sim_render.point_to_polyline_distance_mm(500.0, 0.0, centerline), 0.0, places=5)
        self.assertAlmostEqual(sim_render.point_to_polyline_distance_mm(500.0, 150.0, centerline), 150.0, places=5)
        self.assertAlmostEqual(sim_render.point_to_polyline_distance_mm(1250.0, 500.0, centerline), 250.0, places=5)

        tracker = sim_world.OffTrackTracker(centerline, lane_width_mm=430.0)
        # Inside half-lane (215 mm) -> off_track stays 0
        self.assertAlmostEqual(tracker.update(500.0, 180.0, 0.1), 0.0, places=6)
        self.assertFalse(tracker.is_off_track)
        # Outside half-lane (260 mm > 215 mm) -> accumulates dt
        self.assertAlmostEqual(tracker.update(500.0, 260.0, 0.25), 0.25, places=6)
        self.assertTrue(tracker.is_off_track)
        self.assertAlmostEqual(tracker.update(500.0, 260.0, 0.50), 0.75, places=6)
        tracker.reset()
        self.assertEqual(tracker.off_track_s, 0.0)

    def test_renderer_lighting_and_finals_props(self):
        track_json = REPO_ROOT / "tools" / "sim" / "tracks" / "fork_s.json"
        renderer = sim_render.TrackRenderer(
            str(track_json), random_yellow=2, cones=2, no_entry=2, prop_seed=99
        )
        self.assertEqual(len(renderer.injected_props["random_yellow"]), 2)
        self.assertEqual(len(renderer.injected_props["cones"]), 2)
        self.assertEqual(len(renderer.injected_props["no_entry"]), 2)

        x0, y0 = renderer.meta["centerline_mm"][0]
        x1, y1 = renderer.meta["centerline_mm"][2]
        yaw = math.atan2(y1 - y0, x1 - x0)
        img_clean = renderer.render(x0, y0, yaw)
        img_pert = renderer.render(x0, y0, yaw, blur=0.8, gain=0.85, color_cast="warm", glare=0.4)
        self.assertEqual(img_clean.shape, (480, 640, 3))
        self.assertEqual(img_pert.shape, (480, 640, 3))
        self.assertEqual(img_pert.dtype, np.uint8)
        self.assertFalse(np.array_equal(img_clean, img_pert))

    def test_tune_parse_world_three_tuple_unpack(self):
        with tempfile.TemporaryDirectory() as tmp:
            wlog = Path(tmp) / "world.log"
            wlog.write_text(
                "[INFO] [100.000] [sim_world]: [sim] s=1.20 off=0.00 lat=10mm v=0.400 yaw=0.0\n"
                "[INFO] [101.000] [sim_world]: [sim] s=1.60 off=0.35 lat=240mm v=0.400 yaw=0.0\n"
                "[INFO] [102.000] [sim_world]: [sim] s=2.00 off=0.35 lat=50mm v=0.400 yaw=0.0\n",
                encoding="utf-8",
            )
            off, smax, fps = sim_tune.parse_world(wlog)
            self.assertAlmostEqual(off, 0.35, places=3)
            self.assertAlmostEqual(smax, 2.00, places=3)
            self.assertAlmostEqual(fps, 30.0, places=1)
        # Ensure tune.py source unpacks 3 values properly
        tune_src = (REPO_ROOT / "tools" / "sim" / "tune.py").read_text(encoding="utf-8")
        self.assertIn("off, smax, fps = parse_world(wlog)", tune_src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
