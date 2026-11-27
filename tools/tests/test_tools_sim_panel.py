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

    def test_panel_backend_hardening_events_params_and_logs(self):
        # 1. save_params rejects non-numeric / NaN / bool values cleanly without crashing
        orig_dir = panel_server.CONFIG_DIR
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(REPO_ROOT / "tools" / "config" / "finals_fork.yaml", Path(tmp) / "finals_fork.yaml")
            panel_server.CONFIG_DIR = tmp
            try:
                for bad_val in ("", "nan", "inf", True):
                    ok, msg = panel_server.save_params("finals_fork", {
                        "line_follower_node": {"target_speed": bad_val}
                    })
                    self.assertFalse(ok)
                    self.assertIn("target_speed", msg)
            finally:
                panel_server.CONFIG_DIR = orig_dir

        # 2. read_recent_events preserves commas inside kv and supports >10 rows
        orig_logroot = panel_server.LOGROOT
        orig_state_dir = panel_server.STATE_DIR
        with tempfile.TemporaryDirectory() as tmp:
            run_id = "test_run_01"
            log_dir = Path(tmp) / "logs" / run_id
            state_dir = Path(tmp) / "state"
            log_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "status.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
            rows = ["ts,t,src,area,event,kv\n"]
            for i in range(25):
                rows.append(f"100.{i},{i}.5,follow,red,avoid_on,side=L,d=0.45\n")
            (log_dir / "events.csv").write_text("".join(rows), encoding="utf-8")
            (log_dir / "follow.log").write_text("hello log\n", encoding="utf-8")

            panel_server.LOGROOT = str(Path(tmp) / "logs")
            panel_server.STATE_DIR = str(state_dir)
            try:
                evs = panel_server.read_recent_events()
                self.assertEqual(len(evs), 25)
                self.assertEqual(evs[-1]["kv"], "side=L,d=0.45")

                # 3. _logs rejects path traversal outside ALLOWED_LOG_FILES & Handler serves assets/status
                import io
                def _invoke_get(req_path):
                    buf = io.BytesIO()
                    h = panel_server.Handler.__new__(panel_server.Handler)
                    h.path = req_path
                    h.headers = {}
                    h.wfile = buf
                    h.requestline = f"GET {req_path} HTTP/1.1"
                    h.request_version = "HTTP/1.1"
                    h.client_address = ("127.0.0.1", 12345)
                    captured = {"status": 0, "headers": {}}
                    h.send_response = lambda code, msg=None: captured.__setitem__("status", code)
                    h.send_header = lambda k, v: captured["headers"].__setitem__(k.lower(), v)
                    h.end_headers = lambda: None
                    h.do_GET()
                    return captured["status"], captured["headers"], buf.getvalue()

                st_code, _hdrs, raw = _invoke_get("/api/logs?f=../../secret&offset=bad")
                body = json.loads(raw.decode("utf-8"))
                self.assertEqual(st_code, 400)
                self.assertFalse(body["ok"])

                st_code2, _hdrs2, raw2 = _invoke_get("/api/status")
                body2 = json.loads(raw2.decode("utf-8"))
                self.assertEqual(st_code2, 200)
                self.assertTrue(body2["ok"])
                self.assertIn("frame_age", body2)
                self.assertEqual(len(body2["events"]), 25)

                for asset in ("duo_hero_day.jpg", "atelier_banner.jpg",
                              "gemini_chibi.jpg", "deepseek_chibi.jpg", "duo_standby.jpg"):
                    acode, _ahdrs, abody = _invoke_get(f"/assets/{asset}")
                    self.assertEqual(acode, 200, f"Asset failed: {asset}")
                    self.assertGreater(len(abody), 1000, f"Asset empty: {asset}")

                # 4. read_diagnostics extracts perf metrics and keeps early milestones/faults across 100 perf rows
                diag_rows = [
                    "ts,t,src,area,event,kv\n",
                    "100.0,1.2,follow,yellow,stop_begin,count=1 s=2.40 finish=0\n",
                    "101.0,2.5,follow,state,change,from=tracking to=lost\n",
                    "102.0,3.0,follow,state,change,from=lost to=tracking\n",
                ]
                for i in range(100):
                    diag_rows.append(
                        f"110.{i},{10+i}.0,camera,camera,perf,fps=15.0 read_p50=63.45ms read_p95=67.39ms fails=0\n"
                    )
                    diag_rows.append(
                        f"110.{i},{10+i}.2,yolo,yolo,perf,proc_fps=14.5 dropped=1 e2e_p95=32.17ms\n"
                    )
                diag_rows.append("210.0,115.0,follow,red_src,detect,src=cv_only d=0.12\n")
                (log_dir / "events.csv").write_text("".join(diag_rows), encoding="utf-8")

                st_code3, _hdrs3, raw3 = _invoke_get("/api/status")
                body3 = json.loads(raw3.decode("utf-8"))
                self.assertEqual(st_code3, 200)
                self.assertAlmostEqual(body3["perf"]["cam_fps"], 15.0)
                self.assertAlmostEqual(body3["perf"]["cam_read_p95"], 67.39)
                self.assertEqual(body3["perf"]["cam_fails"], 0)
                self.assertAlmostEqual(body3["perf"]["yolo_fps"], 14.5)
                self.assertEqual(body3["perf"]["yolo_dropped"], 1)
                self.assertAlmostEqual(body3["perf"]["yolo_e2e_p95"], 32.17)
                self.assertTrue(body3["red_hint"]["active"])
                self.assertEqual(body3["red_hint"]["src"], "cv_only")

                # Early yellow stop & lost state survive 200 perf rows; normal tracking & perf rows are filtered out
                m_pairs = [(m["area"], m["event"]) for m in body3["milestones"]]
                self.assertEqual(m_pairs, [("yellow", "stop_begin"), ("state", "change")])
                self.assertIn("to=lost", body3["milestones"][1]["kv"])
            finally:
                panel_server.LOGROOT = orig_logroot
                panel_server.STATE_DIR = orig_state_dir

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

    def test_brake_speed_params_schema_presets_and_save(self):
        expected_specs = {
            "enable_sqrt_brake_envelope": ("bool", 0, 1),
            "enable_closed_loop_brake": ("bool", 0, 1),
            "brake_damping_gain": ("float", 0.2, 2.0),
            "brake_zero_cutoff_mps": ("float", 0.005, 0.08),
            "enable_clear_path_speed": ("bool", 0, 1),
            "clear_straight_speed": ("float", 0.35, 0.95),
            "clear_preview_dist_m": ("float", 0.45, 1.50),
            "max_accel_mps2": ("float", 0.3, 2.5),
            "max_decel_mps2": ("float", 0.8, 4.0),
            "active_decel_kp": ("float", 0.0, 1.2),
        }

        for map_name in ("finals_fork", "finals_roundabout"):
            payload = panel_server.load_params(map_name)
            self.assertIn("presets", payload)
            preset_ids = {p["id"] for p in payload["presets"]}
            self.assertTrue({"run1_safe", "run2_fast", "legacy_compat"}.issubset(preset_ids))

            seen_keys = []
            by_key = {}
            tier_by_key = {}
            for g in payload["groups"]:
                for it in g["items"]:
                    seen_keys.append(it["key"])
                    by_key[it["key"]] = it
                    tier_by_key[it["key"]] = g["tier"]
            self.assertEqual(len(seen_keys), len(set(seen_keys)), f"Duplicate keys in {map_name}")

            for k, (typ, lo, hi) in expected_specs.items():
                self.assertIn(k, by_key, f"Missing brake/speed param '{k}' in {map_name}")
                self.assertEqual(tier_by_key[k], "calibration")
                it = by_key[k]
                self.assertEqual(it["type"], typ)
                if typ == "float":
                    self.assertAlmostEqual(it["min"], lo, places=6)
                    self.assertAlmostEqual(it["max"], hi, places=6)

        # Save, range validation, and preset apply in temp config dir
        orig_dir = panel_server.CONFIG_DIR
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(REPO_ROOT / "tools" / "config" / "finals_fork.yaml", Path(tmp) / "finals_fork.yaml")
            panel_server.CONFIG_DIR = tmp
            try:
                # Reject out-of-range values
                for bad_kv in (
                    {"brake_damping_gain": 2.5},
                    {"brake_zero_cutoff_mps": 0.001},
                    {"clear_straight_speed": 1.20},
                    {"clear_preview_dist_m": 0.20},
                    {"max_accel_mps2": 3.0},
                    {"max_decel_mps2": 0.4},
                    {"active_decel_kp": 1.8},
                ):
                    ok_bad, _ = panel_server.save_params("finals_fork", {"line_follower_node": bad_kv})
                    self.assertFalse(ok_bad, f"Expected range rejection for {bad_kv}")

                # Save valid values and verify reload round-trip
                ok_good, msg_good = panel_server.save_params("finals_fork", {
                    "line_follower_node": {
                        "enable_sqrt_brake_envelope": True,
                        "enable_closed_loop_brake": True,
                        "brake_damping_gain": 1.15,
                        "brake_zero_cutoff_mps": 0.03,
                        "enable_clear_path_speed": True,
                        "clear_straight_speed": 0.74,
                        "clear_preview_dist_m": 1.10,
                        "max_accel_mps2": 1.10,
                        "max_decel_mps2": 2.80,
                        "active_decel_kp": 0.60,
                    }
                })
                self.assertTrue(ok_good, msg_good)
                reloaded = panel_server.load_params("finals_fork")
                kv = {it["key"]: it["value"] for g in reloaded["groups"] for it in g["items"]}
                self.assertAlmostEqual(kv["brake_damping_gain"], 1.15, places=6)
                self.assertAlmostEqual(kv["brake_zero_cutoff_mps"], 0.03, places=6)
                self.assertAlmostEqual(kv["clear_straight_speed"], 0.74, places=6)
                self.assertAlmostEqual(kv["active_decel_kp"], 0.60, places=6)

                # Apply preset run2_fast and legacy_compat
                ok_p1, msg_p1 = panel_server.apply_preset("finals_fork", "run2_fast")
                self.assertTrue(ok_p1, msg_p1)
                kv1 = {it["key"]: it["value"] for g in panel_server.load_params("finals_fork")["groups"] for it in g["items"]}
                self.assertAlmostEqual(kv1["clear_straight_speed"], 0.78, places=6)
                self.assertAlmostEqual(kv1["max_accel_mps2"], 1.20, places=6)

                ok_p2, msg_p2 = panel_server.apply_preset("finals_fork", "legacy_compat")
                self.assertTrue(ok_p2, msg_p2)
                kv2 = {it["key"]: it["value"] for g in panel_server.load_params("finals_fork")["groups"] for it in g["items"]}
                self.assertFalse(kv2["enable_sqrt_brake_envelope"])
                self.assertFalse(kv2["enable_closed_loop_brake"])
                self.assertFalse(kv2["enable_clear_path_speed"])

                ok_unk, _ = panel_server.apply_preset("finals_fork", "nonexistent_preset")
                self.assertFalse(ok_unk)
            finally:
                panel_server.CONFIG_DIR = orig_dir

    def test_brake_speed_dec_telemetry_parsing_and_status_api(self):
        # 1. Direct [DEC] line parser covers short and canonical keys, units, and ignores invalid values
        p1 = panel_server.parse_dec_telemetry_line(
            "[INFO] [123.4] [line_follower]: [DEC] ctrl mode=straight v=0.58 "
            "v_safe_cap=0.684 clear_score=0.915 brake_cmd=-0.145 a_eff=1.782"
        )
        self.assertAlmostEqual(p1["v_safe_cap_mps"], 0.684, places=3)
        self.assertAlmostEqual(p1["clear_path_score"], 0.915, places=3)
        self.assertAlmostEqual(p1["brake_cmd_mps"], -0.145, places=3)
        self.assertAlmostEqual(p1["effective_brake_decel_mps2"], 1.782, places=3)

        p2 = panel_server.parse_dec_telemetry_line(
            "v_safe_cap_mps=0.62m/s clear_path_score=0.75 brake_cmd_mps=0.00m/s effective_brake_decel_mps2=1.90m/s2"
        )
        self.assertAlmostEqual(p2["v_safe_cap_mps"], 0.62, places=3)
        self.assertAlmostEqual(p2["clear_path_score"], 0.75, places=3)
        self.assertAlmostEqual(p2["brake_cmd_mps"], 0.0, places=3)
        self.assertAlmostEqual(p2["effective_brake_decel_mps2"], 1.9, places=3)

        # Non-[DEC] lines are skipped when require_dec_tag=True
        agg = panel_server.extract_brake_speed_telemetry([
            "plain debug line v_safe_cap=9.99 clear_score=0.11",
            "[DEC] ctrl v_safe_cap=0.65 clear_score=0.88",
            "[DEC] yellow brake_begin brake_cmd=-0.21 a_eff=1.75",
        ])
        self.assertAlmostEqual(agg["v_safe_cap_mps"], 0.65, places=3)
        self.assertAlmostEqual(agg["clear_path_score"], 0.88, places=3)
        self.assertAlmostEqual(agg["brake_cmd_mps"], -0.21, places=3)
        self.assertAlmostEqual(agg["effective_brake_decel_mps2"], 1.75, places=3)

        # Missing fields gracefully remain None
        empty_agg = panel_server.extract_brake_speed_telemetry(["[DEC] ctrl mode=straight v=0.40"])
        for k in ("v_safe_cap_mps", "clear_path_score", "brake_cmd_mps", "effective_brake_decel_mps2"):
            self.assertIsNone(empty_agg[k])

        # 2. Integration with read_diagnostics & /api/status (events.csv + follow.log)
        orig_logroot = panel_server.LOGROOT
        orig_state_dir = panel_server.STATE_DIR
        with tempfile.TemporaryDirectory() as tmp:
            run_id = "test_bs_telem"
            log_dir = Path(tmp) / "logs" / run_id
            state_dir = Path(tmp) / "state"
            log_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "status.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
            (log_dir / "events.csv").write_text(
                "ts,t,src,area,event,kv\n"
                "100.1,1.0,follow,ctrl,log,mode=straight v=0.55 v_safe_cap=0.71 clear_score=0.86 a_eff=1.82\n",
                encoding="utf-8",
            )
            (log_dir / "follow.log").write_text(
                "[DEC] yellow brake_begin brake_cmd=-0.185 a_eff=1.790\n",
                encoding="utf-8",
            )
            panel_server.LOGROOT = str(Path(tmp) / "logs")
            panel_server.STATE_DIR = str(state_dir)
            try:
                diag = panel_server.read_diagnostics()
                bs = diag["brake_speed"]
                self.assertAlmostEqual(bs["v_safe_cap_mps"], 0.71, places=3)
                self.assertAlmostEqual(bs["clear_path_score"], 0.86, places=3)
                self.assertAlmostEqual(bs["brake_cmd_mps"], -0.185, places=3)
                self.assertAlmostEqual(bs["effective_brake_decel_mps2"], 1.79, places=3)
                # ctrl rows must still not pollute milestones
                self.assertEqual(diag["milestones"], [])

                import io
                buf = io.BytesIO()
                h = panel_server.Handler.__new__(panel_server.Handler)
                h.path = "/api/status"
                h.headers = {}
                h.wfile = buf
                h.requestline = "GET /api/status HTTP/1.1"
                h.request_version = "HTTP/1.1"
                h.client_address = ("127.0.0.1", 12345)
                captured = {"status": 0}
                h.send_response = lambda code, msg=None: captured.__setitem__("status", code)
                h.send_header = lambda k, v: None
                h.end_headers = lambda: None
                h.do_GET()
                self.assertEqual(captured["status"], 200)
                body = json.loads(buf.getvalue().decode("utf-8"))
                self.assertIn("brake_speed", body)
                self.assertAlmostEqual(body["brake_speed"]["v_safe_cap_mps"], 0.71, places=3)
                self.assertAlmostEqual(body["brake_speed"]["brake_cmd_mps"], -0.185, places=3)
            finally:
                panel_server.LOGROOT = orig_logroot
                panel_server.STATE_DIR = orig_state_dir

        # 3. Panel HTML surfaces compact live telemetry badges with graceful '-' fallback & preset UI
        html = (REPO_ROOT / "tools" / "panel" / "index.html").read_text(encoding="utf-8")
        for token in (
            "brakeSpeedStrip", "bsVSafeCap", "bsClearScore", "bsBrakeCmd", "bsAEff",
            "d.brake_speed", "v_safe_cap_mps", "clear_path_score",
            "brake_cmd_mps", "effective_brake_decel_mps2",
            "presetSel", "btnApplyPreset",
        ):
            self.assertIn(token, html, f"Missing UI hook '{token}' in tools/panel/index.html")


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

    def test_camera_model_pinhole_identity_pitch_jitter_and_homography_roundtrip(self):
        # 1. Default pinhole CameraModel(h_mm=110.0, alpha_deg=12.5, f_px=325.0) 100% backward compatible
        cam_default = sim_render.CameraModel()
        cam_explicit = sim_render.CameraModel(
            h_mm=110.0, alpha_deg=12.5, f_px=325.0, width=640, height=480, pitch_jitter_deg=0.0
        )
        self.assertFalse(cam_default.use_homography)
        self.assertAlmostEqual(cam_default.effective_alpha_deg, 12.5, places=9)
        for w_mm, d_mm in ((0.0, 350.0), (-150.0, 600.0), (215.0, 1000.0)):
            u0, v0 = cam_default.wd_to_uv(w_mm, d_mm)
            u1, v1 = cam_explicit.wd_to_uv(w_mm, d_mm)
            self.assertAlmostEqual(u0, u1, places=9)
            self.assertAlmostEqual(v0, v1, places=9)
            w_back, d_back = cam_default.uv_to_wd(u0, v0)
            self.assertAlmostEqual(w_back, w_mm, places=5)
            self.assertAlmostEqual(d_back, d_mm, places=5)

        # 2. pitch_jitter_deg shifts projection identically to alpha_deg + pitch_jitter_deg
        cam_jit_pos = sim_render.CameraModel(pitch_jitter_deg=1.5)
        cam_alpha_pos = sim_render.CameraModel(alpha_deg=14.0)
        cam_jit_neg = sim_render.CameraModel(pitch_jitter_deg=-1.5)
        for w_mm, d_mm in ((0.0, 500.0), (180.0, 800.0)):
            u_def, v_def = cam_default.wd_to_uv(w_mm, d_mm)
            u_jp, v_jp = cam_jit_pos.wd_to_uv(w_mm, d_mm)
            u_ap, v_ap = cam_alpha_pos.wd_to_uv(w_mm, d_mm)
            _u_jn, v_jn = cam_jit_neg.wd_to_uv(w_mm, d_mm)
            self.assertAlmostEqual(u_jp, u_ap, places=9)
            self.assertAlmostEqual(v_jp, v_ap, places=9)
            # Positive downward pitch moves ground point UP in image (smaller v)
            self.assertLess(v_jp, v_def - 5.0)
            self.assertGreater(v_jn, v_def + 5.0)
        self.assertLess(cam_jit_pos.distance_at_row(300), cam_default.distance_at_row(300))
        self.assertGreater(cam_jit_neg.distance_at_row(300), cam_default.distance_at_row(300))

        # 3. CameraModel.from_homography_matrix round-trip (img_to_ground & ground_to_img + pitch_jitter_deg)
        overhang_m = 0.06
        ground_pts = [
            (0.30, 0.215), (0.30, -0.215),
            (0.60, 0.215), (0.60, -0.215),
            (1.00, 0.215), (1.00, -0.215),
        ]
        img_pts = [
            cam_default.wd_to_uv(w_mm=-y_m * 1000.0, d_mm=(d_m + overhang_m) * 1000.0)
            for d_m, y_m in ground_pts
        ]
        solved = calib_mod.solve_homography(img_pts, ground_pts)
        H_i2g = solved["H"]
        H_g2i = solved["H_inv"]

        cam_h_i2g = sim_render.CameraModel.from_homography_matrix(
            solved["H_flat"], width=640, height=480, pitch_jitter_deg=0.0, direction="img_to_ground"
        )
        cam_h_g2i = sim_render.CameraModel.from_homography_matrix(
            H_g2i, width=640, height=480, pitch_jitter_deg=0.0, direction="ground_to_img"
        )
        self.assertTrue(cam_h_i2g.use_homography)
        self.assertTrue(cam_h_g2i.use_homography)

        for d_m, y_m in ((0.35, 0.12), (0.75, -0.18), (1.10, 0.0)):
            u_ref, v_ref = cam_default.ground_to_pixel_m(d_m, y_m)
            for ch in (cam_h_i2g, cam_h_g2i):
                u_h, v_h = ch.ground_to_pixel_m(d_m, y_m)
                self.assertAlmostEqual(u_h, u_ref, places=2)
                self.assertAlmostEqual(v_h, v_ref, places=2)
                d_back, y_back = ch.pixel_to_ground_m(u_h, v_h)
                self.assertAlmostEqual(d_back, d_m, places=4)
                self.assertAlmostEqual(y_back, y_m, places=4)
                # wd_to_uv <-> uv_to_wd round-trip in mm
                w_mm = -y_m * 1000.0
                d_mm = (d_m + overhang_m) * 1000.0
                u_wd, v_wd = ch.wd_to_uv(w_mm, d_mm)
                w_rt, d_rt = ch.uv_to_wd(u_wd, v_wd)
                self.assertAlmostEqual(w_rt, w_mm, places=2)
                self.assertAlmostEqual(d_rt, d_mm, places=2)

        # Sky pixels above horizon cleanly yield inf / NaN
        self.assertEqual(cam_h_i2g.distance_at_row(50), float("inf"))
        self.assertAlmostEqual(cam_h_i2g.distance_at_row(300), cam_default.distance_at_row(300), delta=2.0)

        # Homography + pitch_jitter_deg matches true 3D GroundProjector pitch rotation to sub-pixel precision,
        # and matches pinhole vertical v-shift identically
        def _gp_pinhole_uv(d_m, y_m, pitch_deg):
            a = math.radians(pitch_deg)
            d_cam = d_m + overhang_m
            xc = -y_m
            yc = -d_cam * math.sin(a) + 0.11 * math.cos(a)
            zc = d_cam * math.cos(a) + 0.11 * math.sin(a)
            return 320.0 + 325.0 * (xc / zc), 240.0 + 325.0 * (yc / zc)

        gp_img_pts = [_gp_pinhole_uv(d_m, y_m, 12.5) for d_m, y_m in ground_pts]
        H_gp = calib_mod.solve_homography(gp_img_pts, ground_pts)["H"]
        cam_gp_jit = sim_render.CameraModel.from_homography_matrix(
            H_gp, width=640, height=480, pitch_jitter_deg=1.5
        )
        for d_m, y_m in ((0.40, 0.15), (0.80, -0.10)):
            u_true, v_true = _gp_pinhole_uv(d_m, y_m, 14.0)
            u_hj, v_hj = cam_gp_jit.ground_to_pixel_m(d_m, y_m)
            self.assertAlmostEqual(u_hj, u_true, delta=0.05)
            self.assertAlmostEqual(v_hj, v_true, delta=0.05)

        cam_h_jit = sim_render.CameraModel.from_homography_matrix(
            H_i2g, width=640, height=480, pitch_jitter_deg=1.5
        )
        for d_m, y_m in ((0.40, 0.15), (0.80, -0.10)):
            _u_pj, v_pj = cam_jit_pos.ground_to_pixel_m(d_m, y_m)
            _u_hj, v_hj = cam_h_jit.ground_to_pixel_m(d_m, y_m)
            self.assertAlmostEqual(v_hj, v_pj, delta=0.05)

        # TrackRenderer integration with homography camera and pitch jitter
        track_json = REPO_ROOT / "tools" / "sim" / "tracks" / "fork_s.json"
        r_pin = sim_render.TrackRenderer(str(track_json), cam=cam_default)
        r_hom = sim_render.TrackRenderer(str(track_json), cam=cam_h_i2g)
        r_jit = sim_render.TrackRenderer(str(track_json), pitch_jitter_deg=2.0)
        x0, y0 = r_pin.meta["centerline_mm"][0]
        x1, y1 = r_pin.meta["centerline_mm"][2]
        yaw = math.atan2(y1 - y0, x1 - x0)
        img_pin = r_pin.render(x0, y0, yaw)
        img_hom = r_hom.render(x0, y0, yaw)
        img_jit = r_jit.render(x0, y0, yaw)
        self.assertEqual(img_hom.shape, (480, 640, 3))
        self.assertLess(float(np.mean(np.abs(img_pin.astype(np.float32) - img_hom.astype(np.float32)))), 1.0)
        self.assertGreater(float(np.mean(np.abs(img_pin.astype(np.float32) - img_jit.astype(np.float32)))), 2.0)

        # Reject invalid / singular homographies cleanly
        with self.assertRaises(ValueError):
            sim_render.CameraModel.from_homography_matrix(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            sim_render.CameraModel.from_homography_matrix([1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            sim_render.CameraModel.from_homography_matrix(H_i2g, direction="bad_dir")


if __name__ == "__main__":
    unittest.main(verbosity=2)
