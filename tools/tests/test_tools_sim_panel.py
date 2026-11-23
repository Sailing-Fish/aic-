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
import params_schema  # noqa: E402
import record_and_compare_cv as cv_cmp  # noqa: E402
import render as sim_render  # noqa: E402
import server as panel_server  # noqa: E402
import synth_finals_dataset as synth_mod  # noqa: E402
import tune as sim_tune  # noqa: E402
import world_node as sim_world  # noqa: E402


class TestYamlAndSchemaParity(unittest.TestCase):
    YAML_FILES = [
        REPO_ROOT / "tools" / "config" / "fork.yaml",
        REPO_ROOT / "tools" / "config" / "roundabout.yaml",
        REPO_ROOT / "tools" / "config" / "finals_fork.yaml",
        REPO_ROOT / "tools" / "config" / "finals_roundabout.yaml",
        REPO_ROOT / "src" / "line_follow_ws" / "src" / "line_follower" / "config" / "line_follower_params.yaml",
    ]

    def test_all_yamls_parse_cleanly(self):
        for p in self.YAML_FILES:
            self.assertTrue(p.is_file(), f"Missing YAML: {p}")
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            self.assertIsInstance(data, dict, f"YAML root not dict: {p}")
            self.assertIn("line_follower_node", data, f"Missing line_follower_node in {p}")
            self.assertIn("cmd_mux_node", data, f"Missing cmd_mux_node in {p}")

    def test_legacy_vs_finals_mode_defaults(self):
        for name in ("fork.yaml", "roundabout.yaml"):
            data = yaml.safe_load((REPO_ROOT / "tools" / "config" / name).read_text(encoding="utf-8"))
            lf = data["line_follower_node"]["ros__parameters"]
            mux = data["cmd_mux_node"]["ros__parameters"]
            self.assertEqual(lf["planner_mode"], "legacy_segment")
            self.assertEqual(lf["lane_seg_mode"], "legacy_hsv")
            self.assertEqual(lf["yellow_detect_mode"], "legacy_area")
            self.assertFalse(lf["enable_semantic_hole_fill"])
            self.assertTrue(mux["enable_detection_override"])

        for name in ("finals_fork.yaml", "finals_roundabout.yaml"):
            data = yaml.safe_load((REPO_ROOT / "tools" / "config" / name).read_text(encoding="utf-8"))
            lf = data["line_follower_node"]["ros__parameters"]
            mux = data["cmd_mux_node"]["ros__parameters"]
            self.assertEqual(lf["planner_mode"], "finals_corridor")
            self.assertEqual(lf["lane_seg_mode"], "relative_border")
            self.assertEqual(lf["yellow_detect_mode"], "geometric_cv")
            self.assertTrue(lf["enable_semantic_hole_fill"])
            self.assertTrue(lf["enable_obstacle_memory"])
            self.assertTrue(lf["yellow_require_odom_still"])
            self.assertTrue(lf["lost_line_safe_mode"])
            self.assertFalse(mux["enable_detection_override"])
            self.assertEqual(len(lf["homography_matrix"]), 9)

    def test_schema_keys_exist_in_cpp_declarations(self):
        lf_cpp = (
            REPO_ROOT / "src" / "line_follow_ws" / "src" / "line_follower" / "src" / "line_follower_node.cpp"
        ).read_text(encoding="utf-8")
        mux_cpp = (
            REPO_ROOT / "src" / "line_follow_ws" / "src" / "line_follower" / "src" / "cmd_mux_node.cpp"
        ).read_text(encoding="utf-8")

        lf_declared = set(re.findall(r'declare_parameter(?:<[^>]+>)?\(\s*"([^"]+)"', lf_cpp))
        # Also include segN_<key> generated dynamically in loop
        seg_base = set(re.findall(r'decl_[db]\(\s*"([^"]+)"', lf_cpp))
        for i in range(1, 5):
            for k in seg_base:
                lf_declared.add(f"seg{i}_{k}")
        mux_declared = set(re.findall(r'declare_parameter(?:<[^>]+>)?\(\s*"([^"]+)"', mux_cpp))

        for grp in params_schema.SCHEMA:
            sec = grp["section"]
            if sec == "line_follower_node":
                for item in grp["items"]:
                    self.assertIn(item[0], lf_declared, f"Schema key '{item[0]}' not declared in line_follower_node.cpp")
            elif sec == "cmd_mux_node":
                for item in grp["items"]:
                    self.assertIn(item[0], mux_declared, f"Schema key '{item[0]}' not declared in cmd_mux_node.cpp")


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
    def test_load_params_all_four_maps(self):
        required_keys = {
            "planner_mode", "lane_seg_mode", "yellow_detect_mode",
            "enable_semantic_hole_fill", "enable_obstacle_memory",
            "allow_dual_digit_cover", "yellow_h_low", "yellow_h_high",
            "yellow_s_min", "yellow_v_min", "red_h_low_1", "red_s_min",
            "enable_detection_override",
        }
        for map_name in ("fork", "roundabout", "finals_fork", "finals_roundabout"):
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

    def test_panel_html_has_voltage_and_finals_maps(self):
        html = (REPO_ROOT / "tools" / "panel" / "index.html").read_text(encoding="utf-8")
        self.assertIn("voltBadge", html)
        self.assertIn("11.6", html)
        self.assertIn("finals_fork", html)
        self.assertIn("finals_roundabout", html)
        self.assertIn("it.type==='enum'", html)


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
