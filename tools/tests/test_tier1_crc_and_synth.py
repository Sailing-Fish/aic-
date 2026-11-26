#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tier 1-④ & 参数契约独立单元测试：
1. tools/finals/calibrate_crc.py（按 session 无泄漏切分、合成路径拦截、保召回阈值、框底 CRC 外扩、YOLO 目录与 CLI）
2. tools/finals/synth_finals_dataset.py（--max-digit-dist-m 1.15 封顶、眩光/泛黄/LED 色温退化增强、禁翻转守卫）
3. Tier 1 YAML / C++ / Panel Schema 参数 100% 对齐与安全默认中性值校验
"""
from __future__ import annotations

import json
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "finals"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "panel"))
sys.path.insert(0, str(REPO_ROOT / "src" / "yolov8"))

import bpu_yolov8_node as yolo_mod  # noqa: E402
import calibrate_crc as crc_mod  # noqa: E402
import params_schema  # noqa: E402
import server as panel_server  # noqa: E402
import synth_finals_dataset as synth_mod  # noqa: E402


def _make_crc_records(num_sessions: int = 5, frames_per_session: int = 12, seed: int = 42):
    """构造多场次真实样式检测校验集（含安全类与数字类、已知框底向上误差与低分误报）。"""
    rng = random.Random(seed)
    records = []
    for s_idx in range(num_sessions):
        sid = f"real_hall_run_{s_idx:02d}"
        for f_idx in range(frames_per_session):
            gts = []
            preds = []

            # 1) red_no_entry：TP 置信度 0.22~0.88，框底向上误差 1.0~8.0px（含少量向下负误差 -> 钳位 0）
            ne_gt = [200.0, 260.0, 320.0, 320.0]
            gts.append({"cls": "red_no_entry", "box": ne_gt})
            ne_up_err = rng.uniform(-2.0, 8.0)
            ne_pred = [201.0, 260.0, 319.0, ne_gt[3] - ne_up_err]
            preds.append({"cls": "red_no_entry", "conf": round(rng.uniform(0.22, 0.88), 4), "box": ne_pred})
            # 低分 FP（在远处其他位置）
            if f_idx % 3 == 0:
                preds.append({"cls": "red_no_entry", "conf": 0.11, "box": [20.0, 150.0, 60.0, 180.0]})

            # 2) red_cone：TP 置信度 0.28~0.92，框底向上误差 0.5~5.0px
            cone_gt = [400.0, 220.0, 450.0, 340.0]
            gts.append({"cls": "red_cone", "box": cone_gt})
            cone_up_err = rng.uniform(-1.0, 5.0)
            cone_pred = [401.0, 221.0, 449.0, cone_gt[3] - cone_up_err]
            preds.append({"cls": "red_cone", "conf": round(rng.uniform(0.28, 0.92), 4), "box": cone_pred})

            # 3) yellow_stop_line：TP 置信度 0.20~0.85，框底向上误差 2.0~10.0px
            yel_gt = [120.0, 370.0, 520.0, 400.0]
            gts.append({"cls": "yellow_stop_line", "box": yel_gt})
            yel_up_err = rng.uniform(0.0, 10.0)
            yel_pred = [122.0, 370.0, 518.0, yel_gt[3] - yel_up_err]
            preds.append({"cls": "yellow_stop_line", "conf": round(rng.uniform(0.20, 0.85), 4), "box": yel_pred})

            # 4) Bonus 数字 '6'：高分 TP (0.65~0.95) + 中低分误报 FP (0.18~0.42)
            dig_gt = [280.0, 300.0, 330.0, 340.0]
            gts.append({"cls": "6", "box": dig_gt})
            preds.append({"cls": "6", "conf": round(rng.uniform(0.65, 0.95), 4), "box": [281.0, 300.0, 329.0, 339.0]})
            preds.append({"cls": "6", "conf": round(rng.uniform(0.18, 0.42), 4), "box": [10.0, 10.0, 45.0, 40.0]})

            records.append({
                "session_id": sid,
                "frame_id": f"{sid}_f{f_idx:04d}",
                "image_width": 640,
                "image_height": 480,
                "gt": gts,
                "preds": preds,
            })
    return records


class TestCalibrateCrc(unittest.TestCase):
    def test_synthetic_path_rejected_by_default_and_overridable(self):
        with tempfile.TemporaryDirectory() as tmp:
            synth_dir = Path(tmp) / "dataset_synth_8cls"
            synth_dir.mkdir(parents=True, exist_ok=True)
            json_p = synth_dir / "val_records.json"
            json_p.write_text(json.dumps(_make_crc_records(3, 4)), encoding="utf-8")

            self.assertTrue(crc_mod.is_synthetic_path(json_p))
            with self.assertRaises(ValueError):
                crc_mod.load_records_json(json_p)

            # 显式放行仅用于自动化测试
            recs = crc_mod.load_records_json(json_p, allow_synthetic=True)
            self.assertEqual(len(recs), 12)

            # YOLO 目录同样拦截 synth 路径
            gt_d = synth_dir / "gt"
            pr_d = synth_dir / "pred"
            gt_d.mkdir()
            pr_d.mkdir()
            with self.assertRaises(ValueError):
                crc_mod.load_records_yolo_dirs(gt_d, pr_d)

    def test_strict_session_split_no_leakage_and_guards(self):
        recs = _make_crc_records(num_sessions=5, frames_per_session=10, seed=7)
        cal, val, cal_s, val_s = crc_mod.split_by_session(recs, calib_ratio=0.6, seed=2026)

        self.assertEqual(len(cal_s), 3)
        self.assertEqual(len(val_s), 2)
        self.assertTrue(set(cal_s).isdisjoint(set(val_s)))
        self.assertEqual({r["session_id"] for r in cal}, set(cal_s))
        self.assertEqual({r["session_id"] for r in val}, set(val_s))
        self.assertEqual(len(cal) + len(val), len(recs))

        # 单场次必须拒绝，避免同视频相邻帧泄漏
        one_session = [r for r in recs if r["session_id"] == cal_s[0]]
        with self.assertRaises(ValueError):
            crc_mod.split_by_session(one_session)

        # 缺失 session_id 必须拒绝
        with tempfile.TemporaryDirectory() as tmp:
            bad_json = Path(tmp) / "real_no_session.json"
            bad_json.write_text(json.dumps([{"image_height": 480, "gt": [], "preds": []}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                crc_mod.load_records_json(bad_json)

    def test_conformal_quantile_exact_finite_sample_formula(self):
        # n = 19, alpha = 0.05 -> k = ceil(20 * 0.95) = 19 -> max element
        vals19 = [float(i) for i in range(1, 20)]
        self.assertAlmostEqual(crc_mod.conformal_quantile(vals19, alpha=0.05), 19.0, places=6)
        # n = 99, alpha = 0.05 -> k = ceil(100 * 0.95) = 95 -> 95.0
        vals99 = [float(i) for i in range(1, 100)]
        self.assertAlmostEqual(crc_mod.conformal_quantile(vals99, alpha=0.05), 95.0, places=6)
        # Empty -> 0.0
        self.assertAlmostEqual(crc_mod.conformal_quantile([], alpha=0.05), 0.0, places=6)

    def test_end_to_end_recall_thresholds_crc_pads_and_yaml_parity(self):
        recs = _make_crc_records(num_sessions=6, frames_per_session=20, seed=123)
        res = crc_mod.calibrate_crc(recs, calib_ratio=0.6, seed=2026, alpha=0.05)

        self.assertFalse(res["split"]["session_leakage"])
        self.assertTrue(set(res["split"]["calib_sessions"]).isdisjoint(set(res["split"]["val_sessions"])))

        # 安全类保召回校验
        for cls_name, min_r in (("red_no_entry", 0.98), ("yellow_stop_line", 0.98), ("red_cone", 0.95)):
            info = res["per_class"][cls_name]
            self.assertGreaterEqual(info["calib"]["recall"], min_r)
            self.assertGreaterEqual(info["val"]["recall"], min_r - 0.05)

        # 数字类兼顾精确率：滤除 0.18~0.42 的误报同时保留 0.65~0.95 真值
        d6 = res["per_class"]["6"]
        self.assertGreater(d6["threshold"], 0.42)
        self.assertLessEqual(d6["threshold"], 0.68)
        self.assertGreaterEqual(d6["val"]["precision"], 0.95)
        self.assertGreaterEqual(d6["val"]["recall"], 0.95)

        # 框底向上残差外扩应为正值且在验证集上达到高覆盖率
        pads = res["crc_pads"]
        self.assertGreater(pads["crc_no_entry_pad_y_px"], 4.0)
        self.assertGreater(pads["crc_cone_pad_y_px"], 2.0)
        self.assertGreater(pads["crc_yellow_pad_y_px"], 5.0)
        for pkey in ("crc_no_entry_pad_y_px", "crc_cone_pad_y_px", "crc_yellow_pad_y_px"):
            pv = res["pad_validation"][pkey]
            self.assertIsNotNone(pv["val_coverage"])
            self.assertGreaterEqual(pv["val_coverage"], 0.85)

        # 输出的 class_conf_overrides 可直接被 bpu_yolov8_node 解析
        cls11 = list(crc_mod.ALL_CLASSES_11)
        parsed_arr = yolo_mod.parse_class_conf_overrides(
            res["class_conf_overrides_json"], cls11, default_conf=0.25
        )
        self.assertAlmostEqual(float(parsed_arr[9]), res["class_conf_overrides_dict"]["red_no_entry"], places=4)
        self.assertAlmostEqual(float(parsed_arr[5]), res["class_conf_overrides_dict"]["6"], places=4)

        # 输出 YAML 片段可干净解析
        snippet = crc_mod.format_yaml_snippet(res)
        parsed_yaml = yaml.safe_load(snippet)
        lf_p = parsed_yaml["line_follower_node"]["ros__parameters"]
        yolo_p = parsed_yaml["bpu_yolov8_digit_node"]["ros__parameters"]
        self.assertAlmostEqual(lf_p["crc_no_entry_pad_y_px"], pads["crc_no_entry_pad_y_px"], places=2)
        self.assertAlmostEqual(lf_p["crc_cone_pad_y_px"], pads["crc_cone_pad_y_px"], places=2)
        self.assertAlmostEqual(lf_p["crc_yellow_pad_y_px"], pads["crc_yellow_pad_y_px"], places=2)
        self.assertIsInstance(yolo_p["class_conf_overrides"], str)
        self.assertEqual(json.loads(yolo_p["class_conf_overrides"]), res["class_conf_overrides_dict"])

    def test_yolo_txt_directory_mode_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "real_val_capture"
            gt_dir = root / "gt"
            pred_dir = root / "pred"
            for sid in ("session_a", "session_b", "session_c"):
                (gt_dir / sid).mkdir(parents=True, exist_ok=True)
                (pred_dir / sid).mkdir(parents=True, exist_ok=True)
                for f in range(5):
                    stem = f"frame_{f:03d}.txt"
                    # cls 9 = red_no_entry; GT bottom cy+bh/2 = 0.70 (v=336px)
                    (gt_dir / sid / stem).write_text("9 0.50 0.65 0.20 0.10\n", encoding="utf-8")
                    # Pred bottom cy+bh/2 = 0.69 (v=331.2px -> upward residual = 4.8px)
                    (pred_dir / sid / stem).write_text("9 0.50 0.64 0.20 0.10 0.55\n", encoding="utf-8")

            out_yaml = root / "crc_out.yaml"
            out_json = root / "crc_report.json"
            rc = crc_mod.main([
                "--gt-dir", str(gt_dir),
                "--pred-dir", str(pred_dir),
                "--classes", "11",
                "--out-yaml", str(out_yaml),
                "--out-json", str(out_json),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_yaml.is_file())
            self.assertTrue(out_json.is_file())
            rep = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertAlmostEqual(rep["crc_pads"]["crc_no_entry_pad_y_px"], 4.8, places=1)


class TestSynthTier1Upgrades(unittest.TestCase):
    def test_max_digit_dist_m_defaults_and_strict_enforcement(self):
        self.assertAlmostEqual(synth_mod.resolve_max_digit_dist_m(8, None), 1.15, places=6)
        self.assertAlmostEqual(synth_mod.resolve_max_digit_dist_m(11, None), 1.55, places=6)
        self.assertAlmostEqual(synth_mod.resolve_max_digit_dist_m(8, 0.95), 0.95, places=6)
        with self.assertRaises(ValueError):
            synth_mod.resolve_max_digit_dist_m(8, 0.10)
        with self.assertRaises(ValueError):
            synth_mod.resolve_max_digit_dist_m(8, 3.00)

        # CLI 解析校验
        ns8 = synth_mod.parse_args(["--out", "tmp_out", "--classes", "8"])
        self.assertIsNone(ns8.max_digit_dist_m)
        ns_custom = synth_mod.parse_args(["--out", "tmp_out", "--classes", "8", "--max-digit-dist-m", "1.15"])
        self.assertAlmostEqual(ns_custom.max_digit_dist_m, 1.15, places=6)

        # Spy on project_ground_prop_bgra to verify EVERY digit placement in 8-class mode obeys d_m <= 1.15
        recorded_digit_dists = []
        orig_proj = synth_mod.project_ground_prop_bgra

        def spy_proj(bg, prop_bgra, d_m, y_m, phys_w_m, phys_h_m, *args, **kwargs):
            # 数字贴纸的 phys_w_m / phys_h_m 均在 [0.09, 0.23] 且不是 0.24x0.13 红禁行框 / 0.28+ 黄线
            if abs(phys_w_m - 0.24) > 1e-6 and phys_w_m < 0.25:
                recorded_digit_dists.append(float(d_m))
            return orig_proj(bg, prop_bgra, d_m, y_m, phys_w_m, phys_h_m, *args, **kwargs)

        synth_mod.project_ground_prop_bgra = spy_proj
        try:
            rng = random.Random(2026)
            for _ in range(40):
                synth_mod.generate_one_sample(
                    rng, [], {c: [] for c in synth_mod.CLASSES}, num_classes=8
                )
        finally:
            synth_mod.project_ground_prop_bgra = orig_proj

        self.assertGreater(len(recorded_digit_dists), 20)
        self.assertLessEqual(
            max(recorded_digit_dists),
            1.15 + 1e-9,
            f"Found digit placed beyond 1.15m cap: max={max(recorded_digit_dists):.3f}m",
        )

    def test_finals_venue_degradation_effects_and_no_flip_guards(self):
        self.assertFalse(synth_mod.ALLOW_HFLIP)
        self.assertFalse(synth_mod.ALLOW_VFLIP)
        self.assertLessEqual(synth_mod.MAX_ROTATION_DEG, 15.0)

        neutral = np.full((240, 320, 3), 160, dtype=np.uint8)
        rng = random.Random(101)

        # 严禁开启翻转
        with self.assertRaises(ValueError):
            synth_mod.apply_finals_venue_degradation(neutral, rng, allow_flip=True)

        # 1) 局部镜面反光高光斑：最大亮度显著高于原图 160
        spec = synth_mod.apply_finals_venue_degradation(
            neutral, random.Random(102), enable_specular=True, enable_yellowing=False, enable_led_temp=False
        )
        self.assertEqual(spec.shape, neutral.shape)
        self.assertEqual(spec.dtype, np.uint8)
        self.assertGreater(int(spec.max()), 200)

        # 2) 赛垫泛黄/老化色偏：地面下半部 B 通道明显低于 R 通道
        yell = synth_mod.apply_finals_venue_degradation(
            neutral, random.Random(103), enable_specular=False, enable_yellowing=True, enable_led_temp=False
        )
        ground = yell[120:, :, :]
        self.assertLess(float(ground[:, :, 0].mean()), float(ground[:, :, 2].mean()) - 15.0)

        # 3) 冷暖 LED 色温漂移：跨多组种子同时产生偏冷(B>R)与偏暖(R>B)
        saw_warm = False
        saw_cool = False
        for s in range(20):
            led = synth_mod.apply_finals_venue_degradation(
                neutral, random.Random(200 + s), enable_specular=False, enable_yellowing=False, enable_led_temp=True
            )
            diff_rb = float(led[:, :, 2].mean()) - float(led[:, :, 0].mean())
            if diff_rb > 8.0:
                saw_warm = True
            if diff_rb < -8.0:
                saw_cool = True
        self.assertTrue(saw_warm, "LED color temp drift should produce warm casts")
        self.assertTrue(saw_cool, "LED color temp drift should produce cool casts")


class TestTier1ConfigAndSchemaParity(unittest.TestCase):
    TIER1_EXPECTED_DEFAULTS = {
        "voltage_topic": "/PowerVoltage",
        "enable_cbf_shield": False,
        "cbf_wheelbase_m": 0.142,
        "cbf_max_steer_deg": 35.0,
        "cbf_horizon_steps": 6,
        "cbf_step_dt_sec": 0.06,
        "cbf_gamma": 0.60,
        "yellow_final_ramp_dist_m": 0.35,
        "yellow_final_creep_speed_mps": 0.18,
        "voltage_decel_slope": 0.0,
        "enable_brake_adaptation": True,
        "enable_risk_aware_decision": True,
        "crc_no_entry_pad_y_px": 0.0,
        "crc_cone_pad_y_px": 0.0,
        "crc_yellow_pad_y_px": 0.0,
    }
    TIER1_PANEL_KEYS = {k for k in TIER1_EXPECTED_DEFAULTS if k != "voltage_topic"}

    def test_tier1_params_100pct_parity_across_cpp_yamls_and_schema(self):
        lf_cpp = (
            REPO_ROOT / "src" / "line_follow_ws" / "src" / "line_follower" / "src" / "line_follower_node.cpp"
        ).read_text(encoding="utf-8")
        lf_declared = set(re.findall(r'declare_parameter(?:<[^>]+>)?\(\s*"([^"]+)"', lf_cpp))

        # 1) 15 项 Tier 1 参数全部在 C++ 中声明
        missing_in_cpp = set(self.TIER1_EXPECTED_DEFAULTS.keys()) - lf_declared
        self.assertFalse(missing_in_cpp, f"Tier 1 params missing in line_follower_node.cpp: {missing_in_cpp}")

        # 2) finals_fork.yaml / finals_roundabout.yaml / line_follower_params.yaml 100% 同步且默认值安全中性
        yaml_paths = [
            REPO_ROOT / "tools" / "config" / "finals_fork.yaml",
            REPO_ROOT / "tools" / "config" / "finals_roundabout.yaml",
            REPO_ROOT / "src" / "line_follow_ws" / "src" / "line_follower" / "config" / "line_follower_params.yaml",
        ]
        for yp in yaml_paths:
            data = yaml.safe_load(yp.read_text(encoding="utf-8"))
            lf = data["line_follower_node"]["ros__parameters"]

            # 0 冗余：YAML 中每一个 line_follower_node 参数均必须在 C++ 中声明
            extra_yaml_keys = set(lf.keys()) - lf_declared
            self.assertFalse(extra_yaml_keys, f"Redundant keys in {yp.name} not in C++: {extra_yaml_keys}")

            # 0 缺漏 + 安全默认中性值校验
            for k, expected in self.TIER1_EXPECTED_DEFAULTS.items():
                self.assertIn(k, lf, f"Missing Tier 1 key '{k}' in {yp.name}")
                actual = lf[k]
                if isinstance(expected, bool):
                    self.assertIs(actual, expected, f"{yp.name}:{k} expected {expected}, got {actual}")
                elif isinstance(expected, float):
                    self.assertAlmostEqual(float(actual), expected, places=6, msg=f"{yp.name}:{k}")
                else:
                    self.assertEqual(actual, expected, f"{yp.name}:{k}")

        # 3) params_schema.py：全部 14 项可调 Tier 1 参数干净归入 calibration (⚙ 开关与标定) 分组，0 重复、0 冗余
        seen_schema_keys = []
        schema_by_key = {}
        tier_by_key = {}
        for grp in params_schema.SCHEMA:
            if grp["section"] != "line_follower_node":
                continue
            for item in grp["items"]:
                k = item[0]
                seen_schema_keys.append(k)
                schema_by_key[k] = item
                tier_by_key[k] = grp["tier"]

        self.assertEqual(len(seen_schema_keys), len(set(seen_schema_keys)), "Duplicate keys in params_schema.SCHEMA")
        extra_schema_keys = set(seen_schema_keys) - lf_declared
        self.assertFalse(extra_schema_keys, f"Schema keys not in C++: {extra_schema_keys}")

        for k in self.TIER1_PANEL_KEYS:
            self.assertIn(k, schema_by_key, f"Tier 1 key '{k}' missing in params_schema.py")
            self.assertEqual(tier_by_key[k], "calibration", f"Tier 1 key '{k}' must live in 'calibration' tier")
            spec = schema_by_key[k]
            typ, lo, hi = spec[2], spec[3], spec[4]
            default_v = self.TIER1_EXPECTED_DEFAULTS[k]
            if typ == "bool":
                self.assertIsInstance(default_v, bool)
            elif typ in ("float", "int"):
                self.assertGreaterEqual(default_v, lo, f"{k} default {default_v} < min {lo}")
                self.assertLessEqual(default_v, hi, f"{k} default {default_v} > max {hi}")

        # 4) Web 面板后端 load_params 对两张决赛地图均可完整加载全部 14 项 Tier 1 参数
        for map_name in ("finals_fork", "finals_roundabout"):
            payload = panel_server.load_params(map_name)
            cal_keys = {
                it["key"]
                for g in payload["groups"]
                if g["tier"] == "calibration"
                for it in g["items"]
            }
            missing_panel = self.TIER1_PANEL_KEYS - cal_keys
            self.assertFalse(missing_panel, f"{map_name} panel missing Tier 1 calibration keys: {missing_panel}")

    def test_panel_tier1_diagnostics_and_html_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_path = Path(tmp) / "events.csv"
            ev_path.write_text(
                "ts_iso,t_rel,source,area,event,kv\n"
                "2026-11-26T20:00:01,1.20,follow,cbf,intervene,src=no_entry v=0.44->0.26 yaw=+0.10->-0.85 clr=0.012\n"
                "2026-11-26T20:00:02,1.70,follow,cbf,intervene,src=no_entry v=0.40->0.24 yaw=+0.05->-0.75 clr=0.015\n"
                "2026-11-26T20:00:03,2.30,follow,cbf,intervene,src=lane_border v=0.30->0.00 yaw=+0.00->+0.40 clr=0.000\n"
                "2026-11-26T20:00:06,5.80,follow,brake_adapt,state,v_in=0.220 travel=0.0310 scale=1.000->0.946 a_eff=1.701 vbat=12.10\n",
                encoding="utf-8",
            )
            orig_run_dir = panel_server.run_dir
            try:
                panel_server.run_dir = lambda: tmp
                diag = panel_server.read_diagnostics()
            finally:
                panel_server.run_dir = orig_run_dir

            t1 = diag["tier1"]
            self.assertEqual(t1["cbf_count"], 3)
            self.assertEqual(t1["cbf_last_src"], "lane_border")
            self.assertTrue(t1["cbf_hard_stop"])
            self.assertEqual(t1["brake_updates"], 1)
            self.assertAlmostEqual(t1["brake_scale"], 0.95, places=2)
            self.assertAlmostEqual(t1["brake_a_eff"], 1.70, places=2)
            self.assertAlmostEqual(t1["brake_vbat"], 12.10, places=2)
            # 第 2 条重复 no_entry 介入被折叠，不刷满里程碑；第 1/3 条 cbf + 1 条 brake_adapt 保留
            cbf_ms = [m for m in diag["milestones"] if m["area"] == "cbf"]
            br_ms = [m for m in diag["milestones"] if m["area"] == "brake_adapt"]
            self.assertEqual(len(cbf_ms), 2)
            self.assertEqual(len(br_ms), 1)

        html = (REPO_ROOT / "tools" / "panel" / "index.html").read_text(encoding="utf-8")
        self.assertIn("e.area === 'cbf'", html)
        self.assertIn("e.area === 'brake_adapt'", html)
        self.assertIn("d.tier1", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
