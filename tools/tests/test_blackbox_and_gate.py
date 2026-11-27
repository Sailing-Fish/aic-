#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Real-Car Blackbox Calibrator & Anti-Overfitting 3-Layer Iteration Gate.

Verifies:
  1. Blackbox directory & .zip parsing (meta.json, events_slice.csv, follow_slice.log,
     frame_timestamps.csv, params_snapshot.yaml).
  2. Calibrated `tools/sim/real_car_dynamics.json` generation & physical parameter ranges.
  3. Offline real-car blackbox replay consistency check (all 5 checks a..e PASS).
  4. Anti-cheat static scanner: catches synthetic `s_total_m > 3.2 && s_total_m < 4.0`,
     `arc_m` literal branching, world XY hardcoding, and holdout seed hardcoding;
     passes clean production C++ codebase with 0 false positives.
  5. Safety margin floor enforcement on candidate YAMLs (`border_margin_m >= 0.020`,
     `no_entry_margin_m >= 0.025`, `yellow_bumper_margin_m >= 0.070`).
  6. Frozen Blind Holdout separation (`101..104` vs `501..505`) and deterministic
     ACCEPT / REVERT verdict logic.

Runnable via:
    C:\\Users\\Lenovo\\miniconda3\\envs\\dl\\python.exe -m unittest tools/tests/test_blackbox_and_gate.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

# Guard against tools/sim/profile.py shadowing Python stdlib `profile`
REPO_ROOT = Path(__file__).resolve().parents[2]
_SIM_DIR = REPO_ROOT / "tools" / "sim"
_sim_dir_norm = os.path.normcase(str(_SIM_DIR))
sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or ".")) != _sim_dir_norm]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sim import auto_iterate_gate  # noqa: E402
from tools.sim import blackbox_calibrator  # noqa: E402

FORK_YAML = REPO_ROOT / "tools" / "config" / "finals_fork.yaml"
ROUNDABOUT_YAML = REPO_ROOT / "tools" / "config" / "finals_roundabout.yaml"
REAL_CAR_DYNAMICS_JSON = REPO_ROOT / "tools" / "sim" / "real_car_dynamics.json"


class TestBlackboxCalibrator(unittest.TestCase):
    """Tests for tools/sim/blackbox_calibrator.py and real_car_dynamics.json."""

    def test_parse_real_blackbox_dirs_and_zip_support(self):
        existing = [p for p in blackbox_calibrator.DEFAULT_BLACKBOX_DIRS if p.exists()]
        self.assertTrue(
            len(existing) >= 2,
            f"Expected both extracted real-car blackbox dirs on disk, found: {existing}",
        )
        for bb_dir in existing:
            bundle = blackbox_calibrator.load_blackbox(bb_dir)
            self.assertTrue(bundle.meta.get("frames", 0) > 100)
            self.assertGreater(len(bundle.frame_timestamps), 100)
            self.assertGreater(len(bundle.events), 50)
            self.assertGreater(len(bundle.ctrl_records), 40)
            self.assertGreaterEqual(len(bundle.brake_adapt_events), 2)
            self.assertGreater(len(bundle.camera_perf), 2)
            self.assertGreater(len(bundle.yolo_perf), 2)
            self.assertIn("target_speed", bundle.params_snapshot)

        # Verify arbitrary .zip bundle loading matches extracted directory parsing
        src_dir = existing[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "bundle_test.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for fname in (
                    "meta.json",
                    "events_slice.csv",
                    "follow_slice.log",
                    "frame_timestamps.csv",
                    "params_snapshot.yaml",
                ):
                    fp = src_dir / fname
                    if fp.is_file():
                        zf.write(fp, arcname=f"nested_run/{fname}")
            z_bundle = blackbox_calibrator.load_blackbox(zip_path)
            d_bundle = blackbox_calibrator.load_blackbox(src_dir)
            self.assertEqual(len(z_bundle.frame_timestamps), len(d_bundle.frame_timestamps))
            self.assertEqual(len(z_bundle.brake_adapt_events), len(d_bundle.brake_adapt_events))
            self.assertEqual(len(z_bundle.ctrl_records), len(d_bundle.ctrl_records))

    def test_generate_and_validate_real_car_dynamics_json(self):
        existing = [p for p in blackbox_calibrator.DEFAULT_BLACKBOX_DIRS if p.exists()]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir) / "dynamics_out.json"
            cal = blackbox_calibrator.calibrate_from_blackboxes(existing, out_json=tmp_out)
            self.assertTrue(tmp_out.is_file())
            loaded = json.loads(tmp_out.read_text(encoding="utf-8"))

        self.assertEqual(loaded["schema_version"], "1.0")
        c = loaded["calibrated"]
        self.assertGreater(c["a_eff_mps2"], 1.0)
        self.assertLess(c["a_eff_mps2"], 2.2)
        self.assertGreaterEqual(c["battery_voltage_nominal_v"], 11.0)
        self.assertLessEqual(c["battery_voltage_nominal_v"], 13.0)
        self.assertGreater(c["camera_fps_hz"], 10.0)
        self.assertGreater(c["camera_read_p50_ms"], 20.0)
        self.assertGreater(c["yolo_e2e_p50_ms"], 10.0)
        self.assertIn("comparison_vs_sim_defaults", loaded)
        self.assertIn("a_eff_mps2", loaded["comparison_vs_sim_defaults"])
        self.assertEqual(len(loaded["runs"]), len(existing))

        # Verify committed/generated file at tools/sim/real_car_dynamics.json exists & matches schema
        self.assertTrue(REAL_CAR_DYNAMICS_JSON.is_file())
        disk_data = json.loads(REAL_CAR_DYNAMICS_JSON.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            disk_data["calibrated"]["a_eff_mps2"], cal["calibrated"]["a_eff_mps2"], places=4
        )

    def test_blackbox_offline_replay_consistency_check(self):
        rep = blackbox_calibrator.run_blackbox_replay_check(
            yaml_fork=FORK_YAML,
            yaml_roundabout=ROUNDABOUT_YAML,
        )
        self.assertEqual(rep["total_count"], 5)
        for ch in rep["checks"]:
            self.assertTrue(ch["passed"], f"Replay check {ch['id']} failed: {ch['reason']}")
        self.assertTrue(rep["passed"])


class TestAntiOverfittingGate(unittest.TestCase):
    """Tests for Layer 1 anti-cheat, safety floors, Layer 3 holdout separation, and verdicts."""

    def test_anti_cheat_catches_synthetic_map_coordinate_hacks(self):
        hack_range = """
        double compute_yaw(double s_total_m, double lat_err) {
          double yaw_cmd = 3.8 * lat_err;
          if (s_total_m > 3.2 && s_total_m < 4.0) {
            yaw_cmd = -1.25;
          }
          return yaw_cmd;
        }
        """
        v1 = auto_iterate_gate.check_anti_cheat_code(hack_range, "src/motion_controller.cpp")
        self.assertTrue(len(v1) >= 1, "Failed to catch s_total_m > 3.2 && s_total_m < 4.0 hack")
        self.assertEqual(v1[0]["rule"], "map_coordinate_range_branching")

        hack_arc_literal = """
        if (arc_m >= 4.5) {
          speed_cmd = 0.18;
        }
        """
        v2 = auto_iterate_gate.check_anti_cheat_code(hack_arc_literal, "src/corridor_planner.cpp")
        self.assertTrue(len(v2) >= 1, "Failed to catch arc_m >= 4.5 literal hack")

        hack_xy = """
        if (odom_x > 1.2 && odom_x < 2.8) {
          yaw_cmd = 0.8;
        }
        """
        v3 = auto_iterate_gate.check_anti_cheat_code(hack_xy, "src/motion_controller.cpp")
        self.assertTrue(len(v3) >= 1, "Failed to catch world XY hardcoding hack")

        hack_seed = """
        if (seed == 501) {
          speed_cmd = 0.30;
        }
        """
        v4 = auto_iterate_gate.check_anti_cheat_code(hack_seed, "src/motion_controller.cpp")
        self.assertTrue(len(v4) >= 1, "Failed to catch holdout seed hardcoding")

    def test_anti_cheat_allows_legitimate_finish_guard_and_clean_repo(self):
        legit_finish = """
        bool FinishGuard::is_finish_armed() const {
          return s_total_m_ >= config_.finish_min_dist_m;
        }
        // Comment mentioning s_total_m > 3.2 && s_total_m < 4.0 must be ignored
        """
        v_fg = auto_iterate_gate.check_anti_cheat_code(legit_finish, "src/finish_guard.cpp")
        self.assertEqual(v_fg, [])

        legit_fork_commit = """
        int CorridorPlanner::committed_branch(double current_arc_s_m) const {
          if ((current_arc_s_m - commit_start_s_m_) >= config_.fork_commit_dist_m) {
            return 0;
          }
          return committed_branch_;
        }
        """
        v_cp = auto_iterate_gate.check_anti_cheat_code(legit_fork_commit, "src/corridor_planner.cpp")
        self.assertEqual(v_cp, [])

        # Scan the entire real C++ production tree
        repo_scan = auto_iterate_gate.check_anti_cheat(auto_iterate_gate.LINE_FOLLOWER_ROOT)
        self.assertGreater(repo_scan["files_scanned"], 10)
        self.assertTrue(
            repo_scan["passed"],
            f"Clean production C++ tree had unexpected anti-cheat hits: {repo_scan['violations']}",
        )

    def test_yaml_safety_floor_enforcement(self):
        # Current production YAMLs must pass
        ok = auto_iterate_gate.check_yaml_safety_floors(FORK_YAML, ROUNDABOUT_YAML)
        self.assertTrue(ok["passed"], f"Production YAMLs failed safety floors: {ok['violations']}")

        # Mutate each safety floor below redline and confirm rejection
        base_text = FORK_YAML.read_text(encoding="utf-8")
        bad_cases = [
            ("border_margin_m: 0.025", "border_margin_m: 0.015", "border_margin_m"),
            ("no_entry_margin_m: 0.030", "no_entry_margin_m: 0.018", "no_entry_margin_m"),
            ("yellow_bumper_margin_m: 0.08", "yellow_bumper_margin_m: 0.055", "yellow_bumper_margin_m"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for old_s, new_s, expected_key in bad_cases:
                self.assertIn(old_s, base_text)
                bad_yml = Path(tmpdir) / f"bad_{expected_key}.yaml"
                bad_yml.write_text(base_text.replace(old_s, new_s, 1), encoding="utf-8")
                res = auto_iterate_gate.check_yaml_safety_floors(bad_yml, ROUNDABOUT_YAML)
                self.assertFalse(res["passed"], f"Expected rejection when violating {expected_key}")
                self.assertTrue(
                    any(v.get("key") == expected_key for v in res["violations"]),
                    f"Missing violation for {expected_key}: {res['violations']}",
                )

    def test_exploration_and_blind_holdout_separation_and_dr_specs(self):
        exp_specs = auto_iterate_gate.build_exploration_specs()
        hld_specs = auto_iterate_gate.build_blind_holdout_specs()

        exp_seeds = {s.seed for s in exp_specs}
        hld_seeds = {s.seed for s in hld_specs}
        self.assertEqual(exp_seeds, {101, 102, 103, 104})
        self.assertEqual(hld_seeds, {501, 502, 503, 504, 505})
        self.assertEqual(exp_seeds & hld_seeds, set())

        # Both maps represented in both suites
        self.assertEqual({s.track for s in exp_specs}, {"fork_s", "roundabout"})
        self.assertEqual({s.track for s in hld_specs}, {"fork_s", "roundabout"})
        self.assertEqual(len(exp_specs), 8)
        self.assertEqual(len(hld_specs), 10)

        # Verify harsh blind-holdout perturbations
        for sp in hld_specs:
            self.assertGreaterEqual(sp.latency_ms, 35.0)
            self.assertLessEqual(sp.a_eff_scale, 0.78 + 1e-9)
            self.assertGreater(sp.glare, 0.0)
            self.assertIsNotNone(sp.color_cast)
            self.assertNotEqual(sp.pitch_jitter_deg, 0.0)
            self.assertTrue(
                abs(sp.steer_scale - 1.0) >= 0.09 and abs(sp.speed_scale - 1.0) >= 0.09,
                f"Expected +/-10..12% actuation mismatch on {sp.name}",
            )

    def test_verdict_accept_and_revert_rules(self):
        l1_ok = {
            "passed": True,
            "compile_and_test": {"passed": True},
            "anti_cheat": {"passed": True, "violations": []},
            "safety_floors": {"passed": True, "violations": []},
        }
        l2_ok = {
            "passed": True,
            "regressions": {"passed": True, "reports": []},
            "blackbox_replay": {"passed": True, "checks": []},
        }
        exp_base = {"total": 8, "passed": 7, "dnf_count": 0, "total_penalty_sum_s": 5.0, "composite_score": 25.0}
        hld_base = {"total": 10, "passed": 8, "dnf_count": 0, "total_penalty_sum_s": 10.0, "composite_score": 30.0}
        baseline = {
            "composite_score": round(0.35 * 25.0 + 0.65 * 30.0, 4),
            "layer3": {"exploration": exp_base, "holdout": hld_base},
        }

        # Improved candidate -> ACCEPT
        exp_good = dict(exp_base, passed=8, total_penalty_sum_s=0.0, composite_score=18.0)
        hld_good = dict(hld_base, passed=9, total_penalty_sum_s=5.0, composite_score=24.0)
        verdict, comp, _ = auto_iterate_gate.evaluate_verdict(
            "full", l1_ok, l2_ok, exp_good, hld_good, baseline=baseline
        )
        self.assertEqual(verdict, "ACCEPT")
        self.assertLess(comp, baseline["composite_score"])

        # Overfitted candidate (exploration improved hugely, holdout regressed) -> REVERT
        exp_overfit = dict(exp_base, passed=8, total_penalty_sum_s=0.0, composite_score=10.0)
        hld_overfit = dict(hld_base, passed=7, total_penalty_sum_s=15.0, composite_score=34.0)
        verdict_ov, _, reasons_ov = auto_iterate_gate.evaluate_verdict(
            "full", l1_ok, l2_ok, exp_overfit, hld_overfit, baseline=baseline
        )
        self.assertEqual(verdict_ov, "REVERT")
        self.assertTrue(any("HOLDOUT" in r for r in reasons_ov))

        # Layer 1 anti-cheat failure -> REVERT even if scores improved
        l1_cheat = {
            "passed": False,
            "compile_and_test": {"passed": True},
            "anti_cheat": {"passed": False, "violations": [{"rule": "map_coordinate_range_branching"}]},
            "safety_floors": {"passed": True, "violations": []},
        }
        verdict_ch, _, _ = auto_iterate_gate.evaluate_verdict(
            "full", l1_cheat, l2_ok, exp_good, hld_good, baseline=baseline
        )
        self.assertEqual(verdict_ch, "REVERT")


if __name__ == "__main__":
    unittest.main()
