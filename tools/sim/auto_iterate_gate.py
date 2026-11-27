#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anti-Overfitting 3-Layer Iteration Hard Gate for Autonomous Sim-to-Real Optimization.

Decides deterministic `VERDICT: ACCEPT` vs `VERDICT: REVERT` for every iteration:

  Layer 1 — Static Structural Redlines & Strict Compile:
    1. Pure C++17 strict compile (`g++ -std=c++17 -Wall -Wextra -Werror`) of all 7 pure
       C++17 modules + `test_finals_modules.cpp` (0 warnings, ALL_FINALS_CPP_TESTS_PASSED)
       + rebuild `tools/sim/build/sim_brain.dll` (0 warnings).
    2. Static anti-cheat check on `src/line_follow_ws/src/line_follower/`: FORBIDS
       map-coordinate hardcoding (e.g., branching steering/speed directly on literal
       `s_total_m` / `arc_m` ranges outside `FinishGuard` finish-arming, world XY
       hardcoding, or holdout seed checks).
    3. Safety margin floor enforcement on both candidate YAMLs:
         border_margin_m >= 0.020
         no_entry_margin_m >= 0.025
         yellow_bumper_margin_m >= 0.070

  Layer 2 — 4 Real-Car Crash Regressions + Real-Car Blackbox Offline Replay:
    1. All 4 scenarios in `tools/sim/regression_suite.py` MUST PASS with 0.0s penalty.
    2. `blackbox_calibrator.run_blackbox_replay_check()` MUST PASS 100% (5/5).

  Layer 3 — Exploration Set (`101..104`) + FROZEN BLIND HOLDOUT Set (`501..505`):
    - Exploration seeds: `101..104` on both `fork_s` and `roundabout` (8 runs).
    - FROZEN BLIND HOLDOUT seeds: `501..505` on both `fork_s` and `roundabout` (10 runs),
      NEVER used for tuning selection, combining unseen random props + lighting/color-cast/
      glare/blur + command latency `+35ms` + slippery `0.78x` + `±12%` steer/speed mismatch
      + `±1.0~1.5°` camera `pitch_jitter_deg`.
    - Parallel execution via `ProcessPoolExecutor` (`--workers 6`).
    - Deterministic composite score + zero-regression check vs `--baseline-json`.

CLI:
    python tools/sim/auto_iterate_gate.py \\
        --mode full|fast|holdout \\
        --fork-yaml tools/config/finals_fork.yaml \\
        --roundabout-yaml tools/config/finals_roundabout.yaml \\
        [--baseline-json runs/auto_iterate/iter_00_baseline.json] \\
        [--out-json runs/auto_iterate/iter_01.json] \\
        [--workers 6]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Guard against tools/sim/profile.py shadowing Python stdlib `profile`
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[1]
_this_dir_norm = os.path.normcase(str(_THIS_DIR))
sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or ".")) != _this_dir_norm]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sim import blackbox_calibrator
from tools.sim import build_brain_dll
from tools.sim import regression_suite
from tools.sim.brain_bridge import load_line_follower_params
from tools.sim.pc_sim import ClosedLoopRunner, SimConfig, SimRunResult
from tools.sim.render import CameraModel

# ---------------------------------------------------------------------------
# Frozen Constants & Redlines
# ---------------------------------------------------------------------------

SAFETY_FLOORS: Dict[str, float] = {
    "border_margin_m": 0.020,
    "no_entry_margin_m": 0.025,
    "yellow_bumper_margin_m": 0.070,
}

DEFAULT_FORK_YAML = REPO_ROOT / "tools" / "config" / "finals_fork.yaml"
DEFAULT_ROUNDABOUT_YAML = REPO_ROOT / "tools" / "config" / "finals_roundabout.yaml"
LINE_FOLLOWER_ROOT = REPO_ROOT / "src" / "line_follow_ws" / "src" / "line_follower"
FINALS_TEST_CPP = LINE_FOLLOWER_ROOT / "test" / "test_finals_modules.cpp"

# Exploration seeds (used for diagnostic iteration analysis)
EXPLORATION_SEEDS: Tuple[int, ...] = (101, 102, 103, 104)

# FROZEN BLIND HOLDOUT seeds — NEVER tune specifically to these!
BLIND_HOLDOUT_SEEDS: Tuple[int, ...] = (501, 502, 503, 504, 505)


@dataclass(frozen=True)
class GateTrialSpec:
    name: str
    suite: str  # "exploration" | "holdout"
    track: str  # "fork_s" | "roundabout"
    seed: int
    cones: int = 2
    no_entry: int = 1
    digits: int = 3
    random_yellow: int = 1
    a_eff: float = 1.44
    a_eff_scale: float = 1.0
    latency_ms: float = 10.0
    latency_jitter_ms: float = 2.0
    steer_scale: float = 1.0
    speed_scale: float = 1.0
    voltage: float = 12.0
    gain: float = 1.0
    color_cast: Optional[str] = None
    glare: float = 0.0
    blur: float = 0.0
    pitch_jitter_deg: float = 0.0
    max_time_s: float = 55.0


def build_exploration_specs() -> List[GateTrialSpec]:
    """Deterministic Exploration Suite (seeds 101..104 on fork_s & roundabout)."""
    profiles = {
        101: dict(a_eff_scale=1.00, latency_ms=10.0, steer_scale=1.00, speed_scale=1.00, voltage=12.0, gain=1.00, color_cast=None, glare=0.00, blur=0.0, pitch_jitter_deg=0.0),
        102: dict(a_eff_scale=0.92, latency_ms=15.0, steer_scale=0.96, speed_scale=1.04, voltage=11.8, gain=1.05, color_cast="warm", glare=0.08, blur=0.0, pitch_jitter_deg=+0.5),
        103: dict(a_eff_scale=0.95, latency_ms=15.0, steer_scale=1.04, speed_scale=0.96, voltage=12.1, gain=0.95, color_cast="cool", glare=0.08, blur=0.2, pitch_jitter_deg=-0.5),
        104: dict(a_eff_scale=0.88, latency_ms=20.0, steer_scale=0.95, speed_scale=1.05, voltage=11.7, gain=1.03, color_cast="yellowed", glare=0.10, blur=0.2, pitch_jitter_deg=0.0),
    }
    specs: List[GateTrialSpec] = []
    for track in ("fork_s", "roundabout"):
        for sd in EXPLORATION_SEEDS:
            p = profiles[sd]
            specs.append(
                GateTrialSpec(
                    name=f"explore_{track}_s{sd}",
                    suite="exploration",
                    track=track,
                    seed=sd,
                    **p,
                )
            )
    return specs


def build_blind_holdout_specs() -> List[GateTrialSpec]:
    """FROZEN BLIND HOLDOUT Suite (seeds 501..505 on BOTH fork_s & roundabout).

    Combines unseen random props + lighting/color-cast/glare/blur + latency +35ms
    + slippery 0.78x + ±12% steer/speed mismatch + ±1.0~1.5 deg pitch jitter.
    """
    frozen_profiles = {
        501: dict(
            a_eff_scale=0.78,
            latency_ms=35.0,
            latency_jitter_ms=5.0,
            steer_scale=0.88,
            speed_scale=1.12,
            voltage=11.5,
            gain=1.12,
            color_cast="warm",
            glare=0.22,
            blur=0.5,
            pitch_jitter_deg=+1.2,
        ),
        502: dict(
            a_eff_scale=0.78,
            latency_ms=35.0,
            latency_jitter_ms=5.0,
            steer_scale=1.12,
            speed_scale=0.88,
            voltage=11.4,
            gain=0.88,
            color_cast="cool",
            glare=0.18,
            blur=0.4,
            pitch_jitter_deg=-1.2,
        ),
        503: dict(
            a_eff_scale=0.78,
            latency_ms=35.0,
            latency_jitter_ms=5.0,
            steer_scale=0.90,
            speed_scale=1.10,
            voltage=12.2,
            gain=1.08,
            color_cast="yellowed",
            glare=0.30,
            blur=0.6,
            pitch_jitter_deg=+1.5,
        ),
        504: dict(
            a_eff_scale=0.78,
            latency_ms=35.0,
            latency_jitter_ms=5.0,
            steer_scale=1.10,
            speed_scale=0.90,
            voltage=11.6,
            gain=0.90,
            color_cast="warm",
            glare=0.25,
            blur=0.3,
            pitch_jitter_deg=-1.5,
        ),
        505: dict(
            a_eff_scale=0.78,
            latency_ms=35.0,
            latency_jitter_ms=5.0,
            steer_scale=0.88,
            speed_scale=0.88,
            voltage=11.8,
            gain=1.15,
            color_cast="yellowed",
            glare=0.28,
            blur=0.5,
            pitch_jitter_deg=+1.0,
        ),
    }
    specs: List[GateTrialSpec] = []
    for track in ("fork_s", "roundabout"):
        for sd in BLIND_HOLDOUT_SEEDS:
            p = frozen_profiles[sd]
            specs.append(
                GateTrialSpec(
                    name=f"holdout_{track}_s{sd}",
                    suite="holdout",
                    track=track,
                    seed=sd,
                    **p,
                )
            )
    return specs


# ---------------------------------------------------------------------------
# Layer 1: Strict C++17 Compile + Anti-Cheat Static Check + Safety Floors
# ---------------------------------------------------------------------------


def run_strict_cpp_compile_and_test(verbose: bool = True) -> Dict[str, Any]:
    """Strict compile (`-std=c++17 -Wall -Wextra -Werror`) of all 7 pure C++17 modules
    + `test_finals_modules.cpp` (0 warnings, all tests PASS) + rebuild `sim_brain.dll`.
    """
    t0 = time.perf_counter()
    gxx = shutil.which("g++")
    if not gxx:
        return {
            "passed": False,
            "compile_ok": False,
            "tests_ok": False,
            "dll_ok": False,
            "error": "g++ compiler not found on PATH",
            "wall_time_s": round(time.perf_counter() - t0, 3),
        }

    if not FINALS_TEST_CPP.is_file():
        return {
            "passed": False,
            "compile_ok": False,
            "tests_ok": False,
            "dll_ok": False,
            "error": f"Missing test_finals_modules.cpp at {FINALS_TEST_CPP}",
            "wall_time_s": round(time.perf_counter() - t0, 3),
        }

    pure_sources = [build_brain_dll.SRC_DIR / name for name in build_brain_dll.PURE_MODULE_NAMES]
    for s in pure_sources:
        if not s.is_file():
            return {
                "passed": False,
                "compile_ok": False,
                "tests_ok": False,
                "dll_ok": False,
                "error": f"Missing pure C++17 module: {s}",
                "wall_time_s": round(time.perf_counter() - t0, 3),
            }

    build_brain_dll.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    exe_name = "gate_test_finals_modules.exe" if os.name == "nt" else "gate_test_finals_modules"
    exe_path = build_brain_dll.BUILD_DIR / exe_name

    compile_cmd = [
        gxx,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-DLINE_FOLLOWER_NO_OPENCV",
        f"-I{build_brain_dll.INCLUDE_DIR}",
        *[str(s) for s in pure_sources],
        str(FINALS_TEST_CPP),
        "-o",
        str(exe_path),
    ]
    if os.name == "nt":
        compile_cmd.extend(["-static-libgcc", "-static-libstdc++"])

    if verbose:
        print("[gate/L1] Compiling 7 pure C++17 modules + test_finals_modules.cpp (-Wall -Wextra -Werror)...", flush=True)

    c_proc = subprocess.run(
        compile_cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    compile_ok = (c_proc.returncode == 0) and (not c_proc.stderr.strip())
    if c_proc.returncode != 0:
        return {
            "passed": False,
            "compile_ok": False,
            "tests_ok": False,
            "dll_ok": False,
            "compile_stderr": c_proc.stderr.strip(),
            "compile_stdout": c_proc.stdout.strip(),
            "wall_time_s": round(time.perf_counter() - t0, 3),
        }

    tests_ok = False
    test_stdout = ""
    test_stderr = ""
    try:
        t_proc = subprocess.run(
            [str(exe_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        test_stdout = t_proc.stdout.strip()
        test_stderr = t_proc.stderr.strip()
        tests_ok = (t_proc.returncode == 0) and ("ALL_FINALS_CPP_TESTS_PASSED" in test_stdout)
    finally:
        try:
            if exe_path.exists():
                exe_path.unlink()
        except OSError:
            pass

    dll_ok = False
    dll_err = ""
    if compile_ok and tests_ok:
        if verbose:
            print("[gate/L1] Rebuilding sim_brain.dll (-Wall -Wextra -Werror)...", flush=True)
        try:
            build_brain_dll.build_dll(force=True, verbose=False)
            dll_ok = True
        except Exception as exc:
            dll_err = str(exc)

    passed = bool(compile_ok and tests_ok and dll_ok)
    return {
        "passed": passed,
        "compile_ok": compile_ok,
        "tests_ok": tests_ok,
        "dll_ok": dll_ok,
        "test_stdout": test_stdout,
        "test_stderr": test_stderr,
        "dll_error": dll_err,
        "wall_time_s": round(time.perf_counter() - t0, 3),
    }


def _strip_cpp_comments_and_strings(code: str) -> str:
    """Strip C++ comments and string literals while preserving newlines (1:1 line numbering)."""
    out: List[str] = []
    n = len(code)
    i = 0
    while i < n:
        ch = code[i]
        nxt = code[i + 1] if (i + 1) < n else ""
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and code[i] != "\n":
                i += 1
        elif ch == "/" and nxt == "*":
            i += 2
            while i < n:
                if code[i] == "*" and (i + 1) < n and code[i + 1] == "/":
                    i += 2
                    break
                if code[i] == "\n":
                    out.append("\n")
                i += 1
        elif ch == '"':
            out.append('"')
            i += 1
            while i < n:
                if code[i] == "\\" and (i + 1) < n:
                    i += 2
                    continue
                if code[i] == '"':
                    out.append('"')
                    i += 1
                    break
                if code[i] == "\n":
                    out.append("\n")
                i += 1
        elif ch == "'":
            out.append("'")
            i += 1
            while i < n:
                if code[i] == "\\" and (i + 1) < n:
                    i += 2
                    continue
                if code[i] == "'":
                    out.append("'")
                    i += 1
                    break
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# Cumulative map-coordinate / arc-length identifiers forbidden for literal range steering/speed hacks
_ARC_VAR_PAT = (
    r"(?:s_total_m_?|total_arc_m_?|current_arc_s_m|arc_s_m|arc_m|odom_s_m_?|track_s_m|map_s_m|s_track_m)"
)
_POS_NUM_PAT = r"(?:(?:[1-9]\d*(?:\.\d+)?)|(?:0\.[0-9]*[1-9][0-9]*))"

# Pattern 1: Literal range check on arc/s variable:
#   e.g. `s_total_m > 3.2 && s_total_m < 4.0` or `3.2 < arc_m && arc_m < 4.0`
_RANGE_HACK_RE_1 = re.compile(
    rf"\b{_ARC_VAR_PAT}\b\s*(?:>=|>|<=|<)\s*{_POS_NUM_PAT}\s*&&\s*\b{_ARC_VAR_PAT}\b\s*(?:<=|<|>=|>)\s*{_POS_NUM_PAT}"
)
_RANGE_HACK_RE_2 = re.compile(
    rf"{_POS_NUM_PAT}\s*(?:<=|<)\s*\b{_ARC_VAR_PAT}\b\s*&&\s*\b{_ARC_VAR_PAT}\b\s*(?:<=|<)\s*{_POS_NUM_PAT}"
)

# Pattern 2: Any conditional comparing cumulative arc/s variable directly to a positive literal
# (excluding FinishGuard config comparisons and relative subtraction `(arc - start_arc)`).
_ARC_LITERAL_CMP_RE = re.compile(
    rf"(?<!-\s)(?<!-)\b(?P<var>{_ARC_VAR_PAT})\b\s*(?P<op>>=|<=|>|<|==)\s*(?P<num>{_POS_NUM_PAT})\b"
    rf"|(?P<num2>{_POS_NUM_PAT})\s*(?P<op2>>=|<=|>|<|==)\s*\b(?P<var2>{_ARC_VAR_PAT})\b(?!\s*-)"
)

# Pattern 3: Literal world XY coordinate branching in control/planning modules
_XY_HACK_RE = re.compile(
    r"\b(?:odom_x|odom_y|odom_pose\.x|odom_pose\.y|pose_x|pose_y)\b\s*(?:>=|>|<=|<)\s*"
    + _POS_NUM_PAT
    + r"\s*&&\s*\b(?:odom_x|odom_y|odom_pose\.x|odom_pose\.y|pose_x|pose_y)\b\s*(?:<=|<|>=|>)\s*"
    + _POS_NUM_PAT
)

# Pattern 4: Hardcoding holdout seeds (501..505)
_SEED_HACK_RE = re.compile(r"\bseed\b\s*(?:==|>=|<=)\s*50[1-5]\b")


def check_anti_cheat_code(code: str, rel_path: str = "motion_controller.cpp") -> List[Dict[str, Any]]:
    """Static AST/token anti-cheat scanner on a single C++ source/header text.

    Forbids map-coordinate hardcoding (branching steering/speed on literal `s_total_m` /
    `arc_m` ranges outside `FinishGuard` finish-arming, world XY ranges, or holdout seeds).
    """
    violations: List[Dict[str, Any]] = []
    cleaned = _strip_cpp_comments_and_strings(code)
    fname = Path(rel_path).name.lower()
    is_finish_guard = fname in ("finish_guard.cpp", "finish_guard.hpp")

    # Join wrapped multi-line conditionals while tracking starting line numbers
    lines = cleaned.splitlines()
    for idx, raw_ln in enumerate(lines, start=1):
        # Also build a 2-line window to catch line-wrapped `if (s_total_m > 3.2 &&\n s_total_m < 4.0)`
        window = raw_ln + (" " + lines[idx] if idx < len(lines) else "")

        if _RANGE_HACK_RE_1.search(window) or _RANGE_HACK_RE_2.search(window):
            violations.append(
                {
                    "file": rel_path,
                    "line": idx,
                    "rule": "map_coordinate_range_branching",
                    "snippet": raw_ln.strip()[:160],
                }
            )
            continue

        if _XY_HACK_RE.search(window):
            violations.append(
                {
                    "file": rel_path,
                    "line": idx,
                    "rule": "world_xy_range_hardcoding",
                    "snippet": raw_ln.strip()[:160],
                }
            )
            continue

        if _SEED_HACK_RE.search(window):
            violations.append(
                {
                    "file": rel_path,
                    "line": idx,
                    "rule": "holdout_seed_hardcoding",
                    "snippet": raw_ln.strip()[:160],
                }
            )
            continue

        # Single-sided literal comparison on cumulative arc/s outside FinishGuard
        if not is_finish_guard:
            # Ignore simple assignments/declarations like `double arc_m = 0.0;` (only check comparisons)
            m = _ARC_LITERAL_CMP_RE.search(raw_ln)
            if m:
                # Allow relative difference checks like `(current_arc_s_m - commit_start_s_m_) >= ...`
                # and allow pure zero/epsilon checks (<= 0.05)
                num_str = m.group("num") or m.group("num2") or "0"
                try:
                    num_val = float(num_str)
                except ValueError:
                    num_val = 0.0
                if num_val > 0.05 and (" - " not in raw_ln.split(m.group(0))[0][-24:]):
                    violations.append(
                        {
                            "file": rel_path,
                            "line": idx,
                            "rule": "map_coordinate_literal_branching",
                            "snippet": raw_ln.strip()[:160],
                        }
                    )

    return violations


def check_anti_cheat(source_root: Path = LINE_FOLLOWER_ROOT) -> Dict[str, Any]:
    """Scan all production C++ sources & headers under `src/line_follow_ws/src/line_follower/`."""
    scanned_files: List[str] = []
    violations: List[Dict[str, Any]] = []

    for sub in ("src", "include"):
        sub_dir = source_root / sub
        if not sub_dir.is_dir():
            continue
        for fpath in sorted(sub_dir.rglob("*")):
            if not fpath.is_file() or fpath.suffix.lower() not in (".cpp", ".hpp", ".h", ".cc"):
                continue
            rel = str(fpath.relative_to(source_root)).replace("\\", "/")
            scanned_files.append(rel)
            text = fpath.read_text(encoding="utf-8", errors="replace")
            violations.extend(check_anti_cheat_code(text, rel_path=rel))

    return {
        "passed": len(violations) == 0,
        "files_scanned": len(scanned_files),
        "violations": violations,
    }


def check_yaml_safety_floors(
    yaml_fork: Union[str, Path],
    yaml_roundabout: Union[str, Path],
) -> Dict[str, Any]:
    """Enforce frozen safety margin floors on candidate YAMLs."""
    violations: List[Dict[str, Any]] = []
    checked: Dict[str, Dict[str, Any]] = {}

    for label, path in (("fork", Path(yaml_fork)), ("roundabout", Path(yaml_roundabout))):
        if not path.is_file():
            violations.append({"yaml": label, "path": str(path), "error": "file_not_found"})
            continue
        params = load_line_follower_params(path)
        entry: Dict[str, Any] = {"path": str(path), "values": {}}
        for key, floor_val in SAFETY_FLOORS.items():
            if key not in params:
                violations.append(
                    {"yaml": label, "key": key, "value": None, "floor": floor_val, "error": "missing_key"}
                )
                entry["values"][key] = None
                continue
            val = float(params[key])
            entry["values"][key] = val
            if val < floor_val - 1e-9:
                violations.append(
                    {
                        "yaml": label,
                        "key": key,
                        "value": val,
                        "floor": floor_val,
                        "error": f"{key}={val} < floor {floor_val}",
                    }
                )
        checked[label] = entry

    return {
        "passed": len(violations) == 0,
        "floors": SAFETY_FLOORS,
        "checked": checked,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Parallel Workers (Windows ProcessPoolExecutor-safe top-level callables)
# ---------------------------------------------------------------------------


def _apply_pitch_jitter_if_supported(runner: ClosedLoopRunner, pitch_jitter_deg: float) -> bool:
    """Apply CameraModel pitch_jitter_deg to runner's renderer & mock detector if render.py supports it."""
    if abs(float(pitch_jitter_deg)) <= 1e-9:
        return False
    sig = inspect.signature(CameraModel.__init__)
    if "pitch_jitter_deg" not in sig.parameters:
        return False
    base_cam = runner.world.renderer.cam
    jitter_cam = CameraModel(
        h_mm=base_cam.h,
        alpha_deg=getattr(base_cam, "alpha_deg", 12.5),
        f_px=base_cam.f,
        width=base_cam.w,
        height=base_cam.h_img,
        pitch_jitter_deg=float(pitch_jitter_deg),
    )
    runner.world.renderer.cam = jitter_cam
    runner.world.cam = jitter_cam
    if hasattr(jitter_cam, "build_ground_grid"):
        gd, gw = jitter_cam.build_ground_grid(runner.world.renderer.max_d_mm)
        runner.world.renderer.grid_d = gd
        runner.world.renderer.grid_w = gw
    if hasattr(runner, "_mock_detector") and runner._mock_detector is not None:
        runner._mock_detector.cam = jitter_cam
    runner.reset()
    return True


def _run_regression_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level worker to execute one of the 4 real-car crash regression scenarios."""
    scenario_name = str(task["scenario_name"])
    yaml_fork = str(task["yaml_fork"])
    yaml_roundabout = str(task["yaml_roundabout"])
    rate_hz = float(task.get("rate_hz", 30.0))
    seed = int(task.get("seed", 42))

    scenarios = regression_suite.build_regression_scenarios(
        detector="mock",
        rate_hz=rate_hz,
        param_overrides={},
        seed=seed,
    )
    sc = next(s for s in scenarios if s.name == scenario_name)
    sc.config.yaml_path = yaml_roundabout if sc.config.track == "roundabout" else yaml_fork

    from tools.sim.pc_sim import run_episode

    res = run_episode(sc.config, stop_condition=sc.stop_condition)
    passed, reason = sc.evaluator(res)
    zero_penalty = res.total_penalty_s <= 1e-6
    strict_pass = bool(passed and zero_penalty)
    if passed and not zero_penalty:
        reason = f"{reason}; non_zero_penalty({res.total_penalty_s:.2f}s)"

    return {
        "name": sc.name,
        "track": res.track,
        "passed": strict_pass,
        "reason": reason,
        "sim_time_s": round(res.sim_time_s, 3),
        "wall_time_s": round(res.wall_time_s, 3),
        "total_penalty_s": round(res.total_penalty_s, 3),
        "total_bonus_s": round(res.total_bonus_s, 2),
        "net_score_s": round(res.net_score_s, 3),
        "min_v_mps": round(res.min_v_mps, 4),
        "max_s_m": round(res.max_s_m, 3),
        "dnf": bool(res.dnf),
        "finished": bool(res.finished),
    }


def _compute_trial_composite_cost(
    net_score_s: float,
    total_penalty_s: float,
    finished: bool,
    dnf: bool,
    min_v_mps: float,
) -> float:
    """Deterministic lower-is-better trial cost (seconds-equivalent).

    Strongly penalizes any DNF/incomplete run, any rule penalty, and any reverse overshoot
    below -0.02 m/s while rewarding faster lap times and higher bonus digits via net_score_s.
    """
    cost = float(net_score_s)
    cost += 20.0 * max(0.0, float(total_penalty_s))
    if dnf or not finished:
        cost += 120.0
    if min_v_mps < -0.02:
        cost += 300.0 * abs(min_v_mps + 0.02)
    return round(cost, 4)


def _run_gate_trial_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level worker to execute one Exploration or Frozen Blind Holdout full-lap trial."""
    spec_d = task["spec"]
    yaml_path = str(task["yaml_path"])
    rate_hz = float(task.get("rate_hz", 30.0))

    cfg = SimConfig(
        track=spec_d["track"],
        yaml_path=yaml_path,
        detector="mock",
        rate_hz=rate_hz,
        max_time_s=float(spec_d["max_time_s"]),
        seed=int(spec_d["seed"]),
        cones=int(spec_d["cones"]),
        no_entry=int(spec_d["no_entry"]),
        digits=int(spec_d["digits"]),
        random_yellow=int(spec_d["random_yellow"]),
        prop_seed=int(spec_d["seed"]),
        include_base_yellow=True,
        a_eff=float(spec_d["a_eff"]),
        a_eff_scale=float(spec_d["a_eff_scale"]),
        latency_ms=float(spec_d["latency_ms"]),
        latency_jitter_ms=float(spec_d["latency_jitter_ms"]),
        steer_scale=float(spec_d["steer_scale"]),
        speed_scale=float(spec_d["speed_scale"]),
        voltage=float(spec_d["voltage"]),
        gain=float(spec_d["gain"]),
        color_cast=spec_d["color_cast"],
        glare=float(spec_d["glare"]),
        blur=float(spec_d["blur"]),
    )

    t0 = time.perf_counter()
    runner = ClosedLoopRunner(cfg)
    pitch_applied = False
    try:
        pitch_deg = float(spec_d.get("pitch_jitter_deg", 0.0))
        if abs(pitch_deg) > 1e-9:
            pitch_applied = _apply_pitch_jitter_if_supported(runner, pitch_deg)

        while not runner.is_done:
            runner.step_once()
        res: SimRunResult = runner.build_result(wall_time_s=time.perf_counter() - t0)
    finally:
        runner.close()

    passed = bool(res.finished and (not res.dnf) and (res.total_penalty_s <= 1e-6))
    cost = _compute_trial_composite_cost(
        net_score_s=res.net_score_s,
        total_penalty_s=res.total_penalty_s,
        finished=res.finished,
        dnf=res.dnf,
        min_v_mps=res.min_v_mps,
    )
    return {
        "name": spec_d["name"],
        "suite": spec_d["suite"],
        "track": res.track,
        "seed": int(spec_d["seed"]),
        "pitch_jitter_deg": float(spec_d.get("pitch_jitter_deg", 0.0)),
        "pitch_applied": pitch_applied,
        "passed": passed,
        "finished": bool(res.finished),
        "dnf": bool(res.dnf),
        "dnf_reason": res.dnf_reason,
        "done_reason": res.done_reason,
        "lap_time_s": round(res.lap_time_s, 3),
        "sim_time_s": round(res.sim_time_s, 3),
        "wall_time_s": round(res.wall_time_s, 3),
        "total_penalty_s": round(res.total_penalty_s, 3),
        "boundary_penalty_s": round(res.boundary_penalty_s, 3),
        "cone_hits": int(res.cone_hits),
        "no_entry_hits": int(res.no_entry_hits),
        "yellow_violations": int(res.yellow_violations),
        "yellow_stops_ok": int(res.yellow_stops_ok),
        "total_bonus_s": round(res.total_bonus_s, 2),
        "bonus_digits_collected": list(res.bonus_digits_collected),
        "net_score_s": round(res.net_score_s, 3),
        "min_v_mps": round(res.min_v_mps, 4),
        "max_s_m": round(res.max_s_m, 3),
        "w_cmd_std": round(res.w_cmd_std, 4),
        "composite_cost": cost,
    }


def _summarize_suite(trials: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not trials:
        return {
            "total": 0,
            "passed": 0,
            "finished": 0,
            "dnf_count": 0,
            "pass_rate": 0.0,
            "total_penalty_sum_s": 0.0,
            "total_bonus_sum_s": 0.0,
            "mean_net_score_s": 0.0,
            "worst_net_score_s": 0.0,
            "mean_lap_time_s": 0.0,
            "min_v_mps": 0.0,
            "composite_score": 0.0,
            "trials": [],
        }
    n = len(trials)
    passed = sum(1 for t in trials if t["passed"])
    finished = sum(1 for t in trials if t["finished"])
    dnf_count = sum(1 for t in trials if t["dnf"])
    pen_sum = sum(float(t["total_penalty_s"]) for t in trials)
    bon_sum = sum(float(t["total_bonus_s"]) for t in trials)
    mean_net = sum(float(t["net_score_s"]) for t in trials) / n
    worst_net = max(float(t["net_score_s"]) for t in trials)
    mean_lap = sum(float(t["lap_time_s"]) for t in trials) / n
    min_v = min(float(t["min_v_mps"]) for t in trials)
    comp = sum(float(t["composite_cost"]) for t in trials) / n
    return {
        "total": n,
        "passed": passed,
        "finished": finished,
        "dnf_count": dnf_count,
        "pass_rate": round(passed / n, 4),
        "total_penalty_sum_s": round(pen_sum, 3),
        "total_bonus_sum_s": round(bon_sum, 2),
        "mean_net_score_s": round(mean_net, 3),
        "worst_net_score_s": round(worst_net, 3),
        "mean_lap_time_s": round(mean_lap, 3),
        "min_v_mps": round(min_v, 4),
        "composite_score": round(comp, 4),
        "trials": list(trials),
    }


def _execute_parallel(fn: Any, tasks: Sequence[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    w = max(1, min(int(workers), len(tasks)))
    if w == 1:
        return [fn(t) for t in tasks]
    results: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=w) as ex:
        fut_to_idx = {ex.submit(fn, t): i for i, t in enumerate(tasks)}
        for fut in concurrent.futures.as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            results[idx] = fut.result()
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Gate Orchestrator & Verdict Logic
# ---------------------------------------------------------------------------


def evaluate_verdict(
    mode: str,
    layer1: Dict[str, Any],
    layer2: Optional[Dict[str, Any]],
    exploration: Optional[Dict[str, Any]],
    holdout: Optional[Dict[str, Any]],
    baseline: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, List[str]]:
    """Compute deterministic composite score and ACCEPT vs REVERT verdict."""
    reasons: List[str] = []
    accept = True

    # Layer 1 hard gate
    if not layer1["passed"]:
        accept = False
        if not layer1["compile_and_test"]["passed"]:
            reasons.append("L1 FAIL: strict C++17 compile/tests/DLL build failed")
        if not layer1["anti_cheat"]["passed"]:
            reasons.append(
                f"L1 FAIL: anti-cheat redline violated ({len(layer1['anti_cheat']['violations'])} hit(s))"
            )
        if not layer1["safety_floors"]["passed"]:
            reasons.append(
                f"L1 FAIL: YAML safety floor violated ({len(layer1['safety_floors']['violations'])} hit(s))"
            )

    # Layer 2 hard gate (when run)
    if layer2 is not None and not layer2["passed"]:
        accept = False
        if not layer2["regressions"]["passed"]:
            failed_regs = [r["name"] for r in layer2["regressions"]["reports"] if not r["passed"]]
            reasons.append(f"L2 FAIL: crash regression(s) failed: {failed_regs}")
        if not layer2["blackbox_replay"]["passed"]:
            failed_bb = [
                c["id"] for c in layer2["blackbox_replay"]["checks"] if not c["passed"]
            ]
            reasons.append(f"L2 FAIL: real-car blackbox replay check(s) failed: {failed_bb}")

    # Compute deterministic composite score (lower is better)
    exp_score = float(exploration["composite_score"]) if exploration and exploration["total"] > 0 else None
    hld_score = float(holdout["composite_score"]) if holdout and holdout["total"] > 0 else None

    if exp_score is not None and hld_score is not None:
        composite = round(0.35 * exp_score + 0.65 * hld_score, 4)
    elif hld_score is not None:
        composite = round(hld_score, 4)
    elif exp_score is not None:
        composite = round(exp_score, 4)
    else:
        composite = 9999.0

    # Baseline zero-regression check (anti-overfitting holdout protection)
    if baseline is not None:
        base_comp = float(baseline.get("composite_score", 9999.0))
        base_hld = baseline.get("layer3", {}).get("holdout")
        base_exp = baseline.get("layer3", {}).get("exploration")

        if holdout is not None and base_hld is not None and base_hld.get("total", 0) > 0:
            if holdout["passed"] < base_hld["passed"]:
                accept = False
                reasons.append(
                    f"L3 HOLDOUT REGRESSION: passed {holdout['passed']}/{holdout['total']} "
                    f"< baseline {base_hld['passed']}/{base_hld['total']}"
                )
            if holdout["dnf_count"] > base_hld["dnf_count"]:
                accept = False
                reasons.append(
                    f"L3 HOLDOUT REGRESSION: DNF {holdout['dnf_count']} > baseline {base_hld['dnf_count']}"
                )
            if holdout["total_penalty_sum_s"] > float(base_hld["total_penalty_sum_s"]) + 1e-6:
                accept = False
                reasons.append(
                    f"L3 HOLDOUT REGRESSION: penalty {holdout['total_penalty_sum_s']:.2f}s "
                    f"> baseline {float(base_hld['total_penalty_sum_s']):.2f}s"
                )
            if holdout["composite_score"] > float(base_hld["composite_score"]) + 0.50:
                accept = False
                reasons.append(
                    f"L3 HOLDOUT OVERFIT GUARD: holdout_score {holdout['composite_score']:.3f} "
                    f"> baseline {float(base_hld['composite_score']):.3f} + 0.50"
                )

        if mode == "fast" and exploration is not None and base_exp is not None and base_exp.get("total", 0) > 0:
            if exploration["passed"] < base_exp["passed"]:
                accept = False
                reasons.append(
                    f"L3 EXPLORE REGRESSION: passed {exploration['passed']}/{exploration['total']} "
                    f"< baseline {base_exp['passed']}/{base_exp['total']}"
                )
            if exploration["total_penalty_sum_s"] > float(base_exp["total_penalty_sum_s"]) + 1e-6:
                accept = False
                reasons.append(
                    f"L3 EXPLORE REGRESSION: penalty {exploration['total_penalty_sum_s']:.2f}s "
                    f"> baseline {float(base_exp['total_penalty_sum_s']):.2f}s"
                )

        if composite > base_comp + 1e-6:
            accept = False
            reasons.append(
                f"COMPOSITE REGRESSION: {composite:.4f} > baseline {base_comp:.4f} (lower is better)"
            )
        else:
            delta = base_comp - composite
            if accept:
                reasons.append(
                    f"ALL GATES PASSED (composite={composite:.4f}, delta vs baseline={-delta:+.4f})"
                )
    else:
        if accept:
            reasons.append(f"ALL HARD GATES PASSED (baseline capture, composite={composite:.4f})")

    verdict = "ACCEPT" if accept else "REVERT"
    return verdict, composite, reasons


def run_gate(
    mode: str = "full",
    fork_yaml: Union[str, Path] = DEFAULT_FORK_YAML,
    roundabout_yaml: Union[str, Path] = DEFAULT_ROUNDABOUT_YAML,
    baseline_json: Optional[Union[str, Path]] = None,
    out_json: Optional[Union[str, Path]] = None,
    workers: int = 6,
    rate_hz: float = 30.0,
    skip_compile: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the 3-Layer Anti-Overfitting Iteration Gate."""
    t_start = time.perf_counter()
    y_fork = Path(fork_yaml).resolve()
    y_rb = Path(roundabout_yaml).resolve()

    baseline_data: Optional[Dict[str, Any]] = None
    if baseline_json:
        bp = Path(baseline_json)
        if not bp.is_file():
            raise FileNotFoundError(f"Baseline JSON not found: {bp}")
        baseline_data = json.loads(bp.read_text(encoding="utf-8"))

    # -------------------- LAYER 1 --------------------
    if skip_compile:
        # Still ensure DLL exists if skipping forced recompile
        dll_ok = build_brain_dll.DLL_PATH.is_file()
        if not dll_ok:
            build_brain_dll.build_dll(force=False, verbose=False)
            dll_ok = build_brain_dll.DLL_PATH.is_file()
        compile_res = {
            "passed": dll_ok,
            "skipped": True,
            "compile_ok": True,
            "tests_ok": True,
            "dll_ok": dll_ok,
            "wall_time_s": 0.0,
        }
    else:
        compile_res = run_strict_cpp_compile_and_test(verbose=verbose)

    anti_cheat_res = check_anti_cheat(LINE_FOLLOWER_ROOT)
    safety_res = check_yaml_safety_floors(y_fork, y_rb)

    layer1 = {
        "passed": bool(compile_res["passed"] and anti_cheat_res["passed"] and safety_res["passed"]),
        "compile_and_test": compile_res,
        "anti_cheat": anti_cheat_res,
        "safety_floors": safety_res,
    }
    if verbose:
        print(
            f"[gate/L1] passed={layer1['passed']} "
            f"(compile={compile_res['passed']}, anti_cheat={anti_cheat_res['passed']}, "
            f"safety_floors={safety_res['passed']})",
            flush=True,
        )

    # Short-circuit if Layer 1 compile/DLL failed (cannot run sim safely without valid DLL)
    layer2: Optional[Dict[str, Any]] = None
    exploration_summary: Optional[Dict[str, Any]] = None
    holdout_summary: Optional[Dict[str, Any]] = None

    can_run_sim = bool(compile_res["passed"])

    # -------------------- LAYER 2 --------------------
    if can_run_sim and mode in ("full", "fast"):
        if verbose:
            print("[gate/L2] Running 4 real-car crash regressions + blackbox offline replay...", flush=True)
        reg_names = [
            "roundabout_red_to_sharp_right",
            "fork_y_merge_anti_uturn",
            "straight_dual_digit_5_2",
            "yellow_stop_no_reverse",
        ]
        reg_tasks = [
            {
                "scenario_name": name,
                "yaml_fork": str(y_fork),
                "yaml_roundabout": str(y_rb),
                "rate_hz": rate_hz,
                "seed": 42,
            }
            for name in reg_names
        ]
        reg_reports = _execute_parallel(_run_regression_worker, reg_tasks, workers=workers)
        reg_passed = all(r["passed"] for r in reg_reports)

        bb_replay = blackbox_calibrator.run_blackbox_replay_check(
            yaml_fork=y_fork,
            yaml_roundabout=y_rb,
        )
        layer2 = {
            "passed": bool(reg_passed and bb_replay["passed"]),
            "regressions": {
                "passed": reg_passed,
                "passed_count": sum(1 for r in reg_reports if r["passed"]),
                "total_count": len(reg_reports),
                "reports": reg_reports,
            },
            "blackbox_replay": bb_replay,
        }
        if verbose:
            print(
                f"[gate/L2] passed={layer2['passed']} "
                f"(regressions={layer2['regressions']['passed_count']}/{layer2['regressions']['total_count']}, "
                f"blackbox_replay={bb_replay['passed_count']}/{bb_replay['total_count']})",
                flush=True,
            )

    # -------------------- LAYER 3 --------------------
    if can_run_sim:
        if mode in ("full", "fast"):
            exp_specs = build_exploration_specs()
            if verbose:
                print(
                    f"[gate/L3] Running Exploration suite ({len(exp_specs)} trials, seeds {list(EXPLORATION_SEEDS)}, workers={workers})...",
                    flush=True,
                )
            exp_tasks = [
                {
                    "spec": dataclasses.asdict(sp),
                    "yaml_path": str(y_rb if sp.track == "roundabout" else y_fork),
                    "rate_hz": rate_hz,
                }
                for sp in exp_specs
            ]
            exp_results = _execute_parallel(_run_gate_trial_worker, exp_tasks, workers=workers)
            exploration_summary = _summarize_suite(exp_results)

        if mode in ("full", "holdout"):
            hld_specs = build_blind_holdout_specs()
            if verbose:
                print(
                    f"[gate/L3] Running FROZEN BLIND HOLDOUT suite ({len(hld_specs)} trials, seeds {list(BLIND_HOLDOUT_SEEDS)}, workers={workers})...",
                    flush=True,
                )
            hld_tasks = [
                {
                    "spec": dataclasses.asdict(sp),
                    "yaml_path": str(y_rb if sp.track == "roundabout" else y_fork),
                    "rate_hz": rate_hz,
                }
                for sp in hld_specs
            ]
            hld_results = _execute_parallel(_run_gate_trial_worker, hld_tasks, workers=workers)
            holdout_summary = _summarize_suite(hld_results)

    verdict, composite_score, reasons = evaluate_verdict(
        mode=mode,
        layer1=layer1,
        layer2=layer2,
        exploration=exploration_summary,
        holdout=holdout_summary,
        baseline=baseline_data,
    )

    wall_s = round(time.perf_counter() - t_start, 2)
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "mode": mode,
        "verdict": verdict,
        "composite_score": composite_score,
        "reasons": reasons,
        "wall_time_s": wall_s,
        "fork_yaml": str(y_fork),
        "roundabout_yaml": str(y_rb),
        "baseline_json": str(baseline_json) if baseline_json else None,
        "layer1": layer1,
        "layer2": layer2,
        "layer3": {
            "exploration_seeds": list(EXPLORATION_SEEDS),
            "blind_holdout_seeds": list(BLIND_HOLDOUT_SEEDS),
            "exploration": exploration_summary,
            "holdout": holdout_summary,
        },
    }

    if out_json:
        out_p = Path(out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if verbose:
        print(format_gate_summary(payload), flush=True)

    return payload


def format_gate_summary(payload: Dict[str, Any]) -> str:
    lines = [
        "=" * 96,
        f"ANTI-OVERFITTING 3-LAYER ITERATION GATE  (mode={payload['mode']} | wall={payload['wall_time_s']:.1f}s)",
        "-" * 96,
    ]
    l1 = payload["layer1"]
    lines.append(
        f"Layer 1 [Static + Compile] : {'PASS' if l1['passed'] else 'FAIL'}  "
        f"(cpp_strict={l1['compile_and_test']['passed']}, anti_cheat={l1['anti_cheat']['passed']}, "
        f"safety_floors={l1['safety_floors']['passed']})"
    )
    l2 = payload.get("layer2")
    if l2 is not None:
        reg = l2["regressions"]
        bb = l2["blackbox_replay"]
        lines.append(
            f"Layer 2 [Crash + Blackbox] : {'PASS' if l2['passed'] else 'FAIL'}  "
            f"(regressions={reg['passed_count']}/{reg['total_count']}, "
            f"blackbox_replay={bb['passed_count']}/{bb['total_count']})"
        )
        for r in reg["reports"]:
            st = "PASS" if r["passed"] else "FAIL"
            lines.append(
                f"    [{st}] {r['name']:<30} pen={r['total_penalty_s']:4.1f}s min_v={r['min_v_mps']:+.3f}  {r['reason']}"
            )
        for c in bb["checks"]:
            st = "PASS" if c["passed"] else "FAIL"
            lines.append(f"    [{st}] {c['id']:<30} {c['reason']}")

    l3 = payload.get("layer3", {})
    exp = l3.get("exploration")
    if exp is not None:
        lines.append(
            f"Layer 3 [Exploration 101..104] : pass={exp['passed']}/{exp['total']} dnf={exp['dnf_count']} "
            f"pen_sum={exp['total_penalty_sum_s']:.1f}s mean_net={exp['mean_net_score_s']:.2f}s "
            f"min_v={exp['min_v_mps']:+.3f} score={exp['composite_score']:.3f}"
        )
    hld = l3.get("holdout")
    if hld is not None:
        lines.append(
            f"Layer 3 [BLIND HOLDOUT 501..505]: pass={hld['passed']}/{hld['total']} dnf={hld['dnf_count']} "
            f"pen_sum={hld['total_penalty_sum_s']:.1f}s mean_net={hld['mean_net_score_s']:.2f}s "
            f"min_v={hld['min_v_mps']:+.3f} score={hld['composite_score']:.3f}"
        )
    lines.append("-" * 96)
    lines.append(f"COMPOSITE SCORE : {payload['composite_score']:.4f}  (lower is better)")
    lines.append(f"VERDICT: {payload['verdict']}")
    for rsn in payload["reasons"]:
        lines.append(f"  - {rsn}")
    lines.append("=" * 96)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Anti-Overfitting 3-Layer Iteration Gate (ACCEPT vs REVERT)"
    )
    ap.add_argument(
        "--mode",
        choices=("full", "fast", "holdout"),
        default="full",
        help="Gate evaluation mode: full (L1+L2+Explore+Holdout), fast (L1+L2+Explore), holdout (L1+Holdout).",
    )
    ap.add_argument(
        "--fork-yaml",
        default=str(DEFAULT_FORK_YAML),
        help="Candidate finals_fork YAML path.",
    )
    ap.add_argument(
        "--roundabout-yaml",
        default=str(DEFAULT_ROUNDABOUT_YAML),
        help="Candidate finals_roundabout YAML path.",
    )
    ap.add_argument(
        "--baseline-json",
        default=None,
        help="Optional baseline gate JSON for zero-regression comparison.",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Optional output JSON report path.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=6,
        help="ProcessPoolExecutor worker count (default: 6).",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Simulation frame rate Hz (default: 30.0).",
    )
    ap.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip forced C++17 recompile/test (only for quick YAML-only checks).",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    res = run_gate(
        mode=args.mode,
        fork_yaml=args.fork_yaml,
        roundabout_yaml=args.roundabout_yaml,
        baseline_json=args.baseline_json,
        out_json=args.out_json,
        workers=args.workers,
        rate_hz=args.fps,
        skip_compile=args.skip_compile,
        verbose=True,
    )
    return 0 if res["verdict"] == "ACCEPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
