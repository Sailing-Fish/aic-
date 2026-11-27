#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unattended Robust Domain-Randomized Auto-Tuner for Windows PC Simulator.

Features:
  - Anti-overfit worst-case domain randomization suite:
      1. Nominal (1.0x brake, +0ms latency, 1.0x steer/speed, random props seed A)
      2. Slippery floor (a_eff * 0.75, random props seed B)
      3. Extra latency (+40ms command latency, random props seed C)
      4. Actuation mismatch A (steer * 0.85, speed * 1.15, random props seed D)
      5. Actuation mismatch B (steer * 1.15, speed * 0.85, random props seed E)
    Every robust-passing candidate MUST achieve 0 penalty & 0 DNF across ALL trials.
  - Frozen safety margin floors (strictly enforced in search space & output validation):
      * border_margin_m        >= 0.020
      * no_entry_margin_m      >= 0.025
      * yellow_bumper_margin_m >= 0.070
  - Outputs TWO candidate YAML profiles per map in --out-dir:
      * run1_conservative.yaml (safety-first, conservative speed & higher margins)
      * run2_aggressive.yaml   (fastest zero-penalty worst-case / mean net score)
      * diff_vs_base.txt       (human-readable parameter diff vs base finals_*.yaml)
      * results.json           (full evaluation log)
  - NEVER touches tools/config/finals_*.yaml unless --apply is explicitly passed!
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Guard against tools/sim/profile.py shadowing Python stdlib `profile`
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[1]
_this_dir_norm = os.path.normcase(str(_THIS_DIR))
sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or ".")) != _this_dir_norm]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from tools.sim.pc_sim import (
    SimConfig,
    SimRunResult,
    load_yaml_params,
    resolve_default_yaml,
    run_episode,
)

# Frozen safety margin floors — NEVER violated by any candidate or generated profile
SAFETY_FLOORS: Dict[str, float] = {
    "border_margin_m": 0.020,
    "no_entry_margin_m": 0.025,
    "yellow_bumper_margin_m": 0.070,
}

# Search space bounds: key -> (low, high, decimals)
SEARCH_BOUNDS: Dict[str, Tuple[float, float, int]] = {
    "target_speed": (0.34, 0.58, 2),
    "max_yaw_rate": (1.45, 2.20, 2),
    "speed_gain_damping": (0.30, 0.72, 2),
    "corridor_kp_y": (2.80, 5.60, 2),
    "corridor_heading_gain": (0.00, 1.05, 2),
    "obstacle_speed_mps": (0.20, 0.34, 2),
    "digit_approach_speed_mps": (0.24, 0.40, 2),
    "yellow_pre_brake_dist_m": (0.62, 0.85, 2),
    "yellow_creep_speed_mps": (0.18, 0.28, 2),
    "border_margin_m": (SAFETY_FLOORS["border_margin_m"], 0.035, 3),
    "no_entry_margin_m": (SAFETY_FLOORS["no_entry_margin_m"], 0.045, 3),
    "yellow_bumper_margin_m": (SAFETY_FLOORS["yellow_bumper_margin_m"], 0.110, 3),
}


@dataclass
class DRScenario:
    tag: str
    a_eff_scale: float = 1.0
    latency_ms: float = 0.0
    steer_scale: float = 1.0
    speed_scale: float = 1.0
    voltage: float = 12.0
    prop_seed: int = 42
    cones: int = 1
    no_entry: int = 1
    digits: int = 2
    random_yellow: int = 1


@dataclass
class TrialSummary:
    tag: str
    passed: bool
    finished: bool
    dnf: bool
    done_reason: str
    lap_time_s: float
    total_penalty_s: float
    total_bonus_s: float
    net_score_s: float
    max_s_m: float
    min_v_mps: float


@dataclass
class CandidateEval:
    idx: int
    label: str
    overrides: Dict[str, Any]
    robust_pass: bool
    trials_passed: int
    trials_total: int
    worst_net_score_s: float
    mean_net_score_s: float
    worst_lap_time_s: float
    mean_lap_time_s: float
    total_penalty_sum_s: float
    safety_margin_score: float
    wall_time_s: float
    trials: List[TrialSummary] = field(default_factory=list)


def enforce_safety_floors(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp all safety margin parameters to strictly satisfy SAFETY_FLOORS."""
    out = dict(params)
    for key, floor_val in SAFETY_FLOORS.items():
        if key in out:
            out[key] = round(max(float(out[key]), floor_val), 4)
    # Keep max_speed >= target_speed + 0.15 so speed cap never silently clips target_speed
    if "target_speed" in out:
        ts = float(out["target_speed"])
        cur_max = float(out.get("max_speed", max(0.80, ts + 0.25)))
        out["max_speed"] = round(max(cur_max, min(0.95, ts + 0.20)), 2)
    return out


def validate_safety_floors(params: Dict[str, Any]) -> None:
    """Assert that every safety floor is strictly met."""
    for key, floor_val in SAFETY_FLOORS.items():
        if key in params:
            val = float(params[key])
            if val < floor_val - 1e-9:
                raise ValueError(
                    f"Safety floor violated: {key}={val} < required floor {floor_val}"
                )


def build_dr_suite(seed: int = 42, prop_density: str = "standard") -> List[DRScenario]:
    """Build the 5-trial anti-overfit domain randomization worst-case suite."""
    if prop_density == "light":
        cones, no_entry, digits, r_yel = 1, 0, 2, 0
    else:
        cones, no_entry, digits, r_yel = 1, 1, 2, 1

    return [
        DRScenario(
            tag="nominal",
            a_eff_scale=1.0,
            latency_ms=0.0,
            steer_scale=1.0,
            speed_scale=1.0,
            voltage=12.0,
            prop_seed=seed + 10,
            cones=cones,
            no_entry=no_entry,
            digits=digits,
            random_yellow=r_yel,
        ),
        DRScenario(
            tag="slippery_0.75",
            a_eff_scale=0.75,
            latency_ms=0.0,
            steer_scale=1.0,
            speed_scale=1.0,
            voltage=11.4,
            prop_seed=seed + 20,
            cones=cones,
            no_entry=no_entry,
            digits=digits,
            random_yellow=r_yel,
        ),
        DRScenario(
            tag="latency_+40ms",
            a_eff_scale=1.0,
            latency_ms=40.0,
            steer_scale=1.0,
            speed_scale=1.0,
            voltage=12.0,
            prop_seed=seed + 30,
            cones=cones,
            no_entry=no_entry,
            digits=digits,
            random_yellow=r_yel,
        ),
        DRScenario(
            tag="mismatch_s0.85_v1.15",
            a_eff_scale=1.0,
            latency_ms=10.0,
            steer_scale=0.85,
            speed_scale=1.15,
            voltage=12.2,
            prop_seed=seed + 40,
            cones=cones,
            no_entry=no_entry,
            digits=digits,
            random_yellow=r_yel,
        ),
        DRScenario(
            tag="mismatch_s1.15_v0.85",
            a_eff_scale=1.0,
            latency_ms=10.0,
            steer_scale=1.15,
            speed_scale=0.85,
            voltage=11.6,
            prop_seed=seed + 50,
            cones=cones,
            no_entry=no_entry,
            digits=digits,
            random_yellow=r_yel,
        ),
    ]


def _extract_tunable(base_params: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, (lo, hi, dec) in SEARCH_BOUNDS.items():
        val = float(base_params.get(k, 0.5 * (lo + hi)))
        out[k] = round(max(lo, min(hi, val)), dec)
    return enforce_safety_floors(out)


def _sample_candidate(
    rng: random.Random,
    anchor: Optional[Dict[str, Any]] = None,
    local_scale: float = 0.25,
) -> Dict[str, Any]:
    cand: Dict[str, Any] = {}
    for k, (lo, hi, dec) in SEARCH_BOUNDS.items():
        if anchor is not None and k in anchor and rng.random() < 0.75:
            span = (hi - lo) * local_scale
            val = float(anchor[k]) + rng.uniform(-span, span)
        else:
            val = rng.uniform(lo, hi)
        val = max(lo, min(hi, val))
        cand[k] = round(val, dec)
    return enforce_safety_floors(cand)


def _compute_safety_margin_score(params: Dict[str, Any], base_speed: float) -> float:
    """Higher = more conservative safety margin & speed headroom."""
    bm = float(params.get("border_margin_m", SAFETY_FLOORS["border_margin_m"]))
    nm = float(params.get("no_entry_margin_m", SAFETY_FLOORS["no_entry_margin_m"]))
    ym = float(params.get("yellow_bumper_margin_m", SAFETY_FLOORS["yellow_bumper_margin_m"]))
    ts = float(params.get("target_speed", base_speed))
    margin_term = (
        (bm - SAFETY_FLOORS["border_margin_m"]) / 0.015
        + (nm - SAFETY_FLOORS["no_entry_margin_m"]) / 0.020
        + (ym - SAFETY_FLOORS["yellow_bumper_margin_m"]) / 0.040
    )
    speed_term = max(0.0, (0.55 - ts) / 0.20)
    return round(margin_term + 1.2 * speed_term, 4)


def evaluate_candidate(
    idx: int,
    label: str,
    overrides: Dict[str, Any],
    track: str,
    yaml_path: Path,
    base_params: Dict[str, Any],
    dr_suite: Sequence[DRScenario],
    rate_hz: float = 30.0,
    max_time_s: float = 48.0,
    fast_fail: bool = True,
) -> CandidateEval:
    """Evaluate one candidate parameter set across the full domain-randomized suite."""
    clean_ov = enforce_safety_floors(overrides)
    validate_safety_floors(clean_ov)

    trials: List[TrialSummary] = []
    t0 = time.perf_counter()

    for sc in dr_suite:
        cfg = SimConfig(
            track=track,
            yaml_path=str(yaml_path),
            param_overrides=clean_ov,
            detector="mock",
            rate_hz=rate_hz,
            max_time_s=max_time_s,
            seed=sc.prop_seed,
            cones=sc.cones,
            no_entry=sc.no_entry,
            digits=sc.digits,
            random_yellow=sc.random_yellow,
            prop_seed=sc.prop_seed,
            include_base_yellow=True,
            a_eff_scale=sc.a_eff_scale,
            latency_ms=sc.latency_ms,
            steer_scale=sc.steer_scale,
            speed_scale=sc.speed_scale,
            voltage=sc.voltage,
        )
        res = run_episode(cfg)
        passed = bool(res.finished and (not res.dnf) and (res.total_penalty_s <= 1e-6))
        trials.append(
            TrialSummary(
                tag=sc.tag,
                passed=passed,
                finished=res.finished,
                dnf=res.dnf,
                done_reason=res.done_reason,
                lap_time_s=round(res.lap_time_s, 3),
                total_penalty_s=round(res.total_penalty_s, 3),
                total_bonus_s=round(res.total_bonus_s, 2),
                net_score_s=round(res.net_score_s, 3),
                max_s_m=round(res.max_s_m, 3),
                min_v_mps=round(res.min_v_mps, 4),
            )
        )
        if fast_fail and not passed:
            break

    wall_s = time.perf_counter() - t0
    trials_passed = sum(1 for t in trials if t.passed)
    robust_pass = (trials_passed == len(dr_suite))

    # For pruned candidates, reflect large sentinel net_score so ranking strictly prefers robust passes
    if robust_pass:
        worst_net = max(t.net_score_s for t in trials)
        mean_net = sum(t.net_score_s for t in trials) / len(trials)
        worst_lap = max(t.lap_time_s for t in trials)
        mean_lap = sum(t.lap_time_s for t in trials) / len(trials)
    else:
        worst_net = max(t.net_score_s for t in trials) + 100.0 * (len(dr_suite) - trials_passed)
        mean_net = (sum(t.net_score_s for t in trials) / max(1, len(trials))) + 100.0 * (
            len(dr_suite) - trials_passed
        )
        worst_lap = max(t.lap_time_s for t in trials)
        mean_lap = sum(t.lap_time_s for t in trials) / max(1, len(trials))

    pen_sum = sum(t.total_penalty_s for t in trials)
    margin_score = _compute_safety_margin_score(clean_ov, float(base_params.get("target_speed", 0.42)))

    return CandidateEval(
        idx=idx,
        label=label,
        overrides=clean_ov,
        robust_pass=robust_pass,
        trials_passed=trials_passed,
        trials_total=len(dr_suite),
        worst_net_score_s=round(worst_net, 3),
        mean_net_score_s=round(mean_net, 3),
        worst_lap_time_s=round(worst_lap, 3),
        mean_lap_time_s=round(mean_lap, 3),
        total_penalty_sum_s=round(pen_sum, 3),
        safety_margin_score=margin_score,
        wall_time_s=round(wall_s, 2),
        trials=trials,
    )


def update_yaml_text_preserving_comments(
    base_yaml_text: str,
    overrides: Dict[str, Any],
    section: str = "line_follower_node",
) -> str:
    """Update scalar parameter values inside `section` while preserving YAML comments/layout."""
    lines = base_yaml_text.splitlines()
    cur_section: Optional[str] = None
    updated_keys = set()
    out_lines: List[str] = []

    for ln in lines:
        if ln and not ln[0].isspace() and ln.rstrip().endswith(":"):
            cur_section = ln.strip()[:-1]
        if cur_section == section:
            m = re.match(r"^(\s*)([A-Za-z0-9_]+)(\s*:\s*)([^\s#]+)(.*)$", ln)
            if m and m.group(2) in overrides:
                key = m.group(2)
                val = overrides[key]
                if isinstance(val, bool):
                    val_str = "true" if val else "false"
                elif isinstance(val, float):
                    val_str = f"{val:.4f}".rstrip("0").rstrip(".")
                    if "." not in val_str:
                        val_str += ".0"
                else:
                    val_str = str(val)
                ln = f"{m.group(1)}{key}{m.group(3)}{val_str}{m.group(5)}"
                updated_keys.add(key)
        out_lines.append(ln)

    text = "\n".join(out_lines) + "\n"
    # Validate resulting YAML parses cleanly and satisfies safety floors
    parsed = yaml.safe_load(text)
    ros_params = parsed[section]["ros__parameters"]
    validate_safety_floors(ros_params)
    return text


def format_diff_vs_base(
    base_params: Dict[str, Any],
    run1: CandidateEval,
    run2: CandidateEval,
) -> str:
    keys = sorted(set(run1.overrides.keys()) | set(run2.overrides.keys()))
    lines = [
        "=" * 92,
        f"{'PARAMETER':<28} {'BASE':>10} {'RUN1_CONS':>12} {'RUN2_AGGR':>12} {'FLOOR':>10}",
        "-" * 92,
    ]
    for k in keys:
        b = base_params.get(k, "-")
        v1 = run1.overrides.get(k, b)
        v2 = run2.overrides.get(k, b)
        fl = f">={SAFETY_FLOORS[k]:.3f}" if k in SAFETY_FLOORS else "-"
        b_s = f"{b:.4g}" if isinstance(b, (int, float)) else str(b)
        v1_s = f"{v1:.4g}" if isinstance(v1, (int, float)) else str(v1)
        v2_s = f"{v2:.4g}" if isinstance(v2, (int, float)) else str(v2)
        mark = "*" if (v1_s != b_s or v2_s != b_s) else " "
        lines.append(f"{mark} {k:<26} {b_s:>10} {v1_s:>12} {v2_s:>12} {fl:>10}")
    lines.append("-" * 92)
    lines.append(
        f"Run 1 (Conservative): [{run1.label}] pass={run1.trials_passed}/{run1.trials_total} "
        f"worst_net={run1.worst_net_score_s:.2f}s mean_net={run1.mean_net_score_s:.2f}s "
        f"safety_score={run1.safety_margin_score:.2f}"
    )
    lines.append(
        f"Run 2 (Aggressive)  : [{run2.label}] pass={run2.trials_passed}/{run2.trials_total} "
        f"worst_net={run2.worst_net_score_s:.2f}s mean_net={run2.mean_net_score_s:.2f}s "
        f"safety_score={run2.safety_margin_score:.2f}"
    )
    lines.append("=" * 92)
    return "\n".join(lines)


def select_run1_and_run2(
    evals: Sequence[CandidateEval],
    base_params: Dict[str, Any],
) -> Tuple[CandidateEval, CandidateEval]:
    """Pick Run 1 (Conservative) and Run 2 (Aggressive) from evaluated candidates."""
    robust = [e for e in evals if e.robust_pass]
    pool = robust if robust else sorted(
        evals, key=lambda e: (-e.trials_passed, e.total_penalty_sum_s, e.worst_net_score_s)
    )

    # Run 2 (Aggressive): minimize worst-case net score, tie-break mean net score
    run2 = min(pool, key=lambda e: (e.worst_net_score_s, e.mean_net_score_s))

    # Run 1 (Conservative): among robust passes (or top pool), prefer higher safety margin
    # and conservative speed without catastrophic slowdown
    base_spd = float(base_params.get("target_speed", 0.42))
    if robust:
        cons_pool = [
            e for e in robust if float(e.overrides.get("target_speed", base_spd)) <= base_spd + 0.04
        ] or robust
        # Maximize safety_margin_score while keeping worst_net_score reasonable
        run1 = max(
            cons_pool,
            key=lambda e: (e.safety_margin_score - 0.05 * e.worst_net_score_s, -e.worst_net_score_s),
        )
    else:
        # Fallback if budget was too small to find a full 5/5 pass: pick highest trials_passed + safety
        best_pass = max(e.trials_passed for e in pool)
        top_pass = [e for e in pool if e.trials_passed == best_pass]
        run1 = max(top_pass, key=lambda e: (e.safety_margin_score, -e.total_penalty_sum_s))

    return run1, run2


def run_auto_tune(
    track: str,
    budget: int,
    out_dir: Path,
    seed: int = 42,
    rate_hz: float = 30.0,
    max_time_s: float = 48.0,
    yaml_path: Optional[Path] = None,
    apply_profile: Optional[str] = None,
    prop_density: str = "standard",
    verbose: bool = True,
) -> Dict[str, Any]:
    base_yaml = Path(yaml_path) if yaml_path else resolve_default_yaml(track)
    base_text = base_yaml.read_text(encoding="utf-8")
    base_params = load_yaml_params(base_yaml)
    base_tunable = _extract_tunable(base_params)
    dr_suite = build_dr_suite(seed=seed, prop_density=prop_density)

    rng = random.Random(seed)
    evals: List[CandidateEval] = []
    budget = max(1, int(budget))

    # Build seeded initial proposals: base, conservative, aggressive, then adaptive search
    cons_seed = enforce_safety_floors(
        {
            **base_tunable,
            "target_speed": round(max(0.36, base_tunable["target_speed"] - 0.04), 2),
            "corridor_kp_y": round(min(5.20, base_tunable["corridor_kp_y"] + 0.70), 2),
            "corridor_heading_gain": round(max(0.15, base_tunable["corridor_heading_gain"] - 0.55), 2),
            "obstacle_speed_mps": round(max(0.20, base_tunable["obstacle_speed_mps"] - 0.02), 2),
            "border_margin_m": round(max(0.028, base_tunable["border_margin_m"] + 0.003), 3),
            "no_entry_margin_m": round(max(0.034, base_tunable["no_entry_margin_m"] + 0.004), 3),
            "yellow_bumper_margin_m": round(max(0.088, base_tunable["yellow_bumper_margin_m"] + 0.008), 3),
            "yellow_pre_brake_dist_m": round(min(0.85, base_tunable["yellow_pre_brake_dist_m"] + 0.06), 2),
        }
    )
    aggr_seed = enforce_safety_floors(
        {
            **base_tunable,
            "target_speed": round(min(0.56, base_tunable["target_speed"] + 0.06), 2),
            "speed_gain_damping": round(max(0.32, base_tunable["speed_gain_damping"] - 0.08), 2),
            "obstacle_speed_mps": round(min(0.32, base_tunable["obstacle_speed_mps"] + 0.03), 2),
            "digit_approach_speed_mps": round(min(0.38, base_tunable["digit_approach_speed_mps"] + 0.04), 2),
            "border_margin_m": max(SAFETY_FLOORS["border_margin_m"], 0.022),
            "no_entry_margin_m": max(SAFETY_FLOORS["no_entry_margin_m"], 0.027),
            "yellow_bumper_margin_m": max(SAFETY_FLOORS["yellow_bumper_margin_m"], 0.075),
        }
    )

    preset_queue: List[Tuple[str, Dict[str, Any]]] = [
        ("baseline", base_tunable),
        ("seed_conservative", cons_seed),
        ("seed_aggressive", aggr_seed),
    ]

    if verbose:
        print(
            f"[auto_tune_pc] track={track} yaml={base_yaml.name} budget={budget} "
            f"dr_trials={len(dr_suite)} floors={SAFETY_FLOORS}",
            flush=True,
        )

    for idx in range(budget):
        if idx < len(preset_queue):
            label, cand = preset_queue[idx]
        else:
            robust_so_far = [e for e in evals if e.robust_pass]
            if robust_so_far and (idx % 2 == 1):
                best_anchor = min(robust_so_far, key=lambda e: (e.worst_net_score_s, e.mean_net_score_s))
                label = f"local_refine_{idx:02d}"
                cand = _sample_candidate(rng, anchor=best_anchor.overrides, local_scale=0.22)
            else:
                label = f"explore_{idx:02d}"
                cand = _sample_candidate(rng, anchor=base_tunable, local_scale=0.45)

        ev = evaluate_candidate(
            idx=idx,
            label=label,
            overrides=cand,
            track=track,
            yaml_path=base_yaml,
            base_params=base_params,
            dr_suite=dr_suite,
            rate_hz=rate_hz,
            max_time_s=max_time_s,
            fast_fail=True,
        )
        evals.append(ev)
        if verbose:
            st = "PASS" if ev.robust_pass else f"FAIL({ev.trials_passed}/{ev.trials_total})"
            print(
                f"  [{idx+1:02d}/{budget:02d}] {label:<18} {st:<10} "
                f"v={ev.overrides['target_speed']:.2f} "
                f"margins=({ev.overrides['border_margin_m']:.3f},{ev.overrides['no_entry_margin_m']:.3f},{ev.overrides['yellow_bumper_margin_m']:.3f}) "
                f"worst={ev.worst_net_score_s:6.2f}s mean={ev.mean_net_score_s:6.2f}s ({ev.wall_time_s:.1f}s)",
                flush=True,
            )

    run1, run2 = select_run1_and_run2(evals, base_params)

    out_dir.mkdir(parents=True, exist_ok=True)
    run1_yaml_text = update_yaml_text_preserving_comments(base_text, run1.overrides)
    run2_yaml_text = update_yaml_text_preserving_comments(base_text, run2.overrides)

    run1_path = out_dir / "run1_conservative.yaml"
    run2_path = out_dir / "run2_aggressive.yaml"
    diff_path = out_dir / "diff_vs_base.txt"
    results_path = out_dir / "results.json"

    run1_path.write_text(run1_yaml_text, encoding="utf-8")
    run2_path.write_text(run2_yaml_text, encoding="utf-8")

    diff_txt = format_diff_vs_base(base_params, run1, run2)
    diff_path.write_text(diff_txt + "\n", encoding="utf-8")

    if verbose:
        print("\n" + diff_txt, flush=True)
        print(f"[auto_tune_pc] Wrote:\n  - {run1_path}\n  - {run2_path}\n  - {diff_path}", flush=True)

    applied_to: Optional[str] = None
    if apply_profile is not None:
        chosen_text = run1_yaml_text if apply_profile == "run1" else run2_yaml_text
        base_yaml.write_text(chosen_text, encoding="utf-8")
        applied_to = str(base_yaml)
        if verbose:
            print(f"[auto_tune_pc] --apply enabled: updated {base_yaml} with {apply_profile} profile.", flush=True)

    summary = {
        "track": track,
        "base_yaml": str(base_yaml),
        "budget": budget,
        "safety_floors": SAFETY_FLOORS,
        "dr_suite": [sc.__dict__ for sc in dr_suite],
        "robust_pass_count": sum(1 for e in evals if e.robust_pass),
        "run1_conservative": {
            "path": str(run1_path),
            "label": run1.label,
            "robust_pass": run1.robust_pass,
            "trials_passed": run1.trials_passed,
            "trials_total": run1.trials_total,
            "worst_net_score_s": run1.worst_net_score_s,
            "mean_net_score_s": run1.mean_net_score_s,
            "overrides": run1.overrides,
        },
        "run2_aggressive": {
            "path": str(run2_path),
            "label": run2.label,
            "robust_pass": run2.robust_pass,
            "trials_passed": run2.trials_passed,
            "trials_total": run2.trials_total,
            "worst_net_score_s": run2.worst_net_score_s,
            "mean_net_score_s": run2.mean_net_score_s,
            "overrides": run2.overrides,
        },
        "diff_path": str(diff_path),
        "applied_to": applied_to,
        "candidates": [
            {
                "idx": e.idx,
                "label": e.label,
                "robust_pass": e.robust_pass,
                "trials_passed": e.trials_passed,
                "trials_total": e.trials_total,
                "worst_net_score_s": e.worst_net_score_s,
                "mean_net_score_s": e.mean_net_score_s,
                "total_penalty_sum_s": e.total_penalty_sum_s,
                "safety_margin_score": e.safety_margin_score,
                "wall_time_s": e.wall_time_s,
                "overrides": e.overrides,
                "trials": [t.__dict__ for t in e.trials],
            }
            for e in evals
        ],
    }
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unattended Domain-Randomized Robust Auto-Tuner for AIC PC Simulator"
    )
    ap.add_argument("--track", choices=("fork_s", "roundabout"), default="fork_s", help="Target official map")
    ap.add_argument("--budget", type=int, default=12, help="Number of candidate evaluations (includes baseline)")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for run1_conservative.yaml, run2_aggressive.yaml, diff_vs_base.txt, results.json",
    )
    ap.add_argument("--yaml", default=None, help="Optional custom base YAML path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--fps", type=float, default=30.0, help="Simulation frame rate (Hz)")
    ap.add_argument("--max-time", type=float, default=48.0, help="Per-trial max simulation time (s)")
    ap.add_argument(
        "--prop-density",
        choices=("standard", "light"),
        default="standard",
        help="Domain randomization prop density",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="OPT-IN ONLY: overwrite tools/config/finals_<track>.yaml with selected profile",
    )
    ap.add_argument(
        "--apply-profile",
        choices=("run1", "run2"),
        default="run1",
        help="Which profile to apply when --apply is passed (default: run1 conservative)",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (REPO_ROOT / "tools" / "sim" / "profiles" / f"auto_tune_{args.track}")
    )

    run_auto_tune(
        track=args.track,
        budget=args.budget,
        out_dir=out_dir,
        seed=args.seed,
        rate_hz=args.fps,
        max_time_s=args.max_time,
        yaml_path=Path(args.yaml) if args.yaml else None,
        apply_profile=args.apply_profile if args.apply else None,
        prop_density=args.prop_density,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
