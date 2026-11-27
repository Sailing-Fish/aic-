#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4-Scenario Real-Car Crash Regression Suite & Multi-Seed Stress Harness.

Codifies the 4 real-car failure modes as deterministic scenarios on official maps:
  1. roundabout_red_to_sharp_right : roundabout.json with red cone/no-entry immediately
     before the 180-deg sharp right curve -> 0 off-track, 0 collision, completes curve.
  2. fork_y_merge_anti_uturn       : fork_s.json through Y-junction entry and Y-merge exit
     -> forward progress past merge, zero U-turn/reversal, zero DNF.
  3. straight_dual_digit_5_2       : fork_s.json straight with closely placed dual digits
     5 and 2 -> clean target lock, bounded yaw oscillation, zero off-track.
  4. yellow_stop_no_reverse        : fork_s.json approach to yellow stop line -> clean stop
     before yellow line, zero reverse velocity (min_v >= -0.02 m/s), zero yellow penalty.

Plus optional randomized multi-seed full-lap stress tests (--seeds N) on both maps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Guard against tools/sim/profile.py shadowing Python stdlib `profile`
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[1]
_this_dir_norm = os.path.normcase(str(_THIS_DIR))
sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or ".")) != _this_dir_norm]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sim.pc_sim import (
    SimConfig,
    SimRunResult,
    load_params_json,
    parse_overrides_list,
    run_episode,
)
from tools.sim.sim_world_pc import StepInfo


@dataclass
class ScenarioSpec:
    name: str
    category: str  # "regression" | "stress"
    description: str
    config: SimConfig
    stop_condition: Optional[Callable[[StepInfo, Dict[str, Any]], bool]]
    evaluator: Callable[[SimRunResult], Tuple[bool, str]]


@dataclass
class ScenarioReport:
    name: str
    category: str
    track: str
    passed: bool
    reason: str
    sim_time_s: float
    wall_time_s: float
    net_score_s: float
    total_penalty_s: float
    total_bonus_s: float
    min_v_mps: float
    max_s_m: float
    w_cmd_std: float
    w_sign_flips: int
    dnf: bool
    finished: bool
    failure_video: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


def build_regression_scenarios(
    detector: str = "mock",
    rate_hz: float = 30.0,
    param_overrides: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> List[ScenarioSpec]:
    """Construct the 4 deterministic real-car crash regression scenarios."""
    overrides = dict(param_overrides or {})
    scenarios: List[ScenarioSpec] = []

    # 1. roundabout_red_to_sharp_right
    # On roundabout.json, s=3.084m is yellow #2; s=3.20..4.35m is straight heading west;
    # s=4.40..6.20m is the 180-deg sharp right U-turn; yellow #3 is at s=6.489m.
    # Place a red cone on the inside (right, lat=-95mm) right before the sharp right curve,
    # plus a red no-entry earlier on the opposite side to stress obstacle-to-curve transition.
    cfg_r1 = SimConfig(
        track="roundabout",
        param_overrides=overrides,
        detector=detector,
        rate_hz=rate_hz,
        max_time_s=25.0,
        seed=seed,
        start_pose={"s_m": 3.22, "lat_mm": 0.0},
        explicit_props=[
            {"type": "red_cone", "s_m": 3.92, "lat_mm": -95.0, "id": "reg_cone_before_right"},
        ],
        include_base_yellow=True,
    )

    def _stop_r1(info: StepInfo, _brain: Dict[str, Any]) -> bool:
        return float(info.referee_status["s_m"]) >= 6.25

    def _eval_r1(res: SimRunResult) -> Tuple[bool, str]:
        errs: List[str] = []
        if res.dnf:
            errs.append(f"DNF({res.dnf_reason})")
        if res.max_s_m < 6.15:
            errs.append(f"incomplete_curve(s={res.max_s_m:.2f}m<6.15m)")
        if res.boundary_penalty_s > 1e-6:
            errs.append(f"off_track({res.boundary_press_time_s:.2f}s,+{res.boundary_penalty_s:.1f}s)")
        if res.cone_hits > 0 or res.no_entry_hits > 0:
            errs.append(f"collision(cones={res.cone_hits},no_entry={res.no_entry_hits})")
        if errs:
            return False, "; ".join(errs)
        return True, f"curve_ok s={res.max_s_m:.2f}m pen=0.0s"

    scenarios.append(
        ScenarioSpec(
            name="roundabout_red_to_sharp_right",
            category="regression",
            description="Roundabout red cone right before 180-deg sharp right curve -> 0 off-track, 0 collision",
            config=cfg_r1,
            stop_condition=_stop_r1,
            evaluator=_eval_r1,
        )
    )

    # 2. fork_y_merge_anti_uturn
    # On fork_s.json, s=2.332m is yellow #2; Y-junction entry is around s=3.1..3.6m;
    # inner vertical leg is s=3.7..4.6m; Y-merge exit is around s=4.7..5.4m; upper S-curve starts ~5.5m.
    cfg_r2 = SimConfig(
        track="fork_s",
        param_overrides=overrides,
        detector=detector,
        rate_hz=rate_hz,
        max_time_s=28.0,
        seed=seed,
        start_pose={"s_m": 2.48, "lat_mm": 0.0},
        explicit_props=[],
        include_base_yellow=True,
    )

    def _stop_r2(info: StepInfo, _brain: Dict[str, Any]) -> bool:
        return float(info.referee_status["s_m"]) >= 6.00

    def _eval_r2(res: SimRunResult) -> Tuple[bool, str]:
        errs: List[str] = []
        if res.dnf:
            errs.append(f"DNF({res.dnf_reason})")
        if res.max_s_m < 5.80:
            errs.append(f"did_not_exit_merge(s={res.max_s_m:.2f}m<5.80m)")
        if res.max_s_backtrack_m > 0.25:
            errs.append(f"uturn_backtrack({res.max_s_backtrack_m:.2f}m)")
        if res.max_heading_diff_deg > 110.0:
            errs.append(f"heading_reversal({res.max_heading_diff_deg:.1f}deg)")
        if res.min_v_mps < -0.05:
            errs.append(f"reverse_vel({res.min_v_mps:+.3f}m/s)")
        if errs:
            return False, "; ".join(errs)
        return True, f"merge_ok s={res.max_s_m:.2f}m backtrack={res.max_s_backtrack_m:.2f}m hdg_max={res.max_heading_diff_deg:.0f}deg"

    scenarios.append(
        ScenarioSpec(
            name="fork_y_merge_anti_uturn",
            category="regression",
            description="Fork Y-junction entry + merge exit -> forward continuation, 0 U-turn, 0 DNF",
            config=cfg_r2,
            stop_condition=_stop_r2,
            evaluator=_eval_r2,
        )
    )

    # 3. straight_dual_digit_5_2
    # On fork_s.json, s=0.85..2.20m is a pure straight between yellow #1 (0.774m) and yellow #2 (2.332m).
    # Place closely spaced dual digits 5 (left) and 2 (right).
    cfg_r3 = SimConfig(
        track="fork_s",
        param_overrides=overrides,
        detector=detector,
        rate_hz=rate_hz,
        max_time_s=15.0,
        seed=seed,
        start_pose={"s_m": 0.88, "lat_mm": 0.0},
        explicit_props=[
            {"type": "digit", "s_m": 1.65, "lat_mm": 85.0, "value": 5, "id": "reg_digit_5"},
            {"type": "digit", "s_m": 1.72, "lat_mm": -85.0, "value": 2, "id": "reg_digit_2"},
        ],
        include_base_yellow=True,
    )

    def _stop_r3(info: StepInfo, _brain: Dict[str, Any]) -> bool:
        return float(info.referee_status["s_m"]) >= 2.15

    def _eval_r3(res: SimRunResult) -> Tuple[bool, str]:
        errs: List[str] = []
        if res.dnf:
            errs.append(f"DNF({res.dnf_reason})")
        if res.max_s_m < 2.08:
            errs.append(f"incomplete_straight(s={res.max_s_m:.2f}m)")
        if res.boundary_penalty_s > 1e-6:
            errs.append(f"off_track(+{res.boundary_penalty_s:.1f}s)")
        locked_target = (5 in res.active_digits_locked) or (5 in res.bonus_digits_collected)
        if not locked_target:
            errs.append(f"missed_digit5(locked={res.active_digits_locked},got={res.bonus_digits_collected})")
        if res.w_cmd_std > 0.70 or res.w_sign_flips > 8:
            errs.append(f"yaw_oscillation(std={res.w_cmd_std:.2f},flips={res.w_sign_flips})")
        if errs:
            return False, "; ".join(errs)
        return (
            True,
            f"lock={res.active_digits_locked} got={res.bonus_digits_collected} w_std={res.w_cmd_std:.2f} flips={res.w_sign_flips}",
        )

    scenarios.append(
        ScenarioSpec(
            name="straight_dual_digit_5_2",
            category="regression",
            description="Straight with closely placed dual digits 5 & 2 -> clean lock on 5, bounded yaw, 0 off-track",
            config=cfg_r3,
            stop_condition=_stop_r3,
            evaluator=_eval_r3,
        )
    )

    # 4. yellow_stop_no_reverse
    # On fork_s.json, start at s=1.00m approaching yellow #2 at s=2.332m at cruising speed.
    cfg_r4 = SimConfig(
        track="fork_s",
        param_overrides=overrides,
        detector=detector,
        rate_hz=rate_hz,
        max_time_s=18.0,
        seed=seed,
        start_pose={"s_m": 1.00, "lat_mm": 0.0},
        explicit_props=[],
        include_base_yellow=True,
    )

    def _stop_r4(info: StepInfo, _brain: Dict[str, Any]) -> bool:
        s_track = float(info.referee_status["s_m"])
        return (
            int(info.referee_status.get("yellow_stops_ok", 0)) >= 1
            and s_track >= 2.55
        ) or s_track >= 2.75

    def _eval_r4(res: SimRunResult) -> Tuple[bool, str]:
        errs: List[str] = []
        if res.dnf:
            errs.append(f"DNF({res.dnf_reason})")
        if res.yellow_violations > 0:
            errs.append(f"yellow_penalty(+{res.yellow_violations * 15}s)")
        if res.yellow_stops_ok < 1:
            errs.append("no_clean_stop")
        if res.min_v_mps < -0.02:
            errs.append(f"reverse_overshoot(min_v={res.min_v_mps:+.3f}<-0.02)")
        if res.max_s_m < 2.50:
            errs.append(f"stuck_after_stop(s={res.max_s_m:.2f}m)")
        if errs:
            return False, "; ".join(errs)
        return True, f"stop_ok={res.yellow_stops_ok} min_v={res.min_v_mps:+.4f}m/s yel_pen=0"

    scenarios.append(
        ScenarioSpec(
            name="yellow_stop_no_reverse",
            category="regression",
            description="Approach to yellow stop line -> clean stop before line, min_v >= -0.02 m/s, 0 penalty",
            config=cfg_r4,
            stop_condition=_stop_r4,
            evaluator=_eval_r4,
        )
    )

    return scenarios


def build_stress_scenarios(
    seeds: int,
    detector: str = "mock",
    rate_hz: float = 30.0,
    param_overrides: Optional[Dict[str, Any]] = None,
    base_seed: int = 100,
) -> List[ScenarioSpec]:
    """Construct randomized multi-seed full-lap stress tests on both fork_s and roundabout."""
    overrides = dict(param_overrides or {})
    scenarios: List[ScenarioSpec] = []
    for track in ("fork_s", "roundabout"):
        for i in range(max(0, int(seeds))):
            sd = base_seed + i
            cfg = SimConfig(
                track=track,
                param_overrides=overrides,
                detector=detector,
                rate_hz=rate_hz,
                max_time_s=60.0,
                seed=sd,
                cones=2,
                no_entry=1,
                digits=3,
                random_yellow=1,
                prop_seed=sd,
                include_base_yellow=True,
            )

            def _eval_stress(res: SimRunResult) -> Tuple[bool, str]:
                errs: List[str] = []
                if res.dnf:
                    errs.append(f"DNF({res.dnf_reason})")
                if not res.finished:
                    errs.append(f"not_finished({res.done_reason},s={res.final_s_m:.2f}m)")
                if res.total_penalty_s > 1e-6:
                    errs.append(
                        f"pen={res.total_penalty_s:.1f}s(line={res.boundary_penalty_s:.1f},"
                        f"c={res.cone_hits},ne={res.no_entry_hits},yv={res.yellow_violations})"
                    )
                if errs:
                    return False, "; ".join(errs)
                return (
                    True,
                    f"FINISH net={res.net_score_s:.2f}s lap={res.lap_time_s:.2f}s bonus=-{res.total_bonus_s:.0f}s",
                )

            scenarios.append(
                ScenarioSpec(
                    name=f"stress_{track}_seed{sd}",
                    category="stress",
                    description=f"Full-lap randomized finals props on {track} (seed={sd})",
                    config=cfg,
                    stop_condition=None,
                    evaluator=_eval_stress,
                )
            )
    return scenarios


def run_suite(
    scenarios: Sequence[ScenarioSpec],
    record_failures_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> List[ScenarioReport]:
    reports: List[ScenarioReport] = []
    fail_dir = Path(record_failures_dir) if record_failures_dir else None
    if fail_dir:
        fail_dir.mkdir(parents=True, exist_ok=True)

    for idx, sc in enumerate(scenarios, start=1):
        if verbose:
            print(f"[{idx}/{len(scenarios)}] Running {sc.name} ({sc.config.track})...", flush=True)
        res = run_episode(sc.config, stop_condition=sc.stop_condition)
        passed, reason = sc.evaluator(res)

        fail_video: Optional[str] = None
        if not passed and fail_dir is not None:
            vid_path = fail_dir / f"fail_{sc.name}.mp4"
            if verbose:
                print(f"    -> FAILED ({reason}); recording failure MP4 to {vid_path}...", flush=True)
            run_episode(sc.config, record_path=vid_path, stop_condition=sc.stop_condition)
            fail_video = str(vid_path)

        rep = ScenarioReport(
            name=sc.name,
            category=sc.category,
            track=res.track,
            passed=passed,
            reason=reason,
            sim_time_s=res.sim_time_s,
            wall_time_s=res.wall_time_s,
            net_score_s=res.net_score_s,
            total_penalty_s=res.total_penalty_s,
            total_bonus_s=res.total_bonus_s,
            min_v_mps=res.min_v_mps,
            max_s_m=res.max_s_m,
            w_cmd_std=res.w_cmd_std,
            w_sign_flips=res.w_sign_flips,
            dnf=res.dnf,
            finished=res.finished,
            failure_video=fail_video,
            details={
                "done_reason": res.done_reason,
                "boundary_press_time_s": res.boundary_press_time_s,
                "boundary_penalty_s": res.boundary_penalty_s,
                "cone_hits": res.cone_hits,
                "no_entry_hits": res.no_entry_hits,
                "yellow_violations": res.yellow_violations,
                "yellow_stops_ok": res.yellow_stops_ok,
                "bonus_digits_collected": res.bonus_digits_collected,
                "active_digits_locked": res.active_digits_locked,
                "max_heading_diff_deg": res.max_heading_diff_deg,
                "max_s_backtrack_m": res.max_s_backtrack_m,
            },
        )
        reports.append(rep)

    return reports


def format_report_table(reports: Sequence[ScenarioReport]) -> str:
    header = (
        f"{'SCENARIO':<32} {'TRACK':<11} {'STATUS':<6} {'SIM_T':>6} {'NET':>7} "
        f"{'PEN':>6} {'BON':>5} {'MIN_V':>7} {'MAX_S':>6}  NOTES"
    )
    sep = "-" * 124
    lines = ["=" * 124, "AIC PC SIMULATOR — REGRESSION & STRESS SUITE SUMMARY", sep, header, sep]
    n_pass = 0
    for r in reports:
        if r.passed:
            n_pass += 1
        st = "PASS" if r.passed else "FAIL"
        lines.append(
            f"{r.name:<32} {r.track:<11} {st:<6} {r.sim_time_s:5.2f}s {r.net_score_s:6.2f}s "
            f"{r.total_penalty_s:5.1f}s {r.total_bonus_s:4.0f}s {r.min_v_mps:+6.3f} {r.max_s_m:5.2f}m  {r.reason}"
        )
        if r.failure_video:
            lines.append(f"    -> failure video: {r.failure_video}")
    lines.append(sep)
    reg = [r for r in reports if r.category == "regression"]
    stress = [r for r in reports if r.category == "stress"]
    reg_pass = sum(1 for r in reg if r.passed)
    stress_pass = sum(1 for r in stress if r.passed)
    total_wall = sum(r.wall_time_s for r in reports)
    lines.append(
        f"TOTAL: {n_pass}/{len(reports)} PASSED  "
        f"(Regression: {reg_pass}/{len(reg)}, Stress: {stress_pass}/{len(stress)})  "
        f"| Wall time: {total_wall:.1f}s"
    )
    lines.append("=" * 124)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="4-Scenario Real-Car Crash Regression Suite + Multi-Seed Stress Tests"
    )
    ap.add_argument(
        "--seeds",
        type=int,
        default=0,
        help="Number of randomized stress seeds per map (fork_s & roundabout). Default 0 (4 regression scenarios only).",
    )
    ap.add_argument("--detector", choices=("mock", "yolo"), default="mock", help="Detector backend")
    ap.add_argument("--fps", type=float, default=30.0, help="Simulation frame rate (Hz)")
    ap.add_argument("--seed", type=int, default=42, help="Base seed for deterministic regression scenarios")
    ap.add_argument("--override", action="append", default=[], help="Parameter override key=val (repeatable)")
    ap.add_argument("--params-json", default=None, help="JSON file or inline JSON string of param overrides")
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated substring filter for scenario names (e.g. 'yellow_stop,fork_y')",
    )
    ap.add_argument(
        "--record-failures",
        default=None,
        help="Directory to save MP4 recordings for any failed scenarios",
    )
    ap.add_argument("--json-out", default=None, help="Optional JSON report output path")
    ap.add_argument(
        "--no-fail-exit",
        action="store_true",
        help="Exit 0 even if some scenarios fail (useful for reporting/diagnostics)",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    overrides = load_params_json(args.params_json)
    overrides.update(parse_overrides_list(args.override))

    scenarios = build_regression_scenarios(
        detector=args.detector,
        rate_hz=args.fps,
        param_overrides=overrides,
        seed=args.seed,
    )
    if args.seeds > 0:
        scenarios.extend(
            build_stress_scenarios(
                seeds=args.seeds,
                detector=args.detector,
                rate_hz=args.fps,
                param_overrides=overrides,
                base_seed=100,
            )
        )

    if args.only:
        filters = [s.strip().lower() for s in args.only.split(",") if s.strip()]
        scenarios = [s for s in scenarios if any(f in s.name.lower() for f in filters)]

    if not scenarios:
        print("[regression_suite] No scenarios matched filter.", flush=True)
        return 1

    t0 = time.perf_counter()
    reports = run_suite(scenarios, record_failures_dir=args.record_failures, verbose=True)
    print("\n" + format_report_table(reports), flush=True)

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "wall_time_s": round(time.perf_counter() - t0, 3),
            "passed": sum(1 for r in reports if r.passed),
            "total": len(reports),
            "reports": [r.__dict__ for r in reports],
        }
        out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    all_pass = all(r.passed for r in reports)
    if args.no_fail_exit:
        return 0
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
