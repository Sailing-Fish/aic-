#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-Car Blackbox Calibrator & Offline Replay Consistency Validator.

Parses real-car fault-replay blackbox bundles (extracted directories or .zip archives)
containing:
  - meta.json
  - events_slice.csv
  - follow_slice.log
  - frame_timestamps.csv
  - params_snapshot.yaml
  - (optional) *_debug.mp4

Responsibilities:
  1. Estimate real-car physical & pipeline parameters:
     - Measured brake deceleration `a_eff_mps2` (both EMA `brake_adapt` & raw kinematic)
     - Battery voltage range (`vbat`)
     - Camera FPS / read+publish latency & YOLO BPU e2e latency stats
     - Steering & speed response / start-up lag stats
     - Comparison table vs current `tools/sim/sim_world_pc.py` defaults
     - Export clean calibrated profile to `tools/sim/real_car_dynamics.json`
  2. Offline real-car blackbox replay consistency validator (`run_blackbox_replay_check`)
     feeding reconstructed/extracted real-car critical telemetry & vision frames through
     the latest `BrainBridge` (`sim_brain.dll`) + `VisionFrontend` mirror to verify:
     (a) `roundabout_212118` off-track window (#133..143 near-right-edge condition):
         brain no longer commands right-turn off-track yaw;
     (b) `fork_215951` yellow-line stop transitions:
         brain commands zero-reverse / clean stop without persistent reverse overshoot;
     (c) `fork_215951` dual-digit #86..93 transition:
         no late right-swerve spike when higher digit leaves FOV;
     (d) `fork_215951` fork exit wide junction #153:
         preserves right-exit continuity rather than flipping left;
     (e) Nominal curve frames (#69/#125):
         bounded, finite, non-divergent `v_cmd` / `w_cmd`.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import io
import json
import math
import os
import re
import statistics
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Guard against tools/sim/profile.py shadowing Python stdlib `profile`
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[1]
_this_dir_norm = os.path.normcase(str(_THIS_DIR))
sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or ".")) != _this_dir_norm]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from tools.sim.brain_bridge import BrainBridge, load_line_follower_params
from tools.sim.render import CameraModel
from tools.sim.sim_world_pc import VehicleDynamics
from tools.sim.vision_mirror import (
    SegmentedFrame,
    VisionFrontend,
    unwrap_ros_params,
)

# Default extracted real-car blackbox directories on this workstation
DEFAULT_BLACKBOX_DIRS: Tuple[Path, ...] = (
    Path(
        r"C:\Users\Lenovo\.gemini\antigravity\brain\2074e89a-af1d-4634-a0f0-907324d9e90a\scratch\roundabout_212118"
    ),
    Path(
        r"C:\Users\Lenovo\.gemini\antigravity\brain\2074e89a-af1d-4634-a0f0-907324d9e90a\scratch\fork_215951"
    ),
)

DEFAULT_OUTPUT_JSON = _THIS_DIR / "real_car_dynamics.json"
DEFAULT_FORK_YAML = REPO_ROOT / "tools" / "config" / "finals_fork.yaml"
DEFAULT_ROUNDABOUT_YAML = REPO_ROOT / "tools" / "config" / "finals_roundabout.yaml"


# ---------------------------------------------------------------------------
# Low-level numeric & parsing helpers
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    k = (len(arr) - 1) * (max(0.0, min(100.0, pct)) / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    return arr[f] * (c - k) + arr[c] * (k - f)


def _summary_stats(values: Sequence[float], ndigits: int = 4) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(vals),
        "min": round(min(vals), ndigits),
        "max": round(max(vals), ndigits),
        "mean": round(statistics.mean(vals), ndigits),
        "p50": round(float(_percentile(vals, 50.0)), ndigits),
        "p95": round(float(_percentile(vals, 95.0)), ndigits),
    }


def _parse_float_token(raw: str) -> Optional[float]:
    s = raw.strip()
    if s.endswith("ms"):
        s = s[:-2]
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


_KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def parse_kv_string(kv_str: str) -> Dict[str, str]:
    """Parse space-separated key=value pairs from events_slice.csv or [DEC] log lines."""
    out: Dict[str, str] = {}
    if not kv_str:
        return out
    for m in _KV_RE.finditer(kv_str):
        out[m.group(1)] = m.group(2)
    return out


def parse_ctrl_fields(kv: Dict[str, str]) -> Dict[str, Any]:
    """Parse typed fields from a `[DEC] ctrl ...` / `follow,ctrl,state` record."""
    rec: Dict[str, Any] = {
        "mode": kv.get("mode", ""),
        "branch": int(kv["branch"]) if "branch" in kv and kv["branch"].lstrip("+-").isdigit() else 0,
        "y_ref": _parse_float_token(kv.get("y_ref", "")),
        "hdg": _parse_float_token(kv.get("hdg", "")),
        "rev": int(kv["rev"]) if "rev" in kv and kv["rev"].isdigit() else 0,
        "v": _parse_float_token(kv.get("v", "")),
        "yaw": _parse_float_token(kv.get("yaw", "")),
        "s": _parse_float_token(kv.get("s", "")),
        "lane_right": None,
        "lane_left": None,
        "obs_near": 0,
        "obs_far": 0,
    }
    lane_raw = kv.get("lane", "")
    m_lane = re.match(r"\[([+\-]?\d+(?:\.\d+)?),([+\-]?\d+(?:\.\d+)?)\]", lane_raw)
    if m_lane:
        rec["lane_right"] = float(m_lane.group(1))
        rec["lane_left"] = float(m_lane.group(2))
    obs_raw = kv.get("obs", "")
    if "/" in obs_raw:
        parts = obs_raw.split("/", 1)
        if parts[0].isdigit():
            rec["obs_near"] = int(parts[0])
        if parts[1].isdigit():
            rec["obs_far"] = int(parts[1])
    return rec


def parse_brake_adapt_fields(kv: Dict[str, str]) -> Dict[str, Any]:
    """Parse `[DEC] brake_adapt v_in=... travel=... scale=a->b a_eff=... vbat=...`."""
    v_in = _parse_float_token(kv.get("v_in", ""))
    travel = _parse_float_token(kv.get("travel", ""))
    a_eff = _parse_float_token(kv.get("a_eff", ""))
    vbat = _parse_float_token(kv.get("vbat", ""))
    scale_before, scale_after = None, None
    sc = kv.get("scale", "")
    if "->" in sc:
        p0, p1 = sc.split("->", 1)
        scale_before = _parse_float_token(p0)
        scale_after = _parse_float_token(p1)
    a_kin_raw = None
    a_kin_lat_adj = None
    if v_in is not None and travel is not None and travel > 1e-4:
        a_kin_raw = (v_in * v_in) / (2.0 * travel)
        adj_travel = travel - v_in * 0.10
        if adj_travel > 0.005:
            a_kin_lat_adj = (v_in * v_in) / (2.0 * adj_travel)
    return {
        "v_in_mps": v_in,
        "travel_m": travel,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "a_eff_mps2": a_eff,
        "a_kin_raw_mps2": round(a_kin_raw, 4) if a_kin_raw is not None else None,
        "a_kin_lat_adj_mps2": round(a_kin_lat_adj, 4) if a_kin_lat_adj is not None else None,
        "vbat_v": vbat,
    }


# ---------------------------------------------------------------------------
# Blackbox bundle loader (supports both extracted directories & .zip files)
# ---------------------------------------------------------------------------


@dataclass
class BlackboxBundle:
    name: str
    source_path: str
    meta: Dict[str, Any] = field(default_factory=dict)
    params_snapshot: Dict[str, Any] = field(default_factory=dict)
    frame_timestamps: List[Dict[str, float]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    follow_log_records: List[Dict[str, Any]] = field(default_factory=list)
    brake_adapt_events: List[Dict[str, Any]] = field(default_factory=list)
    yellow_events: List[Dict[str, Any]] = field(default_factory=list)
    ctrl_records: List[Dict[str, Any]] = field(default_factory=list)
    camera_perf: List[Dict[str, float]] = field(default_factory=list)
    yolo_perf: List[Dict[str, float]] = field(default_factory=list)
    red_src_events: List[Dict[str, Any]] = field(default_factory=list)
    start_signal_epochs: List[Tuple[float, str]] = field(default_factory=list)


def _read_bundle_files(path: Union[str, Path]) -> Tuple[str, Dict[str, str]]:
    """Read text files of interest from either a directory or a .zip archive."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Blackbox path not found: {p}")

    target_names = {
        "meta.json",
        "events_slice.csv",
        "follow_slice.log",
        "frame_timestamps.csv",
        "params_snapshot.yaml",
    }
    texts: Dict[str, str] = {}

    if p.is_file() and p.suffix.lower() == ".zip":
        name = p.stem
        with zipfile.ZipFile(p, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                base = Path(member.filename).name
                if base in target_names and base not in texts:
                    texts[base] = zf.read(member).decode("utf-8", errors="replace")
        return name, texts

    if p.is_dir():
        name = p.name
        for fname in target_names:
            direct = p / fname
            if direct.is_file():
                texts[fname] = direct.read_text(encoding="utf-8", errors="replace")
            else:
                matches = list(p.rglob(fname))
                if matches:
                    texts[fname] = matches[0].read_text(encoding="utf-8", errors="replace")
        return name, texts

    raise ValueError(f"Unsupported blackbox input (expected directory or .zip): {p}")


def parse_frame_timestamps_csv(text: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not text.strip():
        return rows
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        try:
            rows.append(
                {
                    "frame_idx": int(float(r["frame_idx"])),
                    "t_rel_s": float(r["t_rel_s"]),
                    "wall_epoch": float(r["wall_epoch"]),
                }
            )
        except (KeyError, ValueError, TypeError):
            continue
    return rows


def parse_events_slice_csv(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not text.strip():
        return rows
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        kv_raw = r.get("kv", "") or ""
        t_rel = _parse_float_token(r.get("t_rel", "") or "")
        rows.append(
            {
                "ts_iso": r.get("ts_iso", ""),
                "t_rel": t_rel,
                "source": r.get("source", ""),
                "area": r.get("area", ""),
                "event": r.get("event", ""),
                "kv_raw": kv_raw,
                "kv": parse_kv_string(kv_raw),
            }
        )
    return rows


_ROS_LOG_RE = re.compile(
    r"^\[[^\]]+\]\s+\[([A-Z]+)\]\s+\[(\d+\.\d+)\]\s+\[([^\]]+)\]:\s*(.*)$"
)


def parse_follow_slice_log(text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for ln in text.splitlines():
        m = _ROS_LOG_RE.match(ln.strip())
        if not m:
            continue
        level, epoch_s, node, msg = m.group(1), float(m.group(2)), m.group(3), m.group(4)
        rec: Dict[str, Any] = {
            "level": level,
            "epoch": epoch_s,
            "node": node,
            "msg": msg,
            "dec_area": None,
            "dec_event": None,
            "kv": {},
        }
        if msg.startswith("[DEC] "):
            body = msg[6:].strip()
            parts = body.split(None, 2)
            if len(parts) >= 2:
                rec["dec_area"] = parts[0]
                if "=" in parts[1]:
                    rec["dec_event"] = "state"
                    rec["kv"] = parse_kv_string(body[len(parts[0]):].strip())
                else:
                    rec["dec_event"] = parts[1]
                    rec["kv"] = parse_kv_string(parts[2] if len(parts) > 2 else "")
        records.append(rec)
    return records


def load_blackbox(path: Union[str, Path]) -> BlackboxBundle:
    """Load and parse a real-car blackbox directory or .zip bundle."""
    name, texts = _read_bundle_files(path)
    bundle = BlackboxBundle(name=name, source_path=str(path))

    if "meta.json" in texts:
        try:
            bundle.meta = json.loads(texts["meta.json"])
        except json.JSONDecodeError:
            bundle.meta = {}

    if "params_snapshot.yaml" in texts:
        try:
            raw_yaml = yaml.safe_load(texts["params_snapshot.yaml"]) or {}
            bundle.params_snapshot = unwrap_ros_params(raw_yaml)
        except Exception:
            bundle.params_snapshot = {}

    if "frame_timestamps.csv" in texts:
        bundle.frame_timestamps = parse_frame_timestamps_csv(texts["frame_timestamps.csv"])

    if "events_slice.csv" in texts:
        bundle.events = parse_events_slice_csv(texts["events_slice.csv"])
        for ev in bundle.events:
            area = ev["area"]
            event = ev["event"]
            kv = ev["kv"]
            if area == "camera" and event == "perf":
                bundle.camera_perf.append(
                    {
                        k: v
                        for k, v in (
                            ("t_rel", ev["t_rel"]),
                            ("fps", _parse_float_token(kv.get("fps", ""))),
                            ("read_p50_ms", _parse_float_token(kv.get("read_p50", ""))),
                            ("read_p95_ms", _parse_float_token(kv.get("read_p95", ""))),
                            ("pub_p50_ms", _parse_float_token(kv.get("pub_p50", ""))),
                            ("pub_p95_ms", _parse_float_token(kv.get("pub_p95", ""))),
                            ("fails", _parse_float_token(kv.get("fails", ""))),
                        )
                        if v is not None
                    }
                )
            elif area == "yolo" and event == "perf":
                bundle.yolo_perf.append(
                    {
                        k: v
                        for k, v in (
                            ("t_rel", ev["t_rel"]),
                            ("proc_fps", _parse_float_token(kv.get("proc_fps", ""))),
                            ("rx_fps", _parse_float_token(kv.get("rx_fps", ""))),
                            ("dropped", _parse_float_token(kv.get("dropped", ""))),
                            ("infer_p50_ms", _parse_float_token(kv.get("infer_p50", ""))),
                            ("infer_p95_ms", _parse_float_token(kv.get("infer_p95", ""))),
                            ("total_p50_ms", _parse_float_token(kv.get("total_p50", ""))),
                            ("total_p95_ms", _parse_float_token(kv.get("total_p95", ""))),
                            ("e2e_p50_ms", _parse_float_token(kv.get("e2e_p50", ""))),
                            ("e2e_p95_ms", _parse_float_token(kv.get("e2e_p95", ""))),
                        )
                        if v is not None
                    }
                )

    if "follow_slice.log" in texts:
        bundle.follow_log_records = parse_follow_slice_log(texts["follow_slice.log"])
        for rec in bundle.follow_log_records:
            msg = rec["msg"]
            if "Start signal changed:" in msg:
                bundle.start_signal_epochs.append((rec["epoch"], msg))
            area = rec["dec_area"]
            event = rec["dec_event"]
            kv = rec["kv"]
            if area == "brake_adapt":
                parsed = parse_brake_adapt_fields(kv)
                parsed["epoch"] = rec["epoch"]
                bundle.brake_adapt_events.append(parsed)
            elif area == "yellow":
                item = {"epoch": rec["epoch"], "event": event, **kv}
                bundle.yellow_events.append(item)
            elif area == "ctrl":
                parsed_c = parse_ctrl_fields(kv)
                parsed_c["epoch"] = rec["epoch"]
                bundle.ctrl_records.append(parsed_c)
            elif area == "red_src":
                bundle.red_src_events.append({"epoch": rec["epoch"], "event": event, **kv})

    # Fallback: if follow_slice.log was absent, pull brake_adapt & ctrl from events_slice.csv
    if not bundle.brake_adapt_events and bundle.events:
        for ev in bundle.events:
            if ev["area"] == "brake_adapt":
                parsed = parse_brake_adapt_fields(ev["kv"])
                parsed["epoch"] = ev["t_rel"]
                bundle.brake_adapt_events.append(parsed)
    if not bundle.ctrl_records and bundle.events:
        for ev in bundle.events:
            if ev["area"] == "ctrl":
                parsed_c = parse_ctrl_fields(ev["kv"])
                parsed_c["epoch"] = ev["t_rel"]
                bundle.ctrl_records.append(parsed_c)

    return bundle


# ---------------------------------------------------------------------------
# Real-Car Physical Parameter Calibration
# ---------------------------------------------------------------------------


def _get_sim_world_defaults() -> Dict[str, Any]:
    """Introspect current VehicleDynamics defaults from tools/sim/sim_world_pc.py."""
    sig = inspect.signature(VehicleDynamics.__init__)
    out: Dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.default is not inspect.Parameter.empty:
            out[pname] = param.default
    return out


def estimate_bundle_metrics(bundle: BlackboxBundle) -> Dict[str, Any]:
    """Compute per-run physical & telemetry statistics for one blackbox bundle."""
    # Frame recording dt stats
    frame_dts: List[float] = []
    if len(bundle.frame_timestamps) >= 2:
        for i in range(1, len(bundle.frame_timestamps)):
            dt = bundle.frame_timestamps[i]["t_rel_s"] - bundle.frame_timestamps[i - 1]["t_rel_s"]
            if 0.001 < dt < 2.0:
                frame_dts.append(dt)
    rec_fps = (1.0 / statistics.median(frame_dts)) if frame_dts else None

    # Brake decel & voltage
    a_eff_vals = [b["a_eff_mps2"] for b in bundle.brake_adapt_events if b.get("a_eff_mps2") is not None]
    a_kin_raw = [b["a_kin_raw_mps2"] for b in bundle.brake_adapt_events if b.get("a_kin_raw_mps2") is not None]
    a_kin_adj = [
        b["a_kin_lat_adj_mps2"] for b in bundle.brake_adapt_events if b.get("a_kin_lat_adj_mps2") is not None
    ]
    vbat_vals = [b["vbat_v"] for b in bundle.brake_adapt_events if b.get("vbat_v") is not None]

    # Yellow brake duration / dv_dt stats
    brake_end_durs: List[float] = []
    brake_end_vs: List[float] = []
    for yev in bundle.yellow_events:
        if yev.get("event") == "brake_end":
            d_val = _parse_float_token(str(yev.get("dur", "")))
            v_val = _parse_float_token(str(yev.get("v", "")))
            if d_val is not None:
                brake_end_durs.append(d_val)
            if v_val is not None:
                brake_end_vs.append(v_val)

    # Speed & steering response stats from ctrl_records
    v_cmds = [c["v"] for c in bundle.ctrl_records if c.get("v") is not None]
    yaw_cmds = [c["yaw"] for c in bundle.ctrl_records if c.get("yaw") is not None]
    abs_yaws = [abs(w) for w in yaw_cmds]

    # Observed ds/dt actual speed & dyaw/dt from consecutive ctrl records
    actual_speeds: List[float] = []
    speed_ratios: List[float] = []
    dyaw_dts: List[float] = []
    for i in range(1, len(bundle.ctrl_records)):
        prev = bundle.ctrl_records[i - 1]
        cur = bundle.ctrl_records[i]
        t0, t1 = prev.get("epoch"), cur.get("epoch")
        s0, s1 = prev.get("s"), cur.get("s")
        if t0 is not None and t1 is not None and 0.05 <= (t1 - t0) <= 0.60:
            dt = t1 - t0
            if s0 is not None and s1 is not None:
                ds = s1 - s0
                if ds > 0.005:
                    v_act = ds / dt
                    if 0.02 <= v_act <= 1.2:
                        actual_speeds.append(v_act)
                        v_cmd = cur.get("v")
                        if v_cmd is not None and v_cmd >= 0.15:
                            speed_ratios.append(v_act / v_cmd)
            y0, y1 = prev.get("yaw"), cur.get("yaw")
            if y0 is not None and y1 is not None:
                dyaw_dts.append(abs(y1 - y0) / dt)

    # Start-up deadtime / lag: time from `Start signal changed: false -> true` to first s > 0.02
    startup_lag_s: Optional[float] = None
    start_t = None
    for ep, msg in bundle.start_signal_epochs:
        if "false -> true" in msg:
            start_t = ep
            break
    if start_t is not None:
        for c in bundle.ctrl_records:
            if c.get("epoch") is not None and c["epoch"] >= start_t and (c.get("s") or 0.0) >= 0.02:
                startup_lag_s = round(c["epoch"] - start_t, 3)
                break

    max_s_m = max((c["s"] for c in bundle.ctrl_records if c.get("s") is not None), default=0.0)

    return {
        "name": bundle.name,
        "source_path": bundle.source_path,
        "track_mode": bundle.params_snapshot.get("track_mode", "unknown"),
        "duration_sec": bundle.meta.get("duration_sec"),
        "video_frames": bundle.meta.get("frames", len(bundle.frame_timestamps)),
        "debug_record_fps_p50": round(rec_fps, 2) if rec_fps is not None else None,
        "max_arc_s_m": round(max_s_m, 3),
        "camera_fps": _summary_stats([c["fps"] for c in bundle.camera_perf if "fps" in c], 3),
        "camera_read_p50_ms": _summary_stats(
            [c["read_p50_ms"] for c in bundle.camera_perf if "read_p50_ms" in c], 2
        ),
        "camera_read_p95_ms": _summary_stats(
            [c["read_p95_ms"] for c in bundle.camera_perf if "read_p95_ms" in c], 2
        ),
        "camera_pub_p50_ms": _summary_stats(
            [c["pub_p50_ms"] for c in bundle.camera_perf if "pub_p50_ms" in c], 2
        ),
        "yolo_proc_fps": _summary_stats(
            [y["proc_fps"] for y in bundle.yolo_perf if "proc_fps" in y], 3
        ),
        "yolo_e2e_p50_ms": _summary_stats(
            [y["e2e_p50_ms"] for y in bundle.yolo_perf if "e2e_p50_ms" in y], 2
        ),
        "yolo_e2e_p95_ms": _summary_stats(
            [y["e2e_p95_ms"] for y in bundle.yolo_perf if "e2e_p95_ms" in y], 2
        ),
        "brake_adapt_events": bundle.brake_adapt_events,
        "brake_a_eff_mps2": _summary_stats(a_eff_vals, 4),
        "brake_a_kin_raw_mps2": _summary_stats(a_kin_raw, 4),
        "brake_a_kin_lat_adj_mps2": _summary_stats(a_kin_adj, 4),
        "brake_window_dur_s": _summary_stats(brake_end_durs, 3),
        "brake_release_v_mps": _summary_stats(brake_end_vs, 4),
        "battery_voltage_v": _summary_stats(vbat_vals, 3),
        "v_cmd_mps": _summary_stats(v_cmds, 4),
        "v_actual_ds_dt_mps": _summary_stats(actual_speeds, 4),
        "speed_actual_to_cmd_ratio": _summary_stats(speed_ratios, 4),
        "abs_yaw_cmd_rps": _summary_stats(abs_yaws, 4),
        "abs_dyaw_dt_rps2": _summary_stats(dyaw_dts, 4),
        "startup_response_lag_s": startup_lag_s,
    }


def calibrate_from_blackboxes(
    paths: Sequence[Union[str, Path]] = DEFAULT_BLACKBOX_DIRS,
    out_json: Optional[Union[str, Path]] = DEFAULT_OUTPUT_JSON,
) -> Dict[str, Any]:
    """Parse one or more real-car blackbox bundles and write calibrated dynamics JSON."""
    bundles = [load_blackbox(p) for p in paths]
    per_run = [estimate_bundle_metrics(b) for b in bundles]

    all_a_eff: List[float] = []
    all_a_kin_raw: List[float] = []
    all_a_kin_adj: List[float] = []
    all_vbat: List[float] = []
    all_cam_fps: List[float] = []
    all_cam_read_p50: List[float] = []
    all_cam_read_p95: List[float] = []
    all_cam_pub_p50: List[float] = []
    all_yolo_fps: List[float] = []
    all_yolo_e2e_p50: List[float] = []
    all_yolo_e2e_p95: List[float] = []
    all_v_cmd: List[float] = []
    all_v_act: List[float] = []
    all_speed_ratio: List[float] = []
    all_abs_yaw: List[float] = []
    all_dyaw_dt: List[float] = []
    all_startup_lag: List[float] = []
    all_brake_dur: List[float] = []

    for b, m in zip(bundles, per_run):
        all_a_eff.extend(
            ev["a_eff_mps2"] for ev in b.brake_adapt_events if ev.get("a_eff_mps2") is not None
        )
        all_a_kin_raw.extend(
            ev["a_kin_raw_mps2"] for ev in b.brake_adapt_events if ev.get("a_kin_raw_mps2") is not None
        )
        all_a_kin_adj.extend(
            ev["a_kin_lat_adj_mps2"]
            for ev in b.brake_adapt_events
            if ev.get("a_kin_lat_adj_mps2") is not None
        )
        all_vbat.extend(
            ev["vbat_v"] for ev in b.brake_adapt_events if ev.get("vbat_v") is not None
        )
        all_cam_fps.extend(c["fps"] for c in b.camera_perf if "fps" in c)
        all_cam_read_p50.extend(c["read_p50_ms"] for c in b.camera_perf if "read_p50_ms" in c)
        all_cam_read_p95.extend(c["read_p95_ms"] for c in b.camera_perf if "read_p95_ms" in c)
        all_cam_pub_p50.extend(c["pub_p50_ms"] for c in b.camera_perf if "pub_p50_ms" in c)
        all_yolo_fps.extend(y["proc_fps"] for y in b.yolo_perf if "proc_fps" in y)
        all_yolo_e2e_p50.extend(y["e2e_p50_ms"] for y in b.yolo_perf if "e2e_p50_ms" in y)
        all_yolo_e2e_p95.extend(y["e2e_p95_ms"] for y in b.yolo_perf if "e2e_p95_ms" in y)
        all_v_cmd.extend(c["v"] for c in b.ctrl_records if c.get("v") is not None)
        all_abs_yaw.extend(abs(c["yaw"]) for c in b.ctrl_records if c.get("yaw") is not None)
        if m["startup_response_lag_s"] is not None:
            all_startup_lag.append(float(m["startup_response_lag_s"]))
        for yev in b.yellow_events:
            if yev.get("event") == "brake_end":
                d_val = _parse_float_token(str(yev.get("dur", "")))
                if d_val is not None:
                    all_brake_dur.append(d_val)

        # Re-aggregate per-step speed & dyaw/dt
        for i in range(1, len(b.ctrl_records)):
            prev, cur = b.ctrl_records[i - 1], b.ctrl_records[i]
            t0, t1 = prev.get("epoch"), cur.get("epoch")
            s0, s1 = prev.get("s"), cur.get("s")
            if t0 is not None and t1 is not None and 0.05 <= (t1 - t0) <= 0.60:
                dt = t1 - t0
                if s0 is not None and s1 is not None and (s1 - s0) > 0.005:
                    va = (s1 - s0) / dt
                    if 0.02 <= va <= 1.2:
                        all_v_act.append(va)
                        if cur.get("v") is not None and cur["v"] >= 0.15:
                            all_speed_ratio.append(va / cur["v"])
                if prev.get("yaw") is not None and cur.get("yaw") is not None:
                    all_dyaw_dt.append(abs(cur["yaw"] - prev["yaw"]) / dt)

    a_eff_stats = _summary_stats(all_a_eff, 4)
    a_kin_raw_stats = _summary_stats(all_a_kin_raw, 4)
    a_kin_adj_stats = _summary_stats(all_a_kin_adj, 4)
    vbat_stats = _summary_stats(all_vbat, 3)
    cam_fps_stats = _summary_stats(all_cam_fps, 3)
    cam_read_p50_stats = _summary_stats(all_cam_read_p50, 2)
    cam_read_p95_stats = _summary_stats(all_cam_read_p95, 2)
    cam_pub_p50_stats = _summary_stats(all_cam_pub_p50, 2)
    yolo_fps_stats = _summary_stats(all_yolo_fps, 3)
    yolo_e2e_p50_stats = _summary_stats(all_yolo_e2e_p50, 2)
    yolo_e2e_p95_stats = _summary_stats(all_yolo_e2e_p95, 2)
    v_cmd_stats = _summary_stats(all_v_cmd, 4)
    v_act_stats = _summary_stats(all_v_act, 4)
    speed_ratio_stats = _summary_stats(all_speed_ratio, 4)
    abs_yaw_stats = _summary_stats(all_abs_yaw, 4)
    dyaw_dt_stats = _summary_stats(all_dyaw_dt, 4)
    brake_dur_stats = _summary_stats(all_brake_dur, 3)
    startup_lag_stats = _summary_stats(all_startup_lag, 3)

    # Calibrated nominal values
    cal_a_eff = a_eff_stats["p50"] if a_eff_stats["p50"] is not None else 1.44
    cal_a_eff_min = a_eff_stats["min"] if a_eff_stats["min"] is not None else 1.34
    cal_vbat_nom = vbat_stats["mean"] if vbat_stats["mean"] is not None else 12.0
    cal_cam_fps = cam_fps_stats["mean"] if cam_fps_stats["mean"] is not None else 15.0

    # Sensor-to-control pipeline latency estimate:
    # ~0.5 frame exposure/readout (read_p50/2) + ROS pub + YOLO e2e (or full read_p50 + pub)
    read_p50 = cam_read_p50_stats["mean"] or 61.4
    pub_p50 = cam_pub_p50_stats["mean"] or 3.6
    yolo_p50 = yolo_e2e_p50_stats["mean"] or 30.7
    est_vision_latency_ms = round(0.5 * read_p50 + pub_p50, 2)
    est_yolo_total_latency_ms = round(0.5 * read_p50 + pub_p50 + yolo_p50, 2)
    est_worst_p95_latency_ms = round(
        (cam_read_p95_stats["mean"] or 62.5) + (yolo_e2e_p95_stats["mean"] or 35.4), 2
    )

    sim_defaults = _get_sim_world_defaults()
    sim_a_eff = float(sim_defaults.get("a_eff", 1.44))
    sim_voltage = float(sim_defaults.get("voltage", 12.0))
    sim_tau_speed = float(sim_defaults.get("tau_speed", 0.61))
    sim_tau_steer = float(sim_defaults.get("tau_steer", 0.26))
    sim_latency_ms = float(sim_defaults.get("latency_ms", 0.0))

    calibrated_profile = {
        "a_eff_mps2": round(float(cal_a_eff), 4),
        "a_eff_mean_mps2": a_eff_stats["mean"],
        "a_eff_min_mps2": round(float(cal_a_eff_min), 4),
        "a_eff_slippery_078x_mps2": round(float(cal_a_eff) * 0.78, 4),
        "a_kin_raw_mean_mps2": a_kin_raw_stats["mean"],
        "a_kin_lat_adj_mean_mps2": a_kin_adj_stats["mean"],
        "battery_voltage_nominal_v": round(float(cal_vbat_nom), 3),
        "battery_voltage_range_v": [vbat_stats["min"], vbat_stats["max"]],
        "camera_fps_hz": round(float(cal_cam_fps), 2),
        "camera_read_p50_ms": cam_read_p50_stats["mean"],
        "camera_read_p95_ms": cam_read_p95_stats["mean"],
        "camera_pub_p50_ms": cam_pub_p50_stats["mean"],
        "yolo_proc_fps_hz": yolo_fps_stats["mean"],
        "yolo_e2e_p50_ms": yolo_e2e_p50_stats["mean"],
        "yolo_e2e_p95_ms": yolo_e2e_p95_stats["mean"],
        "est_vision_pipeline_latency_ms": est_vision_latency_ms,
        "est_yolo_pipeline_latency_ms": est_yolo_total_latency_ms,
        "est_worst_p95_latency_ms": est_worst_p95_latency_ms,
        "recommended_holdout_extra_latency_ms": 35.0,
        "startup_response_lag_s": startup_lag_stats["mean"],
        "speed_actual_to_cmd_ratio_p50": speed_ratio_stats["p50"],
        "max_observed_abs_yaw_rps": abs_yaw_stats["max"],
        "p95_observed_abs_yaw_rps": abs_yaw_stats["p95"],
        "p95_observed_dyaw_dt_rps2": dyaw_dt_stats["p95"],
        "brake_window_dur_mean_s": brake_dur_stats["mean"],
    }

    comparison_vs_sim_defaults = {
        "a_eff_mps2": {
            "calibrated": calibrated_profile["a_eff_mps2"],
            "calibrated_range": [a_eff_stats["min"], a_eff_stats["max"]],
            "sim_world_pc_default": sim_a_eff,
            "delta": round(calibrated_profile["a_eff_mps2"] - sim_a_eff, 4),
            "note": "Median brake_adapt a_eff matches sim_world_pc.py default (1.44 m/s^2); observed floor is 1.343 m/s^2.",
        },
        "voltage_v": {
            "calibrated": calibrated_profile["battery_voltage_nominal_v"],
            "calibrated_range": calibrated_profile["battery_voltage_range_v"],
            "sim_world_pc_default": sim_voltage,
            "delta": round(calibrated_profile["battery_voltage_nominal_v"] - sim_voltage, 3),
            "note": "Real-car 3S LiPo ran at 11.97..12.13V during test runs vs 12.0V sim nominal.",
        },
        "latency_ms": {
            "calibrated_vision_ms": est_vision_latency_ms,
            "calibrated_yolo_total_ms": est_yolo_total_latency_ms,
            "sim_world_pc_default": sim_latency_ms,
            "delta_vs_vision_ms": round(est_vision_latency_ms - sim_latency_ms, 2),
            "note": "Real camera read_p50~61.4ms + pub~3.6ms + YOLO e2e~30.7ms; blind holdout injects +35ms extra command latency.",
        },
        "camera_fps_hz": {
            "calibrated": calibrated_profile["camera_fps_hz"],
            "sim_world_pc_default_rate_hz": 30.0,
            "note": "Real V4L2 camera runs ~15 Hz (61.4ms read); PC sim runs 30 Hz (2x control rate).",
        },
        "tau_speed_s": {
            "observed_startup_lag_s": calibrated_profile["startup_response_lag_s"],
            "sim_world_pc_default": sim_tau_speed,
            "note": "Observed real-car start-to-motion lag (~0.99-1.05s incl. mux handoff) is consistent with tau_speed=0.61s first-order lag.",
        },
        "tau_steer_s": {
            "observed_p95_dyaw_dt_rps2": calibrated_profile["p95_observed_dyaw_dt_rps2"],
            "sim_world_pc_default": sim_tau_steer,
            "note": "Ackermann servo steering lag retained at 0.26s with +/-12% steer_scale mismatch in blind holdout.",
        },
        "wheelbase_m": {
            "sim_world_pc_default": sim_defaults.get("wheelbase_m", 0.1395),
            "yaml_cbf_wheelbase_m": bundles[0].params_snapshot.get("cbf_wheelbase_m", 0.142)
            if bundles
            else 0.142,
        },
        "max_steer_deg": {
            "sim_world_pc_default": sim_defaults.get("max_steer_deg", 21.0),
        },
    }

    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "sources": [str(p) for p in paths],
        "calibrated": calibrated_profile,
        "aggregates": {
            "brake_a_eff_mps2": a_eff_stats,
            "brake_a_kin_raw_mps2": a_kin_raw_stats,
            "brake_a_kin_lat_adj_mps2": a_kin_adj_stats,
            "brake_window_dur_s": brake_dur_stats,
            "battery_voltage_v": vbat_stats,
            "camera_fps_hz": cam_fps_stats,
            "camera_read_p50_ms": cam_read_p50_stats,
            "camera_read_p95_ms": cam_read_p95_stats,
            "camera_pub_p50_ms": cam_pub_p50_stats,
            "yolo_proc_fps_hz": yolo_fps_stats,
            "yolo_e2e_p50_ms": yolo_e2e_p50_stats,
            "yolo_e2e_p95_ms": yolo_e2e_p95_stats,
            "v_cmd_mps": v_cmd_stats,
            "v_actual_ds_dt_mps": v_act_stats,
            "speed_actual_to_cmd_ratio": speed_ratio_stats,
            "abs_yaw_cmd_rps": abs_yaw_stats,
            "abs_dyaw_dt_rps2": dyaw_dt_stats,
            "startup_response_lag_s": startup_lag_stats,
        },
        "comparison_vs_sim_defaults": comparison_vs_sim_defaults,
        "sim_world_pc_defaults": sim_defaults,
        "runs": per_run,
    }

    if out_json is not None:
        out_p = Path(out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# Offline Real-Car Blackbox Replay Consistency Validator
# ---------------------------------------------------------------------------


def _ground_to_bbox_xywh(
    cam: CameraModel,
    d_bumper_m: float,
    y_lat_m: float,
    front_overhang_m: float = 0.06,
    box_w_px: float = 68.0,
    box_h_px: float = 34.0,
) -> Tuple[float, float, float, float]:
    """Project robot ground point (d_bumper_m forward, y_lat_m left) to a 640x480 pixel bbox
    whose bottom-center (x + w/2, y + h) maps back to (d_bumper_m, y_lat_m) under GroundProjector.
    """
    d_cam_mm = max(80.0, (float(d_bumper_m) + float(front_overhang_m)) * 1000.0)
    w_cam_mm = -float(y_lat_m) * 1000.0  # CameraModel +w is RIGHT, robot +y is LEFT
    u_bc, v_bc = cam.wd_to_uv(w_cam_mm, d_cam_mm)
    x = max(0.0, min(float(cam.w) - box_w_px, u_bc - 0.5 * box_w_px))
    y = max(0.0, min(float(cam.h_img) - box_h_px, v_bc - box_h_px))
    return (round(x, 2), round(y, 2), box_w_px, box_h_px)


def _find_closest_ctrl_for_frame(
    bundle: Optional[BlackboxBundle],
    frame_idx: int,
) -> Optional[Dict[str, Any]]:
    if bundle is None or not bundle.frame_timestamps or not bundle.ctrl_records:
        return None
    if frame_idx < 0 or frame_idx >= len(bundle.frame_timestamps):
        return None
    target_epoch = bundle.frame_timestamps[frame_idx]["wall_epoch"]
    candidates = [c for c in bundle.ctrl_records if c.get("epoch") is not None]
    if not candidates:
        return None
    best = min(candidates, key=lambda c: abs(float(c["epoch"]) - target_epoch))
    out = dict(best)
    out["frame_idx"] = frame_idx
    out["dt_s"] = round(float(best["epoch"]) - target_epoch, 3)
    return out


def _check_a_roundabout_offtrack_window(
    yaml_roundabout: Path,
    roundabout_bundle: Optional[BlackboxBundle],
) -> Dict[str, Any]:
    """(a) roundabout_212118 off-track window (#133..143 near-right-edge condition).

    Real-car debug frames #133..141 (zoom_crash.jpg + follow_slice.log):
      #133: near=-0.43 far=-0.17 hd=+0.26 w=0.54 conf=0.96 (old yaw=-0.15)
      #136: near=-0.53 far=+0.04 hd=+0.57 w=0.58 conf=0.92 (old yaw=-0.38)
      #139: near=-0.73 far=-0.03 hd=+0.69 w=0.53 conf=0.90 (old yaw=-0.37)
      #141: near=-0.90 far=-0.18 hd=+0.72 w=0.46 conf=0.90 (old yaw=-0.32 -> LOST!)
    Verifies the latest BrainBridge no longer commands right-turn off-track yaw.
    """
    frames = [
        # (frame_idx, near_err, far_err, heading_err, width_ratio, conf, t_s, old_yaw)
        (133, -0.43, -0.17, +0.26, 0.54, 0.96, 17.74, -0.15),
        (136, -0.53, +0.04, +0.57, 0.58, 0.92, 18.14, -0.38),
        (139, -0.73, -0.03, +0.69, 0.53, 0.90, 18.49, -0.37),
        (141, -0.90, -0.18, +0.72, 0.46, 0.90, 18.74, -0.32),
    ]

    # Attach exact real-car logged old yaw from blackbox if available
    logged_old: Dict[int, Optional[float]] = {}
    if roundabout_bundle is not None:
        for fidx, *_ in frames:
            c = _find_closest_ctrl_for_frame(roundabout_bundle, fidx)
            logged_old[fidx] = c.get("yaw") if c else None

    results: List[Dict[str, Any]] = []
    with BrainBridge(str(yaml_roundabout), auto_build=False) as brain:
        brain.reset(17.50)
        x_m = 3.40
        for fidx, ne, fe, hd, wr, conf, t_s, fallback_old in frames:
            # Right border is close, left border extends into left side of image
            right_edge_norm = max(-0.98, min(0.95, ne + wr))
            left_edge_norm = max(-1.0, ne - wr)
            obs = {
                "image_width": 640,
                "image_height": 480,
                "lane_valid": True,
                "near_valid": True,
                "far_valid": True,
                "near_error": ne,
                "far_error": fe,
                "heading_error": hd,
                "confidence": conf,
                "near_width_ratio": wr,
                "far_width_ratio": wr,
                "lane_width_ratio": wr,
                "left_edge_norm": left_edge_norm,
                "right_edge_norm": right_edge_norm,
                "has_left_edge": left_edge_norm > -0.98,
                "has_right_edge": True,
                "odom_x": x_m,
                "odom_y": 0.0,
                "odom_yaw": 0.0,
                "odom_v": 0.22,
                "odom_w": 0.0,
                "voltage": 12.10,
                "timestamp_sec": t_s,
            }
            out = brain.step(obs)
            x_m += 0.22 * 0.35
            results.append(
                {
                    "frame_idx": fidx,
                    "near_error": ne,
                    "heading_error": hd,
                    "old_real_yaw": logged_old.get(fidx, fallback_old),
                    "new_v_cmd": round(out["v_cmd"], 4),
                    "new_w_cmd": round(out["w_cmd"], 4),
                    "corridor_target_y_m": round(out["corridor_target_y_m"], 4),
                    "mode": out["active_mode_tag"],
                }
            )

    min_w = min(r["new_w_cmd"] for r in results)
    severe_ws = [r["new_w_cmd"] for r in results if r["near_error"] <= -0.50]
    passed = (min_w >= -0.05) and all(w > 0.0 for w in severe_ws)
    reason = (
        f"no_right_offtrack_yaw min_w={min_w:+.3f} severe_w={[round(w, 3) for w in severe_ws]}"
        if passed
        else f"right_offtrack_yaw_detected min_w={min_w:+.3f} severe_w={severe_ws}"
    )
    return {
        "id": "a_roundabout_offtrack_133_143",
        "passed": passed,
        "reason": reason,
        "min_w_cmd": round(min_w, 4),
        "frames": results,
    }


def _check_b_fork_yellow_stop_no_reverse(
    yaml_fork: Path,
    fork_bundle: Optional[BlackboxBundle],
) -> Dict[str, Any]:
    """(b) fork_215951 yellow-line stop transitions: clean stop, zero persistent reverse overshoot."""
    # Use measured real-car entry velocities and voltages from fork_215951 if available
    entry_cases: List[Tuple[float, float, float]] = []
    if fork_bundle is not None and fork_bundle.brake_adapt_events:
        for ev in fork_bundle.brake_adapt_events:
            v_in = float(ev.get("v_in_mps") or 0.28)
            vbat = float(ev.get("vbat_v") or 11.97)
            a_eff = float(ev.get("a_eff_mps2") or 1.44)
            entry_cases.append((v_in, vbat, a_eff))
    if not entry_cases:
        entry_cases = [(0.255, 11.98, 1.579), (0.307, 11.97, 1.343)]

    params = load_line_follower_params(yaml_fork)
    cam = CameraModel(
        h_mm=float(params.get("cam_height_m", 0.11)) * 1000.0,
        alpha_deg=float(params.get("cam_pitch_deg", 12.5)),
        f_px=float(params.get("focal_px", 325.0)),
    )
    front_overhang = float(params.get("front_overhang_m", 0.06))

    sub_reports: List[Dict[str, Any]] = []
    all_ok = True
    dt = 1.0 / 30.0

    for case_i, (v_in, vbat, a_eff) in enumerate(entry_cases, start=1):
        dyn = VehicleDynamics(
            a_eff=a_eff,
            voltage=vbat,
            latency_ms=35.0,
            seed=100 + case_i,
        )
        dyn.reset(x_mm=0.0, y_mm=0.0, yaw=0.0, v=v_in)
        yellow_x_m = 0.78  # bumper starts 0.78m before yellow line
        bumper_x_m = 0.0
        min_v = v_in
        saw_stop = False
        saw_resume = False
        persistent_rev_steps = 0
        max_persistent_rev = 0
        min_v_cmd_when_slow = 0.0

        with BrainBridge(str(yaml_fork), auto_build=False) as brain:
            brain.reset(0.0)
            for step_i in range(210):  # up to 7.0s
                t_s = step_i * dt
                odom = dyn.get_odom()
                gt = dyn.get_gt_pose()
                bumper_x_m = gt["x_mm"] / 1000.0
                d_bumper = yellow_x_m - bumper_x_m
                yellow_vis = 0.02 <= d_bumper <= 0.85
                if yellow_vis:
                    _, v_px = cam.wd_to_uv(0.0, (d_bumper + front_overhang) * 1000.0)
                    v_px = max(175.0, min(478.0, v_px))
                else:
                    v_px = 0.0

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
                    "near_width_ratio": 0.68,
                    "far_width_ratio": 0.48,
                    "left_edge_norm": -0.68,
                    "right_edge_norm": 0.68,
                    "has_left_edge": True,
                    "has_right_edge": True,
                    "yellow_detected": yellow_vis,
                    "yellow_is_geometric": yellow_vis,
                    "yellow_bottom_y": v_px,
                    "yellow_center_x": 320.0,
                    "yellow_width_ratio": 0.65 if yellow_vis else 0.0,
                    "yellow_dead_end": False,
                    "odom_x": float(odom["x"]),
                    "odom_y": float(odom["y"]),
                    "odom_yaw": float(odom["yaw"]),
                    "odom_v": float(odom["v"]),
                    "odom_w": float(odom["w"]),
                    "voltage": vbat,
                    "timestamp_sec": t_s,
                }
                out = brain.step(obs)
                v_cmd = float(out["v_cmd"])
                w_cmd = float(out["w_cmd"])

                if out["yellow_stop_active"] or out["publish_stop"]:
                    saw_stop = True
                if saw_stop and (not out["yellow_stop_active"]) and v_cmd > 0.10:
                    saw_resume = True
                if float(odom["v"]) <= 0.04 and v_cmd < -0.02:
                    persistent_rev_steps += 1
                    max_persistent_rev = max(max_persistent_rev, persistent_rev_steps)
                    min_v_cmd_when_slow = min(min_v_cmd_when_slow, v_cmd)
                else:
                    persistent_rev_steps = 0

                gt_next, _ = dyn.step(v_cmd, w_cmd, dt)
                min_v = min(min_v, float(gt_next["v"]))
                if saw_resume and gt_next["x_mm"] / 1000.0 > yellow_x_m + 0.10:
                    break

        case_ok = (
            saw_stop
            and saw_resume
            and (min_v >= -0.02)
            and (max_persistent_rev <= 1)
        )
        if not case_ok:
            all_ok = False
        sub_reports.append(
            {
                "case": case_i,
                "v_in_mps": v_in,
                "vbat_v": vbat,
                "a_eff_mps2": a_eff,
                "passed": case_ok,
                "saw_stop": saw_stop,
                "saw_resume": saw_resume,
                "min_v_mps": round(min_v, 4),
                "max_persistent_rev_steps": max_persistent_rev,
                "min_v_cmd_when_slow": round(min_v_cmd_when_slow, 4),
                "final_d_to_yellow_m": round(yellow_x_m - bumper_x_m, 4),
            }
        )

    worst_min_v = min(r["min_v_mps"] for r in sub_reports)
    reason = (
        f"clean_yellow_stops={sum(1 for r in sub_reports if r['passed'])}/{len(sub_reports)} min_v={worst_min_v:+.4f}m/s"
        if all_ok
        else f"yellow_stop_failed min_v={worst_min_v:+.4f}m/s"
    )
    return {
        "id": "b_fork_yellow_stop_no_reverse",
        "passed": all_ok,
        "reason": reason,
        "worst_min_v_mps": round(worst_min_v, 4),
        "cases": sub_reports,
    }


def _check_c_fork_dual_digit_86_93(
    yaml_fork: Path,
    fork_bundle: Optional[BlackboxBundle],
) -> Dict[str, Any]:
    """(c) fork_215951 dual-digit #86..93 transition: no late right-swerve spike when higher digit leaves FOV.

    Exact sequence from zoom_digit_wobble.jpg + follow_slice.log:
      #84..87: digit 5 (left y=+0.07m, d=0.29..0.15m) & digit 2 (right y=-0.12m, d=0.29..0.14m)
      #88..89: digit 5 exits bottom of FOV; only digit 2 visible (d=0.09..0.08m, y=-0.11..-0.10m)
      #90..93: both passed; old brain swerved hard right (yaw=-0.51) toward digit 2!
    """
    params = load_line_follower_params(yaml_fork)
    cam = CameraModel(
        h_mm=float(params.get("cam_height_m", 0.11)) * 1000.0,
        alpha_deg=float(params.get("cam_pitch_deg", 12.5)),
        f_px=float(params.get("focal_px", 325.0)),
    )
    front_overhang = float(params.get("front_overhang_m", 0.06))

    # (frame_idx, t_s, near_err, far_err, hd, wr, conf, [(digit, d_m, y_m, conf), ...])
    # Matches zoom_digit_wobble.jpg + follow_slice.log (lane=[-0.24,+0.19], cor=[-0.12,+0.07]):
    seq = [
        (84, 10.97, 0.12, 0.05, -0.07, 0.76, 0.76, [(5, 0.29, +0.065, 0.75), (2, 0.29, -0.115, 0.69)]),
        (85, 11.11, 0.12, 0.05, -0.07, 0.77, 0.72, [(5, 0.24, +0.065, 0.77), (2, 0.23, -0.115, 0.69)]),
        (86, 11.24, 0.12, 0.05, -0.07, 0.76, 0.76, [(5, 0.20, +0.065, 0.70), (2, 0.19, -0.115, 0.79)]),
        (87, 11.37, 0.12, 0.05, -0.07, 0.76, 0.76, [(5, 0.15, +0.065, 0.72), (2, 0.14, -0.115, 0.82)]),
        # Higher digit 5 exits bottom of FOV; only smaller sibling digit 2 remains close on right
        (88, 11.50, 0.14, 0.06, -0.08, 0.76, 0.76, [(2, 0.09, -0.110, 0.83)]),
        (89, 11.64, 0.15, 0.09, -0.06, 0.75, 0.82, [(2, 0.08, -0.100, 0.81)]),
        (90, 11.77, 0.18, 0.17, -0.01, 0.72, 0.93, []),
        (91, 11.91, 0.18, 0.18, -0.00, 0.71, 0.94, []),
        (93, 12.17, 0.14, 0.00, -0.14, 0.78, 0.66, []),
    ]

    old_min_yaw = -0.51
    if fork_bundle is not None:
        old_yaws = []
        for fidx in range(86, 94):
            c = _find_closest_ctrl_for_frame(fork_bundle, fidx)
            if c and c.get("yaw") is not None:
                old_yaws.append(float(c["yaw"]))
        if old_yaws:
            old_min_yaw = min(old_yaws)

    trace: List[Dict[str, Any]] = []
    with BrainBridge(str(yaml_fork), auto_build=False) as brain:
        brain.reset(10.80)
        x_m = 1.70
        for fidx, t_s, ne, fe, hd, wr, conf, dig_specs in seq:
            dets = []
            for dval, d_m, y_m, dconf in dig_specs:
                bx, by, bw, bh = _ground_to_bbox_xywh(cam, d_m, y_m, front_overhang)
                dets.append(
                    {
                        "class_name": f"digit_{dval}",
                        "digit": dval,
                        "conf": dconf,
                        "bbox": [bx, by, bw, bh],
                        "track_id": 100 + dval,
                    }
                )
            obs = {
                "image_width": 640,
                "image_height": 480,
                "lane_valid": True,
                "near_valid": True,
                "far_valid": True,
                "near_error": ne,
                "far_error": fe,
                "heading_error": hd,
                "confidence": conf,
                "near_width_ratio": wr,
                "far_width_ratio": wr,
                "lane_width_ratio": wr,
                "left_edge_norm": max(-1.0, ne - 0.95),
                "right_edge_norm": min(1.0, ne + 0.95),
                "has_left_edge": True,
                "has_right_edge": False,
                "detections": dets,
                "odom_x": x_m,
                "odom_y": 0.0,
                "odom_yaw": 0.0,
                "odom_v": 0.30,
                "odom_w": 0.0,
                "voltage": 11.97,
                "timestamp_sec": t_s,
            }
            out = brain.step(obs)
            x_m += 0.30 * 0.13
            trace.append(
                {
                    "frame_idx": fidx,
                    "active_digit": int(out["active_digit"]),
                    "digit_target_active": bool(out["digit_target_active"]),
                    "v_cmd": round(out["v_cmd"], 4),
                    "w_cmd": round(out["w_cmd"], 4),
                    "corridor_target_y_m": round(out["corridor_target_y_m"], 4),
                    "mode": out["active_mode_tag"],
                }
            )

    locked_5 = any(r["active_digit"] == 5 for r in trace if r["frame_idx"] <= 87)
    post_exit = [r for r in trace if 88 <= r["frame_idx"] <= 93]
    switched_to_2 = any(r["active_digit"] == 2 for r in post_exit)
    min_post_w = min(r["w_cmd"] for r in post_exit)

    passed = locked_5 and (not switched_to_2) and (min_post_w > -0.35)
    reason = (
        f"locked_5={locked_5} switched_to_2={switched_to_2} min_post_w={min_post_w:+.3f} (old={old_min_yaw:+.2f})"
    )
    return {
        "id": "c_fork_dual_digit_86_93",
        "passed": passed,
        "reason": reason,
        "locked_digit_5": locked_5,
        "switched_to_digit_2": switched_to_2,
        "min_post_exit_w_cmd": round(min_post_w, 4),
        "old_real_min_yaw": round(old_min_yaw, 4),
        "frames": trace,
    }


def _check_d_fork_exit_wide_junction_153(
    yaml_fork: Path,
    fork_bundle: Optional[BlackboxBundle],
) -> Dict[str, Any]:
    """(d) fork_215951 fork exit wide junction #153: preserves right-exit continuity rather than flipping left.

    Replays the exact #150 -> #153 wide-junction transition through VisionFrontend's
    LaneGeometryMirror (1:1 mirror of lane_geometry.cpp) -> BrainBridge(yaml_fork):
      - #150: right-turning approach (near center left of image center, far center right of center -> hd > 0)
      - #153: wide junction where far band spans full image width (left_clipped & right_clipped, far=0)
              while near band has right border visible (near ~ -0.26).
    Old code set raw_heading = near_center_norm (-0.26), flipping yaw strongly LEFT!
    Fixed code holds decaying positive heading (0.75 * filtered_heading > 0), preserving right-exit continuity.
    """
    params = load_line_follower_params(yaml_fork)
    vf = VisionFrontend(params, detector=None)
    roi_top = int(480 * (1.0 - float(params.get("roi_top_ratio", 0.68))))
    roi_bot = int(480 * (1.0 - float(params.get("roi_bottom_ratio", 0.18))))
    roi_h = max(20, roi_bot - roi_top)
    roi_w = 640

    def _make_mask(near_x0: int, near_x1: int, far_x0: int, far_x1: int) -> np.ndarray:
        m = np.zeros((roi_h, roi_w), dtype=np.uint8)
        mid = int(roi_h * 0.50)
        m[:mid, max(0, far_x0):min(roi_w, far_x1)] = 255
        m[mid:, max(0, near_x0):min(roi_w, near_x1)] = 255
        return m

    # #149..150: NORMAL-state approach into right-curving fork merge (near=-0.10..-0.20, far=+0.29 -> hd=+0.49)
    # #153..154: wide junction (far spans 0..640 double-clipped, near right edge at x=505 -> near~-0.21)
    mask_specs = [
        (149, 19.50, (80, 560, 210, 610)),
        (150, 19.64, (0, 520, 235, 630)),
        (153, 20.04, (0, 505, 0, 640)),
        (154, 20.17, (0, 500, 0, 640)),
    ]

    old_153_yaw = None
    if fork_bundle is not None:
        c153 = _find_closest_ctrl_for_frame(fork_bundle, 153)
        if c153 is not None:
            old_153_yaw = c153.get("yaw")

    trace: List[Dict[str, Any]] = []
    with BrainBridge(str(yaml_fork), auto_build=False) as brain:
        brain.reset(19.30)
        x_m = 3.55
        for fidx, t_s, (nx0, nx1, fx0, fx1) in mask_specs:
            bmask = _make_mask(nx0, nx1, fx0, fx1)
            seg = SegmentedFrame(
                roi_bgr=np.zeros((roi_h, roi_w, 3), dtype=np.uint8),
                binary_mask=bmask,
                roi_top=roi_top,
                roi_bottom=roi_bot,
                frame_width=roi_w,
                frame_height=480,
                contour_valid=True,
                contour_centroid=(320.0, roi_h * 0.5),
            )
            obs_lane = vf.lane_geometry.observe(seg, vf.curve_state, preferred_branch=0)
            brain_obs = {
                "image_width": 640,
                "image_height": 480,
                "lane_valid": bool(obs_lane.valid),
                "near_valid": bool(obs_lane.near_band.valid),
                "far_valid": bool(obs_lane.far_band.valid),
                "lateral_error": float(obs_lane.lateral_error),
                "near_error": float(obs_lane.near_center_norm),
                "far_error": float(obs_lane.far_center_norm),
                "heading_error": float(obs_lane.heading_proxy),
                "curvature_proxy": float(obs_lane.curvature_proxy),
                "confidence": float(obs_lane.confidence),
                "near_width_ratio": float(obs_lane.near_width_ratio),
                "far_width_ratio": float(obs_lane.far_width_ratio),
                "lane_width_ratio": float(obs_lane.lane_width_ratio),
                "near_left_edge_px": float(obs_lane.near_left_edge_px),
                "near_right_edge_px": float(obs_lane.near_right_edge_px),
                "has_left_edge": bool(obs_lane.near_left_valid),
                "has_right_edge": bool(obs_lane.near_right_valid),
                "odom_x": x_m,
                "odom_y": 0.0,
                "odom_yaw": 0.0,
                "odom_v": 0.35,
                "odom_w": 0.0,
                "voltage": 11.97,
                "timestamp_sec": t_s,
            }
            out = brain.step(brain_obs)
            x_m += 0.35 * 0.14
            trace.append(
                {
                    "frame_idx": fidx,
                    "near_err": round(float(obs_lane.near_center_norm), 4),
                    "far_err": round(float(obs_lane.far_center_norm), 4),
                    "heading_proxy": round(float(obs_lane.heading_proxy), 4),
                    "far_double_clipped": bool(
                        obs_lane.far_band.left_clipped and obs_lane.far_band.right_clipped
                    ),
                    "v_cmd": round(out["v_cmd"], 4),
                    "w_cmd": round(out["w_cmd"], 4),
                }
            )

    f153 = next(r for r in trace if r["frame_idx"] == 153)
    # Must preserve positive right-turn heading continuity (NOT flip negative like -0.26)
    # and MUST NOT command a sharp left-flip yaw (old real-car flipped to +0.42..+0.98)
    passed = (
        f153["far_double_clipped"]
        and (f153["heading_proxy"] > 0.05)
        and (f153["w_cmd"] <= 0.15)
    )
    reason = (
        f"f153_hdg={f153['heading_proxy']:+.3f} w_cmd={f153['w_cmd']:+.3f} "
        f"(old_real_yaw={old_153_yaw})"
    )
    return {
        "id": "d_fork_exit_wide_junction_153",
        "passed": passed,
        "reason": reason,
        "frame_153": f153,
        "old_real_yaw_153": old_153_yaw,
        "frames": trace,
    }


def _check_e_nominal_curve_frames_69_125(
    yaml_fork: Path,
    yaml_roundabout: Path,
    fork_bundle: Optional[BlackboxBundle],
    roundabout_bundle: Optional[BlackboxBundle],
) -> Dict[str, Any]:
    """(e) Nominal curve frames (#69 / #125): bounded, finite, non-divergent v_cmd / w_cmd."""
    # Fallback real-car extracted snapshots for #69 and #125 if bundles aren't on disk
    fallback_samples = [
        ("fork", 69, -0.02, -0.11, 0.38, 0.90),
        ("fork", 125, +0.06, +0.51, 0.33, 0.85),
        ("roundabout", 69, -0.15, +0.60, 0.25, 0.90),
        ("roundabout", 125, -0.09, -0.02, 0.26, 0.92),
    ]

    samples: List[Tuple[str, int, float, float, float, float]] = []
    for map_name, fidx, fb_yref, fb_hdg, fb_v, fb_conf in fallback_samples:
        bundle = fork_bundle if map_name == "fork" else roundabout_bundle
        c = _find_closest_ctrl_for_frame(bundle, fidx)
        if c is not None and c.get("y_ref") is not None and c.get("hdg") is not None:
            samples.append(
                (
                    map_name,
                    fidx,
                    float(c["y_ref"]),
                    float(c["hdg"]),
                    max(0.15, float(c.get("v") or fb_v)),
                    fb_conf,
                )
            )
        else:
            samples.append((map_name, fidx, fb_yref, fb_hdg, fb_v, fb_conf))

    fork_params = load_line_follower_params(yaml_fork)
    rb_params = load_line_follower_params(yaml_roundabout)

    checked: List[Dict[str, Any]] = []
    all_ok = True
    for map_name, fidx, y_ref, hdg, v_odom, conf in samples:
        yml = yaml_fork if map_name == "fork" else yaml_roundabout
        params = fork_params if map_name == "fork" else rb_params
        max_spd = float(params.get("max_speed", 0.85))
        max_yaw = float(params.get("max_yaw_rate", 2.0))

        # Convert y_ref (~[-0.2, 0.2]m) to near_error (~ -y_ref / 0.215)
        near_err = max(-0.85, min(0.85, -y_ref / 0.215))
        far_err = max(-0.95, min(0.95, near_err + hdg))

        with BrainBridge(str(yml), auto_build=False) as brain:
            brain.reset(1.0)
            out = brain.step(
                {
                    "image_width": 640,
                    "image_height": 480,
                    "lane_valid": True,
                    "near_valid": True,
                    "far_valid": True,
                    "near_error": near_err,
                    "far_error": far_err,
                    "heading_error": hdg,
                    "confidence": conf,
                    "near_width_ratio": 0.65,
                    "far_width_ratio": 0.48,
                    "lane_width_ratio": 0.58,
                    "left_edge_norm": max(-1.0, near_err - 0.65),
                    "right_edge_norm": min(1.0, near_err + 0.65),
                    "has_left_edge": True,
                    "has_right_edge": True,
                    "odom_x": 1.5,
                    "odom_y": 0.0,
                    "odom_yaw": 0.0,
                    "odom_v": v_odom,
                    "odom_w": 0.0,
                    "voltage": 12.0,
                    "timestamp_sec": 1.1,
                }
            )

        v_cmd = float(out["v_cmd"])
        w_cmd = float(out["w_cmd"])
        ok = (
            math.isfinite(v_cmd)
            and math.isfinite(w_cmd)
            and (0.05 <= v_cmd <= max_spd + 1e-6)
            and (abs(w_cmd) <= max_yaw + 1e-6)
        )
        if not ok:
            all_ok = False
        checked.append(
            {
                "map": map_name,
                "frame_idx": fidx,
                "input_y_ref": round(y_ref, 4),
                "input_hdg": round(hdg, 4),
                "v_cmd": round(v_cmd, 4),
                "w_cmd": round(w_cmd, 4),
                "bounds": {"max_speed": max_spd, "max_yaw_rate": max_yaw},
                "passed": ok,
            }
        )

    return {
        "id": "e_nominal_curve_frames_69_125",
        "passed": all_ok,
        "reason": f"nominal_bounded={sum(1 for c in checked if c['passed'])}/{len(checked)}",
        "frames": checked,
    }


def run_blackbox_replay_check(
    yaml_fork: Optional[Union[str, Path]] = None,
    yaml_roundabout: Optional[Union[str, Path]] = None,
    blackbox_dirs: Optional[Sequence[Union[str, Path]]] = None,
) -> Dict[str, Any]:
    """Run all 5 offline real-car blackbox replay consistency checks through BrainBridge."""
    y_fork = Path(yaml_fork) if yaml_fork else DEFAULT_FORK_YAML
    y_rb = Path(yaml_roundabout) if yaml_roundabout else DEFAULT_ROUNDABOUT_YAML

    fork_bundle: Optional[BlackboxBundle] = None
    rb_bundle: Optional[BlackboxBundle] = None
    search_paths = list(blackbox_dirs) if blackbox_dirs is not None else list(DEFAULT_BLACKBOX_DIRS)
    loaded_sources: List[str] = []
    for p in search_paths:
        pp = Path(p)
        if not pp.exists():
            continue
        try:
            b = load_blackbox(pp)
            loaded_sources.append(str(pp))
            tmode = str(b.params_snapshot.get("track_mode", "")).lower()
            lname = b.name.lower()
            if "round" in tmode or "round" in lname:
                rb_bundle = b
            elif "fork" in tmode or "fork" in lname:
                fork_bundle = b
        except Exception:
            continue

    checks = [
        _check_a_roundabout_offtrack_window(y_rb, rb_bundle),
        _check_b_fork_yellow_stop_no_reverse(y_fork, fork_bundle),
        _check_c_fork_dual_digit_86_93(y_fork, fork_bundle),
        _check_d_fork_exit_wide_junction_153(y_fork, fork_bundle),
        _check_e_nominal_curve_frames_69_125(y_fork, y_rb, fork_bundle, rb_bundle),
    ]
    passed_count = sum(1 for c in checks if c["passed"])
    return {
        "passed": passed_count == len(checks),
        "passed_count": passed_count,
        "total_count": len(checks),
        "yaml_fork": str(y_fork),
        "yaml_roundabout": str(y_rb),
        "loaded_blackbox_sources": loaded_sources,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Real-Car Blackbox Dynamics Calibrator & Offline Replay Consistency Validator"
    )
    ap.add_argument(
        "--input",
        action="append",
        default=[],
        help="Blackbox directory or .zip archive path (repeatable; defaults to both real-car runs).",
    )
    ap.add_argument(
        "--out-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to write calibrated real-car dynamics JSON (default: tools/sim/real_car_dynamics.json).",
    )
    ap.add_argument(
        "--fork-yaml",
        default=str(DEFAULT_FORK_YAML),
        help="Candidate fork YAML for offline blackbox replay check.",
    )
    ap.add_argument(
        "--roundabout-yaml",
        default=str(DEFAULT_ROUNDABOUT_YAML),
        help="Candidate roundabout YAML for offline blackbox replay check.",
    )
    ap.add_argument(
        "--skip-replay",
        action="store_true",
        help="Only run calibration and skip offline BrainBridge replay checks.",
    )
    ap.add_argument(
        "--replay-only",
        action="store_true",
        help="Only run offline BrainBridge replay consistency check.",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    inputs = [Path(p) for p in args.input] if args.input else [p for p in DEFAULT_BLACKBOX_DIRS if p.exists()]

    if not args.replay_only:
        if not inputs:
            print("[blackbox_calibrator] ERROR: No blackbox input directories or .zip files found.", flush=True)
            return 2
        cal = calibrate_from_blackboxes(inputs, out_json=args.out_json)
        c = cal["calibrated"]
        print(
            f"[blackbox_calibrator] Calibrated from {len(inputs)} run(s) -> {args.out_json}\n"
            f"  a_eff_mps2={c['a_eff_mps2']} (min={c['a_eff_min_mps2']}, 0.78x={c['a_eff_slippery_078x_mps2']})\n"
            f"  vbat={c['battery_voltage_nominal_v']}V range={c['battery_voltage_range_v']}\n"
            f"  camera_fps={c['camera_fps_hz']}Hz read_p50={c['camera_read_p50_ms']}ms "
            f"yolo_e2e_p50={c['yolo_e2e_p50_ms']}ms est_pipeline_lat={c['est_yolo_pipeline_latency_ms']}ms",
            flush=True,
        )

    if not args.skip_replay:
        rep = run_blackbox_replay_check(
            yaml_fork=args.fork_yaml,
            yaml_roundabout=args.roundabout_yaml,
            blackbox_dirs=inputs if inputs else None,
        )
        print(
            f"[blackbox_calibrator] Offline Replay Check: {rep['passed_count']}/{rep['total_count']} PASSED",
            flush=True,
        )
        for ch in rep["checks"]:
            st = "PASS" if ch["passed"] else "FAIL"
            print(f"  [{st}] {ch['id']}: {ch['reason']}", flush=True)
        if not rep["passed"]:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
