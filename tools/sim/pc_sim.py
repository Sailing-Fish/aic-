#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Windows PC Closed-Loop Simulator Runner & Dual-Pane OpenCV GUI.

Wires:
    PCSimWorld -> VisionFrontend (MockNoisyDetector | RealYoloDetector)
               -> BrainBridge (real-car pure C++17 DLL) -> PCSimWorld

Supports:
    - Interactive dual-pane GUI (--gui, 1280x480 resizable):
        Left pane  : Top-down global map, 4 wheels, 3 discs, props, trail, AIC scoreboard
        Right pane : 640x480 camera view + lane/band/edge/corridor/yellow/red/digit/CBF HUD
        Keys       : Space pause/resume, n single-step, 1/2/4/0 speed, r reset,
                     m toggle mock/YOLO, q/Esc quit
    - Headless CLI mode (--headless, --record out.mp4, --snapshot out.png)
    - Direct loading of tools/config/finals_fork.yaml / finals_roundabout.yaml
      plus --override key=val and --params-json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Guard against tools/sim/profile.py shadowing Python stdlib `profile` (needed for torch/ultralytics)
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[1]
_this_dir_norm = os.path.normcase(str(_THIS_DIR))
sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or ".")) != _this_dir_norm]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import yaml

from tools.sim.brain_bridge import BrainBridge
from tools.sim.sim_world_pc import PCSimWorld, StepInfo, resolve_track_json
from tools.sim.vision_mirror import (
    DEFAULT_YOLO_WEIGHTS,
    MockNoisyDetector,
    RealYoloDetector,
    VisionFrontend,
    render_debug_overlay,
    unwrap_ros_params,
)


TRACK_CONFIG_MAP: Dict[str, Path] = {
    "fork_s": REPO_ROOT / "tools" / "config" / "finals_fork.yaml",
    "fork": REPO_ROOT / "tools" / "config" / "finals_fork.yaml",
    "roundabout": REPO_ROOT / "tools" / "config" / "finals_roundabout.yaml",
}


def resolve_default_yaml(track: str) -> Path:
    key = Path(track).stem.lower()
    if key in TRACK_CONFIG_MAP:
        return TRACK_CONFIG_MAP[key]
    if "round" in key:
        return TRACK_CONFIG_MAP["roundabout"]
    return TRACK_CONFIG_MAP["fork_s"]


def parse_override_value(raw: str) -> Any:
    """Parse CLI key=val string into bool/int/float/list/dict/str."""
    s = raw.strip()
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        if s.startswith(("0x", "-0x")):
            return int(s, 16)
        if "." not in s and "e" not in low:
            return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def parse_overrides_list(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"Invalid --override '{item}': expected key=val")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Empty key in --override '{item}'")
        out[k] = parse_override_value(v)
    return out


def load_params_json(spec: Optional[str]) -> Dict[str, Any]:
    """Load parameter overrides from a JSON file path or inline JSON object string."""
    if not spec:
        return {}
    s = spec.strip()
    p = Path(s)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
    elif s.startswith("{"):
        data = json.loads(s)
    else:
        raise FileNotFoundError(f"params JSON file not found: {spec}")
    if not isinstance(data, dict):
        raise ValueError("params-json must decode to a JSON object/dict")
    return unwrap_ros_params(data) if "line_follower_node" in data else dict(data)


def load_yaml_params(
    yaml_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load line_follower_node.ros__parameters from YAML and apply overrides."""
    params = unwrap_ros_params(Path(yaml_path))
    if overrides:
        params.update(overrides)
    return params


def save_image_unicode(path: Union[str, Path], bgr_img: np.ndarray) -> Path:
    """Save image safely on Windows even when path contains non-ASCII characters."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, bgr_img)
    if not ok:
        raise RuntimeError(f"Failed to encode image as {ext}")
    out_path.write_bytes(buf.tobytes())
    return out_path


def _open_video_writer(path: Union[str, Path], fps: float, width: int, height: int) -> Tuple[cv2.VideoWriter, Path]:
    """Open an OpenCV VideoWriter, handling Windows Unicode paths safely via temp fallback if needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_str in ("mp4v", "avc1", "XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(width), int(height)))
        if writer.isOpened():
            return writer, out_path
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {out_path}")


@dataclass
class SimConfig:
    track: str = "fork_s"
    yaml_path: Optional[str] = None
    param_overrides: Dict[str, Any] = field(default_factory=dict)
    detector: str = "mock"  # "mock" | "yolo"
    yolo_weights: str = str(DEFAULT_YOLO_WEIGHTS)
    rate_hz: float = 30.0
    max_time_s: float = 60.0
    seed: int = 42
    # Props
    random_yellow: int = 0
    cones: int = 0
    no_entry: int = 0
    digits: int = 0
    prop_seed: int = 42
    include_base_yellow: bool = True
    clear_base_yellow: bool = False
    explicit_props: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    start_pose: Optional[Union[Dict[str, Any], Sequence[float]]] = None
    # Vehicle dynamics & domain randomization
    wheelbase_m: float = 0.1395
    max_steer_deg: float = 21.0
    tau_speed: float = 0.61
    tau_steer: float = 0.26
    a_eff: float = 1.44
    voltage: float = 12.0
    voltage_decel_slope: float = 0.08
    a_eff_scale: float = 1.0
    steer_scale: float = 1.0
    speed_scale: float = 1.0
    odom_scale_mismatch: float = 1.0
    latency_ms: float = 0.0
    latency_jitter_ms: float = 0.0
    # Camera / lighting
    cam_offset_mm: float = 120.0
    blur: float = 0.0
    gain: float = 1.0
    color_cast: Optional[Union[str, Sequence[float]]] = None
    glare: float = 0.0
    # Mock detector noise
    mock_bbox_jitter_px: float = 2.0
    mock_conf_noise_std: float = 0.05
    mock_base_recall: float = 0.96
    mock_flicker_prob: float = 0.03
    # Episode termination guard for permanent brain stop
    permanent_stop_grace_s: float = 1.5


@dataclass
class SimRunResult:
    track: str
    detector: str
    steps: int
    sim_time_s: float
    wall_time_s: float
    done: bool
    done_reason: str
    finished: bool
    dnf: bool
    dnf_reason: str
    lap_time_s: float
    total_penalty_s: float
    total_bonus_s: float
    net_score_s: float
    boundary_press_time_s: float
    boundary_penalty_s: float
    cone_hits: int
    no_entry_hits: int
    yellow_violations: int
    yellow_stops_ok: int
    bonus_digits_collected: List[int]
    max_s_m: float
    final_s_m: float
    min_v_mps: float
    max_v_mps: float
    max_abs_w_cmd: float
    w_cmd_std: float
    w_sign_flips: int
    cbf_active_steps: int
    active_digits_locked: List[int]
    yellow_phases_seen: List[str]
    max_heading_diff_deg: float
    max_s_backtrack_m: float
    referee_status: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    snapshot_path: Optional[str] = None
    record_path: Optional[str] = None


def _wrap_angle(rad: float) -> float:
    return math.atan2(math.sin(rad), math.cos(rad))


def draw_camera_hud(
    bgr_img: np.ndarray,
    debug_data: Dict[str, Any],
    brain_out: Optional[Dict[str, Any]] = None,
    step_info: Optional[StepInfo] = None,
    detector_mode: str = "mock",
    speed_mult: float = 1.0,
    paused: bool = False,
    fps_actual: float = 0.0,
) -> np.ndarray:
    """Render 640x480 right-pane camera view with vision overlay, corridor bounds, and clean HUD."""
    # Base vision overlay (bands, edges, yellow, CV red, detections, mask)
    overlay = render_debug_overlay(bgr_img, debug_data, brain_info=None)
    h, w = overlay.shape[:2]

    # Draw solid dark top (48px) & bottom (54px) bars matching the left top-down HUD height
    # (y=0..48 is sky above horizon ~168; y=426..480 is below active lane ROI bottom ~393)
    top_bar_h = 48
    bot_bar_h = 54
    cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (20, 22, 26), -1)
    cv2.line(overlay, (0, top_bar_h), (w - 1, top_bar_h), (65, 70, 80), 1)

    cv2.rectangle(overlay, (0, h - bot_bar_h), (w, h), (20, 22, 26), -1)
    cv2.line(overlay, (0, h - bot_bar_h), (w - 1, h - bot_bar_h), (65, 70, 80), 1)

    # Draw corridor bounds from BrainBridge (green bracket + magenta target marker)
    if brain_out:
        c_row = int(brain_out.get("corridor_row_y", -1))
        c_l = int(brain_out.get("corridor_left_px", -1))
        c_r = int(brain_out.get("corridor_right_px", -1))
        c_ref = int(brain_out.get("corridor_ref_px", -1))
        if top_bar_h < c_row < (h - bot_bar_h) and 0 <= c_l < c_r <= w:
            cl_i = max(2, min(w - 3, c_l))
            cr_i = max(2, min(w - 3, c_r))
            cv2.line(overlay, (cl_i, c_row), (cr_i, c_row), (40, 245, 90), 3, cv2.LINE_AA)
            cv2.line(overlay, (cl_i, c_row - 10), (cl_i, c_row + 10), (40, 245, 90), 2, cv2.LINE_AA)
            cv2.line(overlay, (cr_i, c_row - 10), (cr_i, c_row + 10), (40, 245, 90), 2, cv2.LINE_AA)
            if 0 <= c_ref < w:
                cv2.drawMarker(
                    overlay,
                    (int(c_ref), c_row),
                    (255, 80, 255),
                    cv2.MARKER_DIAMOND,
                    12,
                    2,
                    cv2.LINE_AA,
                )

    # Top bar line 1: State / Mode / Detector / Yellow status
    state_name = str(brain_out.get("state_name", "INIT")) if brain_out else "INIT"
    mode_tag = str(brain_out.get("active_mode_tag", "-")) if brain_out else "-"
    yel_phase = str(brain_out.get("yellow_phase", "none")) if brain_out else "none"
    cbf_on = bool(brain_out.get("cbf_active", False)) if brain_out else False
    cbf_src = str(brain_out.get("cbf_limiting_source", "")) if brain_out else ""

    det_label = detector_mode.upper()
    row1 = f"STATE:{state_name}  MODE:{mode_tag}  DET:{det_label}  YEL:{yel_phase}"
    cv2.putText(
        overlay, row1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 242, 250), 1, cv2.LINE_AA
    )

    # CBF badge on top-right
    if cbf_on:
        cbf_txt = f"CBF:{cbf_src or 'ACT'}"
        cbf_bg = (30, 50, 220)
    else:
        cbf_txt = "CBF:OFF"
        cbf_bg = (55, 65, 78)
    cv2.rectangle(overlay, (w - 118, 4), (w - 6, 23), cbf_bg, -1)
    cv2.putText(
        overlay, cbf_txt[:14], (w - 113, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA
    )

    # Top bar line 2: Vision lane geometry & yellow distance
    curve = debug_data.get("curve", {})
    yel = debug_data.get("yellow", {})
    yel_str = ""
    if yel.get("detected"):
        yel_str = f" Yd={yel.get('d_bumper_m', 0.0):.2f}m" + (" [END]" if yel.get("dead_end_ahead") else "")
    row2 = (
        f"SEG:{debug_data.get('lane_seg_mode', '?')[:8]} "
        f"near={curve.get('near_error', 0.0):+.2f} far={curve.get('far_error', 0.0):+.2f} "
        f"hdg={curve.get('heading_proxy', 0.0):+.2f} conf={curve.get('confidence', 0.0):.2f}{yel_str}"
    )
    cv2.putText(
        overlay, row2, (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (140, 235, 255), 1, cv2.LINE_AA
    )

    # Bottom bar line 1: Cmd / Actual / Corridor / Obstacles / Digit
    v_cmd = float(brain_out.get("v_cmd", 0.0)) if brain_out else 0.0
    w_cmd = float(brain_out.get("w_cmd", 0.0)) if brain_out else 0.0
    v_act = float(step_info.gt_pose["v"]) if step_info else 0.0
    w_act = float(step_info.gt_pose["w"]) if step_info else 0.0
    obs_n = int(brain_out.get("obstacle_count", 0)) if brain_out else 0
    act_dig = int(brain_out.get("active_digit", 0)) if brain_out else 0
    pref_br = int(brain_out.get("preferred_branch", 0)) if brain_out else 0
    c_lm = float(brain_out.get("corridor_left_m", 0.0)) if brain_out else 0.0
    c_rm = float(brain_out.get("corridor_right_m", 0.0)) if brain_out else 0.0
    c_ym = float(brain_out.get("corridor_target_y_m", 0.0)) if brain_out else 0.0

    b_row1 = (
        f"CMD v={v_cmd:+.2f} w={w_cmd:+.2f} | ACT v={v_act:+.2f} w={w_act:+.2f} | "
        f"OBS:{obs_n} DIG:{act_dig} BR:{pref_br:+d} CORR:[{c_rm:+.2f},{c_lm:+.2f}]->{c_ym:+.2f}m"
    )
    cv2.putText(
        overlay, b_row1, (8, h - 27), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (240, 245, 250), 1, cv2.LINE_AA
    )

    # Bottom bar line 2: Playback & interactive hotkeys
    spd_txt = "PAUSED" if paused else ("MAX" if speed_mult <= 0 else f"{speed_mult:g}x")
    fps_txt = f"{fps_actual:.0f}fps" if fps_actual > 0 else ""
    b_row2 = (
        f"SPD:{spd_txt} {fps_txt}  |  [Space]Pause [n]Step [1/2/4/0]Spd [m]Mock/YOLO [r]Rst [q]Quit"
    )
    cv2.putText(
        overlay, b_row2, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (175, 190, 205), 1, cv2.LINE_AA
    )

    return overlay


def compose_dual_pane(topdown_bgr: np.ndarray, cam_hud_bgr: np.ndarray) -> np.ndarray:
    """Combine left top-down map (640x480) and right camera HUD (640x480) into 1280x480 BGR."""
    if topdown_bgr.shape[:2] != (480, 640):
        topdown_bgr = cv2.resize(topdown_bgr, (640, 480), interpolation=cv2.INTER_AREA)
    if cam_hud_bgr.shape[:2] != (480, 640):
        cam_hud_bgr = cv2.resize(cam_hud_bgr, (640, 480), interpolation=cv2.INTER_AREA)
    dual = np.hstack([topdown_bgr, cam_hud_bgr])
    cv2.line(dual, (640, 0), (640, 479), (80, 88, 100), 2)
    return dual


class ClosedLoopRunner:
    """Closed-loop orchestrator wiring PCSimWorld, VisionFrontend, and BrainBridge."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.yaml_path = Path(config.yaml_path) if config.yaml_path else resolve_default_yaml(config.track)
        self.params = load_yaml_params(self.yaml_path, config.param_overrides)

        self.world = PCSimWorld(
            track=config.track,
            rate_hz=config.rate_hz,
            wheelbase_m=config.wheelbase_m,
            max_steer_deg=config.max_steer_deg,
            tau_speed=config.tau_speed,
            tau_steer=config.tau_steer,
            a_eff=config.a_eff,
            voltage=config.voltage,
            voltage_decel_slope=config.voltage_decel_slope,
            a_eff_scale=config.a_eff_scale,
            steer_scale=config.steer_scale,
            speed_scale=config.speed_scale,
            odom_scale_mismatch=config.odom_scale_mismatch,
            latency_ms=config.latency_ms,
            latency_jitter_ms=config.latency_jitter_ms,
            cam_offset_mm=config.cam_offset_mm,
            blur=config.blur,
            gain=config.gain,
            color_cast=config.color_cast,
            glare=config.glare,
            random_yellow=config.random_yellow,
            cones=config.cones,
            no_entry=config.no_entry,
            digits=config.digits,
            prop_seed=config.prop_seed,
            include_base_yellow=config.include_base_yellow,
            clear_base_yellow=config.clear_base_yellow,
            max_time_s=config.max_time_s,
            seed=config.seed,
        )

        self.brain = BrainBridge(str(self.yaml_path), param_overrides=config.param_overrides)

        self.detector_mode = config.detector.strip().lower()
        if self.detector_mode not in ("mock", "yolo"):
            raise ValueError(f"Unsupported detector mode '{config.detector}', expected 'mock' or 'yolo'")

        self._mock_detector = self._make_mock_detector()
        self._yolo_detector: Optional[RealYoloDetector] = None
        if self.detector_mode == "yolo":
            self._yolo_detector = self._make_yolo_detector()

        active_det = self._yolo_detector if self.detector_mode == "yolo" else self._mock_detector
        self.vision = VisionFrontend(self.params, detector=active_det)

        self.step_info: StepInfo = self.world.last_step_info  # type: ignore[assignment]
        self.last_obs: Dict[str, Any] = {}
        self.last_debug: Dict[str, Any] = {}
        self.last_brain_out: Dict[str, Any] = {}
        self._obs_fresh: bool = False

        # Episode telemetry accumulators
        self.steps = 0
        self.min_v_mps = 0.0
        self.max_v_mps = 0.0
        self.max_s_m = 0.0
        self.max_s_backtrack_m = 0.0
        self.max_heading_diff_deg = 0.0
        self.w_cmd_list: List[float] = []
        self.cbf_active_steps = 0
        self.active_digits_locked: List[int] = []
        self.yellow_phases_seen: List[str] = []
        self._perm_stop_hold_s = 0.0
        self._done_override_reason = ""

        self.reset()

    def _make_mock_detector(self) -> MockNoisyDetector:
        return MockNoisyDetector(
            seed=self.config.seed,
            bbox_jitter_px=self.config.mock_bbox_jitter_px,
            conf_noise_std=self.config.mock_conf_noise_std,
            base_recall=self.config.mock_base_recall,
            flicker_prob=self.config.mock_flicker_prob,
            detect_digits=True,
            detect_obstacles=False,
            detect_yellow=False,
            cam=self.world.cam,
        )

    def _make_yolo_detector(self) -> RealYoloDetector:
        det = RealYoloDetector(weights_path=self.config.yolo_weights, conf_thresh=0.25)
        det.warmup()
        return det

    def set_detector_mode(self, mode: str) -> str:
        mode = mode.strip().lower()
        if mode == "yolo":
            if self._yolo_detector is None:
                self._yolo_detector = self._make_yolo_detector()
            self._yolo_detector.reset()
            self.vision.set_detector(self._yolo_detector)
            self.detector_mode = "yolo"
        else:
            self._mock_detector.reset(seed=self.config.seed)
            self.vision.set_detector(self._mock_detector)
            self.detector_mode = "mock"
        self._obs_fresh = False
        return self.detector_mode

    def toggle_detector_mode(self) -> str:
        return self.set_detector_mode("yolo" if self.detector_mode == "mock" else "mock")

    def reset(
        self,
        pose: Optional[Union[Dict[str, Any], Sequence[float]]] = None,
        props: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ) -> StepInfo:
        use_pose = self.config.start_pose if pose is None else pose
        use_props = self.config.explicit_props if props is None else props
        self.step_info = self.world.reset(pose=use_pose, props=use_props, seed=seed)
        self.brain.reset(now_sec=0.0)
        self.vision.reset()
        if self.detector_mode == "mock":
            self._mock_detector.reset(seed=self.config.seed if seed is None else int(seed))
        elif self._yolo_detector is not None:
            self._yolo_detector.reset()

        self.steps = 0
        v0 = float(self.step_info.gt_pose["v"])
        s0 = float(self.step_info.referee_status["s_m"])
        self.min_v_mps = v0
        self.max_v_mps = v0
        self.max_s_m = s0
        self.max_s_backtrack_m = 0.0
        self.max_heading_diff_deg = 0.0
        self.w_cmd_list = []
        self.cbf_active_steps = 0
        self.active_digits_locked = []
        self.yellow_phases_seen = []
        self._perm_stop_hold_s = 0.0
        self._done_override_reason = ""
        self._obs_fresh = False

        # Compute initial vision observation (no world step yet) so frame 0 HUD is populated
        self._observe_current_frame()
        self.last_brain_out = {}
        return self.step_info

    def _observe_current_frame(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        gt = self.step_info.gt_pose
        world_ctx = {
            "pose": (gt["cam_x_mm"], gt["cam_y_mm"], gt["cam_yaw"]),
            "world_props": self.step_info.world_props,
            "cam": self.world.cam,
        }
        obs, dbg = self.vision.process(
            self.step_info.bgr_img,
            world_context=world_ctx,
            preferred_branch=self.brain.preferred_branch,
            timestamp_sec=float(self.step_info.odom["timestamp_sec"]),
        )
        self.last_obs = obs
        self.last_debug = dbg
        self._obs_fresh = True
        return obs, dbg

    def step_once(self) -> Tuple[StepInfo, Dict[str, Any], Dict[str, Any]]:
        """Execute one closed-loop control cycle: Vision -> Brain -> World."""
        if self._obs_fresh and self.last_obs:
            obs, dbg = self.last_obs, self.last_debug
        else:
            obs, dbg = self._observe_current_frame()

        odom = self.step_info.odom
        brain_in = dict(obs)
        brain_in.update(
            {
                "odom_x": float(odom["x"]),
                "odom_y": float(odom["y"]),
                "odom_yaw": float(odom["yaw"]),
                "odom_v": float(odom["v"]),
                "odom_w": float(odom["w"]),
                "voltage": float(odom["voltage"]),
                "timestamp_sec": float(odom["timestamp_sec"]),
            }
        )
        brain_out = self.brain.step(brain_in)
        self.last_brain_out = brain_out

        # Step physics & referee
        v_cmd = float(brain_out["v_cmd"])
        w_cmd = float(brain_out["w_cmd"])
        self.step_info = self.world.step(v_cmd, w_cmd)
        self._obs_fresh = False
        self.steps += 1

        # Telemetry tracking
        gt = self.step_info.gt_pose
        ref = self.step_info.referee_status
        v_now = float(gt["v"])
        s_now = float(ref["s_m"])
        self.min_v_mps = min(self.min_v_mps, v_now)
        self.max_v_mps = max(self.max_v_mps, v_now)
        if s_now > self.max_s_m:
            self.max_s_m = s_now
        else:
            self.max_s_backtrack_m = max(self.max_s_backtrack_m, self.max_s_m - s_now)

        # Heading alignment vs forward track tangent
        _, _, track_yaw, _, _ = self.world.renderer.pose_at_s(s_now)
        hdg_err_deg = abs(math.degrees(_wrap_angle(float(gt["yaw"]) - track_yaw)))
        if hdg_err_deg > self.max_heading_diff_deg:
            self.max_heading_diff_deg = hdg_err_deg

        self.w_cmd_list.append(w_cmd)
        if brain_out.get("cbf_active"):
            self.cbf_active_steps += 1
        dig = int(brain_out.get("active_digit", 0))
        if dig > 0 and (not self.active_digits_locked or self.active_digits_locked[-1] != dig):
            self.active_digits_locked.append(dig)
        yp = str(brain_out.get("yellow_phase", "none"))
        if yp != "none" and (not self.yellow_phases_seen or self.yellow_phases_seen[-1] != yp):
            self.yellow_phases_seen.append(yp)

        # Detect permanent finish stop even if slightly before referee finish zone
        if brain_out.get("permanent_stop") and abs(v_now) <= 0.03:
            self._perm_stop_hold_s += self.world.dt
            if (
                not self.step_info.done
                and self._perm_stop_hold_s >= self.config.permanent_stop_grace_s
            ):
                self._done_override_reason = "brain_permanent_stop"
        else:
            self._perm_stop_hold_s = 0.0

        return self.step_info, brain_out, self.last_debug

    @property
    def is_done(self) -> bool:
        return bool(self.step_info.done or self._done_override_reason)

    @property
    def done_reason(self) -> str:
        return self.step_info.done_reason or self._done_override_reason

    def render_dual_pane(
        self, speed_mult: float = 1.0, paused: bool = False, fps_actual: float = 0.0
    ) -> np.ndarray:
        if not self._obs_fresh:
            self._observe_current_frame()
        topdown = self.world.render_topdown(width=640, height=480)
        cam_hud = draw_camera_hud(
            self.step_info.bgr_img,
            self.last_debug,
            brain_out=self.last_brain_out,
            step_info=self.step_info,
            detector_mode=self.detector_mode,
            speed_mult=speed_mult,
            paused=paused,
            fps_actual=fps_actual,
        )
        return compose_dual_pane(topdown, cam_hud)

    def build_result(
        self,
        wall_time_s: float,
        snapshot_path: Optional[str] = None,
        record_path: Optional[str] = None,
    ) -> SimRunResult:
        ref = copy.deepcopy(self.step_info.referee_status)
        w_arr = np.asarray(self.w_cmd_list, dtype=np.float64) if self.w_cmd_list else np.zeros(1)
        w_std = float(np.std(w_arr))
        max_abs_w = float(np.max(np.abs(w_arr)))
        # Count significant sign flips (|w| > 0.15 rad/s)
        flips = 0
        prev_sign = 0
        for wv in self.w_cmd_list:
            if abs(wv) > 0.15:
                sgn = 1 if wv > 0 else -1
                if prev_sign != 0 and sgn != prev_sign:
                    flips += 1
                prev_sign = sgn

        return SimRunResult(
            track=self.world.track_name,
            detector=self.detector_mode,
            steps=self.steps,
            sim_time_s=float(self.step_info.gt_pose["t_s"]),
            wall_time_s=float(wall_time_s),
            done=self.is_done,
            done_reason=self.done_reason,
            finished=bool(ref.get("finished", False)),
            dnf=bool(ref.get("dnf", False)),
            dnf_reason=str(ref.get("dnf_reason", "")),
            lap_time_s=float(ref.get("lap_time_s", 0.0)),
            total_penalty_s=float(ref.get("total_penalty_s", 0.0)),
            total_bonus_s=float(ref.get("total_bonus_s", 0.0)),
            net_score_s=float(ref.get("net_score_s", 0.0)),
            boundary_press_time_s=float(ref.get("boundary_press_time_s", 0.0)),
            boundary_penalty_s=float(ref.get("boundary_penalty_s", 0.0)),
            cone_hits=int(ref.get("cone_hits", 0)),
            no_entry_hits=int(ref.get("no_entry_hits", 0)),
            yellow_violations=int(ref.get("yellow_violations", 0)),
            yellow_stops_ok=int(ref.get("yellow_stops_ok", 0)),
            bonus_digits_collected=list(ref.get("bonus_digits_collected", [])),
            max_s_m=float(self.max_s_m),
            final_s_m=float(ref.get("s_m", self.step_info.gt_pose["s_m"])),
            min_v_mps=float(self.min_v_mps),
            max_v_mps=float(self.max_v_mps),
            max_abs_w_cmd=max_abs_w,
            w_cmd_std=w_std,
            w_sign_flips=flips,
            cbf_active_steps=self.cbf_active_steps,
            active_digits_locked=list(self.active_digits_locked),
            yellow_phases_seen=list(self.yellow_phases_seen),
            max_heading_diff_deg=float(self.max_heading_diff_deg),
            max_s_backtrack_m=float(self.max_s_backtrack_m),
            referee_status=ref,
            events=list(ref.get("events", [])),
            snapshot_path=snapshot_path,
            record_path=record_path,
        )

    def run_headless(
        self,
        record_path: Optional[Union[str, Path]] = None,
        snapshot_path: Optional[Union[str, Path]] = None,
        snapshot_time_s: Optional[float] = None,
        stop_condition: Optional[Callable[[StepInfo, Dict[str, Any]], bool]] = None,
    ) -> SimRunResult:
        """Run full episode headlessly with optional MP4 recording and dual-pane PNG snapshot."""
        writer: Optional[cv2.VideoWriter] = None
        saved_record: Optional[str] = None
        saved_snapshot: Optional[str] = None
        snapshot_taken = False
        best_snapshot_frame: Optional[np.ndarray] = None
        best_snapshot_score = -1.0

        t_wall0 = time.perf_counter()
        try:
            if record_path:
                writer, rec_p = _open_video_writer(record_path, self.world.rate_hz, 1280, 480)
                saved_record = str(rec_p)

            while not self.is_done:
                info, brain_out, dbg = self.step_once()
                t_sim = float(info.gt_pose["t_s"])

                need_frame = (writer is not None) or (snapshot_path is not None and not snapshot_taken)
                frame_dual: Optional[np.ndarray] = None
                if need_frame:
                    # Only render full dual-pane when recording or considering snapshot
                    if writer is not None:
                        frame_dual = self.render_dual_pane(speed_mult=0.0, paused=False)
                        writer.write(frame_dual)

                    if snapshot_path is not None and not snapshot_taken:
                        if snapshot_time_s is not None:
                            if t_sim >= float(snapshot_time_s):
                                if frame_dual is None:
                                    frame_dual = self.render_dual_pane(speed_mult=0.0, paused=False)
                                save_image_unicode(snapshot_path, frame_dual)
                                saved_snapshot = str(Path(snapshot_path).resolve())
                                snapshot_taken = True
                        else:
                            # Auto-pick a rich active mid-run frame (with motion + trajectory + visible features)
                            if t_sim >= 1.2 and (self.steps % 3 == 0 or t_sim >= 3.0):
                                score = 1.0
                                if info.gt_pose["v"] > 0.15:
                                    score += 1.5
                                if dbg.get("cv_red_boxes"):
                                    score += 2.5
                                if dbg.get("detections"):
                                    score += 2.5
                                if dbg.get("yellow", {}).get("detected"):
                                    score += 2.0
                                if brain_out.get("cbf_active"):
                                    score += 1.5
                                if score > best_snapshot_score:
                                    best_snapshot_score = score
                                    best_snapshot_frame = (
                                        frame_dual
                                        if frame_dual is not None
                                        else self.render_dual_pane(speed_mult=0.0, paused=False)
                                    )
                                if t_sim >= 4.0 and best_snapshot_score >= 4.5:
                                    save_image_unicode(snapshot_path, best_snapshot_frame)
                                    saved_snapshot = str(Path(snapshot_path).resolve())
                                    snapshot_taken = True

                if stop_condition is not None and stop_condition(info, brain_out):
                    self._done_override_reason = "stop_condition"
                    break

            # Ensure snapshot is always written if requested
            if snapshot_path is not None and not snapshot_taken:
                final_frame = (
                    best_snapshot_frame
                    if best_snapshot_frame is not None
                    else self.render_dual_pane(speed_mult=0.0, paused=False)
                )
                save_image_unicode(snapshot_path, final_frame)
                saved_snapshot = str(Path(snapshot_path).resolve())
        finally:
            if writer is not None:
                writer.release()

        wall_s = time.perf_counter() - t_wall0
        return self.build_result(
            wall_time_s=wall_s,
            snapshot_path=saved_snapshot,
            record_path=saved_record,
        )

    def run_gui(
        self,
        record_path: Optional[Union[str, Path]] = None,
        snapshot_path: Optional[Union[str, Path]] = None,
        window_name: str = "AIC PC Simulator (Dual-Pane)",
    ) -> SimRunResult:
        """Interactive 1280x480 resizable dual-pane OpenCV GUI."""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 480)

        paused = False
        speed_mult = 1.0  # 1, 2, 4, or 0 (unthrottled)
        writer: Optional[cv2.VideoWriter] = None
        saved_record: Optional[str] = None
        saved_snapshot: Optional[str] = None
        fps_ema = self.world.rate_hz

        t_wall0 = time.perf_counter()
        last_tick = t_wall0
        try:
            if record_path:
                writer, rec_p = _open_video_writer(record_path, self.world.rate_hz, 1280, 480)
                saved_record = str(rec_p)

            while True:
                step_now = False
                if not paused and not self.is_done:
                    step_now = True

                t_frame0 = time.perf_counter()
                if step_now:
                    self.step_once()

                now = time.perf_counter()
                dt_wall = max(1e-4, now - last_tick)
                last_tick = now
                fps_ema = 0.85 * fps_ema + 0.15 * (1.0 / dt_wall)

                dual = self.render_dual_pane(
                    speed_mult=speed_mult,
                    paused=paused or self.is_done,
                    fps_actual=fps_ema,
                )
                if writer is not None and step_now:
                    writer.write(dual)

                cv2.imshow(window_name, dual)

                # Compute waitKey delay for real-time pacing
                if paused or self.is_done:
                    wait_ms = 30
                elif speed_mult <= 0.0:
                    wait_ms = 1
                else:
                    target_ms = (self.world.dt / speed_mult) * 1000.0
                    elapsed_ms = (time.perf_counter() - t_frame0) * 1000.0
                    wait_ms = max(1, int(round(target_ms - elapsed_ms)))

                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("n"):
                    paused = True
                    if not self.is_done:
                        self.step_once()
                elif key == ord("1"):
                    speed_mult = 1.0
                elif key == ord("2"):
                    speed_mult = 2.0
                elif key == ord("4"):
                    speed_mult = 4.0
                elif key == ord("0"):
                    speed_mult = 0.0
                elif key == ord("r"):
                    self.reset()
                    paused = False
                elif key == ord("m"):
                    new_mode = self.toggle_detector_mode()
                    print(f"[pc_sim] Switched detector -> {new_mode}", flush=True)
                elif key == ord("s") and snapshot_path:
                    save_image_unicode(snapshot_path, dual)
                    saved_snapshot = str(Path(snapshot_path).resolve())
                    print(f"[pc_sim] Saved snapshot -> {saved_snapshot}", flush=True)

                # Stop if user closed the window via X button
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

            if snapshot_path and not saved_snapshot:
                dual = self.render_dual_pane(speed_mult=speed_mult, paused=paused, fps_actual=fps_ema)
                save_image_unicode(snapshot_path, dual)
                saved_snapshot = str(Path(snapshot_path).resolve())
        finally:
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

        wall_s = time.perf_counter() - t_wall0
        return self.build_result(
            wall_time_s=wall_s,
            snapshot_path=saved_snapshot,
            record_path=saved_record,
        )

    def close(self) -> None:
        self.brain.close()

    def __enter__(self) -> "ClosedLoopRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def run_episode(
    config: SimConfig,
    record_path: Optional[Union[str, Path]] = None,
    snapshot_path: Optional[Union[str, Path]] = None,
    snapshot_time_s: Optional[float] = None,
    stop_condition: Optional[Callable[[StepInfo, Dict[str, Any]], bool]] = None,
) -> SimRunResult:
    """Convenience helper to run a single headless closed-loop episode and clean up resources."""
    with ClosedLoopRunner(config) as runner:
        return runner.run_headless(
            record_path=record_path,
            snapshot_path=snapshot_path,
            snapshot_time_s=snapshot_time_s,
            stop_condition=stop_condition,
        )


def _format_summary(res: SimRunResult) -> str:
    status = "FINISH" if res.finished else ("DNF" if res.dnf else res.done_reason.upper() or "STOP")
    rt_factor = (res.sim_time_s / max(1e-6, res.wall_time_s))
    lines = [
        "=" * 76,
        f"AIC PC Simulator Result | track={res.track} | detector={res.detector} | status={status}",
        "-" * 76,
        f"  Sim Time     : {res.sim_time_s:6.2f}s ({res.steps} steps, wall {res.wall_time_s:.2f}s, {rt_factor:.2f}x RT)",
        f"  Distance     : s={res.final_s_m:.2f}m (max={res.max_s_m:.2f}m, backtrack={res.max_s_backtrack_m:.3f}m)",
        f"  Velocity     : min={res.min_v_mps:+.3f} m/s, max={res.max_v_mps:+.3f} m/s | max|w|={res.max_abs_w_cmd:.2f} rad/s",
        f"  Net Score    : {res.net_score_s:6.2f}s  = Lap {res.lap_time_s:.2f}s + Pen {res.total_penalty_s:.2f}s - Bonus {res.total_bonus_s:.1f}s",
        f"  Penalties    : line={res.boundary_penalty_s:.2f}s ({res.boundary_press_time_s:.2f}s), "
        f"cones={res.cone_hits}, no_entry={res.no_entry_hits}, yellow_viol={res.yellow_violations}",
        f"  Bonuses/Stop : digits={res.bonus_digits_collected} (-{res.total_bonus_s:.0f}s), "
        f"yellow_stops_ok={res.yellow_stops_ok}, cbf_steps={res.cbf_active_steps}",
    ]
    if res.snapshot_path:
        lines.append(f"  Snapshot PNG : {res.snapshot_path}")
    if res.record_path:
        lines.append(f"  Recorded MP4 : {res.record_path}")
    lines.append("=" * 76)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Windows PC Closed-Loop AIC Simulator & Dual-Pane OpenCV GUI"
    )
    ap.add_argument("--track", default="fork_s", help="Track name ('fork_s', 'roundabout') or path to track .json")
    ap.add_argument("--yaml", default=None, help="Path to ROS2 finals_*.yaml (defaults to track's official YAML)")
    ap.add_argument("--override", action="append", default=[], help="Parameter override key=val (repeatable)")
    ap.add_argument("--params-json", default=None, help="Path to JSON file or inline JSON string of param overrides")
    ap.add_argument("--detector", choices=("mock", "yolo"), default="mock", help="Digit detector backend")
    ap.add_argument("--yolo-weights", default=str(DEFAULT_YOLO_WEIGHTS), help="Path to YOLOv8 .pt weights")

    mode_grp = ap.add_mutually_exclusive_group()
    mode_grp.add_argument("--gui", action="store_true", help="Launch interactive 1280x480 dual-pane OpenCV GUI")
    mode_grp.add_argument("--headless", action="store_true", help="Run headless without GUI window (default)")

    ap.add_argument("--record", default=None, help="Save 1280x480 dual-pane MP4 video to path")
    ap.add_argument("--snapshot", default=None, help="Save 1280x480 dual-pane PNG snapshot to path")
    ap.add_argument("--snapshot-time", type=float, default=None, help="Exact sim time (s) to capture snapshot")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for dynamics, mock detector, and props")
    ap.add_argument("--max-time", type=float, default=60.0, help="Maximum simulation time (s)")
    ap.add_argument("--fps", type=float, default=30.0, help="Simulation frame rate (Hz, e.g. 15 or 30)")

    # Finals prop generation
    ap.add_argument("--cones", type=int, default=0, help="Number of random 3D red cones")
    ap.add_argument("--no-entry", type=int, default=0, help="Number of random hollow red no-entry boxes")
    ap.add_argument("--digits", type=int, default=0, help="Number of random bonus digits (1..8)")
    ap.add_argument("--random-yellow", type=int, default=0, help="Number of extra random yellow stop lines")
    ap.add_argument("--finals-props", action="store_true", help="Preset: 2 cones, 1 no-entry, 3 digits, 1 random yellow")

    # Domain randomization / physics perturbations
    ap.add_argument("--a-eff-scale", type=float, default=1.0, help="Braking deceleration multiplier")
    ap.add_argument("--steer-scale", type=float, default=1.0, help="Steering actuation multiplier")
    ap.add_argument("--speed-scale", type=float, default=1.0, help="Speed actuation multiplier")
    ap.add_argument("--latency-ms", type=float, default=0.0, help="Command latency (ms)")
    ap.add_argument("--voltage", type=float, default=12.0, help="Battery voltage (V)")
    ap.add_argument("--color-cast", default=None, help="Lighting color cast ('none','warm','cool','yellowed')")
    ap.add_argument("--glare", type=float, default=0.0, help="Directional glare strength (0..1)")
    ap.add_argument("--json-out", default=None, help="Optional path to save run summary JSON")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    overrides = load_params_json(args.params_json)
    overrides.update(parse_overrides_list(args.override))

    cones = args.cones
    no_entry = args.no_entry
    digits = args.digits
    random_yellow = args.random_yellow
    if args.finals_props:
        cones = max(cones, 2)
        no_entry = max(no_entry, 1)
        digits = max(digits, 3)
        random_yellow = max(random_yellow, 1)

    cfg = SimConfig(
        track=args.track,
        yaml_path=args.yaml,
        param_overrides=overrides,
        detector=args.detector,
        yolo_weights=args.yolo_weights,
        rate_hz=args.fps,
        max_time_s=args.max_time,
        seed=args.seed,
        cones=cones,
        no_entry=no_entry,
        digits=digits,
        random_yellow=random_yellow,
        prop_seed=args.seed,
        a_eff_scale=args.a_eff_scale,
        steer_scale=args.steer_scale,
        speed_scale=args.speed_scale,
        latency_ms=args.latency_ms,
        voltage=args.voltage,
        color_cast=args.color_cast,
        glare=args.glare,
    )

    with ClosedLoopRunner(cfg) as runner:
        if args.gui:
            res = runner.run_gui(record_path=args.record, snapshot_path=args.snapshot)
        else:
            res = runner.run_headless(
                record_path=args.record,
                snapshot_path=args.snapshot,
                snapshot_time_s=args.snapshot_time,
            )

    print(_format_summary(res), flush=True)

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in res.__dict__.items() if k != "referee_status"}
        payload["referee_status"] = {
            k: v for k, v in res.referee_status.items() if k not in ("wheel_coords_mm", "disc_centers_mm")
        }
        out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
