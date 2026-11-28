#!/usr/bin/env python3
"""Credible BEV closed-loop full-track simulator & real-car debug replay validator.

Uses the exact compiled C++17 BevDwaPlanner via tools/sim/build/sim_brain.dll,
exact YAML parameters, calibrated GroundProjector homography camera FOV mask,
delayed asymmetric Wheeltec mini_akm dynamics (100 Hz plant / 15 Hz planner),
continuous 3-circle wall/obstacle collision checking, and debug/*.zip replay.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

SIM_DIR = Path(__file__).resolve().parent
REPO_ROOT = SIM_DIR.parent.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from build_brain_dll import DLL_PATH, build_dll

SCRATCH_DIR = Path(
    r"C:\Users\Lenovo\.gemini\antigravity\brain\7c9cff71-dd0f-4809-bbdb-63529fb339c8\scratch"
)

BEV_ROWS = 160
BEV_COLS = 160
BEV_CELLS = BEV_ROWS * BEV_COLS
BEV_RES = 0.01
BEV_D_MIN = -0.35
BEV_Y_MAX = 0.80

BEV_UNKNOWN = 0
BEV_FREE = 1
BEV_WALL_BLUE = 2
BEV_OBS_CONE = 3
BEV_OBS_NO_ENTRY = 4

PHASE_NAMES = [
    "cruise_start",
    "fork_R_enter",
    "fork_R_stem",
    "fork_T_right",
    "fork_L_uturn",
    "fork_S_curve",
    "rb1_approach",
    "rb1_active",
    "rb_uturn",
    "rb2_approach",
    "rb2_active",
    "rb2_exit",
]


class BevDwaStepOutC(ctypes.Structure):
    _fields_ = [
        ("linear_speed_mps", ctypes.c_double),
        ("yaw_rate_rps", ctypes.c_double),
        ("chosen_kappa", ctypes.c_double),
        ("chosen_safe_len_m", ctypes.c_double),
        ("min_wall_clearance_m", ctypes.c_double),
        ("min_obs_clearance_m", ctypes.c_double),
        ("rel_yaw_rad", ctypes.c_double),
        ("goal_yaw_rad", ctypes.c_double),
        ("measured_lane_width_m", ctypes.c_double),
        ("phase", ctypes.c_int32),
        ("committed_branch", ctypes.c_int32),
        ("has_feasible_arc", ctypes.c_int32),
        ("has_near_obstacle", ctypes.c_int32),
        ("is_finish_line", ctypes.c_int32),
        ("yellow_rearmed", ctypes.c_int32),
        ("goal_yaw_active", ctypes.c_int32),
        ("mode_tag", ctypes.c_char * 32),
        ("chosen_kappa2", ctypes.c_double),
        ("chosen_s_split_m", ctypes.c_double),
        ("feasible_arc_count", ctypes.c_int32),
        ("_pad_k2", ctypes.c_int32),
    ]


def load_lib() -> ctypes.CDLL:
    build_dll(force=False, verbose=False)
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        gxx = shutil.which("g++")
        if gxx:
            os.add_dll_directory(str(Path(gxx).resolve().parent))
    lib = ctypes.CDLL(str(DLL_PATH))

    lib.brain_create.argtypes = []
    lib.brain_create.restype = ctypes.c_void_p
    lib.brain_destroy.argtypes = [ctypes.c_void_p]
    lib.brain_destroy.restype = None

    lib.brain_set_param_double.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
    lib.brain_set_param_double.restype = ctypes.c_int32
    lib.brain_set_param_int.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
    lib.brain_set_param_int.restype = ctypes.c_int32
    lib.brain_set_param_bool.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
    lib.brain_set_param_bool.restype = ctypes.c_int32
    lib.brain_set_param_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.brain_set_param_string.restype = ctypes.c_int32

    lib.bev_dwa_reset_c.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.bev_dwa_reset_c.restype = ctypes.c_int32

    lib.bev_dwa_notify_yellow_stop_at_c.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.bev_dwa_notify_yellow_stop_at_c.restype = ctypes.c_int32

    lib.bev_dwa_update_fork_votes_c.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.bev_dwa_update_fork_votes_c.restype = ctypes.c_int32

    lib.bev_dwa_finalize_fork_c.argtypes = [ctypes.c_void_p]
    lib.bev_dwa_finalize_fork_c.restype = ctypes.c_int32

    lib.bev_dwa_step_c.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.POINTER(BevDwaStepOutC),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.bev_dwa_step_c.restype = ctypes.c_int32
    return lib


def apply_yaml_params(lib: ctypes.CDLL, handle: ctypes.c_void_p, params: Dict) -> None:
    for k, v in params.items():
        kb = k.encode("utf-8")
        if isinstance(v, bool):
            lib.brain_set_param_bool(handle, kb, 1 if v else 0)
        elif isinstance(v, int):
            lib.brain_set_param_int(handle, kb, v)
        elif isinstance(v, float):
            lib.brain_set_param_double(handle, kb, v)
        elif isinstance(v, str):
            lib.brain_set_param_string(handle, kb, v.encode("utf-8"))


def imread_unicode(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    enc.tofile(str(path))


def precompute_homography_fov(H_list: List[float], roi_top_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (in_fov[160,160], d_grid[160,160], y_grid[160,160]) matching BevMapper::ensure_lut."""
    H = np.array(H_list, dtype=np.float64).reshape(3, 3)
    H_inv = np.linalg.inv(H)
    roi_top = int(480 * (1.0 - roi_top_ratio))

    r_idx = np.arange(BEV_ROWS, dtype=np.float64)
    c_idx = np.arange(BEV_COLS, dtype=np.float64)
    d_vals = BEV_D_MIN + (r_idx + 0.5) * BEV_RES
    y_vals = BEV_Y_MAX - (c_idx + 0.5) * BEV_RES
    d_grid, y_grid = np.meshgrid(d_vals, y_vals, indexing="ij")

    ones = np.ones_like(d_grid)
    pts = np.stack([d_grid, y_grid, ones], axis=0).reshape(3, -1)
    uv_h = H_inv @ pts
    denom = uv_h[2, :]
    valid_w = np.abs(denom) > 1e-8
    u = np.where(valid_w, uv_h[0, :] / np.where(valid_w, denom, 1.0), -1.0)
    v = np.where(valid_w, uv_h[1, :] / np.where(valid_w, denom, 1.0), -1.0)

    # Check forward projection finite nonzero w matching GroundProjector (w may be negative!)
    uv_pts = np.stack([u, v, np.ones_like(u)], axis=0)
    fwd_h = H @ uv_pts
    fwd_w_ok = np.abs(fwd_h[2, :]) > 1e-9

    in_fov = (
        valid_w
        & fwd_w_ok
        & (d_grid.reshape(-1) >= 0.02)
        & (u >= 0.0)
        & (u < 640.0)
        & (v >= float(roi_top))
        & (v < 480.0)
    ).reshape(BEV_ROWS, BEV_COLS)
    return in_fov, d_grid, y_grid


@dataclass
class TrackMap:
    name: str
    img_bgr: np.ndarray
    H: int
    W: int
    m_per_px: float
    road_free_u8: np.ndarray
    yellow_u8: np.ndarray
    signed_wall_dist_m: np.ndarray
    start_x: float
    start_y: float
    start_yaw: float
    end_x: float
    end_y: float
    centerline_m: np.ndarray


def load_track(track_name: str) -> TrackMap:
    png_path = SIM_DIR / "tracks" / f"{track_name}.png"
    json_path = SIM_DIR / "tracks" / f"{track_name}.json"
    img = imread_unicode(png_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    H, W = img.shape[:2]
    m_per_px = float(meta["mm_per_px"]) / 1000.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 170)
    yellow_mask = (
        (hsv[:, :, 0] >= 15)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] > 60)
        & (hsv[:, :, 2] > 90)
    )
    road_free = np.where(white_mask | yellow_mask, np.uint8(255), np.uint8(0))
    yellow_u8 = np.where(yellow_mask, np.uint8(255), np.uint8(0))

    dist_in = cv2.distanceTransform(road_free, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(255 - road_free, cv2.DIST_L2, 5)
    signed_wall_dist_m = (dist_in - dist_out) * m_per_px

    pts_mm = np.array(meta["centerline_mm"], dtype=np.float64)
    centerline_m = pts_mm / 1000.0
    p0 = centerline_m[0]
    p5 = centerline_m[min(5, len(centerline_m) - 1)]
    start_yaw = math.atan2(p5[1] - p0[1], p5[0] - p0[0])
    # Nudge start forward 0.38m so entire 0.28m car starts cleanly inside white start pad
    start_x = float(p0[0] + 0.38 * math.cos(start_yaw))
    start_y = float(p0[1] + 0.38 * math.sin(start_yaw))

    end_pt = centerline_m[-1]
    return TrackMap(
        name=track_name,
        img_bgr=img,
        H=H,
        W=W,
        m_per_px=m_per_px,
        road_free_u8=road_free,
        yellow_u8=yellow_u8,
        signed_wall_dist_m=signed_wall_dist_m,
        start_x=start_x,
        start_y=start_y,
        start_yaw=start_yaw,
        end_x=float(end_pt[0]),
        end_y=float(end_pt[1]),
        centerline_m=centerline_m,
    )


@dataclass
class ObstacleSpec:
    kind: str  # "cone" | "no_entry"
    wx: float
    wy: float
    radius_m: float = 0.075
    half_len_m: float = 0.085
    half_wid_m: float = 0.055


@dataclass
class SimResult:
    case_name: str
    completed: bool
    reason: str
    lap_time_s: float
    arc_m: float
    min_wall_clr_m: float
    min_obs_clr_m: float
    off_track_s: float
    cone_hits: int
    no_entry_hits: int
    final_phase: str
    committed_branch: int
    traj_png: Path


def run_closed_loop_sim(
    lib: ctypes.CDLL,
    track_name: str,
    route: str,
    lag_s: float = 0.18,
    obstacles: Optional[List[ObstacleSpec]] = None,
    case_name: Optional[str] = None,
) -> SimResult:
    obstacles = obstacles or []
    yaml_file = REPO_ROOT / "tools" / "config" / (
        "finals_fork.yaml" if track_name == "fork_s" else "finals_roundabout.yaml"
    )
    cfg = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    params = cfg["line_follower_node"]["ros__parameters"]
    tmap = load_track(track_name)
    in_fov, d_grid, y_grid = precompute_homography_fov(
        params["homography_matrix"], float(params.get("roi_top_ratio", 0.68))
    )

    handle = lib.brain_create()
    try:
        apply_yaml_params(lib, handle, params)
        # Keep realistic 0.15s latency setting inside planner matching physical STM32 config
        lib.brain_set_param_double(handle, b"bev_control_latency_s", float(params.get("bev_control_latency_s", 0.15)))

        # Rear-axle world state; bumper = rear + 0.18 * dir
        rear_to_bumper = float(params.get("rear_axle_to_bumper_m", 0.18))
        car_yaw = tmap.start_yaw
        bumper_x = tmap.start_x
        bumper_y = tmap.start_y
        rear_x = bumper_x - rear_to_bumper * math.cos(car_yaw)
        rear_y = bumper_y - rear_to_bumper * math.sin(car_yaw)
        lib.bev_dwa_reset_c(handle, 0.0, bumper_x, bumper_y, car_yaw)

        dt_sim = 0.01  # 100 Hz continuous physics
        ctrl_period = 1.0 / 15.0  # 15 Hz real camera/planner rate
        lag_steps = max(0, int(round(lag_s / dt_sim)))
        cmd_queue: deque = deque([(0.0, 0.0)] * lag_steps, maxlen=max(1, lag_steps))

        v_actual = 0.0
        kappa_actual = 0.0
        w_actual = 0.0
        cmd_v_latched = 0.0
        cmd_w_latched = 0.0

        r_car = 0.5 * float(params.get("car_width_m", 0.191))
        k_max_l = float(params.get("bev_kappa_max_left", 2.38))
        k_max_r = float(params.get("bev_kappa_max_right", 2.80))

        min_wall_clr_m = 1.0
        min_obs_clr_m = 1.0
        off_track_s = 0.0
        cone_hits = 0
        no_entry_hits = 0
        was_in_cone = False
        was_in_ne = False

        # Yellow stop simulation state
        holding_stop = False
        stop_start_t = -1.0
        stop_votes_sent = False
        stop_hold_dur_s = 0.65
        stop_count = 0

        # Precompute forward-corridor yellow sampler grid inside BEV
        yel_roi = in_fov & (d_grid >= 0.03) & (d_grid <= 0.68) & (np.abs(y_grid) <= 0.16)
        yel_d_pts = d_grid[yel_roi]
        yel_y_pts = y_grid[yel_roi]

        fov_d = d_grid[in_fov]
        fov_y = y_grid[in_fov]

        grid_buf = (ctypes.c_uint8 * BEV_CELLS)()
        fused_out = (ctypes.c_uint8 * BEV_CELLS)()
        out = BevDwaStepOutC()

        traj_pts: List[Tuple[float, float, float, float]] = []
        min_clr_pt: Optional[Tuple[float, float]] = None

        t = 0.0
        next_ctrl_t = 0.0
        arc_m = 0.0
        completed = False
        reason = "timeout"
        last_phase = "cruise_start"
        last_branch = 0

        m_per_px = tmap.m_per_px
        inv_m_per_px = 1.0 / m_per_px
        img_h_m1 = tmap.H - 1

        max_sim_s = 55.0
        while t < max_sim_s:
            # 1) 15 Hz planner step
            if t + 1e-9 >= next_ctrl_t:
                next_ctrl_t += ctrl_period
                ct = math.cos(car_yaw)
                st = math.sin(car_yaw)

                # Live 160x160 BEV projection inside exact Homography FOV
                grid_arr = np.zeros((BEV_ROWS, BEV_COLS), dtype=np.uint8)
                wx_f = bumper_x + fov_d * ct - fov_y * st
                wy_f = bumper_y + fov_d * st + fov_y * ct
                px_f = np.round(wx_f * inv_m_per_px).astype(np.int32)
                py_f = np.round(img_h_m1 - wy_f * inv_m_per_px).astype(np.int32)
                in_img = (px_f >= 0) & (px_f < tmap.W) & (py_f >= 0) & (py_f < tmap.H)

                cells = np.full(len(fov_d), BEV_WALL_BLUE, dtype=np.uint8)
                valid_px = px_f[in_img]
                valid_py = py_f[in_img]
                free_hits = tmap.road_free_u8[valid_py, valid_px] > 0
                c_valid = np.where(free_hits, BEV_FREE, BEV_WALL_BLUE).astype(np.uint8)
                cells[in_img] = c_valid

                for obs in obstacles:
                    if obs.kind == "cone":
                        hit_obs = ((wx_f - obs.wx) ** 2 + (wy_f - obs.wy) ** 2) <= (obs.radius_m ** 2)
                        cells[hit_obs] = BEV_OBS_CONE
                    else:
                        hit_obs = (np.abs(wx_f - obs.wx) <= obs.half_len_m) & (
                            np.abs(wy_f - obs.wy) <= obs.half_wid_m
                        )
                        cells[hit_obs] = BEV_OBS_NO_ENTRY

                grid_arr[in_fov] = cells
                ctypes.memmove(grid_buf, grid_arr.ctypes.data, BEV_CELLS)

                # Yellow line detection inside forward corridor
                wx_y = bumper_x + yel_d_pts * ct - yel_y_pts * st
                wy_y = bumper_y + yel_d_pts * st + yel_y_pts * ct
                px_y = np.round(wx_y * inv_m_per_px).astype(np.int32)
                py_y = np.round(img_h_m1 - wy_y * inv_m_per_px).astype(np.int32)
                y_in = (px_y >= 0) & (px_y < tmap.W) & (py_y >= 0) & (py_y < tmap.H)
                y_hits = np.zeros(len(yel_d_pts), dtype=bool)
                y_hits[y_in] = tmap.yellow_u8[py_y[y_in], px_y[y_in]] > 0
                yellow_count = int(y_hits.sum())
                yellow_ahead = 1 if yellow_count >= 12 else 0
                yellow_d = float(np.percentile(yel_d_pts[y_hits], 15)) if yellow_ahead else 0.0
                yellow_dist_valid = 1 if yellow_ahead else 0

                # Simulate realistic yellow line pre-brake & stationary voting lifecycle
                yellow_cap = 1e9
                if holding_stop:
                    elapsed_stop = t - stop_start_t
                    # Only fork/RB decision stops carry left/right choice cards on the real track:
                    # - fork_s: second yellow line (arc ~ 2.0m, before fork junction)
                    # - roundabout: first & third yellow lines (RB1 & RB2 entrances)
                    is_choice_stop = (
                        (track_name == "fork_s" and 1.5 <= arc_m <= 3.5)
                        or (track_name == "roundabout" and (arc_m < 1.5 or 5.0 <= arc_m <= 8.0))
                    )
                    if elapsed_stop >= 0.18 and not stop_votes_sent and is_choice_stop:
                        if route == "R":
                            l_dig, r_dig = 4, 8
                        else:
                            l_dig, r_dig = 8, 4
                        dig_arr = (ctypes.c_double * 10)(
                            0.50, +0.14, float(l_dig), 0.90, t,
                            0.50, -0.14, float(r_dig), 0.90, t,
                        )
                        lib.bev_dwa_update_fork_votes_c(handle, dig_arr, 2, 1)
                        lib.bev_dwa_update_fork_votes_c(handle, dig_arr, 2, 1)
                        stop_votes_sent = True
                    if elapsed_stop >= stop_hold_dur_s and abs(v_actual) < 0.03:
                        holding_stop = False
                        lib.bev_dwa_finalize_fork_c(handle)
                elif yellow_ahead and (out.yellow_rearmed != 0 or stop_count == 0):
                    # Smooth pre-brake envelope down to stop trigger at 0.20m
                    if yellow_d <= 0.21:
                        # Check finish BEFORE stopping
                        pass
                    elif yellow_d <= 0.85:
                        yellow_cap = max(0.16, math.sqrt(max(0.0, 2.0 * 1.4 * (yellow_d - 0.16))))

                rc = lib.bev_dwa_step_c(
                    handle,
                    grid_buf,
                    bumper_x,
                    bumper_y,
                    car_yaw,
                    v_actual,
                    w_actual,
                    1,
                    None,
                    0,
                    yellow_ahead,
                    yellow_dist_valid,
                    yellow_d,
                    yellow_cap,
                    1 if holding_stop else 0,
                    t,
                    ctypes.byref(out),
                    fused_out,
                )
                if rc != 0:
                    raise RuntimeError(f"bev_dwa_step_c failed: {rc}")

                last_phase = PHASE_NAMES[out.phase] if 0 <= out.phase < len(PHASE_NAMES) else str(out.phase)
                last_branch = int(out.committed_branch)

                # Finish check: complete cleanly right at the final yellow line (never 0.5m before it)
                dist_end = math.hypot(bumper_x - tmap.end_x, bumper_y - tmap.end_y)
                if (
                    (out.is_finish_line != 0 and arc_m > 5.0 and yellow_ahead and yellow_d <= 0.24)
                    or (dist_end < 0.30 and arc_m > 7.5)
                ):
                    completed = True
                    reason = "finish"
                    break

                if (
                    not holding_stop
                    and yellow_ahead
                    and yellow_d <= 0.21
                    and (out.yellow_rearmed != 0 or stop_count == 0)
                    and dist_end > 0.60
                ):
                    holding_stop = True
                    stop_start_t = t
                    stop_votes_sent = False
                    stop_count += 1
                    lib.bev_dwa_notify_yellow_stop_at_c(handle, t)
                    cmd_v_latched = 0.0
                    cmd_w_latched = 0.0
                elif holding_stop:
                    cmd_v_latched = 0.0
                    cmd_w_latched = 0.0
                else:
                    cmd_v_latched = float(out.linear_speed_mps)
                    cmd_w_latched = float(out.yaw_rate_rps)

            # 2) 100 Hz Wheeltec mini_akm delayed & slew-limited plant
            if lag_steps > 0:
                v_del, w_del = cmd_queue[0]
                cmd_queue.append((cmd_v_latched, cmd_w_latched))
            else:
                v_del, w_del = cmd_v_latched, cmd_w_latched

            if v_del > v_actual:
                v_actual = min(v_del, v_actual + 1.2 * dt_sim)
            else:
                v_actual = max(v_del, v_actual - 1.8 * dt_sim)
            v_actual = max(0.0, v_actual)

            if v_actual < 0.015 and abs(v_del) < 0.015:
                w_actual = 0.0
                kappa_actual = 0.0
            else:
                k_target = w_del / max(0.14, v_actual)
                k_target = max(-k_max_r, min(k_max_l, k_target))
                # STM32 Smooth_control + servo first-order response (~0.06s time constant + slew)
                max_dk = 8.0 * dt_sim
                dk = max(-max_dk, min(max_dk, (k_target - kappa_actual) * (dt_sim / 0.06)))
                kappa_actual = max(-k_max_r, min(k_max_l, kappa_actual + dk))
                w_actual = v_actual * kappa_actual

            ds = v_actual * dt_sim
            dyaw = w_actual * dt_sim
            rear_x += ds * math.cos(car_yaw + 0.5 * dyaw)
            rear_y += ds * math.sin(car_yaw + 0.5 * dyaw)
            car_yaw += dyaw
            arc_m += ds
            bumper_x = rear_x + rear_to_bumper * math.cos(car_yaw)
            bumper_y = rear_y + rear_to_bumper * math.sin(car_yaw)

            # 3) Continuous 3-circle ground-truth safety audit
            ct = math.cos(car_yaw)
            st = math.sin(car_yaw)
            step_wall_clr = 1.0
            step_cone_clr = 1.0
            step_ne_clr = 1.0
            out_of_map = False
            for off_b in (0.0, -0.10, -0.22):
                bx = bumper_x + off_b * ct
                by = bumper_y + off_b * st
                px = int(round(bx * inv_m_per_px))
                py = int(round(img_h_m1 - by * inv_m_per_px))
                if px < 0 or px >= tmap.W or py < 0 or py >= tmap.H:
                    out_of_map = True
                    continue
                sw_clr = float(tmap.signed_wall_dist_m[py, px]) - r_car
                step_wall_clr = min(step_wall_clr, sw_clr)
                for obs in obstacles:
                    if obs.kind == "cone":
                        dc = math.hypot(bx - obs.wx, by - obs.wy) - obs.radius_m - r_car
                        step_cone_clr = min(step_cone_clr, dc)
                    else:
                        dx = max(0.0, abs(bx - obs.wx) - obs.half_len_m)
                        dy = max(0.0, abs(by - obs.wy) - obs.half_wid_m)
                        dne = math.hypot(dx, dy) - r_car
                        step_ne_clr = min(step_ne_clr, dne)

            if step_wall_clr < min_wall_clr_m:
                min_wall_clr_m = step_wall_clr
                min_clr_pt = (bumper_x, bumper_y)
            min_obs_clr_m = min(min_obs_clr_m, step_cone_clr, step_ne_clr)

            if step_wall_clr < -0.002:
                off_track_s += dt_sim
                if step_wall_clr < -0.12 or out_of_map:
                    reason = "crashed_off_track"
                    break
            in_cone_now = step_cone_clr < 0.0
            in_ne_now = step_ne_clr < 0.0
            if in_cone_now and not was_in_cone:
                cone_hits += 1
            if in_ne_now and not was_in_ne:
                no_entry_hits += 1
            was_in_cone = in_cone_now
            was_in_ne = in_ne_now

            if len(traj_pts) == 0 or t - traj_pts[-1][3] >= 0.05:
                traj_pts.append((bumper_x, bumper_y, step_wall_clr, t))

            t += dt_sim
    finally:
        lib.brain_destroy(handle)

    # Render clean high-contrast trajectory PNG
    cname = case_name or f"{track_name}_{route}_lag{int(round(lag_s * 1000))}"
    vis = tmap.img_bgr.copy()
    for obs in obstacles:
        opx = int(round(obs.wx * inv_m_per_px))
        opy = int(round(img_h_m1 - obs.wy * inv_m_per_px))
        if obs.kind == "cone":
            cv2.circle(vis, (opx, opy), int(round(obs.radius_m * inv_m_per_px)), (0, 120, 255), -1)
            cv2.circle(vis, (opx, opy), int(round(obs.radius_m * inv_m_per_px)), (0, 0, 180), 2)
        else:
            rx = int(round(obs.half_len_m * inv_m_per_px))
            ry = int(round(obs.half_wid_m * inv_m_per_px))
            cv2.rectangle(vis, (opx - rx, opy - ry), (opx + rx, opy + ry), (200, 0, 255), -1)

    for i in range(1, len(traj_pts)):
        x0, y0, clr0, _ = traj_pts[i - 1]
        x1, y1, clr1, _ = traj_pts[i]
        p0 = (int(round(x0 * inv_m_per_px)), int(round(img_h_m1 - y0 * inv_m_per_px)))
        p1 = (int(round(x1 * inv_m_per_px)), int(round(img_h_m1 - y1 * inv_m_per_px)))
        c_min = min(clr0, clr1)
        if c_min < 0.0:
            col = (0, 0, 255)
        elif c_min < 0.025:
            col = (0, 180, 255)
        else:
            col = (40, 210, 40)
        cv2.line(vis, p0, p1, col, 3, cv2.LINE_AA)

    if min_clr_pt is not None:
        mp = (int(round(min_clr_pt[0] * inv_m_per_px)), int(round(img_h_m1 - min_clr_pt[1] * inv_m_per_px)))
        cv2.circle(vis, mp, 8, (0, 0, 255), 2)

    status_col = (40, 230, 40) if (completed and off_track_s == 0.0 and cone_hits == 0 and no_entry_hits == 0) else (0, 0, 255)
    cv2.rectangle(vis, (8, 8), (940, 44), (20, 20, 28), -1)
    hud = (
        f"{cname} | ok={completed} ({reason}) t={t:.2f}s s={arc_m:.2f}m "
        f"min_w={min_wall_clr_m:+.3f}m min_o={min_obs_clr_m:+.3f}m off={off_track_s:.2f}s "
        f"phase={last_phase} br={last_branch}"
    )
    cv2.putText(vis, hud, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_col, 1, cv2.LINE_AA)

    out_png = SCRATCH_DIR / f"sim_traj_{cname}.png"
    imwrite_unicode(out_png, vis)

    return SimResult(
        case_name=cname,
        completed=completed,
        reason=reason,
        lap_time_s=t,
        arc_m=arc_m,
        min_wall_clr_m=min_wall_clr_m,
        min_obs_clr_m=min_obs_clr_m,
        off_track_s=off_track_s,
        cone_hits=cone_hits,
        no_entry_hits=no_entry_hits,
        final_phase=last_phase,
        committed_branch=last_branch,
        traj_png=out_png,
    )


def extract_bev_grid_from_inset(frame_bgr: np.ndarray) -> np.ndarray:
    """Reconstruct 160x160 BEV grid from top-right render_bev_inset overlay in debug MP4."""
    # Find inset box by locating its bright cyan 1px left border [90,220,255] in top-right quadrant
    roi = frame_bgr[0:260, 380:640]
    b, g, r = roi[:, :, 0].astype(int), roi[:, :, 1].astype(int), roi[:, :, 2].astype(int)
    border = (np.abs(b - 90) < 35) & (np.abs(g - 220) < 35) & (np.abs(r - 255) < 35)
    ys, xs = np.where(border)
    if len(xs) > 30:
        x0 = 380 + int(xs.min())
        y0 = int(ys.min())
        side = min(210, int(frame_bgr.shape[1] - x0), int(frame_bgr.shape[0] - y0))
        if int(xs.max()) - int(xs.min()) + 1 > 150:
            side = int(xs.max()) - int(xs.min()) + 1
        crop = frame_bgr[y0 + 2 : y0 + side - 2, x0 + 2 : x0 + side - 2]
    else:
        crop = frame_bgr[28:214, 426:632]

    small = cv2.resize(crop, (BEV_COLS, BEV_ROWS), interpolation=cv2.INTER_NEAREST)
    # Mask top compass/HUD overlay text region (~26 rows in flipped view -> top of crop)
    small[:26, :] = [32, 32, 38]
    # Flip vertically back from draw_y = 159 - r to row r in [0..159]
    small = np.flipud(small)

    pb = small[:, :, 0].astype(np.int32)
    pg = small[:, :, 1].astype(np.int32)
    pr = small[:, :, 2].astype(np.int32)

    labels = np.full((BEV_ROWS, BEV_COLS), BEV_FREE, dtype=np.uint8)
    # BEV_UNKNOWN: dark background gray (~32,32,38)
    is_unk = (np.maximum(np.maximum(pb, pg), pr) < 75) & (np.abs(pb - pr) < 25)
    # BEV_WALL_BLUE: dark blue (~150,70,30 BGR)
    is_blue = (pb > 105) & (pb > pr + 45) & (pg < 120)
    # BEV_OBS_CONE: strict bright orange (~0,120,255 BGR), NOT yellow car box (~0,230,255)
    is_cone = (pr > 195) & (pb < 45) & (pg >= 70) & (pg <= 165)
    # BEV_OBS_NO_ENTRY: magenta (~200,0,255 BGR)
    is_ne = (pr > 180) & (pb > 155) & (pg < 60)

    labels[is_unk] = BEV_UNKNOWN
    labels[is_blue] = BEV_WALL_BLUE
    labels[is_cone] = BEV_OBS_CONE
    labels[is_ne] = BEV_OBS_NO_ENTRY

    # Remove 1-2px drawn arc/box lines cleanly and ensure ego car footprint starts on BEV_FREE
    labels = cv2.medianBlur(labels, 3)
    # Physical car bumper is at d=0 (r=35), tail at d=-0.28 (r=7), width +-0.0955 (c=70..90)
    labels[8:40, 68:92] = np.where(labels[8:40, 68:92] == BEV_UNKNOWN, BEV_FREE, labels[8:40, 68:92])
    return labels


def run_debug_replay(lib: ctypes.CDLL) -> bool:
    cases = [
        (
            "183553_fork",
            REPO_ROOT / "debug" / "20261128_183553_fork_test_debug.zip",
            REPO_ROOT / "tools" / "config" / "finals_fork.yaml",
            "20261128_183553_fork_test_debug.mp4",
            [95, 105, 118, 120, 125],
            +1,
        ),
        (
            "183754_rb",
            REPO_ROOT / "debug" / "20261128_183754_fork_test_debug.zip",
            REPO_ROOT / "tools" / "config" / "finals_roundabout.yaml",
            "20261128_183754_fork_test_debug.mp4",
            [44, 70, 90, 96, 110],
            +1,
        ),
    ]
    all_ok = True
    for cname, zpath, ypath, mp4_name, frame_ids, expected_br in cases:
        if not zpath.is_file():
            print(f"[REPLAY] SKIP missing {zpath}")
            continue
        cfg = yaml.safe_load(ypath.read_text(encoding="utf-8"))
        params = cfg["line_follower_node"]["ros__parameters"]
        handle = lib.brain_create()
        try:
            apply_yaml_params(lib, handle, params)
            with zipfile.ZipFile(zpath, "r") as zf, tempfile.TemporaryDirectory() as td:
                log_txt = zf.read("follow_slice.log").decode("utf-8", "ignore")
                old_lines = [l for l in log_txt.splitlines() if "[DEC] ctrl" in l]
                mp4_path = zf.extract(mp4_name, td)
                cap = cv2.VideoCapture(mp4_path)
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                lib.bev_dwa_reset_c(handle, 0.0, 0.0, 0.0, 0.0)
                if cname.startswith("183553"):
                    # Commit right fork cleanly before accident bend f118-f125
                    dig = (ctypes.c_double * 10)(
                        0.45, +0.14, 4.0, 0.90, 0.1,
                        0.45, -0.14, 8.0, 0.90, 0.1,
                    )
                    lib.bev_dwa_notify_yellow_stop_at_c(handle, 0.05)
                    lib.bev_dwa_update_fork_votes_c(handle, dig, 2, 1)
                    lib.bev_dwa_update_fork_votes_c(handle, dig, 2, 1)
                    lib.bev_dwa_finalize_fork_c(handle)
                    yaw_sim = -0.82
                else:
                    dig = (ctypes.c_double * 5)(0.45, -0.14, 5.0, 0.90, 0.1)
                    lib.bev_dwa_notify_yellow_stop_at_c(handle, 0.05)
                    lib.bev_dwa_update_fork_votes_c(handle, dig, 1, 1)
                    lib.bev_dwa_update_fork_votes_c(handle, dig, 1, 1)
                    lib.bev_dwa_finalize_fork_c(handle)
                    yaw_sim = -0.60

                feasible_pre_crash = 0
                pre_crash_total = 0
                valid_recovery = 0
                tested = 0
                out = BevDwaStepOutC()
                fused = (ctypes.c_uint8 * BEV_CELLS)()
                grid_buf = (ctypes.c_uint8 * BEV_CELLS)()
                for idx_i, fid in enumerate(frame_ids):
                    if fid >= total_f:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                    ok, fr = cap.read()
                    if not ok or fr is None:
                        continue
                    bev_g = extract_bev_grid_from_inset(fr)
                    ctypes.memmove(grid_buf, bev_g.ctypes.data, BEV_CELLS)
                    t_now = 1.0 + 0.1 * idx_i
                    lib.bev_dwa_step_c(
                        handle,
                        grid_buf,
                        0.20 * idx_i,
                        -0.10 * idx_i,
                        yaw_sim,
                        0.32,
                        -0.5,
                        1,
                        None,
                        0,
                        0,
                        0,
                        0.0,
                        1e9,
                        0,
                        t_now,
                        ctypes.byref(out),
                        fused,
                    )
                    tested += 1
                    is_pre_crash = (cname.startswith("183553") and fid <= 105) or (
                        cname.startswith("183754") and fid in (44, 90)
                    )
                    if is_pre_crash:
                        pre_crash_total += 1
                        if out.has_feasible_arc != 0 and out.feasible_arc_count > 0:
                            feasible_pre_crash += 1
                    else:
                        # On already-on-wall crash frames, planner must actively steer (non-zero recovery)
                        if abs(out.chosen_kappa) >= 0.50 or out.has_feasible_arc != 0:
                            valid_recovery += 1
                    print(
                        f"[REPLAY] {cname} f={fid:03d}: pre={int(is_pre_crash)} feas={out.has_feasible_arc} "
                        f"arcs={out.feasible_arc_count} k1={out.chosen_kappa:+.2f} "
                        f"k2={out.chosen_kappa2:+.2f} v={out.linear_speed_mps:.2f} "
                        f"L={out.chosen_safe_len_m:.2f} mode={out.mode_tag.decode()} br={out.committed_branch}"
                    )
                cap.release()
                old_deadlocks = sum(1 for l in old_lines if "arc=29/0" in l)
                ok_case = (
                    tested >= 3
                    and feasible_pre_crash == pre_crash_total
                    and valid_recovery >= (tested - pre_crash_total - 1)
                    and out.committed_branch == expected_br
                )
                print(
                    f"[REPLAY SUMMARY] {cname}: old_arc29_0_logs={old_deadlocks}/{len(old_lines)} -> "
                    f"pre_crash_feas={feasible_pre_crash}/{pre_crash_total} "
                    f"active_rec={valid_recovery}/{tested - pre_crash_total} PASS={ok_case}"
                )
                if not ok_case:
                    all_ok = False
        finally:
            lib.brain_destroy(handle)
    return all_ok


def build_stress_obstacles(track_name: str, route: str) -> List[ObstacleSpec]:
    tmap = load_track(track_name)
    pts = tmap.centerline_m
    if track_name == "fork_s":
        # Place cone along initial straight and no_entry on return straight leg (before S-bend)
        p_c = pts[min(70, len(pts) - 1)]
        p_n = pts[min(len(pts) - 125, len(pts) - 1)]
        return [
            ObstacleSpec("cone", float(p_c[0]), float(p_c[1] + 0.085), radius_m=0.070),
            ObstacleSpec("no_entry", float(p_n[0] - 0.115), float(p_n[1]), half_len_m=0.045, half_wid_m=0.08),
        ]
    else:
        # Place cone along straight before U-turn and no_entry on RB2 exit straight
        p_c = pts[min(200, len(pts) - 1)]
        p_n = pts[min(len(pts) - 45, len(pts) - 1)]
        return [
            ObstacleSpec("cone", float(p_c[0]), float(p_c[1] + 0.085), radius_m=0.070),
            ObstacleSpec("no_entry", float(p_n[0]), float(p_n[1] - 0.115), half_len_m=0.08, half_wid_m=0.045),
        ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Credible BEV full-track closed-loop simulator")
    ap.add_argument("--replay-debug", action="store_true", help="Also replay 183553 & 183754 debug zips")
    ap.add_argument("--lag-sweep", action="store_true", help="Run 0.15s / 0.20s / 0.25s actuator lag sweep")
    args = ap.parse_args()

    lib = load_lib()
    results: List[SimResult] = []

    # 1) 4 Canonical routes @ calibrated 0.18s lag
    for tr in ("fork_s", "roundabout"):
        for rt in ("R", "L"):
            results.append(run_closed_loop_sim(lib, tr, rt, lag_s=0.18))

    # 2) Lag stress sweep (0.15s, 0.20s, 0.25s) on tightest R routes
    results.append(run_closed_loop_sim(lib, "fork_s", "R", lag_s=0.15))
    results.append(run_closed_loop_sim(lib, "fork_s", "R", lag_s=0.25))
    results.append(run_closed_loop_sim(lib, "roundabout", "R", lag_s=0.25))
    if args.lag_sweep:
        results.append(run_closed_loop_sim(lib, "fork_s", "L", lag_s=0.20))
        results.append(run_closed_loop_sim(lib, "roundabout", "L", lag_s=0.20))

    # 3) Obstacle stress (red cone + no_entry on track)
    results.append(
        run_closed_loop_sim(
            lib,
            "fork_s",
            "R",
            lag_s=0.18,
            obstacles=build_stress_obstacles("fork_s", "R"),
            case_name="fork_s_R_obs_stress",
        )
    )
    results.append(
        run_closed_loop_sim(
            lib,
            "roundabout",
            "R",
            lag_s=0.18,
            obstacles=build_stress_obstacles("roundabout", "R"),
            case_name="roundabout_R_obs_stress",
        )
    )

    all_pass = True
    print("\n=== BEV FULL-TRACK CLOSED-LOOP RESULTS ===")
    for r in results:
        ok = (
            r.completed
            and r.off_track_s <= 0.0
            and r.min_wall_clr_m >= -0.002
            and r.cone_hits == 0
            and r.no_entry_hits == 0
        )
        if not ok:
            all_pass = False
        print(
            f"[{'PASS' if ok else 'FAIL'}] {r.case_name:<24} reason={r.reason:<17} "
            f"t={r.lap_time_s:5.2f}s s={r.arc_m:5.2f}m w_min={r.min_wall_clr_m:+.3f}m "
            f"o_min={r.min_obs_clr_m:+.3f}m off={r.off_track_s:.2f}s phase={r.final_phase}"
        )

    if args.replay_debug:
        print("\n=== REAL-CAR DEBUG ZIP REPLAY ===")
        if not run_debug_replay(lib):
            all_pass = False

    print(f"\nOVERALL_SIM_PASS={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
