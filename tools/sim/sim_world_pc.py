#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-fidelity Windows PC simulator world: 3D finals prop renderer, real-measured
Ackermann vehicle dynamics (with reverse-brake & latency), official AIC Finals
auto-referee, and top-down visualizer.

Owned components:
  - FinalsTrackRenderer (subclasses render.TrackRenderer)
  - VehicleDynamics
  - AICReferee
  - StepInfo
  - PCSimWorld
"""
from __future__ import annotations

import copy
import importlib.util
import math
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw as PILImageDraw
    from PIL import ImageFont as PILImageFont
except ImportError:  # pragma: no cover
    PILImage = None
    PILImageDraw = None
    PILImageFont = None


def _load_render_module():
    """Load tools/sim/render.py without polluting sys.path[0] (avoids shadowing stdlib profile)."""
    if "tools.sim.render" in sys.modules:
        return sys.modules["tools.sim.render"]
    if "render" in sys.modules and hasattr(sys.modules["render"], "TrackRenderer"):
        return sys.modules["render"]
    try:
        from tools.sim import render as mod
        return mod
    except ImportError:
        render_path = Path(__file__).resolve().parent / "render.py"
        spec = importlib.util.spec_from_file_location("_aic_sim_render", str(render_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load render.py from {render_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_render_mod = _load_render_module()
BG_BGR: Tuple[int, int, int] = _render_mod.BG_BGR
WHITE_BGR: Tuple[int, int, int] = _render_mod.WHITE_BGR
YELLOW_BGR: Tuple[int, int, int] = _render_mod.YELLOW_BGR
RED_BGR: Tuple[int, int, int] = _render_mod.RED_BGR
CameraModel = _render_mod.CameraModel
TrackRenderer = _render_mod.TrackRenderer
parse_color_cast = _render_mod.parse_color_cast
point_to_polyline_distance_mm = _render_mod.point_to_polyline_distance_mm


# ---------------------------------------------------------------------------
# Constants & Geometry Helpers
# ---------------------------------------------------------------------------

TRACKS_DIR = Path(__file__).resolve().parent / "tracks"

PROP_TYPE_ALIASES: Dict[str, str] = {
    "red_cone": "red_cone",
    "cone": "red_cone",
    "cones": "red_cone",
    "red_no_entry": "red_no_entry",
    "no_entry": "red_no_entry",
    "noentry": "red_no_entry",
    "digit": "digit",
    "bonus_digit": "digit",
    "bonus": "digit",
    "yellow_stop_line": "yellow_stop_line",
    "yellow": "yellow_stop_line",
    "yellow_line": "yellow_stop_line",
    "random_yellow": "yellow_stop_line",
}

BOLD_FONT_CANDIDATES: Tuple[str, ...] = (
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\verdanab.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)
_BOLD_FONTS_CACHE: Optional[List[str]] = None


def _get_bold_fonts() -> List[str]:
    global _BOLD_FONTS_CACHE
    if _BOLD_FONTS_CACHE is None:
        _BOLD_FONTS_CACHE = [p for p in BOLD_FONT_CANDIDATES if os.path.isfile(p)]
    return _BOLD_FONTS_CACHE


def resolve_track_json(track: Union[str, Path]) -> str:
    """Resolve 'fork_s' / 'roundabout' or a filesystem path to a track .json path."""
    p = Path(track)
    if p.is_file():
        return str(p)
    name = str(track).strip()
    if not name.endswith(".json"):
        name = name + ".json"
    cand = TRACKS_DIR / name
    if cand.is_file():
        return str(cand)
    raise FileNotFoundError(f"Track JSON not found: {track} (checked {p} and {cand})")


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _obb_corners_2d(
    cx: float, cy: float, half_len: float, half_wid: float, yaw_rad: float
) -> np.ndarray:
    """Return 4 corners (4, 2) of an OBB in world coords: [FL, FR, RR, RL]."""
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    # local (+hl, +hw) -> world
    pts = np.array(
        [
            [cx + half_len * c - half_wid * s, cy + half_len * s + half_wid * c],
            [cx + half_len * c + half_wid * s, cy + half_len * s - half_wid * c],
            [cx - half_len * c + half_wid * s, cy - half_len * s - half_wid * c],
            [cx - half_len * c - half_wid * s, cy - half_len * s + half_wid * c],
        ],
        dtype=np.float64,
    )
    return pts


def obb_overlap_2d(
    cx1: float,
    cy1: float,
    hl1: float,
    hw1: float,
    yaw1: float,
    cx2: float,
    cy2: float,
    hl2: float,
    hw2: float,
    yaw2: float,
) -> bool:
    """Exact 2D Separating Axis Theorem (SAT) overlap test between two OBBs."""
    # Quick bounding circle rejection
    r_sum = math.hypot(hl1, hw1) + math.hypot(hl2, hw2)
    if math.hypot(cx1 - cx2, cy1 - cy2) > r_sum:
        return False

    c1 = _obb_corners_2d(cx1, cy1, hl1, hw1, yaw1)
    c2 = _obb_corners_2d(cx2, cy2, hl2, hw2, yaw2)

    axes = [
        (math.cos(yaw1), math.sin(yaw1)),
        (-math.sin(yaw1), math.cos(yaw1)),
        (math.cos(yaw2), math.sin(yaw2)),
        (-math.sin(yaw2), math.cos(yaw2)),
    ]
    for ax, ay in axes:
        p1 = c1[:, 0] * ax + c1[:, 1] * ay
        p2 = c2[:, 0] * ax + c2[:, 1] * ay
        if float(p1.max()) < float(p2.min()) or float(p2.max()) < float(p1.min()):
            return False
    return True


def point_to_obb_sdf_2d(
    px: float,
    py: float,
    cx: float,
    cy: float,
    half_len: float,
    half_wid: float,
    yaw_rad: float,
) -> float:
    """Analytical signed distance from point (px, py) to OBB (negative inside)."""
    dx = px - cx
    dy = py - cy
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    qx = abs(lx) - max(0.0, half_len)
    qy = abs(ly) - max(0.0, half_wid)
    out_d = math.hypot(max(qx, 0.0), max(qy, 0.0))
    in_d = min(max(qx, qy), 0.0)
    return out_d + in_d


def make_bonus_digit_bgra(
    digit: Union[int, str],
    seed: int = 42,
    size: int = 160,
    style: str = "sticker",
) -> np.ndarray:
    """Create a crisp BGRA patch for AIC finals bonus digit (1..8).

    Default 'sticker' style matches the official circular royal-blue/navy disc with
    crisp bold white digit used by the trained YOLOv8n 8-class finals detector.
    Optional 'flat_navy' style renders a bold navy digit directly on transparent ground.
    """
    d_str = str(int(digit))
    rng = random.Random(int(seed) + int(digit) * 97)
    s = int(size)
    bgr = np.zeros((s, s, 3), dtype=np.uint8)
    alpha = np.zeros((s, s), dtype=np.uint8)

    if style == "flat_navy":
        fg_bgr = (145, 48, 22)
        mask = _render_digit_mask(d_str, s, s, rng)
        bgr[:, :] = fg_bgr
        alpha = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        return np.dstack([bgr, alpha])

    # Official AIC finals circular blue/navy sticker with bold white digit
    # BGR chosen in the high-recall center of real YOLO training distribution
    b = rng.randint(148, 184)
    g = rng.randint(38, 62)
    r = rng.randint(14, 32)
    bgr[:, :] = (b, g, r)

    # Subtle shading gradient for realism
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    grad = 1.0 + 0.07 * ((yy / s) - 0.5) - 0.05 * ((xx / s) - 0.5)
    bgr = np.clip(bgr.astype(np.float32) * grad[:, :, None], 0, 255).astype(np.uint8)

    center = (s // 2, s // 2)
    rad = int(s * 0.46)
    cv2.circle(alpha, center, rad, 255, -1, cv2.LINE_AA)

    mask = _render_digit_mask(d_str, s, s, rng)
    fg = np.array([248.0, 248.0, 250.0], dtype=np.float32)[None, None, :]
    blended = bgr.astype(np.float32) * (1.0 - mask[:, :, None]) + fg * mask[:, :, None]
    bgr = np.clip(blended, 0, 255).astype(np.uint8)
    return np.dstack([bgr, alpha])


def _render_digit_mask(digit: str, out_w: int, out_h: int, rng: random.Random) -> np.ndarray:
    """Render bold legible digit mask in [0, 1] using PIL TrueType bold font or OpenCV."""
    fonts = _get_bold_fonts()
    if PILImage is not None and PILImageDraw is not None and PILImageFont is not None and fonts:
        # Prefer Arial Bold (index 0) for crisp 3 vs 8 distinction
        font_path = fonts[0]
        font_px = max(18, int(min(out_w, out_h) * 0.70))
        try:
            font = PILImageFont.truetype(font_path, font_px)
            pil_img = PILImage.new("L", (out_w, out_h), 0)
            draw = PILImageDraw.Draw(pil_img)
            bbox = draw.textbbox((0, 0), digit, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = (out_w - tw) * 0.5 - bbox[0]
            ty = (out_h - th) * 0.5 - bbox[1]
            draw.text((tx, ty), digit, fill=255, font=font)
            return np.asarray(pil_img, dtype=np.float32) / 255.0
        except Exception:
            pass

    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    scale = 2.4 * (min(out_w, out_h) / 96.0)
    thick = max(3, int(round(7.5 * (min(out_w, out_h) / 96.0))))
    (tw, th), _ = cv2.getTextSize(digit, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    org = (max(2, (out_w - tw) // 2), min(out_h - 4, (out_h + th) // 2))
    cv2.putText(canvas, digit, org, cv2.FONT_HERSHEY_DUPLEX, scale, 255, thick, cv2.LINE_AA)
    return canvas.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# 1. FinalsTrackRenderer (3D Standing Cones + Ground Flat Props)
# ---------------------------------------------------------------------------

class FinalsTrackRenderer(TrackRenderer):
    """Enhanced renderer with 3D perspective standing red cones, hollow red no-entry
    rectangles, random yellow lines, and YOLO-readable bonus digits (1..8).

    Supports both random seeded prop placement and deterministic scenario prop specs.
    """

    def __init__(
        self,
        track_json: Union[str, Path],
        cam: Optional[CameraModel] = None,
        max_d_mm: float = 5000.0,
        random_yellow: int = 0,
        cones: int = 0,
        no_entry: int = 0,
        digits: int = 0,
        prop_seed: int = 42,
        props: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_base_yellow: bool = True,
        clear_base_yellow: bool = False,
    ):
        resolved = resolve_track_json(track_json)
        super().__init__(
            resolved,
            cam=cam,
            max_d_mm=max_d_mm,
            random_yellow=0,
            cones=0,
            no_entry=0,
            prop_seed=prop_seed,
        )
        self.max_d_mm = float(max_d_mm)
        self.track_json_path = resolved

        # Build exact drivable mask from base track PNG (white + yellow = drivable; blue = out)
        b_ch = self.base_map_bgr[:, :, 0].astype(np.int16)
        g_ch = self.base_map_bgr[:, :, 1].astype(np.int16)
        r_ch = self.base_map_bgr[:, :, 2].astype(np.int16)
        self.drivable_mask: np.ndarray = (b_ch - r_ch) <= 40

        # Precompute clean base map without yellow lines (for scenarios with clear_base_yellow=True)
        is_yellow_base = (r_ch > 180) & (g_ch > 150) & (b_ch < 100)
        self._clean_white_map_bgr = self.base_map_bgr.copy()
        self._clean_white_map_bgr[is_yellow_base] = WHITE_BGR

        # Segment vectors for fast s_m projection
        self._seg_a = self.centerline_mm[:-1]
        self._seg_b = self.centerline_mm[1:]
        self._seg_ab = self._seg_b - self._seg_a
        self._seg_len2 = np.sum(self._seg_ab * self._seg_ab, axis=1)

        self.world_props: List[Dict[str, Any]] = []
        self._digit_patches: Dict[str, np.ndarray] = {}

        self.set_props(
            props=props,
            random_yellow=random_yellow,
            cones=cones,
            no_entry=no_entry,
            digits=digits,
            seed=prop_seed,
            include_base_yellow=include_base_yellow,
            clear_base_yellow=clear_base_yellow,
        )

    def is_point_drivable(self, x_mm: float, y_mm: float) -> bool:
        """Check whether world ground point (x_mm, y_mm) lies inside the white/yellow track."""
        px, py = self._mm_to_px(x_mm, y_mm)
        if px < 0 or px >= self.W or py < 0 or py >= self.H:
            return False
        return bool(self.drivable_mask[py, px])

    def pose_at_s(
        self, s_val: float
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """Smoothly interpolate centerline position & tangent at arc-length s_val (m).

        Returns (x_mm, y_mm, yaw_rad, t_hat, n_left).
        """
        s_clamped = _clamp(float(s_val), float(self.s_m[0]), float(self.s_m[-1]))
        idx = int(np.searchsorted(self.s_m, s_clamped, side="right")) - 1
        idx = max(0, min(len(self.s_m) - 2, idx))
        s0 = float(self.s_m[idx])
        s1 = float(self.s_m[idx + 1])
        alpha = (s_clamped - s0) / max(1e-9, s1 - s0)
        p0 = self.centerline_mm[idx]
        p1 = self.centerline_mm[idx + 1]
        p = p0 + alpha * (p1 - p0)

        # Use a short symmetric window for clean tangent direction
        i0 = max(0, idx - 1)
        i1 = min(len(self.centerline_mm) - 1, idx + 2)
        t = self.centerline_mm[i1] - self.centerline_mm[i0]
        tn = float(np.linalg.norm(t))
        if tn < 1e-9:
            t = np.array([1.0, 0.0], dtype=np.float64)
        else:
            t = t / tn
        n_left = np.array([-t[1], t[0]], dtype=np.float64)
        yaw = math.atan2(float(t[1]), float(t[0]))
        return float(p[0]), float(p[1]), yaw, t, n_left

    def project_s_from_xy(self, x_mm: float, y_mm: float) -> Tuple[float, float, float]:
        """Project world point onto track centerline -> (s_m, signed_lat_mm, abs_dist_mm)."""
        p = np.array([float(x_mm), float(y_mm)], dtype=np.float64)
        ap = p - self._seg_a
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(
                self._seg_len2 > 1e-12,
                np.sum(ap * self._seg_ab, axis=1) / self._seg_len2,
                0.0,
            )
        t = np.clip(t, 0.0, 1.0)
        proj = self._seg_a + self._seg_ab * t[:, None]
        dists = np.hypot(proj[:, 0] - p[0], proj[:, 1] - p[1])
        best_i = int(np.argmin(dists))
        best_t = float(t[best_i])
        s_val = float(self.s_m[best_i] + best_t * (self.s_m[best_i + 1] - self.s_m[best_i]))
        seg = self._seg_ab[best_i]
        seg_n = float(np.linalg.norm(seg)) + 1e-9
        t_hat = seg / seg_n
        n_left = np.array([-t_hat[1], t_hat[0]], dtype=np.float64)
        rel = p - proj[best_i]
        signed_lat = float(np.dot(rel, n_left))
        return s_val, signed_lat, float(dists[best_i])

    def _normalize_prop_spec(
        self, spec: Dict[str, Any], idx: int, rng: random.Random
    ) -> Dict[str, Any]:
        raw_type = str(spec.get("type", "")).strip().lower()
        if raw_type in ("1", "2", "3", "4", "5", "6", "7", "8"):
            ptype = "digit"
            inferred_val: Optional[int] = int(raw_type)
        else:
            ptype = PROP_TYPE_ALIASES.get(raw_type, raw_type)
            inferred_val = None
        if ptype not in ("red_cone", "red_no_entry", "digit", "yellow_stop_line"):
            raise ValueError(f"Unsupported prop type: {spec.get('type')!r}")

        # Resolve world pose (x_mm, y_mm, yaw_rad, s_m, lat_mm)
        if "x_mm" in spec and "y_mm" in spec:
            x_mm = float(spec["x_mm"])
            y_mm = float(spec["y_mm"])
            s_val, lat_mm, _ = self.project_s_from_xy(x_mm, y_mm)
            _, _, track_yaw, _, _ = self.pose_at_s(s_val)
            if "yaw_rad" in spec:
                yaw_rad = float(spec["yaw_rad"])
            elif "yaw_deg" in spec:
                yaw_rad = math.radians(float(spec["yaw_deg"]))
            else:
                yaw_off = float(spec.get("yaw_offset", 0.0))
                if "yaw_offset_deg" in spec:
                    yaw_off = math.radians(float(spec["yaw_offset_deg"]))
                yaw_rad = _wrap_angle(track_yaw + yaw_off)
            if "s_m" in spec:
                s_val = float(spec["s_m"])
            if "lat_mm" in spec:
                lat_mm = float(spec["lat_mm"])
        elif "s_m" in spec:
            s_val = float(spec["s_m"])
            lat_mm = float(spec.get("lat_mm", 0.0))
            cx, cy, track_yaw, _, n_left = self.pose_at_s(s_val)
            x_mm = cx + float(n_left[0]) * lat_mm
            y_mm = cy + float(n_left[1]) * lat_mm
            yaw_off = float(spec.get("yaw_offset", 0.0))
            if "yaw_offset_deg" in spec:
                yaw_off = math.radians(float(spec["yaw_offset_deg"]))
            yaw_rad = _wrap_angle(track_yaw + yaw_off)
        else:
            raise ValueError(f"Prop spec must specify either 's_m' or ('x_mm','y_mm'): {spec}")

        prop_id = str(spec.get("id", f"{ptype}_{idx}"))
        is_finish = bool(spec.get("is_finish", False))
        from_base = bool(spec.get("from_base_map", False))

        if ptype == "red_cone":
            base_mm = float(spec.get("base_mm", 135.0))
            height_mm = float(spec.get("height_mm", 150.0))
            return {
                "id": prop_id,
                "type": "red_cone",
                "class_name": "red_cone",
                "s_m": round(s_val, 4),
                "lat_mm": round(lat_mm, 2),
                "x_mm": round(x_mm, 2),
                "y_mm": round(y_mm, 2),
                "yaw_rad": round(yaw_rad, 5),
                "length_mm": base_mm,
                "width_mm": base_mm,
                "height_mm": height_mm,
                "radius_mm": 0.5 * base_mm,
                "value": None,
                "is_finish": False,
                "from_base_map": False,
                "triggered": False,
                "collected": False,
                "stopped_ok": False,
            }

        if ptype == "red_no_entry":
            length_mm = float(spec.get("length_mm", 240.0))
            width_mm = float(spec.get("width_mm", 130.0))
            border_mm = float(spec.get("border_mm", 10.0))
            return {
                "id": prop_id,
                "type": "red_no_entry",
                "class_name": "red_no_entry",
                "s_m": round(s_val, 4),
                "lat_mm": round(lat_mm, 2),
                "x_mm": round(x_mm, 2),
                "y_mm": round(y_mm, 2),
                "yaw_rad": round(yaw_rad, 5),
                "length_mm": length_mm,
                "width_mm": width_mm,
                "height_mm": 0.0,
                "radius_mm": 0.5 * math.hypot(length_mm, width_mm),
                "border_mm": border_mm,
                "value": None,
                "is_finish": False,
                "from_base_map": False,
                "triggered": False,
                "collected": False,
                "stopped_ok": False,
            }

        if ptype == "digit":
            val = spec.get("value", inferred_val)
            if val is None:
                val = rng.randint(1, 8)
            val_int = int(val)
            if not (1 <= val_int <= 8):
                raise ValueError(f"Bonus digit value must be in 1..8, got {val_int}")
            size_mm = float(spec.get("size_mm", 130.0))
            style = str(spec.get("style", "sticker"))
            return {
                "id": prop_id,
                "type": "digit",
                "class_name": str(val_int),
                "s_m": round(s_val, 4),
                "lat_mm": round(lat_mm, 2),
                "x_mm": round(x_mm, 2),
                "y_mm": round(y_mm, 2),
                "yaw_rad": round(yaw_rad, 5),
                "length_mm": size_mm,
                "width_mm": size_mm,
                "height_mm": 0.0,
                "radius_mm": 0.5 * size_mm,
                "value": val_int,
                "style": style,
                "is_finish": False,
                "from_base_map": False,
                "triggered": False,
                "collected": False,
                "stopped_ok": False,
            }

        # yellow_stop_line
        thick_mm = float(spec.get("width_mm", spec.get("length_mm", 25.0)))
        span_mm = float(spec.get("span_mm", self.lane_width_mm))
        return {
            "id": prop_id,
            "type": "yellow_stop_line",
            "class_name": "yellow_stop_line",
            "s_m": round(s_val, 4),
            "lat_mm": round(lat_mm, 2),
            "x_mm": round(x_mm, 2),
            "y_mm": round(y_mm, 2),
            "yaw_rad": round(yaw_rad, 5),
            "length_mm": thick_mm,
            "width_mm": span_mm,
            "height_mm": 0.0,
            "radius_mm": 0.5 * span_mm,
            "span_mm": span_mm,
            "value": None,
            "is_finish": is_finish,
            "from_base_map": from_base,
            "triggered": False,
            "collected": False,
            "stopped_ok": False,
        }

    def set_props(
        self,
        props: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        random_yellow: int = 0,
        cones: int = 0,
        no_entry: int = 0,
        digits: int = 0,
        seed: int = 42,
        include_base_yellow: bool = True,
        clear_base_yellow: bool = False,
    ) -> List[Dict[str, Any]]:
        """Configure deterministic and/or randomly generated finals props and paint flat map texture."""
        self.map_bgr = (
            self._clean_white_map_bgr.copy() if clear_base_yellow else self.base_map_bgr.copy()
        )
        self.world_props = []
        self._digit_patches = {}
        self.injected_props = {"random_yellow": [], "cones": [], "no_entry": [], "digits": []}

        specs: List[Dict[str, Any]] = []
        if isinstance(props, list):
            specs = list(props)
        elif isinstance(props, dict):
            specs = list(props.get("specs", []))
            random_yellow = int(props.get("random_yellow", random_yellow))
            cones = int(props.get("cones", cones))
            no_entry = int(props.get("no_entry", no_entry))
            digits = int(props.get("digits", digits))
            seed = int(props.get("seed", seed))
            include_base_yellow = bool(props.get("include_base_yellow", include_base_yellow))
            clear_base_yellow = bool(props.get("clear_base_yellow", clear_base_yellow))
            if clear_base_yellow:
                self.map_bgr = self._clean_white_map_bgr.copy()

        rng = random.Random(int(seed))
        total_s = float(self.s_m[-1])
        placed_s: List[float] = []

        # 1) Base-map yellow stop lines (already painted on base_map_bgr)
        base_yellows = list(self.meta.get("yellow_lines_s_m", []))
        if include_base_yellow and not clear_base_yellow and base_yellows:
            max_y_s = max(base_yellows)
            for i, ys in enumerate(base_yellows):
                is_fin = abs(float(ys) - float(max_y_s)) < 1e-3 and float(ys) > 0.7 * total_s
                p_dict = self._normalize_prop_spec(
                    {
                        "id": f"base_yellow_{i}",
                        "type": "yellow_stop_line",
                        "s_m": float(ys),
                        "lat_mm": 0.0,
                        "is_finish": is_fin,
                        "from_base_map": True,
                    },
                    i,
                    rng,
                )
                self.world_props.append(p_dict)
                placed_s.append(float(ys))

        # 2) Explicit deterministic scenario specs
        for idx, sp in enumerate(specs):
            p_dict = self._normalize_prop_spec(sp, idx, rng)
            self.world_props.append(p_dict)
            placed_s.append(float(p_dict["s_m"]))

        # 3) Random seeded prop generation
        half_lane = 0.5 * self.lane_width_mm

        def pick_s(min_gap: float = 0.65) -> Optional[float]:
            lo_s = 0.85
            hi_s = max(1.2, total_s - 0.90)
            for _ in range(100):
                sv = rng.uniform(lo_s, hi_s)
                if all(abs(sv - prev) >= min_gap for prev in placed_s):
                    placed_s.append(sv)
                    return sv
            return None

        # Random yellow lines
        for i in range(max(0, int(random_yellow))):
            sv = pick_s(min_gap=0.75)
            if sv is None:
                break
            p_dict = self._normalize_prop_spec(
                {
                    "id": f"rand_yellow_{i}",
                    "type": "yellow_stop_line",
                    "s_m": sv,
                    "lat_mm": 0.0,
                    "width_mm": 25.0,
                },
                len(self.world_props),
                rng,
            )
            self.world_props.append(p_dict)

        # Random 3D cones (135x135x150 mm, offset ±95..140 mm leaving passable corridor)
        for i in range(max(0, int(cones))):
            sv = pick_s(min_gap=0.60)
            if sv is None:
                break
            side = rng.choice([-1.0, 1.0])
            lat_mm = side * rng.uniform(95.0, max(105.0, half_lane - 75.0))
            p_dict = self._normalize_prop_spec(
                {
                    "id": f"rand_cone_{i}",
                    "type": "red_cone",
                    "s_m": sv,
                    "lat_mm": lat_mm,
                    "yaw_offset": rng.uniform(-0.35, 0.35),
                },
                len(self.world_props),
                rng,
            )
            self.world_props.append(p_dict)

        # Random hollow red no-entry rectangles (240x130 mm)
        for i in range(max(0, int(no_entry))):
            sv = pick_s(min_gap=0.70)
            if sv is None:
                break
            side = rng.choice([-1.0, 1.0])
            lat_mm = side * rng.uniform(100.0, max(110.0, half_lane - 75.0))
            p_dict = self._normalize_prop_spec(
                {
                    "id": f"rand_no_entry_{i}",
                    "type": "red_no_entry",
                    "s_m": sv,
                    "lat_mm": lat_mm,
                },
                len(self.world_props),
                rng,
            )
            self.world_props.append(p_dict)

        # Random bonus digits (1..8, flat on white lane)
        avail_digits = list(range(1, 9))
        rng.shuffle(avail_digits)
        for i in range(max(0, int(digits))):
            sv = pick_s(min_gap=0.55)
            if sv is None:
                break
            val = avail_digits[i % len(avail_digits)]
            lat_mm = rng.uniform(-90.0, 90.0)
            p_dict = self._normalize_prop_spec(
                {
                    "id": f"rand_digit_{i}",
                    "type": "digit",
                    "s_m": sv,
                    "lat_mm": lat_mm,
                    "value": val,
                    "size_mm": rng.uniform(118.0, 136.0),
                },
                len(self.world_props),
                rng,
            )
            self.world_props.append(p_dict)

        # 4) Paint ground-flat props onto self.map_bgr & cache digit BGRA patches
        for p in self.world_props:
            ptype = p["type"]
            if ptype == "yellow_stop_line":
                if not p.get("from_base_map", False):
                    self._paint_yellow_line_on_map(p)
                    self.injected_props["random_yellow"].append(p)
            elif ptype == "red_no_entry":
                self._paint_no_entry_on_map(p)
                self.injected_props["no_entry"].append(p)
            elif ptype == "digit":
                patch = make_bonus_digit_bgra(
                    p["value"],
                    seed=int(seed) + hash(p["id"]) % 10000,
                    size=160,
                    style=str(p.get("style", "sticker")),
                )
                self._digit_patches[p["id"]] = patch
                self._paint_digit_on_map(p, patch)
                self.injected_props["digits"].append(p)
            elif ptype == "red_cone":
                # Do NOT paint flat circles for cones on map_bgr; cones are projected in 3D in render()
                self.injected_props["cones"].append(p)

        return self.world_props

    def _paint_yellow_line_on_map(self, prop: Dict[str, Any]) -> None:
        cx, cy = float(prop["x_mm"]), float(prop["y_mm"])
        yaw = float(prop["yaw_rad"])
        t = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        n_left = np.array([-t[1], t[0]], dtype=np.float64)
        half_span = 0.5 * float(prop.get("span_mm", self.lane_width_mm))
        half_thick = 0.5 * float(prop.get("length_mm", 25.0))
        center = np.array([cx, cy], dtype=np.float64)
        corners = [
            center - t * half_thick - n_left * half_span,
            center + t * half_thick - n_left * half_span,
            center + t * half_thick + n_left * half_span,
            center - t * half_thick + n_left * half_span,
        ]
        poly = np.array([self._mm_to_px(c[0], c[1]) for c in corners], dtype=np.int32)
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        paint_mask = (mask > 0) & self.drivable_mask
        self.map_bgr[paint_mask] = YELLOW_BGR

    def _paint_no_entry_on_map(self, prop: Dict[str, Any]) -> None:
        cx, cy = float(prop["x_mm"]), float(prop["y_mm"])
        yaw = float(prop["yaw_rad"])
        hl = 0.5 * float(prop.get("length_mm", 240.0))
        hw = 0.5 * float(prop.get("width_mm", 130.0))
        border_mm = float(prop.get("border_mm", 10.0))
        outer_mm = _obb_corners_2d(cx, cy, hl, hw, yaw)
        inner_mm = _obb_corners_2d(
            cx, cy, max(1.0, hl - border_mm), max(1.0, hw - border_mm), yaw
        )
        outer_px = np.array([self._mm_to_px(c[0], c[1]) for c in outer_mm], dtype=np.int32)
        inner_px = np.array([self._mm_to_px(c[0], c[1]) for c in inner_mm], dtype=np.int32)
        ring_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.fillPoly(ring_mask, [outer_px], 255)
        cv2.fillPoly(ring_mask, [inner_px], 0)
        # Ensure minimum 2px crisp border at map resolution
        if border_mm / self.mpp < 2.2:
            cv2.polylines(ring_mask, [outer_px], True, 255, 2)
        self.map_bgr[ring_mask > 0] = RED_BGR

    def _paint_digit_on_map(self, prop: Dict[str, Any], patch_bgra: np.ndarray) -> None:
        """Warp upright digit BGRA onto self.map_bgr so camera approaching +yaw sees it upright."""
        cx, cy = float(prop["x_mm"]), float(prop["y_mm"])
        yaw = float(prop["yaw_rad"])
        t = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        n_left = np.array([-t[1], t[0]], dtype=np.float64)
        hh = 0.5 * float(prop.get("length_mm", 130.0))
        hw = 0.5 * float(prop.get("width_mm", 130.0))
        center = np.array([cx, cy], dtype=np.float64)

        # Upright orientation when viewed along +t:
        # top = +t (farther), bottom = -t (closer), left = +n_left, right = -n_left
        corners_mm = [
            center + t * hh + n_left * hw,  # TL
            center + t * hh - n_left * hw,  # TR
            center - t * hh - n_left * hw,  # BR
            center - t * hh + n_left * hw,  # BL
        ]
        dst_px = np.array(
            [[c[0] / self.mpp, (self.H - 1) - c[1] / self.mpp] for c in corners_mm],
            dtype=np.float32,
        )
        bx0 = max(0, int(math.floor(float(dst_px[:, 0].min()))) - 1)
        by0 = max(0, int(math.floor(float(dst_px[:, 1].min()))) - 1)
        bx1 = min(self.W, int(math.ceil(float(dst_px[:, 0].max()))) + 2)
        by1 = min(self.H, int(math.ceil(float(dst_px[:, 1].max()))) + 2)
        rw, rh = bx1 - bx0, by1 - by0
        if rw < 2 or rh < 2:
            return
        ph, pw = patch_bgra.shape[:2]
        src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
        dst_local = dst_px - np.float32([[bx0, by0]])
        H_mat = cv2.getPerspectiveTransform(src, dst_local)
        warped = cv2.warpPerspective(
            patch_bgra,
            H_mat,
            (rw, rh),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        a_f = warped[:, :, 3:4].astype(np.float32) / 255.0
        roi = self.map_bgr[by0:by1, bx0:bx1].astype(np.float32)
        out = a_f * warped[:, :, :3].astype(np.float32) + (1.0 - a_f) * roi
        self.map_bgr[by0:by1, bx0:bx1] = np.clip(out, 0, 255).astype(np.uint8)

    def _world_to_cam_wd(
        self, wx_mm: float, wy_mm: float, cam_x_mm: float, cam_y_mm: float, cam_yaw: float
    ) -> Tuple[float, float]:
        """Convert world ground (wx, wy) to camera ground frame (w_right_mm, d_fwd_mm)."""
        dx = wx_mm - cam_x_mm
        dy = wy_mm - cam_y_mm
        c = math.cos(cam_yaw)
        s = math.sin(cam_yaw)
        d_mm = dx * c + dy * s
        w_mm = dx * s - dy * c
        return w_mm, d_mm

    def _project_3d_point(
        self, w_mm: float, d_mm: float, z_mm: float
    ) -> Optional[Tuple[float, float]]:
        dz = self.cam.h - z_mm
        denom = d_mm + dz * self.cam.tan_a
        if denom <= 10.0:
            return None
        return self.cam.wdz_to_uv(w_mm, d_mm, z_mm)

    def _overlay_crisp_digits_3d(
        self, img: np.ndarray, cam_x_mm: float, cam_y_mm: float, cam_yaw: float
    ) -> None:
        """Project high-res 160x160 digit stickers directly onto camera frame for razor-sharp YOLO detection."""
        h_img, w_img = img.shape[:2]
        for prop in self.world_props:
            if prop["type"] != "digit":
                continue
            patch = self._digit_patches.get(prop["id"])
            if patch is None:
                continue
            cx, cy = float(prop["x_mm"]), float(prop["y_mm"])
            wc, dc = self._world_to_cam_wd(cx, cy, cam_x_mm, cam_y_mm, cam_yaw)
            if dc < 120.0 or dc > min(2400.0, self.max_d_mm):
                continue
            if abs(wc) > dc * 1.35 + 150.0:
                continue

            yaw = float(prop["yaw_rad"])
            t = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
            n_left = np.array([-t[1], t[0]], dtype=np.float64)
            hh = 0.5 * float(prop["length_mm"])
            hw = 0.5 * float(prop["width_mm"])
            center = np.array([cx, cy], dtype=np.float64)
            corners_mm = [
                center + t * hh + n_left * hw,  # TL
                center + t * hh - n_left * hw,  # TR
                center - t * hh - n_left * hw,  # BR
                center - t * hh + n_left * hw,  # BL
            ]
            dst_pts = []
            valid = True
            for pt in corners_mm:
                pw_mm, pd_mm = self._world_to_cam_wd(
                    float(pt[0]), float(pt[1]), cam_x_mm, cam_y_mm, cam_yaw
                )
                if pd_mm <= 40.0:
                    valid = False
                    break
                uv = self._project_3d_point(pw_mm, pd_mm, 0.0)
                if uv is None:
                    valid = False
                    break
                dst_pts.append([uv[0], uv[1]])
            if not valid:
                continue

            dst_arr = np.array(dst_pts, dtype=np.float32)
            bx0 = max(0, int(math.floor(float(dst_arr[:, 0].min()))) - 1)
            by0 = max(0, int(math.floor(float(dst_arr[:, 1].min()))) - 1)
            bx1 = min(w_img, int(math.ceil(float(dst_arr[:, 0].max()))) + 2)
            by1 = min(h_img, int(math.ceil(float(dst_arr[:, 1].max()))) + 2)
            rw, rh = bx1 - bx0, by1 - by0
            if rw < 5 or rh < 4:
                continue

            ph, pw = patch.shape[:2]
            src_arr = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
            H_mat = cv2.getPerspectiveTransform(src_arr, dst_arr - np.float32([[bx0, by0]]))
            warped = cv2.warpPerspective(
                patch,
                H_mat,
                (rw, rh),
                flags=cv2.INTER_AREA if rh < ph * 0.6 else cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            a_f = warped[:, :, 3:4].astype(np.float32) / 255.0
            roi = img[by0:by1, bx0:bx1].astype(np.float32)
            blended = a_f * warped[:, :, :3].astype(np.float32) + (1.0 - a_f) * roi
            img[by0:by1, bx0:bx1] = np.clip(blended, 0, 255).astype(np.uint8)

    def _render_3d_cones(
        self, img: np.ndarray, cam_x_mm: float, cam_y_mm: float, cam_yaw: float
    ) -> None:
        """Project 3D standing red cones (135x135mm base, 150mm height) far-to-near."""
        visible: List[Tuple[float, float, Dict[str, Any]]] = []
        for prop in self.world_props:
            if prop["type"] != "red_cone":
                continue
            w_c, d_c = self._world_to_cam_wd(
                float(prop["x_mm"]), float(prop["y_mm"]), cam_x_mm, cam_y_mm, cam_yaw
            )
            if d_c < 85.0 or d_c > self.max_d_mm:
                continue
            if abs(w_c) > d_c * 1.45 + 220.0:
                continue
            visible.append((d_c, w_c, prop))

        # Far-to-near depth sort (painter's algorithm)
        visible.sort(key=lambda item: -item[0])

        h_img, w_img = img.shape[:2]
        n_seg = 16
        angles = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)

        for d_c, w_c, prop in visible:
            base_mm = float(prop.get("length_mm", 135.0))
            height_mm = float(prop.get("height_mm", 150.0))
            hb = 0.5 * base_mm  # 67.5 mm
            base_h = 15.0       # 15 mm square rubber base plate
            r_bot = base_mm * 0.385  # ~52 mm lower cone radius
            r_top = base_mm * 0.090  # ~12 mm top cone radius
            z_c1 = height_mm * 0.48  # collar lower z (~72 mm)
            z_c2 = height_mm * 0.70  # collar upper z (~105 mm)
            rel_yaw = float(prop["yaw_rad"]) - cam_yaw
            cr, sr = math.cos(rel_yaw), math.sin(rel_yaw)

            def _base_pt(lx: float, ly: float, z: float) -> Optional[Tuple[float, float]]:
                # lx along cone forward, ly along cone left -> cam (d, w)
                dd = d_c + lx * cr - ly * sr
                ww = w_c - (lx * sr + ly * cr)
                return self._project_3d_point(ww, dd, z)

            # 1) Soft dark ground contact shadow (z=0, slightly wider than base)
            sh_r = hb * 1.14
            sh_local = [(-sh_r, -sh_r), (sh_r, -sh_r), (sh_r, sh_r), (-sh_r, sh_r)]
            sh_uv = [_base_pt(lx, ly, 0.0) for lx, ly in sh_local]
            if all(p is not None for p in sh_uv):
                sh_poly = np.array(sh_uv, dtype=np.int32)
                bx0 = max(0, int(sh_poly[:, 0].min()) - 2)
                by0 = max(0, int(sh_poly[:, 1].min()) - 2)
                bx1 = min(w_img, int(sh_poly[:, 0].max()) + 3)
                by1 = min(h_img, int(sh_poly[:, 1].max()) + 3)
                if bx1 > bx0 and by1 > by0:
                    smask = np.zeros((by1 - by0, bx1 - bx0), dtype=np.uint8)
                    cv2.fillPoly(smask, [sh_poly - np.array([bx0, by0], dtype=np.int32)], 255)
                    roi = img[by0:by1, bx0:bx1].astype(np.float32)
                    a_s = (smask.astype(np.float32) / 255.0) * 0.34
                    roi = roi * (1.0 - a_s[:, :, None]) + 20.0 * a_s[:, :, None]
                    img[by0:by1, bx0:bx1] = np.clip(roi, 0, 255).astype(np.uint8)

            # 2) 3D Square Base Plate (z = 0..base_h)
            b_local = [(-hb, -hb), (hb, -hb), (hb, hb), (-hb, hb)]
            bot_uv = [_base_pt(lx, ly, 0.0) for lx, ly in b_local]
            top_uv = [_base_pt(lx, ly, base_h) for lx, ly in b_local]
            if all(p is not None for p in bot_uv) and all(p is not None for p in top_uv):
                # Draw the 4 side faces sorted back-to-front by their edge midpoint d
                sides = []
                for k in range(4):
                    kn = (k + 1) % 4
                    mx_l = 0.5 * (b_local[k][0] + b_local[kn][0])
                    my_l = 0.5 * (b_local[k][1] + b_local[kn][1])
                    md = d_c + mx_l * cr - my_l * sr
                    sides.append((md, k, kn))
                sides.sort(key=lambda s_it: -s_it[0])
                for _, k, kn in sides:
                    quad = np.array(
                        [bot_uv[k], bot_uv[kn], top_uv[kn], top_uv[k]], dtype=np.int32
                    )
                    cv2.fillPoly(img, [quad], (18, 20, 172), cv2.LINE_AA)
                top_poly = np.array(top_uv, dtype=np.int32)
                cv2.fillPoly(img, [top_poly], (22, 24, 205), cv2.LINE_AA)

            # 3) 3D Tapered Frustum Body (z = base_h .. height_mm) with cylindrical shading
            def _ring(z_val: float) -> Tuple[List[Optional[Tuple[float, float]]], List[float]]:
                frac = _clamp((z_val - base_h) / max(1.0, height_mm - base_h), 0.0, 1.0)
                rad = r_bot * (1.0 - frac) + r_top * frac
                pts_uv: List[Optional[Tuple[float, float]]] = []
                ds: List[float] = []
                for ang in angles:
                    ww = w_c + rad * math.cos(float(ang))
                    dd = d_c + rad * math.sin(float(ang))
                    pts_uv.append(self._project_3d_point(ww, dd, z_val))
                    ds.append(dd)
                return pts_uv, ds

            ring_bot, d_bot = _ring(base_h)
            ring_c1, _ = _ring(z_c1)
            ring_c2, _ = _ring(z_c2)
            ring_top, _ = _ring(height_mm)

            # Build vertical strip quads (lower red, white reflective collar, upper red)
            # Sort strips back-to-front by strip midpoint depth
            strips: List[Tuple[float, int, int]] = []
            for k in range(n_seg):
                kn = (k + 1) % n_seg
                mid_d = 0.5 * (d_bot[k] + d_bot[kn])
                strips.append((mid_d, k, kn))
            strips.sort(key=lambda st: -st[0])

            for _, k, kn in strips:
                # Subtle cylindrical diffuse shading (light from upper-left front)
                ang_mid = float(angles[k]) + math.pi / n_seg
                shade = 0.80 + 0.24 * max(0.0, -math.sin(ang_mid) * 0.85 - math.cos(ang_mid) * 0.45)
                red_col = (
                    int(_clamp(RED_BGR[0] * shade, 10, 255)),
                    int(_clamp(RED_BGR[1] * shade, 10, 255)),
                    int(_clamp(RED_BGR[2] * shade, 140, 255)),
                )
                wht_col = (
                    int(_clamp(238 * (0.88 + 0.14 * shade), 185, 255)),
                    int(_clamp(240 * (0.88 + 0.14 * shade), 188, 255)),
                    int(_clamp(244 * (0.88 + 0.14 * shade), 192, 255)),
                )

                for r_lo, r_hi, col in (
                    (ring_bot, ring_c1, red_col),
                    (ring_c1, ring_c2, wht_col),
                    (ring_c2, ring_top, red_col),
                ):
                    p0, p1, p2, p3 = r_lo[k], r_lo[kn], r_hi[kn], r_hi[k]
                    if p0 is None or p1 is None or p2 is None or p3 is None:
                        continue
                    quad = np.array([p0, p1, p2, p3], dtype=np.int32)
                    cv2.fillPoly(img, [quad], col, cv2.LINE_AA)

            # Top cap
            valid_top = [p for p in ring_top if p is not None]
            if len(valid_top) >= 3:
                cv2.fillPoly(img, [np.array(valid_top, dtype=np.int32)], (20, 20, 185), cv2.LINE_AA)

    def project_prop_bbox_uv(
        self,
        prop: Dict[str, Any],
        cam_x_mm: float,
        cam_y_mm: float,
        cam_yaw: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Project a world prop's 3D/2D bounding corners into pixel bbox (x1, y1, x2, y2)."""
        cx, cy = float(prop["x_mm"]), float(prop["y_mm"])
        yaw = float(prop["yaw_rad"])
        hl = 0.5 * float(prop.get("length_mm", 130.0))
        hw = 0.5 * float(prop.get("width_mm", 130.0))
        hz = float(prop.get("height_mm", 0.0))

        base_corners = _obb_corners_2d(cx, cy, hl, hw, yaw)
        uv_pts: List[Tuple[float, float]] = []
        z_levels = (0.0, hz) if hz > 1.0 else (0.0,)
        for z in z_levels:
            for pt in base_corners:
                w_mm, d_mm = self._world_to_cam_wd(
                    float(pt[0]), float(pt[1]), cam_x_mm, cam_y_mm, cam_yaw
                )
                if d_mm <= 40.0 or d_mm > self.max_d_mm:
                    continue
                uv = self._project_3d_point(w_mm, d_mm, z)
                if uv is not None:
                    uv_pts.append(uv)
        if len(uv_pts) < 2:
            return None
        us = [p[0] for p in uv_pts]
        vs = [p[1] for p in uv_pts]
        x1 = _clamp(min(us), 0.0, float(self.cam.w - 1))
        y1 = _clamp(min(vs), 0.0, float(self.cam.h_img - 1))
        x2 = _clamp(max(us), 0.0, float(self.cam.w - 1))
        y2 = _clamp(max(vs), 0.0, float(self.cam.h_img - 1))
        if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
            return None
        return (x1, y1, x2, y2)

    def render(
        self,
        x_mm: float,
        y_mm: float,
        yaw: float,
        blur: float = 0.0,
        gain: float = 1.0,
        color_cast: Optional[Union[str, Sequence[float]]] = None,
        glare: float = 0.0,
    ) -> np.ndarray:
        """Render 640x480 BGR camera frame with ground map, crisp digits, 3D cones, and lighting."""
        # 1) Ground plane remap (no lighting perturbations yet so 3D props receive them too)
        out = super().render(x_mm, y_mm, yaw, blur=0.0, gain=1.0, color_cast=None, glare=0.0)

        # 2) High-res perspective bonus digits on ground
        self._overlay_crisp_digits_3d(out, x_mm, y_mm, yaw)

        # 3) 3D standing red cones (depth sorted far-to-near)
        self._render_3d_cones(out, x_mm, y_mm, yaw)

        # 4) Camera optics & lighting perturbations
        if blur > 0.0:
            k = int(blur * 2) * 2 + 1
            out = cv2.GaussianBlur(out, (k, k), blur)

        cast_bgr = parse_color_cast(color_cast)
        need_float = (gain != 1.0) or (cast_bgr != (1.0, 1.0, 1.0)) or (glare > 0.0)
        if need_float:
            arr = out.astype(np.float32)
            arr[:, :, 0] *= float(gain) * cast_bgr[0]
            arr[:, :, 1] *= float(gain) * cast_bgr[1]
            arr[:, :, 2] *= float(gain) * cast_bgr[2]
            if glare > 0.0:
                arr += float(glare) * 140.0 * self._glare_field
            out = np.clip(arr, 0, 255).astype(np.uint8)
        return out


# ---------------------------------------------------------------------------
# 2. VehicleDynamics (Real-Measured Ackermann + Reverse-Brake + Latency)
# ---------------------------------------------------------------------------

class VehicleDynamics:
    """Real-measured Ackermann bicycle model with braking/reverse-drag physics,
    voltage-dependent decel scaling, command latency queue, and domain randomization.
    """

    def __init__(
        self,
        wheelbase_m: float = 0.1395,
        max_steer_deg: float = 21.0,
        tau_speed: float = 0.61,
        tau_steer: float = 0.26,
        a_eff: float = 1.44,
        voltage: float = 12.0,
        voltage_nominal: float = 12.0,
        voltage_decel_slope: float = 0.08,
        a_eff_scale: float = 1.0,
        steer_scale: float = 1.0,
        speed_scale: float = 1.0,
        odom_scale_mismatch: float = 1.0,
        latency_ms: float = 0.0,
        latency_jitter_ms: float = 0.0,
        cam_offset_mm: float = 120.0,
        rear_axle_to_bumper_m: float = 0.180,
        seed: int = 42,
    ):
        self.wheelbase_m = float(wheelbase_m)
        self.max_steer_rad = math.radians(float(max_steer_deg))
        self.tau_speed = max(1e-3, float(tau_speed))
        self.tau_steer = max(1e-3, float(tau_steer))
        self.a_eff = max(0.1, float(a_eff))
        self.voltage = float(voltage)
        self.voltage_nominal = float(voltage_nominal)
        self.voltage_decel_slope = float(voltage_decel_slope)
        self.a_eff_scale = max(0.2, float(a_eff_scale))
        self.steer_scale = float(steer_scale)
        self.speed_scale = float(speed_scale)
        self.odom_scale_mismatch = float(odom_scale_mismatch)
        self.latency_ms = max(0.0, float(latency_ms))
        self.latency_jitter_ms = max(0.0, float(latency_jitter_ms))
        self.cam_offset_mm = float(cam_offset_mm)
        self.rear_axle_to_bumper_m = float(rear_axle_to_bumper_m)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

        # State
        self.t_s: float = 0.0
        self.x_mm: float = 0.0
        self.y_mm: float = 0.0
        self.yaw: float = 0.0
        self.v: float = 0.0
        self.w: float = 0.0
        self.delta: float = 0.0
        self.s_m: float = 0.0

        # Odom state (potentially scaled by odom_scale_mismatch)
        self.odom_x_m: float = 0.0
        self.odom_y_m: float = 0.0
        self.odom_yaw: float = 0.0
        self.odom_v: float = 0.0
        self.odom_w: float = 0.0
        self.odom_s_m: float = 0.0

        # Command latency queue: deque of (apply_time_s, v_cmd, w_cmd)
        self._cmd_queue: Deque[Tuple[float, float, float]] = deque()
        self._active_v_cmd: float = 0.0
        self._active_w_cmd: float = 0.0

        self.min_v_seen: float = 0.0
        self.max_v_seen: float = 0.0

    @property
    def effective_brake_decel(self) -> float:
        """Effective braking deceleration [m/s^2] including voltage & domain scale."""
        v_scale = _clamp(
            1.0 + self.voltage_decel_slope * (self.voltage - self.voltage_nominal),
            0.55,
            1.35,
        )
        return self.a_eff * self.a_eff_scale * v_scale

    def reset(
        self,
        x_mm: float,
        y_mm: float,
        yaw: float,
        v: float = 0.0,
        delta: float = 0.0,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        if seed is not None:
            self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self.t_s = 0.0
        self.x_mm = float(x_mm)
        self.y_mm = float(y_mm)
        self.yaw = _wrap_angle(float(yaw))
        self.v = float(v)
        self.delta = _clamp(float(delta), -self.max_steer_rad, self.max_steer_rad)
        self.w = self.v * math.tan(self.delta) / self.wheelbase_m
        self.s_m = 0.0

        self.odom_x_m = self.x_mm / 1000.0
        self.odom_y_m = self.y_mm / 1000.0
        self.odom_yaw = self.yaw
        self.odom_v = self.v * self.odom_scale_mismatch
        self.odom_w = self.w
        self.odom_s_m = 0.0

        self._cmd_queue.clear()
        self._active_v_cmd = 0.0
        self._active_w_cmd = 0.0
        self.min_v_seen = self.v
        self.max_v_seen = self.v
        return self.get_gt_pose(), self.get_odom()

    def _enqueue_and_pop_cmd(self, v_cmd: float, w_cmd: float) -> Tuple[float, float]:
        if self.latency_ms <= 1e-6 and self.latency_jitter_ms <= 1e-6:
            self._active_v_cmd = float(v_cmd)
            self._active_w_cmd = float(w_cmd)
            return self._active_v_cmd, self._active_w_cmd

        jitter = (
            self._rng.uniform(-self.latency_jitter_ms, self.latency_jitter_ms)
            if self.latency_jitter_ms > 0.0
            else 0.0
        )
        delay_s = max(0.0, (self.latency_ms + jitter) / 1000.0)
        deliver_t = self.t_s + delay_s
        # Maintain monotonic delivery order
        if self._cmd_queue and deliver_t < self._cmd_queue[-1][0]:
            deliver_t = self._cmd_queue[-1][0]
        self._cmd_queue.append((deliver_t, float(v_cmd), float(w_cmd)))

        while self._cmd_queue and self._cmd_queue[0][0] <= self.t_s + 1e-9:
            _, av, aw = self._cmd_queue.popleft()
            self._active_v_cmd = av
            self._active_w_cmd = aw
        return self._active_v_cmd, self._active_w_cmd

    def step(
        self, v_cmd: float, w_cmd: float, dt: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Advance vehicle physics by dt seconds with 4x sub-stepping."""
        dt = max(1e-4, float(dt))
        act_v_cmd, act_w_cmd = self._enqueue_and_pop_cmd(v_cmd, w_cmd)

        v_cmd_eff = act_v_cmd * self.speed_scale
        w_cmd_eff = act_w_cmd * self.steer_scale
        a_brake = self.effective_brake_decel

        n_sub = 4
        dt_s = dt / n_sub
        for _ in range(n_sub):
            # --- Longitudinal velocity dynamics ---
            if self.v > 1e-5 and v_cmd_eff < -1e-5:
                # Active reverse-braking while moving forward:
                # At nominal yellow_brake_speed = -0.25 -> a_act == a_brake (1.44 m/s^2)
                brake_gain = _clamp(0.75 + 1.0 * abs(v_cmd_eff), 0.75, 1.40)
                a_act = a_brake * brake_gain
                dv = -a_act * dt_s
                if self.v + dv >= 0.0:
                    self.v += dv
                else:
                    # Crossed zero while negative v_cmd persists -> vehicle REVERSES!
                    t_zero = self.v / a_act
                    dt_rem = max(0.0, dt_s - t_zero)
                    self.v = 0.0
                    # Accelerate backward toward negative v_cmd_eff
                    dv_rev = (dt_rem / (self.tau_speed + dt_rem)) * (v_cmd_eff - self.v)
                    max_rev_step = a_act * dt_rem
                    self.v += max(-max_rev_step, dv_rev)
            elif self.v < -1e-5 and v_cmd_eff > 1e-5:
                # Braking out of reverse into forward
                a_act = a_brake
                dv = a_act * dt_s
                if self.v + dv <= 0.0:
                    self.v += dv
                else:
                    t_zero = (-self.v) / a_act
                    dt_rem = max(0.0, dt_s - t_zero)
                    self.v = 0.0
                    self.v += (dt_rem / (self.tau_speed + dt_rem)) * (v_cmd_eff - self.v)
            elif abs(v_cmd_eff) < 1e-5:
                # Zero command: coasting drag + back-EMF/rolling friction to clean stop
                dv_fo = (dt_s / (self.tau_speed + dt_s)) * (0.0 - self.v)
                fric = 0.14 * dt_s
                if self.v > 0.0:
                    self.v = max(0.0, self.v + dv_fo - fric)
                elif self.v < 0.0:
                    self.v = min(0.0, self.v + dv_fo + fric)
                if abs(self.v) < 0.004:
                    self.v = 0.0
            else:
                # Same-direction acceleration or gradual speed change (forward or reverse!)
                self.v += (dt_s / (self.tau_speed + dt_s)) * (v_cmd_eff - self.v)

            # --- Steering dynamics ---
            v_ref = self.v if abs(self.v) > 0.05 else (v_cmd_eff if abs(v_cmd_eff) > 0.05 else 0.0)
            if abs(v_ref) > 0.02:
                d_cmd = math.atan(w_cmd_eff * self.wheelbase_m / v_ref)
            else:
                d_cmd = (
                    math.copysign(self.max_steer_rad, w_cmd_eff)
                    if abs(w_cmd_eff) > 1e-4
                    else 0.0
                )
            d_cmd = _clamp(d_cmd, -self.max_steer_rad, self.max_steer_rad)
            self.delta += (dt_s / (self.tau_steer + dt_s)) * (d_cmd - self.delta)

            # --- Ackermann kinematics integration ---
            omega = self.v * math.tan(self.delta) / self.wheelbase_m
            ds_m = self.v * dt_s
            self.x_mm += (ds_m * 1000.0) * math.cos(self.yaw)
            self.y_mm += (ds_m * 1000.0) * math.sin(self.yaw)
            self.yaw = _wrap_angle(self.yaw + omega * dt_s)
            self.s_m += abs(ds_m)
            self.w = omega

            # Odom integration (with optional scale mismatch)
            ds_odom = ds_m * self.odom_scale_mismatch
            self.odom_x_m += ds_odom * math.cos(self.odom_yaw)
            self.odom_y_m += ds_odom * math.sin(self.odom_yaw)
            self.odom_yaw = _wrap_angle(self.odom_yaw + omega * dt_s)
            self.odom_s_m += abs(ds_odom)

            if self.v < self.min_v_seen:
                self.min_v_seen = self.v
            if self.v > self.max_v_seen:
                self.max_v_seen = self.v

        self.t_s += dt
        self.odom_v = self.v * self.odom_scale_mismatch
        self.odom_w = self.w
        return self.get_gt_pose(), self.get_odom()

    def get_gt_pose(self) -> Dict[str, float]:
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        cam_x = self.x_mm + self.cam_offset_mm * c
        cam_y = self.y_mm + self.cam_offset_mm * s
        bumper_mm = self.rear_axle_to_bumper_m * 1000.0
        fb_x = self.x_mm + bumper_mm * c
        fb_y = self.y_mm + bumper_mm * s
        return {
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "yaw": self.yaw,
            "v": self.v,
            "w": self.w,
            "delta": self.delta,
            "s_m": self.s_m,
            "t_s": self.t_s,
            "cam_x_mm": cam_x,
            "cam_y_mm": cam_y,
            "cam_yaw": self.yaw,
            "front_bumper_x_mm": fb_x,
            "front_bumper_y_mm": fb_y,
            "min_v_seen": self.min_v_seen,
            "max_v_seen": self.max_v_seen,
        }

    def get_odom(self) -> Dict[str, float]:
        return {
            "x": self.odom_x_m,
            "y": self.odom_y_m,
            "yaw": self.odom_yaw,
            "v": self.odom_v,
            "w": self.odom_w,
            "s_m": self.odom_s_m,
            "voltage": self.voltage,
            "timestamp_sec": self.t_s,
        }


# ---------------------------------------------------------------------------
# 3. AICReferee (Official AIC Finals Auto-Referee)
# ---------------------------------------------------------------------------

class AICReferee:
    """Official AIC Finals rule checker & scorer.

    Rules enforced:
      - Exact 4-wheel coordinates (car_length=0.280m, car_width=0.191m,
        rear_axle_to_bumper=0.180m, wheelbase=0.1395m) & 3-disc chassis centers.
      - Lane boundary press (via exact track PNG drivable mask):
        +15s/s accumulated while >= 1 wheel is outside white lane;
        >= 3 wheels outside simultaneously -> DNF.
      - Red no-entry overlap (SAT OBB-vs-OBB): +30s (once per prop).
      - Red cone collision (SAT OBB-vs-OBB): +10s (once per prop).
      - Yellow stop line: must achieve full stop (|v| <= 0.02 m/s sustained >= 0.15s)
        BEFORE front bumper touches yellow line; touching before stopping or passing
        without full stop -> +15s (once per yellow line).
      - Bonus digit collection: vehicle chassis covers digit -> -5s * digit_value (once).
      - Finish line detection & net_score = lap_time + total_penalty - total_bonus.
    """

    def __init__(
        self,
        renderer: FinalsTrackRenderer,
        car_length_m: float = 0.280,
        car_width_m: float = 0.191,
        rear_axle_to_bumper_m: float = 0.180,
        wheelbase_m: float = 0.1395,
        still_speed_mps: float = 0.02,
        still_hold_s: float = 0.15,
        yellow_stop_window_mm: float = 520.0,
        digit_cover_tol_mm: float = 25.0,
    ):
        self.renderer = renderer
        self.car_length_mm = float(car_length_m) * 1000.0
        self.car_width_mm = float(car_width_m) * 1000.0
        self.rear_axle_to_bumper_mm = float(rear_axle_to_bumper_m) * 1000.0
        self.wheelbase_mm = float(wheelbase_m) * 1000.0
        self.still_speed_mps = float(still_speed_mps)
        self.still_hold_s = float(still_hold_s)
        self.yellow_stop_window_mm = float(yellow_stop_window_mm)
        self.digit_cover_tol_mm = float(digit_cover_tol_mm)
        self.end_s_m = float(renderer.meta.get("end_s_m", float(renderer.s_m[-1])))

        # 3-disc offsets matching AckermannCbfShield::compute_disc_offsets
        self.disc_radius_mm = 0.5 * self.car_width_mm
        x_mid_mm = self.rear_axle_to_bumper_mm - 0.5 * self.car_length_mm
        half_span_mm = max(0.0, 0.5 * self.car_length_mm - self.disc_radius_mm)
        self.disc_offsets_mm = (
            x_mid_mm - half_span_mm,
            x_mid_mm,
            x_mid_mm + half_span_mm,
        )

        # Runtime state
        self._init_counters()

    def _init_counters(self) -> None:
        self.lap_time_s: float = 0.0
        self.boundary_press_time_s: float = 0.0
        self.boundary_penalty_s: float = 0.0
        self.cone_hits: int = 0
        self.cone_penalty_s: float = 0.0
        self.no_entry_hits: int = 0
        self.no_entry_penalty_s: float = 0.0
        self.yellow_violations: int = 0
        self.yellow_penalty_s: float = 0.0
        self.yellow_stops_ok: int = 0
        self.bonus_digits_collected: List[int] = []
        self.total_bonus_s: float = 0.0
        self.finished: bool = False
        self.dnf: bool = False
        self.dnf_reason: str = ""
        self.events: List[Dict[str, Any]] = []

        self.wheels_out: int = 0
        self.wheel_states: Dict[str, bool] = {"FL": False, "FR": False, "RL": False, "RR": False}
        self.last_geom: Dict[str, Any] = {}
        self.centerline_dist_mm: float = 0.0
        self.current_s_m: float = 0.0

        # Per-yellow-line tracking
        self._yellow_ignored: Dict[str, bool] = {}
        self._yellow_stop_hold_s: Dict[str, float] = {}
        self._yellow_prev_d_before_mm: Dict[str, float] = {}
        self._has_moved: bool = False
        self._start_xy_mm: Tuple[float, float] = (0.0, 0.0)

    def compute_vehicle_geometry(self, x_mm: float, y_mm: float, yaw: float) -> Dict[str, Any]:
        """Compute exact 4-wheel world coords, 4 chassis corners, front bumper pts, and 3-disc centers."""
        c = math.cos(yaw)
        s = math.sin(yaw)
        hw = 0.5 * self.car_width_mm
        wb = self.wheelbase_mm
        fb = self.rear_axle_to_bumper_mm
        rb = fb - self.car_length_mm

        def to_world(lx: float, ly: float) -> Tuple[float, float]:
            return (x_mm + lx * c - ly * s, y_mm + lx * s + ly * c)

        wheels_mm = {
            "FL": to_world(wb, +hw),
            "FR": to_world(wb, -hw),
            "RL": to_world(0.0, +hw),
            "RR": to_world(0.0, -hw),
        }
        corners_mm = [
            to_world(fb, +hw),
            to_world(fb, -hw),
            to_world(rb, -hw),
            to_world(rb, +hw),
        ]
        front_bumper_pts_mm = [
            to_world(fb, +hw),
            to_world(fb, 0.0),
            to_world(fb, -hw),
        ]
        x_mid_local = fb - 0.5 * self.car_length_mm
        chassis_center_mm = to_world(x_mid_local, 0.0)
        disc_centers_mm = [to_world(off, 0.0) for off in self.disc_offsets_mm]
        return {
            "wheels_mm": wheels_mm,
            "corners_mm": corners_mm,
            "front_bumper_pts_mm": front_bumper_pts_mm,
            "chassis_center_mm": chassis_center_mm,
            "disc_centers_mm": disc_centers_mm,
            "disc_radius_mm": self.disc_radius_mm,
        }

    def _front_bumper_yellow_coords(
        self, front_bumper_pts: Sequence[Tuple[float, float]], prop: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Project front bumper points into yellow line local frame -> (max_along_mm, min_abs_lat_mm)."""
        px, py = float(prop["x_mm"]), float(prop["y_mm"])
        pyaw = float(prop["yaw_rad"])
        tx, ty = math.cos(pyaw), math.sin(pyaw)
        nx, ny = -ty, tx
        alongs: List[float] = []
        abs_lats: List[float] = []
        for fx, fy in front_bumper_pts:
            dx = fx - px
            dy = fy - py
            alongs.append(dx * tx + dy * ty)
            abs_lats.append(abs(dx * nx + dy * ny))
        return max(alongs), min(abs_lats)

    def reset(
        self,
        x_mm: float,
        y_mm: float,
        yaw: float,
        world_props: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        self._init_counters()
        self._start_xy_mm = (float(x_mm), float(y_mm))
        for p in world_props:
            p["triggered"] = False
            p["collected"] = False
            p["stopped_ok"] = False

        geom = self.compute_vehicle_geometry(x_mm, y_mm, yaw)
        self.last_geom = geom
        self.current_s_m, _, self.centerline_dist_mm = self.renderer.project_s_from_xy(x_mm, y_mm)

        # Ignore yellow lines that are already behind the spawn front bumper on the same corridor
        for p in world_props:
            if p["type"] != "yellow_stop_line":
                continue
            pid = p["id"]
            half_thick = 0.5 * float(p.get("length_mm", 25.0))
            half_span = 0.5 * float(p.get("span_mm", self.renderer.lane_width_mm))
            max_along, min_abs_lat = self._front_bumper_yellow_coords(
                geom["front_bumper_pts_mm"], p
            )
            d_before = (-half_thick) - max_along
            self._yellow_prev_d_before_mm[pid] = d_before
            self._yellow_stop_hold_s[pid] = 0.0
            # Already past or behind on same lane segment at spawn -> ignore
            if min_abs_lat <= half_span + 80.0 and d_before < -half_thick:
                self._yellow_ignored[pid] = True
            else:
                self._yellow_ignored[pid] = False

        self._check_wheels(geom)
        return self.get_status()

    def _check_wheels(self, geom: Dict[str, Any]) -> int:
        out_count = 0
        for name, (wx, wy) in geom["wheels_mm"].items():
            is_out = not self.renderer.is_point_drivable(wx, wy)
            self.wheel_states[name] = is_out
            if is_out:
                out_count += 1
        self.wheels_out = out_count
        return out_count

    def update(
        self,
        x_mm: float,
        y_mm: float,
        yaw: float,
        v_mps: float,
        dt: float,
        t_s: float,
        world_props: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate all official AIC rules over step dt."""
        if self.finished or self.dnf:
            return self.get_status()

        self.lap_time_s = float(t_s)
        geom = self.compute_vehicle_geometry(x_mm, y_mm, yaw)
        self.last_geom = geom
        self.current_s_m, _, self.centerline_dist_mm = self.renderer.project_s_from_xy(x_mm, y_mm)

        # Track whether vehicle has moved since reset (prevents false stop credit at t=0 spawn)
        moved_mm = math.hypot(x_mm - self._start_xy_mm[0], y_mm - self._start_xy_mm[1])
        if moved_mm >= 18.0 or abs(v_mps) >= 0.04:
            self._has_moved = True

        # 1) 4-wheel lane boundary press (+15s/s) and >= 3 wheels DNF
        wheels_out = self._check_wheels(geom)
        if wheels_out >= 1 and dt > 0.0:
            self.boundary_press_time_s += dt
            self.boundary_penalty_s += 15.0 * dt
        if wheels_out >= 3:
            self.dnf = True
            self.dnf_reason = "off_track_3_wheels"
            self.events.append({
                "type": "dnf",
                "t_s": round(t_s, 3),
                "detail": f"wheels_out={wheels_out}",
                "delta_s": 0.0,
            })
            return self.get_status()

        # Vehicle chassis OBB parameters
        car_cx, car_cy = geom["chassis_center_mm"]
        car_hl = 0.5 * self.car_length_mm
        car_hw = 0.5 * self.car_width_mm

        has_finish_yellow = False

        # 2) Evaluate props (no-entry, cones, yellow stop lines, bonus digits)
        for prop in world_props:
            ptype = prop["type"]
            pid = prop["id"]

            if ptype == "red_no_entry":
                if not prop["triggered"]:
                    hit = obb_overlap_2d(
                        car_cx,
                        car_cy,
                        car_hl,
                        car_hw,
                        yaw,
                        float(prop["x_mm"]),
                        float(prop["y_mm"]),
                        0.5 * float(prop["length_mm"]),
                        0.5 * float(prop["width_mm"]),
                        float(prop["yaw_rad"]),
                    )
                    if hit:
                        prop["triggered"] = True
                        self.no_entry_hits += 1
                        self.no_entry_penalty_s += 30.0
                        self.events.append({
                            "type": "no_entry_hit",
                            "t_s": round(t_s, 3),
                            "prop_id": pid,
                            "delta_s": 30.0,
                        })

            elif ptype == "red_cone":
                if not prop["triggered"]:
                    hit = obb_overlap_2d(
                        car_cx,
                        car_cy,
                        car_hl,
                        car_hw,
                        yaw,
                        float(prop["x_mm"]),
                        float(prop["y_mm"]),
                        0.5 * float(prop["length_mm"]),
                        0.5 * float(prop["width_mm"]),
                        float(prop["yaw_rad"]),
                    )
                    if hit:
                        prop["triggered"] = True
                        self.cone_hits += 1
                        self.cone_penalty_s += 10.0
                        self.events.append({
                            "type": "cone_hit",
                            "t_s": round(t_s, 3),
                            "prop_id": pid,
                            "delta_s": 10.0,
                        })

            elif ptype == "digit":
                if not prop["collected"]:
                    sdf = point_to_obb_sdf_2d(
                        float(prop["x_mm"]),
                        float(prop["y_mm"]),
                        car_cx,
                        car_cy,
                        car_hl,
                        car_hw,
                        yaw,
                    )
                    if sdf <= self.digit_cover_tol_mm:
                        val = int(prop["value"])
                        prop["collected"] = True
                        prop["triggered"] = True
                        self.bonus_digits_collected.append(val)
                        bonus = 5.0 * val
                        self.total_bonus_s += bonus
                        self.events.append({
                            "type": "digit_collected",
                            "t_s": round(t_s, 3),
                            "prop_id": pid,
                            "value": val,
                            "delta_s": -bonus,
                        })

            elif ptype == "yellow_stop_line":
                if prop.get("is_finish", False):
                    has_finish_yellow = True
                if self._yellow_ignored.get(pid, False):
                    continue
                half_thick = 0.5 * float(prop.get("length_mm", 25.0))
                half_span = 0.5 * float(prop.get("span_mm", self.renderer.lane_width_mm))
                max_along, min_abs_lat = self._front_bumper_yellow_coords(
                    geom["front_bumper_pts_mm"], prop
                )
                d_before = (-half_thick) - max_along
                prev_d = self._yellow_prev_d_before_mm.get(pid, d_before)
                in_corridor = min_abs_lat <= (half_span + 80.0)

                if in_corridor and not prop["stopped_ok"] and not prop["triggered"]:
                    # Check touching or crossing BEFORE full stop
                    touched_now = (d_before <= 0.0) and (max_along <= 320.0) and (
                        prev_d > -120.0 or abs(max_along) <= half_thick + 30.0
                    )
                    crossed_fast = (prev_d > 0.0) and (d_before <= 0.0)
                    if touched_now or crossed_fast:
                        prop["triggered"] = True
                        self.yellow_violations += 1
                        self.yellow_penalty_s += 15.0
                        self.events.append({
                            "type": "yellow_violation",
                            "t_s": round(t_s, 3),
                            "prop_id": pid,
                            "delta_s": 15.0,
                        })
                        if prop.get("is_finish", False):
                            self.finished = True
                            self.events.append({
                                "type": "finish",
                                "t_s": round(t_s, 3),
                                "detail": "crossed_finish_yellow_without_stop",
                                "delta_s": 0.0,
                            })
                    elif 0.0 < d_before <= self.yellow_stop_window_mm and self._has_moved:
                        # Approaching yellow line cleanly without touching yet
                        if abs(v_mps) <= self.still_speed_mps:
                            self._yellow_stop_hold_s[pid] = (
                                self._yellow_stop_hold_s.get(pid, 0.0) + dt
                            )
                            if self._yellow_stop_hold_s[pid] >= self.still_hold_s:
                                prop["stopped_ok"] = True
                                self.yellow_stops_ok += 1
                                self.events.append({
                                    "type": "yellow_stop_ok",
                                    "t_s": round(t_s, 3),
                                    "prop_id": pid,
                                    "d_before_mm": round(d_before, 1),
                                    "delta_s": 0.0,
                                })
                                if prop.get("is_finish", False):
                                    self.finished = True
                                    self.events.append({
                                        "type": "finish",
                                        "t_s": round(t_s, 3),
                                        "detail": "clean_stop_at_finish_yellow",
                                        "delta_s": 0.0,
                                    })
                        else:
                            self._yellow_stop_hold_s[pid] = 0.0
                    else:
                        self._yellow_stop_hold_s[pid] = 0.0

                self._yellow_prev_d_before_mm[pid] = d_before

        # 3) Fallback finish detection if front bumper reaches track end_s_m
        if not self.finished:
            fb_cx, fb_cy = geom["front_bumper_pts_mm"][1]
            fb_s, _, fb_dist = self.renderer.project_s_from_xy(fb_cx, fb_cy)
            end_thresh = (self.end_s_m - 0.06) if has_finish_yellow else (self.end_s_m - 0.15)
            if fb_s >= end_thresh and fb_dist <= self.renderer.lane_width_mm:
                self.finished = True
                self.events.append({
                    "type": "finish",
                    "t_s": round(t_s, 3),
                    "detail": f"reached_end_s={fb_s:.3f}m",
                    "delta_s": 0.0,
                })

        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        total_penalty = (
            self.boundary_penalty_s
            + self.no_entry_penalty_s
            + self.cone_penalty_s
            + self.yellow_penalty_s
        )
        net_score = self.lap_time_s + total_penalty - self.total_bonus_s
        return {
            "lap_time_s": round(self.lap_time_s, 4),
            "total_penalty_s": round(total_penalty, 4),
            "total_bonus_s": round(self.total_bonus_s, 4),
            "net_score_s": round(net_score, 4),
            "boundary_press_time_s": round(self.boundary_press_time_s, 4),
            "boundary_penalty_s": round(self.boundary_penalty_s, 4),
            "wheels_out": int(self.wheels_out),
            "wheel_states": dict(self.wheel_states),
            "wheel_coords_mm": dict(self.last_geom.get("wheels_mm", {})),
            "disc_centers_mm": list(self.last_geom.get("disc_centers_mm", [])),
            "disc_radius_mm": float(self.disc_radius_mm),
            "centerline_dist_mm": round(self.centerline_dist_mm, 2),
            "s_m": round(self.current_s_m, 4),
            "cone_hits": int(self.cone_hits),
            "cone_penalty_s": round(self.cone_penalty_s, 4),
            "no_entry_hits": int(self.no_entry_hits),
            "no_entry_penalty_s": round(self.no_entry_penalty_s, 4),
            "yellow_violations": int(self.yellow_violations),
            "yellow_penalty_s": round(self.yellow_penalty_s, 4),
            "yellow_stops_ok": int(self.yellow_stops_ok),
            "bonus_digits_collected": list(self.bonus_digits_collected),
            "finished": bool(self.finished),
            "dnf": bool(self.dnf),
            "dnf_reason": self.dnf_reason,
            "events": list(self.events),
        }


# ---------------------------------------------------------------------------
# 4. StepInfo & PCSimWorld + Top-Down Visualizer
# ---------------------------------------------------------------------------

@dataclass
class StepInfo:
    bgr_img: np.ndarray
    odom: Dict[str, float]
    gt_pose: Dict[str, float]
    world_props: List[Dict[str, Any]]
    referee_status: Dict[str, Any]
    done: bool
    done_reason: str


class PCSimWorld:
    """Complete Windows PC simulator world: 3D renderer + real dynamics + AIC referee + topdown HUD."""

    def __init__(
        self,
        track: Union[str, Path] = "fork_s",
        rate_hz: float = 30.0,
        # Vehicle dynamics & domain randomization
        wheelbase_m: float = 0.1395,
        max_steer_deg: float = 21.0,
        tau_speed: float = 0.61,
        tau_steer: float = 0.26,
        a_eff: float = 1.44,
        voltage: float = 12.0,
        voltage_decel_slope: float = 0.08,
        a_eff_scale: float = 1.0,
        steer_scale: float = 1.0,
        speed_scale: float = 1.0,
        odom_scale_mismatch: float = 1.0,
        latency_ms: float = 0.0,
        latency_jitter_ms: float = 0.0,
        # Camera / lighting
        cam_offset_mm: float = 120.0,
        blur: float = 0.0,
        gain: float = 1.0,
        color_cast: Optional[Union[str, Sequence[float]]] = None,
        glare: float = 0.0,
        # Props & episode settings
        random_yellow: int = 0,
        cones: int = 0,
        no_entry: int = 0,
        digits: int = 0,
        prop_seed: int = 42,
        include_base_yellow: bool = True,
        clear_base_yellow: bool = False,
        max_time_s: float = 120.0,
        seed: int = 42,
    ):
        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / self.rate_hz
        self.max_time_s = float(max_time_s)
        self.seed = int(seed)

        self.render_blur = float(blur)
        self.render_gain = float(gain)
        self.render_color_cast = color_cast
        self.render_glare = float(glare)

        self._default_prop_cfg = {
            "random_yellow": int(random_yellow),
            "cones": int(cones),
            "no_entry": int(no_entry),
            "digits": int(digits),
            "seed": int(prop_seed),
            "include_base_yellow": bool(include_base_yellow),
            "clear_base_yellow": bool(clear_base_yellow),
        }

        self.renderer = FinalsTrackRenderer(
            track_json=track,
            random_yellow=random_yellow,
            cones=cones,
            no_entry=no_entry,
            digits=digits,
            prop_seed=prop_seed,
            include_base_yellow=include_base_yellow,
            clear_base_yellow=clear_base_yellow,
        )
        self.cam = self.renderer.cam
        self.meta = self.renderer.meta
        self.track_name = str(self.meta.get("name", Path(str(track)).stem))

        self.dynamics = VehicleDynamics(
            wheelbase_m=wheelbase_m,
            max_steer_deg=max_steer_deg,
            tau_speed=tau_speed,
            tau_steer=tau_steer,
            a_eff=a_eff,
            voltage=voltage,
            voltage_decel_slope=voltage_decel_slope,
            a_eff_scale=a_eff_scale,
            steer_scale=steer_scale,
            speed_scale=speed_scale,
            odom_scale_mismatch=odom_scale_mismatch,
            latency_ms=latency_ms,
            latency_jitter_ms=latency_jitter_ms,
            cam_offset_mm=cam_offset_mm,
            rear_axle_to_bumper_m=0.180,
            seed=seed,
        )
        self.referee = AICReferee(
            renderer=self.renderer,
            car_length_m=0.280,
            car_width_m=0.191,
            rear_axle_to_bumper_m=0.180,
            wheelbase_m=wheelbase_m,
        )

        self.trajectory: List[Tuple[float, float]] = []
        self.last_step_info: Optional[StepInfo] = None
        self.reset()

    @property
    def world_props(self) -> List[Dict[str, Any]]:
        return self.renderer.world_props

    def _resolve_start_pose(
        self, pose: Optional[Union[Dict[str, Any], Sequence[float]]]
    ) -> Tuple[float, float, float, float]:
        """Resolve pose argument into (x_mm, y_mm, yaw_rad, v_mps)."""
        if pose is None:
            cl = self.renderer.centerline_mm
            n_cl = len(cl)
            max_idx = min(15, max(1, n_cl - 2))
            for idx in range(max_idx):
                x0, y0 = float(cl[idx, 0]), float(cl[idx, 1])
                i_next = min(idx + 2, n_cl - 1)
                x1, y1 = float(cl[i_next, 0]), float(cl[i_next, 1])
                yaw0 = math.atan2(y1 - y0, x1 - x0)
                geom = self.referee.compute_vehicle_geometry(x0, y0, yaw0)
                if all(
                    self.renderer.is_point_drivable(wx, wy)
                    for wx, wy in geom["wheels_mm"].values()
                ):
                    return x0, y0, yaw0, 0.0
            x0, y0 = float(cl[0, 0]), float(cl[0, 1])
            x1, y1 = float(cl[min(2, n_cl - 1), 0]), float(cl[min(2, n_cl - 1), 1])
            return x0, y0, math.atan2(y1 - y0, x1 - x0), 0.0

        if isinstance(pose, dict):
            v0 = float(pose.get("v", 0.0))
            if "x_mm" in pose and "y_mm" in pose:
                x0 = float(pose["x_mm"])
                y0 = float(pose["y_mm"])
                if "yaw" in pose:
                    yaw0 = float(pose["yaw"])
                elif "yaw_rad" in pose:
                    yaw0 = float(pose["yaw_rad"])
                elif "yaw_deg" in pose:
                    yaw0 = math.radians(float(pose["yaw_deg"]))
                else:
                    s_val, _, _ = self.renderer.project_s_from_xy(x0, y0)
                    _, _, yaw0, _, _ = self.renderer.pose_at_s(s_val)
                return x0, y0, yaw0, v0
            if "s_m" in pose:
                s_val = float(pose["s_m"])
                lat_mm = float(pose.get("lat_mm", 0.0))
                cx, cy, track_yaw, _, n_left = self.renderer.pose_at_s(s_val)
                x0 = cx + float(n_left[0]) * lat_mm
                y0 = cy + float(n_left[1]) * lat_mm
                yaw_off = float(pose.get("yaw_offset", 0.0))
                if "yaw_offset_deg" in pose:
                    yaw_off = math.radians(float(pose["yaw_offset_deg"]))
                return x0, y0, _wrap_angle(track_yaw + yaw_off), v0
            raise ValueError(f"Unsupported pose dict: {pose}")

        seq = list(pose)
        if len(seq) == 3:
            return float(seq[0]), float(seq[1]), float(seq[2]), 0.0
        if len(seq) == 4:
            return float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])
        raise ValueError(f"Pose sequence must have 3 or 4 items, got {len(seq)}")

    def reset(
        self,
        pose: Optional[Union[Dict[str, Any], Sequence[float]]] = None,
        props: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ) -> StepInfo:
        """Reset world props, vehicle pose/dynamics, and AIC referee; returns initial StepInfo."""
        if seed is not None:
            self.seed = int(seed)

        if props is None:
            self.renderer.set_props(
                props=None,
                random_yellow=self._default_prop_cfg["random_yellow"],
                cones=self._default_prop_cfg["cones"],
                no_entry=self._default_prop_cfg["no_entry"],
                digits=self._default_prop_cfg["digits"],
                seed=self._default_prop_cfg["seed"] if seed is None else int(seed),
                include_base_yellow=self._default_prop_cfg["include_base_yellow"],
                clear_base_yellow=self._default_prop_cfg["clear_base_yellow"],
            )
        else:
            self.renderer.set_props(
                props=props,
                seed=self._default_prop_cfg["seed"] if seed is None else int(seed),
                include_base_yellow=self._default_prop_cfg["include_base_yellow"],
                clear_base_yellow=self._default_prop_cfg["clear_base_yellow"],
            )

        x0, y0, yaw0, v0 = self._resolve_start_pose(pose)
        gt_pose, odom = self.dynamics.reset(x0, y0, yaw0, v=v0, seed=self.seed)
        ref_status = self.referee.reset(x0, y0, yaw0, self.renderer.world_props)

        self.trajectory = [(x0, y0)]
        img = self.renderer.render(
            gt_pose["cam_x_mm"],
            gt_pose["cam_y_mm"],
            gt_pose["cam_yaw"],
            blur=self.render_blur,
            gain=self.render_gain,
            color_cast=self.render_color_cast,
            glare=self.render_glare,
        )
        info = StepInfo(
            bgr_img=img,
            odom=odom,
            gt_pose=gt_pose,
            world_props=copy.deepcopy(self.renderer.world_props),
            referee_status=ref_status,
            done=False,
            done_reason="",
        )
        self.last_step_info = info
        return info

    def step(self, v_cmd: float, w_cmd: float) -> StepInfo:
        """Step physics by dt, update AIC referee, render camera frame, and return StepInfo."""
        gt_pose, odom = self.dynamics.step(float(v_cmd), float(w_cmd), self.dt)
        ref_status = self.referee.update(
            gt_pose["x_mm"],
            gt_pose["y_mm"],
            gt_pose["yaw"],
            gt_pose["v"],
            self.dt,
            gt_pose["t_s"],
            self.renderer.world_props,
        )
        self.trajectory.append((gt_pose["x_mm"], gt_pose["y_mm"]))

        done = False
        done_reason = ""
        if ref_status["dnf"]:
            done = True
            done_reason = f"dnf:{ref_status['dnf_reason']}"
        elif ref_status["finished"]:
            done = True
            done_reason = "finish"
        elif self.max_time_s > 0.0 and gt_pose["t_s"] >= self.max_time_s:
            done = True
            done_reason = "timeout"

        img = self.renderer.render(
            gt_pose["cam_x_mm"],
            gt_pose["cam_y_mm"],
            gt_pose["cam_yaw"],
            blur=self.render_blur,
            gain=self.render_gain,
            color_cast=self.render_color_cast,
            glare=self.render_glare,
        )
        info = StepInfo(
            bgr_img=img,
            odom=odom,
            gt_pose=gt_pose,
            world_props=copy.deepcopy(self.renderer.world_props),
            referee_status=ref_status,
            done=done,
            done_reason=done_reason,
        )
        self.last_step_info = info
        return info

    def render_topdown(
        self,
        width: int = 640,
        height: int = 480,
        trajectory: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Render top-down global track map with props, 4-wheel footprint, 3-disc circles,
        trajectory trail, and live AIC scoreboard HUD.
        """
        W_out, H_out = int(width), int(height)
        canvas = np.full((H_out, W_out, 3), (28, 30, 34), dtype=np.uint8)

        hud_top = 48
        hud_bot = 54
        map_avail_h = max(60, H_out - hud_top - hud_bot)
        map_avail_w = max(80, W_out - 12)

        # Aspect-preserving fit of the track map
        src_h, src_w = self.renderer.map_bgr.shape[:2]
        scale = min(map_avail_w / float(src_w), map_avail_h / float(src_h))
        dst_w = max(1, int(round(src_w * scale)))
        dst_h = max(1, int(round(src_h * scale)))
        ox = (W_out - dst_w) // 2
        oy = hud_top + (map_avail_h - dst_h) // 2

        resized_map = cv2.resize(
            self.renderer.map_bgr, (dst_w, dst_h), interpolation=cv2.INTER_AREA
        )
        canvas[oy:oy + dst_h, ox:ox + dst_w] = resized_map

        mpp = self.renderer.mpp
        map_H = self.renderer.H

        def mm_to_canvas(x_mm: float, y_mm: float) -> Tuple[int, int]:
            px = x_mm / mpp
            py = (map_H - 1) - y_mm / mpp
            cx = int(round(ox + px * scale))
            cy = int(round(oy + py * scale))
            return cx, cy

        def mm_to_px_len(d_mm: float) -> float:
            return (d_mm / mpp) * scale

        # 1) Draw props cleanly on top-down map
        for prop in self.renderer.world_props:
            ptype = prop["type"]
            cx_i, cy_i = mm_to_canvas(float(prop["x_mm"]), float(prop["y_mm"]))
            if ptype == "yellow_stop_line":
                if prop.get("stopped_ok", False):
                    cv2.circle(canvas, (cx_i, cy_i), 7, (60, 230, 80), 2, cv2.LINE_AA)
                elif prop.get("triggered", False):
                    cv2.circle(canvas, (cx_i, cy_i), 8, (30, 40, 255), 2, cv2.LINE_AA)
            elif ptype == "red_cone":
                r_px = max(4, int(round(mm_to_px_len(float(prop["radius_mm"])))))
                col = (25, 25, 240) if not prop.get("triggered") else (0, 120, 255)
                cv2.circle(canvas, (cx_i, cy_i), r_px, col, -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx_i, cy_i), max(2, r_px // 2), (240, 240, 245), -1, cv2.LINE_AA)
                if prop.get("triggered"):
                    cv2.drawMarker(canvas, (cx_i, cy_i), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            elif ptype == "red_no_entry":
                corners = _obb_corners_2d(
                    float(prop["x_mm"]),
                    float(prop["y_mm"]),
                    0.5 * float(prop["length_mm"]),
                    0.5 * float(prop["width_mm"]),
                    float(prop["yaw_rad"]),
                )
                poly = np.array([mm_to_canvas(c[0], c[1]) for c in corners], dtype=np.int32)
                if prop.get("triggered"):
                    overlay = canvas.copy()
                    cv2.fillPoly(overlay, [poly], (30, 30, 235))
                    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
                cv2.polylines(canvas, [poly], True, (25, 25, 245), 2, cv2.LINE_AA)
            elif ptype == "digit":
                r_px = max(6, int(round(mm_to_px_len(float(prop["radius_mm"])))))
                ring_col = (50, 235, 90) if prop.get("collected") else (220, 160, 40)
                cv2.circle(canvas, (cx_i, cy_i), r_px + 2, ring_col, 2, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    str(prop["value"]),
                    (cx_i - 4, cy_i + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 2) Trajectory trail
        traj = self.trajectory if trajectory is None else trajectory
        if traj and len(traj) >= 2:
            pts = np.array([mm_to_canvas(p[0], p[1]) for p in traj], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (230, 215, 40), 2, cv2.LINE_AA)

        # 3) Vehicle 3-disc circles, OBB body, heading arrow, and 4 exact wheels
        gt = self.dynamics.get_gt_pose()
        ref = self.referee.get_status()
        geom = self.referee.compute_vehicle_geometry(gt["x_mm"], gt["y_mm"], gt["yaw"])

        # 3-disc circles
        r_disc_px = max(3, int(round(mm_to_px_len(geom["disc_radius_mm"]))))
        for dcx, dcy in geom["disc_centers_mm"]:
            cv2.circle(canvas, mm_to_canvas(dcx, dcy), r_disc_px, (180, 235, 90), 1, cv2.LINE_AA)

        # Chassis OBB
        body_poly = np.array(
            [mm_to_canvas(c[0], c[1]) for c in geom["corners_mm"]], dtype=np.int32
        )
        cv2.polylines(canvas, [body_poly], True, (30, 230, 255), 2, cv2.LINE_AA)

        # Heading arrow (rear axle -> front bumper center)
        p_rear = mm_to_canvas(gt["x_mm"], gt["y_mm"])
        fb_c = geom["front_bumper_pts_mm"][1]
        p_front = mm_to_canvas(fb_c[0], fb_c[1])
        cv2.arrowedLine(canvas, p_rear, p_front, (0, 240, 255), 2, tipLength=0.35)

        # 4 Wheels (green = inside lane, bright red = outside lane)
        w_states = ref["wheel_states"]
        for wname, (wx, wy) in geom["wheels_mm"].items():
            wpt = mm_to_canvas(wx, wy)
            w_col = (30, 40, 255) if w_states.get(wname, False) else (40, 235, 80)
            cv2.circle(canvas, wpt, 4, w_col, -1, cv2.LINE_AA)
            cv2.circle(canvas, wpt, 4, (15, 15, 15), 1, cv2.LINE_AA)

        # 4) Top & bottom scoreboard HUD
        cv2.rectangle(canvas, (0, 0), (W_out, hud_top), (20, 22, 26), -1)
        cv2.rectangle(canvas, (0, H_out - hud_bot), (W_out, H_out), (20, 22, 26), -1)
        cv2.line(canvas, (0, hud_top), (W_out, hud_top), (65, 70, 80), 1)
        cv2.line(canvas, (0, H_out - hud_bot), (W_out, H_out - hud_bot), (65, 70, 80), 1)

        # Top row 1: Track + dynamics telemetry
        top_line1 = (
            f"TRACK: {self.track_name}  |  t={ref['lap_time_s']:.2f}s  "
            f"s={ref['s_m']:.2f}m  v={gt['v']:+.2f}m/s  W_out={ref['wheels_out']}/4"
        )
        cv2.putText(
            canvas, top_line1, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 240, 245), 1, cv2.LINE_AA
        )

        # Top row 2: Net score summary + status badge
        top_line2 = (
            f"NET: {ref['net_score_s']:.2f}s  "
            f"(Time {ref['lap_time_s']:.2f} + Pen {ref['total_penalty_s']:.2f} - Bonus {ref['total_bonus_s']:.1f})"
        )
        net_col = (80, 240, 110) if ref["total_penalty_s"] < 1e-6 else (80, 190, 255)
        cv2.putText(
            canvas, top_line2, (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.48, net_col, 1, cv2.LINE_AA
        )

        # Right-aligned status badge
        if ref["dnf"]:
            badge, bcol = "DNF", (30, 40, 235)
        elif ref["finished"]:
            badge, bcol = "FINISH", (40, 190, 60)
        else:
            badge, bcol = "RUN", (180, 120, 35)
        cv2.rectangle(canvas, (W_out - 86, 8), (W_out - 8, 38), bcol, -1)
        cv2.putText(
            canvas, badge, (W_out - 76, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Bottom row 1: Detailed penalty breakdown
        bot_line1 = (
            f"PENALTY  Line:{ref['boundary_penalty_s']:.1f}s ({ref['boundary_press_time_s']:.2f}s)  "
            f"Cone:{ref['cone_hits']} (+{ref['cone_penalty_s']:.0f}s)  "
            f"NoEnt:{ref['no_entry_hits']} (+{ref['no_entry_penalty_s']:.0f}s)  "
            f"YelViol:{ref['yellow_violations']} (+{ref['yellow_penalty_s']:.0f}s)"
        )
        cv2.putText(
            canvas, bot_line1, (10, H_out - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (215, 220, 230), 1, cv2.LINE_AA
        )

        # Bottom row 2: Bonus & yellow stops
        digits_str = ",".join(str(d) for d in ref["bonus_digits_collected"]) or "-"
        bot_line2 = (
            f"BONUS: [{digits_str}] (-{ref['total_bonus_s']:.0f}s)  |  "
            f"YelStopOK: {ref['yellow_stops_ok']}  |  Vbat: {self.dynamics.voltage:.1f}V"
        )
        cv2.putText(
            canvas, bot_line2, (10, H_out - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 235, 190), 1, cv2.LINE_AA
        )

        return canvas


# ---------------------------------------------------------------------------
# Built-in Self-Test Suite
# ---------------------------------------------------------------------------

def _run_selftest() -> None:
    print("=== sim_world_pc.py self-test ===")

    # 1) 3D standing cone projection check (must rise above ground base -> vertical height > 40px)
    world = PCSimWorld(track="fork_s", rate_hz=30.0)
    info = world.reset(
        pose={"s_m": 1.0, "lat_mm": 0.0},
        props=[
            {"type": "red_cone", "s_m": 1.65, "lat_mm": 0.0, "id": "test_cone"},
            {"type": "red_no_entry", "s_m": 2.0, "lat_mm": 100.0, "id": "test_ne"},
            {"type": "digit", "s_m": 1.55, "lat_mm": -90.0, "value": 5, "id": "test_d5"},
        ],
    )
    img = info.bgr_img
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    cone_prop = next(p for p in world.world_props if p["id"] == "test_cone")
    bbox = world.renderer.project_prop_bbox_uv(
        cone_prop, info.gt_pose["cam_x_mm"], info.gt_pose["cam_y_mm"], info.gt_pose["cam_yaw"]
    )
    assert bbox is not None
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    assert bh > 45.0, f"Expected tall 3D standing cone, got bh={bh:.1f}px"
    # Verify actual red pixels and white collar pixels appear inside projected cone bbox
    x1, y1, x2, y2 = map(int, bbox)
    crop = img[y1:y2, x1:x2]
    red_px = ((crop[:, :, 2].astype(int) - crop[:, :, 0].astype(int)) > 100).sum()
    assert red_px > 200, f"Expected red cone pixels in bbox, got {red_px}"
    print(f"[PASS] 3D cone projection: bbox={tuple(round(v,1) for v in bbox)} w={bw:.1f} h={bh:.1f} red_px={red_px}")

    # 2) Reverse-brake physics: holding negative v_cmd after v crosses 0 causes actual reverse!
    dyn = VehicleDynamics(a_eff=1.44, latency_ms=0.0)
    dyn.reset(0.0, 0.0, 0.0, v=0.40)
    # Brake with v_cmd=-0.25 for 0.8s -> should cross 0 around 0.28s and reverse!
    vs = []
    for _ in range(24):
        gt, _ = dyn.step(-0.25, 0.0, 1.0 / 30.0)
        vs.append(gt["v"])
    assert min(vs) < -0.05, f"Expected actual reverse when holding negative v_cmd, got min_v={min(vs):.4f}"
    # Release cleanly at v <= 0.10 -> zero reverse!
    dyn.reset(0.0, 0.0, 0.0, v=0.40)
    min_clean_v = 1.0
    for _ in range(30):
        cmd_v = -0.25 if dyn.v > 0.10 else 0.0
        gt, _ = dyn.step(cmd_v, 0.0, 1.0 / 30.0)
        min_clean_v = min(min_clean_v, gt["v"])
    assert min_clean_v >= -0.005, f"Clean brake release reversed: min_v={min_clean_v:.4f}"
    print(f"[PASS] Reverse-brake physics: hold_neg min_v={min(vs):.3f}m/s, clean_release min_v={min_clean_v:.4f}m/s")

    # 3) Referee: cone (+10s), no-entry (+30s), bonus digit (-5*v), yellow violation (+15s) & clean stop (0s)
    w_ref = PCSimWorld(track="fork_s", rate_hz=30.0, clear_base_yellow=True)
    # 3a) Cone hit + no-entry hit + digit 6 collect
    w_ref.reset(
        pose={"s_m": 1.0, "lat_mm": 0.0, "v": 0.50},
        props=[
            {"type": "red_cone", "s_m": 1.30, "lat_mm": 0.0, "id": "c1"},
            {"type": "red_no_entry", "s_m": 1.60, "lat_mm": 0.0, "id": "ne1"},
            {"type": "digit", "s_m": 1.90, "lat_mm": 0.0, "value": 6, "id": "d6"},
        ],
    )
    for _ in range(85):
        si = w_ref.step(0.50, 0.0)
    rs = si.referee_status
    assert rs["cone_hits"] == 1 and abs(rs["cone_penalty_s"] - 10.0) < 1e-6
    assert rs["no_entry_hits"] == 1 and abs(rs["no_entry_penalty_s"] - 30.0) < 1e-6
    assert rs["bonus_digits_collected"] == [6] and abs(rs["total_bonus_s"] - 30.0) < 1e-6
    expected_net = rs["lap_time_s"] + rs["total_penalty_s"] - rs["total_bonus_s"]
    assert abs(rs["net_score_s"] - expected_net) < 1e-3
    print(f"[PASS] Referee cone(+10)/no_entry(+30)/digit6(-30): net={rs['net_score_s']:.2f}s")

    # 3b) Yellow stop line: clean stop before touching -> 0 penalty; running through -> +15s
    w_ref.reset(
        pose={"s_m": 1.0, "lat_mm": 0.0, "v": 0.30},
        props=[{"type": "yellow_stop_line", "s_m": 1.50, "lat_mm": 0.0, "id": "y1"}],
    )
    # Approach then stop cleanly before 1.50m (front bumper starts at 1.18m, gap 0.32m)
    for _ in range(45):
        v_now = w_ref.dynamics.v
        cmd = -0.25 if v_now > 0.06 else 0.0
        si = w_ref.step(cmd, 0.0)
    assert si.referee_status["yellow_stops_ok"] == 1
    assert si.referee_status["yellow_violations"] == 0
    # Now drive across it -> still 0 penalty!
    for _ in range(45):
        si = w_ref.step(0.45, 0.0)
    assert si.referee_status["yellow_violations"] == 0
    print(f"[PASS] Referee yellow clean stop: stops_ok={si.referee_status['yellow_stops_ok']}, viol=0")

    # Running straight through yellow without stopping -> +15s
    w_ref.reset(
        pose={"s_m": 1.0, "lat_mm": 0.0, "v": 0.45},
        props=[{"type": "yellow_stop_line", "s_m": 1.50, "lat_mm": 0.0, "id": "y_viol"}],
    )
    for _ in range(35):
        si = w_ref.step(0.45, 0.0)
    assert si.referee_status["yellow_violations"] == 1
    assert abs(si.referee_status["yellow_penalty_s"] - 15.0) < 1e-6
    print(f"[PASS] Referee yellow violation: viol=1 (+15s)")

    # 3c) Boundary press (+15s/s) and >= 3 wheels DNF
    w_ref.reset(pose={"s_m": 1.0, "lat_mm": 160.0, "v": 0.0}, props=[])
    si = w_ref.step(0.0, 0.0)
    assert si.referee_status["wheels_out"] in (1, 2)
    assert si.referee_status["boundary_penalty_s"] > 0.0 and not si.referee_status["dnf"]
    w_ref.reset(pose={"s_m": 1.0, "lat_mm": 265.0, "yaw_offset": 0.45, "v": 0.0}, props=[])
    si3 = w_ref.step(0.0, 0.0)
    assert si3.referee_status["wheels_out"] == 3 and si3.referee_status["dnf"] and si3.done
    w_ref.reset(pose={"s_m": 1.0, "lat_mm": 360.0, "v": 0.0}, props=[])
    si = w_ref.step(0.0, 0.0)
    assert si.referee_status["wheels_out"] == 4 and si.referee_status["dnf"] and si.done
    print(f"[PASS] Referee boundary press & 3/4-wheel DNF: w3={si3.referee_status['wheels_out']} w4={si.referee_status['wheels_out']} done_reason={si.done_reason}")

    # 4) Top-down visualizer
    td = world.render_topdown(640, 480)
    assert td.shape == (480, 640, 3) and td.dtype == np.uint8
    print(f"[PASS] Top-down visualizer: shape={td.shape}")
    print("=== ALL SELF-TESTS PASSED ===")


if __name__ == "__main__":
    _run_selftest()
