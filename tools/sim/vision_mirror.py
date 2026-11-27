#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1:1 Python OpenCV Vision Frontend Mirror & Dual-Mode Detector for AIC 2026 PC Simulator.

Mirrors the real-car C++ vision pipeline verbatim:
  - `ImageProcessor` (`src/line_follow_ws/src/line_follower/src/image_processor.cpp`):
      `relative_border`, `hybrid`, and `legacy_hsv` lane segmentation,
      semantic hole filling (digit cards / yellow bars + interior closed holes),
      independent yellow ROI (`yellow_roi_bottom_ratio`), geometric & YOLO-hybrid
      yellow bar extraction, and ahead-strip dead-end blue/dark ratio.
  - `LaneGeometry` (`src/line_follow_ws/src/line_follower/src/lane_geometry.cpp`):
      far/near band column scanning, multi-run fork bimodal split detection,
      width-adaptive fusion, wide-junction heading continuity, EMA observation filter,
      and continuous curve assist.
  - `GroundProjector` (`src/line_follow_ws/src/line_follower/src/ground_projector.cpp`):
      calibrated 3x3 homography + pinhole fallback (`pixel_to_ground`, `ground_to_pixel`).
  - `detect_cv_red_in_lane` & single-side visible near-band boundary clamp
      (`src/line_follow_ws/src/line_follower/src/line_follower_node.cpp`).

Also provides dual-mode switchable object detectors:
  - `RealYoloDetector`: runs real `ultralytics.YOLO` (CUDA GPU with graceful CPU fallback)
      returning detections matching `bpu_yolov8_digit_node` + persistent `SimpleBoxTracker`.
  - `MockNoisyDetector`: fast statistical mock detector projecting ground-truth `world_props`
      via `CameraModel` + vehicle pose `(x_mm, y_mm, yaw)` with realistic distance-dependent
      recall, bbox pixel jitter, confidence noise, and optional brief dropout/flicker.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import yaml

try:
    from .render import CameraModel
except Exception:
    try:
        from tools.sim.render import CameraModel  # type: ignore
    except Exception:
        from render import CameraModel  # type: ignore

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_WEIGHTS = (
    REPO_ROOT / "runs" / "detect" / "runs" / "digits8" / "yolov8n_digits8_v3" / "weights" / "best.pt"
)
DEFAULT_OUT_LIST = REPO_ROOT / "src" / "yolov8" / "mod1" / "out.list"

SEMANTIC_UNKNOWN = "unknown"
SEMANTIC_RED_CONE = "red_cone"
SEMANTIC_RED_NO_ENTRY = "red_no_entry"
SEMANTIC_YELLOW_STOP_LINE = "yellow_stop_line"
SEMANTIC_BONUS_DIGIT = "bonus_digit"


def _clamp(val: float, lo: float, hi: float) -> float:
    return lo if val < lo else (hi if val > hi else val)


def _clamp_int(val: int, lo: int, hi: int) -> int:
    return lo if val < lo else (hi if val > hi else val)


def _rect_intersect(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return (0, 0, 0, 0)
    return (x1, y1, w, h)


def _rect_union(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def classify_detection_label(label: str) -> Tuple[str, int]:
    """Exact mirror of `LineFollowerNode::classify_detection_label`."""
    if not label:
        return SEMANTIC_UNKNOWN, 0
    lower = str(label).strip().lower()
    if lower in ("red_cone", "cone"):
        return SEMANTIC_RED_CONE, 0
    if lower in ("red_no_entry", "no_entry", "forbid", "forbidden"):
        return SEMANTIC_RED_NO_ENTRY, 0
    if lower in ("yellow_stop_line", "yellow_line", "stop_line"):
        return SEMANTIC_YELLOW_STOP_LINE, 0

    num_part = lower
    if num_part.startswith("digit_"):
        num_part = num_part[6:]
    elif len(num_part) >= 2 and num_part[0] == "d":
        num_part = num_part[1:]
    if len(num_part) == 1 and "1" <= num_part <= "8":
        return SEMANTIC_BONUS_DIGIT, ord(num_part) - ord("0")

    return SEMANTIC_UNKNOWN, 0


def unwrap_ros_params(params: Optional[Union[Dict[str, Any], str, Path]]) -> Dict[str, Any]:
    """Accept flat dict, parsed ROS2 YAML dict, or YAML file path and return flat params."""
    if params is None:
        return {}
    if isinstance(params, (str, Path)):
        with open(params, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return unwrap_ros_params(data)
    if not isinstance(params, dict):
        return {}
    if "line_follower_node" in params and isinstance(params["line_follower_node"], dict):
        inner = params["line_follower_node"]
        if "ros__parameters" in inner and isinstance(inner["ros__parameters"], dict):
            return dict(inner["ros__parameters"])
    if "ros__parameters" in params and isinstance(params["ros__parameters"], dict):
        return dict(params["ros__parameters"])
    return dict(params)


# ---------------------------------------------------------------------------
# SimpleBoxTracker (matches src/yolov8/bpu_yolov8_node.py)
# ---------------------------------------------------------------------------


class SimpleBoxTracker:
    """Lightweight per-class IoU + centroid distance tracker matching `bpu_yolov8_node.py`."""

    def __init__(
        self,
        iou_thresh: float = 0.30,
        max_center_dist_px: float = 80.0,
        max_missed_frames: int = 5,
    ) -> None:
        self.iou_thresh = float(max(0.0, iou_thresh))
        self.max_center_dist_px = float(max(1.0, max_center_dist_px))
        self.max_missed_frames = int(max(0, max_missed_frames))
        self._next_track_id = 1
        self._tracks: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _box_iou(
        a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
    ) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        union = max(1e-6, aw * ah + bw * bh - inter)
        return float(inter / union)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out_dets = [dict(d) for d in detections]
        det_indices_by_cls: Dict[str, List[int]] = {}
        for idx, det in enumerate(out_dets):
            cls_key = str(det.get("class_name", det.get("type", "")))
            det_indices_by_cls.setdefault(cls_key, []).append(idx)

        matched_track_ids = set()
        matched_det_indices = set()

        for cls_name, det_indices in det_indices_by_cls.items():
            cls_tracks = [
                (tid, trk) for tid, trk in self._tracks.items() if trk["type"] == cls_name
            ]
            if not cls_tracks:
                continue

            candidates: List[Tuple[float, int, int]] = []
            for tid, trk in cls_tracks:
                t_box = (trk["bx"], trk["by"], trk["bw"], trk["bh"])
                tcx, tcy = trk["cx"], trk["cy"]
                for det_idx in det_indices:
                    det = out_dets[det_idx]
                    d_box = (
                        float(det["x1"]),
                        float(det["y1"]),
                        float(det["x2"] - det["x1"]),
                        float(det["y2"] - det["y1"]),
                    )
                    iou = self._box_iou(t_box, d_box)
                    if iou >= self.iou_thresh:
                        candidates.append((2.0 + iou, tid, det_idx))
                    else:
                        dist = math.hypot(float(det["cx"]) - tcx, float(det["cy"]) - tcy)
                        if dist <= self.max_center_dist_px:
                            candidates.append((1.0 - dist / self.max_center_dist_px, tid, det_idx))

            candidates.sort(key=lambda item: item[0], reverse=True)
            for _score, tid, det_idx in candidates:
                if tid in matched_track_ids or det_idx in matched_det_indices:
                    continue
                matched_track_ids.add(tid)
                matched_det_indices.add(det_idx)
                det = out_dets[det_idx]
                trk = self._tracks[tid]
                trk["bx"] = float(det["x1"])
                trk["by"] = float(det["y1"])
                trk["bw"] = float(det["x2"] - det["x1"])
                trk["bh"] = float(det["y2"] - det["y1"])
                trk["cx"] = float(det["cx"])
                trk["cy"] = float(det["cy"])
                trk["missed"] = 0
                trk["hits"] += 1
                det["track_id"] = int(tid)
                det["track_hits"] = int(trk["hits"])

        for idx, det in enumerate(out_dets):
            if idx in matched_det_indices:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            cls_name = str(det.get("class_name", det.get("type", "")))
            self._tracks[tid] = {
                "type": cls_name,
                "bx": float(det["x1"]),
                "by": float(det["y1"]),
                "bw": float(det["x2"] - det["x1"]),
                "bh": float(det["y2"] - det["y1"]),
                "cx": float(det["cx"]),
                "cy": float(det["cy"]),
                "missed": 0,
                "hits": 1,
            }
            det["track_id"] = int(tid)
            det["track_hits"] = 1

        expired = []
        for tid, trk in self._tracks.items():
            if tid not in matched_track_ids:
                trk["missed"] += 1
                if trk["missed"] > self.max_missed_frames:
                    expired.append(tid)
        for tid in expired:
            del self._tracks[tid]

        return out_dets


# ---------------------------------------------------------------------------
# GroundProjector (exact mirror of ground_projector.cpp)
# ---------------------------------------------------------------------------


@dataclass
class GroundProjectorConfig:
    use_homography: bool = False
    homography_matrix: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    cam_height_m: float = 0.11
    cam_pitch_deg: float = 12.5
    focal_px: float = 325.0
    cx: float = 320.0
    cy: float = 240.0
    front_overhang_m: float = 0.06
    image_width: int = 640
    image_height: int = 480
    min_valid_d_m: float = -0.5
    max_valid_d_m: float = 6.0


def _invert_3x3(m: Sequence[float]) -> Optional[Tuple[float, ...]]:
    if len(m) != 9:
        return None
    if any(not math.isfinite(float(v)) for v in m):
        return None
    m0, m1, m2, m3, m4, m5, m6, m7, m8 = [float(v) for v in m]
    c00 = m4 * m8 - m5 * m7
    c01 = -(m3 * m8 - m5 * m6)
    c02 = m3 * m7 - m4 * m6
    c10 = -(m1 * m8 - m2 * m7)
    c11 = m0 * m8 - m2 * m6
    c12 = -(m0 * m7 - m1 * m6)
    c20 = m1 * m5 - m2 * m4
    c21 = -(m0 * m5 - m2 * m3)
    c22 = m0 * m4 - m1 * m3
    det = m0 * c00 + m1 * c01 + m2 * c02
    if not math.isfinite(det) or abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    inv = (
        c00 * inv_det,
        c10 * inv_det,
        c20 * inv_det,
        c01 * inv_det,
        c11 * inv_det,
        c21 * inv_det,
        c02 * inv_det,
        c12 * inv_det,
        c22 * inv_det,
    )
    if any(not math.isfinite(v) for v in inv):
        return None
    return inv


class GroundProjectorMirror:
    """1:1 mirror of `line_follower::GroundProjector`."""

    def __init__(self, config: Optional[GroundProjectorConfig] = None) -> None:
        self.config = GroundProjectorConfig()
        self._sin_pitch = 0.0
        self._cos_pitch = 1.0
        self._homography_valid = False
        self._inv_homography: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self.set_config(config or GroundProjectorConfig())

    def set_config(self, config: GroundProjectorConfig) -> None:
        cfg = copy.copy(config)
        cfg.cam_height_m = max(0.01, float(cfg.cam_height_m))
        cfg.cam_pitch_deg = _clamp(float(cfg.cam_pitch_deg), -45.0, 85.0)
        cfg.focal_px = max(1.0, float(cfg.focal_px))
        cfg.image_width = max(1, int(cfg.image_width))
        cfg.image_height = max(1, int(cfg.image_height))
        if cfg.max_valid_d_m < cfg.min_valid_d_m:
            cfg.min_valid_d_m, cfg.max_valid_d_m = cfg.max_valid_d_m, cfg.min_valid_d_m
        self.config = cfg
        pitch_rad = math.radians(cfg.cam_pitch_deg)
        self._sin_pitch = math.sin(pitch_rad)
        self._cos_pitch = math.cos(pitch_rad)
        inv = _invert_3x3(cfg.homography_matrix)
        self._homography_valid = inv is not None
        if inv is not None:
            self._inv_homography = inv

    def set_homography_from_vector(self, h9: Sequence[float], enable: bool = True) -> bool:
        if len(h9) != 9:
            return False
        inv = _invert_3x3(h9)
        if inv is None:
            return False
        self.config.homography_matrix = tuple(float(v) for v in h9)
        self.config.use_homography = bool(enable)
        self._inv_homography = inv
        self._homography_valid = True
        return True

    def using_homography(self) -> bool:
        return bool(self.config.use_homography and self._homography_valid)

    def pixel_to_ground(self, u: float, v: float) -> Tuple[bool, float, float]:
        if not (math.isfinite(u) and math.isfinite(v)):
            return False, 0.0, 0.0
        cfg = self.config
        if self.using_homography():
            h = cfg.homography_matrix
            w = h[6] * u + h[7] * v + h[8]
            if not math.isfinite(w) or abs(w) < 1e-9:
                return False, 0.0, 0.0
            d = (h[0] * u + h[1] * v + h[2]) / w
            y = (h[3] * u + h[4] * v + h[5]) / w
            if not (math.isfinite(d) and math.isfinite(y)):
                return False, 0.0, 0.0
            if d < cfg.min_valid_d_m or d > cfg.max_valid_d_m:
                return False, 0.0, 0.0
            return True, d, y

        x_n = (u - cfg.cx) / cfg.focal_px
        y_n = (v - cfg.cy) / cfg.focal_px
        denom = self._sin_pitch + y_n * self._cos_pitch
        if not math.isfinite(denom) or denom <= 1e-6:
            return False, 0.0, 0.0
        d_cam_m = cfg.cam_height_m * (self._cos_pitch - y_n * self._sin_pitch) / denom
        y = -cfg.cam_height_m * x_n / denom
        d = d_cam_m - cfg.front_overhang_m
        if not (math.isfinite(d_cam_m) and math.isfinite(d) and math.isfinite(y)):
            return False, 0.0, 0.0
        if d_cam_m <= 0.0 or d < cfg.min_valid_d_m or d > cfg.max_valid_d_m:
            return False, 0.0, 0.0
        return True, d, y

    def ground_to_pixel(self, d_bumper_m: float, y_lat_m: float) -> Tuple[bool, float, float]:
        if not (math.isfinite(d_bumper_m) and math.isfinite(y_lat_m)):
            return False, 0.0, 0.0
        cfg = self.config
        if self.using_homography():
            hi = self._inv_homography
            w = hi[6] * d_bumper_m + hi[7] * y_lat_m + hi[8]
            if not math.isfinite(w) or abs(w) < 1e-9:
                return False, 0.0, 0.0
            px_u = (hi[0] * d_bumper_m + hi[1] * y_lat_m + hi[2]) / w
            px_v = (hi[3] * d_bumper_m + hi[4] * y_lat_m + hi[5]) / w
            if not (math.isfinite(px_u) and math.isfinite(px_v)):
                return False, 0.0, 0.0
            return True, px_u, px_v

        d_cam_m = d_bumper_m + cfg.front_overhang_m
        x_c = -y_lat_m
        y_c = -d_cam_m * self._sin_pitch + cfg.cam_height_m * self._cos_pitch
        z_c = d_cam_m * self._cos_pitch + cfg.cam_height_m * self._sin_pitch
        if not math.isfinite(z_c) or z_c <= 1e-6:
            return False, 0.0, 0.0
        px_u = cfg.cx + cfg.focal_px * (x_c / z_c)
        px_v = cfg.cy + cfg.focal_px * (y_c / z_c)
        if not (math.isfinite(px_u) and math.isfinite(px_v)):
            return False, 0.0, 0.0
        return True, px_u, px_v


# ---------------------------------------------------------------------------
# ImageProcessor (exact 1:1 mirror of image_processor.hpp / .cpp)
# ---------------------------------------------------------------------------


@dataclass
class ImageProcessorConfig:
    roi_top_ratio: float = 0.68
    roi_bottom_ratio: float = 0.18
    yellow_roi_bottom_ratio: float = 0.0

    gaussian_kernel: int = 5
    binary_threshold: float = 160.0
    morph_kernel: int = 3
    morph_iterations: int = 2
    min_contour_area: float = 2500.0
    invert_binary: bool = False

    use_white_filter: bool = True
    white_h_low: int = 0
    white_h_high: int = 180
    white_s_max: int = 45
    white_v_min: int = 80

    use_yellow_filter: bool = True
    yellow_h_low: int = 20
    yellow_h_high: int = 40
    yellow_s_min: int = 80
    yellow_v_min: int = 120

    lane_seg_mode: str = "relative_border"
    blue_margin: int = 22
    lab_blue_b_max: int = 116
    black_block_size: int = 31
    black_c_offset: float = 14.0
    black_v_max: int = 115
    lab_chroma_radius: float = 24.0
    lab_l_min_offset: float = 38.0
    hybrid_white_s_max: int = 75
    hybrid_white_v_min: int = 65

    enable_semantic_hole_fill: bool = True
    semantic_hole_fill_dilate_px: int = 2

    yellow_detect_mode: str = "geometric_cv"
    yellow_min_aspect_ratio: float = 2.5
    yellow_min_width_ratio: float = 0.20
    yellow_min_area_px: float = 120.0
    yellow_ahead_strip_height_px: int = 28
    yellow_dead_end_blue_ratio: float = 0.35
    yellow_dead_end_dark_ratio: float = 0.20
    yellow_dead_end_combined_ratio: float = 0.50


@dataclass
class SegmentedFrame:
    roi_bgr: np.ndarray
    binary_mask: np.ndarray
    roi_top: int = 0
    roi_bottom: int = 0
    frame_width: int = 0
    frame_height: int = 0
    yellow_line_detected: bool = False
    contour_valid: bool = False
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    contour_centroid: Tuple[float, float] = (0.0, 0.0)
    yellow_is_geometric: bool = False
    yellow_bottom_y_full: float = 0.0
    yellow_top_y_full: float = 0.0
    yellow_bar_width_px: float = 0.0
    yellow_center_x_full: float = 0.0
    yellow_bbox_full: Tuple[int, int, int, int] = (0, 0, 0, 0)
    yellow_from_yolo_hybrid: bool = False
    yellow_ahead_blue_ratio: float = 0.0
    yellow_ahead_dark_ratio: float = 0.0
    yellow_has_dead_end_ahead: bool = False
    active_lane_seg_mode: str = "relative_border"


def _cpp_nth_median(arr: np.ndarray, fallback: float) -> float:
    """Exact match to C++ `std::nth_element` at `values.size() / 2`."""
    n = int(arr.size)
    if n == 0:
        return fallback
    mid = n // 2
    return float(np.partition(arr, mid)[mid])


def _find_lowest_yellow_row(yellow_mask: np.ndarray, box: Tuple[int, int, int, int]) -> int:
    h, w = yellow_mask.shape[:2]
    cx, cy, cw, ch = _rect_intersect(box, (0, 0, w, h))
    if cw <= 0 or ch <= 0:
        return box[1] + box[3] - 1
    min_row_pixels = max(3, cw // 4)
    sub = yellow_mask[cy : cy + ch, cx : cx + cw]
    counts = np.count_nonzero(sub, axis=1)
    hits = np.flatnonzero(counts >= min_row_pixels)
    if hits.size > 0:
        return int(cy + hits[-1])
    return cy + ch - 1


def _find_highest_yellow_row(yellow_mask: np.ndarray, box: Tuple[int, int, int, int]) -> int:
    h, w = yellow_mask.shape[:2]
    cx, cy, cw, ch = _rect_intersect(box, (0, 0, w, h))
    if cw <= 0 or ch <= 0:
        return box[1]
    min_row_pixels = max(3, cw // 4)
    sub = yellow_mask[cy : cy + ch, cx : cx + cw]
    counts = np.count_nonzero(sub, axis=1)
    hits = np.flatnonzero(counts >= min_row_pixels)
    if hits.size > 0:
        return int(cy + hits[0])
    return cy


def _build_relative_border_mask(blurred_bgr: np.ndarray, cfg: ImageProcessorConfig) -> np.ndarray:
    """Vectorized 1:1 exact mirror of `build_relative_border_mask` in `image_processor.cpp`."""
    rows, cols = blurred_bgr.shape[:2]
    lab = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2Lab)
    l_chan = lab[:, :, 0]

    local_dark = cv2.adaptiveThreshold(
        l_chan,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        cfg.black_block_size,
        cfg.black_c_offset,
    )

    b = blurred_bgr[:, :, 0].astype(np.int32)
    g = blurred_bgr[:, :, 1].astype(np.int32)
    r = blurred_bgr[:, :, 2].astype(np.int32)
    l_val = lab[:, :, 0].astype(np.int32)
    b_lab = lab[:, :, 2].astype(np.int32)

    max_rg = np.maximum(g, r)
    blue_by_bgr = ((b - max_rg) > cfg.blue_margin) & (b_lab < 126)
    blue_by_lab = (b_lab < cfg.lab_blue_b_max) & (b > max_rg)
    is_blue = blue_by_bgr | blue_by_lab
    is_dark = (l_val < 45) | ((local_dark > 0) & (l_val <= cfg.black_v_max))
    non_blue_non_dark = (~is_blue) & (~is_dark)

    # Dynamic intra-frame Lab white reference (exact sample bounds & stride=2)
    sample_y0 = _clamp_int(int(rows * 0.55), 0, max(0, rows - 1))
    sample_y1 = _clamp_int(int(rows * 0.95), min(rows, sample_y0 + 1), rows)
    sample_x0 = _clamp_int(int(cols * 0.25), 0, max(0, cols - 1))
    sample_x1 = _clamp_int(int(cols * 0.75), min(cols, sample_x0 + 1), cols)

    ok_sub = non_blue_non_dark[sample_y0:sample_y1:2, sample_x0:sample_x1:2]
    lab_sub = lab[sample_y0:sample_y1:2, sample_x0:sample_x1:2]
    l_sub = lab_sub[:, :, 0].astype(np.int32)
    a_sub = lab_sub[:, :, 1].astype(np.int32)
    b_sub = lab_sub[:, :, 2].astype(np.int32)

    valid_ref = ok_sub & (np.abs(a_sub - 128) <= 30) & ((b_sub - 128) <= 40)
    n_ref = int(np.count_nonzero(valid_ref))

    med_l, med_a, med_b = 175.0, 128.0, 128.0
    if n_ref >= 40:
        med_l = _clamp(_cpp_nth_median(l_sub[valid_ref].astype(np.float64), 175.0), 85.0, 245.0)
        med_a = _clamp(_cpp_nth_median(a_sub[valid_ref].astype(np.float64), 128.0), 112.0, 144.0)
        med_b = _clamp(_cpp_nth_median(b_sub[valid_ref].astype(np.float64), 128.0), 114.0, 152.0)

    chroma_r2 = cfg.lab_chroma_radius * cfg.lab_chroma_radius
    min_l = max(55.0, med_l - cfg.lab_l_min_offset)

    da = lab[:, :, 1].astype(np.float64) - med_a
    db = lab[:, :, 2].astype(np.float64) - med_b
    white_ok = non_blue_non_dark & (l_val >= min_l) & ((da * da + db * db) <= chroma_r2)
    return np.where(white_ok, np.uint8(255), np.uint8(0))


class ImageProcessorMirror:
    """1:1 Python/OpenCV mirror of `line_follower::ImageProcessor`."""

    def __init__(self, config: Optional[ImageProcessorConfig] = None) -> None:
        self.config = ImageProcessorConfig()
        self._morph_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.set_config(config or ImageProcessorConfig())

    def set_config(self, config: ImageProcessorConfig) -> None:
        cfg = copy.copy(config)
        cfg.roi_top_ratio = _clamp(float(cfg.roi_top_ratio), 0.0, 1.0)
        cfg.roi_bottom_ratio = _clamp(float(cfg.roi_bottom_ratio), 0.0, 0.5)
        cfg.yellow_roi_bottom_ratio = _clamp(float(cfg.yellow_roi_bottom_ratio), 0.0, 0.5)

        cfg.gaussian_kernel = int(cfg.gaussian_kernel)
        if cfg.gaussian_kernel <= 0:
            cfg.gaussian_kernel = 1
        if cfg.gaussian_kernel % 2 == 0:
            cfg.gaussian_kernel += 1

        cfg.morph_kernel = max(1, int(cfg.morph_kernel))
        cfg.morph_iterations = max(1, int(cfg.morph_iterations))

        cfg.white_h_low = _clamp_int(int(cfg.white_h_low), 0, 180)
        cfg.white_h_high = _clamp_int(int(cfg.white_h_high), 0, 180)
        if cfg.white_h_low > cfg.white_h_high:
            cfg.white_h_low, cfg.white_h_high = cfg.white_h_high, cfg.white_h_low
        cfg.white_s_max = _clamp_int(int(cfg.white_s_max), 0, 255)
        cfg.white_v_min = _clamp_int(int(cfg.white_v_min), 0, 255)

        cfg.yellow_h_low = _clamp_int(int(cfg.yellow_h_low), 0, 180)
        cfg.yellow_h_high = _clamp_int(int(cfg.yellow_h_high), 0, 180)
        if cfg.yellow_h_low > cfg.yellow_h_high:
            cfg.yellow_h_low, cfg.yellow_h_high = cfg.yellow_h_high, cfg.yellow_h_low
        cfg.yellow_s_min = _clamp_int(int(cfg.yellow_s_min), 0, 255)
        cfg.yellow_v_min = _clamp_int(int(cfg.yellow_v_min), 0, 255)

        if cfg.lane_seg_mode not in ("legacy_hsv", "relative_border", "hybrid"):
            cfg.lane_seg_mode = "relative_border"

        cfg.blue_margin = _clamp_int(int(cfg.blue_margin), 0, 255)
        cfg.lab_blue_b_max = _clamp_int(int(cfg.lab_blue_b_max), 0, 255)
        cfg.black_block_size = max(3, int(cfg.black_block_size))
        if cfg.black_block_size % 2 == 0:
            cfg.black_block_size += 1
        cfg.black_c_offset = float(cfg.black_c_offset)
        cfg.black_v_max = _clamp_int(int(cfg.black_v_max), 0, 255)
        cfg.lab_chroma_radius = max(5.0, float(cfg.lab_chroma_radius))
        cfg.lab_l_min_offset = _clamp(float(cfg.lab_l_min_offset), 5.0, 200.0)
        cfg.hybrid_white_s_max = _clamp_int(int(cfg.hybrid_white_s_max), 0, 255)
        cfg.hybrid_white_v_min = _clamp_int(int(cfg.hybrid_white_v_min), 0, 255)
        cfg.semantic_hole_fill_dilate_px = max(0, int(cfg.semantic_hole_fill_dilate_px))

        if cfg.yellow_detect_mode not in ("geometric_cv", "yolo_cv_hybrid"):
            cfg.yellow_detect_mode = "geometric_cv"
        cfg.yellow_min_aspect_ratio = max(1.0, float(cfg.yellow_min_aspect_ratio))
        cfg.yellow_min_width_ratio = _clamp(float(cfg.yellow_min_width_ratio), 0.01, 1.0)
        cfg.yellow_min_area_px = max(1.0, float(cfg.yellow_min_area_px))
        cfg.yellow_ahead_strip_height_px = max(4, int(cfg.yellow_ahead_strip_height_px))
        cfg.yellow_dead_end_blue_ratio = _clamp(float(cfg.yellow_dead_end_blue_ratio), 0.0, 1.0)
        cfg.yellow_dead_end_dark_ratio = _clamp(float(cfg.yellow_dead_end_dark_ratio), 0.0, 1.0)
        cfg.yellow_dead_end_combined_ratio = _clamp(
            float(cfg.yellow_dead_end_combined_ratio), 0.0, 1.0
        )

        self.config = cfg
        self._morph_element = cv2.getStructuringElement(
            cv2.MORPH_RECT, (cfg.morph_kernel, cfg.morph_kernel)
        )

    def segment(
        self,
        frame_bgr: np.ndarray,
        fill_white_rects: Sequence[Tuple[int, int, int, int]] = (),
        yolo_yellow_rects: Sequence[Tuple[int, int, int, int]] = (),
    ) -> Optional[SegmentedFrame]:
        if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None

        cfg = self.config
        height, width = frame_bgr.shape[:2]

        roi_top = _clamp_int(int(height * (1.0 - cfg.roi_top_ratio)), 0, height - 1)
        roi_bottom = _clamp_int(height - int(height * cfg.roi_bottom_ratio), roi_top + 1, height)
        roi_height = roi_bottom - roi_top
        roi_rect = (0, roi_top, width, roi_height)
        roi = frame_bgr[roi_top:roi_bottom, 0:width]

        blurred = cv2.GaussianBlur(roi, (cfg.gaussian_kernel, cfg.gaussian_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 1. Independent yellow stop line ROI [roi_top, yellow_roi_bottom]
        yellow_roi_bottom = _clamp_int(
            height - int(height * cfg.yellow_roi_bottom_ratio), roi_top + 1, height
        )
        yellow_roi_height = yellow_roi_bottom - roi_top
        yellow_roi_rect = (0, roi_top, width, yellow_roi_height)

        yellow_is_geometric = False
        yellow_from_yolo_hybrid = False
        yellow_bottom_y_full = 0.0
        yellow_top_y_full = 0.0
        yellow_bar_width_px = 0.0
        yellow_center_x_full = 0.0
        yellow_bbox_full: Tuple[int, int, int, int] = (0, 0, 0, 0)

        if cfg.use_yellow_filter:
            if yellow_roi_bottom == roi_bottom:
                yellow_hsv = hsv
            else:
                yellow_roi = frame_bgr[roi_top:yellow_roi_bottom, 0:width]
                yellow_blurred = cv2.GaussianBlur(
                    yellow_roi, (cfg.gaussian_kernel, cfg.gaussian_kernel), 0
                )
                yellow_hsv = cv2.cvtColor(yellow_blurred, cv2.COLOR_BGR2HSV)

            lower_yellow = np.array([cfg.yellow_h_low, cfg.yellow_s_min, cfg.yellow_v_min], dtype=np.uint8)
            upper_yellow = np.array([cfg.yellow_h_high, 255, 255], dtype=np.uint8)
            yellow_mask = cv2.inRange(yellow_hsv, lower_yellow, upper_yellow)
            yellow_pixels = cv2.countNonZero(yellow_mask)

            open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            yellow_clean = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, open_k, iterations=1)
            yellow_clean = cv2.morphologyEx(yellow_clean, cv2.MORPH_CLOSE, close_k, iterations=1)

            # YOLO-guided CV refinement
            if cfg.yellow_detect_mode == "yolo_cv_hybrid" and len(yolo_yellow_rects) > 0:
                best_yolo_idx = -1
                best_yolo_bottom = -1
                for i, r in enumerate(yolo_yellow_rects):
                    rx, ry, rw, rh = r
                    if rw <= 0 or rh <= 0:
                        continue
                    inter = _rect_intersect(r, yellow_roi_rect)
                    if inter[2] <= 0 or inter[3] <= 0:
                        continue
                    bot = ry + rh
                    if bot > best_yolo_bottom:
                        best_yolo_bottom = bot
                        best_yolo_idx = i

                if best_yolo_idx >= 0:
                    yx, yy, yw, yh = yolo_yellow_rects[best_yolo_idx]
                    px0 = _clamp_int(yx - 4, 0, width - 1)
                    py0 = _clamp_int(yy - 6, roi_top, yellow_roi_bottom - 1)
                    px1 = _clamp_int(yx + yw + 4, px0 + 1, width)
                    py1 = _clamp_int(yy + yh + 6, py0 + 1, yellow_roi_bottom)
                    local_box = (px0, py0 - roi_top, px1 - px0, py1 - py0)
                    local_clipped = _rect_intersect(local_box, (0, 0, width, yellow_roi_height))

                    yellow_from_yolo_hybrid = True
                    yellow_is_geometric = True

                    if local_clipped[2] > 0 and local_clipped[3] > 0:
                        lx, ly, lw, lh = local_clipped
                        local_nz = cv2.countNonZero(yellow_clean[ly : ly + lh, lx : lx + lw])
                        min_local = max(20, int(0.04 * (lw * lh)))
                        if local_nz >= min_local:
                            bot_roi = _find_lowest_yellow_row(yellow_clean, local_clipped)
                            top_roi = _find_highest_yellow_row(yellow_clean, local_clipped)
                            yellow_bottom_y_full = float(roi_top + bot_roi)
                            yellow_top_y_full = float(roi_top + top_roi)
                            yellow_bar_width_px = float(yw)
                            yellow_center_x_full = float(yx) + 0.5 * float(yw)
                            yellow_bbox_full = (
                                int(yx),
                                int(roi_top + top_roi),
                                int(yw),
                                max(1, int(bot_roi - top_roi + 1)),
                            )
                        else:
                            yellow_bottom_y_full = float(yy + yh - 1)
                            yellow_top_y_full = float(yy)
                            yellow_bar_width_px = float(yw)
                            yellow_center_x_full = float(yx) + 0.5 * float(yw)
                            yellow_bbox_full = (int(yx), int(yy), int(yw), int(yh))

            # Pure CV geometric horizontal yellow bar detection
            if not yellow_from_yolo_hybrid and yellow_pixels > 0:
                y_contours, _ = cv2.findContours(
                    yellow_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                best_bar_idx = -1
                best_bar_bottom = -1
                best_bar_area = 0.0
                min_w = cfg.yellow_min_width_ratio * float(width)

                for i, cnt in enumerate(y_contours):
                    area = float(cv2.contourArea(cnt))
                    if area < cfg.yellow_min_area_px:
                        continue
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    if bw <= 0 or bh <= 0:
                        continue
                    aspect = float(bw) / float(bh)
                    if aspect < cfg.yellow_min_aspect_ratio or float(bw) < min_w:
                        continue
                    bot = by + bh
                    if bot > best_bar_bottom or (bot == best_bar_bottom and area > best_bar_area):
                        best_bar_bottom = bot
                        best_bar_area = area
                        best_bar_idx = i

                if best_bar_idx >= 0:
                    bx, by, bw, bh = cv2.boundingRect(y_contours[best_bar_idx])
                    bot_roi = _find_lowest_yellow_row(yellow_clean, (bx, by, bw, bh))
                    top_roi = _find_highest_yellow_row(yellow_clean, (bx, by, bw, bh))
                    yellow_is_geometric = True
                    yellow_bottom_y_full = float(roi_top + bot_roi)
                    yellow_top_y_full = float(roi_top + top_roi)
                    yellow_bar_width_px = float(bw)
                    yellow_center_x_full = float(bx) + 0.5 * float(bw)
                    yellow_bbox_full = (
                        int(bx),
                        int(roi_top + top_roi),
                        int(bw),
                        max(1, int(bot_roi - top_roi + 1)),
                    )

        yellow_line_detected = (
            (yellow_from_yolo_hybrid or yellow_is_geometric)
            if cfg.yellow_detect_mode == "yolo_cv_hybrid"
            else yellow_is_geometric
        )

        # 2. Dead-end-ahead check above yellow bar
        yellow_ahead_blue_ratio = 0.0
        yellow_ahead_dark_ratio = 0.0
        yellow_has_dead_end_ahead = False

        if yellow_is_geometric and yellow_bbox_full[2] > 0:
            strip_bot = _clamp_int(yellow_bbox_full[1] - 2, 0, height)
            strip_top = _clamp_int(strip_bot - cfg.yellow_ahead_strip_height_px, 0, strip_bot)
            strip_x0 = _clamp_int(yellow_bbox_full[0], 0, width - 1)
            strip_x1 = _clamp_int(yellow_bbox_full[0] + yellow_bbox_full[2], strip_x0 + 1, width)

            if strip_bot > strip_top and strip_x1 > strip_x0:
                strip_bgr = frame_bgr[strip_top:strip_bot, strip_x0:strip_x1]
                strip_lab = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2Lab)
                total_px = int(strip_bgr.shape[0] * strip_bgr.shape[1])

                sb = strip_bgr[:, :, 0].astype(np.int32)
                sg = strip_bgr[:, :, 1].astype(np.int32)
                sr = strip_bgr[:, :, 2].astype(np.int32)
                sl = strip_lab[:, :, 0].astype(np.int32)
                sb_lab = strip_lab[:, :, 2].astype(np.int32)
                s_max_rg = np.maximum(sg, sr)

                s_blue = (((sb - s_max_rg) > cfg.blue_margin) & (sb_lab < 126)) | (
                    (sb_lab < cfg.lab_blue_b_max) & (sb > s_max_rg)
                )
                s_dark = (~s_blue) & (sl <= min(90, cfg.black_v_max))

                if total_px > 0:
                    yellow_ahead_blue_ratio = float(np.count_nonzero(s_blue)) / float(total_px)
                    yellow_ahead_dark_ratio = float(np.count_nonzero(s_dark)) / float(total_px)
                    combined = yellow_ahead_blue_ratio + yellow_ahead_dark_ratio
                    yellow_has_dead_end_ahead = (
                        yellow_ahead_blue_ratio >= cfg.yellow_dead_end_blue_ratio
                    ) or (
                        combined >= cfg.yellow_dead_end_combined_ratio
                        and yellow_ahead_dark_ratio >= cfg.yellow_dead_end_dark_ratio
                    )

        # 3. Lane binary mask segmentation (relative_border | hybrid | legacy_hsv)
        if cfg.lane_seg_mode == "relative_border":
            binary = _build_relative_border_mask(blurred, cfg)
        elif cfg.lane_seg_mode == "hybrid":
            rel_mask = _build_relative_border_mask(blurred, cfg)
            lower_relaxed = np.array([cfg.white_h_low, 0, cfg.hybrid_white_v_min], dtype=np.uint8)
            upper_relaxed = np.array([cfg.white_h_high, cfg.hybrid_white_s_max, 255], dtype=np.uint8)
            relaxed_hsv = cv2.inRange(hsv, lower_relaxed, upper_relaxed)
            binary = cv2.bitwise_and(rel_mask, relaxed_hsv)
        else:
            if cfg.use_white_filter:
                lower_white = np.array([cfg.white_h_low, 0, cfg.white_v_min], dtype=np.uint8)
                upper_white = np.array([cfg.white_h_high, cfg.white_s_max, 255], dtype=np.uint8)
                binary = cv2.inRange(hsv, lower_white, upper_white)
            else:
                gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
                thresh_type = cv2.THRESH_BINARY_INV if cfg.invert_binary else cv2.THRESH_BINARY
                _, binary = cv2.threshold(gray, cfg.binary_threshold, 255, thresh_type)

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._morph_element, iterations=1)

        # 4. Semantic hole-filling
        if cfg.enable_semantic_hole_fill:
            pad = cfg.semantic_hole_fill_dilate_px

            def _paint_full_rect(r_full: Tuple[int, int, int, int]) -> None:
                rx, ry, rw, rh = r_full
                if rw <= 0 or rh <= 0:
                    return
                padded = (rx - pad, ry - pad, rw + 2 * pad, rh + 2 * pad)
                inter = _rect_intersect(padded, roi_rect)
                if inter[2] <= 0 or inter[3] <= 0:
                    return
                lx = inter[0] - roi_rect[0]
                ly = inter[1] - roi_rect[1]
                cv2.rectangle(binary, (lx, ly), (lx + inter[2] - 1, ly + inter[3] - 1), 255, cv2.FILLED)

            for rect in fill_white_rects:
                _paint_full_rect(rect)
            if yellow_is_geometric and yellow_bbox_full[2] > 0:
                _paint_full_rect(yellow_bbox_full)

            # Fill small closed interior holes completely surrounded by white lane
            hole_contours, hole_hierarchy = cv2.findContours(
                binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            if hole_hierarchy is not None and len(hole_contours) > 0:
                hier = hole_hierarchy[0]
                max_hole_area = 0.06 * float(binary.shape[1] * binary.shape[0])
                b_cols = binary.shape[1]
                b_rows = binary.shape[0]
                for i, hcnt in enumerate(hole_contours):
                    if hier[i][3] < 0:
                        continue
                    a = float(cv2.contourArea(hcnt))
                    if a <= 4.0 or a > max_hole_area:
                        continue
                    hx, hy, hw, hh = cv2.boundingRect(hcnt)
                    if hx <= 1 or hy <= 1 or (hx + hw) >= (b_cols - 1) or (hy + hh) >= (b_rows - 1):
                        continue
                    cv2.drawContours(binary, hole_contours, i, 255, cv2.FILLED)

        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, self._morph_element, iterations=cfg.morph_iterations
        )

        out = SegmentedFrame(
            roi_bgr=roi.copy(),
            binary_mask=binary.copy(),
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            frame_width=width,
            frame_height=height,
            yellow_line_detected=yellow_line_detected,
            yellow_is_geometric=yellow_is_geometric,
            yellow_bottom_y_full=yellow_bottom_y_full,
            yellow_top_y_full=yellow_top_y_full,
            yellow_bar_width_px=yellow_bar_width_px,
            yellow_center_x_full=yellow_center_x_full,
            yellow_bbox_full=yellow_bbox_full,
            yellow_from_yolo_hybrid=yellow_from_yolo_hybrid,
            yellow_ahead_blue_ratio=yellow_ahead_blue_ratio,
            yellow_ahead_dark_ratio=yellow_ahead_dark_ratio,
            yellow_has_dead_end_ahead=yellow_has_dead_end_ahead,
            active_lane_seg_mode=cfg.lane_seg_mode,
            contour_valid=False,
            contour_centroid=(float(width) * 0.5, float(roi_height) * 0.70),
        )

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_area = 0.0
            best_idx = 0
            for i, cnt in enumerate(contours):
                area = float(cv2.contourArea(cnt))
                if area > max_area:
                    max_area = area
                    best_idx = i
            if max_area >= cfg.min_contour_area:
                out.contour_valid = True
                bx, by, bw, bh = cv2.boundingRect(contours[best_idx])
                out.bounding_box = (int(bx), int(by), int(bw), int(bh))
                moments = cv2.moments(contours[best_idx])
                if moments["m00"] != 0.0:
                    out.contour_centroid = (
                        float(moments["m10"] / moments["m00"]),
                        float(moments["m01"] / moments["m00"]),
                    )
                else:
                    out.contour_centroid = (float(bx + bw / 2.0), float(by + bh / 2.0))

        return out


# ---------------------------------------------------------------------------
# LaneGeometry (exact 1:1 mirror of lane_geometry.hpp / .cpp)
# ---------------------------------------------------------------------------


@dataclass
class LaneGeometryConfig:
    far_band_top_ratio: float = 0.28
    far_band_bottom_ratio: float = 0.48
    near_band_top_ratio: float = 0.55
    near_band_bottom_ratio: float = 0.80

    near_band_weight: float = 0.60
    far_band_weight: float = 0.40

    column_white_ratio: float = 0.40
    min_run_width: int = 18

    enable_continuous_curve_assist: bool = True
    reverse_curve_min_error: float = 0.12
    reverse_curve_threshold: float = 0.35
    reverse_curve_confirm_frames: int = 3
    reverse_curve_clear_frames: int = 4

    nominal_near_width_ratio: float = 0.68
    nominal_far_width_ratio: float = 0.48
    wide_road_ratio_threshold: float = 0.84
    wide_band_min_reliability: float = 0.18
    enable_width_adaptive_fusion: bool = True
    wide_lane_boundary_blend: float = 0.45
    observation_filter_alpha: float = 0.85

    fork_min_gap_px: int = 14
    fork_min_branch_width_px: int = 18


@dataclass
class BandObservation:
    valid: bool = False
    center_px: float = 0.0
    raw_center_px: float = 0.0
    center_norm: float = 0.0
    left_px: int = 0
    right_px: int = 0
    width_px: int = 0
    width_ratio: float = 0.0
    left_clipped: bool = False
    right_clipped: bool = False
    reliability: float = 1.0
    left_edge_px: float = 0.0
    right_edge_px: float = 0.0
    left_valid: bool = False
    right_valid: bool = False
    run_count: int = 0
    has_fork_split: bool = False
    left_branch_center_px: float = 0.0
    right_branch_center_px: float = 0.0
    left_branch_valid: bool = False
    right_branch_valid: bool = False


@dataclass
class CurveAssistState:
    reverse_curve_active: bool = False
    reverse_confirm_count: int = 0
    reverse_clear_count: int = 0
    near_valid: bool = False
    far_valid: bool = False
    near_error: float = 0.0
    far_error: float = 0.0
    curve_level: float = 0.0
    filter_initialized: bool = False
    filtered_error: float = 0.0
    filtered_heading: float = 0.0
    prev_far_center_px: float = 0.0
    last_confidence: float = 0.0
    last_heading_proxy: float = 0.0
    last_curvature_proxy: float = 0.0
    last_lane_width_ratio: float = 0.0


@dataclass
class LaneObservation:
    valid: bool = False
    yellow_detected: bool = False
    roi: Tuple[int, int, int, int] = (0, 0, 0, 0)
    near_band: BandObservation = field(default_factory=BandObservation)
    far_band: BandObservation = field(default_factory=BandObservation)
    near_center_px: float = 0.0
    far_center_px: float = 0.0
    near_center_norm: float = 0.0
    far_center_norm: float = 0.0
    near_width_ratio: float = 0.0
    far_width_ratio: float = 0.0
    lane_width_ratio: float = 0.0
    near_left_edge_px: float = 0.0
    near_right_edge_px: float = 0.0
    near_left_valid: bool = False
    near_right_valid: bool = False
    has_fork_split: bool = False
    left_branch_center_px: float = 0.0
    right_branch_center_px: float = 0.0
    left_branch_valid: bool = False
    right_branch_valid: bool = False
    active_branch_choice: int = 0
    lateral_error: float = 0.0
    heading_proxy: float = 0.0
    curvature_proxy: float = 0.0
    confidence: float = 0.0
    target_centroid: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ProcessingResult:
    line_detected: bool = False
    yellow_line_detected: bool = False
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    centroid: Tuple[float, float] = (0.0, 0.0)
    error: float = 0.0
    state: str = "lost"
    roi_bgr: Optional[np.ndarray] = None
    binary_mask: Optional[np.ndarray] = None
    roi_top: int = 0
    roi_bottom: int = 0
    yellow_is_geometric: bool = False
    yellow_bottom_y_full: float = 0.0
    yellow_top_y_full: float = 0.0
    yellow_bar_width_px: float = 0.0
    yellow_center_x_full: float = 0.0
    yellow_bbox_full: Tuple[int, int, int, int] = (0, 0, 0, 0)
    yellow_from_yolo_hybrid: bool = False
    yellow_ahead_blue_ratio: float = 0.0
    yellow_ahead_dark_ratio: float = 0.0
    yellow_has_dead_end_ahead: bool = False
    active_lane_seg_mode: str = "relative_border"
    observation: LaneObservation = field(default_factory=LaneObservation)


class LaneGeometryMirror:
    """1:1 Python mirror of `line_follower::LaneGeometry`."""

    def __init__(self, config: Optional[LaneGeometryConfig] = None) -> None:
        self.config = LaneGeometryConfig()
        self.set_config(config or LaneGeometryConfig())

    def set_config(self, config: LaneGeometryConfig) -> None:
        cfg = copy.copy(config)
        cfg.far_band_top_ratio = _clamp(float(cfg.far_band_top_ratio), 0.0, 1.0)
        cfg.far_band_bottom_ratio = _clamp(float(cfg.far_band_bottom_ratio), 0.0, 1.0)
        if cfg.far_band_top_ratio > cfg.far_band_bottom_ratio:
            cfg.far_band_top_ratio, cfg.far_band_bottom_ratio = (
                cfg.far_band_bottom_ratio,
                cfg.far_band_top_ratio,
            )

        cfg.near_band_top_ratio = _clamp(float(cfg.near_band_top_ratio), 0.0, 1.0)
        cfg.near_band_bottom_ratio = _clamp(float(cfg.near_band_bottom_ratio), 0.0, 1.0)
        if cfg.near_band_top_ratio > cfg.near_band_bottom_ratio:
            cfg.near_band_top_ratio, cfg.near_band_bottom_ratio = (
                cfg.near_band_bottom_ratio,
                cfg.near_band_top_ratio,
            )

        cfg.column_white_ratio = _clamp(float(cfg.column_white_ratio), 0.01, 1.0)
        cfg.min_run_width = max(1, int(cfg.min_run_width))
        cfg.reverse_curve_min_error = max(0.0, float(cfg.reverse_curve_min_error))
        cfg.reverse_curve_threshold = max(0.0, float(cfg.reverse_curve_threshold))
        cfg.reverse_curve_confirm_frames = max(1, int(cfg.reverse_curve_confirm_frames))
        cfg.reverse_curve_clear_frames = max(1, int(cfg.reverse_curve_clear_frames))
        cfg.nominal_near_width_ratio = _clamp(float(cfg.nominal_near_width_ratio), 0.10, 1.0)
        cfg.nominal_far_width_ratio = _clamp(float(cfg.nominal_far_width_ratio), 0.10, 1.0)
        cfg.wide_road_ratio_threshold = _clamp(float(cfg.wide_road_ratio_threshold), 0.20, 1.0)
        cfg.wide_band_min_reliability = _clamp(float(cfg.wide_band_min_reliability), 0.05, 1.0)
        cfg.wide_lane_boundary_blend = _clamp(float(cfg.wide_lane_boundary_blend), 0.0, 1.0)
        cfg.observation_filter_alpha = _clamp(float(cfg.observation_filter_alpha), 0.05, 1.0)
        cfg.fork_min_gap_px = max(2, int(cfg.fork_min_gap_px))
        cfg.fork_min_branch_width_px = max(1, int(cfg.fork_min_branch_width_px))
        self.config = cfg

    def observe_band(
        self,
        binary: np.ndarray,
        y0: int,
        y1: int,
        reference_x: float,
        nominal_width_ratio: float,
        dir_hint_norm: float = 0.0,
        preferred_branch: int = 0,
    ) -> BandObservation:
        del dir_hint_norm
        obs = BandObservation()
        if binary is None or binary.size == 0 or binary.shape[0] <= 0 or binary.shape[1] <= 0:
            return obs

        rows, cols = binary.shape[:2]
        y0 = _clamp_int(int(y0), 0, rows - 1)
        y1 = _clamp_int(int(y1), y0 + 1, rows)
        band = binary[y0:y1, :]
        band_height = band.shape[0]
        if band_height <= 0:
            return obs

        cfg = self.config
        min_white_pixels = max(1, int(float(band_height) * cfg.column_white_ratio))
        active = np.count_nonzero(band, axis=0) >= min_white_pixels

        # Extract contiguous active column runs
        padded = np.empty(cols + 2, dtype=np.int8)
        padded[0] = 0
        padded[1 : cols + 1] = active.astype(np.int8)
        padded[cols + 1] = 0
        diffs = np.diff(padded)
        starts = np.flatnonzero(diffs == 1)
        ends = np.flatnonzero(diffs == -1) - 1

        runs: List[Tuple[int, int, int, float]] = []
        for s_i, e_i in zip(starts, ends):
            s_int = int(s_i)
            e_int = int(e_i)
            w_int = e_int - s_int + 1
            if w_int >= cfg.min_run_width:
                runs.append((s_int, e_int, w_int, 0.5 * float(s_int + e_int)))

        if not runs:
            return obs

        obs.run_count = len(runs)

        # Multi-run fork split detection
        left_branch_idx = -1
        right_branch_idx = -1
        for i, (_rs, _re, rw, _rc) in enumerate(runs):
            if rw < cfg.fork_min_branch_width_px:
                continue
            if left_branch_idx < 0:
                left_branch_idx = i
            right_branch_idx = i

        if left_branch_idx >= 0 and right_branch_idx > left_branch_idx:
            lrun = runs[left_branch_idx]
            rrun = runs[right_branch_idx]
            gap_px = rrun[0] - lrun[1] - 1
            if gap_px >= cfg.fork_min_gap_px:
                obs.has_fork_split = True
                obs.left_branch_valid = True
                obs.left_branch_center_px = lrun[3]
                obs.right_branch_valid = True
                obs.right_branch_center_px = rrun[3]

        # Select active run
        chosen_idx = 0
        if preferred_branch < 0 and obs.left_branch_valid:
            chosen_idx = left_branch_idx
        elif preferred_branch > 0 and obs.right_branch_valid:
            chosen_idx = right_branch_idx
        else:
            best_distance = 1e9
            best_width = 0
            for i, (rs, re, rw, rc) in enumerate(runs):
                distance = 0.0
                if reference_x < float(rs) or reference_x > float(re):
                    distance = abs(rc - reference_x)
                if (
                    i == 0
                    or distance < best_distance
                    or (abs(distance - best_distance) < 1e-6 and rw > best_width)
                ):
                    chosen_idx = i
                    best_distance = distance
                    best_width = rw

        best_start, best_end, best_width, _ = runs[chosen_idx]
        cols_d = float(cols)
        image_center_x = 0.5 * cols_d

        obs.valid = True
        obs.left_px = best_start
        obs.right_px = best_end
        obs.width_px = best_width
        obs.width_ratio = _clamp(float(best_width) / cols_d, 0.0, 1.0)
        obs.left_clipped = best_start <= 1
        obs.right_clipped = best_end >= cols - 2
        obs.left_edge_px = float(best_start)
        obs.right_edge_px = float(best_end)
        obs.left_valid = not obs.left_clipped
        obs.right_valid = not obs.right_clipped
        obs.raw_center_px = 0.5 * float(best_start + best_end)

        if obs.width_ratio <= cfg.wide_road_ratio_threshold:
            obs.reliability = 1.0
        else:
            excess = (obs.width_ratio - cfg.wide_road_ratio_threshold) / max(
                1e-3, 1.0 - cfg.wide_road_ratio_threshold
            )
            obs.reliability = _clamp(
                1.0 - excess * (1.0 - cfg.wide_band_min_reliability),
                cfg.wide_band_min_reliability,
                1.0,
            )
        if obs.left_clipped and obs.right_clipped:
            obs.reliability = min(obs.reliability, cfg.wide_band_min_reliability)

        effective_center = obs.raw_center_px
        if cfg.wide_lane_boundary_blend > 0.0:
            nominal_w = _clamp(nominal_width_ratio, 0.15, 0.95) * cols_d
            if obs.left_clipped and not obs.right_clipped:
                anchored = min(float(obs.right_px) - 0.5 * nominal_w, 0.5 * nominal_w)
                effective_center = (
                    (1.0 - cfg.wide_lane_boundary_blend) * obs.raw_center_px
                    + cfg.wide_lane_boundary_blend * anchored
                )
            elif obs.right_clipped and not obs.left_clipped:
                anchored = max(float(obs.left_px) + 0.5 * nominal_w, cols_d - 0.5 * nominal_w)
                effective_center = (
                    (1.0 - cfg.wide_lane_boundary_blend) * obs.raw_center_px
                    + cfg.wide_lane_boundary_blend * anchored
                )
            effective_center = _clamp(effective_center, 0.0, cols_d - 1.0)

        obs.center_px = effective_center
        obs.center_norm = _clamp((obs.center_px - image_center_x) / image_center_x, -1.0, 1.0)
        return obs

    def observe(
        self,
        seg: SegmentedFrame,
        curve_state: CurveAssistState,
        preferred_branch: int = 0,
    ) -> LaneObservation:
        cfg = self.config
        obs = LaneObservation(
            valid=False,
            yellow_detected=seg.yellow_line_detected,
            roi=(0, seg.roi_top, seg.frame_width, seg.roi_bottom - seg.roi_top),
            active_branch_choice=int(preferred_branch),
        )

        if seg.binary_mask is None or seg.binary_mask.size == 0 or seg.frame_width <= 0:
            curve_state.last_confidence = 0.0
            return obs

        roi_height = seg.roi_bottom - seg.roi_top
        width = seg.frame_width
        image_center_x = float(width) * 0.5

        far_y0 = int(float(roi_height) * cfg.far_band_top_ratio)
        far_y1 = int(float(roi_height) * cfg.far_band_bottom_ratio)
        near_y0 = int(float(roi_height) * cfg.near_band_top_ratio)
        near_y1 = int(float(roi_height) * cfg.near_band_bottom_ratio)

        near_obs = self.observe_band(
            seg.binary_mask,
            near_y0,
            near_y1,
            image_center_x,
            cfg.nominal_near_width_ratio,
            0.0,
            preferred_branch,
        )
        far_reference = near_obs.center_px if near_obs.valid else image_center_x
        far_obs = self.observe_band(
            seg.binary_mask,
            far_y0,
            far_y1,
            far_reference,
            cfg.nominal_far_width_ratio,
            0.0,
            preferred_branch,
        )

        near_valid = near_obs.valid
        far_valid = far_obs.valid
        near_error = near_obs.center_norm if near_valid else 0.0
        far_error = far_obs.center_norm if far_valid else 0.0

        curve_state.near_valid = near_valid
        curve_state.far_valid = far_valid
        curve_state.near_error = near_error
        curve_state.far_error = far_error
        curve_state.curve_level = 0.0
        if near_valid:
            curve_state.curve_level = max(curve_state.curve_level, abs(near_error))
        if far_valid:
            curve_state.curve_level = max(curve_state.curve_level, abs(far_error))

        reverse_curve_candidate = False
        if cfg.enable_continuous_curve_assist and near_valid and far_valid:
            opposite_direction = near_error * far_error < 0.0
            both_significant = (
                abs(near_error) >= cfg.reverse_curve_min_error
                and abs(far_error) >= cfg.reverse_curve_min_error
            )
            large_difference = abs(far_error - near_error) >= cfg.reverse_curve_threshold
            reverse_curve_candidate = opposite_direction and both_significant and large_difference

        if not cfg.enable_continuous_curve_assist:
            curve_state.reverse_curve_active = False
            curve_state.reverse_confirm_count = 0
            curve_state.reverse_clear_count = 0
        elif reverse_curve_candidate:
            curve_state.reverse_clear_count = 0
            curve_state.reverse_confirm_count += 1
            if (
                not curve_state.reverse_curve_active
                and curve_state.reverse_confirm_count >= cfg.reverse_curve_confirm_frames
            ):
                curve_state.reverse_curve_active = True
                curve_state.reverse_confirm_count = cfg.reverse_curve_confirm_frames
        else:
            curve_state.reverse_confirm_count = 0
            if curve_state.reverse_curve_active:
                curve_state.reverse_clear_count += 1
                if curve_state.reverse_clear_count >= cfg.reverse_curve_clear_frames:
                    curve_state.reverse_curve_active = False
                    curve_state.reverse_clear_count = 0
            else:
                curve_state.reverse_clear_count = 0

        target_x = float(seg.contour_centroid[0])
        target_y = float(seg.contour_centroid[1])
        local_target_valid = False
        near_y = 0.5 * float(near_y0 + near_y1)
        far_y = 0.5 * float(far_y0 + far_y1)

        if near_valid and far_valid:
            use_reverse_near_only = (
                cfg.enable_continuous_curve_assist
                and curve_state.reverse_curve_active
                and (not cfg.enable_width_adaptive_fusion or near_obs.reliability >= 0.5)
            )
            if use_reverse_near_only:
                target_x = near_obs.center_px
                target_y = near_y
            else:
                w_near = max(0.0, cfg.near_band_weight)
                w_far = max(0.0, cfg.far_band_weight)
                if cfg.enable_width_adaptive_fusion:
                    w_near *= near_obs.reliability
                    w_far *= far_obs.reliability
                w_sum = w_near + w_far
                if w_sum <= 1e-9:
                    w_near, w_far = 0.70, 0.30
                elif abs(w_sum - 1.0) > 1e-9:
                    w_near /= w_sum
                    w_far /= w_sum
                target_x = w_near * near_obs.center_px + w_far * far_obs.center_px
                target_y = w_near * near_y + w_far * far_y
            local_target_valid = True
        elif near_valid:
            target_x = near_obs.center_px
            target_y = near_y
            local_target_valid = True
        elif far_valid:
            target_x = far_obs.center_px
            target_y = far_y
            local_target_valid = True

        if not local_target_valid and not seg.contour_valid:
            curve_state.last_confidence = 0.0
            return obs

        obs.valid = True
        obs.near_band = near_obs
        obs.far_band = far_obs
        obs.near_center_px = near_obs.center_px
        obs.far_center_px = far_obs.center_px
        obs.near_center_norm = near_obs.center_norm
        obs.far_center_norm = far_obs.center_norm
        obs.near_width_ratio = near_obs.width_ratio
        obs.far_width_ratio = far_obs.width_ratio
        obs.near_left_edge_px = near_obs.left_edge_px
        obs.near_right_edge_px = near_obs.right_edge_px
        obs.near_left_valid = near_obs.valid and near_obs.left_valid
        obs.near_right_valid = near_obs.valid and near_obs.right_valid

        if far_obs.has_fork_split:
            obs.has_fork_split = True
            obs.left_branch_center_px = far_obs.left_branch_center_px
            obs.right_branch_center_px = far_obs.right_branch_center_px
            obs.left_branch_valid = far_obs.left_branch_valid
            obs.right_branch_valid = far_obs.right_branch_valid
        elif near_obs.has_fork_split:
            obs.has_fork_split = True
            obs.left_branch_center_px = near_obs.left_branch_center_px
            obs.right_branch_center_px = near_obs.right_branch_center_px
            obs.left_branch_valid = near_obs.left_branch_valid
            obs.right_branch_valid = near_obs.right_branch_valid

        if near_valid and far_valid:
            obs.lane_width_ratio = 0.5 * (near_obs.width_ratio + far_obs.width_ratio)
        elif near_valid:
            obs.lane_width_ratio = near_obs.width_ratio
        elif far_valid:
            obs.lane_width_ratio = far_obs.width_ratio
        else:
            obs.lane_width_ratio = _clamp(float(seg.bounding_box[2]) / float(width), 0.0, 1.0)

        raw_centroid_x = float(target_x)
        raw_centroid_y = float(target_y)
        raw_error = _clamp((raw_centroid_x - image_center_x) / image_center_x, -1.0, 1.0)

        # Wide-junction heading continuity (matches latest lane_geometry.cpp:482-498)
        raw_heading = 0.0
        if near_valid and far_valid:
            if (
                cfg.enable_width_adaptive_fusion
                and far_obs.left_clipped
                and far_obs.right_clipped
                and not (near_obs.left_clipped and near_obs.right_clipped)
            ):
                raw_heading = (
                    0.75 * curve_state.filtered_heading if curve_state.filter_initialized else 0.0
                )
            else:
                raw_heading = _clamp(far_obs.center_norm - near_obs.center_norm, -1.0, 1.0)
        elif far_valid:
            raw_heading = far_obs.center_norm
        elif near_valid:
            raw_heading = (
                0.5 * curve_state.filtered_heading if curve_state.filter_initialized else 0.0
            )

        # 4-part confidence score
        width_score = 0.30
        if near_valid and far_valid:
            width_score = 0.5 * (near_obs.reliability + far_obs.reliability)
        elif near_valid:
            width_score = 0.75 * near_obs.reliability
        elif far_valid:
            width_score = 0.75 * far_obs.reliability

        consistency_score = 0.25
        if near_valid and far_valid:
            consistency_score = (
                _clamp(1.0 - 0.5 * abs(far_obs.center_norm - near_obs.center_norm), 0.30, 1.0)
                * width_score
            )
        elif near_valid or far_valid:
            consistency_score = 0.55 * width_score

        stability_base = (
            _clamp(1.0 - abs(raw_error - curve_state.filtered_error) / 0.50, 0.0, 1.0)
            if curve_state.filter_initialized
            else 1.0
        )
        stability_score = stability_base * math.sqrt(_clamp(width_score, 0.0, 1.0))
        contour_score = 1.0 if seg.contour_valid else 0.30

        obs.confidence = _clamp(
            0.45 * width_score
            + 0.25 * consistency_score
            + 0.15 * stability_score
            + 0.15 * contour_score,
            0.0,
            1.0,
        )

        both_double_clipped = (
            near_valid
            and far_valid
            and near_obs.left_clipped
            and near_obs.right_clipped
            and far_obs.left_clipped
            and far_obs.right_clipped
        )

        alpha = cfg.observation_filter_alpha
        if not curve_state.filter_initialized or alpha >= 0.999:
            obs.lateral_error = raw_error
            obs.heading_proxy = raw_heading
            obs.target_centroid = (raw_centroid_x, raw_centroid_y)
        elif cfg.enable_width_adaptive_fusion and both_double_clipped:
            obs.lateral_error = raw_error
            obs.heading_proxy = raw_heading
            obs.target_centroid = (raw_centroid_x, raw_centroid_y)
        else:
            obs.lateral_error = _clamp(
                alpha * raw_error + (1.0 - alpha) * curve_state.filtered_error, -1.0, 1.0
            )
            obs.heading_proxy = _clamp(
                alpha * raw_heading + (1.0 - alpha) * curve_state.filtered_heading, -1.0, 1.0
            )
            filtered_target_x = _clamp(
                image_center_x + obs.lateral_error * image_center_x, 0.0, float(width - 1)
            )
            obs.target_centroid = (filtered_target_x, raw_centroid_y)

        base_curve = (
            abs(far_obs.center_norm - near_obs.center_norm)
            if (near_valid and far_valid)
            else abs(obs.heading_proxy)
        )
        if cfg.enable_width_adaptive_fusion:
            max_w = max(obs.near_width_ratio, obs.far_width_ratio)
            if max_w > cfg.wide_road_ratio_threshold:
                w_excess = (max_w - cfg.wide_road_ratio_threshold) / max(
                    1e-3, 1.0 - cfg.wide_road_ratio_threshold
                )
                base_curve = max(base_curve, 0.35 * _clamp(w_excess, 0.0, 1.0))
        obs.curvature_proxy = _clamp(base_curve, 0.0, 1.0)

        curve_state.filter_initialized = True
        if cfg.enable_width_adaptive_fusion and both_double_clipped:
            curve_state.filtered_error *= 0.92
            curve_state.filtered_heading *= 0.92
        else:
            curve_state.filtered_error = obs.lateral_error
            curve_state.filtered_heading = obs.heading_proxy
        if far_valid:
            curve_state.prev_far_center_px = far_obs.center_px
        curve_state.last_confidence = obs.confidence
        curve_state.last_heading_proxy = obs.heading_proxy
        curve_state.last_curvature_proxy = obs.curvature_proxy
        curve_state.last_lane_width_ratio = obs.lane_width_ratio

        return obs

    def evaluate(
        self,
        seg: SegmentedFrame,
        curve_state: CurveAssistState,
        preferred_branch: int = 0,
    ) -> ProcessingResult:
        result = ProcessingResult(
            line_detected=False,
            yellow_line_detected=seg.yellow_line_detected,
            yellow_is_geometric=seg.yellow_is_geometric,
            yellow_bottom_y_full=seg.yellow_bottom_y_full,
            yellow_top_y_full=seg.yellow_top_y_full,
            yellow_bar_width_px=seg.yellow_bar_width_px,
            yellow_center_x_full=seg.yellow_center_x_full,
            yellow_bbox_full=seg.yellow_bbox_full,
            yellow_from_yolo_hybrid=seg.yellow_from_yolo_hybrid,
            yellow_ahead_blue_ratio=seg.yellow_ahead_blue_ratio,
            yellow_ahead_dark_ratio=seg.yellow_ahead_dark_ratio,
            yellow_has_dead_end_ahead=seg.yellow_has_dead_end_ahead,
            active_lane_seg_mode=seg.active_lane_seg_mode,
            bounding_box=seg.bounding_box,
            roi_bgr=seg.roi_bgr,
            binary_mask=seg.binary_mask,
            roi_top=seg.roi_top,
            roi_bottom=seg.roi_bottom,
        )
        obs = self.observe(seg, curve_state, preferred_branch)
        result.observation = obs
        if not obs.valid:
            return result
        result.line_detected = True
        result.centroid = obs.target_centroid
        result.error = obs.lateral_error
        result.state = "tracking"
        return result


# ---------------------------------------------------------------------------
# Dual-Mode Detectors: RealYoloDetector & MockNoisyDetector
# ---------------------------------------------------------------------------


def _format_detection_dict(
    class_name: str,
    cls_id: int,
    conf: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    track_id: int = -1,
    source: str = "yolo",
) -> Dict[str, Any]:
    w_f = max(1.0, float(img_w))
    h_f = max(1.0, float(img_h))
    x1_c = _clamp(float(x1), 0.0, w_f)
    y1_c = _clamp(float(y1), 0.0, h_f)
    x2_c = _clamp(float(x2), x1_c, w_f)
    y2_c = _clamp(float(y2), y1_c, h_f)
    bw = max(1.0, x2_c - x1_c)
    bh = max(1.0, y2_c - y1_c)
    x_i = _clamp_int(int(round(x1_c)), 0, max(0, img_w - 1))
    y_i = _clamp_int(int(round(y1_c)), 0, max(0, img_h - 1))
    w_i = min(max(1, int(round(bw))), max(1, img_w - x_i))
    h_i = min(max(1, int(round(bh))), max(1, img_h - y_i))
    return {
        "class_name": str(class_name),
        "type": str(class_name),
        "cls_id": int(cls_id),
        "conf": float(conf),
        "confidence": float(conf),
        "x": int(x_i),
        "y": int(y_i),
        "w": int(w_i),
        "h": int(h_i),
        "x1": float(x1_c),
        "y1": float(y1_c),
        "x2": float(x2_c),
        "y2": float(y2_c),
        "x1_norm": float(x1_c / w_f),
        "y1_norm": float(y1_c / h_f),
        "x2_norm": float(x2_c / w_f),
        "y2_norm": float(y2_c / h_f),
        "cx": float(x1_c + 0.5 * bw),
        "cy": float(y1_c + 0.5 * bh),
        "track_id": int(track_id),
        "source": str(source),
    }


def _load_out_list(path: Optional[Union[str, Path]]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        items = [line.strip() for line in f.readlines() if line.strip()]
    return items if items else None


def _parse_class_conf_overrides(
    overrides: Optional[Union[Dict[str, Any], str]],
    names_map: Dict[int, str],
    default_conf: float,
) -> Dict[int, float]:
    out = {int(cid): float(default_conf) for cid in names_map.keys()}
    if not overrides:
        return out
    parsed: Optional[Dict[str, Any]] = None
    if isinstance(overrides, dict):
        parsed = overrides
    elif isinstance(overrides, str):
        s = overrides.strip()
        if s:
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    parsed = obj
            except Exception:
                parsed = None
    if not parsed:
        return out
    name_to_id = {str(v): int(k) for k, v in names_map.items()}
    for k, v in parsed.items():
        try:
            thr = _clamp(float(v), 1e-4, 1.0 - 1e-4)
        except Exception:
            continue
        ks = str(k).strip()
        if ks in name_to_id:
            out[name_to_id[ks]] = thr
        elif ks.isdigit() and int(ks) in out:
            out[int(ks)] = thr
    return out


def _safe_import_ultralytics() -> Tuple[Any, Any]:
    """Import `torch` & `ultralytics.YOLO` while guarding against `tools/sim/profile.py`
    shadowing Python's stdlib `profile` module (which `torchvision` -> `cProfile` imports
    when scripts are launched directly as `python tools/sim/<script>.py`).
    """
    sim_dir = os.path.normcase(os.path.abspath(str(Path(__file__).resolve().parent)))
    saved_path = list(sys.path)
    saved_profile = sys.modules.get("profile")
    profile_shadowed = (
        saved_profile is not None
        and getattr(saved_profile, "__file__", None) is not None
        and os.path.normcase(os.path.abspath(os.path.dirname(saved_profile.__file__))) == sim_dir
    )
    try:
        sys.path[:] = [
            p for p in sys.path
            if os.path.normcase(os.path.abspath(p or ".")) != sim_dir
        ]
        if profile_shadowed:
            sys.modules.pop("profile", None)
        import cProfile  # noqa: F401  (pre-binds stdlib `profile` cleanly)
        import torch
        import torchvision  # noqa: F401
        from ultralytics import YOLO
        return torch, YOLO
    finally:
        sys.path[:] = saved_path
        if profile_shadowed and saved_profile is not None:
            sys.modules["profile"] = saved_profile


class RealYoloDetector:
    """Real `ultralytics.YOLO` detector running on CUDA GPU with graceful CPU fallback.

    Output format matches `bpu_yolov8_digit_node` (pixel & normalized bboxes, class_name,
    conf, persistent track_id).
    """

    def __init__(
        self,
        weights_path: Union[str, Path] = DEFAULT_YOLO_WEIGHTS,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.55,
        imgsz: int = 640,
        min_box_size: int = 6,
        top_k: int = 20,
        device: Optional[str] = None,
        cls_list_path: Optional[Union[str, Path]] = None,
        class_conf_overrides: Optional[Union[Dict[str, Any], str]] = None,
        enable_tracker: bool = True,
    ) -> None:
        wpath = Path(weights_path)
        if not wpath.is_absolute():
            wpath = REPO_ROOT / wpath
        if not wpath.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {wpath}")
        self.weights_path = wpath
        self.conf_thresh = float(_clamp(conf_thresh, 1e-4, 0.9999))
        self.iou_thresh = float(_clamp(iou_thresh, 0.05, 0.99))
        self.imgsz = int(max(160, imgsz))
        self.min_box_size = int(max(1, min_box_size))
        self.top_k = int(max(1, top_k))
        self.enable_tracker = bool(enable_tracker)
        self.tracker = SimpleBoxTracker()

        torch, YOLO = _safe_import_ultralytics()

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                self.device = "cpu"

        self.model = YOLO(str(self.weights_path))
        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"
            self.model.to("cpu")

        # Build class id -> name map from model.names (optionally overridden by out.list)
        raw_names = getattr(self.model, "names", None)
        if isinstance(raw_names, dict) and raw_names:
            self.names: Dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}
        elif isinstance(raw_names, (list, tuple)) and raw_names:
            self.names = {i: str(v) for i, v in enumerate(raw_names)}
        else:
            loaded = _load_out_list(cls_list_path or DEFAULT_OUT_LIST) or [str(i) for i in range(1, 9)]
            self.names = {i: name for i, name in enumerate(loaded)}

        if cls_list_path:
            loaded = _load_out_list(cls_list_path)
            if loaded and len(loaded) == len(self.names):
                self.names = {i: name for i, name in enumerate(loaded)}

        self._cls_thresholds = _parse_class_conf_overrides(
            class_conf_overrides, self.names, self.conf_thresh
        )
        self._min_conf = min(self._cls_thresholds.values()) if self._cls_thresholds else self.conf_thresh

    def reset(self) -> None:
        self.tracker.reset()

    def warmup(self, width: int = 640, height: int = 480) -> None:
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        self.detect(dummy)
        self.tracker.reset()

    def detect(
        self,
        bgr_img: np.ndarray,
        world_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        del world_context
        if bgr_img is None or bgr_img.size == 0:
            return []
        img_h, img_w = bgr_img.shape[:2]

        try:
            results = self.model.predict(
                source=bgr_img,
                conf=self._min_conf,
                iou=self.iou_thresh,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                max_det=self.top_k,
            )
        except Exception:
            if self.device != "cpu":
                self.device = "cpu"
                self.model.to("cpu")
                results = self.model.predict(
                    source=bgr_img,
                    conf=self._min_conf,
                    iou=self.iou_thresh,
                    imgsz=self.imgsz,
                    device="cpu",
                    verbose=False,
                    max_det=self.top_k,
                )
            else:
                raise

        dets: List[Dict[str, Any]] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy_arr = boxes.xyxy.detach().cpu().numpy()
            conf_arr = boxes.conf.detach().cpu().numpy()
            cls_arr = boxes.cls.detach().cpu().numpy()
            for xyxy, conf_v, cls_v in zip(xyxy_arr, conf_arr, cls_arr):
                cid = int(cls_v)
                conf_f = float(conf_v)
                thr = self._cls_thresholds.get(cid, self.conf_thresh)
                if conf_f < thr:
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                if (x2 - x1) < self.min_box_size or (y2 - y1) < self.min_box_size:
                    continue
                cname = self.names.get(cid, str(cid))
                dets.append(
                    _format_detection_dict(
                        class_name=cname,
                        cls_id=cid,
                        conf=conf_f,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        img_w=img_w,
                        img_h=img_h,
                        track_id=-1,
                        source="yolo",
                    )
                )

        if self.enable_tracker:
            dets = self.tracker.update(dets)
        return dets


def _project_point_3d(
    cam: CameraModel, w_mm: float, d_mm: float, z_mm: float = 0.0
) -> Optional[Tuple[float, float]]:
    """Project vehicle-frame 3D point (w right, d forward, z up, in mm) to (u, v)."""
    h_eff = cam.h - z_mm
    denom = d_mm + h_eff * cam.tan_a
    if denom <= 1e-3:
        return None
    # Note: CameraModel.wd_to_uv formula generalized to arbitrary elevation z_mm
    # Ray in camera frame:
    #   x_c = w_mm
    #   y_c = -d_mm * sin(a) + (h - z_mm) * cos(a)
    #   z_c =  d_mm * cos(a) + (h - z_mm) * sin(a)
    # Dividing numerator & denominator by cos(a):
    #   z_c / cos(a) = d_mm + h_eff * tan(a) = denom
    #   y_c / cos(a) = h_eff - d_mm * tan(a)
    u = cam.cx + cam.f * w_mm / denom
    v = cam.cy + cam.f * (h_eff - d_mm * cam.tan_a) / denom
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return float(u), float(v)


def _normalize_world_props(raw_props: Any) -> List[Dict[str, Any]]:
    """Accept list[dict] or dict-of-lists (TrackRenderer.injected_props) into uniform list."""
    if raw_props is None:
        return []
    if isinstance(raw_props, list):
        return [p for p in raw_props if isinstance(p, dict)]
    if isinstance(raw_props, dict):
        out: List[Dict[str, Any]] = []
        mapping = {
            "digits": "digit",
            "bonus_digits": "digit",
            "cones": "cone",
            "red_cones": "cone",
            "no_entry": "no_entry",
            "red_no_entry": "no_entry",
            "random_yellow": "yellow_line",
            "yellow_lines": "yellow_line",
        }
        for key, val in raw_props.items():
            if not isinstance(val, list):
                continue
            inferred = mapping.get(str(key).lower(), str(key).lower())
            for item in val:
                if isinstance(item, dict):
                    d = dict(item)
                    d.setdefault("type", inferred)
                    out.append(d)
        return out
    return []


def _extract_pose_mm(world_context: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    pose = world_context.get("pose", world_context.get("gt_pose"))
    if pose is None:
        if "x_mm" in world_context and "y_mm" in world_context:
            return (
                float(world_context["x_mm"]),
                float(world_context["y_mm"]),
                float(world_context.get("yaw", 0.0)),
            )
        if "x_m" in world_context and "y_m" in world_context:
            return (
                float(world_context["x_m"]) * 1000.0,
                float(world_context["y_m"]) * 1000.0,
                float(world_context.get("yaw", 0.0)),
            )
        return None
    if isinstance(pose, (tuple, list)) and len(pose) >= 3:
        return float(pose[0]), float(pose[1]), float(pose[2])
    if isinstance(pose, dict):
        if "x_mm" in pose and "y_mm" in pose:
            return float(pose["x_mm"]), float(pose["y_mm"]), float(pose.get("yaw", 0.0))
        if "x_m" in pose and "y_m" in pose:
            return (
                float(pose["x_m"]) * 1000.0,
                float(pose["y_m"]) * 1000.0,
                float(pose.get("yaw", 0.0)),
            )
        if "x" in pose and "y" in pose:
            # Heuristic: track coordinates in mm are > 50, in meters are < 50
            x_v = float(pose["x"])
            y_v = float(pose["y"])
            scale = 1000.0 if (abs(x_v) < 30.0 and abs(y_v) < 30.0) else 1.0
            return x_v * scale, y_v * scale, float(pose.get("yaw", 0.0))
    return None


class MockNoisyDetector:
    """Fast statistical mock detector (10-20x+ faster than YOLO) for batch/auto-tune.

    Projects ground-truth scene props (`world_props` + `CameraModel` + vehicle pose
    `(x_mm, y_mm, yaw)`) into image bboxes with realistic distance-dependent recall,
    bbox pixel jitter, confidence noise, and optional brief dropout/flicker.

    By default mirrors the real-car 8-class digit model (`detect_digits=True`,
    `detect_obstacles=False`, `detect_yellow=False`), since red cones/no-entry are
    detected by `detect_cv_red_in_lane` and yellow lines by geometric CV. Set
    `detect_obstacles=True` / `detect_yellow=True` to simulate an 11-class model.
    """

    def __init__(
        self,
        seed: int = 42,
        conf_thresh: float = 0.25,
        bbox_jitter_px: float = 2.0,
        conf_noise_std: float = 0.05,
        base_recall: float = 0.96,
        flicker_prob: float = 0.03,
        flicker_max_frames: int = 2,
        min_dist_mm: float = 130.0,
        max_dist_mm: float = 2100.0,
        min_box_size: int = 6,
        detect_digits: bool = True,
        detect_obstacles: bool = False,
        detect_yellow: bool = False,
        cam: Optional[CameraModel] = None,
        enable_tracker: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.conf_thresh = float(_clamp(conf_thresh, 0.01, 0.99))
        self.bbox_jitter_px = float(max(0.0, bbox_jitter_px))
        self.conf_noise_std = float(max(0.0, conf_noise_std))
        self.base_recall = float(_clamp(base_recall, 0.0, 1.0))
        self.flicker_prob = float(_clamp(flicker_prob, 0.0, 0.5))
        self.flicker_max_frames = int(max(0, flicker_max_frames))
        self.min_dist_mm = float(max(10.0, min_dist_mm))
        self.max_dist_mm = float(max(self.min_dist_mm + 50.0, max_dist_mm))
        self.min_box_size = int(max(1, min_box_size))
        self.detect_digits = bool(detect_digits)
        self.detect_obstacles = bool(detect_obstacles)
        self.detect_yellow = bool(detect_yellow)
        self.default_cam = cam or CameraModel()
        self.enable_tracker = bool(enable_tracker)

        self._rng = np.random.default_rng(self.seed)
        self._dropout_countdown: Dict[str, int] = {}
        self.tracker = SimpleBoxTracker()

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._dropout_countdown.clear()
        self.tracker.reset()

    def _distance_recall(self, dist_m: float, box_min_dim_px: float) -> float:
        # High recall in 0.25m..1.25m sweet spot; smooth roll-off very close or far
        if dist_m < 0.22:
            near_f = _clamp((dist_m - 0.12) / 0.10, 0.35, 1.0)
        else:
            near_f = 1.0
        if dist_m > 1.25:
            far_f = _clamp(1.0 - (dist_m - 1.25) / 0.95, 0.15, 1.0)
        else:
            far_f = 1.0
        size_f = _clamp((box_min_dim_px - 4.0) / 14.0, 0.25, 1.0)
        return float(_clamp(self.base_recall * near_f * far_f * size_f, 0.0, 1.0))

    def _distance_base_conf(self, dist_m: float) -> float:
        # ~0.91 at 0.45m tapering smoothly to ~0.58 at 1.8m
        raw = 0.93 - 0.20 * max(0.0, dist_m - 0.35)
        return float(_clamp(raw, 0.45, 0.96))

    def detect(
        self,
        bgr_img: Optional[np.ndarray],
        world_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not world_context:
            return []

        pose = _extract_pose_mm(world_context)
        if pose is None:
            return []
        x_mm, y_mm, yaw = pose

        cam = world_context.get("cam") or world_context.get("camera") or self.default_cam
        if bgr_img is not None and bgr_img.size > 0:
            img_h, img_w = bgr_img.shape[:2]
        else:
            img_h, img_w = int(cam.h_img), int(cam.w)

        raw_props = world_context.get("world_props", world_context.get("props"))
        props = _normalize_world_props(raw_props)
        if not props:
            return []

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        dets: List[Dict[str, Any]] = []
        for idx, prop in enumerate(props):
            ptype_raw = str(prop.get("type", "")).strip().lower()
            class_name: Optional[str] = None
            cls_id = 0
            kind = ""

            if ptype_raw in ("digit", "bonus_digit", "bonus"):
                if not self.detect_digits:
                    continue
                val = int(prop.get("value", prop.get("digit", 0)))
                if val < 1 or val > 8:
                    continue
                class_name = str(val)
                cls_id = val - 1
                kind = "digit"
            elif ptype_raw in ("cone", "red_cone"):
                if not self.detect_obstacles:
                    continue
                class_name = "red_cone"
                cls_id = 8
                kind = "cone"
            elif ptype_raw in ("no_entry", "red_no_entry", "forbidden"):
                if not self.detect_obstacles:
                    continue
                class_name = "red_no_entry"
                cls_id = 9
                kind = "no_entry"
            elif ptype_raw in ("yellow", "yellow_line", "yellow_stop_line", "random_yellow"):
                if not self.detect_yellow:
                    continue
                class_name = "yellow_stop_line"
                cls_id = 10
                kind = "yellow"
            else:
                continue

            if "x_mm" in prop and "y_mm" in prop:
                px_mm = float(prop["x_mm"])
                py_mm = float(prop["y_mm"])
            elif "x_m" in prop and "y_m" in prop:
                px_mm = float(prop["x_m"]) * 1000.0
                py_mm = float(prop["y_m"]) * 1000.0
            else:
                continue

            # World -> vehicle frame (render.py convention: +d forward, +w right)
            dx = px_mm - x_mm
            dy = py_mm - y_mm
            d_mm = dx * cos_y + dy * sin_y
            w_mm = dx * sin_y - dy * cos_y
            if d_mm < self.min_dist_mm or d_mm > self.max_dist_mm:
                continue

            # Construct 3D bounding corners in world/vehicle frame
            prop_yaw = float(prop.get("yaw", prop.get("heading_rad", yaw)))
            rel_yaw = prop_yaw - yaw
            c_r = math.cos(rel_yaw)
            s_r = math.sin(rel_yaw)

            pts_3d: List[Tuple[float, float, float]] = []
            if kind == "digit":
                hl = float(prop.get("length_mm", 125.0)) * 0.5
                hw = float(prop.get("width_mm", 95.0)) * 0.5
                for dl in (-hl, hl):
                    for dw in (-hw, hw):
                        dd = dl * c_r - dw * s_r
                        ww = -(dl * s_r + dw * c_r)
                        pts_3d.append((w_mm + ww, d_mm + dd, 0.0))
            elif kind == "cone":
                r_base = float(prop.get("base_mm", 135.0)) * 0.5
                h_cone = float(prop.get("height_mm", 150.0))
                for dw in (-r_base, r_base):
                    for dd in (-r_base, r_base):
                        pts_3d.append((w_mm + dw, d_mm + dd, 0.0))
                pts_3d.append((w_mm, d_mm, h_cone))
            elif kind == "no_entry":
                hl = float(prop.get("length_mm", 240.0)) * 0.5
                hw = float(prop.get("width_mm", 130.0)) * 0.5
                for dl in (-hl, hl):
                    for dw in (-hw, hw):
                        dd = dl * c_r - dw * s_r
                        ww = -(dl * s_r + dw * c_r)
                        pts_3d.append((w_mm + ww, d_mm + dd, 0.0))
            else:  # yellow
                hl = float(prop.get("length_mm", 25.0)) * 0.5
                hw = float(prop.get("width_mm", 430.0)) * 0.5
                for dl in (-hl, hl):
                    for dw in (-hw, hw):
                        dd = dl * c_r - dw * s_r
                        ww = -(dl * s_r + dw * c_r)
                        pts_3d.append((w_mm + ww, d_mm + dd, 0.0))

            proj_uv: List[Tuple[float, float]] = []
            for pw, pd, pz in pts_3d:
                if pd <= 20.0:
                    continue
                uv = _project_point_3d(cam, pw, pd, pz)
                if uv is not None:
                    proj_uv.append(uv)
            if len(proj_uv) < 2:
                continue

            us = [p[0] for p in proj_uv]
            vs = [p[1] for p in proj_uv]
            u_min, u_max = min(us), max(us)
            v_min, v_max = min(vs), max(vs)

            raw_area = max(1.0, (u_max - u_min) * (v_max - v_min))
            x1 = _clamp(u_min, 0.0, float(img_w - 1))
            y1 = _clamp(v_min, 0.0, float(img_h - 1))
            x2 = _clamp(u_max, 0.0, float(img_w - 1))
            y2 = _clamp(v_max, 0.0, float(img_h - 1))
            bw = x2 - x1
            bh = y2 - y1
            if bw < self.min_box_size or bh < self.min_box_size:
                continue
            vis_frac = (bw * bh) / raw_area
            if vis_frac < 0.25:
                continue

            # Optional brief dropout/flicker per prop
            prop_key = str(prop.get("id", f"{kind}_{idx}_{round(px_mm)}_{round(py_mm)}"))
            rem = self._dropout_countdown.get(prop_key, 0)
            if rem > 0:
                self._dropout_countdown[prop_key] = rem - 1
                continue
            if self.flicker_prob > 0.0 and self.flicker_max_frames > 0:
                if float(self._rng.random()) < self.flicker_prob:
                    dur = int(self._rng.integers(1, self.flicker_max_frames + 1))
                    self._dropout_countdown[prop_key] = dur - 1
                    continue

            # Distance-dependent recall check
            dist_m = d_mm / 1000.0
            recall = self._distance_recall(dist_m, min(bw, bh))
            if float(self._rng.random()) > recall:
                continue

            # Pixel jitter & confidence noise
            if self.bbox_jitter_px > 0.0:
                jit = self._rng.normal(0.0, self.bbox_jitter_px, size=4)
                x1 = _clamp(x1 + float(jit[0]), 0.0, float(img_w - 2))
                y1 = _clamp(y1 + float(jit[1]), 0.0, float(img_h - 2))
                x2 = _clamp(x2 + float(jit[2]), x1 + self.min_box_size, float(img_w - 1))
                y2 = _clamp(y2 + float(jit[3]), y1 + self.min_box_size, float(img_h - 1))

            base_conf = self._distance_base_conf(dist_m)
            conf = base_conf + (
                float(self._rng.normal(0.0, self.conf_noise_std))
                if self.conf_noise_std > 0.0
                else 0.0
            )
            conf = _clamp(conf, 0.01, 0.99)
            if conf < self.conf_thresh:
                continue

            dets.append(
                _format_detection_dict(
                    class_name=class_name,
                    cls_id=cls_id,
                    conf=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    img_w=img_w,
                    img_h=img_h,
                    track_id=-1,
                    source="mock",
                )
            )

        if self.enable_tracker:
            dets = self.tracker.update(dets)
        return dets


# ---------------------------------------------------------------------------
# VisionFrontend (full 1:1 mirror of LineFollowerNode vision frontend)
# ---------------------------------------------------------------------------


class VisionFrontend:
    """Complete 1:1 Python OpenCV vision frontend mirroring `line_follower_node.cpp`.

    Usage:
        vf = VisionFrontend(params_dict, detector=MockNoisyDetector(seed=42))
        obs_for_brain, debug_overlay = vf.process(bgr_img, world_context=ctx)
    """

    def __init__(
        self,
        params: Optional[Union[Dict[str, Any], str, Path]] = None,
        detector: Optional[Any] = None,
    ) -> None:
        self.raw_params: Dict[str, Any] = unwrap_ros_params(params)
        self.detector = detector

        self.image_processor = ImageProcessorMirror()
        self.lane_geometry = LaneGeometryMirror()
        self.ground_projector = GroundProjectorMirror()
        self.curve_state = CurveAssistState()

        # Runtime switches & red fallback parameters (defaults match LineFollowerNode::declare_parameters)
        self.enable_semantic_hole_fill: bool = True
        self.enable_cv_red_fallback: bool = True
        self.red_h_low_1: int = 0
        self.red_h_high_1: int = 10
        self.red_h_low_2: int = 160
        self.red_h_high_2: int = 180
        self.red_s_min: int = 90
        self.red_v_min: int = 70
        self.red_cone_area_ratio_threshold: float = 0.0006
        self.crc_no_entry_pad_y_px: float = 0.0
        self.crc_cone_pad_y_px: float = 0.0
        self.crc_yellow_pad_y_px: float = 0.0
        self.default_lane_width_m: float = 0.430
        self.preferred_branch: int = 0

        self.update_params(self.raw_params)

    def _p(self, key: str, default: Any) -> Any:
        return self.raw_params.get(key, default)

    def update_params(self, params: Optional[Union[Dict[str, Any], str, Path]]) -> None:
        if params is not None:
            self.raw_params.update(unwrap_ros_params(params))

        self.enable_semantic_hole_fill = bool(self._p("enable_semantic_hole_fill", True))
        self.enable_cv_red_fallback = bool(self._p("enable_cv_red_fallback", True))
        self.crc_no_entry_pad_y_px = max(0.0, float(self._p("crc_no_entry_pad_y_px", 0.0)))
        self.crc_cone_pad_y_px = max(0.0, float(self._p("crc_cone_pad_y_px", 0.0)))
        self.crc_yellow_pad_y_px = max(0.0, float(self._p("crc_yellow_pad_y_px", 0.0)))
        self.default_lane_width_m = max(0.10, float(self._p("default_lane_width_m", 0.430)))

        # Red HSV thresholds
        r_h1_lo = _clamp_int(int(self._p("red_h_low_1", 0)), 0, 180)
        r_h1_hi = _clamp_int(int(self._p("red_h_high_1", 10)), 0, 180)
        if r_h1_lo > r_h1_hi:
            r_h1_lo, r_h1_hi = r_h1_hi, r_h1_lo
        r_h2_lo = _clamp_int(int(self._p("red_h_low_2", 160)), 0, 180)
        r_h2_hi = _clamp_int(int(self._p("red_h_high_2", 180)), 0, 180)
        if r_h2_lo > r_h2_hi:
            r_h2_lo, r_h2_hi = r_h2_hi, r_h2_lo
        self.red_h_low_1 = r_h1_lo
        self.red_h_high_1 = r_h1_hi
        self.red_h_low_2 = r_h2_lo
        self.red_h_high_2 = r_h2_hi
        self.red_s_min = _clamp_int(int(self._p("red_s_min", 90)), 0, 255)
        self.red_v_min = _clamp_int(int(self._p("red_v_min", 70)), 0, 255)
        self.red_cone_area_ratio_threshold = max(
            0.0, float(self._p("red_cone_area_ratio_threshold", 0.0006))
        )

        # Sync ImageProcessorConfig
        img_cfg = ImageProcessorConfig(
            roi_top_ratio=float(self._p("roi_top_ratio", 0.68)),
            roi_bottom_ratio=float(self._p("roi_bottom_ratio", 0.18)),
            yellow_roi_bottom_ratio=float(self._p("yellow_roi_bottom_ratio", 0.0)),
            gaussian_kernel=int(self._p("gaussian_kernel", 5)),
            binary_threshold=float(self._p("binary_threshold", 160.0)),
            morph_kernel=int(self._p("morph_kernel", 3)),
            morph_iterations=int(self._p("morph_iterations", 2)),
            min_contour_area=float(self._p("min_contour_area", 2500.0)),
            invert_binary=bool(self._p("invert_binary", False)),
            use_white_filter=bool(self._p("use_white_filter", True)),
            white_h_low=int(self._p("white_h_low", 0)),
            white_h_high=int(self._p("white_h_high", 180)),
            white_s_max=int(self._p("white_s_max", 45)),
            white_v_min=int(self._p("white_v_min", 80)),
            use_yellow_filter=bool(self._p("use_yellow_filter", True)),
            yellow_h_low=int(self._p("yellow_h_low", 20)),
            yellow_h_high=int(self._p("yellow_h_high", 40)),
            yellow_s_min=int(self._p("yellow_s_min", 80)),
            yellow_v_min=int(self._p("yellow_v_min", 120)),
            lane_seg_mode=str(self._p("lane_seg_mode", "relative_border")),
            blue_margin=int(self._p("blue_margin", 22)),
            lab_blue_b_max=int(self._p("lab_blue_b_max", 116)),
            black_block_size=int(self._p("black_block_size", 31)),
            black_c_offset=float(self._p("black_c_offset", 14.0)),
            black_v_max=int(self._p("black_v_max", 115)),
            lab_chroma_radius=float(self._p("lab_chroma_radius", 24.0)),
            lab_l_min_offset=float(self._p("lab_l_min_offset", 38.0)),
            hybrid_white_s_max=int(self._p("hybrid_white_s_max", 75)),
            hybrid_white_v_min=int(self._p("hybrid_white_v_min", 65)),
            enable_semantic_hole_fill=self.enable_semantic_hole_fill,
            semantic_hole_fill_dilate_px=int(self._p("semantic_hole_fill_dilate_px", 2)),
            yellow_detect_mode=str(self._p("yellow_detect_mode", "geometric_cv")),
            yellow_min_aspect_ratio=float(self._p("yellow_min_aspect_ratio", 2.5)),
            yellow_min_width_ratio=float(self._p("yellow_min_width_ratio", 0.20)),
            yellow_min_area_px=float(self._p("yellow_min_area_px", 120.0)),
            yellow_ahead_strip_height_px=int(self._p("yellow_ahead_strip_height_px", 28)),
            yellow_dead_end_blue_ratio=float(self._p("yellow_dead_end_blue_ratio", 0.35)),
            yellow_dead_end_dark_ratio=float(self._p("yellow_dead_end_dark_ratio", 0.20)),
            yellow_dead_end_combined_ratio=float(self._p("yellow_dead_end_combined_ratio", 0.50)),
        )
        self.image_processor.set_config(img_cfg)

        # Sync LaneGeometryConfig
        geo_cfg = LaneGeometryConfig(
            far_band_top_ratio=float(self._p("far_band_top_ratio", 0.28)),
            far_band_bottom_ratio=float(self._p("far_band_bottom_ratio", 0.48)),
            near_band_top_ratio=float(self._p("near_band_top_ratio", 0.55)),
            near_band_bottom_ratio=float(self._p("near_band_bottom_ratio", 0.80)),
            near_band_weight=float(self._p("near_band_weight", 0.60)),
            far_band_weight=float(self._p("far_band_weight", 0.40)),
            column_white_ratio=float(self._p("band_column_white_ratio", 0.40)),
            min_run_width=int(self._p("band_min_run_width", 18)),
            enable_continuous_curve_assist=bool(self._p("enable_continuous_curve_assist", True)),
            reverse_curve_min_error=float(self._p("reverse_curve_min_error", 0.12)),
            reverse_curve_threshold=float(self._p("reverse_curve_threshold", 0.35)),
            reverse_curve_confirm_frames=int(self._p("reverse_curve_confirm_frames", 3)),
            reverse_curve_clear_frames=int(self._p("reverse_curve_clear_frames", 4)),
            nominal_near_width_ratio=float(self._p("nominal_near_width_ratio", 0.68)),
            nominal_far_width_ratio=float(self._p("nominal_far_width_ratio", 0.48)),
            wide_road_ratio_threshold=float(self._p("wide_road_ratio_threshold", 0.84)),
            wide_band_min_reliability=float(self._p("wide_band_min_reliability", 0.18)),
            enable_width_adaptive_fusion=bool(self._p("enable_width_adaptive_fusion", True)),
            wide_lane_boundary_blend=float(self._p("wide_lane_boundary_blend", 0.45)),
            observation_filter_alpha=float(self._p("observation_filter_alpha", 0.85)),
            fork_min_gap_px=int(self._p("fork_min_gap_px", 14)),
            fork_min_branch_width_px=int(self._p("fork_min_branch_width_px", 18)),
        )
        self.lane_geometry.set_config(geo_cfg)

        # Sync GroundProjectorConfig
        proj_cfg = GroundProjectorConfig(
            use_homography=False,
            cam_height_m=float(self._p("cam_height_m", 0.11)),
            cam_pitch_deg=float(self._p("cam_pitch_deg", 12.5)),
            focal_px=float(self._p("focal_px", 325.0)),
            cx=float(self._p("cam_cx", 320.0)),
            cy=float(self._p("cam_cy", 240.0)),
            front_overhang_m=float(self._p("front_overhang_m", 0.06)),
        )
        self.ground_projector.set_config(proj_cfg)
        if bool(self._p("use_homography", False)):
            h_vec = self._p("homography_matrix", None)
            if isinstance(h_vec, (list, tuple)) and len(h_vec) == 9:
                self.ground_projector.set_homography_from_vector(h_vec, enable=True)

    def reset(self) -> None:
        self.curve_state = CurveAssistState()
        self.preferred_branch = 0
        if self.detector is not None and hasattr(self.detector, "reset"):
            self.detector.reset()

    def set_detector(self, detector: Optional[Any]) -> None:
        self.detector = detector

    def set_preferred_branch(self, preferred_branch: int) -> None:
        self.preferred_branch = int(preferred_branch)

    def detect_cv_red_in_lane(
        self,
        frame_bgr: np.ndarray,
        result: ProcessingResult,
        exclude_rects: Sequence[Tuple[int, int, int, int]] = (),
        now_sec: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Exact 1:1 mirror of `LineFollowerNode::detect_cv_red_in_lane`."""
        if (
            frame_bgr is None
            or frame_bgr.size == 0
            or result.binary_mask is None
            or result.binary_mask.size == 0
        ):
            return []

        img_h, img_w = frame_bgr.shape[:2]
        roi_top = _clamp_int(result.roi_top, 0, img_h - 1)
        roi_bottom = _clamp_int(result.roi_bottom, roi_top + 1, img_h)
        roi_h = roi_bottom - roi_top
        roi_w = img_w
        if (
            roi_h <= 4
            or roi_w <= 4
            or result.binary_mask.shape[0] != roi_h
            or result.binary_mask.shape[1] != roi_w
        ):
            return []

        # 1) Row-span fill of white lane interior (+4px border pad)
        has_white = result.binary_mask > 0
        row_any = has_white.any(axis=1)
        lane_span_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        if np.any(row_any):
            first_idx = np.argmax(has_white, axis=1)
            last_idx = (roi_w - 1) - np.argmax(has_white[:, ::-1], axis=1)
            valid_rows = np.flatnonzero(row_any & ((last_idx - first_idx) >= 16))
            for r in valid_rows:
                p0 = max(0, int(first_idx[r]) - 4)
                p1 = min(roi_w - 1, int(last_idx[r]) + 4)
                lane_span_mask[r, p0 : p1 + 1] = 255

        # 2) HSV red two-range threshold inside ROI gated by lane_span_mask
        roi_bgr = frame_bgr[roi_top:roi_bottom, 0:roi_w]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(
            hsv,
            np.array([self.red_h_low_1, self.red_s_min, self.red_v_min], dtype=np.uint8),
            np.array([self.red_h_high_1, 255, 255], dtype=np.uint8),
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([self.red_h_low_2, self.red_s_min, self.red_v_min], dtype=np.uint8),
            np.array([self.red_h_high_2, 255, 255], dtype=np.uint8),
        )
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.bitwise_and(red_mask, lane_span_mask)

        # 3) Mask out YOLO digit/yellow boxes (±4px)
        for ex_x, ex_y, ex_w, ex_h in exclude_rects:
            x1 = _clamp_int(int(ex_x) - 4, 0, roi_w)
            x2 = _clamp_int(int(ex_x + ex_w) + 4, 0, roi_w)
            y1 = _clamp_int(int(ex_y) - roi_top - 4, 0, roi_h)
            y2 = _clamp_int(int(ex_y + ex_h) - roi_top + 4, 0, roi_h)
            if x2 > x1 and y2 > y1:
                red_mask[y1:y2, x1:x2] = 0

        # 4) Open (3x3) + Close (11x9) morphology
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 9))
        raw_red_clean = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, k_open)
        grouped_red = cv2.morphologyEx(raw_red_clean, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(grouped_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_area = float(roi_w * roi_h)
        min_area_px = max(70.0, self.red_cone_area_ratio_threshold * roi_area)

        # Filter contours & merge fragmented hollow red no-entry boxes
        merged_rois: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area_px:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rw < 6 or rh < 6:
                continue
            touches_lr_border = (rx <= 1) or (rx + rw >= roi_w - 2)
            if touches_lr_border and rw < 16 and rh < 16:
                continue
            merged_rois.append((int(rx), int(ry), int(rw), int(rh)))

        did_merge = True
        max_merged_w = int(0.55 * float(roi_w))
        while did_merge and len(merged_rois) >= 2:
            did_merge = False
            for i in range(len(merged_rois)):
                if did_merge:
                    break
                ax, ay, aw, ah = merged_rois[i]
                for j in range(i + 1, len(merged_rois)):
                    bx, by, bw, bh = merged_rois[j]
                    dx = max(0, max(ax - (bx + bw), bx - (ax + aw)))
                    dy = max(0, max(ay - (by + bh), by - (ay + ah)))
                    if dx <= 24 and dy <= 16:
                        u = _rect_union(merged_rois[i], merged_rois[j])
                        if u[2] <= max_merged_w:
                            merged_rois[i] = u
                            merged_rois.pop(j)
                            did_merge = True
                            break

        out_boxes: List[Dict[str, Any]] = []
        max_v_px = float(max(0, img_h - 1))

        for rx, ry, rw, rh in merged_rois:
            # Reject off-track red objects whose base center lies outside lane span
            u_base_roi = _clamp_int(rx + rw // 2, 0, roi_w - 1)
            v_base_roi = _clamp_int(ry + rh - 1, 0, roi_h - 1)
            if lane_span_mask[v_base_roi, u_base_roi] == 0:
                continue

            img_x = rx
            img_y = ry + roi_top
            u_bot = float(img_x) + 0.5 * float(rw)
            v_raw = _clamp(float(img_y + rh), 0.0, max_v_px)
            ok, d_m, y_m = self.ground_projector.pixel_to_ground(u_bot, v_raw)
            if not ok:
                continue

            est_width_m = 0.135
            ok_l, _dl, yl = self.ground_projector.pixel_to_ground(float(img_x), v_raw)
            ok_r, _dr, yr = self.ground_projector.pixel_to_ground(float(img_x + rw), v_raw)
            if ok_l and ok_r:
                est_width_m = _clamp(abs(yl - yr), 0.10, 0.26)

            raw_red_px = cv2.countNonZero(raw_red_clean[ry : ry + rh, rx : rx + rw])
            fill_ratio = float(raw_red_px) / max(1.0, float(rw * rh))
            looks_no_entry = (fill_ratio < 0.42 and rw >= 18) or (est_width_m > 0.155)

            crc_pad_y = self.crc_no_entry_pad_y_px if looks_no_entry else self.crc_cone_pad_y_px
            v_bot = _clamp(float(img_y + rh) + crc_pad_y, 0.0, max_v_px)
            if abs(v_bot - v_raw) > 1e-6:
                ok, d_m, y_m = self.ground_projector.pixel_to_ground(u_bot, v_bot)
                if not ok:
                    continue
                ok_l, _dl, yl = self.ground_projector.pixel_to_ground(float(img_x), v_bot)
                ok_r, _dr, yr = self.ground_projector.pixel_to_ground(float(img_x + rw), v_bot)
                if ok_l and ok_r:
                    est_width_m = _clamp(abs(yl - yr), 0.10, 0.26)

            if d_m < 0.02 or d_m > 1.45 or abs(y_m) > 0.35:
                continue

            if looks_no_entry:
                half_wid_m = _clamp(0.5 * est_width_m, 0.065, 0.13)
                half_len_m = 0.065 if half_wid_m > 0.095 else 0.12
                cls_short = "no_entry"
                cls_full = "red_no_entry"
            else:
                half_wid_m = _clamp(0.5 * est_width_m, 0.0675, 0.10)
                half_len_m = 0.0675
                cls_short = "cone"
                cls_full = "red_cone"

            out_boxes.append(
                {
                    "x": int(img_x),
                    "y": int(img_y),
                    "w": int(rw),
                    "h": int(rh),
                    "x1": int(img_x),
                    "y1": int(img_y),
                    "x2": int(img_x + rw),
                    "y2": int(img_y + rh),
                    "class_name": cls_short,
                    "type": cls_full,
                    "is_no_entry": bool(looks_no_entry),
                    "confidence": 0.60,
                    "conf": 0.60,
                    "d_bumper_m": float(d_m),
                    "y_lat_m": float(y_m),
                    "half_len_m": float(half_len_m),
                    "half_wid_m": float(half_wid_m),
                    "fill_ratio": float(fill_ratio),
                    "est_width_m": float(est_width_m),
                    "stamp_sec": float(now_sec),
                }
            )

        return out_boxes

    def _compute_clamped_lane_bounds(
        self,
        result: ProcessingResult,
        active_branch: int,
    ) -> Tuple[float, float, float, str]:
        """Exact mirror of `line_follower_node.cpp:1547-1595` (single-side boundary clamp)."""
        obs = result.observation
        geo_cfg = self.lane_geometry.config
        roi_h = float(result.roi_bottom - result.roi_top)
        v_near = (
            float(result.roi_top)
            + 0.5 * (geo_cfg.near_band_top_ratio + geo_cfg.near_band_bottom_ratio) * roi_h
        )
        half_w_def = 0.5 * self.default_lane_width_m

        center_ok, _d_tmp, y_center_m = self.ground_projector.pixel_to_ground(
            float(result.centroid[0]), v_near
        )
        y_lane_left_m = (y_center_m + half_w_def) if center_ok else half_w_def
        y_lane_right_m = (y_center_m - half_w_def) if center_ok else -half_w_def
        clamp_mode = "default_center"

        left_proj_ok, _dl, y_l = (
            self.ground_projector.pixel_to_ground(obs.near_left_edge_px, v_near)
            if obs.near_left_valid
            else (False, 0.0, 0.0)
        )
        right_proj_ok, _dr, y_r = (
            self.ground_projector.pixel_to_ground(obs.near_right_edge_px, v_near)
            if obs.near_right_valid
            else (False, 0.0, 0.0)
        )

        if (
            not obs.has_fork_split
            and left_proj_ok
            and right_proj_ok
            and 0.30 <= (y_l - y_r) <= 0.52
            and (active_branch == 0 or (center_ok and abs(y_center_m - 0.5 * (y_l + y_r)) <= 0.04))
        ):
            y_lane_left_m = y_l
            y_lane_right_m = y_r
            clamp_mode = "measured_both"
        elif not obs.has_fork_split and active_branch == 0 and obs.near_band.valid:
            def_w = self.default_lane_width_m
            if (
                obs.near_band.left_clipped
                and not obs.near_band.right_clipped
                and right_proj_ok
                and -0.45 <= y_r <= 0.25
                and y_r > y_lane_right_m
            ):
                y_lane_right_m = y_r
                y_lane_left_m = y_r + def_w
                clamp_mode = "clamp_right_edge"
            elif (
                obs.near_band.right_clipped
                and not obs.near_band.left_clipped
                and left_proj_ok
                and -0.25 <= y_l <= 0.45
                and y_l < y_lane_left_m
            ):
                y_lane_left_m = y_l
                y_lane_right_m = y_l - def_w
                clamp_mode = "clamp_left_edge"

        return float(y_lane_left_m), float(y_lane_right_m), float(v_near), clamp_mode

    def _normalize_input_detections(
        self, raw_dets: Sequence[Dict[str, Any]], img_w: int, img_h: int
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for idx, d in enumerate(raw_dets):
            if not isinstance(d, dict):
                continue
            cname = str(d.get("class_name", d.get("type", d.get("label", ""))))
            conf = float(d.get("conf", d.get("confidence", 0.0)))
            if "x1" in d and "y1" in d and "x2" in d and "y2" in d:
                x1, y1, x2, y2 = float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])
            elif "x" in d and "y" in d and "w" in d and "h" in d:
                x1, y1 = float(d["x"]), float(d["y"])
                x2, y2 = x1 + float(d["w"]), y1 + float(d["h"])
            elif "bx" in d and "by" in d and "bw" in d and "bh" in d:
                x1, y1 = float(d["bx"]), float(d["by"])
                x2, y2 = x1 + float(d["bw"]), y1 + float(d["bh"])
            else:
                continue
            item = _format_detection_dict(
                class_name=cname,
                cls_id=int(d.get("cls_id", 0)),
                conf=conf,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                img_w=img_w,
                img_h=img_h,
                track_id=int(d.get("track_id", idx)),
                source=str(d.get("source", "external")),
            )
            normalized.append(item)
        return normalized

    def process(
        self,
        bgr_img: np.ndarray,
        detections: Optional[Sequence[Dict[str, Any]]] = None,
        world_context: Optional[Dict[str, Any]] = None,
        preferred_branch: Optional[int] = None,
        timestamp_sec: float = 0.0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Process a BGR camera frame and return `(obs_for_brain, debug_overlay_data)`."""
        if bgr_img is None or bgr_img.size == 0:
            raise ValueError("bgr_img is empty")
        img_h, img_w = bgr_img.shape[:2]
        active_branch = self.preferred_branch if preferred_branch is None else int(preferred_branch)

        # 1) Obtain detections (explicit override OR configured detector)
        if detections is not None:
            det_list = self._normalize_input_detections(detections, img_w, img_h)
        elif self.detector is not None:
            det_list = self.detector.detect(bgr_img, world_context=world_context)
        else:
            det_list = []

        # 2) Extract semantic hole-fill rects, red-exclusion rects, and YOLO yellow rects
        fill_white_rects: List[Tuple[int, int, int, int]] = []
        cv_red_exclude_rects: List[Tuple[int, int, int, int]] = []
        yolo_yellow_rects: List[Tuple[int, int, int, int]] = []

        max_v_px = float(max(0, img_h - 1))
        enriched_dets: List[Dict[str, Any]] = []
        for det in det_list:
            sem_type, digit_val = classify_detection_label(det["class_name"])
            rect = (int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"]))
            if sem_type in (SEMANTIC_BONUS_DIGIT, SEMANTIC_YELLOW_STOP_LINE):
                cv_red_exclude_rects.append(rect)
                if self.enable_semantic_hole_fill:
                    fill_white_rects.append(rect)
            if sem_type == SEMANTIC_YELLOW_STOP_LINE:
                yolo_yellow_rects.append(rect)

            # Project bottom-center for HUD & optional downstream use
            u_bot = float(rect[0]) + 0.5 * float(rect[2])
            crc_pad_y = 0.0
            if sem_type == SEMANTIC_RED_NO_ENTRY:
                crc_pad_y = self.crc_no_entry_pad_y_px
            elif sem_type == SEMANTIC_RED_CONE:
                crc_pad_y = self.crc_cone_pad_y_px
            elif sem_type == SEMANTIC_YELLOW_STOP_LINE:
                crc_pad_y = self.crc_yellow_pad_y_px
            v_bot = _clamp(float(rect[1] + rect[3]) + crc_pad_y, 0.0, max_v_px)
            ok_proj, d_m, y_m = self.ground_projector.pixel_to_ground(u_bot, v_bot)

            d_out = dict(det)
            d_out["semantic_type"] = sem_type
            d_out["digit"] = int(digit_val)
            d_out["proj_valid"] = bool(ok_proj)
            d_out["d_bumper_m"] = float(d_m) if ok_proj else 0.0
            d_out["y_lat_m"] = float(y_m) if ok_proj else 0.0
            enriched_dets.append(d_out)

        # 3) Image segmentation & lane geometry
        seg = self.image_processor.segment(bgr_img, fill_white_rects, yolo_yellow_rects)
        if seg is None:
            raise RuntimeError("ImageProcessorMirror.segment returned None")

        result = self.lane_geometry.evaluate(seg, self.curve_state, active_branch)
        obs = result.observation

        # 4) White-lane-gated CV red obstacle detection
        cv_red_boxes: List[Dict[str, Any]] = []
        if self.enable_cv_red_fallback and result.line_detected:
            cv_red_boxes = self.detect_cv_red_in_lane(
                bgr_img, result, cv_red_exclude_rects, timestamp_sec
            )

        # 5) Near-band lane boundary projection with single-side visible clamp
        if result.line_detected:
            y_lane_left_m, y_lane_right_m, v_near_px, clamp_mode = (
                self._compute_clamped_lane_bounds(result, active_branch)
            )
        else:
            roi_h = float(result.roi_bottom - result.roi_top)
            geo_cfg = self.lane_geometry.config
            v_near_px = (
                float(result.roi_top)
                + 0.5 * (geo_cfg.near_band_top_ratio + geo_cfg.near_band_bottom_ratio) * roi_h
            )
            half_w = 0.5 * self.default_lane_width_m
            y_lane_left_m, y_lane_right_m, clamp_mode = half_w, -half_w, "lost"

        # Yellow metric distance (for debug overlay)
        yellow_dist_valid = False
        yellow_d_bumper_m = 0.0
        if result.yellow_line_detected and result.yellow_is_geometric:
            v_yel_bot = _clamp(
                result.yellow_bottom_y_full + self.crc_yellow_pad_y_px, 0.0, max_v_px
            )
            yellow_dist_valid, yellow_d_bumper_m, _yy = self.ground_projector.pixel_to_ground(
                result.yellow_center_x_full, v_yel_bot
            )

        half_w_px = 0.5 * float(img_w)
        left_edge_norm = (
            _clamp((obs.near_left_edge_px - half_w_px) / half_w_px, -1.0, 1.0)
            if obs.near_left_valid
            else -1.0
        )
        right_edge_norm = (
            _clamp((obs.near_right_edge_px - half_w_px) / half_w_px, -1.0, 1.0)
            if obs.near_right_valid
            else 1.0
        )
        left_branch_error = (
            _clamp((obs.left_branch_center_px - half_w_px) / half_w_px, -1.0, 1.0)
            if obs.left_branch_valid
            else 0.0
        )
        right_branch_error = (
            _clamp((obs.right_branch_center_px - half_w_px) / half_w_px, -1.0, 1.0)
            if obs.right_branch_valid
            else 0.0
        )

        # Build obs_for_brain (all keys required by BrainBridge.step + full C++ parity fields)
        obs_for_brain: Dict[str, Any] = {
            "lane_valid": bool(result.line_detected),
            "lateral_error": float(obs.lateral_error),
            "near_error": float(obs.near_center_norm if obs.near_band.valid else 0.0),
            "far_error": float(obs.far_center_norm if obs.far_band.valid else 0.0),
            "near_valid": bool(obs.near_band.valid),
            "far_valid": bool(obs.far_band.valid),
            "heading_error": float(obs.heading_proxy),
            "curvature_proxy": float(obs.curvature_proxy),
            "curve_level": float(self.curve_state.curve_level),
            "reverse_curve_active": bool(self.curve_state.reverse_curve_active),
            "confidence": float(obs.confidence),
            "near_width_ratio": float(obs.near_width_ratio),
            "far_width_ratio": float(obs.far_width_ratio),
            "lane_width_ratio": float(obs.lane_width_ratio),
            "left_edge_norm": float(left_edge_norm),
            "right_edge_norm": float(right_edge_norm),
            "near_left_edge_px": float(obs.near_left_edge_px),
            "near_right_edge_px": float(obs.near_right_edge_px),
            "has_left_edge": bool(obs.near_left_valid),
            "has_right_edge": bool(obs.near_right_valid),
            "near_left_clipped": bool(obs.near_band.left_clipped),
            "near_right_clipped": bool(obs.near_band.right_clipped),
            "far_left_clipped": bool(obs.far_band.left_clipped),
            "far_right_clipped": bool(obs.far_band.right_clipped),
            "centroid_x_px": float(result.centroid[0]),
            "centroid_y_roi_px": float(result.centroid[1]),
            "centroid_y_full_px": float(result.centroid[1] + result.roi_top),
            "roi_top": int(result.roi_top),
            "roi_bottom": int(result.roi_bottom),
            "frame_width": int(img_w),
            "frame_height": int(img_h),
            "v_near_px": float(v_near_px),
            "y_lane_left_m": float(y_lane_left_m),
            "y_lane_right_m": float(y_lane_right_m),
            "lane_clamp_mode": clamp_mode,
            "fork_detected": bool(obs.has_fork_split),
            "left_branch_valid": bool(obs.left_branch_valid),
            "right_branch_valid": bool(obs.right_branch_valid),
            "left_branch_error": float(left_branch_error),
            "right_branch_error": float(right_branch_error),
            "left_branch_center_px": float(obs.left_branch_center_px),
            "right_branch_center_px": float(obs.right_branch_center_px),
            "yellow_detected": bool(result.yellow_line_detected),
            "yellow_is_geometric": bool(result.yellow_is_geometric),
            "yellow_from_yolo_hybrid": bool(result.yellow_from_yolo_hybrid),
            "yellow_bottom_y": float(result.yellow_bottom_y_full),
            "yellow_top_y": float(result.yellow_top_y_full),
            "yellow_center_x": float(result.yellow_center_x_full),
            "yellow_bar_width_px": float(result.yellow_bar_width_px),
            "yellow_width_ratio": float(result.yellow_bar_width_px / max(1.0, float(img_w))),
            "yellow_bbox": tuple(int(v) for v in result.yellow_bbox_full),
            "yellow_dead_end": bool(result.yellow_has_dead_end_ahead),
            "yellow_ahead_blue_ratio": float(result.yellow_ahead_blue_ratio),
            "yellow_ahead_dark_ratio": float(result.yellow_ahead_dark_ratio),
            "cv_red_boxes": cv_red_boxes,
            "detections": enriched_dets,
            "active_lane_seg_mode": str(result.active_lane_seg_mode),
        }

        roi_h_int = result.roi_bottom - result.roi_top
        geo_cfg = self.lane_geometry.config
        far_y0_full = result.roi_top + int(float(roi_h_int) * geo_cfg.far_band_top_ratio)
        far_y1_full = result.roi_top + int(float(roi_h_int) * geo_cfg.far_band_bottom_ratio)
        near_y0_full = result.roi_top + int(float(roi_h_int) * geo_cfg.near_band_top_ratio)
        near_y1_full = result.roi_top + int(float(roi_h_int) * geo_cfg.near_band_bottom_ratio)

        debug_overlay_data: Dict[str, Any] = {
            "binary_mask": result.binary_mask,
            "roi_top": int(result.roi_top),
            "roi_bottom": int(result.roi_bottom),
            "frame_width": int(img_w),
            "frame_height": int(img_h),
            "lane_seg_mode": str(result.active_lane_seg_mode),
            "line_detected": bool(result.line_detected),
            "bounding_box": tuple(int(v) for v in result.bounding_box),
            "centroid_px": (float(result.centroid[0]), float(result.centroid[1] + result.roi_top)),
            "near_band": {
                "valid": bool(obs.near_band.valid),
                "y0_full": int(near_y0_full),
                "y1_full": int(near_y1_full),
                "left_px": int(obs.near_band.left_px),
                "right_px": int(obs.near_band.right_px),
                "center_px": float(obs.near_band.center_px),
                "raw_center_px": float(obs.near_band.raw_center_px),
                "width_ratio": float(obs.near_band.width_ratio),
                "reliability": float(obs.near_band.reliability),
                "left_clipped": bool(obs.near_band.left_clipped),
                "right_clipped": bool(obs.near_band.right_clipped),
                "left_valid": bool(obs.near_band.left_valid),
                "right_valid": bool(obs.near_band.right_valid),
            },
            "far_band": {
                "valid": bool(obs.far_band.valid),
                "y0_full": int(far_y0_full),
                "y1_full": int(far_y1_full),
                "left_px": int(obs.far_band.left_px),
                "right_px": int(obs.far_band.right_px),
                "center_px": float(obs.far_band.center_px),
                "raw_center_px": float(obs.far_band.raw_center_px),
                "width_ratio": float(obs.far_band.width_ratio),
                "reliability": float(obs.far_band.reliability),
                "left_clipped": bool(obs.far_band.left_clipped),
                "right_clipped": bool(obs.far_band.right_clipped),
                "left_valid": bool(obs.far_band.left_valid),
                "right_valid": bool(obs.far_band.right_valid),
            },
            "fork": {
                "has_fork_split": bool(obs.has_fork_split),
                "left_branch_valid": bool(obs.left_branch_valid),
                "left_branch_center_px": float(obs.left_branch_center_px),
                "right_branch_valid": bool(obs.right_branch_valid),
                "right_branch_center_px": float(obs.right_branch_center_px),
                "preferred_branch": int(active_branch),
            },
            "edges": {
                "near_left_valid": bool(obs.near_left_valid),
                "near_left_edge_px": float(obs.near_left_edge_px),
                "near_right_valid": bool(obs.near_right_valid),
                "near_right_edge_px": float(obs.near_right_edge_px),
                "v_near_px": float(v_near_px),
                "y_lane_left_m": float(y_lane_left_m),
                "y_lane_right_m": float(y_lane_right_m),
                "clamp_mode": clamp_mode,
            },
            "curve": {
                "enable_continuous_curve_assist": bool(geo_cfg.enable_continuous_curve_assist),
                "reverse_curve_active": bool(self.curve_state.reverse_curve_active),
                "near_error": float(self.curve_state.near_error),
                "far_error": float(self.curve_state.far_error),
                "curve_level": float(self.curve_state.curve_level),
                "heading_proxy": float(obs.heading_proxy),
                "curvature_proxy": float(obs.curvature_proxy),
                "confidence": float(obs.confidence),
                "lane_width_ratio": float(obs.lane_width_ratio),
            },
            "yellow": {
                "detected": bool(result.yellow_line_detected),
                "is_geometric": bool(result.yellow_is_geometric),
                "from_yolo_hybrid": bool(result.yellow_from_yolo_hybrid),
                "bbox_full": tuple(int(v) for v in result.yellow_bbox_full),
                "bottom_y_full": float(result.yellow_bottom_y_full),
                "top_y_full": float(result.yellow_top_y_full),
                "center_x_full": float(result.yellow_center_x_full),
                "bar_width_px": float(result.yellow_bar_width_px),
                "dead_end_ahead": bool(result.yellow_has_dead_end_ahead),
                "ahead_blue_ratio": float(result.yellow_ahead_blue_ratio),
                "ahead_dark_ratio": float(result.yellow_ahead_dark_ratio),
                "dist_valid": bool(yellow_dist_valid),
                "d_bumper_m": float(yellow_d_bumper_m),
            },
            "cv_red_boxes": cv_red_boxes,
            "detections": enriched_dets,
        }

        return obs_for_brain, debug_overlay_data


def render_debug_overlay(
    bgr_img: np.ndarray,
    debug_data: Dict[str, Any],
    brain_info: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Render C++-style vision HUD overlay onto a copy of `bgr_img`."""
    overlay = bgr_img.copy()
    h, w = overlay.shape[:2]
    roi_top = int(debug_data.get("roi_top", 0))
    roi_bottom = int(debug_data.get("roi_bottom", h))
    mask = debug_data.get("binary_mask")

    if mask is not None and mask.size > 0:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        roi_slice = overlay[roi_top:roi_bottom, 0:w]
        if roi_slice.shape == mask_bgr.shape:
            overlay[roi_top:roi_bottom, 0:w] = cv2.addWeighted(roi_slice, 0.65, mask_bgr, 0.35, 0.0)

    # ROI border & center line
    cv2.line(overlay, (0, roi_top), (w - 1, roi_top), (200, 160, 0), 1)
    cv2.line(overlay, (0, roi_bottom - 1), (w - 1, roi_bottom - 1), (200, 160, 0), 1)
    cv2.line(overlay, (w // 2, roi_top), (w // 2, roi_bottom), (0, 255, 255), 1)

    # Near & far bands
    for band_key, col in (("far_band", (255, 200, 0)), ("near_band", (0, 220, 120))):
        b = debug_data.get(band_key, {})
        if b.get("valid"):
            y_mid = (int(b["y0_full"]) + int(b["y1_full"])) // 2
            cv2.line(overlay, (int(b["left_px"]), y_mid), (int(b["right_px"]), y_mid), col, 2)
            cv2.circle(overlay, (int(round(b["center_px"])), y_mid), 5, col, -1)

    # Centroid & edges
    if debug_data.get("line_detected"):
        cx, cy = debug_data.get("centroid_px", (w * 0.5, roi_top + 20.0))
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), 6, (0, 0, 255), -1)

        edges = debug_data.get("edges", {})
        vy = _clamp_int(int(round(edges.get("v_near_px", roi_top + 20))), 0, h - 1)
        if edges.get("near_left_valid"):
            lx = _clamp_int(int(round(edges["near_left_edge_px"])), 0, w - 1)
            cv2.line(overlay, (lx, max(0, vy - 12)), (lx, min(h - 1, vy + 12)), (255, 255, 0), 2)
        if edges.get("near_right_valid"):
            rx = _clamp_int(int(round(edges["near_right_edge_px"])), 0, w - 1)
            cv2.line(overlay, (rx, max(0, vy - 12)), (rx, min(h - 1, vy + 12)), (255, 255, 0), 2)

    # Fork markers
    fork = debug_data.get("fork", {})
    if fork.get("has_fork_split"):
        fy = _clamp_int(roi_top + int((roi_bottom - roi_top) * 0.42), 0, h - 1)
        pbr = int(fork.get("preferred_branch", 0))
        if fork.get("left_branch_valid"):
            bx = _clamp_int(int(round(fork["left_branch_center_px"])), 0, w - 1)
            cv2.circle(overlay, (bx, fy), 6, (0, 255, 0) if pbr < 0 else (200, 200, 0), 2)
        if fork.get("right_branch_valid"):
            bx = _clamp_int(int(round(fork["right_branch_center_px"])), 0, w - 1)
            cv2.circle(overlay, (bx, fy), 6, (0, 255, 0) if pbr > 0 else (200, 200, 0), 2)

    # Yellow box & bottom edge
    yel = debug_data.get("yellow", {})
    if yel.get("detected") and yel.get("is_geometric"):
        yx, yy, yw, yh = yel.get("bbox_full", (0, 0, 0, 0))
        if yw > 0 and yh > 0:
            cv2.rectangle(overlay, (yx, yy), (yx + yw, yy + yh), (0, 255, 255), 2)
            yb = _clamp_int(int(round(yel.get("bottom_y_full", yy + yh))), 0, h - 1)
            cv2.line(overlay, (yx, yb), (yx + yw, yb), (0, 165, 255), 2)
        lbl = f"YELLOW d={yel.get('d_bumper_m', 0.0):.2f}m" + (
            " [DEAD_END]" if yel.get("dead_end_ahead") else ""
        )
        cv2.putText(overlay, lbl, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # CV red obstacle boxes
    for rb in debug_data.get("cv_red_boxes", []):
        rx, ry, rw, rh = int(rb["x"]), int(rb["y"]), int(rb["w"]), int(rb["h"])
        is_ne = rb.get("class_name") == "no_entry"
        col = (255, 0, 255) if is_ne else (0, 0, 255)
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), col, 2)
        txt = f"[CV]{rb['class_name']} d={rb['d_bumper_m']:.2f} y={rb['y_lat_m']:+.2f}"
        cv2.putText(
            overlay, txt, (_clamp_int(rx, 4, max(4, w - 180)), max(16, ry - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1,
        )

    # Detector boxes (YOLO / Mock)
    for det in debug_data.get("detections", []):
        dx, dy, dw, dh = int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])
        src_tag = "Y" if det.get("source") == "yolo" else "M"
        col = (255, 200, 0) if det.get("semantic_type") == SEMANTIC_BONUS_DIGIT else (0, 255, 0)
        cv2.rectangle(overlay, (dx, dy), (dx + dw, dy + dh), col, 2)
        txt = f"[{src_tag}]{det['class_name']}#{det.get('track_id', -1)} {det['conf']:.2f}"
        cv2.putText(
            overlay, txt, (_clamp_int(dx, 4, max(4, w - 160)), max(16, dy - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1,
        )

    # Top status text
    curve = debug_data.get("curve", {})
    mode_txt = "REV_CURVE" if curve.get("reverse_curve_active") else "NORMAL"
    line1 = (
        f"{debug_data.get('lane_seg_mode', '?')} | {mode_txt} "
        f"near={curve.get('near_error', 0.0):+.2f} far={curve.get('far_error', 0.0):+.2f} "
        f"hdg={curve.get('heading_proxy', 0.0):+.2f} conf={curve.get('confidence', 0.0):.2f} "
        f"w={curve.get('lane_width_ratio', 0.0):.2f}"
    )
    cv2.putText(overlay, line1, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    if brain_info:
        line2 = (
            f"state={brain_info.get('state_name', '?')} "
            f"v={brain_info.get('v_cmd', 0.0):.2f} w={brain_info.get('w_cmd', 0.0):+.2f} "
            f"obs={brain_info.get('obstacle_count', 0)} "
            f"digit={brain_info.get('active_digit', 0)} "
            f"cbf={int(bool(brain_info.get('cbf_active', False)))}"
        )
        cv2.putText(overlay, line2, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

    return overlay


# ---------------------------------------------------------------------------
# Self-test (runs directly via `python tools/sim/vision_mirror.py`)
# ---------------------------------------------------------------------------


def _run_selftest() -> int:
    import time

    try:
        from .render import TrackRenderer
    except Exception:
        from render import TrackRenderer  # type: ignore

    tracks_dir = Path(__file__).resolve().parent / "tracks"
    configs_dir = REPO_ROOT / "tools" / "config"

    cases = [
        ("fork_s", tracks_dir / "fork_s.json", configs_dir / "finals_fork.yaml"),
        ("roundabout", tracks_dir / "roundabout.json", configs_dir / "finals_roundabout.yaml"),
    ]

    print("=== VisionFrontend & Dual-Mode Detector Self-Test ===")
    for name, track_json, cfg_yaml in cases:
        renderer = TrackRenderer(
            str(track_json), random_yellow=2, cones=2, no_entry=2, prop_seed=42
        )
        params = unwrap_ros_params(cfg_yaml)
        cl = renderer.centerline_mm
        s_arr = renderer.s_m

        # Build synthetic world_props list including digits for MockNoisyDetector
        world_props: List[Dict[str, Any]] = []
        for c in renderer.injected_props["cones"]:
            world_props.append({"type": "cone", **c})
        for ne in renderer.injected_props["no_entry"]:
            world_props.append({"type": "no_entry", **ne})
        for yl in renderer.injected_props["random_yellow"]:
            p, _t, _n = renderer._pose_at_s(yl["s_m"])
            world_props.append({"type": "yellow_line", "x_mm": float(p[0]), "y_mm": float(p[1]), **yl})
        # Place 2 synthetic bonus digits ahead of start pose
        for dig_val, dig_s, lat in ((5, 0.55, -70.0), (2, 0.70, 75.0)):
            p, t, n_left = renderer._pose_at_s(dig_s)
            pos = p + n_left * lat
            world_props.append({
                "type": "digit",
                "value": dig_val,
                "s_m": dig_s,
                "x_mm": float(pos[0]),
                "y_mm": float(pos[1]),
                "yaw": float(math.atan2(t[1], t[0])),
            })

        mock_det = MockNoisyDetector(seed=42, detect_digits=True, detect_obstacles=False)
        vf = VisionFrontend(params, detector=mock_det)

        # Sample frames along track (start, cone, no_entry, yellow, curve)
        sample_indices = [0, 20, 50, 100, 160]
        # Add poses ~0.55m before injected cones, no_entry, and yellow lines
        extra_s = (
            [max(0.0, c["s_m"] - 0.55) for c in renderer.injected_props["cones"][:1]]
            + [max(0.0, ne["s_m"] - 0.55) for ne in renderer.injected_props["no_entry"][:1]]
            + [max(0.0, yl["s_m"] - 0.55) for yl in renderer.injected_props["random_yellow"][:1]]
        )
        for es in extra_s:
            sample_indices.append(int(np.argmin(np.abs(s_arr - es))))

        valid_cnt = 0
        yellow_cnt = 0
        red_cnt = 0
        mock_digit_cnt = 0
        vf_ms = 0.0

        for idx in sample_indices:
            i0 = min(idx, len(cl) - 3)
            x0, y0 = float(cl[i0][0]), float(cl[i0][1])
            x1, y1 = float(cl[i0 + 2][0]), float(cl[i0 + 2][1])
            yaw = math.atan2(y1 - y0, x1 - x0)
            frame = renderer.render(x0, y0, yaw)
            ctx = {"pose": (x0, y0, yaw), "world_props": world_props, "cam": renderer.cam}
            t_vf0 = time.perf_counter()
            obs, dbg = vf.process(frame, world_context=ctx, timestamp_sec=float(idx) * 0.05)
            vf_ms += (time.perf_counter() - t_vf0) * 1000.0
            valid_cnt += int(obs["lane_valid"])
            yellow_cnt += int(obs["yellow_detected"])
            red_cnt += len(obs["cv_red_boxes"])
            mock_digit_cnt += sum(1 for d in obs["detections"] if d["semantic_type"] == SEMANTIC_BONUS_DIGIT)

        print(
            f"[{name}] Mock+CV ({len(sample_indices)} frames, vision={vf_ms:.1f}ms, "
            f"{vf_ms/len(sample_indices):.2f}ms/f): "
            f"lane_valid={valid_cnt}/{len(sample_indices)} yellow_hits={yellow_cnt} "
            f"cv_red_hits={red_cnt} mock_digits={mock_digit_cnt}"
        )

        # Test all 3 segmentation modes on start frame
        x0, y0 = float(cl[0][0]), float(cl[0][1])
        x1, y1 = float(cl[2][0]), float(cl[2][1])
        yaw0 = math.atan2(y1 - y0, x1 - x0)
        frame0 = renderer.render(x0, y0, yaw0)
        for mode in ("relative_border", "hybrid", "legacy_hsv"):
            vf_m = VisionFrontend({**params, "lane_seg_mode": mode})
            obs_m, _ = vf_m.process(frame0)
            print(
                f"  mode={mode:<15} valid={obs_m['lane_valid']} "
                f"err={obs_m['lateral_error']:+.3f} conf={obs_m['confidence']:.3f} "
                f"lane_y=[{obs_m['y_lane_right_m']:+.3f},{obs_m['y_lane_left_m']:+.3f}] ({obs_m['lane_clamp_mode']})"
            )

    # Test RealYoloDetector load + GPU inference on rendered frame with drawn digit
    print("\n--- Testing RealYoloDetector ---")
    yolo_det = RealYoloDetector(weights_path=DEFAULT_YOLO_WEIGHTS, conf_thresh=0.25)
    print(f"Loaded RealYoloDetector on device={yolo_det.device}, names={yolo_det.names}")
    r_fork = TrackRenderer(str(tracks_dir / "fork_s.json"))
    cl = r_fork.centerline_mm
    x0, y0 = float(cl[0][0]), float(cl[0][1])
    x1, y1 = float(cl[2][0]), float(cl[2][1])
    yaw0 = math.atan2(y1 - y0, x1 - x0)
    frame_y = r_fork.render(x0, y0, yaw0)
    # Warmup + timed inference
    yolo_det.warmup()
    t0 = time.perf_counter()
    dets_y = yolo_det.detect(frame_y)
    dt_y_ms = (time.perf_counter() - t0) * 1000.0
    vf_y = VisionFrontend(unwrap_ros_params(configs_dir / "finals_fork.yaml"), detector=yolo_det)
    obs_y, dbg_y = vf_y.process(frame_y)
    print(
        f"RealYoloDetector inference: {dt_y_ms:.2f}ms on {yolo_det.device}, "
        f"dets={len(dets_y)}, lane_valid={obs_y['lane_valid']}, conf={obs_y['confidence']:.3f}"
    )

    # Also test RealYoloDetector on a synthetic digit card rendered onto the white lane
    frame_dig = frame_y.copy()
    cv2.putText(frame_dig, "5", (295, 340), cv2.FONT_HERSHEY_DUPLEX, 2.5, (140, 60, 20), 6, cv2.LINE_AA)
    dets_dig = yolo_det.detect(frame_dig)
    print(f"RealYoloDetector synthetic '5' test: dets={[(d['class_name'], round(d['conf'], 3), (d['x'], d['y'], d['w'], d['h'])) for d in dets_dig]}")

    print("=== Self-Test PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_selftest())
