#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""赛道鸟瞰图 -> 相机视角渲染（含可选光照扰动与决赛道具注入）。

相机模型（实测标定值）：
    h      = 110 mm   镜头高度
    alpha  = 12.5 deg 俯角（光轴向下）
    f_px   = 325 px   像素焦距（由地平线落在 roi_top=168 反推）

地面点 (w, d) 为车体系：w 向右、d 向前，单位 mm。
投影：
    u = cx + f * w / d
    v = cy + f * (h/d - tan(alpha)) / (1 + (h/d) * tan(alpha))
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

BG_BGR = (166, 85, 37)      # 地图蓝底
WHITE_BGR = (255, 255, 255)
YELLOW_BGR = (0, 215, 255)
RED_BGR = (25, 25, 230)

COLOR_CAST_PRESETS: Dict[str, Tuple[float, float, float]] = {
    "none": (1.0, 1.0, 1.0),
    "warm": (0.86, 1.00, 1.14),
    "cool": (1.14, 1.00, 0.88),
    "yellowed": (0.80, 0.98, 1.06),
}


def parse_color_cast(spec: Optional[Union[str, Sequence[float]]]) -> Tuple[float, float, float]:
    """解析色温/偏色参数：支持预设名 ('warm'/'cool'/'yellowed'/'none') 或 'b,g,r' 三元组。"""
    if spec is None:
        return (1.0, 1.0, 1.0)
    if isinstance(spec, str):
        s = spec.strip().lower()
        if not s or s == "none":
            return (1.0, 1.0, 1.0)
        if s in COLOR_CAST_PRESETS:
            return COLOR_CAST_PRESETS[s]
        parts = [float(x) for x in s.replace(" ", ",").split(",") if x]
        if len(parts) != 3:
            raise ValueError(f"color_cast 需为预设名或 3 个 BGR 系数，收到: {spec}")
        return (parts[0], parts[1], parts[2])
    vals = [float(x) for x in spec]
    if len(vals) != 3:
        raise ValueError(f"color_cast 需为 3 元组，收到: {spec}")
    return (vals[0], vals[1], vals[2])


def point_to_polyline_distance_mm(x_mm: float, y_mm: float, centerline_mm: np.ndarray) -> float:
    """计算地面点 (x_mm, y_mm) 到折线中心线的精确垂直最短距离（单位 mm）。"""
    pts = np.asarray(centerline_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 0.0
    if pts.shape[0] == 1:
        return float(math.hypot(x_mm - pts[0, 0], y_mm - pts[0, 1]))

    p = np.array([float(x_mm), float(y_mm)], dtype=np.float64)
    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    ap = p - a
    ab_len2 = np.sum(ab * ab, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(ab_len2 > 1e-12, np.sum(ap * ab, axis=1) / ab_len2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + ab * t[:, None]
    dists = np.hypot(proj[:, 0] - p[0], proj[:, 1] - p[1])
    return float(np.min(dists))


class CameraModel:
    def __init__(self, h_mm=110.0, alpha_deg=12.5, f_px=325.0,
                 width=640, height=480):
        self.h = float(h_mm)
        self.alpha = math.radians(alpha_deg)
        self.f = float(f_px)
        self.w = int(width)
        self.h_img = int(height)
        self.cx = self.w / 2.0
        self.cy = self.h_img / 2.0
        self.tan_a = math.tan(self.alpha)

    def wd_to_uv(self, w_mm, d_mm):
        """车体系地面点 -> 像素。"""
        denom = d_mm + self.h * self.tan_a
        u = (self.f * w_mm + self.cx * denom) / denom
        v = ((self.cy - self.f * self.tan_a) * d_mm
             + self.h * (self.cy * self.tan_a + self.f)) / denom
        return u, v

    def wdz_to_uv(self, w_mm, d_mm, z_mm=0.0):
        """车体系 3D 点 (w_mm 向右, d_mm 向前, z_mm 向上) -> 像素 (u, v)。"""
        dz = self.h - float(z_mm)
        denom = d_mm + dz * self.tan_a
        u = self.cx + self.f * w_mm / denom
        v = self.cy + self.f * (dz - d_mm * self.tan_a) / denom
        return u, v

    def distance_at_row(self, row):
        """图像第 row 行对应的地面距离(mm)；地平线以上返回 inf。"""
        horizon = self.cy - self.f * self.tan_a
        if row <= horizon + 1e-6:
            return float("inf")
        A = self.cy - self.f * self.tan_a
        B = self.h * (self.cy * self.tan_a + self.f)
        return (B - row * self.h * self.tan_a) / (row - A)

    def preview_table(self, rows):
        return [(r, self.distance_at_row(r)) for r in rows]


class TrackRenderer:
    """把 tracks/<name>.png 当作纹理，按位姿裁剪并投影到相机视角。"""

    def __init__(
        self,
        track_json,
        cam=None,
        max_d_mm=5000.0,
        random_yellow: int = 0,
        cones: int = 0,
        no_entry: int = 0,
        prop_seed: int = 42,
    ):
        with open(track_json, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        base = os.path.splitext(track_json)[0]
        png_path = base + ".png"
        if not os.path.isfile(png_path):
            raise FileNotFoundError(png_path)
        buf = np.fromfile(png_path, dtype=np.uint8)
        self.map_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if self.map_bgr is None:
            raise FileNotFoundError(png_path)
        self.base_map_bgr = self.map_bgr.copy()
        self.mpp = float(self.meta["mm_per_px"])
        self.H, self.W = self.map_bgr.shape[:2]
        self.lane_width_mm = float(self.meta.get("lane_width_mm", 430.0))
        self.centerline_mm = np.asarray(self.meta["centerline_mm"], dtype=np.float64)
        self.s_m = np.asarray(self.meta["s_m"], dtype=np.float64)
        self.cam = cam or CameraModel()
        self.injected_props: Dict[str, List[dict]] = {
            "random_yellow": [],
            "cones": [],
            "no_entry": [],
        }

        if random_yellow > 0 or cones > 0 or no_entry > 0:
            self.inject_finals_props(
                random_yellow=random_yellow,
                cones=cones,
                no_entry=no_entry,
                seed=prop_seed,
            )

        uu, vv = np.meshgrid(np.arange(self.cam.w, dtype=np.float32),
                             np.arange(self.cam.h_img, dtype=np.float32))
        A = self.cam.cy - self.cam.f * self.cam.tan_a
        B = self.cam.h * (self.cam.cy * self.cam.tan_a + self.cam.f)
        denom_v = vv - A
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (B - vv * self.cam.h * self.cam.tan_a) / denom_v
        d = np.where(denom_v > 1e-3, d, np.nan)
        d[(d < 0) | (d > max_d_mm)] = np.nan
        w = (uu - self.cam.cx) * (d + self.cam.h * self.cam.tan_a) / self.cam.f
        self.grid_d = d.astype(np.float32)
        self.grid_w = w.astype(np.float32)

        # 预计算标准化网格供定向眩光叠加
        yy_g, xx_g = np.mgrid[0:self.cam.h_img, 0:self.cam.w].astype(np.float32)
        self._glare_field = np.exp(
            -((xx_g - self.cam.w * 0.55) ** 2 + (yy_g - self.cam.h_img * 0.62) ** 2)
            / (2.0 * (0.28 * self.cam.w) ** 2)
        )[:, :, None].astype(np.float32)

        finite = np.isfinite(d)
        print(f"[render] grid d {np.nanmin(d):.0f}..{np.nanmax(d):.0f}mm  "
              f"w {np.nanmin(w):.0f}..{np.nanmax(w):.0f}mm  "
              f"valid={finite.mean()*100:.0f}%", flush=True)

    def _mm_to_px(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        px = int(round(x_mm / self.mpp))
        py = int(round((self.H - 1) - y_mm / self.mpp))
        return px, py

    def _pose_at_s(self, s_val: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = int(np.argmin(np.abs(self.s_m - s_val)))
        i0 = max(0, idx - 1)
        i1 = min(len(self.centerline_mm) - 1, idx + 1)
        p = self.centerline_mm[idx]
        t = self.centerline_mm[i1] - self.centerline_mm[i0]
        tn = float(np.linalg.norm(t)) + 1e-9
        t = t / tn
        n_left = np.array([-t[1], t[0]], dtype=np.float64)
        return p, t, n_left

    def inject_finals_props(
        self,
        random_yellow: int = 0,
        cones: int = 0,
        no_entry: int = 0,
        seed: int = 42,
    ) -> Dict[str, List[dict]]:
        """在鸟瞰赛道纹理上注入随机黄线、红锥 (13.5x13.5cm) 和空心红禁行区 (24x13cm)。"""
        self.map_bgr = self.base_map_bgr.copy()
        self.injected_props = {"random_yellow": [], "cones": [], "no_entry": []}
        rng = random.Random(seed)
        total_s = float(self.s_m[-1])
        existing_y = list(self.meta.get("yellow_lines_s_m", []))
        placed_s: List[float] = list(existing_y)

        def pick_s(min_gap: float = 0.65) -> Optional[float]:
            for _ in range(80):
                sv = rng.uniform(0.8, max(1.0, total_s - 0.8))
                if all(abs(sv - prev) >= min_gap for prev in placed_s):
                    placed_s.append(sv)
                    return sv
            return None

        # 1) 随机黄线（宽 25mm 横贯赛道）
        half_lane = 0.5 * self.lane_width_mm
        for _ in range(max(0, int(random_yellow))):
            sv = pick_s(min_gap=0.70)
            if sv is None:
                break
            p, _t, n_left = self._pose_at_s(sv)
            a = self._mm_to_px(*(p + n_left * half_lane))
            b = self._mm_to_px(*(p - n_left * half_lane))
            thick_px = max(2, int(round(25.0 / self.mpp)))
            cv2.line(self.map_bgr, a, b, YELLOW_BGR, thick_px)
            self.injected_props["random_yellow"].append({"s_m": round(sv, 3)})

        # 2) 红锥（135mm x 135mm，偏离中心 ±90..135mm，留出可行走廊）
        for _ in range(max(0, int(cones))):
            sv = pick_s(min_gap=0.55)
            if sv is None:
                break
            p, _t, n_left = self._pose_at_s(sv)
            side = rng.choice([-1.0, 1.0])
            lat_mm = side * rng.uniform(85.0, max(95.0, half_lane - 85.0))
            center_mm = p + n_left * lat_mm
            cx, cy = self._mm_to_px(center_mm[0], center_mm[1])
            r_px = max(4, int(round(67.5 / self.mpp)))
            cv2.circle(self.map_bgr, (cx, cy), r_px, RED_BGR, -1)
            cv2.circle(self.map_bgr, (cx, cy), max(2, r_px // 3), (235, 235, 235), -1)
            self.injected_props["cones"].append({
                "s_m": round(sv, 3), "lat_mm": round(lat_mm, 1),
                "x_mm": round(float(center_mm[0]), 1), "y_mm": round(float(center_mm[1]), 1),
            })

        # 3) 空心红禁行区（240mm 沿车道 × 130mm 宽，10mm 红色空心边框）
        for _ in range(max(0, int(no_entry))):
            sv = pick_s(min_gap=0.65)
            if sv is None:
                break
            p, t, n_left = self._pose_at_s(sv)
            side = rng.choice([-1.0, 1.0])
            lat_mm = side * rng.uniform(95.0, max(105.0, half_lane - 80.0))
            center_mm = p + n_left * lat_mm
            hl, hw = 120.0, 65.0  # 240x130 mm
            corners_mm = [
                center_mm - t * hl - n_left * hw,
                center_mm + t * hl - n_left * hw,
                center_mm + t * hl + n_left * hw,
                center_mm - t * hl + n_left * hw,
            ]
            poly = np.array([self._mm_to_px(c[0], c[1]) for c in corners_mm], dtype=np.int32)
            border_px = max(2, int(round(10.0 / self.mpp)))
            cv2.polylines(self.map_bgr, [poly], True, RED_BGR, border_px, cv2.LINE_AA)
            self.injected_props["no_entry"].append({
                "s_m": round(sv, 3), "lat_mm": round(lat_mm, 1),
                "x_mm": round(float(center_mm[0]), 1), "y_mm": round(float(center_mm[1]), 1),
            })

        return self.injected_props

    def distance_to_centerline_mm(self, x_mm: float, y_mm: float) -> float:
        """返回车辆位置 (x_mm, y_mm) 到赛道中心线的精确最短距离（mm）。"""
        return point_to_polyline_distance_mm(x_mm, y_mm, self.centerline_mm)

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
        """返回 640x480 BGR 相机图（支持亮度增益、色温偏色、眩光与模糊扰动）。"""
        c, s = math.cos(yaw), math.sin(yaw)
        d, w = self.grid_d, self.grid_w
        x = x_mm + d * c + w * s
        y = y_mm + d * s - w * c
        mx = np.nan_to_num((x / self.mpp).astype(np.float32), nan=-1e6)
        my = np.nan_to_num(((self.H - 1) - y / self.mpp).astype(np.float32), nan=-1e6)
        out = cv2.remap(self.map_bgr, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=BG_BGR)
        black = (out == 0).all(axis=2)
        if black.any():
            out[black] = BG_BGR
        if blur > 0:
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


def _selftest(track, out_png):
    r = TrackRenderer(track)
    s = r.meta["centerline_mm"]
    x0, y0 = s[0]
    x1, y1 = s[2]
    yaw = math.atan2(y1 - y0, x1 - x0)
    cam = r.cam
    print("preview table (row -> mm):")
    for row, d in cam.preview_table([168, 200, 240, 300, 384]):
        print(f"  v={row:3d} -> {'inf' if d == float('inf') else f'{d:.0f} mm'}")
    img = r.render(x0, y0, yaw)
    cv2.imwrite(out_png, img)
    print(f"saved {out_png}  size={img.shape}  mean={img.mean():.1f}")


if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tracks", "fork_s.json")
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/aic_maps/render_test.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _selftest(tk, out)
