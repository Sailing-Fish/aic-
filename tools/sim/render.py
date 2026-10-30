#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""赛道鸟瞰图 -> 相机视角渲染。

相机模型（实测标定值）：
    h      = 110 mm   镜头高度
    alpha  = 12.5 deg 俯角（光轴向下）
    f_px   = 325 px   像素焦距（由地平线落在 roi_top=168 反推）

地面点 (w, d) 为车体系：w 向右、d 向前，单位 mm。
投影：
    u = cx + f * w / d
    v = cy + f * (h/d - tan(alpha)) / (1 + (h/d) * tan(alpha))
齐次形式可写成 3x3 单应，因此整幅图一次 warpPerspective 就能出。
"""
import json
import math
import os

import cv2
import numpy as np

BG_BGR = (166, 85, 37)      # 地图蓝底
WHITE_BGR = (255, 255, 255)
YELLOW_BGR = (0, 215, 255)


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

    def distance_at_row(self, row):
        """图像第 row 行对应的地面距离(mm)；地平线以上返回 inf。"""
        # v = 168 附近是地平线
        horizon = self.cy - self.f * self.tan_a
        if row <= horizon + 1e-6:
            return float("inf")
        # 由 v 反解 d：v*(d + h*tan) = A*d + B, A = cy - f*tan, B = h*(cy*tan + f)
        A = self.cy - self.f * self.tan_a
        B = self.h * (self.cy * self.tan_a + self.f)
        return (B - row * self.h * self.tan_a) / (row - A)

    def preview_table(self, rows):
        return [(r, self.distance_at_row(r)) for r in rows]


class TrackRenderer:
    """把 tracks/<name>.png 当作纹理，按位姿裁剪并投影到相机视角。"""

    def __init__(self, track_json, cam=None, max_d_mm=5000.0):
        # 逐像素反向映射(remap)：先预计算每个像素对应的车体地面点 (w,d)，
        # 每帧只把它按位姿旋转平移到地图像素坐标。相比 warpPerspective，
        # 角点落到地图外时不会整幅崩掉。
        with open(track_json, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        base = os.path.splitext(track_json)[0]
        self.map_bgr = cv2.imread(base + ".png")
        if self.map_bgr is None:
            raise FileNotFoundError(base + ".png")
        self.mpp = float(self.meta["mm_per_px"])
        self.H, self.W = self.map_bgr.shape[:2]
        self.cam = cam or CameraModel()

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
        finite = np.isfinite(d)
        print(f"[render] grid d {np.nanmin(d):.0f}..{np.nanmax(d):.0f}mm  "
              f"w {np.nanmin(w):.0f}..{np.nanmax(w):.0f}mm  "
              f"valid={finite.mean()*100:.0f}%", flush=True)

    def render(self, x_mm, y_mm, yaw, blur=0.0, gain=1.0):
        """返回 640x480 BGR 相机图。"""
        c, s = math.cos(yaw), math.sin(yaw)
        d, w = self.grid_d, self.grid_w
        x = x_mm + d * c + w * s
        y = y_mm + d * s - w * c
        mx = np.nan_to_num((x / self.mpp).astype(np.float32), nan=-1e6)
        my = np.nan_to_num(((self.H - 1) - y / self.mpp).astype(np.float32), nan=-1e6)
        out = cv2.remap(self.map_bgr, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=BG_BGR)
        # remap 在坐标越界时会填 (0,0,0)，不是 borderValue。纯黑会被巡线的
        # binary_threshold(160) 当成障碍，污染 ROI 的列白像素比例 -> 质心跳动 -> 丢线。
        # 统一改回地图背景色，与真实地垫一致。
        black = (out == 0).all(axis=2)
        if black.any():
            out[black] = BG_BGR
        if blur > 0:
            k = int(blur * 2) * 2 + 1
            out = cv2.GaussianBlur(out, (k, k), blur)
        if gain != 1.0:
            out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
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
