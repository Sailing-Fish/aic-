#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AIC 2026 决赛 11 类 YOLO 合成与域随机化数据生成器。

类别映射（与 src/yolov8/mod1/out_finals11.list 严格一致）：
    0..7 : Bonus 数字 1..8（标准印刷体 + 手写风格 + 彩色/纹理干扰背景）
    8    : red_cone         (红色路锥 13.5x13.5x23 cm)
    9    : red_no_entry     (红色空心禁行区 24x13 cm，边框 1 cm，内空)
    10   : yellow_stop_line (黄色停车线)

安全硬约束：
    - 严禁水平/垂直翻转（fliplr / flipud 会破坏数字 1-8 语义，例如 2/5/6/9 镜像失真）
    - 旋转角度硬限幅在 ±15° 以内（防止 6/9 或大角度失真）
    - 支持真实赛场背景目录 (--bg-dir) 与实拍道具抠图目录 (--props-dir)，
      未提供时自动使用内置仿真底图与合成卡片/路锥/禁行框冷启动。

输出结构：
    <out_dir>/images/{train,val}/
    <out_dir>/labels/{train,val}/
    <out_dir>/dataset.yaml
    <out_dir>/calib_list.txt    (供 RDK X5 Horizon OE INT8 PTQ 量化校准)
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CLASSES: List[str] = [
    "1", "2", "3", "4", "5", "6", "7", "8",
    "red_cone", "red_no_entry", "yellow_stop_line",
]
CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
MAX_ROTATION_DEG = 15.0

# 严禁开启翻转的硬校验标记（供单元测试与外部调用验证）
ALLOW_HFLIP = False
ALLOW_VFLIP = False


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------- 合成背景

def make_synthetic_track_bg(rng: random.Random, width: int = 640, height: int = 480) -> np.ndarray:
    """生成近似真实赛场视角的蓝底 + 白赛道 + 黑边合成背景图。"""
    # 随机轻微老化的蓝垫颜色
    blue_bgr = np.array([
        rng.randint(135, 190),
        rng.randint(70, 115),
        rng.randint(25, 65),
    ], dtype=np.uint8)
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:, :] = blue_bgr

    # 地平线以上画场馆杂乱背景（墙面/柱子/灯光）
    horizon = rng.randint(120, 175)
    wall = np.array([
        rng.randint(150, 220),
        rng.randint(155, 225),
        rng.randint(155, 225),
    ], dtype=np.uint8)
    img[:horizon, :] = wall
    for _ in range(rng.randint(3, 8)):
        x1 = rng.randint(0, width - 20)
        x2 = min(width, x1 + rng.randint(20, 120))
        y1 = rng.randint(0, max(1, horizon - 20))
        y2 = min(horizon, y1 + rng.randint(15, 60))
        col = (rng.randint(60, 230), rng.randint(60, 230), rng.randint(60, 230))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)

    # 梯形/微弯曲白赛道
    curve = rng.uniform(-90.0, 90.0)
    cx_bot = width * 0.5 + rng.uniform(-70.0, 70.0)
    cx_top = width * 0.5 + curve
    w_bot = rng.uniform(340.0, 480.0)
    w_top = rng.uniform(90.0, 175.0)

    ys = np.linspace(horizon, height - 1, 24)
    t = (ys - horizon) / max(1.0, (height - 1 - horizon))
    cxs = cx_top + (cx_bot - cx_top) * (t ** 1.15) + curve * 0.15 * np.sin(math.pi * t)
    half_ws = 0.5 * (w_top + (w_bot - w_top) * (t ** 1.05))

    left_pts = np.stack([cxs - half_ws, ys], axis=1)
    right_pts = np.stack([cxs + half_ws, ys], axis=1)[::-1]
    poly = np.vstack([left_pts, right_pts]).astype(np.int32)

    # 老化/偏暖/微灰白路面
    white_bgr = (
        rng.randint(210, 252),
        rng.randint(215, 254),
        rng.randint(215, 255),
    )
    cv2.fillPoly(img, [poly], white_bgr)

    # 黑色车道边界线（模拟 2-3cm 黑边）
    black_bgr = (rng.randint(15, 50), rng.randint(15, 50), rng.randint(15, 50))
    cv2.polylines(img, [left_pts.astype(np.int32)], False, black_bgr, rng.randint(4, 9), cv2.LINE_AA)
    cv2.polylines(img, [right_pts.astype(np.int32)], False, black_bgr, rng.randint(4, 9), cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- 道具合成 (BGRA)

def _draw_handwritten_digit(
    canvas: np.ndarray, digit: str, color: Tuple[int, int, int], rng: random.Random
) -> None:
    """用带抖动控制点的手绘折线生成手写体数字 1-8，避免依赖外部字体。"""
    h, w = canvas.shape[:2]
    # 归一化笔画定义（坐标系 [0.22..0.78] x [0.16..0.84]）
    strokes_map: Dict[str, List[List[Tuple[float, float]]]] = {
        "1": [[(0.38, 0.28), (0.52, 0.17), (0.52, 0.83)], [(0.34, 0.83), (0.70, 0.83)]],
        "2": [[(0.28, 0.30), (0.40, 0.18), (0.62, 0.18), (0.72, 0.30), (0.66, 0.48), (0.28, 0.82), (0.74, 0.82)]],
        "3": [[(0.28, 0.24), (0.55, 0.17), (0.72, 0.30), (0.52, 0.48), (0.74, 0.65), (0.58, 0.83), (0.28, 0.78)]],
        "4": [[(0.60, 0.18), (0.26, 0.58), (0.76, 0.58)], [(0.60, 0.18), (0.60, 0.84)]],
        "5": [[(0.70, 0.19), (0.32, 0.19), (0.29, 0.47), (0.58, 0.44), (0.73, 0.62), (0.58, 0.83), (0.28, 0.78)]],
        "6": [[(0.68, 0.22), (0.46, 0.18), (0.28, 0.45), (0.30, 0.75), (0.54, 0.84), (0.72, 0.68), (0.64, 0.50), (0.32, 0.52)]],
        "7": [[(0.26, 0.20), (0.74, 0.20), (0.44, 0.84)]],
        "8": [
            [(0.50, 0.48), (0.30, 0.34), (0.36, 0.18), (0.64, 0.18), (0.70, 0.34), (0.50, 0.48)],
            [(0.50, 0.48), (0.27, 0.64), (0.34, 0.83), (0.66, 0.83), (0.73, 0.64), (0.50, 0.48)],
        ],
    }
    thickness = max(3, int(min(h, w) * rng.uniform(0.08, 0.14)))
    for stroke in strokes_map[digit]:
        pts = []
        for nx, ny in stroke:
            jx = clamp(nx + rng.uniform(-0.04, 0.04), 0.12, 0.88)
            jy = clamp(ny + rng.uniform(-0.04, 0.04), 0.10, 0.90)
            pts.append((int(jx * w), int(jy * h)))
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr], False, color, thickness, cv2.LINE_AA)


def make_digit_card_bgra(
    digit: str,
    rng: random.Random,
    handwritten: bool = False,
    size: Tuple[int, int] = (96, 112),
) -> np.ndarray:
    """生成带 2026 决赛风格彩色/纹理/条纹干扰背景的数字卡片 BGRA。"""
    w, h = size
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bg_type = rng.choice(["plain_color", "stripes", "checker", "gradient", "noisy_white"])

    if bg_type == "noisy_white":
        base = np.array([rng.randint(205, 250), rng.randint(205, 250), rng.randint(205, 250)], dtype=np.uint8)
        bgr[:, :] = base
    elif bg_type == "plain_color":
        base = np.array([rng.randint(90, 245), rng.randint(90, 245), rng.randint(90, 245)], dtype=np.uint8)
        bgr[:, :] = base
    elif bg_type == "stripes":
        c1 = (rng.randint(120, 245), rng.randint(120, 245), rng.randint(120, 245))
        c2 = (rng.randint(70, 210), rng.randint(70, 210), rng.randint(70, 210))
        bgr[:, :] = c1
        step = rng.randint(8, 16)
        horiz = rng.random() < 0.5
        for p in range(0, (h if horiz else w), step * 2):
            if horiz:
                bgr[p:p + step, :] = c2
            else:
                bgr[:, p:p + step] = c2
    elif bg_type == "checker":
        c1 = (rng.randint(140, 245), rng.randint(140, 245), rng.randint(140, 245))
        c2 = (rng.randint(95, 200), rng.randint(95, 200), rng.randint(95, 200))
        tile = rng.randint(10, 18)
        for yy in range(0, h, tile):
            for xx in range(0, w, tile):
                col = c1 if ((yy // tile + xx // tile) % 2 == 0) else c2
                bgr[yy:yy + tile, xx:xx + tile] = col
    else:  # gradient
        c1 = np.array([rng.randint(100, 240) for _ in range(3)], dtype=np.float32)
        c2 = np.array([rng.randint(100, 240) for _ in range(3)], dtype=np.float32)
        alpha_g = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
        grad = (1.0 - alpha_g) * c1 + alpha_g * c2
        bgr[:, :] = np.repeat(grad.astype(np.uint8), h, axis=0)

    # 细碎视觉纹理噪声（斑点/细线干扰）
    for _ in range(rng.randint(2, 6)):
        p1 = (rng.randint(0, w - 1), rng.randint(0, h - 1))
        p2 = (rng.randint(0, w - 1), rng.randint(0, h - 1))
        col = (rng.randint(80, 220), rng.randint(80, 220), rng.randint(80, 220))
        cv2.line(bgr, p1, p2, col, 1, cv2.LINE_AA)

    # 数字笔迹颜色（与背景亮度拉开对比）
    bg_luma = float(bgr.mean())
    if bg_luma > 135:
        fg = (rng.randint(5, 55), rng.randint(5, 55), rng.randint(5, 65))
    else:
        fg = (rng.randint(215, 255), rng.randint(215, 255), rng.randint(215, 255))

    if handwritten:
        _draw_handwritten_digit(bgr, digit, fg, rng)
    else:
        font = rng.choice([
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
        ])
        scale = rng.uniform(2.2, 2.9) * (min(w, h) / 96.0)
        thick = max(3, int(round(rng.uniform(5.0, 9.0) * (min(w, h) / 96.0))))
        (tw, th), _ = cv2.getTextSize(digit, font, scale, thick)
        org = (max(4, (w - tw) // 2 + rng.randint(-4, 4)),
               min(h - 6, (h + th) // 2 + rng.randint(-4, 4)))
        cv2.putText(bgr, digit, org, font, scale, fg, thick, cv2.LINE_AA)

    # 卡片外框轻微描边
    if rng.random() < 0.5:
        cv2.rectangle(bgr, (1, 1), (w - 2, h - 2), (40, 40, 40), 2)

    alpha = np.full((h, w), 255, dtype=np.uint8)
    return np.dstack([bgr, alpha])


def make_red_cone_bgra(rng: random.Random, size: Tuple[int, int] = (72, 112)) -> np.ndarray:
    """生成 13.5x13.5x23cm 比例的红色锥桶 BGRA（带底座与可选反光条）。"""
    w, h = size
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)
    red_main = (rng.randint(15, 55), rng.randint(15, 55), rng.randint(175, 245))
    red_dark = (max(0, red_main[0] - 10), max(0, red_main[1] - 10), max(110, red_main[2] - 45))

    # 方形底座
    base_y = int(h * 0.80)
    cv2.rectangle(bgr, (int(w * 0.08), base_y), (int(w * 0.92), h - 2), red_dark, -1)
    cv2.rectangle(alpha, (int(w * 0.08), base_y), (int(w * 0.92), h - 2), 255, -1)

    # 锥体梯形
    cone_pts = np.array([
        [int(w * 0.42), int(h * 0.06)],
        [int(w * 0.58), int(h * 0.06)],
        [int(w * 0.82), base_y + 4],
        [int(w * 0.18), base_y + 4],
    ], dtype=np.int32)
    cv2.fillPoly(bgr, [cone_pts], red_main)
    cv2.fillPoly(alpha, [cone_pts], 255)

    # 可选白色反光条
    if rng.random() < 0.65:
        band_y1, band_y2 = int(h * 0.36), int(h * 0.52)
        t1 = (band_y1 - h * 0.06) / (base_y - h * 0.06)
        t2 = (band_y2 - h * 0.06) / (base_y - h * 0.06)
        bpts = np.array([
            [int(w * (0.42 - 0.24 * t1)), band_y1],
            [int(w * (0.58 + 0.24 * t1)), band_y1],
            [int(w * (0.58 + 0.24 * t2)), band_y2],
            [int(w * (0.42 - 0.24 * t2)), band_y2],
        ], dtype=np.int32)
        cv2.fillPoly(bgr, [bpts], (235, 235, 235))
    return np.dstack([bgr, alpha])


def make_red_no_entry_bgra(rng: random.Random, size: Tuple[int, int] = (144, 78)) -> np.ndarray:
    """生成 24x13cm 比例、边框约 1cm 的空心红色禁行区框 BGRA（内部透明露出路面）。"""
    w, h = size
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)
    red_bgr = (rng.randint(15, 55), rng.randint(15, 55), rng.randint(180, 245))
    # 1cm / 13cm ≈ 7.7% 高度
    border = max(4, int(round(h * rng.uniform(0.075, 0.11))))
    cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), red_bgr, border)
    cv2.rectangle(alpha, (0, 0), (w - 1, h - 1), 255, border)
    return np.dstack([bgr, alpha])


def make_yellow_stop_line_bgra(rng: random.Random, size: Tuple[int, int] = (240, 24)) -> np.ndarray:
    """生成横向黄色停车线胶带 BGRA。"""
    w, h = size
    out = np.zeros((h, w, 4), dtype=np.uint8)
    yellow_bgr = (rng.randint(0, 45), rng.randint(185, 245), rng.randint(205, 255))
    out[:, :, :3] = yellow_bgr
    out[:, :, 3] = 255
    return out


# ---------------------------------------------------------------- 几何与光度增强（禁止翻转！）

def warp_prop_bgra(
    prop: np.ndarray,
    rng: random.Random,
    max_rot_deg: float = MAX_ROTATION_DEG,
    perspective_strength: float = 0.08,
    allow_flip: bool = False,
) -> np.ndarray:
    """对道具 BGRA 做小角度旋转 (±<=15°) 与轻度透视变形。严禁任何翻转！"""
    if allow_flip:
        raise ValueError("严格禁止水平/垂直翻转（会破坏数字 1-8 类别语义）！")
    max_rot_deg = min(abs(float(max_rot_deg)), MAX_ROTATION_DEG)
    h, w = prop.shape[:2]
    pad = int(max(h, w) * 0.25) + 4
    padded = cv2.copyMakeBorder(prop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    ph, pw = padded.shape[:2]

    angle = rng.uniform(-max_rot_deg, max_rot_deg)
    M_rot = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), angle, 1.0)
    rot = cv2.warpAffine(
        padded, M_rot, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

    if perspective_strength > 0:
        s = perspective_strength * min(w, h)
        src = np.float32([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]])
        dst = src + np.float32([
            [rng.uniform(-s, s), rng.uniform(-s, s)],
            [rng.uniform(-s, s), rng.uniform(-s, s)],
            [rng.uniform(-s, s), rng.uniform(-s, s)],
            [rng.uniform(-s, s), rng.uniform(-s, s)],
        ])
        M_p = cv2.getPerspectiveTransform(src, dst)
        rot = cv2.warpPerspective(
            rot, M_p, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

    ys, xs = np.nonzero(rot[:, :, 3] > 20)
    if len(xs) == 0:
        return prop
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return rot[y0:y1, x0:x1]


def composite_bgra(
    bg: np.ndarray, prop: np.ndarray, cx: int, cy: int
) -> Optional[Tuple[int, int, int, int]]:
    """按 alpha 将道具贴到背景图上，返回裁剪后 bbox (x1, y1, x2, y2)。"""
    H, W = bg.shape[:2]
    ph, pw = prop.shape[:2]
    x1 = cx - pw // 2
    y1 = cy - ph // 2
    x2 = x1 + pw
    y2 = y1 + ph

    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(W, x2), min(H, y2)
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return None

    px1, py1 = ix1 - x1, iy1 - y1
    px2, py2 = px1 + (ix2 - ix1), py1 + (iy2 - iy1)
    patch = prop[py1:py2, px1:px2]
    alpha = (patch[:, :, 3:4].astype(np.float32)) / 255.0
    roi = bg[iy1:iy2, ix1:ix2].astype(np.float32)
    blended = alpha * patch[:, :, :3].astype(np.float32) + (1.0 - alpha) * roi
    bg[iy1:iy2, ix1:ix2] = np.clip(blended, 0, 255).astype(np.uint8)
    return ix1, iy1, ix2, iy2


def apply_photometric_augmentation(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """光照与镜头域随机化：色温偏置、亮度增益、定向眩光梯度、轻微运动模糊、噪声。"""
    out = img.astype(np.float32)
    h, w = out.shape[:2]

    # 1) 亮度与色温偏移（暖黄/冷蓝场馆灯光）
    gain = rng.uniform(0.72, 1.28)
    b_mul = rng.uniform(0.86, 1.16)
    g_mul = rng.uniform(0.92, 1.08)
    r_mul = rng.uniform(0.86, 1.16)
    out[:, :, 0] *= gain * b_mul
    out[:, :, 1] *= gain * g_mul
    out[:, :, 2] *= gain * r_mul

    # 2) 定向反光 / 眩光梯度
    if rng.random() < 0.65:
        gx = rng.uniform(0.15, 0.85) * w
        gy = rng.uniform(0.25, 0.90) * h
        sigma = rng.uniform(0.18, 0.45) * max(w, h)
        strength = rng.uniform(25.0, 85.0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        glare = strength * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * sigma * sigma))
        out += glare[:, :, None]

    out = np.clip(out, 0, 255).astype(np.uint8)

    # 3) 轻微运动模糊
    if rng.random() < 0.35:
        k = rng.choice([3, 5])
        kernel = np.zeros((k, k), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)

    # 4) 轻度高斯噪声
    if rng.random() < 0.45:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0.0, rng.uniform(2.0, 6.0), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


# ---------------------------------------------------------------- 数据集生成主流程

def load_image_paths(directory: Optional[str]) -> List[Path]:
    if not directory:
        return []
    p = Path(directory)
    if not p.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])


def load_prop_bank(props_dir: Optional[str]) -> Dict[str, List[Path]]:
    bank: Dict[str, List[Path]] = {c: [] for c in CLASSES}
    if not props_dir:
        return bank
    root = Path(props_dir)
    if not root.is_dir():
        return bank
    for cls_name in CLASSES:
        sub = root / cls_name
        if sub.is_dir():
            bank[cls_name] = load_image_paths(str(sub))
    return bank


def read_prop_bgra(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    return img


def bbox_to_yolo(cls_id: int, box: Tuple[int, int, int, int], W: int, H: int) -> Optional[str]:
    x1, y1, x2, y2 = box
    bw = max(0.0, min(float(W), float(x2)) - max(0.0, float(x1)))
    bh = max(0.0, min(float(H), float(y2)) - max(0.0, float(y1)))
    if bw < 3.0 or bh < 3.0:
        return None
    cx = clamp((x1 + x2) * 0.5 / W, 0.0, 1.0)
    cy = clamp((y1 + y2) * 0.5 / H, 0.0, 1.0)
    nw = clamp(bw / W, 1e-4, 1.0)
    nh = clamp(bh / H, 1e-4, 1.0)
    return f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def generate_one_sample(
    rng: random.Random,
    bg_paths: Sequence[Path],
    prop_bank: Dict[str, List[Path]],
    width: int = 640,
    height: int = 480,
    max_rot_deg: float = MAX_ROTATION_DEG,
) -> Tuple[np.ndarray, List[str]]:
    """生成单张 640x480 训练样本及其 11 类 YOLO 标注行。"""
    if bg_paths:
        raw_bg = cv2.imdecode(np.fromfile(str(rng.choice(bg_paths)), dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(raw_bg, (width, height)) if raw_bg is not None else make_synthetic_track_bg(rng, width, height)
    else:
        img = make_synthetic_track_bg(rng, width, height)

    labels: List[str] = []

    def get_prop(cls_name: str, handwritten: bool = False) -> np.ndarray:
        if prop_bank.get(cls_name) and rng.random() < 0.7:
            loaded = read_prop_bgra(rng.choice(prop_bank[cls_name]))
            if loaded is not None:
                return loaded
        if cls_name in ("1", "2", "3", "4", "5", "6", "7", "8"):
            return make_digit_card_bgra(cls_name, rng, handwritten=handwritten)
        if cls_name == "red_cone":
            return make_red_cone_bgra(rng)
        if cls_name == "red_no_entry":
            return make_red_no_entry_bgra(rng)
        return make_yellow_stop_line_bgra(rng)

    # 1) 45% 概率放置黄色停车线（随机直道停车线）
    if rng.random() < 0.45:
        cy_y = rng.randint(int(height * 0.48), int(height * 0.88))
        depth_scale = clamp((cy_y - height * 0.30) / (height * 0.65), 0.35, 1.25)
        yw = int(clamp(230 * depth_scale, 90, width - 40))
        yh = int(clamp(22 * depth_scale, 8, 36))
        y_prop = cv2.resize(get_prop("yellow_stop_line"), (yw, yh))
        y_prop = warp_prop_bgra(y_prop, rng, max_rot_deg=min(6.0, max_rot_deg), perspective_strength=0.04)
        cx_y = width // 2 + rng.randint(-60, 60)
        box = composite_bgra(img, y_prop, cx_y, cy_y)
        if box:
            ln = bbox_to_yolo(CLASS_TO_ID["yellow_stop_line"], box, width, height)
            if ln:
                labels.append(ln)

    # 2) 75% 概率放置双数字 Bonus 对（模拟 10-15cm 间距的直道手写双数字或岔路标准双数字）
    if rng.random() < 0.75:
        d1, d2 = rng.sample(CLASSES[:8], 2)
        handwritten = rng.random() < 0.5
        cy_d = rng.randint(int(height * 0.42), int(height * 0.82))
        depth_scale = clamp((cy_d - height * 0.25) / (height * 0.65), 0.32, 1.15)
        cw = int(clamp(76 * depth_scale, 28, 120))
        ch = int(clamp(90 * depth_scale, 34, 140))
        gap_px = int(clamp((105 if not handwritten else 82) * depth_scale, 32, 175))
        mid_x = width // 2 + rng.randint(-80, 80)

        for digit, sign in ((d1, -1), (d2, +1)):
            card = cv2.resize(get_prop(digit, handwritten=handwritten), (cw, ch))
            card = warp_prop_bgra(card, rng, max_rot_deg=max_rot_deg, perspective_strength=0.08)
            box = composite_bgra(img, card, mid_x + sign * (gap_px // 2), cy_d + rng.randint(-5, 5))
            if box:
                ln = bbox_to_yolo(CLASS_TO_ID[digit], box, width, height)
                if ln:
                    labels.append(ln)

    # 3) 60% 概率放置 1-2 个红色路锥
    if rng.random() < 0.60:
        for _ in range(rng.randint(1, 2)):
            cy_c = rng.randint(int(height * 0.42), int(height * 0.86))
            depth_scale = clamp((cy_c - height * 0.25) / (height * 0.65), 0.30, 1.15)
            cw = int(clamp(68 * depth_scale, 22, 110))
            ch = int(clamp(108 * depth_scale, 34, 170))
            cone = cv2.resize(get_prop("red_cone"), (cw, ch))
            cone = warp_prop_bgra(cone, rng, max_rot_deg=min(8.0, max_rot_deg), perspective_strength=0.05)
            cx_c = width // 2 + rng.randint(-150, 150)
            box = composite_bgra(img, cone, cx_c, cy_c)
            if box:
                ln = bbox_to_yolo(CLASS_TO_ID["red_cone"], box, width, height)
                if ln:
                    labels.append(ln)

    # 4) 55% 概率放置 1 个空心红色禁行区框 (24x13cm)
    if rng.random() < 0.55:
        cy_n = rng.randint(int(height * 0.46), int(height * 0.85))
        depth_scale = clamp((cy_n - height * 0.25) / (height * 0.65), 0.35, 1.15)
        nw = int(clamp(138 * depth_scale, 48, 220))
        nh = int(clamp(74 * depth_scale, 26, 120))
        frame = cv2.resize(get_prop("red_no_entry"), (nw, nh))
        frame = warp_prop_bgra(frame, rng, max_rot_deg=min(10.0, max_rot_deg), perspective_strength=0.09)
        cx_n = width // 2 + rng.randint(-120, 120)
        box = composite_bgra(img, frame, cx_n, cy_n)
        if box:
            ln = bbox_to_yolo(CLASS_TO_ID["red_no_entry"], box, width, height)
            if ln:
                labels.append(ln)

    img = apply_photometric_augmentation(img, rng)
    return img, labels


def generate_dataset(
    out_dir: str | Path,
    num_train: int = 200,
    num_val: int = 40,
    bg_dir: Optional[str] = None,
    props_dir: Optional[str] = None,
    seed: int = 2026,
    max_rot_deg: float = MAX_ROTATION_DEG,
    calib_count: int = 100,
    width: int = 640,
    height: int = 480,
) -> Dict[str, object]:
    """生成完整 Ultralytics 11 类 YOLO 数据集与 Horizon OE PTQ 校准列表。"""
    if abs(max_rot_deg) > MAX_ROTATION_DEG:
        raise ValueError(f"max_rot_deg={max_rot_deg} 超过允许上限 ±{MAX_ROTATION_DEG}°")

    root = Path(out_dir).resolve()
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    bg_paths = load_image_paths(bg_dir)
    prop_bank = load_prop_bank(props_dir)
    rng = random.Random(seed)

    train_images: List[Path] = []
    val_images: List[Path] = []
    total_boxes = 0

    for split, count, bucket in (("train", num_train, train_images), ("val", num_val, val_images)):
        for i in range(count):
            img, lbls = generate_one_sample(
                rng, bg_paths, prop_bank, width=width, height=height, max_rot_deg=max_rot_deg
            )
            stem = f"{split}_{i:05d}"
            img_path = root / "images" / split / f"{stem}.jpg"
            lbl_path = root / "labels" / split / f"{stem}.txt"
            ok_enc, enc_buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok_enc:
                enc_buf.tofile(str(img_path))
            lbl_path.write_text("\n".join(lbls) + ("\n" if lbls else ""), encoding="utf-8")
            bucket.append(img_path)
            total_boxes += len(lbls)

    # 写入 dataset.yaml
    names_yaml = "\n".join(f"  {i}: '{name}'" for i, name in enumerate(CLASSES))
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(
        f"# AIC 2026 Finals 11-class YOLO dataset\n"
        f"# CRITICAL training hyperparameters: fliplr=0.0, flipud=0.0, degrees={min(max_rot_deg, MAX_ROTATION_DEG):.1f}\n"
        f"path: {root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )

    # 写入 RDK X5 Horizon OE INT8 PTQ 校准列表
    pool = list(train_images) + list(val_images)
    rng.shuffle(pool)
    calib_sel = pool[: min(calib_count, len(pool))]
    calib_list = root / "calib_list.txt"
    calib_list.write_text(
        "\n".join(p.as_posix() for p in calib_sel) + ("\n" if calib_sel else ""),
        encoding="utf-8",
    )

    return {
        "out_dir": str(root),
        "num_train": len(train_images),
        "num_val": len(val_images),
        "total_boxes": total_boxes,
        "dataset_yaml": str(dataset_yaml),
        "calib_list": str(calib_list),
        "calib_count": len(calib_sel),
        "allow_hflip": ALLOW_HFLIP,
        "allow_vflip": ALLOW_VFLIP,
        "max_rot_deg": min(abs(max_rot_deg), MAX_ROTATION_DEG),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="AIC 2026 决赛 11 类 YOLO 合成增强数据集生成器（严禁翻转）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out", required=True, help="输出数据集目录")
    ap.add_argument("--num-train", type=int, default=400, help="训练集张数")
    ap.add_argument("--num-val", type=int, default=80, help="验证集张数")
    ap.add_argument("--bg-dir", default=None, help="真实赛场背景图目录（可选）")
    ap.add_argument("--props-dir", default=None, help="实拍道具抠图目录（按类名建子目录，可选）")
    ap.add_argument("--seed", type=int, default=2026, help="随机种子")
    ap.add_argument("--max-rot-deg", type=float, default=15.0, help="最大旋转角（上限 15°）")
    ap.add_argument("--calib-count", type=int, default=100, help="INT8 PTQ 校准集图片数量")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    info = generate_dataset(
        out_dir=args.out,
        num_train=args.num_train,
        num_val=args.num_val,
        bg_dir=args.bg_dir,
        props_dir=args.props_dir,
        seed=args.seed,
        max_rot_deg=args.max_rot_deg,
        calib_count=args.calib_count,
        width=args.width,
        height=args.height,
    )
    print(
        f"[DONE] train={info['num_train']} val={info['num_val']} "
        f"boxes={info['total_boxes']} calib={info['calib_count']} "
        f"(hflip={info['allow_hflip']}, vflip={info['allow_vflip']}, max_rot=±{info['max_rot_deg']}°)"
    )
    print(f"  dataset.yaml : {info['dataset_yaml']}")
    print(f"  calib_list   : {info['calib_list']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
