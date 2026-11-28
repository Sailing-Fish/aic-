#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AIC 2026 决赛 8 类数字直替 / 11 类全目标 YOLO 合成与域随机化数据生成器。

模式说明：
    --classes 11（默认，保持向后兼容）：
        0..7 : Bonus 数字 1..8
        8    : red_cone
        9    : red_no_entry
        10   : yellow_stop_line
    --classes 8（8 类决赛数字直替，与 src/yolov8/mod1/out.list 严格对齐）：
        0..7 : Bonus 数字 1..8（仅标注 1..8；红锥/红禁行区/黄停车线仍作为无标签背景干扰物绘制）

安全硬约束：
    - 严禁水平/垂直翻转（fliplr / flipud 会破坏数字 1-8 语义，例如 2/5/6/9 镜像失真）
    - 旋转角度硬限幅在 ±15° 以内（防止 6/9 或大角度失真）
    - 支持真实赛场背景目录 (--bg-dir)、实拍道具抠图目录 (--props-dir) 与 MNIST 手写缓存 (--mnist-dir)

输出结构：
    <out_dir>/images/{train,val}/
    <out_dir>/labels/{train,val}/
    <out_dir>/dataset.yaml
    <out_dir>/calib_list.txt    (供 RDK X5 Horizon OE INT8 PTQ 量化校准)
"""
from __future__ import annotations

import argparse
import gzip
import math
import os
import random
import struct
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw as PILImageDraw
    from PIL import ImageFont as PILImageFont
except ImportError:
    PILImage = None
    PILImageDraw = None
    PILImageFont = None

DIGITS8_CLASSES: List[str] = ["1", "2", "3", "4", "5", "6", "7", "8"]
DIGITS8_CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(DIGITS8_CLASSES)}

CLASSES: List[str] = DIGITS8_CLASSES + ["red_cone", "red_no_entry", "yellow_stop_line"]
CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
MAX_ROTATION_DEG = 15.0

# 严禁开启翻转的硬校验标记（供单元测试与外部调用验证）
ALLOW_HFLIP = False
ALLOW_VFLIP = False

# Tier 1-④-b 默认数字贴纸最远投影距离（m）：
# 8 类数字直替模式封顶 1.15m（消除 1.45~1.55m 处 4~6px 不可读超远贴纸拖累 Recall），
# 11 类全目标模式维持 1.55m 保持旧行为与单测兼容。
DEFAULT_MAX_DIGIT_DIST_8CLS_M = 1.15
DEFAULT_MAX_DIGIT_DIST_11CLS_M = 1.55


def resolve_max_digit_dist_m(num_classes: int, max_digit_dist_m: Optional[float] = None) -> float:
    """解析并校验数字贴纸最远投影距离（m）。"""
    if max_digit_dist_m is None:
        return DEFAULT_MAX_DIGIT_DIST_8CLS_M if num_classes == 8 else DEFAULT_MAX_DIGIT_DIST_11CLS_M
    val = float(max_digit_dist_m)
    if not (0.25 <= val <= 2.50):
        raise ValueError(f"max_digit_dist_m={val} 超出合理范围 [0.25, 2.50] m")
    return val

MNIST_MIRRORS: Tuple[str, ...] = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
MNIST_TRAIN_IMAGES_GZ = "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABELS_GZ = "train-labels-idx1-ubyte.gz"

# 与实车 finals_fork.yaml / finals_roundabout.yaml 对齐的地面单应矩阵 (u, v, 1) -> (x_fwd, y_left, w)
CALIBRATED_H_P2G = np.array(
    [
        [-1.3985236e-05, 0.00074556476, -0.41829529],
        [0.0007049316, -2.6281263e-05, -0.21497408],
        [-4.7978586e-06, -0.0063203042, 1.0],
    ],
    dtype=np.float64,
)
CALIBRATED_H_G2P = np.linalg.inv(CALIBRATED_H_P2G)

BOLD_FONT_CANDIDATES: Tuple[str, ...] = (
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\verdanab.ttf",
    r"C:\Windows\Fonts\bahnschrift.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\trebucbd.ttf",
    r"C:\Windows\Fonts\impact.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\ARIALNB.TTF",
    r"C:\Windows\Fonts\consolab.ttf",
    r"C:\Windows\Fonts\georgiab.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyhbd.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)
HANDWRITTEN_FONT_CANDIDATES: Tuple[str, ...] = (
    r"C:\Windows\Fonts\Inkfree.ttf",
    r"C:\Windows\Fonts\segoepr.ttf",
    r"C:\Windows\Fonts\segoeprb.ttf",
    r"C:\Windows\Fonts\segoesc.ttf",
    r"C:\Windows\Fonts\segoescb.ttf",
    r"C:\Windows\Fonts\BRADHITC.TTF",
    r"C:\Windows\Fonts\LHANDW.TTF",
    r"C:\Windows\Fonts\mvboli.ttf",
    r"C:\Windows\Fonts\PRISTINA.TTF",
    r"C:\Windows\Fonts\comic.ttf",
    r"C:\Windows\Fonts\comicbd.ttf",
    r"C:\Windows\Fonts\FRSCRIPT.TTF",
    r"C:\Windows\Fonts\MISTRAL.TTF",
)
_FONT_PATH_CACHE: Optional[List[str]] = None
_HW_FONT_PATH_CACHE: Optional[List[str]] = None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _discover_bold_fonts() -> List[str]:
    global _FONT_PATH_CACHE
    if _FONT_PATH_CACHE is None:
        _FONT_PATH_CACHE = [p for p in BOLD_FONT_CANDIDATES if os.path.isfile(p)]
    return _FONT_PATH_CACHE


def _discover_handwritten_fonts() -> List[str]:
    global _HW_FONT_PATH_CACHE
    if _HW_FONT_PATH_CACHE is None:
        _HW_FONT_PATH_CACHE = [p for p in HANDWRITTEN_FONT_CANDIDATES if os.path.isfile(p)]
    return _HW_FONT_PATH_CACHE


# ---------------------------------------------------------------- MNIST 手写数字缓存与加载

def _parse_idx_images(gz_bytes: bytes) -> np.ndarray:
    raw = gzip.decompress(gz_bytes)
    magic, num, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"Invalid IDX3 magic: {magic}")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(num, rows, cols)


def _parse_idx_labels(gz_bytes: bytes) -> np.ndarray:
    raw = gzip.decompress(gz_bytes)
    magic, num = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"Invalid IDX1 magic: {magic}")
    return np.frombuffer(raw, dtype=np.uint8, offset=8)


def _download_bytes(filename: str, timeout_s: float = 25.0) -> Optional[bytes]:
    for base_url in MNIST_MIRRORS:
        url = base_url + filename
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if getattr(resp, "status", 200) == 200:
                    return resp.read()
        except Exception:
            continue
    return None


def load_mnist_digits(
    cache_dir: Optional[str | Path],
    digits: Sequence[str] = DIGITS8_CLASSES,
    allow_download: bool = True,
) -> Dict[str, np.ndarray]:
    """加载或下载 MNIST 真实手写数字（1..8）并缓存到 <cache_dir>/mnist_1_8.npz。

    离线或下载失败时优雅回退为空字典映射，调用方自动使用内置笔画生成器。
    """
    bank: Dict[str, np.ndarray] = {d: np.empty((0, 28, 28), dtype=np.uint8) for d in digits}
    if not cache_dir:
        return bank

    cache_path = Path(cache_dir)
    npz_path = cache_path / "mnist_1_8.npz"
    if npz_path.is_file():
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                for d in digits:
                    key = f"d{d}"
                    if key in data:
                        bank[d] = data[key]
            if any(len(v) > 0 for v in bank.values()):
                return bank
        except Exception:
            pass

    if not allow_download:
        return bank

    imgs_all: Optional[np.ndarray] = None
    lbls_all: Optional[np.ndarray] = None

    img_gz = _download_bytes(MNIST_TRAIN_IMAGES_GZ)
    lbl_gz = _download_bytes(MNIST_TRAIN_LABELS_GZ)
    if img_gz is not None and lbl_gz is not None:
        try:
            imgs_all = _parse_idx_images(img_gz)
            lbls_all = _parse_idx_labels(lbl_gz)
        except Exception:
            imgs_all, lbls_all = None, None

    # 若镜像下载失败且环境装有 torchvision，尝试 torchvision.datasets.MNIST
    if imgs_all is None or lbls_all is None:
        try:
            from torchvision import datasets  # type: ignore

            cache_path.mkdir(parents=True, exist_ok=True)
            ds = datasets.MNIST(root=str(cache_path), train=True, download=True)
            imgs_all = ds.data.numpy().astype(np.uint8)
            lbls_all = ds.targets.numpy().astype(np.uint8)
        except Exception:
            return bank

    if imgs_all is None or lbls_all is None:
        return bank

    cache_path.mkdir(parents=True, exist_ok=True)
    save_dict: Dict[str, np.ndarray] = {}
    for d in digits:
        d_int = int(d)
        subset = imgs_all[lbls_all == d_int].copy()
        bank[d] = subset
        save_dict[f"d{d}"] = subset
    try:
        np.savez_compressed(npz_path, **save_dict)
    except Exception:
        pass
    return bank


# ---------------------------------------------------------------- 合成背景

def make_synthetic_track_bg(rng: random.Random, width: int = 640, height: int = 480) -> np.ndarray:
    """生成近似真实赛场视角的蓝底 + 白赛道 + 黑边合成背景图（含可选折痕与岔道）。"""
    # 宝蓝/深蓝赛场垫（含微老化色温变化）
    blue_bgr = np.array([
        rng.randint(135, 192),
        rng.randint(65, 115),
        rng.randint(22, 62),
    ], dtype=np.uint8)
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:, :] = blue_bgr

    # 地平线以上画场馆杂乱背景（墙面/踢脚线/柱子/灯光）
    horizon = rng.randint(120, 178)
    wall = np.array([
        rng.randint(100, 210),
        rng.randint(105, 215),
        rng.randint(105, 215),
    ], dtype=np.uint8)
    img[:horizon, :] = wall
    # 场馆墙脚黑条带（模拟实拍图背景地砖/墙脚线）
    if rng.random() < 0.65:
        kick_h = rng.randint(8, 22)
        cv2.rectangle(img, (0, max(0, horizon - kick_h)), (width, horizon), (35, 38, 42), -1)
    for _ in range(rng.randint(2, 7)):
        x1 = rng.randint(0, width - 20)
        x2 = min(width, x1 + rng.randint(20, 120))
        y1 = rng.randint(0, max(1, horizon - 20))
        y2 = min(horizon, y1 + rng.randint(15, 60))
        col = (rng.randint(50, 225), rng.randint(50, 225), rng.randint(50, 225))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)

    # 梯形/微弯曲白赛道
    curve = rng.uniform(-95.0, 95.0)
    cx_bot = width * 0.5 + rng.uniform(-70.0, 70.0)
    cx_top = width * 0.5 + curve
    w_bot = rng.uniform(340.0, 480.0)
    w_top = rng.uniform(95.0, 180.0)

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

    # 可选岔道中间分叉蓝岛（约 25% 概率模拟赛场 Y 形岔道）
    if rng.random() < 0.25:
        split_t = rng.uniform(0.25, 0.55)
        split_y = int(horizon + split_t * (height - 1 - horizon))
        island = np.array([
            [int(cx_top - w_top * 0.14), horizon],
            [int(cx_top + w_top * 0.14), horizon],
            [int(cx_top + rng.uniform(-20, 20)), split_y],
        ], dtype=np.int32)
        cv2.fillPoly(img, [island], blue_bgr.tolist())
        cv2.polylines(img, [island], True, (25, 25, 25), rng.randint(4, 7), cv2.LINE_AA)

    # 黑色车道边界线（模拟 2-3cm 黑边）
    black_bgr = (rng.randint(15, 50), rng.randint(15, 50), rng.randint(15, 50))
    cv2.polylines(img, [left_pts.astype(np.int32)], False, black_bgr, rng.randint(4, 9), cv2.LINE_AA)
    cv2.polylines(img, [right_pts.astype(np.int32)], False, black_bgr, rng.randint(4, 9), cv2.LINE_AA)

    # 赛场布垫微妙折痕阴影（模拟实拍褶皱感，保持路面整体高亮）
    if rng.random() < 0.50:
        for _ in range(rng.randint(1, 3)):
            cy_f = rng.randint(horizon + 20, height - 20)
            tilt = rng.randint(-18, 18)
            overlay = img.copy()
            cv2.line(overlay, (0, cy_f), (width, cy_f + tilt), (175, 178, 182), rng.randint(3, 8), cv2.LINE_AA)
            cv2.addWeighted(overlay, rng.uniform(0.18, 0.35), img, 0.72, 0, img)

    return img


# ---------------------------------------------------------------- 道具合成 (BGRA)

def _apply_elastic_warp_u8(mask_u8: np.ndarray, rng: random.Random, alpha: float = 5.5, sigma: float = 4.0) -> np.ndarray:
    """对字形灰度遮罩施加 Simard 风格随机弹性形变，模拟不同书写者手抖与笔顺变化。"""
    h, w = mask_u8.shape[:2]
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    ksize = max(3, int(round(sigma * 4.0)) | 1)
    dx = cv2.GaussianBlur((np_rng.uniform(-1.0, 1.0, (h, w))).astype(np.float32), (ksize, ksize), sigma) * float(alpha)
    dy = cv2.GaussianBlur((np_rng.uniform(-1.0, 1.0, (h, w))).astype(np.float32), (ksize, ksize), sigma) * float(alpha)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = np.clip(xx + dx, 0, w - 1)
    map_y = np.clip(yy + dy, 0, h - 1)
    return cv2.remap(mask_u8, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _apply_stroke_width_and_dry_ink(mask_u8: np.ndarray, rng: random.Random, allow_very_thin: bool = True) -> np.ndarray:
    """模拟不同笔型与出墨质感：极细圆珠笔/细白笔、中号油漆笔、扁头宽马克笔、干笔粉笔微断墨。"""
    modes = ["keep", "thin1", "thick_iso", "thick_horiz", "thick_vert", "dry_chalk"]
    weights = [0.28, 0.24 if allow_very_thin else 0.12, 0.16, 0.10, 0.10, 0.12]
    mode = rng.choices(modes, weights=weights, k=1)[0]

    out = mask_u8.copy()
    if mode == "thin1":
        # 对过细笔画若腐蚀过狠先检查像素保留比例，防止笔迹断裂消失
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        er = cv2.erode(out, k, iterations=1)
        if int((er > 64).sum()) >= max(30, int((out > 64).sum() * 0.35)):
            out = er
    elif mode == "thick_iso":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rng.choice([2, 3]), rng.choice([2, 3])))
        out = cv2.dilate(out, k, iterations=1)
    elif mode == "thick_horiz":
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (rng.choice([3, 4]), 2))
        out = cv2.dilate(out, k, iterations=1)
    elif mode == "thick_vert":
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, rng.choice([3, 4])))
        out = cv2.dilate(out, k, iterations=1)
    elif mode == "dry_chalk":
        h, w = out.shape[:2]
        np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        tex = np_rng.uniform(0.55, 1.15, (max(4, h // 2), max(4, w // 2))).astype(np.float32)
        tex = cv2.resize(tex, (w, h), interpolation=cv2.INTER_LINEAR)
        out = np.clip(out.astype(np.float32) * tex, 0, 255).astype(np.uint8)
    return out


def _draw_handwritten_digit(
    canvas: np.ndarray, digit: str, color: Tuple[int, int, int], rng: random.Random
) -> None:
    """多笔形、多写法、多线宽手绘折线数字 1-8，显式覆盖换人换笔写法差异。"""
    h, w = canvas.shape[:2]
    variants_map: Dict[str, List[List[List[Tuple[float, float]]]]] = {
        "1": [
            [[(0.36, 0.28), (0.52, 0.17), (0.52, 0.83)], [(0.33, 0.83), (0.70, 0.83)]],
            [[(0.50, 0.16), (0.50, 0.84)]],
            [[(0.37, 0.31), (0.54, 0.17), (0.50, 0.84)]],
        ],
        "2": [
            [[(0.28, 0.30), (0.40, 0.18), (0.62, 0.18), (0.72, 0.30), (0.66, 0.48), (0.28, 0.82), (0.74, 0.82)]],
            [[(0.26, 0.28), (0.45, 0.17), (0.68, 0.25), (0.62, 0.50), (0.30, 0.81), (0.42, 0.72), (0.74, 0.81)]],
        ],
        "3": [
            [[(0.28, 0.24), (0.55, 0.17), (0.72, 0.30), (0.50, 0.48), (0.74, 0.65), (0.58, 0.83), (0.28, 0.78)]],
            [[(0.30, 0.19), (0.70, 0.19), (0.46, 0.46), (0.70, 0.56), (0.66, 0.78), (0.30, 0.82)]],
        ],
        "4": [
            [[(0.60, 0.18), (0.26, 0.58), (0.76, 0.58)], [(0.60, 0.18), (0.60, 0.84)]],  # 闭口印刷风手写4
            [[(0.32, 0.19), (0.26, 0.55), (0.74, 0.55)], [(0.64, 0.22), (0.64, 0.84)]],  # 开口手写4
            [[(0.62, 0.17), (0.28, 0.60), (0.77, 0.58)], [(0.62, 0.40), (0.62, 0.84)]],
        ],
        "5": [
            [[(0.70, 0.19), (0.32, 0.19), (0.29, 0.47), (0.58, 0.44), (0.73, 0.62), (0.58, 0.83), (0.28, 0.78)]],
            [[(0.34, 0.20), (0.31, 0.48), (0.60, 0.47), (0.72, 0.65), (0.52, 0.83), (0.28, 0.76)], [(0.34, 0.20), (0.72, 0.20)]],
        ],
        "6": [
            [[(0.66, 0.20), (0.44, 0.18), (0.28, 0.46), (0.30, 0.75), (0.54, 0.84), (0.72, 0.68), (0.64, 0.50), (0.32, 0.52)]],
            [[(0.58, 0.17), (0.32, 0.42), (0.31, 0.72), (0.52, 0.83), (0.69, 0.68), (0.56, 0.52), (0.34, 0.58)]],
        ],
        "7": [
            [[(0.26, 0.20), (0.74, 0.20), (0.44, 0.84)]],
            [[(0.26, 0.20), (0.74, 0.20), (0.44, 0.84)], [(0.38, 0.52), (0.67, 0.52)]],  # 带横杠7
            [[(0.26, 0.30), (0.26, 0.19), (0.74, 0.19), (0.48, 0.84)]],
        ],
        "8": [
            [
                [(0.50, 0.48), (0.30, 0.34), (0.36, 0.18), (0.64, 0.18), (0.70, 0.34), (0.50, 0.48)],
                [(0.50, 0.48), (0.27, 0.64), (0.34, 0.83), (0.66, 0.83), (0.73, 0.64), (0.50, 0.48)],
            ],
            [[(0.55, 0.20), (0.32, 0.28), (0.36, 0.44), (0.67, 0.56), (0.69, 0.75), (0.48, 0.84), (0.30, 0.72), (0.35, 0.55), (0.66, 0.40), (0.66, 0.24), (0.55, 0.20)]],
        ],
    }
    mask_u8 = np.zeros((h, w), dtype=np.uint8)
    chosen_strokes = rng.choice(variants_map[digit])
    # 线宽覆盖极细白板笔 (0.038) 到粗马克笔 (0.145)
    thickness = max(2, int(round(min(h, w) * rng.uniform(0.038, 0.145))))
    for stroke in chosen_strokes:
        pts = []
        for nx, ny in stroke:
            jx = clamp(nx + rng.uniform(-0.045, 0.045), 0.12, 0.88)
            jy = clamp(ny + rng.uniform(-0.045, 0.045), 0.10, 0.90)
            pts.append((int(jx * w), int(jy * h)))
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask_u8, [arr], False, 255, thickness, cv2.LINE_AA)

    if rng.random() < 0.65:
        mask_u8 = _apply_elastic_warp_u8(mask_u8, rng, alpha=rng.uniform(2.5, 6.0), sigma=rng.uniform(3.5, 5.5))
    mask_u8 = _apply_stroke_width_and_dry_ink(mask_u8, rng, allow_very_thin=False)
    mask_f = (mask_u8.astype(np.float32) / 255.0)[:, :, None]
    fg_arr = np.array(color, dtype=np.float32)[None, None, :]
    out = canvas.astype(np.float32) * (1.0 - mask_f) + fg_arr * mask_f
    canvas[:, :] = np.clip(out, 0, 255).astype(np.uint8)


def _render_mnist_glyph_mask(
    glyph28: np.ndarray, out_w: int, out_h: int, rng: random.Random
) -> np.ndarray:
    """将 28x28 MNIST 手写体增强（随机弹性形变 + 多笔宽腐蚀/加粗 + 宽高比 + 微倾），输出 [0,1] float 遮罩。"""
    up = cv2.resize(glyph28, (64, 64), interpolation=cv2.INTER_CUBIC)
    ys, xs = np.nonzero(up > 35)
    if len(xs) > 8:
        x0, x1 = max(0, int(xs.min()) - 2), min(64, int(xs.max()) + 3)
        y0, y1 = max(0, int(ys.min()) - 2), min(64, int(ys.max()) + 3)
        crop = up[y0:y1, x0:x1]
    else:
        crop = up

    # 弹性形变 + 换笔粗细与干墨纹理
    if rng.random() < 0.70:
        crop = _apply_elastic_warp_u8(crop, rng, alpha=rng.uniform(2.5, 7.5), sigma=rng.uniform(3.2, 5.5))
    crop = _apply_stroke_width_and_dry_ink(crop, rng, allow_very_thin=True)

    ch, cw = crop.shape[:2]
    target_frac = rng.uniform(0.50, 0.74)
    aspect_jitter = rng.uniform(0.80, 1.24)
    base_scale = min((out_w * target_frac) / max(1, cw), (out_h * target_frac) / max(1, ch))
    nw = max(6, int(round(cw * base_scale * math.sqrt(aspect_jitter))))
    nh = max(6, int(round(ch * base_scale / math.sqrt(aspect_jitter))))
    nw = min(int(out_w * 0.84), nw)
    nh = min(int(out_h * 0.84), nh)
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if base_scale < 1.0 else cv2.INTER_CUBIC)

    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    ox = (out_w - nw) // 2 + rng.randint(-max(1, out_w // 18), max(1, out_w // 18))
    oy = (out_h - nh) // 2 + rng.randint(-max(1, out_h // 18), max(1, out_h // 18))
    x1, y1 = max(0, ox), max(0, oy)
    x2, y2 = min(out_w, ox + nw), min(out_h, oy + nh)
    if x2 > x1 and y2 > y1:
        sx1, sy1 = x1 - ox, y1 - oy
        mask[y1:y2, x1:x2] = resized[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]

    # 微倾剪切与微旋转（硬限幅 <= 10°，绝不翻转）
    ang = rng.uniform(-10.0, 10.0)
    shear = rng.uniform(-0.16, 0.16)
    M_rot = cv2.getRotationMatrix2D((out_w * 0.5, out_h * 0.5), ang, 1.0)
    M_rot[0, 1] += shear
    M_rot[0, 2] -= shear * (out_h * 0.5)
    mask = cv2.warpAffine(mask, M_rot, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=0)

    mask_f = np.clip((mask.astype(np.float32) - 20.0) / 172.0, 0.0, 1.0)
    return mask_f


def _draw_printed_digit_mask(
    digit: str, out_w: int, out_h: int, rng: random.Random, font_pool: str = "printed"
) -> np.ndarray:
    """绘制多样印刷字体或手写风格系统字体数字遮罩 [0,1]。"""
    fonts = _discover_handwritten_fonts() if font_pool == "handwritten" else _discover_bold_fonts()
    if not fonts:
        fonts = _discover_bold_fonts()
    if PILImage is not None and PILImageDraw is not None and PILImageFont is not None and fonts:
        font_path = rng.choice(fonts)
        frac = rng.uniform(0.54, 0.78) if font_pool == "handwritten" else rng.uniform(0.58, 0.78)
        font_px = max(16, int(min(out_w, out_h) * frac))
        try:
            font = PILImageFont.truetype(font_path, font_px)
            pil_img = PILImage.new("L", (out_w, out_h), 0)
            draw = PILImageDraw.Draw(pil_img)
            bbox = draw.textbbox((0, 0), digit, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = (out_w - tw) * 0.5 - bbox[0] + rng.uniform(-out_w * 0.05, out_w * 0.05)
            ty = (out_h - th) * 0.5 - bbox[1] + rng.uniform(-out_h * 0.05, out_h * 0.05)
            draw.text((tx, ty), digit, fill=255, font=font)
            arr = np.asarray(pil_img, dtype=np.uint8)
            if font_pool == "handwritten":
                if rng.random() < 0.70:
                    arr = _apply_elastic_warp_u8(arr, rng, alpha=rng.uniform(2.5, 6.5), sigma=rng.uniform(3.5, 5.5))
                arr = _apply_stroke_width_and_dry_ink(arr, rng, allow_very_thin=True)
            elif rng.random() < 0.35:
                arr = _apply_stroke_width_and_dry_ink(arr, rng, allow_very_thin=False)
            return arr.astype(np.float32) / 255.0
        except Exception:
            pass

    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    font = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX])
    scale = rng.uniform(1.8, 2.7) * (min(out_w, out_h) / 96.0)
    thick = max(2, int(round(rng.uniform(3.5, 9.5) * (min(out_w, out_h) / 96.0))))
    (tw, th), _ = cv2.getTextSize(digit, font, scale, thick)
    org = (max(2, (out_w - tw) // 2 + rng.randint(-4, 4)),
           min(out_h - 4, (out_h + th) // 2 + rng.randint(-4, 4)))
    cv2.putText(canvas, digit, org, font, scale, 255, thick, cv2.LINE_AA)
    return canvas.astype(np.float32) / 255.0


def _draw_digit_on_bgr(
    bgr: np.ndarray,
    digit: str,
    fg_bgr: Tuple[int, int, int],
    rng: random.Random,
    handwritten: bool,
    mnist_bank: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """将多样印刷体 / 系统手写字体 / 弹性 MNIST / 多变体手绘笔画合成到 bgr 画布上。"""
    h, w = bgr.shape[:2]
    mask_f: Optional[np.ndarray] = None

    if handwritten:
        p_hw = rng.random()
        samples = mnist_bank.get(digit) if mnist_bank else None
        if p_hw < 0.45 and samples is not None and len(samples) > 0:
            glyph = samples[rng.randrange(len(samples))]
            mask_f = _render_mnist_glyph_mask(glyph, w, h, rng)
        elif p_hw < 0.78 and _discover_handwritten_fonts():
            mask_f = _draw_printed_digit_mask(digit, w, h, rng, font_pool="handwritten")
        else:
            _draw_handwritten_digit(bgr, digit, fg_bgr, rng)
            return
    else:
        mask_f = _draw_printed_digit_mask(digit, w, h, rng, font_pool="printed")

    fg_arr = np.array(fg_bgr, dtype=np.float32)[None, None, :]
    out = bgr.astype(np.float32) * (1.0 - mask_f[:, :, None]) + fg_arr * mask_f[:, :, None]
    bgr[:, :] = np.clip(out, 0, 255).astype(np.uint8)


def _apply_disc_background_interference(
    bgr: np.ndarray,
    alpha_mask: np.ndarray,
    rng: random.Random,
    strength: str = "random",
) -> np.ndarray:
    """针对 2026 决赛“蓝圆片加底噪、局部纹理、彩色斑块干扰”专项合成模块。"""
    h, w = bgr.shape[:2]
    out = bgr.astype(np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    inside = (alpha_mask > 64)

    strong = (strength == "heavy") or (strength == "random" and rng.random() < 0.45)

    # 1) 局部纹理叠加（细条纹、网格、棋盘格、半色调圆点、纸张/布料纹理）
    if strength == "heavy" or rng.random() < 0.72:
        tex_mode = rng.choice(["stripes", "grid", "checker", "dots", "rings", "grain"])
        overlay = out.copy()
        tex_col = np.array(
            [
                rng.randint(25, 225),
                rng.randint(25, 225),
                rng.randint(25, 225),
            ],
            dtype=np.float32,
        )
        if tex_mode == "stripes":
            step = rng.randint(8, 18)
            thick = max(2, step // 2)
            ang_horiz = rng.random() < 0.5
            for p in range(0, (h if ang_horiz else w), step):
                if ang_horiz:
                    overlay[p:p + thick, :] = tex_col
                else:
                    overlay[:, p:p + thick] = tex_col
        elif tex_mode == "grid":
            step = rng.randint(10, 22)
            lw = rng.randint(1, 2)
            for p in range(0, h, step):
                overlay[p:p + lw, :] = tex_col
            for p in range(0, w, step):
                overlay[:, p:p + lw] = tex_col
        elif tex_mode == "checker":
            tile = rng.randint(10, 20)
            for yy in range(0, h, tile):
                for xx in range(0, w, tile):
                    if (yy // tile + xx // tile) % 2 == 0:
                        overlay[yy:yy + tile, xx:xx + tile] = tex_col
        elif tex_mode == "dots":
            step = rng.randint(12, 22)
            rad = rng.randint(2, max(2, step // 4))
            for yy in range(step // 2, h, step):
                for xx in range(step // 2, w, step):
                    cv2.circle(overlay, (xx, yy), rad, tex_col.tolist(), -1, cv2.LINE_AA)
        elif tex_mode == "rings":
            cx, cy = w // 2 + rng.randint(-15, 15), h // 2 + rng.randint(-15, 15)
            for r_rad in range(12, max(h, w), rng.randint(12, 20)):
                cv2.circle(overlay, (cx, cy), r_rad, tex_col.tolist(), rng.randint(1, 3), cv2.LINE_AA)
        else:  # grain
            coarse = np_rng.uniform(0.0, 255.0, (max(8, h // 4), max(8, w // 4), 3)).astype(np.float32)
            overlay = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
        t_alpha = rng.uniform(0.22, 0.48) if strong else rng.uniform(0.12, 0.30)
        out[inside] = out[inside] * (1.0 - t_alpha) + overlay[inside] * t_alpha

    # 2) 低频云斑 / 彩色污渍色块（不规则黄/绿/青/紫/灰斑）
    if strength == "heavy" or rng.random() < 0.65:
        n_blobs = rng.randint(2, 5) if strong else rng.randint(1, 3)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for _ in range(n_blobs):
            bx = rng.uniform(0.15, 0.85) * w
            by = rng.uniform(0.15, 0.85) * h
            rx = rng.uniform(0.12, 0.34) * w
            ry = rng.uniform(0.10, 0.30) * h
            blob_env = np.exp(-(((xx - bx) / max(1.0, rx)) ** 2 + ((yy - by) / max(1.0, ry)) ** 2))
            blob_col = np.array(
                [rng.randint(20, 210), rng.randint(25, 220), rng.randint(25, 220)], dtype=np.float32
            )[None, None, :]
            b_alpha = (blob_env * (rng.uniform(0.25, 0.52) if strong else rng.uniform(0.15, 0.34)))[:, :, None]
            out = out * (1.0 - b_alpha) + blob_col * b_alpha

    # 3) 高斯底噪 + RGB 通道色噪 + 椒盐斑点
    if strength == "heavy" or rng.random() < 0.80:
        sigma = rng.uniform(18.0, 40.0) if strong else rng.uniform(9.0, 24.0)
        g_noise = np_rng.normal(0.0, sigma, (h, w, 3)).astype(np.float32)
        out += g_noise
    if strength == "heavy" or rng.random() < 0.55:
        sp_prob = rng.uniform(0.025, 0.07) if strong else rng.uniform(0.008, 0.03)
        sp_rand = np_rng.random((h, w))
        salt = (sp_rand < sp_prob * 0.5) & inside
        pepper = (sp_rand > 1.0 - sp_prob * 0.5) & inside
        out[salt] = np.clip(out[salt] + rng.uniform(60.0, 145.0), 0, 255)
        out[pepper] = np.clip(out[pepper] - rng.uniform(45.0, 120.0), 0, 255)

    # 4) 蓝盘表面轻微彩色杂线（保留中心清晰度）
    if strength == "heavy" or rng.random() < 0.50:
        for _ in range(rng.randint(2, 6)):
            p1 = (rng.randint(int(w * 0.1), int(w * 0.9)), rng.randint(int(h * 0.1), int(h * 0.9)))
            p2 = (rng.randint(int(w * 0.1), int(w * 0.9)), rng.randint(int(h * 0.1), int(h * 0.9)))
            lcol = (rng.randint(25, 210), rng.randint(25, 210), rng.randint(25, 210))
            cv2.line(out, p1, p2, lcol, rng.randint(1, 2), cv2.LINE_AA)

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_pvc_gloss_and_scratches(
    bgr: np.ndarray, alpha_mask: np.ndarray, rng: random.Random
) -> np.ndarray:
    """复刻官方亮面 PVC 贴纸的顶部凹凸反光斑与表面细划痕。"""
    h, w = bgr.shape[:2]
    out = bgr.astype(np.float32)

    # 1) 顶部/斜侧凹凸不规则 PVC 亮面高光斑
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    low_noise = np_rng.uniform(0.0, 1.0, (14, 14)).astype(np.float32)
    crinkle = cv2.resize(low_noise, (w, h), interpolation=cv2.INTER_CUBIC)
    crinkle = cv2.GaussianBlur(crinkle, (5, 5), 0.0)

    gx = rng.uniform(0.28, 0.72) * w
    gy = rng.uniform(0.14, 0.42) * h
    rx = rng.uniform(0.18, 0.36) * w
    ry = rng.uniform(0.12, 0.28) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    env = np.exp(-(((xx - gx) / max(1.0, rx)) ** 2 + ((yy - gy) / max(1.0, ry)) ** 2))
    spec_raw = env * np.clip((crinkle - 0.32) * 2.2, 0.0, 1.0)
    spec_alpha = np.clip(spec_raw * rng.uniform(0.55, 1.15), 0.0, 0.92) * (alpha_mask.astype(np.float32) / 255.0)
    glare_col = np.array([252.0, 250.0, 245.0], dtype=np.float32)[None, None, :]
    out = out * (1.0 - spec_alpha[:, :, None]) + glare_col * spec_alpha[:, :, None]

    # 2) 细微表面反光划痕（2-6 条）
    scratch_layer = np.zeros_like(out)
    for _ in range(rng.randint(2, 6)):
        x1 = rng.randint(int(w * 0.1), int(w * 0.9))
        y1 = rng.randint(int(h * 0.1), int(h * 0.9))
        dx = rng.randint(-int(w * 0.3), int(w * 0.3))
        dy = rng.randint(-int(h * 0.3), int(h * 0.3))
        cv2.line(scratch_layer, (x1, y1), (x1 + dx, y1 + dy), (240, 240, 245), 1, cv2.LINE_AA)
    s_alpha = rng.uniform(0.12, 0.28) * (alpha_mask.astype(np.float32) / 255.0)[:, :, None]
    out = out * (1.0 - s_alpha) + np.maximum(out, scratch_layer) * s_alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def make_official_sticker_bgra(
    digit: str,
    rng: random.Random,
    handwritten: bool = False,
    glossy_pvc: bool = False,
    mnist_bank: Optional[Dict[str, np.ndarray]] = None,
    size: int = 160,
    blank_disc_bank: Optional[Sequence[np.ndarray]] = None,
    disc_interference_prob: float = 0.60,
    interference_strength: str = "random",
) -> np.ndarray:
    """复刻官方宝蓝圆贴纸 BGRA（支持真实去字蓝盘底材 + 2026 决赛底噪/纹理/彩斑干扰 + 多样白字）。"""
    s = int(size)
    alpha = np.zeros((s, s), dtype=np.uint8)
    center = (s // 2, s // 2)
    rad = int(s * 0.46)

    used_real_blank = False
    if blank_disc_bank and rng.random() < 0.42:
        tpl = rng.choice(blank_disc_bank)
        tpl_rs = cv2.resize(tpl, (s, s), interpolation=cv2.INTER_LINEAR)
        bgr = tpl_rs[:, :, :3].copy()
        alpha = tpl_rs[:, :, 3].copy() if tpl_rs.shape[2] >= 4 else np.zeros((s, s), dtype=np.uint8)
        if int((alpha > 64).sum()) < 200:
            cv2.circle(alpha, center, rad, 255, -1, cv2.LINE_AA)
        # 轻度亮度/色调微调防止贴纸颜色单一
        bgr = np.clip(bgr.astype(np.float32) * np.array(
            [rng.uniform(0.85, 1.18), rng.uniform(0.85, 1.18), rng.uniform(0.85, 1.22)], dtype=np.float32
        )[None, None, :], 0, 255).astype(np.uint8)
        used_real_blank = True

    if not used_real_blank:
        bgr = np.zeros((s, s, 3), dtype=np.uint8)
        # 覆盖多种蓝贴纸色调：亮宝蓝、暗藏青灰蓝、偏靛蓝/青蓝
        p_pal = rng.random()
        if p_pal < 0.50:
            b = rng.randint(135, 200)
            g = rng.randint(28, 72)
            r = rng.randint(8, 42)
        elif p_pal < 0.85:
            b = rng.randint(68, 145)
            g = rng.randint(min(48, b - 8), min(110, b - 4))
            r = rng.randint(min(38, g - 6), min(96, g))
        else:
            b = rng.randint(120, 190)
            g = rng.randint(65, 125)
            r = rng.randint(18, 65)
        bgr[:, :] = (b, g, r)

        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        grad = 1.0 + rng.uniform(-0.14, 0.14) * ((yy / s) - 0.5) + rng.uniform(-0.12, 0.12) * ((xx / s) - 0.5)
        bgr = np.clip(bgr.astype(np.float32) * grad[:, :, None], 0, 255).astype(np.uint8)
        cv2.circle(alpha, center, rad, 255, -1, cv2.LINE_AA)

        # ~15% 概率保留少许白色底纸边
        if rng.random() < 0.15:
            pad_r = rad + rng.randint(3, 6)
            white_bg = np.full((s, s, 3), rng.randint(225, 250), dtype=np.uint8)
            circ_m = (alpha.astype(np.float32) / 255.0)[:, :, None]
            bgr = np.clip(bgr.astype(np.float32) * circ_m + white_bg.astype(np.float32) * (1.0 - circ_m), 0, 255).astype(np.uint8)
            cv2.circle(alpha, center, min(s // 2 - 1, pad_r), 255, -1, cv2.LINE_AA)

    had_interference = (rng.random() < disc_interference_prob)
    if had_interference:
        bgr = _apply_disc_background_interference(bgr, alpha, rng, strength=interference_strength)

    # 字迹颜色以高对比白/微暖白/微冷灰白为主
    fg_white = rng.choice([
        (rng.randint(235, 255), rng.randint(235, 255), rng.randint(235, 255)),
        (rng.randint(208, 240), rng.randint(228, 252), rng.randint(235, 255)),
        (rng.randint(235, 255), rng.randint(225, 248), rng.randint(210, 238)),
    ])
    _draw_digit_on_bgr(bgr, digit, fg_white, rng, handwritten=handwritten, mnist_bank=mnist_bank)

    # 轻度字迹表面磨损噪点（不破坏数字拓扑）
    if had_interference and rng.random() < 0.35:
        np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        dust = np_rng.normal(0.0, rng.uniform(5.0, 12.0), bgr.shape).astype(np.float32)
        bgr = np.clip(bgr.astype(np.float32) + dust, 0, 255).astype(np.uint8)

    if glossy_pvc:
        bgr = _apply_pvc_gloss_and_scratches(bgr, alpha, rng)

    return np.dstack([bgr, alpha])


def make_digit_card_bgra(
    digit: str,
    rng: random.Random,
    handwritten: bool = False,
    size: Tuple[int, int] = (96, 112),
    mnist_bank: Optional[Dict[str, np.ndarray]] = None,
    circular: Optional[bool] = None,
) -> np.ndarray:
    """生成带 2026 决赛风格彩色/纹理/条纹/涂鸦干扰背景的数字卡片或圆片 BGRA。"""
    w, h = size
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bg_type = rng.choice(["plain_color", "stripes", "checker", "gradient", "noisy_white", "scribble"])

    if bg_type == "noisy_white":
        base = np.array([rng.randint(205, 250), rng.randint(205, 250), rng.randint(205, 250)], dtype=np.uint8)
        bgr[:, :] = base
    elif bg_type == "plain_color":
        base = np.array([rng.randint(70, 245), rng.randint(70, 245), rng.randint(70, 245)], dtype=np.uint8)
        bgr[:, :] = base
    elif bg_type == "stripes":
        c1 = (rng.randint(110, 245), rng.randint(110, 245), rng.randint(110, 245))
        c2 = (rng.randint(60, 210), rng.randint(60, 210), rng.randint(60, 210))
        bgr[:, :] = c1
        step = rng.randint(8, 16)
        horiz = rng.random() < 0.5
        for p in range(0, (h if horiz else w), step * 2):
            if horiz:
                bgr[p:p + step, :] = c2
            else:
                bgr[:, p:p + step] = c2
    elif bg_type == "checker":
        c1 = (rng.randint(135, 245), rng.randint(135, 245), rng.randint(135, 245))
        c2 = (rng.randint(85, 200), rng.randint(85, 200), rng.randint(85, 200))
        tile = rng.randint(10, 18)
        for yy in range(0, h, tile):
            for xx in range(0, w, tile):
                col = c1 if ((yy // tile + xx // tile) % 2 == 0) else c2
                bgr[yy:yy + tile, xx:xx + tile] = col
    elif bg_type == "scribble":
        bgr[:, :] = (rng.randint(120, 235), rng.randint(120, 235), rng.randint(120, 235))
        for _ in range(rng.randint(8, 16)):
            p1 = (rng.randint(0, w - 1), rng.randint(0, h - 1))
            p2 = (rng.randint(0, w - 1), rng.randint(0, h - 1))
            col = (rng.randint(40, 220), rng.randint(40, 220), rng.randint(40, 220))
            cv2.line(bgr, p1, p2, col, rng.randint(1, 3), cv2.LINE_AA)
    else:  # gradient
        c1 = np.array([rng.randint(85, 240) for _ in range(3)], dtype=np.float32)
        c2 = np.array([rng.randint(85, 240) for _ in range(3)], dtype=np.float32)
        alpha_g = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
        grad = (1.0 - alpha_g) * c1 + alpha_g * c2
        bgr[:, :] = np.repeat(grad.astype(np.uint8), h, axis=0)

    # 细碎视觉纹理噪声（斑点/细线干扰）
    for _ in range(rng.randint(2, 6)):
        p1 = (rng.randint(0, w - 1), rng.randint(0, h - 1))
        p2 = (rng.randint(0, w - 1), rng.randint(0, h - 1))
        col = (rng.randint(70, 225), rng.randint(70, 225), rng.randint(70, 225))
        cv2.line(bgr, p1, p2, col, 1, cv2.LINE_AA)

    # 数字笔迹颜色（与背景亮度拉开对比，含高对比黑/白/彩色字）
    bg_luma = float(bgr.mean())
    if bg_luma > 135:
        fg = rng.choice([
            (rng.randint(5, 50), rng.randint(5, 50), rng.randint(5, 60)),
            (rng.randint(140, 200), rng.randint(15, 55), rng.randint(10, 45)),
            (rng.randint(10, 45), rng.randint(15, 55), rng.randint(160, 225)),
        ])
    else:
        fg = (rng.randint(218, 255), rng.randint(218, 255), rng.randint(218, 255))

    _draw_digit_on_bgr(bgr, digit, fg, rng, handwritten=handwritten, mnist_bank=mnist_bank)

    use_circle = (rng.random() < 0.45) if circular is None else circular
    if use_circle:
        alpha = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(alpha, (w // 2, h // 2), (int(w * 0.46), int(h * 0.46)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    else:
        if rng.random() < 0.5:
            cv2.rectangle(bgr, (1, 1), (w - 2, h - 2), (40, 40, 40), 2)
        alpha = np.full((h, w), 255, dtype=np.uint8)

    return np.dstack([bgr, alpha])


def make_digit_sticker_by_mix(
    digit: str,
    rng: random.Random,
    mnist_bank: Optional[Dict[str, np.ndarray]] = None,
    blank_disc_bank: Optional[Sequence[np.ndarray]] = None,
    disc_interference_prob: float = 0.60,
    interference_strength: str = "random",
) -> np.ndarray:
    """按决赛强泛化分布生成数字贴纸：蓝底圆片（含底噪/纹理/反光）+ 2026 非蓝彩色干扰底纹圆片/卡片。"""
    p = rng.random()
    handwritten = rng.random() < 0.58
    if p < 0.42:
        return make_official_sticker_bgra(
            digit, rng, handwritten=handwritten, glossy_pvc=False, mnist_bank=mnist_bank, size=160,
            blank_disc_bank=blank_disc_bank, disc_interference_prob=disc_interference_prob,
            interference_strength=interference_strength,
        )
    if p < 0.76:
        return make_official_sticker_bgra(
            digit, rng, handwritten=handwritten, glossy_pvc=True, mnist_bank=mnist_bank, size=160,
            blank_disc_bank=blank_disc_bank, disc_interference_prob=disc_interference_prob,
            interference_strength=interference_strength,
        )
    is_circ = rng.random() < 0.65
    sz = (152, 152) if is_circ else (128, 152)
    return make_digit_card_bgra(
        digit, rng, handwritten=handwritten, size=sz, mnist_bank=mnist_bank, circular=is_circ
    )


def make_red_cone_bgra(rng: random.Random, size: Tuple[int, int] = (72, 112)) -> np.ndarray:
    """生成 13.5x13.5x23cm 比例的红色锥桶 BGRA（带底座与可选反光条）。"""
    w, h = size
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)
    red_main = (rng.randint(15, 55), rng.randint(15, 55), rng.randint(175, 245))
    red_dark = (max(0, red_main[0] - 10), max(0, red_main[1] - 10), max(110, red_main[2] - 45))

    base_y = int(h * 0.80)
    cv2.rectangle(bgr, (int(w * 0.08), base_y), (int(w * 0.92), h - 2), red_dark, -1)
    cv2.rectangle(alpha, (int(w * 0.08), base_y), (int(w * 0.92), h - 2), 255, -1)

    cone_pts = np.array([
        [int(w * 0.42), int(h * 0.06)],
        [int(w * 0.58), int(h * 0.06)],
        [int(w * 0.82), base_y + 4],
        [int(w * 0.18), base_y + 4],
    ], dtype=np.int32)
    cv2.fillPoly(bgr, [cone_pts], red_main)
    cv2.fillPoly(alpha, [cone_pts], 255)

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


# ---------------------------------------------------------------- 3D 地面单应透视与几何增强（禁止翻转！）

def project_3d_to_uv(
    w_mm: float,
    d_mm: float,
    z_mm: float = 0.0,
    cam_h_mm: float = 110.0,
    pitch_deg: float = 12.5,
    f_px: float = 325.0,
    width: int = 640,
    height: int = 480,
) -> Tuple[float, float]:
    """与 tools/sim/render.py CameraModel 一致的 3D->像素投影（z_mm=0 与 wd_to_uv 完全一致）。"""
    scale = width / 640.0
    f = f_px * scale
    cx = width * 0.5
    cy = height * 0.5
    tan_a = math.tan(math.radians(pitch_deg))
    dz = cam_h_mm - z_mm
    denom = d_mm + dz * tan_a
    if denom <= 1.0:
        denom = 1.0
    u = cx + f * w_mm / denom
    v = cy + f * (dz - d_mm * tan_a) / denom
    return u, v


def project_ground_point_homography(
    x_fwd_m: float,
    y_left_m: float,
    width: int = 640,
    height: int = 480,
    H_g2p: np.ndarray = CALIBRATED_H_G2P,
) -> Optional[Tuple[float, float]]:
    """使用实车标定单应逆矩阵将地面点 (x_fwd_m, y_left_m) 映射到图像坐标 (u, v)。"""
    p = H_g2p @ np.array([float(x_fwd_m), float(y_left_m), 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-8:
        return None
    u = float(p[0] / p[2]) * (float(width) / 640.0)
    v = float(p[1] / p[2]) * (float(height) / 480.0)
    return u, v


def project_ground_prop_bgra(
    bg: np.ndarray,
    prop_bgra: np.ndarray,
    d_m: float,
    y_m: float,
    phys_w_m: float,
    phys_h_m: float,
    rot_deg: float = 0.0,
    curl_deg: float = 0.0,
    cam_h_mm: float = 110.0,
    pitch_deg: float = 12.5,
    f_px: float = 325.0,
    allow_flip: bool = False,
    min_box_w: int = 5,
    min_box_h: int = 4,
    use_homography: bool = False,
    allow_wide_yaw: bool = False,
) -> Optional[Tuple[int, int, int, int]]:
    """按小车相机单应/针孔内参将平面贴纸/道具做真实 3D 地面透视并混合到 bg 上。

    allow_wide_yaw=True 仅用于地面圆形数字贴纸在平面内 ±180° 任意朝向（如 8 横贴成 ∞、6 斜贴），
    仍严禁任何镜像翻转（allow_flip=True 必报错）。
    """
    if allow_flip:
        raise ValueError("严格禁止水平/垂直翻转（会破坏数字 1-8 类别语义）！")
    max_allowed = 180.0 if allow_wide_yaw else MAX_ROTATION_DEG
    if abs(rot_deg) > max_allowed + 1e-6:
        raise ValueError(f"rot_deg={rot_deg} 超过允许上限 ±{max_allowed}°")

    H_img, W_img = bg.shape[:2]
    ph, pw = prop_bgra.shape[:2]
    hw_mm = phys_w_m * 500.0
    hh_mm = phys_h_m * 500.0
    d_mm = d_m * 1000.0
    w_mm = -y_m * 1000.0  # ROS +y 向左 -> 车体 w_mm 向右为负

    local_pts = [
        (-hw_mm, +hh_mm),
        (+hw_mm, +hh_mm),
        (+hw_mm, -hh_mm),
        (-hw_mm, -hh_mm),
    ]
    rad_r = math.radians(rot_deg)
    cos_r, sin_r = math.cos(rad_r), math.sin(rad_r)
    curl_rad = math.radians(max(0.0, min(35.0, curl_deg)))

    dst_pts: List[List[float]] = []
    shadow_pts: List[List[float]] = []
    pose_scale = cam_h_mm / 110.0
    for lx, ly in local_pts:
        rx = lx * cos_r - ly * sin_r
        ry = lx * sin_r + ly * cos_r
        lift_frac = max(0.0, (ly + hh_mm) / max(1e-3, 2.0 * hh_mm))
        z_mm = (2.0 * hh_mm * lift_frac) * math.sin(curl_rad)
        dd_mm = d_mm + ry * math.cos(curl_rad)
        ww_mm = w_mm + rx
        if use_homography and curl_deg <= 1.0:
            # 轻度车身俯仰颠簸微调（绕标定单应扰动 ±4%）
            pt_h = project_ground_point_homography(
                (dd_mm / 1000.0) * pose_scale, -ww_mm / 1000.0, width=W_img, height=H_img
            )
            if pt_h is None:
                return None
            u, v = pt_h
        else:
            u, v = project_3d_to_uv(ww_mm, dd_mm, z_mm, cam_h_mm, pitch_deg, f_px, W_img, H_img)
        dst_pts.append([u, v])
        if curl_deg > 1.0:
            su, sv = project_3d_to_uv(ww_mm, dd_mm, 0.0, cam_h_mm, pitch_deg, f_px, W_img, H_img)
            shadow_pts.append([su, sv])

    dst_arr = np.array(dst_pts, dtype=np.float32)
    min_u, min_v = float(dst_arr[:, 0].min()), float(dst_arr[:, 1].min())
    max_u, max_v = float(dst_arr[:, 0].max()), float(dst_arr[:, 1].max())

    bx0 = max(0, int(math.floor(min_u)) - 2)
    by0 = max(0, int(math.floor(min_v)) - 2)
    bx1 = min(W_img, int(math.ceil(max_u)) + 3)
    by1 = min(H_img, int(math.ceil(max_v)) + 3)
    roi_w, roi_h = bx1 - bx0, by1 - by0
    if roi_w < min_box_w or roi_h < min_box_h:
        return None

    # 绘制翘边软阴影（在地面投影多边形内微微压暗路面）
    if curl_deg > 2.0 and len(shadow_pts) == 4:
        sh_arr = np.array(shadow_pts, dtype=np.int32)
        sh_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillPoly(sh_mask, [sh_arr - np.array([bx0, by0], dtype=np.int32)], 255)
        sh_mask = cv2.GaussianBlur(sh_mask, (7, 7), 0)
        sh_f = (sh_mask.astype(np.float32) / 255.0) * min(0.45, curl_deg / 45.0)
        roi_bg = bg[by0:by1, bx0:bx1].astype(np.float32)
        bg[by0:by1, bx0:bx1] = np.clip(roi_bg * (1.0 - sh_f[:, :, None]), 0, 255).astype(np.uint8)

    src_arr = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    dst_local = dst_arr - np.float32([[bx0, by0]])
    H_mat = cv2.getPerspectiveTransform(src_arr, dst_local)
    warped = cv2.warpPerspective(
        prop_bgra,
        H_mat,
        (roi_w, roi_h),
        flags=cv2.INTER_AREA if roi_h < ph * 0.6 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    ys, xs = np.nonzero(warped[:, :, 3] > 28)
    if len(xs) < 12:
        return None
    px0, px1 = int(xs.min()), int(xs.max()) + 1
    py0, py1 = int(ys.min()), int(ys.max()) + 1
    if (px1 - px0) < min_box_w or (py1 - py0) < min_box_h:
        return None

    alpha_f = warped[:, :, 3:4].astype(np.float32) / 255.0
    roi = bg[by0:by1, bx0:bx1].astype(np.float32)
    blended = alpha_f * warped[:, :, :3].astype(np.float32) + (1.0 - alpha_f) * roi
    bg[by0:by1, bx0:bx1] = np.clip(blended, 0, 255).astype(np.uint8)
    return bx0 + px0, by0 + py0, bx0 + px1, by0 + py1


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


def apply_finals_venue_degradation(
    img: np.ndarray,
    rng: random.Random,
    enable_specular: bool = True,
    enable_yellowing: bool = True,
    enable_led_temp: bool = True,
    allow_flip: bool = False,
) -> np.ndarray:
    """Tier 1-④-b 决赛赛场退化增强：局部镜面反光高光斑 + 赛垫泛黄/老化色偏 + 冷暖 LED 色温漂移。

    严格禁止翻转或几何镜像操作（仅作用于光度/色度域）。
    """
    if allow_flip:
        raise ValueError("严格禁止水平/垂直翻转（会破坏数字 1-8 类别语义）！")

    out = img.astype(np.float32)
    h, w = out.shape[:2]

    # 1) 赛垫泛黄 / 老化色偏（衰减蓝色通道并略升红绿，伴随地面不均匀老化渐变）
    if enable_yellowing:
        yellow_strength = rng.uniform(0.10, 0.28)
        yy_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        # 地面区域（图像下半部）泛黄更明显
        ground_mask = np.clip((yy_norm - 0.20) / 0.55, 0.25, 1.0)
        b_scale = 1.0 - yellow_strength * ground_mask
        g_scale = 1.0 - 0.12 * yellow_strength * ground_mask
        r_scale = 1.0 + 0.18 * yellow_strength * ground_mask
        out[:, :, 0] *= b_scale
        out[:, :, 1] *= g_scale
        out[:, :, 2] *= r_scale

    # 2) 冷暖 LED 色温漂移（暖灯 ~3200K 增红降蓝 / 冷灯 ~7000K 增蓝降红 + 轻微荧光绿偏移）
    if enable_led_temp:
        # temp_shift > 0 偏暖，< 0 偏冷；避开 0 附近保证有效漂移
        sign = -1.0 if rng.random() < 0.5 else 1.0
        mag = rng.uniform(0.06, 0.16)
        shift = sign * mag
        g_tint = rng.uniform(-0.04, 0.04)
        out[:, :, 0] *= (1.0 - shift)
        out[:, :, 1] *= (1.0 + g_tint)
        out[:, :, 2] *= (1.0 + shift)

    # 3) 局部镜面反光高光斑（模拟顶灯/聚光灯在光滑赛垫与 PVC 贴纸上的各向异性亮斑）
    if enable_specular:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        n_spots = rng.randint(1, 2)
        for _ in range(n_spots):
            gx = rng.uniform(0.15, 0.85) * w
            gy = rng.uniform(0.32, 0.88) * h
            rx = rng.uniform(0.06, 0.22) * w
            ry = rng.uniform(0.03, 0.11) * h
            theta = math.radians(rng.uniform(-35.0, 35.0))
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            dx = xx - gx
            dy = yy - gy
            xr = dx * cos_t + dy * sin_t
            yr = -dx * sin_t + dy * cos_t
            spot = np.exp(-0.5 * ((xr / max(1.0, rx)) ** 2 + (yr / max(1.0, ry)) ** 2))
            intensity = rng.uniform(55.0, 135.0)
            tint = np.array(
                [rng.uniform(0.92, 1.0), rng.uniform(0.96, 1.0), 1.0], dtype=np.float32
            )[None, None, :]
            out += (intensity * spot)[:, :, None] * tint

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_photometric_augmentation(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """光照与镜头域随机化：色温偏置、亮度增益、决赛赛场退化增强、定向眩光、轻微运动模糊、噪声。"""
    # 随机触发决赛赛场退化增强（镜面高光斑 / 赛垫泛黄 / 冷暖 LED 色温漂移）
    p_spec = rng.random() < 0.45
    p_yell = rng.random() < 0.45
    p_led = rng.random() < 0.50
    if p_spec or p_yell or p_led:
        img = apply_finals_venue_degradation(
            img,
            rng,
            enable_specular=p_spec,
            enable_yellowing=p_yell,
            enable_led_temp=p_led,
            allow_flip=False,
        )

    out = img.astype(np.float32)
    h, w = out.shape[:2]

    gain = rng.uniform(0.75, 1.24)
    b_mul = rng.uniform(0.87, 1.15)
    g_mul = rng.uniform(0.93, 1.07)
    r_mul = rng.uniform(0.87, 1.15)
    out[:, :, 0] *= gain * b_mul
    out[:, :, 1] *= gain * g_mul
    out[:, :, 2] *= gain * r_mul

    if rng.random() < 0.60:
        gx = rng.uniform(0.15, 0.85) * w
        gy = rng.uniform(0.25, 0.90) * h
        sigma = rng.uniform(0.18, 0.45) * max(w, h)
        strength = rng.uniform(20.0, 75.0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        glare = strength * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * sigma * sigma))
        out += glare[:, :, None]

    out = np.clip(out, 0, 255).astype(np.uint8)

    if rng.random() < 0.30:
        k = rng.choice([3, 5])
        kernel = np.zeros((k, k), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)
    elif rng.random() < 0.24:
        # RDK X5 UVC 镜头轻度离焦模糊
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.5, 1.1))

    if rng.random() < 0.40:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0.0, rng.uniform(1.8, 5.5), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.35:
        # 复刻实车 UVC MJPEG 压缩块效应与色度轻度抽样失真
        ok_m, enc_m = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(58, 86)])
        if ok_m:
            dec_m = cv2.imdecode(enc_m, cv2.IMREAD_COLOR)
            if dec_m is not None:
                out = dec_m

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


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def load_blank_disc_bank(props_dir: Optional[str]) -> List[np.ndarray]:
    """加载真实去字空白蓝圆盘底材 (_blank_discs/*.png) 用于合成全 8 类数字的真实纹理贴纸。"""
    if not props_dir:
        return []
    bdir = Path(props_dir) / "_blank_discs"
    if not bdir.is_dir():
        return []
    out: List[np.ndarray] = []
    for p in load_image_paths(str(bdir)):
        arr = read_prop_bgra(p)
        if arr is not None:
            out.append(arr)
    return out


def apply_roi_crop_remap(
    img: np.ndarray,
    labels: Sequence[str],
    roi_crop_top_px: int,
    out_w: int = 640,
    out_h: int = 640,
    min_box_w_px: float = 5.0,
    min_box_h_px: float = 4.0,
) -> Tuple[np.ndarray, List[str]]:
    """对图像与 YOLO 标注执行与实车 bpu_yolov8_node 一致的顶部裁剪 + 垂直拉伸。"""
    crop_top = int(roi_crop_top_px or 0)
    if crop_top <= 0:
        return img, list(labels)

    H, W = img.shape[:2]
    roi_top = max(0, min(crop_top, H - 32))
    crop_img = img[roi_top:H, :]
    roi_h = float(max(1, H - roi_top))
    if crop_img.shape[1] != out_w or crop_img.shape[0] != out_h:
        resized = cv2.resize(crop_img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = crop_img

    remapped: List[str] = []
    for ln in labels:
        parts = ln.strip().split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        x1 = (cx - bw * 0.5) * W
        x2 = (cx + bw * 0.5) * W
        y1 = (cy - bh * 0.5) * H - float(roi_top)
        y2 = (cy + bh * 0.5) * H - float(roi_top)

        cx1 = clamp(x1, 0.0, float(W))
        cx2 = clamp(x2, 0.0, float(W))
        cy1 = clamp(y1, 0.0, roi_h)
        cy2 = clamp(y2, 0.0, roi_h)

        # 转入 out_w x out_h 模型像素空间校验最低可辨尺寸
        pw = (cx2 - cx1) * (float(out_w) / float(max(1, W)))
        ph = (cy2 - cy1) * (float(out_h) / roi_h)
        if pw < min_box_w_px or ph < min_box_h_px:
            continue

        ncx = clamp(0.5 * (cx1 + cx2) / float(W), 0.0, 1.0)
        ncy = clamp(0.5 * (cy1 + cy2) / roi_h, 0.0, 1.0)
        nbw = clamp((cx2 - cx1) / float(W), 1e-4, 1.0)
        nbh = clamp((cy2 - cy1) / roi_h, 1e-4, 1.0)
        remapped.append(f"{cid} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")

    return resized, remapped


def _weighted_sample_distinct(
    rng: random.Random,
    items: Sequence[str],
    weights: Sequence[float],
    k: int,
) -> List[str]:
    avail = list(items)
    w_avail = [max(1e-4, float(w)) for w in weights]
    chosen: List[str] = []
    for _ in range(min(k, len(avail))):
        pick = rng.choices(avail, weights=w_avail, k=1)[0]
        idx = avail.index(pick)
        chosen.append(pick)
        avail.pop(idx)
        w_avail.pop(idx)
    return chosen


def _generate_digits8_sample(
    rng: random.Random,
    bg_paths: Sequence[Path],
    prop_bank: Dict[str, List[Path]],
    mnist_bank: Optional[Dict[str, np.ndarray]],
    width: int,
    height: int,
    max_rot_deg: float,
    max_digit_dist_m: float = DEFAULT_MAX_DIGIT_DIST_8CLS_M,
    blank_disc_bank: Optional[Sequence[np.ndarray]] = None,
    use_homography: bool = False,
    allow_wide_yaw: bool = False,
    roi_crop_top_px: int = 0,
    empty_ratio: float = 0.08,
    prop_reuse_prob: float = 0.70,
    disc_interference_prob: float = 0.60,
    interference_strength: str = "random",
    class_weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """生成 8 类（1..8）真实 3D 地面单应透视样本，并将红锥/红禁行区/黄线作为无标签干扰背景绘制。"""
    max_d = resolve_max_digit_dist_m(8, max_digit_dist_m)
    c_weights = list(class_weights) if (class_weights is not None and len(class_weights) == 8) else [1.0] * 8

    using_real_bg = False
    if bg_paths:
        raw_bg = cv2.imdecode(np.fromfile(str(rng.choice(bg_paths)), dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw_bg is not None:
            img = cv2.resize(raw_bg, (width, height))
            using_real_bg = True
        else:
            img = make_synthetic_track_bg(rng, width, height)
    else:
        img = make_synthetic_track_bg(rng, width, height)

    # 真实背景上自动提取白道区域掩码，确保贴纸落在真实白道而非蓝垫/红锥上
    real_white_mask: Optional[np.ndarray] = None
    if using_real_bg:
        hsv_bg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        real_white_mask = ((hsv_bg[:, :, 1] < 82) & (hsv_bg[:, :, 2] > 102)).astype(np.uint8)
        real_white_mask[: int(height * 0.34), :] = 0
        k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
        real_white_mask = cv2.erode(real_white_mask, k_er, iterations=1)

    # 相机参数微扰（围绕 h=0.11m, pitch=12.5°, f=325px）
    cam_h_mm = rng.uniform(104.0, 118.0)
    pitch_deg = rng.uniform(11.0, 14.5)
    f_px = rng.uniform(318.0, 332.0)

    def get_prop(cls_name: str) -> np.ndarray:
        if prop_bank.get(cls_name) and rng.random() < prop_reuse_prob:
            loaded = read_prop_bgra(rng.choice(prop_bank[cls_name]))
            if loaded is not None:
                return loaded
        if cls_name in DIGITS8_CLASS_TO_ID:
            return make_digit_sticker_by_mix(
                cls_name,
                rng,
                mnist_bank=mnist_bank,
                blank_disc_bank=blank_disc_bank,
                disc_interference_prob=disc_interference_prob,
                interference_strength=interference_strength,
            )
        if cls_name == "red_cone":
            return make_red_cone_bgra(rng)
        if cls_name == "red_no_entry":
            return make_red_no_entry_bgra(rng)
        return make_yellow_stop_line_bgra(rng)

    # 1) 无标签背景干扰物：真实背景自带红锥/红禁行区/黄线，仅在合成背景上高频绘制
    if not using_real_bg and rng.random() < 0.45:
        d_y = rng.uniform(0.20, 0.90) if use_homography else rng.uniform(0.30, 1.05)
        y_lat = rng.uniform(-0.08, 0.08)
        y_prop = get_prop("yellow_stop_line")
        project_ground_prop_bgra(
            img, y_prop, d_m=d_y, y_m=y_lat, phys_w_m=rng.uniform(0.28, 0.42), phys_h_m=rng.uniform(0.025, 0.045),
            rot_deg=rng.uniform(-min(6.0, max_rot_deg), min(6.0, max_rot_deg)),
            cam_h_mm=cam_h_mm, pitch_deg=pitch_deg, f_px=f_px, allow_flip=False, min_box_w=12, min_box_h=2,
            use_homography=use_homography, allow_wide_yaw=False,
        )

    if not using_real_bg and rng.random() < 0.45:
        d_n = rng.uniform(0.24, 0.95) if use_homography else rng.uniform(0.35, 1.15)
        y_n = rng.uniform(-0.14, 0.14)
        n_prop = get_prop("red_no_entry")
        project_ground_prop_bgra(
            img, n_prop, d_m=d_n, y_m=y_n, phys_w_m=0.24, phys_h_m=0.13,
            rot_deg=rng.uniform(-min(10.0, max_rot_deg), min(10.0, max_rot_deg)),
            cam_h_mm=cam_h_mm, pitch_deg=pitch_deg, f_px=f_px, allow_flip=False, min_box_w=10, min_box_h=3,
            use_homography=use_homography, allow_wide_yaw=False,
        )

    # 2) 地面数字贴纸布局（直道并排 / 岔道双支路 / 单弯道 / 近处边缘截断），所有距离严格 <= max_d
    labels: List[str] = []
    placed_boxes: List[Tuple[int, int, int, int]] = []
    placements: List[Tuple[str, float, float, float, float]] = []

    def _u_dist(lo: float, hi: float) -> float:
        hi_c = min(hi, max_d)
        lo_c = min(lo, hi_c)
        return rng.uniform(lo_c, hi_c)

    if rng.random() >= clamp(empty_ratio, 0.0, 0.5):
        scene_mode = rng.choices(
            ["straight_dual", "fork_branches", "single_curve", "near_clipped", "multi_depth"],
            weights=[0.34, 0.24, 0.18, 0.14, 0.10],
        )[0]

        def rand_diam(d_m: float) -> float:
            if use_homography:
                # 实车蓝圆盘真实直径 ~0.10–0.14 m
                return rng.uniform(0.105, 0.148)
            if d_m > 0.90:
                return rng.uniform(0.14, 0.22)
            return rng.uniform(0.11, 0.16)

        if scene_mode == "straight_dual":
            d1, d2 = _weighted_sample_distinct(rng, DIGITS8_CLASSES, c_weights, 2)
            d_base = _u_dist(0.12, 0.72) if use_homography else _u_dist(0.24, 0.85)
            sep = rng.uniform(0.11, 0.19)
            y_mid = rng.uniform(-0.07, 0.07)
            d_floor = 0.06 if use_homography else 0.15
            placements.append((d1, clamp(d_base + rng.uniform(-0.02, 0.02), d_floor, max_d), clamp(y_mid + sep * 0.5, -0.20, 0.20), rand_diam(d_base), 0.0))
            placements.append((d2, clamp(d_base + rng.uniform(-0.02, 0.02), d_floor, max_d), clamp(y_mid - sep * 0.5, -0.20, 0.20), rand_diam(d_base), 0.0))
        elif scene_mode == "fork_branches":
            d1, d2 = _weighted_sample_distinct(rng, DIGITS8_CLASSES, c_weights, 2)
            da = _u_dist(0.18, 1.00) if use_homography else _u_dist(0.32, 1.15)
            db = clamp(da + rng.uniform(-0.14, 0.14), min(0.16, max_d), max_d)
            placements.append((d1, da, rng.uniform(0.08, 0.19), rand_diam(da), rng.uniform(0.0, 12.0) if rng.random() < 0.20 else 0.0))
            placements.append((d2, db, rng.uniform(-0.19, -0.08), rand_diam(db), rng.uniform(0.0, 12.0) if rng.random() < 0.20 else 0.0))
        elif scene_mode == "single_curve":
            d1 = _weighted_sample_distinct(rng, DIGITS8_CLASSES, c_weights, 1)[0]
            da = _u_dist(0.12, 1.05) if use_homography else _u_dist(0.26, 1.30)
            placements.append((d1, da, rng.uniform(-0.18, 0.18), rand_diam(da), rng.uniform(0.0, 18.0) if rng.random() < 0.18 else 0.0))
            if rng.random() < 0.45:
                rem = [c for c in DIGITS8_CLASSES if c != d1]
                rem_w = [c_weights[DIGITS8_CLASS_TO_ID[c]] for c in rem]
                d2 = _weighted_sample_distinct(rng, rem, rem_w, 1)[0]
                db = _u_dist(0.38, 1.10) if use_homography else _u_dist(0.55, 1.50)
                placements.append((d2, db, rng.uniform(-0.18, 0.18), rand_diam(db), rng.uniform(0.0, 18.0) if rng.random() < 0.25 else 0.0))
        elif scene_mode == "near_clipped":
            d1 = _weighted_sample_distinct(rng, DIGITS8_CLASSES, c_weights, 1)[0]
            da = _u_dist(0.055, 0.14) if use_homography else _u_dist(0.15, 0.24)
            ya = rng.choice([rng.uniform(-0.19, -0.10), rng.uniform(0.10, 0.19), rng.uniform(-0.08, 0.08)])
            placements.append((d1, da, ya, rng.uniform(0.105, 0.15), 0.0))
            if rng.random() < 0.5:
                rem = [c for c in DIGITS8_CLASSES if c != d1]
                rem_w = [c_weights[DIGITS8_CLASS_TO_ID[c]] for c in rem]
                d2 = _weighted_sample_distinct(rng, rem, rem_w, 1)[0]
                db = _u_dist(0.30, 0.95) if use_homography else _u_dist(0.45, 1.10)
                placements.append((d2, db, rng.uniform(-0.14, 0.14), rand_diam(db), 0.0))
        else:  # multi_depth
            chosen = _weighted_sample_distinct(rng, DIGITS8_CLASSES, c_weights, rng.randint(2, 3))
            if use_homography:
                depths = sorted([
                    _u_dist(0.09, 0.22),
                    _u_dist(0.28, min(0.58, max_d)),
                    _u_dist(0.62, 1.05),
                ][:len(chosen)])
            else:
                depths = sorted([
                    _u_dist(0.22, 0.45),
                    _u_dist(0.48, min(0.85, max_d)),
                    _u_dist(0.80, 1.52),
                ][:len(chosen)])
            for dg, dm in zip(chosen, depths):
                curl = rng.uniform(5.0, 20.0) if (dm > 0.75 and rng.random() < 0.4) else 0.0
                placements.append((dg, dm, rng.uniform(-0.17, 0.17), rand_diam(dm), curl))

        # 按由远到近顺序绘制（正确遮挡关系），并对每张贴纸距离强制封顶 <= max_d
        min_bw_eff = 10 if max_d <= 0.85 else 6
        min_bh_eff = 8 if max_d <= 0.85 else 4
        placements.sort(key=lambda item: item[1], reverse=True)
        for digit, d_m, y_m, diam_m, curl_deg in placements:
            d_m = min(d_m, max_d)
            if real_white_mask is not None:
                if use_homography:
                    pt0 = project_ground_point_homography(d_m, y_m, width=width, height=height)
                    u0, v0 = pt0 if pt0 is not None else (-1.0, -1.0)
                else:
                    u0, v0 = project_3d_to_uv(-y_m * 1000.0, d_m * 1000.0, 0.0, cam_h_mm, pitch_deg, f_px, width, height)
                ui, vi = int(round(u0)), int(round(v0))
                on_white = (0 <= ui < width and 0 <= vi < height and real_white_mask[vi, ui] > 0)
                if not on_white and 0 <= vi < height:
                    xs_w = np.nonzero(real_white_mask[vi, :] > 0)[0]
                    if len(xs_w) >= 24:
                        u_new = float(xs_w[rng.randint(int(len(xs_w) * 0.15), int(len(xs_w) * 0.85))])
                        if use_homography:
                            gp = CALIBRATED_H_P2G @ np.array([u_new * (640.0 / width), float(vi) * (480.0 / height), 1.0], dtype=np.float64)
                            if abs(gp[2]) > 1e-8:
                                d_m = clamp(float(gp[0] / gp[2]), 0.05, max_d)
                                y_m = clamp(float(gp[1] / gp[2]), -0.35, 0.35)
                                on_white = True
                        else:
                            scale_w = width / 640.0
                            denom = max(1.0, d_m * 1000.0 + cam_h_mm * math.tan(math.radians(pitch_deg)))
                            w_mm_new = (u_new - width * 0.5) * denom / max(1.0, f_px * scale_w)
                            y_m = clamp(-w_mm_new / 1000.0, -0.35, 0.35)
                            on_white = True
                if not on_white:
                    continue

            prop = get_prop(digit)
            ph_p, pw_p = prop.shape[:2]
            aspect = pw_p / max(1.0, float(ph_p))
            phys_w = diam_m * (aspect if aspect < 1.0 else 1.0)
            phys_h = diam_m / (aspect if aspect > 1.0 else 1.0)
            if allow_wide_yaw:
                p_rot = rng.random()
                if p_rot < 0.50:
                    rot = rng.uniform(-25.0, 25.0)
                elif p_rot < 0.80:
                    rot = rng.choice([-1.0, 1.0]) * rng.uniform(25.0, 115.0)
                else:
                    rot = rng.choice([-1.0, 1.0]) * rng.uniform(115.0, 180.0)
            else:
                rot = rng.uniform(-max_rot_deg, max_rot_deg)

            box = project_ground_prop_bgra(
                img, prop, d_m=d_m, y_m=y_m, phys_w_m=phys_w, phys_h_m=phys_h,
                rot_deg=rot, curl_deg=curl_deg, cam_h_mm=cam_h_mm, pitch_deg=pitch_deg,
                f_px=f_px, allow_flip=False, min_box_w=min_bw_eff, min_box_h=min_bh_eff,
                use_homography=use_homography, allow_wide_yaw=allow_wide_yaw,
            )
            if box is None:
                continue
            if any(_box_iou(box, prev) > 0.25 for prev in placed_boxes):
                continue
            ln = bbox_to_yolo(DIGITS8_CLASS_TO_ID[digit], box, width, height)
            if ln:
                placed_boxes.append(box)
                labels.append(ln)

    # 3) 站立红锥作为无标签背景干扰物（仅合成背景绘制；真实背景已自带真实红锥）
    if not using_real_bg and rng.random() < 0.50:
        for _ in range(rng.randint(1, 2)):
            d_c = rng.uniform(0.35, 1.10) if use_homography else rng.uniform(0.45, 1.45)
            y_c = rng.uniform(-0.22, 0.22)
            u_c, v_base = project_3d_to_uv(-y_c * 1000.0, d_c * 1000.0, 0.0, cam_h_mm, pitch_deg, f_px, width, height)
            _, v_top = project_3d_to_uv(-y_c * 1000.0, d_c * 1000.0, 230.0, cam_h_mm, pitch_deg, f_px, width, height)
            ch = int(clamp(v_base - v_top, 20, 190))
            cw = int(clamp(ch * 0.60, 12, 115))
            cy_c = int(round((v_base + v_top) * 0.5))
            cx_c = int(round(u_c))
            cand_box = (cx_c - cw // 2, cy_c - ch // 2, cx_c + cw // 2, cy_c + ch // 2)
            if any(_box_iou(cand_box, pb) > 0.12 for pb in placed_boxes):
                continue
            cone = cv2.resize(get_prop("red_cone"), (cw, ch))
            cone = warp_prop_bgra(cone, rng, max_rot_deg=min(8.0, max_rot_deg), perspective_strength=0.04)
            composite_bgra(img, cone, cx_c, cy_c)

    img = apply_photometric_augmentation(img, rng)
    if roi_crop_top_px > 0:
        img, labels = apply_roi_crop_remap(img, labels, roi_crop_top_px=roi_crop_top_px, out_w=640, out_h=640)
    return img, labels


def generate_one_sample(
    rng: random.Random,
    bg_paths: Sequence[Path],
    prop_bank: Dict[str, List[Path]],
    width: int = 640,
    height: int = 480,
    max_rot_deg: float = MAX_ROTATION_DEG,
    num_classes: int = 11,
    mnist_bank: Optional[Dict[str, np.ndarray]] = None,
    max_digit_dist_m: Optional[float] = None,
    blank_disc_bank: Optional[Sequence[np.ndarray]] = None,
    use_homography: bool = False,
    allow_wide_yaw: bool = False,
    roi_crop_top_px: int = 0,
    empty_ratio: float = 0.08,
    prop_reuse_prob: float = 0.70,
    disc_interference_prob: float = 0.60,
    interference_strength: str = "random",
    class_weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """生成单张训练样本及其 8 类或 11 类 YOLO 标注行。"""
    if num_classes not in (8, 11):
        raise ValueError(f"num_classes 仅支持 8 或 11，收到: {num_classes}")
    eff_max_dist_m = resolve_max_digit_dist_m(num_classes, max_digit_dist_m)
    if num_classes == 8:
        return _generate_digits8_sample(
            rng=rng, bg_paths=bg_paths, prop_bank=prop_bank, mnist_bank=mnist_bank,
            width=width, height=height, max_rot_deg=max_rot_deg,
            max_digit_dist_m=eff_max_dist_m,
            blank_disc_bank=blank_disc_bank,
            use_homography=use_homography,
            allow_wide_yaw=allow_wide_yaw,
            roi_crop_top_px=roi_crop_top_px,
            empty_ratio=empty_ratio,
            prop_reuse_prob=prop_reuse_prob,
            disc_interference_prob=disc_interference_prob,
            interference_strength=interference_strength,
            class_weights=class_weights,
        )

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
            return make_digit_card_bgra(cls_name, rng, handwritten=handwritten, mnist_bank=mnist_bank)
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

    # 2) 75% 概率放置双数字 Bonus 对
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


def save_preview_grid(
    dataset_dir: str | Path,
    out_path: str | Path,
    split: str = "train",
    rows: int = 3,
    cols: int = 3,
    class_names: Sequence[str] = DIGITS8_CLASSES,
    seed: int = 42,
) -> Path:
    """从生成的数据集中抽样绘制带 YOLO 真值绿框和类别标签的 NxN 预览拼图。"""
    ds = Path(dataset_dir)
    imgs = sorted((ds / "images" / split).glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"No images found in {ds / 'images' / split}")

    rng = random.Random(seed)
    n_cells = rows * cols
    # 优先挑选带标注的图以便直观检查框质量
    labeled = [p for p in imgs if (ds / "labels" / split / f"{p.stem}.txt").is_file()
               and (ds / "labels" / split / f"{p.stem}.txt").read_text(encoding="utf-8").strip()]
    pool = labeled if len(labeled) >= n_cells else imgs
    chosen = rng.sample(pool, min(n_cells, len(pool)))

    cells: List[np.ndarray] = []
    for p in chosen:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        lbl_p = ds / "labels" / split / f"{p.stem}.txt"
        if lbl_p.is_file():
            for ln in lbl_p.read_text(encoding="utf-8").splitlines():
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = int(round((cx - bw * 0.5) * W))
                y1 = int(round((cy - bh * 0.5) * H))
                x2 = int(round((cx + bw * 0.5) * W))
                y2 = int(round((cy + bh * 0.5) * H))
                cv2.rectangle(img, (x1, y1), (x2, y2), (40, 235, 60), 2)
                cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
                cv2.putText(img, cname, (max(2, x1), max(16, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 235, 60), 2, cv2.LINE_AA)
        cells.append(img)

    while len(cells) < n_cells:
        cells.append(np.zeros_like(cells[0]))

    grid_rows = [np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)]
    canvas = np.vstack(grid_rows)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", canvas)
    if ok:
        buf.tofile(str(out_p))
    return out_p


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
    num_classes: int = 11,
    mnist_dir: Optional[str] = None,
    allow_mnist_download: bool = True,
    max_digit_dist_m: Optional[float] = None,
    use_homography: bool = False,
    allow_wide_yaw: bool = False,
    roi_crop_top_px: int = 0,
    empty_ratio: float = 0.08,
    prop_reuse_prob: float = 0.70,
    disc_interference_prob: float = 0.60,
    interference_strength: str = "random",
    class_weights: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """生成完整 Ultralytics 8 类或 11 类 YOLO 数据集与 Horizon OE PTQ 校准列表。"""
    if abs(max_rot_deg) > MAX_ROTATION_DEG:
        raise ValueError(f"max_rot_deg={max_rot_deg} 超过允许上限 ±{MAX_ROTATION_DEG}°")
    if num_classes not in (8, 11):
        raise ValueError(f"num_classes 仅支持 8 或 11，收到: {num_classes}")
    eff_max_digit_dist_m = resolve_max_digit_dist_m(num_classes, max_digit_dist_m)

    active_classes = DIGITS8_CLASSES if num_classes == 8 else CLASSES
    root = Path(out_dir).resolve()
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    bg_paths = load_image_paths(bg_dir)
    prop_bank = load_prop_bank(props_dir)
    blank_disc_bank = load_blank_disc_bank(props_dir)
    mnist_bank = load_mnist_digits(mnist_dir, allow_download=allow_mnist_download) if mnist_dir else None
    mnist_total = sum(len(v) for v in mnist_bank.values()) if mnist_bank else 0
    rng = random.Random(seed)

    train_images: List[Path] = []
    val_images: List[Path] = []
    total_boxes = 0

    for split, count, bucket in (("train", num_train, train_images), ("val", num_val, val_images)):
        for i in range(count):
            img, lbls = generate_one_sample(
                rng,
                bg_paths,
                prop_bank,
                width=width,
                height=height,
                max_rot_deg=max_rot_deg,
                num_classes=num_classes,
                mnist_bank=mnist_bank,
                max_digit_dist_m=eff_max_digit_dist_m,
                blank_disc_bank=blank_disc_bank,
                use_homography=use_homography,
                allow_wide_yaw=allow_wide_yaw,
                roi_crop_top_px=roi_crop_top_px,
                empty_ratio=empty_ratio,
                prop_reuse_prob=prop_reuse_prob,
                disc_interference_prob=disc_interference_prob,
                interference_strength=interference_strength,
                class_weights=class_weights,
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

    names_yaml = "\n".join(f"  {i}: '{name}'" for i, name in enumerate(active_classes))
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(
        f"# AIC 2026 Finals {num_classes}-class YOLO dataset (max_digit_dist_m={eff_max_digit_dist_m:.2f}, roi_crop_top_px={roi_crop_top_px})\n"
        f"# CRITICAL training hyperparameters: fliplr=0.0, flipud=0.0, degrees={min(max_rot_deg, MAX_ROTATION_DEG):.1f}\n"
        f"path: {root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(active_classes)}\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )

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
        "num_classes": num_classes,
        "num_train": len(train_images),
        "num_val": len(val_images),
        "total_boxes": total_boxes,
        "mnist_samples": mnist_total,
        "blank_discs": len(blank_disc_bank),
        "dataset_yaml": str(dataset_yaml),
        "calib_list": str(calib_list),
        "calib_count": len(calib_sel),
        "allow_hflip": ALLOW_HFLIP,
        "allow_vflip": ALLOW_VFLIP,
        "max_rot_deg": min(abs(max_rot_deg), MAX_ROTATION_DEG),
        "max_digit_dist_m": eff_max_digit_dist_m,
        "roi_crop_top_px": roi_crop_top_px,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="AIC 2026 决赛 8 类数字直替 / 11 类全目标 YOLO 合成增强数据集生成器（严禁翻转）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out", required=True, help="输出数据集目录")
    ap.add_argument("--classes", type=int, choices=[8, 11], default=11, help="类别数：8=仅标注数字1..8，11=全目标")
    ap.add_argument("--num-train", type=int, default=400, help="训练集张数")
    ap.add_argument("--num-val", type=int, default=80, help="验证集张数")
    ap.add_argument("--bg-dir", default=None, help="真实赛场背景图目录（可选）")
    ap.add_argument("--props-dir", default=None, help="实拍道具抠图目录（按类名建子目录，可选）")
    ap.add_argument("--mnist-dir", default="dataset_work/mnist_cache", help="MNIST 真实手写数字缓存目录（--classes 8 自动加载）")
    ap.add_argument("--no-mnist", action="store_true", help="禁用 MNIST 加载，仅使用内置笔画/印刷字体")
    ap.add_argument("--preview-out", default=None, help="可选：保存 3x3 标注预览拼图路径")
    ap.add_argument("--seed", type=int, default=2026, help="随机种子")
    ap.add_argument("--max-rot-deg", type=float, default=15.0, help="最大旋转角（上限 15°）")
    ap.add_argument(
        "--max-digit-dist-m",
        type=float,
        default=None,
        help="数字贴纸最远投影距离（m）；未指定时 --classes 8 默认 1.15m，--classes 11 默认 1.55m",
    )
    ap.add_argument("--use-homography", action="store_true", help="启用标定地面单应矩阵投影")
    ap.add_argument("--allow-wide-yaw", action="store_true", help="允许地面圆形贴纸平面内宽角旋转（严禁镜像翻转）")
    ap.add_argument("--roi-crop-top", type=int, default=0, help="可选：顶部裁剪像素（如 160）并重映射为 640x640")
    ap.add_argument("--empty-ratio", type=float, default=0.08, help="空道无数字负样本比例")
    ap.add_argument("--prop-reuse-prob", type=float, default=0.70, help="复用 --props-dir 实拍贴纸的概率（防过拟合建议 0.15）")
    ap.add_argument("--disc-interference-prob", type=float, default=0.60, help="蓝圆盘底噪/纹理/彩斑干扰概率")
    ap.add_argument("--interference-strength", choices=["random", "heavy"], default="random", help="蓝圆盘干扰强度")
    ap.add_argument("--class-weights", default=None, help="可选 8 类采样权重（如 1.35,1,1,1,1,1,1.35,1）")
    ap.add_argument("--calib-count", type=int, default=100, help="INT8 PTQ 校准集图片数量")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    use_mnist_dir = None if args.no_mnist else (args.mnist_dir if args.classes == 8 or args.mnist_dir != "dataset_work/mnist_cache" else None)
    parsed_cw = None
    if args.class_weights:
        parsed_cw = [float(x.strip()) for x in args.class_weights.split(",") if x.strip()]
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
        num_classes=args.classes,
        mnist_dir=use_mnist_dir,
        max_digit_dist_m=args.max_digit_dist_m,
        use_homography=args.use_homography,
        allow_wide_yaw=args.allow_wide_yaw,
        roi_crop_top_px=args.roi_crop_top,
        empty_ratio=args.empty_ratio,
        prop_reuse_prob=args.prop_reuse_prob,
        disc_interference_prob=args.disc_interference_prob,
        interference_strength=args.interference_strength,
        class_weights=parsed_cw,
    )
    if args.preview_out:
        cnames = DIGITS8_CLASSES if args.classes == 8 else CLASSES
        p_out = save_preview_grid(args.out, args.preview_out, split="train", class_names=cnames, seed=args.seed)
        print(f"  preview_grid : {p_out}")
    print(
        f"[DONE] classes={info['num_classes']} train={info['num_train']} val={info['num_val']} "
        f"boxes={info['total_boxes']} mnist={info['mnist_samples']} calib={info['calib_count']} "
        f"(max_digit_dist={info['max_digit_dist_m']:.2f}m, hflip={info['allow_hflip']}, "
        f"vflip={info['allow_vflip']}, max_rot=±{info['max_rot_deg']}°)"
    )
    print(f"  dataset.yaml : {info['dataset_yaml']}")
    print(f"  calib_list   : {info['calib_list']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
