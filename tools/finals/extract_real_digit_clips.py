#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从用户新录制的实车 `/image_raw` 原始视频中自动提取紧致贴合蓝圆盘的真实标注帧、
真实空白赛道路面底图、以及去透视真实数字贴纸抠图与真实空白蓝圆盘质感模板。

关键原则：
1. 严格按视频片段隔离 train / val：`193341_c1` 完整留出作零泄漏验证集，不参与训练与抠图提取。
2. 排除提车搬运切换镜头，仅在稳定贴地路段内按通道号/质心轨迹指派左右类别。
3. 使用 `finals_fork.yaml` 实车单应矩阵 `H_p2g` 校验物理投影宽度，并用白道环邻域筛选排除黑边外蓝区与环岛中心蓝岛。
4. 自动输出标注抽检拼图 `audit_train.jpg` / `audit_val.jpg` / `audit_props.jpg` 供人工/视觉复核。
"""
from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 与 tools/config/finals_fork.yaml 完全一致的地面单应矩阵 (u, v, 1) -> (x_fwd, y_left, w)
CALIBRATED_H_P2G = np.array(
    [
        [-1.3985236e-05, 0.00074556476, -0.41829529],
        [0.0007049316, -2.6281263e-05, -0.21497408],
        [-4.7978586e-06, -0.0063203042, 1.0],
    ],
    dtype=np.float64,
)

DIGITS8_CLASSES = ["1", "2", "3", "4", "5", "6", "7", "8"]
CLASS_TO_ID = {c: i for i, c in enumerate(DIGITS8_CLASSES)}

# 六段核对无误的实车地面行进数字片段：
# (clip_id, video_relpath, f_start, f_end, left_digit, right_digit, split)
CLIP_SPECS: Tuple[Tuple[str, str, int, int, str, str, str], ...] = (
    # 留出独立实拍验证集（零泄漏，不进训练集、不进真实抠图库）
    ("193341_c1", "dataset_capture/raw_videos/20261128_193341_digits_normal.mp4", 12, 94, "3", "8", "val"),
    # 实拍训练集片段
    ("193341_c2", "dataset_capture/raw_videos/20261128_193341_digits_normal.mp4", 148, 210, "4", "8", "train"),
    ("193439_c1", "dataset_capture/raw_videos/20261128_193439_digits_normal.mp4", 40, 102, "6", "8", "train"),
    ("193439_c2", "dataset_capture/raw_videos/20261128_193439_digits_normal.mp4", 286, 350, "2", "5", "train"),
    ("193439_c3", "dataset_capture/raw_videos/20261128_193439_digits_normal.mp4", 412, 506, "3", "8", "train"),
    ("193439_c4", "dataset_capture/raw_videos/20261128_193439_digits_normal.mp4", 570, 634, "6", "5", "train"),
)

# 干净无数字实拍路面时段，补充入 real_bg_v4 扩充真实底图多样度
CLEAN_BG_SPECS: Tuple[Tuple[str, str, int, int, int], ...] = (
    ("bg_193341_end", "dataset_capture/raw_videos/20261128_193341_digits_normal.mp4", 216, 290, 3),
    ("bg_193439_init", "dataset_capture/raw_videos/20261128_193439_digits_normal.mp4", 0, 28, 3),
    ("bg_193439_mid", "dataset_capture/raw_videos/20261128_193439_digits_normal.mp4", 224, 272, 3),
    ("bg_193439_end", "dataset_capture/raw_videos/20261128_193439_digits_normal.mp4", 638, 671, 3),
)

# 仅从训练集近距离帧提取顶视贴纸模板与去字空白蓝圆片
TEMPLATE_SOURCE_FRAMES: Dict[str, Tuple[str, List[int], str, str]] = {
    "193341_c2": ("dataset_capture/raw_videos/20261128_193341_digits_normal.mp4", [187, 193, 199], "4", "8"),
    "193439_c1": ("dataset_capture/raw_videos/20261128_193439_digits_normal.mp4", [78, 84, 88], "6", "8"),
    "193439_c2": ("dataset_capture/raw_videos/20261128_193439_digits_normal.mp4", [328, 336, 344], "2", "5"),
    "193439_c3": ("dataset_capture/raw_videos/20261128_193439_digits_normal.mp4", [464, 471, 476], "3", "8"),
    "193439_c4": ("dataset_capture/raw_videos/20261128_193439_digits_normal.mp4", [606, 613, 620], "6", "5"),
}


def pixel_to_ground(u: float, v: float, H_p2g: np.ndarray = CALIBRATED_H_P2G) -> Optional[Tuple[float, float]]:
    p = H_p2g @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-8:
        return None
    x_fwd = float(p[0] / p[2])
    y_left = float(p[1] / p[2])
    if x_fwd <= 0.02 or x_fwd > 3.0:
        return None
    return x_fwd, y_left


def ground_to_pixel(x_fwd: float, y_left: float, H_g2p: np.ndarray) -> Optional[Tuple[float, float]]:
    p = H_g2p @ np.array([float(x_fwd), float(y_left), 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-8:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def detect_blue_discs_in_frame(frame_bgr: np.ndarray, roi_top: int = 165) -> List[Dict[str, float]]:
    """
    从实拍赛道图白道内部紧致分割出蓝色数字圆贴纸。
    返回列表按 x1 从左到右排序，每项含 {x1, y1, x2, y2, cx, cy, d_m, y_m, gw_m, area}。
    """
    H, W = frame_bgr.shape[:2]
    roi = frame_bgr[roi_top:H, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)

    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)
    max_rg = np.maximum(r, g)

    # 蓝贴纸像素掩码（兼容宝蓝与轻微过曝/暗场偏灰蓝）
    blue_raw = (
        ((b - max_rg) >= 14) & (lab[:, :, 2] < 124) & (hsv[:, :, 2] > 32)
    ).astype(np.uint8) * 255

    # 白道底面像素掩码（用于验证贴纸外围是否处于白色赛道内部）
    white_raw = ((hsv[:, :, 1] < 86) & (hsv[:, :, 2] > 98)).astype(np.uint8)
    black_raw = (hsv[:, :, 2] < 65).astype(np.uint8)

    # 闭运算愈合圆盘内部白色数字笔画与局部反光斑块
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 9))
    blue_closed = cv2.morphologyEx(blue_raw, cv2.MORPH_CLOSE, k_close, iterations=2)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue_clean = cv2.morphologyEx(blue_closed, cv2.MORPH_OPEN, k_open, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_clean, connectivity=8)
    rh, rw = roi.shape[:2]
    candidates: List[Dict[str, float]] = []

    for lbl in range(1, num_labels):
        x, y, bw, bh, area = [int(v) for v in stats[lbl]]
        if area < 40 or bw < 10 or bh < 4:
            continue

        # 只有同时贴屏幕边缘且紧挨黑边界线（外侧蓝垫）才排除；贴边但处于白道内部的大贴纸予以保留
        touches_side = (x <= 2) or ((x + bw) >= rw - 2)
        if touches_side and bw > 190:
            continue
        # 画面远/中景贴到左右侧边缘的必然是弯道外侧蓝垫，贴纸仅在近景出画时才可能触碰左右边
        if touches_side and (y + roi_top) < 295:
            continue

        aspect = bw / float(max(1, bh))
        if not (0.95 <= aspect <= 7.0):
            continue

        # 用紧贴的蓝像素分量外接框（不做过度扩张）
        comp_mask = (labels[y:y + bh, x:x + bw] == lbl).astype(np.uint8)
        fill_ratio = float(comp_mask.sum()) / float(max(1, bw * bh))
        if fill_ratio < 0.44:
            continue

        # 外围邻域白道比例校验：排除被黑线包裹的环岛中心蓝岛与外侧蓝垫
        pad_x = max(6, int(bw * 0.16))
        pad_y = max(5, int(bh * 0.28))
        rx1, ry1 = max(0, x - pad_x), max(0, y - pad_y)
        rx2, ry2 = min(rw, x + bw + pad_x), min(rh, y + bh + pad_y)
        sub_lbl = (labels[ry1:ry2, rx1:rx2] == lbl).astype(np.uint8)
        dil = cv2.dilate(sub_lbl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 9)), iterations=1)
        ring = (dil > 0) & (sub_lbl == 0)
        # 检查左右与上下未贴画面边缘像素的白/黑比例
        side_ring = ring.copy()
        if y + bh >= rh - 3:
            side_ring[max(0, side_ring.shape[0] - 4):, :] = False
        if x <= 3:
            side_ring[:, :min(4, side_ring.shape[1])] = False
        if x + bw >= rw - 3:
            side_ring[:, max(0, side_ring.shape[1] - 4):] = False
        ring_cnt = int(side_ring.sum())
        if ring_cnt < 12:
            continue
        white_ring_frac = float(white_raw[ry1:ry2, rx1:rx2][side_ring].mean())
        black_ring_frac = float(black_raw[ry1:ry2, rx1:rx2][side_ring].mean())
        if white_ring_frac < 0.38 or black_ring_frac > 0.22:
            continue

        # 用单应矩阵校核地面物理尺寸与位置（真实圆盘直径 ~0.10–0.13 m）
        x1_full = float(x)
        y1_full = float(y + roi_top)
        x2_full = float(x + bw)
        y2_full = float(y + bh + roi_top)
        u_c = 0.5 * (x1_full + x2_full)
        v_b = min(float(H - 1), y2_full)

        gp_c = pixel_to_ground(u_c, v_b)
        gp_l = pixel_to_ground(x1_full, v_b)
        gp_r = pixel_to_ground(x2_full, v_b)
        if gp_c is None or gp_l is None or gp_r is None:
            continue
        d_m, y_m = gp_c
        gw_m = abs(gp_l[1] - gp_r[1])
        if not (0.04 <= d_m <= 1.25 and abs(y_m) <= 0.48):
            continue
        min_gw = 0.025 if touches_side else 0.050
        if not (min_gw <= gw_m <= 0.22):
            continue

        candidates.append(
            {
                "x1": x1_full,
                "y1": y1_full,
                "x2": x2_full,
                "y2": y2_full,
                "cx": u_c,
                "cy": 0.5 * (y1_full + y2_full),
                "d_m": d_m,
                "y_m": y_m,
                "gw_m": gw_m,
                "area": float(area),
            }
        )

    # 若检出超过 2 个，保留面积/白道贴合度最优且横向分布合理的前 2 个
    candidates.sort(key=lambda c: c["area"], reverse=True)
    if len(candidates) > 2:
        candidates = candidates[:2]
    candidates.sort(key=lambda c: c["cx"])
    return candidates


def _yolo_line(cls_id: int, box: Dict[str, float], W: int, H: int) -> str:
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    cx = max(0.0, min(1.0, 0.5 * (x1 + x2) / W))
    cy = max(0.0, min(1.0, 0.5 * (y1 + y2) / H))
    nw = max(1e-4, min(1.0, (x2 - x1) / W))
    nh = max(1e-4, min(1.0, (y2 - y1) / H))
    return f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def extract_clips(out_dir: Path, max_static_diff: float = 2.0) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val"):
        (out_dir / sp / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / sp / "labels").mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Dict[str, int]] = {}
    audit_buckets: Dict[str, List[Tuple[np.ndarray, List[Tuple[str, Dict[str, float]]]]]] = {
        "train": [],
        "val": [],
    }

    for clip_id, vid_rel, f_start, f_end, left_cls, right_cls, split in CLIP_SPECS:
        cap = cv2.VideoCapture(str(vid_rel))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {vid_rel}")

        prev_gray: Optional[np.ndarray] = None
        last_left_cx: Optional[float] = 220.0
        last_right_cx: Optional[float] = 420.0
        prev_pair_cxs: Optional[Tuple[float, float]] = None
        saved = 0
        box_cnt = 0
        single_cnt = 0

        for fi in range(f_start, f_end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            H, W = frame.shape[:2]
            cands = detect_blue_discs_in_frame(frame)
            if not cands:
                continue

            # 去重极度静止帧，同时保留接近与转弯中的全部动态视角
            gray_roi = cv2.cvtColor(cv2.resize(frame[165:, :], (160, 80)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                mad = float(np.mean(np.abs(gray_roi.astype(np.float32) - prev_gray.astype(np.float32))))
                if mad < max_static_diff and saved > 0:
                    continue
            prev_gray = gray_roi

            assigned: List[Tuple[str, Dict[str, float]]] = []
            if len(cands) == 2:
                b_l, b_r = cands[0], cands[1]
                # 确认两框有一定横向间距，防止同一框裂成两半误标
                if (b_r["cx"] - b_l["cx"]) < 45.0:
                    continue
                assigned = [(left_cls, b_l), (right_cls, b_r)]
                last_left_cx = b_l["cx"]
                last_right_cx = b_r["cx"]
                prev_pair_cxs = (b_l["cx"], b_r["cx"])
            else:
                b_only = cands[0]
                single_cnt += 1
                # 所有 6 段都是双数字并排场景：只有当圆盘已经推进至近景底边/左右边缘出画区域，才允许单框标注；
                # 远端画面中部的单框必定漏掉了可见兄弟贴纸，直接跳过，严防教会 YOLO 漏检！
                near_exit = (b_only["y2"] >= 395.0) or (b_only["x1"] <= 48.0) or (b_only["x2"] >= W - 48.0)
                if not near_exit:
                    continue
                if prev_pair_cxs is not None:
                    pl, pr = prev_pair_cxs
                    if 70.0 < pl < 570.0 and 70.0 < pr < 570.0 and b_only["y2"] < 425.0:
                        continue
                # 用上一帧追踪质心 + 画幅左右位置判断剩余单个贴纸归属
                dl = abs(b_only["cx"] - (last_left_cx if last_left_cx is not None else W * 0.30))
                dr = abs(b_only["cx"] - (last_right_cx if last_right_cx is not None else W * 0.70))
                if b_only["cx"] < W * 0.44 and dl <= dr + 40.0:
                    assigned = [(left_cls, b_only)]
                    last_left_cx = b_only["cx"]
                elif b_only["cx"] > W * 0.56 and dr <= dl + 40.0:
                    assigned = [(right_cls, b_only)]
                    last_right_cx = b_only["cx"]
                elif dl <= dr:
                    assigned = [(left_cls, b_only)]
                    last_left_cx = b_only["cx"]
                else:
                    assigned = [(right_cls, b_only)]
                    last_right_cx = b_only["cx"]

            stem = f"{clip_id}_f{fi:04d}"
            img_path = out_dir / split / "images" / f"{stem}.jpg"
            lbl_path = out_dir / split / "labels" / f"{stem}.txt"
            ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok_enc:
                buf.tofile(str(img_path))

            yolo_lines = [_yolo_line(CLASS_TO_ID[cls_name], bx, W, H) for cls_name, bx in assigned]
            lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            saved += 1
            box_cnt += len(assigned)
            audit_buckets[split].append((frame, assigned))

        cap.release()
        stats[clip_id] = {"split": split, "frames": saved, "boxes": box_cnt, "single_frames": single_cnt}
        print(f"[CLIP] {clip_id} ({split}): frames={saved}, boxes={box_cnt}, single_disc_frames={single_cnt}, pair=({left_cls},{right_cls})")

    # 生成标注抽检拼图
    for split in ("train", "val"):
        items = audit_buckets[split]
        if not items:
            continue
        step = max(1, len(items) // 24)
        sample_items = items[::step][:24]
        cells = []
        for idx, (fr, ass) in enumerate(sample_items):
            vis = fr[140:480, :].copy()
            for cls_name, bx in ass:
                p1 = (int(bx["x1"]), int(bx["y1"]) - 140)
                p2 = (int(bx["x2"]), int(bx["y2"]) - 140)
                cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"{cls_name} gw={bx['gw_m']*100:.1f}cm",
                    (p1[0], max(14, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2,
                )
            cells.append(cv2.resize(vis, (320, 170)))
        cols = 4
        rows = int(math.ceil(len(cells) / cols))
        while len(cells) < rows * cols:
            cells.append(np.zeros((170, 320, 3), dtype=np.uint8))
        grid = np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)])
        out_audit = out_dir / f"audit_{split}.jpg"
        cv2.imwrite(str(out_audit), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"[AUDIT] saved {out_audit} ({len(sample_items)}/{len(items)} frames)")

    return stats


def extract_clean_backgrounds(bg_out_dir: Path, old_bg_dir: Path) -> int:
    bg_out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    if old_bg_dir.is_dir():
        for p in sorted(old_bg_dir.glob("*.*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy2(p, bg_out_dir / f"v2_{p.name}")
                count += 1

    for tag, vid_rel, f_start, f_end, step in CLEAN_BG_SPECS:
        cap = cv2.VideoCapture(vid_rel)
        if not cap.isOpened():
            continue
        for fi in range(f_start, f_end + 1, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out_p = bg_out_dir / f"{tag}_f{fi:04d}.jpg"
            ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok_enc:
                buf.tofile(str(out_p))
                count += 1
        cap.release()
    print(f"[BG] extracted total {count} real track backgrounds -> {bg_out_dir}")
    return count


def extract_dewarped_templates(props_out_dir: Path) -> Dict[str, int]:
    """
    仅从训练集近距清晰帧按 CALIBRATED_H_P2G 做顶视去透视抠图：
    1. 每张真实圆片单独保存为 160x160 椭圆柔边 BGRA 模板到 <cls>/；
    2. 同时用 inpaint 抹掉白字，保存真实亮面蓝圆盘空白质感模板到 _blank_discs/。
    """
    props_out_dir.mkdir(parents=True, exist_ok=True)
    blank_dir = props_out_dir / "_blank_discs"
    blank_dir.mkdir(parents=True, exist_ok=True)

    saved_counts: Dict[str, int] = {}
    blank_saved = 0
    audit_cells: List[np.ndarray] = []

    for clip_id, (vid_rel, fidxs, left_cls, right_cls) in TEMPLATE_SOURCE_FRAMES.items():
        cap = cv2.VideoCapture(vid_rel)
        if not cap.isOpened():
            continue
        for fi in fidxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            cands = detect_blue_discs_in_frame(frame)
            if len(cands) != 2:
                continue

            for cls_name, bx in zip((left_cls, right_cls), cands):
                # 仅选贴地完整且不太小的近距贴纸
                if bx["d_m"] > 0.45 or bx["y2"] > 468 or bx["x1"] < 14 or bx["x2"] > 626:
                    continue
                # 用质心反投影得到贴纸真实几何中心（而非底边 v_b，避免顶部被裁）
                gp_center = pixel_to_ground(bx["cx"], bx["cy"])
                if gp_center is None:
                    continue
                x_c, y_c = gp_center
                half_m = max(0.088, bx["gw_m"] * 0.78)
                out_sz = 160
                dst_corners = np.float32([[0, 0], [out_sz, 0], [out_sz, out_sz], [0, out_sz]])
                H_g2p = np.linalg.inv(CALIBRATED_H_P2G)
                src_corners = []
                for dx_l, df_l in ((-half_m, +half_m), (+half_m, +half_m), (+half_m, -half_m), (-half_m, -half_m)):
                    pt = ground_to_pixel(x_c + df_l, y_c - dx_l, H_g2p)
                    if pt is None:
                        src_corners = []
                        break
                    src_corners.append([pt[0], pt[1]])
                if len(src_corners) != 4:
                    continue

                M = cv2.getPerspectiveTransform(np.float32(src_corners), dst_corners)
                patch = cv2.warpPerspective(frame, M, (out_sz, out_sz), flags=cv2.INTER_CUBIC)

                # 在去透视局部图上提取精准圆形贴纸掩码并做居中重采样
                pb = patch[:, :, 0].astype(np.int16)
                pg = patch[:, :, 1].astype(np.int16)
                pr = patch[:, :, 2].astype(np.int16)
                blue_m = ((pb - np.maximum(pr, pg)) >= 14).astype(np.uint8) * 255
                blue_m = cv2.morphologyEx(
                    blue_m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=2
                )
                ys_m, xs_m = np.nonzero(blue_m > 0)
                if len(xs_m) < 1200:
                    continue
                cx_m, cy_m = float(xs_m.mean()), float(ys_m.mean())
                rad_x = 0.5 * float(np.percentile(xs_m, 97) - np.percentile(xs_m, 3))
                rad_y = 0.5 * float(np.percentile(ys_m, 97) - np.percentile(ys_m, 3))
                rad = float(np.clip(0.5 * (rad_x + rad_y), 38.0, 68.0))
                # 平移缩放将圆盘规范对齐到中心 (80, 80)、半径 66px
                scale_c = 66.0 / rad
                M_align = np.float32([
                    [scale_c, 0.0, 80.0 - scale_c * cx_m],
                    [0.0, scale_c, 80.0 - scale_c * cy_m],
                ])
                patch = cv2.warpAffine(patch, M_align, (out_sz, out_sz), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                alpha = np.zeros((out_sz, out_sz), dtype=np.uint8)
                cv2.circle(alpha, (80, 80), 66, 255, -1, cv2.LINE_AA)
                bgra = np.dstack([patch, alpha])

                cls_dir = props_out_dir / cls_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                out_png = cls_dir / f"{clip_id}_f{fi:04d}_{cls_name}.png"
                ok_png, buf_png = cv2.imencode(".png", bgra)
                if ok_png:
                    buf_png.tofile(str(out_png))
                    saved_counts[cls_name] = saved_counts.get(cls_name, 0) + 1
                    vis_cell = patch.copy()
                    cv2.circle(vis_cell, (80, 80), 66, (0, 255, 0), 1)
                    cv2.putText(vis_cell, f"{cls_name}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    audit_cells.append(vis_cell)

                # 同时用 Telea inpaint 提取空白真实蓝圆盘质感（扩大膨胀消除白字抗锯齿暗边）
                hsv_p = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                stroke_mask = ((hsv_p[:, :, 1] < 110) & (hsv_p[:, :, 2] > 95) & (alpha > 128)).astype(np.uint8) * 255
                stroke_mask = cv2.dilate(stroke_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)
                if 80 < int(stroke_mask.sum() // 255) < 9500:
                    blank_bgr = cv2.inpaint(patch, stroke_mask, 9, cv2.INPAINT_TELEA)
                    blank_bgra = np.dstack([blank_bgr, alpha])
                    b_png = blank_dir / f"blank_{clip_id}_f{fi:04d}_{cls_name}.png"
                    ok_b, buf_b = cv2.imencode(".png", blank_bgra)
                    if ok_b:
                        buf_b.tofile(str(b_png))
                        blank_saved += 1
                        vis_b = blank_bgr.copy()
                        cv2.putText(vis_b, f"blk_{cls_name}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                        audit_cells.append(vis_b)
        cap.release()

    if audit_cells:
        cols = min(6, len(audit_cells))
        rows = int(math.ceil(len(audit_cells) / cols))
        cells = list(audit_cells[: rows * cols])
        while len(cells) < rows * cols:
            cells.append(np.zeros((160, 160, 3), dtype=np.uint8))
        grid = np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)])
        cv2.imwrite(str(props_out_dir / "audit_props.jpg"), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print(f"[PROPS] saved counts={saved_counts}, blank_blue_discs={blank_saved} -> {props_out_dir}")
    return saved_counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="提取并校验实车数字视频标注、路面底图与真实圆盘模板")
    ap.add_argument("--digits-out", default="dataset_work/real_digits_v4")
    ap.add_argument("--bg-out", default="dataset_work/real_bg_v4")
    ap.add_argument("--old-bg-dir", default="dataset_work/real_bg_v2/images")
    ap.add_argument("--props-out", default="dataset_work/real_props_v4")
    args = ap.parse_args(argv)

    extract_clips(Path(args.digits_out))
    extract_clean_backgrounds(Path(args.bg_out), Path(args.old_bg_dir))
    extract_dewarped_templates(Path(args.props_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
