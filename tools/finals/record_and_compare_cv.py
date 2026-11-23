#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""赛场实拍录视频 + 离线 CV 三模式 A/B 对比工具。

功能：
  1) --record 模式：录制原始摄像头视频与帧时间戳，可选自动设置 50Hz 抗频闪
     (--power-line-50hz) 及批量对比不同曝光值 (--exposure 0 80 150)。
  2) --compare 模式：离线回放视频文件或图片目录，用 Python 复现 C++ ImageProcessor
     的三种车道分割模式：
       - legacy_hsv      : 旧版固定 HSV 白色过滤
       - relative_border : 零标定抗光照/抗老化相对特征（蓝垫抑制 + 局部黑边 + 帧内 Lab 白参考）
       - hybrid          : relative_border 与宽松 HSV 联合
     输出：
       - 每帧 CSV 指标（中心偏差、置信度、丢线状态）
       - 汇总 JSON + 终端对比表（丢线率 dropout_rate、中心标准差 std、帧间抖动 jitter）
       - 三联屏对比视频（直观检查眩光/老化/阴影下哪种模式最稳）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

MODES = ("legacy_hsv", "relative_border", "hybrid")


@dataclass
class CVParams:
    roi_top_ratio: float = 0.65
    roi_bottom_ratio: float = 0.20
    gaussian_kernel: int = 5
    morph_kernel: int = 3
    morph_iterations: int = 2
    # legacy HSV
    white_s_max: int = 45
    white_v_min: int = 80
    # relative_border
    blue_margin: int = 22
    blue_lab_b_max: int = 115
    black_block_size: int = 31
    black_c_offset: int = 14
    white_lab_chroma_max: float = 28.0
    white_min_l_offset: float = 35.0
    # relaxed HSV for hybrid
    hybrid_s_max: int = 85
    hybrid_v_min: int = 55
    # dual-band lane center
    near_top: float = 0.55
    near_bot: float = 0.85
    far_top: float = 0.25
    far_bot: float = 0.50
    col_white_ratio: float = 0.35
    min_run_width: int = 18


def crop_roi(bgr: np.ndarray, params: CVParams) -> Tuple[np.ndarray, int, int]:
    h = bgr.shape[0]
    y0 = int(max(0, min(h - 2, round(h * (1.0 - params.roi_top_ratio)))))
    y1 = int(max(y0 + 2, min(h, round(h * (1.0 - params.roi_bottom_ratio)))))
    return bgr[y0:y1], y0, y1


def _morph_clean(mask: np.ndarray, params: CVParams) -> np.ndarray:
    k = max(1, int(params.morph_kernel) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker, iterations=max(1, params.morph_iterations))
    return out


def segment_legacy_hsv(roi_bgr: np.ndarray, params: CVParams) -> np.ndarray:
    """复现旧版固定 HSV 白道分割。"""
    gk = max(1, int(params.gaussian_kernel) | 1)
    blur = cv2.GaussianBlur(roi_bgr, (gk, gk), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, params.white_v_min), (180, params.white_s_max, 255))
    return _morph_clean(mask, params)


def segment_relative_border(roi_bgr: np.ndarray, params: CVParams) -> np.ndarray:
    """零手动标定相对特征分割：蓝垫抑制 + 局部暗边检测 + 帧内 Lab 动态白参考。"""
    gk = max(1, int(params.gaussian_kernel) | 1)
    blur = cv2.GaussianBlur(roi_bgr, (gk, gk), 0)
    b_ch, g_ch, r_ch = cv2.split(blur)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)
    L_ch, a_ch, lab_b = cv2.split(lab)

    # 1) 亮度不变蓝垫抑制：B - max(R, G) > blue_margin 或 Lab b 偏蓝
    rg_max = np.maximum(r_ch, g_ch).astype(np.int16)
    blue_excess = b_ch.astype(np.int16) - rg_max
    blue_mask = (blue_excess > params.blue_margin) | (lab_b < params.blue_lab_b_max)

    # 2) 局部黑边/暗带检测（自适应阈值抑制光照渐变）
    bs = max(3, int(params.black_block_size) | 1)
    dark_local = cv2.adaptiveThreshold(
        L_ch, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, bs, params.black_c_offset
    ) > 0
    # 绝对极暗保底（避免宽黑带内部中心失效）
    dark_abs = L_ch < 42
    dark_mask = dark_local | dark_abs

    # 3) 帧内底部中央区域动态白参考（抗地图发黄/色温偏移）
    rh, rw = roi_bgr.shape[:2]
    y_ref0, y_ref1 = int(rh * 0.55), rh
    x_ref0, x_ref1 = int(rw * 0.25), int(rw * 0.75)
    valid_ref = (~blue_mask[y_ref0:y_ref1, x_ref0:x_ref1]) & (~dark_mask[y_ref0:y_ref1, x_ref0:x_ref1])

    if int(valid_ref.sum()) >= 80:
        ref_L = float(np.median(L_ch[y_ref0:y_ref1, x_ref0:x_ref1][valid_ref]))
        ref_a = float(np.median(a_ch[y_ref0:y_ref1, x_ref0:x_ref1][valid_ref]))
        ref_b = float(np.median(lab_b[y_ref0:y_ref1, x_ref0:x_ref1][valid_ref]))
        # 合理限幅，防止被眩光或彩色卡片污染参考点
        ref_L = float(np.clip(ref_L, 110.0, 245.0))
        ref_a = float(np.clip(ref_a, 112.0, 144.0))
        ref_b = float(np.clip(ref_b, 112.0, 152.0))
    else:
        ref_L, ref_a, ref_b = 185.0, 128.0, 128.0

    chroma_dist = np.hypot(
        a_ch.astype(np.float32) - ref_a,
        lab_b.astype(np.float32) - ref_b,
    )
    l_min = max(55.0, ref_L - params.white_min_l_offset)
    white_like = (chroma_dist <= params.white_lab_chroma_max) & (L_ch.astype(np.float32) >= l_min)
    drivable = white_like & (~blue_mask) & (~dark_mask)
    mask = (drivable.astype(np.uint8)) * 255
    return _morph_clean(mask, params)


def segment_hybrid(roi_bgr: np.ndarray, params: CVParams) -> np.ndarray:
    """结合 relative_border 与宽松 HSV 保底的混合模式。"""
    rel = segment_relative_border(roi_bgr, params)
    gk = max(1, int(params.gaussian_kernel) | 1)
    blur = cv2.GaussianBlur(roi_bgr, (gk, gk), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    relaxed_hsv = cv2.inRange(hsv, (0, 0, params.hybrid_v_min), (180, params.hybrid_s_max, 255))
    # 用宽松 HSV 过滤极端高饱和杂物，同时保留 relative_border 的抗阴影/老化能力
    comb = cv2.bitwise_and(rel, relaxed_hsv)
    return _morph_clean(comb, params)


def segment_by_mode(roi_bgr: np.ndarray, mode: str, params: CVParams) -> np.ndarray:
    if mode == "legacy_hsv":
        return segment_legacy_hsv(roi_bgr, params)
    if mode == "relative_border":
        return segment_relative_border(roi_bgr, params)
    if mode == "hybrid":
        return segment_hybrid(roi_bgr, params)
    raise ValueError(f"未知分割模式: {mode}")


def _extract_band_center(
    mask: np.ndarray, y0_frac: float, y1_frac: float, ref_x: float, params: CVParams
) -> Optional[Tuple[float, int]]:
    rh, rw = mask.shape[:2]
    r0 = int(max(0, min(rh - 1, round(rh * y0_frac))))
    r1 = int(max(r0 + 1, min(rh, round(rh * y1_frac))))
    band = mask[r0:r1, :] > 0
    col_ratio = band.mean(axis=0)
    active = col_ratio >= params.col_white_ratio

    # 找所有连续白色列段
    runs: List[Tuple[int, int]] = []
    start = None
    for x, val in enumerate(active):
        if val and start is None:
            start = x
        elif not val and start is not None:
            if x - start >= params.min_run_width:
                runs.append((start, x - 1))
            start = None
    if start is not None and rw - start >= params.min_run_width:
        runs.append((start, rw - 1))

    if not runs:
        return None
    best = min(runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - ref_x))
    return 0.5 * (best[0] + best[1]), (best[1] - best[0] + 1)


def extract_lane_center(mask: np.ndarray, params: CVParams) -> Tuple[bool, float, float]:
    """返回 (valid, center_px, norm_err in [-1, 1])。"""
    rw = mask.shape[1]
    mid = rw * 0.5
    near = _extract_band_center(mask, params.near_top, params.near_bot, mid, params)
    far_ref = near[0] if near is not None else mid
    far = _extract_band_center(mask, params.far_top, params.far_bot, far_ref, params)

    if near is not None and far is not None:
        c = 0.65 * near[0] + 0.35 * far[0]
        return True, c, (c - mid) / max(1.0, mid)
    if near is not None:
        return True, near[0], (near[0] - mid) / max(1.0, mid)
    if far is not None:
        return True, far[0], (far[0] - mid) / max(1.0, mid)
    return False, mid, 0.0


def compare_frames(
    frames: Sequence[np.ndarray],
    params: Optional[CVParams] = None,
    out_csv: Optional[str | Path] = None,
    out_video: Optional[str | Path] = None,
    fps: float = 20.0,
) -> Dict[str, Dict[str, float]]:
    """对帧序列运行 3 种 CV 模式并计算汇总指标，可选输出 CSV 和三联对比视频。"""
    params = params or CVParams()
    if not frames:
        raise ValueError("输入帧序列为空")

    per_mode_valid: Dict[str, List[bool]] = {m: [] for m in MODES}
    per_mode_err: Dict[str, List[float]] = {m: [] for m in MODES}
    rows_csv: List[Dict[str, object]] = []

    writer = None
    if out_video is not None:
        out_video = Path(out_video)
        out_video.parent.mkdir(parents=True, exist_ok=True)
        h0, w0 = frames[0].shape[:2]
        panel_w, panel_h = w0 // 2, h0 // 2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (panel_w * 3, panel_h))

    try:
        for idx, bgr in enumerate(frames):
            roi, y0, y1 = crop_roi(bgr, params)
            row: Dict[str, object] = {"frame": idx}
            panels = []

            for mode in MODES:
                mask = segment_by_mode(roi, mode, params)
                valid, cpx, err = extract_lane_center(mask, params)
                per_mode_valid[mode].append(valid)
                if valid:
                    per_mode_err[mode].append(err)
                row[f"{mode}_valid"] = int(valid)
                row[f"{mode}_err"] = round(err, 5) if valid else ""

                if writer is not None:
                    vis = bgr.copy()
                    overlay = vis[y0:y1].copy()
                    overlay[mask > 0] = (
                        overlay[mask > 0].astype(np.float32) * 0.55
                        + np.array([0, 220, 0], dtype=np.float32) * 0.45
                    ).astype(np.uint8)
                    vis[y0:y1] = overlay
                    cv2.rectangle(vis, (0, y0), (vis.shape[1] - 1, y1 - 1), (255, 180, 0), 1)
                    if valid:
                        cx_i = int(round(cpx))
                        cv2.line(vis, (cx_i, y0), (cx_i, y1), (0, 0, 255), 2)
                    status_txt = f"{mode}: err={err:+.3f}" if valid else f"{mode}: LOST"
                    col = (0, 255, 0) if valid else (0, 0, 255)
                    cv2.putText(vis, status_txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
                    panels.append(cv2.resize(vis, (bgr.shape[1] // 2, bgr.shape[0] // 2)))

            rows_csv.append(row)
            if writer is not None and len(panels) == 3:
                writer.write(np.hstack(panels))
    finally:
        if writer is not None:
            writer.release()

    summary: Dict[str, Dict[str, float]] = {}
    n_total = len(frames)
    for mode in MODES:
        valids = per_mode_valid[mode]
        errs = np.asarray(per_mode_err[mode], dtype=np.float64)
        n_ok = int(sum(valids))
        dropout_rate = 1.0 - (n_ok / max(1, n_total))
        if len(errs) >= 2:
            diffs = np.abs(np.diff(errs))
            err_std = float(np.std(errs))
            jitter_mean = float(np.mean(diffs))
            jitter_p95 = float(np.percentile(diffs, 95))
        elif len(errs) == 1:
            err_std = 0.0
            jitter_mean = 0.0
            jitter_p95 = 0.0
        else:
            err_std = float("nan")
            jitter_mean = float("nan")
            jitter_p95 = float("nan")
        summary[mode] = {
            "frames": float(n_total),
            "valid_frames": float(n_ok),
            "dropout_rate": round(dropout_rate, 4),
            "center_std": round(err_std, 5) if np.isfinite(err_std) else -1.0,
            "jitter_mean": round(jitter_mean, 5) if np.isfinite(jitter_mean) else -1.0,
            "jitter_p95": round(jitter_p95, 5) if np.isfinite(jitter_p95) else -1.0,
        }

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["frame"] + [f"{m}_{k}" for m in MODES for k in ("valid", "err")]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_csv)

    return summary


# ---------------------------------------------------------------- 视频读取与录制

def load_source_frames(source: str | Path, max_frames: int = 0) -> Tuple[List[np.ndarray], float]:
    """从视频文件或图片目录读取帧序列。"""
    p = Path(source)
    frames: List[np.ndarray] = []
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
        if max_frames > 0:
            paths = paths[:max_frames]
        for fp in paths:
            img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        return frames, 20.0

    if not p.is_file():
        raise FileNotFoundError(f"输入路径不存在: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {p}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    if fps <= 1.0 or fps > 120.0:
        fps = 20.0
    try:
        while True:
            ok, frm = cap.read()
            if not ok or frm is None:
                break
            frames.append(frm)
            if max_frames > 0 and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return frames, fps


def _apply_v4l2_ctrl(device: str, ctrl: str, val: int) -> bool:
    if os.name == "nt":
        return False
    try:
        r = subprocess.run(
            ["v4l2-ctl", "-d", device, f"--set-ctrl={ctrl}={val}"],
            capture_output=True, text=True, timeout=3.0,
        )
        return r.returncode == 0
    except Exception:
        return False


def record_clips(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_str = str(args.device)
    dev_open = int(dev_str) if dev_str.isdigit() else dev_str

    if args.power_line_50hz:
        ok = _apply_v4l2_ctrl(dev_str, "power_line_frequency", 1)
        print(f"[v4l2] power_line_frequency=1 (50Hz) -> {'OK' if ok else 'SKIP/UNSUPPORTED'}")

    exposures = args.exposure if args.exposure else [None]
    for exp in exposures:
        tag = f"exp{exp}" if exp is not None else "auto"
        if exp is not None:
            if exp == 0:
                _apply_v4l2_ctrl(dev_str, "auto_exposure", 3)
            else:
                _apply_v4l2_ctrl(dev_str, "auto_exposure", 1)
                _apply_v4l2_ctrl(dev_str, "exposure_time_absolute", int(exp))

        cap = cv2.VideoCapture(dev_open, cv2.CAP_V4L2 if os.name != "nt" else cv2.CAP_ANY)
        if not cap.isOpened():
            cap = cv2.VideoCapture(dev_open)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开摄像头: {dev_str}", file=sys.stderr)
            return 2

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        vid_path = out_dir / f"clip_{tag}.mp4"
        ts_path = out_dir / f"clip_{tag}_timestamps.csv"
        writer = cv2.VideoWriter(
            str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
        )
        print(f"[REC] 开始录制 {vid_path} ({args.duration:.1f}s)...")
        t0 = time.monotonic()
        n = 0
        try:
            with open(ts_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "t_rel_s", "epoch_s"])
                while time.monotonic() - t0 < args.duration:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    now = time.monotonic()
                    if frame.shape[1] != args.width or frame.shape[0] != args.height:
                        frame = cv2.resize(frame, (args.width, args.height))
                    writer.write(frame)
                    w.writerow([n, f"{now - t0:.4f}", f"{time.time():.4f}"])
                    n += 1
        finally:
            writer.release()
            cap.release()
        print(f"[REC] 完成 {vid_path}：{n} 帧 ({n / max(0.1, args.duration):.1f} fps)")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="AIC 2026 赛场录视频 & 离线 CV 三模式 A/B 对比工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true", help="录制摄像头视频片段")
    mode.add_argument("--compare", metavar="VIDEO_OR_DIR", help="离线回放视频或图片目录并对比三种 CV 模式")

    ap.add_argument("--device", default="/dev/video0", help="录制设备路径或序号")
    ap.add_argument("--duration", type=float, default=15.0, help="单段录制时长 s")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--power-line-50hz", action="store_true", help="尝试打开 V4L2 50Hz 抗频闪")
    ap.add_argument("--exposure", type=int, nargs="+", default=None,
                    help="录制多段对比曝光值（0=自动，>0=手动 exposure_time_absolute）")

    ap.add_argument("--out", default="tools/logs/cv_compare", help="输出目录")
    ap.add_argument("--max-frames", type=int, default=0, help="最多对比帧数（0=全部）")
    ap.add_argument("--no-video", action="store_true", help="不生成三联对比视频，只算 CSV/JSON 指标")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.record:
        return record_clips(args)

    frames, fps = load_source_frames(args.compare, max_frames=args.max_frames)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    vid_path = None if args.no_video else (out_dir / "compare_side_by_side.mp4")

    summary = compare_frames(frames, out_csv=csv_path, out_video=vid_path, fps=fps)
    sum_path = out_dir / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"{'模式':<18} {'有效帧':>10} {'丢线率':>9} {'中心std':>10} {'抖动均值':>10} {'抖动P95':>10}")
    print("-" * 76)
    for m in MODES:
        s = summary[m]
        print(
            f"{m:<18} {int(s['valid_frames']):4d}/{int(s['frames']):<5d} "
            f"{s['dropout_rate']*100:8.2f}% {s['center_std']:10.4f} "
            f"{s['jitter_mean']:10.4f} {s['jitter_p95']:10.4f}"
        )
    print("=" * 76)
    print(f"[OUT] CSV     : {csv_path}")
    print(f"[OUT] Summary : {sum_path}")
    if vid_path:
        print(f"[OUT] Video   : {vid_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
