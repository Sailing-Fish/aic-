#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从面板录制的原始赛道视频自动去重抽帧，生成待标注 YOLO 关键帧集。

典型用法（在 PC 上 git pull 拉回 dataset_capture/raw_videos 后运行）：
    python tools/finals/extract_keyframes.py ^
        --input dataset_capture/raw_videos ^
        --output dataset_work/keyframes ^
        --interval-sec 0.4

核心防坑设计：
  1. 画面差异去重（--min-diff）：手推停顿/慢走时自动跳过几乎重复的静态帧，避免假高 mAP 与过拟合
  2. 模糊度过滤（--min-sharpness）：滤掉严重拖影帧，同时保留自然运动模糊
  3. 文件名保留完整视频前缀（日期+场景+光照），方便后续“按视频/光照整段划分 train/val”
  4. 含 hard_neg 的负样本视频自动生成对应空 .txt 标签（0 标注成本防赛场误检）
  5. 输出 manifest.json + 按光照汇总表，直接提示建议留哪几段视频作为纯实拍 Val 集
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def compute_sharpness(bgr: np.ndarray) -> float:
    """Laplacian 方差清晰度指标（值越大越清晰）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_frame_diff(prev_small: Optional[np.ndarray], cur_small: np.ndarray) -> float:
    """缩放灰度图平均绝对像素差（0..255），用于跳过静止/重复帧。"""
    if prev_small is None:
        return 255.0
    return float(np.mean(np.abs(cur_small.astype(np.float32) - prev_small.astype(np.float32))))


def _downsample_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)


def extract_from_video(
    video_path: Path,
    images_dir: Path,
    labels_dir: Optional[Path] = None,
    interval_sec: float = 0.4,
    min_diff: float = 6.0,
    min_sharpness: float = 35.0,
    max_frames_per_video: int = 120,
    jpg_quality: int = 95,
    auto_empty_for_hard_neg: bool = True,
) -> Dict[str, object]:
    """从单个视频按时间步长 + 清晰度 + 差异去重抽取关键帧。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    if fps <= 1.0 or fps > 120.0:
        fps = 15.0
    step_frames = max(1, int(round(fps * max(0.05, interval_sec))))

    stem = video_path.stem
    is_hard_neg = "hard_neg" in stem.lower()
    images_dir.mkdir(parents=True, exist_ok=True)
    if labels_dir is not None and is_hard_neg and auto_empty_for_hard_neg:
        labels_dir.mkdir(parents=True, exist_ok=True)

    total_read = 0
    saved = 0
    skipped_blur = 0
    skipped_dup = 0
    prev_small: Optional[np.ndarray] = None
    saved_files: List[str] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            idx = total_read
            total_read += 1
            if idx % step_frames != 0:
                continue

            sharp = compute_sharpness(frame)
            if sharp < min_sharpness:
                skipped_blur += 1
                continue

            cur_small = _downsample_gray(frame)
            diff = compute_frame_diff(prev_small, cur_small)
            if diff < min_diff:
                skipped_dup += 1
                continue

            out_name = f"{stem}_f{idx:05d}.jpg"
            out_img = images_dir / out_name
            cv2.imwrite(str(out_img), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            if labels_dir is not None and is_hard_neg and auto_empty_for_hard_neg:
                (labels_dir / f"{stem}_f{idx:05d}.txt").write_text("", encoding="utf-8")

            prev_small = cur_small
            saved += 1
            saved_files.append(out_name)
            if max_frames_per_video > 0 and saved >= max_frames_per_video:
                break
    finally:
        cap.release()

    return {
        "video": video_path.name,
        "stem": stem,
        "fps": round(fps, 2),
        "total_read": total_read,
        "saved": saved,
        "skipped_blur": skipped_blur,
        "skipped_dup": skipped_dup,
        "is_hard_neg": is_hard_neg,
        "files": saved_files,
    }


def extract_directory(
    input_dir: Path,
    output_dir: Path,
    interval_sec: float = 0.4,
    min_diff: float = 6.0,
    min_sharpness: float = 35.0,
    max_frames_per_video: int = 120,
    jpg_quality: int = 95,
) -> Dict[str, object]:
    """批量处理目录内全部视频，并生成 manifest.json 便于整段划分 train/val。"""
    if input_dir.is_file():
        videos = [input_dir]
    else:
        videos = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    per_video = []
    total_saved = 0
    for vp in videos:
        info = extract_from_video(
            vp,
            images_dir=images_dir,
            labels_dir=labels_dir,
            interval_sec=interval_sec,
            min_diff=min_diff,
            min_sharpness=min_sharpness,
            max_frames_per_video=max_frames_per_video,
            jpg_quality=jpg_quality,
        )
        per_video.append(info)
        total_saved += int(info["saved"])

    summary = {
        "input": str(input_dir),
        "output": str(output_dir),
        "video_count": len(videos),
        "total_saved": total_saved,
        "params": {
            "interval_sec": interval_sec,
            "min_diff": min_diff,
            "min_sharpness": min_sharpness,
            "max_frames_per_video": max_frames_per_video,
        },
        "videos": per_video,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="从面板录制的原始赛道视频去重抽取 YOLO 待标关键帧")
    ap.add_argument("--input", default="dataset_capture/raw_videos", help="原始视频目录或单个视频文件")
    ap.add_argument("--output", default="dataset_work/keyframes", help="输出关键帧目录")
    ap.add_argument("--interval-sec", type=float, default=0.4, help="抽帧最小时间间隔（秒），默认 0.4s (≈2.5fps)")
    ap.add_argument("--min-diff", type=float, default=6.0, help="与上一保留帧的最小灰度差异（越小保留越多）")
    ap.add_argument("--min-sharpness", type=float, default=35.0, help="Laplacian 清晰度下限（0=不过滤模糊）")
    ap.add_argument("--max-per-video", type=int, default=120, help="每段视频最多抽取帧数，0=不限")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f"[ERROR] 输入路径不存在: {inp}", file=sys.stderr)
        return 1

    res = extract_directory(
        input_dir=inp,
        output_dir=Path(args.output),
        interval_sec=args.interval_sec,
        min_diff=args.min_diff,
        min_sharpness=args.min_sharpness,
        max_frames_per_video=args.max_per_video,
    )
    print(f"处理视频 {res['video_count']} 段，共抽取关键帧 {res['total_saved']} 张 -> {Path(args.output) / 'images'}")
    for v in res["videos"]:
        tag = " [负样本已生成空标签]" if v["is_hard_neg"] else ""
        print(
            f"  - {v['video']}: 保留 {v['saved']} 帧 "
            f"(读 {v['total_read']} / 去重 {v['skipped_dup']} / 模糊 {v['skipped_blur']}){tag}"
        )
    print("提示：划分 train/val 时请按视频文件名前缀整段切分（例如把 dim_shadow 或 glare 视频整段留作 val）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
