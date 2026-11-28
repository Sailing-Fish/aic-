#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 AIC 2026 决赛 8 类数字 v4 强泛化训练/验证/压测数据集与 X5 INT8 PTQ 校准集：
1. 生成 5500 张基于真实赛道底图与真实去字蓝盘纹理、叠加 2026 蓝盘底噪/纹理/彩斑与多笔迹的合成训练图（含 roi_crop_top=160 垂直 2× 拉伸）；
2. 并入零泄漏实拍训练集（2× 适度采样，避免过拟合单一书写人与单支白笔），并独立留出未见片段 193341_c1 作为实拍盲测集；
3. 使用完全独立种子与重度蓝盘干扰、零实拍抠图（prop_reuse_prob=0）生成 240 张“换人换笔+强底噪”压力验证集；
4. 按精确相同的 roi_crop_top=160 裁剪生成 160 张实车对齐的 INT8 PTQ 校准图像。
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "finals"))

import synth_finals_dataset as synth_mod  # noqa: E402


def _crop_copy_real_split(
    src_dir: Path,
    dst_root: Path,
    dst_split: str,
    prefix: str,
    repeats: int,
    roi_crop_top_px: int,
) -> Dict[str, int]:
    img_in = src_dir / "images"
    lbl_in = src_dir / "labels"
    img_out = dst_root / "images" / dst_split
    lbl_out = dst_root / "labels" / dst_split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    imgs = sorted([p for p in img_in.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    frames_written = 0
    boxes_written = 0

    for p in imgs:
        raw = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        lbl_p = lbl_in / f"{p.stem}.txt"
        raw_lines = lbl_p.read_text(encoding="utf-8").splitlines() if lbl_p.is_file() else []
        crop_img, remapped = synth_mod.apply_roi_crop_remap(
            raw, raw_lines, roi_crop_top_px=roi_crop_top_px, out_w=640, out_h=640
        )
        if raw_lines and not remapped:
            continue

        ok_enc, enc_buf = cv2.imencode(".jpg", crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok_enc:
            continue
        lbl_text = "\n".join(remapped) + ("\n" if remapped else "")

        for r_idx in range(repeats):
            rep_tag = f"_r{r_idx}" if repeats > 1 else ""
            stem = f"{prefix}{rep_tag}_{p.stem}"
            enc_buf.tofile(str(img_out / f"{stem}.jpg"))
            (lbl_out / f"{stem}.txt").write_text(lbl_text, encoding="utf-8")
            frames_written += 1
            boxes_written += len(remapped)
            for ln in remapped:
                cid = int(ln.split()[0])
                counts[synth_mod.DIGITS8_CLASSES[cid]] += 1

    print(f"[REAL->{dst_split}] frames={frames_written} (x{repeats}) boxes={boxes_written} cls={dict(sorted(counts.items()))}")
    return {"frames": frames_written, "boxes": boxes_written}


def _count_dataset_classes(ds_root: Path, split: str) -> Dict[str, int]:
    cnt: Counter[str] = Counter()
    lbl_dir = ds_root / "labels" / split
    if not lbl_dir.is_dir():
        return {}
    for lp in lbl_dir.glob("*.txt"):
        for ln in lp.read_text(encoding="utf-8").splitlines():
            parts = ln.strip().split()
            if len(parts) == 5:
                cid = int(parts[0])
                cnt[synth_mod.DIGITS8_CLASSES[cid]] += 1
    return dict(sorted(cnt.items()))


def _write_dataset_yaml(root: Path, train_rel: str, val_rel: str, note: str) -> Path:
    names_yaml = "\n".join(f"  {i}: '{name}'" for i, name in enumerate(synth_mod.DIGITS8_CLASSES))
    yp = root / "dataset.yaml"
    yp.write_text(
        f"# {note}\n"
        f"# CRITICAL training hyperparameters: fliplr=0.0, flipud=0.0, degrees=10.0\n"
        f"path: {root.as_posix()}\n"
        f"train: {train_rel}\n"
        f"val: {val_rel}\n"
        f"nc: 8\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )
    return yp


def build_cal_data(
    real_train_dir: Path,
    real_bg_dir: Path,
    synth_train_dir: Path,
    cal_out_dir: Path,
    roi_crop_top_px: int = 160,
    seed: int = 2026,
) -> int:
    if cal_out_dir.exists():
        shutil.rmtree(cal_out_dir)
    cal_out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = 0

    def _save_png(img_bgr: np.ndarray) -> None:
        nonlocal written
        out_p = cal_out_dir / f"cal_{written:03d}.png"
        ok, buf = cv2.imencode(".png", img_bgr)
        if ok:
            buf.tofile(str(out_p))
            written += 1

    # 1) 70 张真实行进数字帧（已按 roi_crop_top_px 裁剪成 640x640）
    real_imgs = sorted((real_train_dir / "images").glob("*.jpg"))
    if real_imgs:
        step = max(1, len(real_imgs) // 70)
        sel_real = real_imgs[::step][:70]
        for p in sel_real:
            im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                crop_im, _ = synth_mod.apply_roi_crop_remap(im, [], roi_crop_top_px=roi_crop_top_px)
                _save_png(crop_im)

    # 2) 30 张真实空白赛道底图（防空道误检校准漂移）
    bg_imgs = sorted([p for p in real_bg_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if bg_imgs:
        rng.shuffle(bg_imgs)
        for p in bg_imgs[:30]:
            im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                crop_im, _ = synth_mod.apply_roi_crop_remap(im, [], roi_crop_top_px=roi_crop_top_px)
                _save_png(crop_im)

    # 3) 60 张合成域增强图（已是 640x640 ROI，覆盖 1/7、蓝盘底噪与非蓝干扰卡片）
    synth_imgs = sorted((synth_train_dir / "images" / "train").glob("train_*.jpg"))
    rng.shuffle(synth_imgs)
    for p in synth_imgs:
        if written >= 160:
            break
        im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            _save_png(im)

    print(f"[CAL] generated {written} aligned 640x640 calibration PNGs -> {cal_out_dir}")
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="构建 AIC 2026 v4 强泛化 YOLO 训练/留出/压测数据集与 BPU 校准集")
    ap.add_argument("--out", default="dataset_work/digits8_v4_roi160")
    ap.add_argument("--stress-out", default="dataset_work/digits8_v4_stress_roi160")
    ap.add_argument("--cal-out", default="dataset_work/x5_convert_v4/cal_data")
    ap.add_argument("--real-dir", default="dataset_work/real_digits_v4")
    ap.add_argument("--bg-dir", default="dataset_work/real_bg_v4")
    ap.add_argument("--props-dir", default="dataset_work/real_props_v4")
    ap.add_argument("--mnist-dir", default="dataset_work/mnist_cache")
    ap.add_argument("--num-synth-train", type=int, default=5500)
    ap.add_argument("--num-synth-val", type=int, default=450)
    ap.add_argument("--num-stress", type=int, default=240)
    ap.add_argument("--real-oversample", type=int, default=2)
    ap.add_argument("--roi-crop-top", type=int, default=160)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    out_root = Path(args.out).resolve()
    stress_root = Path(args.stress_out).resolve()
    cal_root = Path(args.cal_out).resolve()
    real_root = Path(args.real_dir).resolve()
    bg_root = Path(args.bg_dir).resolve()
    props_root = Path(args.props_dir).resolve()

    if out_root.exists():
        shutil.rmtree(out_root)
    if stress_root.exists():
        shutil.rmtree(stress_root)

    # 1) 生成强泛化真实底图 + 真实蓝盘质感 + 2026 干扰噪声合成数据集
    print("=== [1/4] Generating v4 strong-generalization synthetic train/val ===")
    synth_info = synth_mod.generate_dataset(
        out_dir=out_root,
        num_train=args.num_synth_train,
        num_val=args.num_synth_val,
        bg_dir=str(bg_root),
        props_dir=str(props_root),
        seed=args.seed,
        max_rot_deg=15.0,
        calib_count=120,
        num_classes=8,
        mnist_dir=args.mnist_dir,
        max_digit_dist_m=1.15,
        use_homography=True,
        allow_wide_yaw=True,
        roi_crop_top_px=args.roi_crop_top,
        empty_ratio=0.05,
        prop_reuse_prob=0.15,
        disc_interference_prob=0.62,
        interference_strength="random",
        class_weights=[1.35, 1.0, 1.0, 1.0, 1.0, 1.0, 1.35, 1.0],
    )
    print(f"[SYNTH] {synth_info}")

    # 2) 并入真实视频标注（2× 适度权重）+ 构建独立实拍留出集 real_val_only
    print("=== [2/4] Merging ROI-cropped real train/val clips ===")
    _crop_copy_real_split(
        real_root / "train", out_root, "train", prefix="realtr", repeats=args.real_oversample, roi_crop_top_px=args.roi_crop_top
    )
    _crop_copy_real_split(
        real_root / "val", out_root, "val", prefix="realva", repeats=1, roi_crop_top_px=args.roi_crop_top
    )
    real_val_only = out_root / "real_val_only"
    _crop_copy_real_split(
        real_root / "val", real_val_only, "val", prefix="realva", repeats=1, roi_crop_top_px=args.roi_crop_top
    )
    _write_dataset_yaml(real_val_only, "images/val", "images/val", "Held-out unseen real clip (193341_c1) ROI160")

    # 3) 构建“换人换笔 + 重度蓝盘底噪 + 零实拍抠图”独立决赛压测集
    print("=== [3/4] Generating extreme unseen-writer / heavy-disc-noise stress test set ===")
    synth_mod.generate_dataset(
        out_dir=stress_root,
        num_train=0,
        num_val=args.num_stress,
        bg_dir=str(bg_root),
        props_dir=str(props_root),
        seed=9999,
        max_rot_deg=15.0,
        calib_count=0,
        num_classes=8,
        mnist_dir=args.mnist_dir,
        max_digit_dist_m=1.15,
        use_homography=True,
        allow_wide_yaw=True,
        roi_crop_top_px=args.roi_crop_top,
        empty_ratio=0.0,
        prop_reuse_prob=0.0,  # 严禁复用实拍数字抠图，纯检未见书写体泛化力！
        disc_interference_prob=1.0,
        interference_strength="heavy",
        class_weights=[1.0] * 8,
    )
    _write_dataset_yaml(stress_root, "images/val", "images/val", "Finals Stress Set: 0% real props, 100% heavy disc noise, ROI160")

    # 4) 生成对齐的 INT8 PTQ 校准集与预览拼图
    print("=== [4/4] Building aligned X5 INT8 cal_data and visual previews ===")
    build_cal_data(real_root / "train", bg_root, out_root, cal_root, roi_crop_top_px=args.roi_crop_top, seed=args.seed)

    synth_mod.save_preview_grid(out_root, out_root / "preview_train.png", split="train", rows=4, cols=4, seed=args.seed + 1)
    synth_mod.save_preview_grid(real_val_only, out_root / "preview_real_roi.png", split="val", rows=3, cols=3, seed=args.seed + 2)
    synth_mod.save_preview_grid(stress_root, stress_root / "preview_stress.png", split="val", rows=4, cols=4, seed=args.seed + 3)

    print("\n================ v4 数据集统计汇总 ================")
    print("  train class counts    :", _count_dataset_classes(out_root, "train"))
    print("  val class counts      :", _count_dataset_classes(out_root, "val"))
    print("  real_val_only counts  :", _count_dataset_classes(real_val_only, "val"))
    print("  stress val counts     :", _count_dataset_classes(stress_root, "val"))
    print(f"  main dataset.yaml     : {out_root / 'dataset.yaml'}")
    print(f"  real_val dataset.yaml : {real_val_only / 'dataset.yaml'}")
    print(f"  stress dataset.yaml   : {stress_root / 'dataset.yaml'}")
    print(f"  X5 cal_data dir       : {cal_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
