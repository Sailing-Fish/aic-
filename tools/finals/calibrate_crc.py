#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AIC 2026 决赛 Tier 1-④-a：PC 端离线 CRC 置信度阈值与框底残差标定工具。

核心职责（纯 Python + NumPy + PyYAML，零 ROS2 / OpenCV 依赖）：
  1. 从真实赛场校验集加载预测框与真值框：
     - JSON 格式：
       [
         {
           "session_id": "run_hall_morning_01",
           "image_height": 480,
           "gt":    [{"cls": "red_no_entry", "box": [x1, y1, x2, y2]}],
           "preds": [{"cls": "red_no_entry", "conf": 0.62, "box": [x1, y1, x2, y2]}]
         },
         ...
       ]
     - YOLO txt 目录模式（--gt-dir + --pred-dir）：
       GT 每行: `cls_id cx cy w h`（归一化 [0,1]）
       Pred 每行: `cls_id cx cy w h conf`（Ultralytics save_txt + save_conf 格式）
  2. Sim-to-Real 纪律守卫：
     - 默认拒绝路径中含 "synth"（不区分大小写）的合成数据路径，
       严防用合成分布标定实车 CRC 参数；仅在单元测试显式传 `--allow-synthetic-for-test` 时放行。
  3. 严格按 `session_id`（视频场次）切分校准集 / 验证集：
     - 同一场次视频的所有帧要么全在校准集，要么全在验证集，彻底杜绝相邻帧时间相关性泄漏。
  4. 按类保召回置信度阈值标定（输出可直接粘贴到 `bpu_yolov8_digit_node` 的 `class_conf_overrides`）：
     - `red_no_entry` (+30s 禁行区)：目标召回 >= 0.98
     - `yellow_stop_line` (+15s 黄线)：目标召回 >= 0.98
     - `red_cone` (+5s 红锥)：目标召回 >= 0.95
     - Bonus 数字 `1..8`：在兼顾召回的同时守住精确率（防止误吃低分数字或误导岔路投票）
  5. 框底向上误差分位数外扩（CRC Conformal Quantile）：
     - 地面投影依赖框底像素行 `v_bot = y2`；若预测框底偏高（`v_bot_pred < v_bot_gt`），
       小车会低估障碍物/黄线的贴近程度。
     - 定义单侧非一致性残差：r_i = max(0, v_bot_gt - v_bot_pred)（统一换算至 480p 像素）。
     - 按有限样本保形分位数 k = ceil((n + 1) * (1 - alpha)) 求取：
       `crc_no_entry_pad_y_px`、`crc_cone_pad_y_px`、`crc_yellow_pad_y_px`，
       并在独立验证场次上报告实际覆盖率与召回/精确率。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

DIGITS8_CLASSES: Tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7", "8")
SAFETY_CLASSES: Tuple[str, ...] = ("red_cone", "red_no_entry", "yellow_stop_line")
ALL_CLASSES_11: Tuple[str, ...] = DIGITS8_CLASSES + SAFETY_CLASSES

# 默认按类目标召回率（安全类宁可多报不漏报；数字类兼顾精确率）
DEFAULT_TARGET_RECALL: Dict[str, float] = {
    "red_no_entry": 0.98,
    "yellow_stop_line": 0.98,
    "red_cone": 0.95,
    **{d: 0.85 for d in DIGITS8_CLASSES},
}

# 数字类最小精确率目标（与目标召回一起寻优）
DEFAULT_DIGIT_MIN_PRECISION: float = 0.85

# 框底 CRC 外扩像素参数映射（仅安全相关 3 类）
CRC_PAD_PARAM_MAP: Dict[str, str] = {
    "red_no_entry": "crc_no_entry_pad_y_px",
    "red_cone": "crc_cone_pad_y_px",
    "yellow_stop_line": "crc_yellow_pad_y_px",
}

DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480


# ---------------------------------------------------------------- Sim-to-Real 路径守卫

def is_synthetic_path(path: str | Path) -> bool:
    """检测路径字符串或任意父目录/文件名中是否包含 'synth'（不区分大小写）。"""
    p = Path(path)
    if "synth" in str(p).lower():
        return True
    return any("synth" in part.lower() for part in p.parts)


def assert_not_synthetic_path(path: str | Path, allow_synthetic: bool = False) -> None:
    """守住 Sim-to-Real 纪律：默认拒绝路径含 'synth' 的合成数据。"""
    if not allow_synthetic and is_synthetic_path(path):
        raise ValueError(
            f"Sim-to-Real 纪律拦截：检测到疑似合成数据路径 '{path}'（包含 'synth'）。"
            f"CRC 阈值与框底残差必须基于真实赛场视频校验集标定！"
            f"若仅用于自动化单测，请显式传入 --allow-synthetic-for-test。"
        )


# ---------------------------------------------------------------- 数据加载（JSON / YOLO txt）

def _normalize_box(box: Sequence[float]) -> List[float]:
    if len(box) != 4:
        raise ValueError(f"bbox 必须为 4 元素 [x1, y1, x2, y2]，收到: {box}")
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _normalize_record(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    session_id = str(raw.get("session_id", "")).strip()
    if not session_id:
        raise ValueError(
            f"第 {idx} 条记录缺少非空 'session_id' 字段；CRC 必须严格按视频场次切分以防数据泄漏"
        )
    img_h = float(raw.get("image_height", DEFAULT_IMAGE_HEIGHT))
    img_w = float(raw.get("image_width", DEFAULT_IMAGE_WIDTH))
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"第 {idx} 条记录 image_width/image_height 非法: ({img_w}, {img_h})")

    gts = []
    for g in raw.get("gt", []):
        gts.append({
            "cls": str(g["cls"]),
            "box": _normalize_box(g["box"]),
        })

    preds = []
    for p in raw.get("preds", []):
        preds.append({
            "cls": str(p["cls"]),
            "conf": float(p["conf"]),
            "box": _normalize_box(p["box"]),
        })

    return {
        "session_id": session_id,
        "frame_id": str(raw.get("frame_id", f"{session_id}_{idx:05d}")),
        "image_width": img_w,
        "image_height": img_h,
        "gt": gts,
        "preds": preds,
    }


def load_records_json(
    json_path: str | Path,
    allow_synthetic: bool = False,
) -> List[Dict[str, Any]]:
    """从 JSON 文件读取真实校验集预测与真值记录。"""
    p = Path(json_path)
    assert_not_synthetic_path(p, allow_synthetic=allow_synthetic)
    if not p.is_file():
        raise FileNotFoundError(f"校验集 JSON 文件不存在: {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "records" in raw:
            rows = raw["records"]
        elif "frames" in raw:
            rows = raw["frames"]
        else:
            raise ValueError("JSON 顶层字典必须包含 'records' 或 'frames' 数组")
    elif isinstance(raw, list):
        rows = raw
    else:
        raise ValueError("JSON 必须为帧记录数组或包含 'records'/'frames' 的字典")

    return [_normalize_record(item, i) for i, item in enumerate(rows)]


def _infer_session_id_from_path(
    rel_path: Path,
    session_regex: Optional[str] = None,
) -> str:
    """从 YOLO 标签相对路径提取 session_id（优先父子目录名，其次正则，再次下划线前缀）。"""
    if session_regex:
        m = re.search(session_regex, rel_path.as_posix())
        if m:
            return m.group(1) if m.groups() else m.group(0)
    if len(rel_path.parts) > 1:
        return rel_path.parts[0]
    stem = rel_path.stem
    m_def = re.match(r"^(.+?)(?:_frame\d+|_\d{3,}|-\d{3,})$", stem, flags=re.IGNORECASE)
    if m_def:
        return m_def.group(1)
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def _parse_yolo_txt(
    txt_path: Path,
    class_names: Sequence[str],
    img_w: float,
    img_h: float,
    is_pred: bool,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not txt_path.is_file():
        return items
    for ln in txt_path.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split()
        if not parts:
            continue
        if is_pred:
            if len(parts) < 6:
                raise ValueError(f"预测 YOLO 文件 {txt_path} 缺少置信度列（需 >=6 列）: {ln}")
            cid = int(parts[0])
            cx, cy, bw, bh, conf = map(float, parts[1:6])
        else:
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            conf = 1.0
        cls_name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        x1 = (cx - 0.5 * bw) * img_w
        y1 = (cy - 0.5 * bh) * img_h
        x2 = (cx + 0.5 * bw) * img_w
        y2 = (cy + 0.5 * bh) * img_h
        entry: Dict[str, Any] = {"cls": cls_name, "box": _normalize_box([x1, y1, x2, y2])}
        if is_pred:
            entry["conf"] = conf
        items.append(entry)
    return items


def load_records_yolo_dirs(
    gt_dir: str | Path,
    pred_dir: str | Path,
    class_names: Sequence[str] = ALL_CLASSES_11,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    session_regex: Optional[str] = None,
    allow_synthetic: bool = False,
) -> List[Dict[str, Any]]:
    """从成对 YOLO txt 目录（GT 与 Pred）加载校验记录。"""
    gt_root = Path(gt_dir)
    pred_root = Path(pred_dir)
    assert_not_synthetic_path(gt_root, allow_synthetic=allow_synthetic)
    assert_not_synthetic_path(pred_root, allow_synthetic=allow_synthetic)
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT YOLO 目录不存在: {gt_root}")
    if not pred_root.is_dir():
        raise FileNotFoundError(f"Pred YOLO 目录不存在: {pred_root}")

    gt_files = {p.relative_to(gt_root): p for p in sorted(gt_root.rglob("*.txt"))}
    pred_files = {p.relative_to(pred_root): p for p in sorted(pred_root.rglob("*.txt"))}
    all_rels = sorted(set(gt_files.keys()) | set(pred_files.keys()))

    records: List[Dict[str, Any]] = []
    for rel in all_rels:
        session_id = _infer_session_id_from_path(rel, session_regex=session_regex)
        gts = _parse_yolo_txt(gt_files[rel], class_names, image_width, image_height, is_pred=False) if rel in gt_files else []
        preds = _parse_yolo_txt(pred_files[rel], class_names, image_width, image_height, is_pred=True) if rel in pred_files else []
        records.append({
            "session_id": session_id,
            "frame_id": rel.with_suffix("").as_posix(),
            "image_width": float(image_width),
            "image_height": float(image_height),
            "gt": gts,
            "preds": preds,
        })
    return records


# ---------------------------------------------------------------- 严格按 session_id 切分

def split_by_session(
    records: Sequence[Dict[str, Any]],
    calib_ratio: float = 0.6,
    seed: int = 2026,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
    """严格按 session_id（视频场次）切分校准集与验证集，杜绝相邻帧数据泄漏。"""
    if not (0.0 < calib_ratio < 1.0):
        raise ValueError(f"calib_ratio 必须在 (0, 1) 内，收到: {calib_ratio}")
    if not records:
        raise ValueError("校验集为空，无法切分")

    sessions = sorted({str(r["session_id"]).strip() for r in records if str(r.get("session_id", "")).strip()})
    if len(sessions) < 2:
        raise ValueError(
            f"至少需要 2 个不同 session_id 才能进行无泄漏场次切分，当前只有 {len(sessions)} 个: {sessions}"
        )

    rng = random.Random(seed)
    shuffled = list(sessions)
    rng.shuffle(shuffled)

    n_calib = int(round(len(shuffled) * calib_ratio))
    n_calib = max(1, min(len(shuffled) - 1, n_calib))

    calib_sessions = sorted(shuffled[:n_calib])
    val_sessions = sorted(shuffled[n_calib:])
    calib_set = set(calib_sessions)
    val_set = set(val_sessions)
    assert calib_set.isdisjoint(val_set), "内部错误：校准集与验证集存在重叠 session_id"

    calib_records = [r for r in records if r["session_id"] in calib_set]
    val_records = [r for r in records if r["session_id"] in val_set]
    return calib_records, val_records, calib_sessions, val_sessions


# ---------------------------------------------------------------- 框匹配、阈值求解与 CRC 残差分位数

def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    area_b = max(0.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-12 else 0.0


def match_class_in_frame(
    gt_boxes: Sequence[Sequence[float]],
    preds: Sequence[Dict[str, Any]],
    iou_thresh: float = 0.45,
    conf_thresh: float = 0.0,
) -> Tuple[List[Tuple[int, Dict[str, Any], float]], List[int], List[Dict[str, Any]]]:
    """按置信度降序对单帧同类框做贪心 1-to-1 IoU 匹配。

    Returns:
        matches: list of (gt_idx, pred_dict, iou)
        unmatched_gt_indices: list of int
        false_positives: list of pred_dict
    """
    filtered = [p for p in preds if float(p["conf"]) >= conf_thresh]
    filtered.sort(key=lambda p: float(p["conf"]), reverse=True)

    matched_gt = set()
    matches: List[Tuple[int, Dict[str, Any], float]] = []
    fps: List[Dict[str, Any]] = []

    for pred in filtered:
        best_iou = 0.0
        best_idx = -1
        for gi, gbox in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            ov = box_iou(gbox, pred["box"])
            if ov >= iou_thresh and ov > best_iou:
                best_iou = ov
                best_idx = gi
        if best_idx >= 0:
            matched_gt.add(best_idx)
            matches.append((best_idx, pred, best_iou))
        else:
            fps.append(pred)

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return matches, unmatched_gt, fps


def evaluate_class_at_threshold(
    records: Sequence[Dict[str, Any]],
    cls_name: str,
    conf_thresh: float,
    iou_thresh: float = 0.45,
    target_image_height: float = DEFAULT_IMAGE_HEIGHT,
) -> Dict[str, Any]:
    """在指定置信度阈值下统计某类的 TP/FP/FN、召回率、精确率与框底向上误差列表。"""
    tp = 0
    fp = 0
    fn = 0
    total_gt = 0
    bottom_residuals: List[float] = []

    for rec in records:
        img_h = float(rec.get("image_height", target_image_height))
        v_scale = target_image_height / img_h if img_h > 0 else 1.0
        gts = [g["box"] for g in rec["gt"] if g["cls"] == cls_name]
        preds = [p for p in rec["preds"] if p["cls"] == cls_name]
        total_gt += len(gts)

        matches, unmatched, fps = match_class_in_frame(
            gts, preds, iou_thresh=iou_thresh, conf_thresh=conf_thresh
        )
        tp += len(matches)
        fp += len(fps)
        fn += len(unmatched)

        for gi, pred, _iou in matches:
            v_bot_gt = float(gts[gi][3]) * v_scale
            v_bot_pred = float(pred["box"][3]) * v_scale
            r_i = max(0.0, v_bot_gt - v_bot_pred)
            bottom_residuals.append(r_i)

    recall = tp / float(total_gt) if total_gt > 0 else 1.0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else (1.0 if total_gt == 0 else 0.0)
    return {
        "cls": cls_name,
        "conf_thresh": float(conf_thresh),
        "total_gt": total_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "bottom_residuals_px": bottom_residuals,
    }


def conformal_quantile(residuals: Sequence[float], alpha: float = 0.05) -> float:
    """有限样本保形预测 (Split Conformal / CRC) (1 - alpha) 上分位数。

    对升序残差 r_(1) <= ... <= r_(n)，取 k = min(n, ceil((n + 1) * (1 - alpha))) 处的 r_(k)，
    在可交换假设下保证新样本 P(r_new <= q_hat) >= 1 - alpha。
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha 必须在 (0, 1) 内，收到: {alpha}")
    n = len(residuals)
    if n == 0:
        return 0.0
    arr = np.sort(np.asarray(residuals, dtype=np.float64))
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(n, k))
    return float(arr[k - 1])


def solve_class_conf_threshold(
    calib_records: Sequence[Dict[str, Any]],
    cls_name: str,
    target_recall: float,
    iou_thresh: float = 0.45,
    min_conf: float = 0.05,
    max_conf: float = 0.95,
    digit_min_precision: float = DEFAULT_DIGIT_MIN_PRECISION,
    default_conf: float = 0.25,
) -> Dict[str, Any]:
    """为单类别求解置信度阈值。

    - 安全类 (`red_no_entry`, `yellow_stop_line`, `red_cone`):
      严格优先满足有限样本保召回目标（若满足目标召回，在所有可行阈值中取最大者以最大化精确率）。
    - Bonus 数字类 (`1..8`):
      兼顾精确率与召回率：优先在满足 `precision >= digit_min_precision` 且
      `recall >= target_recall` 的候选中取最大精确率/最高阈值；若无法同时满足，
      在所有候选中最大化 F_0.7 分数（微偏向精确率，避免误吃错号或误导岔路投票）。
    """
    # 收集该类所有 GT 数量及所有预测置信度作为候选扫描点
    all_confs: List[float] = []
    total_gt = 0
    for rec in calib_records:
        total_gt += sum(1 for g in rec["gt"] if g["cls"] == cls_name)
        for p in rec["preds"]:
            if p["cls"] == cls_name:
                all_confs.append(float(p["conf"]))

    if total_gt == 0:
        return {
            "cls": cls_name,
            "threshold": round(float(default_conf), 4),
            "target_recall": float(target_recall),
            "calib_recall": 1.0,
            "calib_precision": 1.0,
            "calib_gt": 0,
            "achievable": True,
            "note": "no_gt_in_calib",
        }

    # 候选阈值：每个预测置信度及其微调中点，限幅在 [min_conf, max_conf]
    uniq = sorted({round(max(min_conf, min(max_conf, c)), 6) for c in all_confs} | {min_conf})
    candidates: List[float] = []
    for i, c in enumerate(uniq):
        candidates.append(c)
        if i + 1 < len(uniq):
            mid = 0.5 * (c + uniq[i + 1])
            candidates.append(round(mid, 6))
    candidates = sorted(set(candidates))

    evals = [
        evaluate_class_at_threshold(calib_records, cls_name, t, iou_thresh=iou_thresh)
        for t in candidates
    ]

    is_digit = cls_name in DIGITS8_CLASSES
    if not is_digit:
        # 安全类：有限样本 CRC 保召回，要求 (fn + 1) / (n + 1) <= 1 - target_recall
        # 若样本数太少使得 1/(n+1) > 1 - target_recall，则退化为要求经验召回 >= target_recall（即 fn=0）
        alpha_miss = max(0.0, 1.0 - target_recall)
        crc_feasible = [
            e for e in evals
            if (e["fn"] + 1) / float(total_gt + 1) <= alpha_miss + 1e-12
        ]
        emp_feasible = [e for e in evals if e["recall"] >= target_recall - 1e-12]

        if crc_feasible:
            # 满足严格 CRC 风险上界，取最高置信度阈值以压低误报
            chosen = max(crc_feasible, key=lambda e: (e["precision"], e["conf_thresh"]))
            achievable = True
            note = "crc_recall_guaranteed"
        elif emp_feasible:
            # 样本量偏小时满足经验召回 >= target_recall，取最高阈值
            chosen = max(emp_feasible, key=lambda e: (e["precision"], e["conf_thresh"]))
            achievable = True
            note = "empirical_recall_guaranteed"
        else:
            # 即使最低阈值也无法达到目标召回（模型本身漏检），取最高召回里精确率最佳者
            best_rec = max(e["recall"] for e in evals)
            top_rec = [e for e in evals if abs(e["recall"] - best_rec) <= 1e-9]
            chosen = max(top_rec, key=lambda e: (e["precision"], e["conf_thresh"]))
            achievable = False
            note = f"recall_below_target(max={best_rec:.3f})"
    else:
        # 数字类 1..8：兼顾精确率与召回率
        both_ok = [
            e for e in evals
            if e["recall"] >= target_recall - 1e-12 and e["precision"] >= digit_min_precision - 1e-12
        ]
        if both_ok:
            # 在均达标的候选中，最大化 F_0.75，平手选更稳健的高阈值
            def _f_score(e: Dict[str, Any], beta: float = 0.75) -> float:
                p, r = e["precision"], e["recall"]
                denom = beta * beta * p + r
                return (1.0 + beta * beta) * p * r / denom if denom > 1e-12 else 0.0

            chosen = max(both_ok, key=lambda e: (_f_score(e), e["precision"], e["conf_thresh"]))
            achievable = True
            note = "digit_precision_recall_balanced"
        else:
            def _f_beta(e: Dict[str, Any], beta: float = 0.75) -> float:
                p, r = e["precision"], e["recall"]
                denom = beta * beta * p + r
                return (1.0 + beta * beta) * p * r / denom if denom > 1e-12 else 0.0

            chosen = max(evals, key=lambda e: (_f_beta(e), e["precision"], e["conf_thresh"]))
            achievable = (chosen["recall"] >= target_recall - 1e-12)
            note = "digit_f075_optimal"

    # 对阈值做极微小下移避免边界浮点抖动，仍留在 [min_conf, max_conf]
    thresh_out = round(max(min_conf, min(max_conf, float(chosen["conf_thresh"]))), 4)
    return {
        "cls": cls_name,
        "threshold": thresh_out,
        "target_recall": float(target_recall),
        "calib_recall": float(chosen["recall"]),
        "calib_precision": float(chosen["precision"]),
        "calib_gt": int(total_gt),
        "calib_tp": int(chosen["tp"]),
        "calib_fp": int(chosen["fp"]),
        "calib_fn": int(chosen["fn"]),
        "achievable": bool(achievable),
        "note": note,
    }


def calibrate_crc(
    records: Sequence[Dict[str, Any]],
    calib_ratio: float = 0.6,
    seed: int = 2026,
    alpha: float = 0.05,
    iou_thresh: float = 0.45,
    target_recalls: Optional[Dict[str, float]] = None,
    digit_min_precision: float = DEFAULT_DIGIT_MIN_PRECISION,
    min_conf: float = 0.05,
    max_conf: float = 0.95,
    default_conf: float = 0.25,
    max_pad_y_px: float = 40.0,
    target_image_height: float = DEFAULT_IMAGE_HEIGHT,
) -> Dict[str, Any]:
    """端到端运行按场次切分 -> 按类保召回阈值求解 -> 框底 CRC 分位数外扩 -> 验证集覆盖率评估。"""
    calib_recs, val_recs, calib_sessions, val_sessions = split_by_session(
        records, calib_ratio=calib_ratio, seed=seed
    )

    recalls_cfg = dict(DEFAULT_TARGET_RECALL)
    if target_recalls:
        recalls_cfg.update({str(k): float(v) for k, v in target_recalls.items()})

    # 发现数据中出现的所有类别（保持 11 类标准顺序，未知类追加在后）
    present_classes = set()
    for r in records:
        for g in r["gt"]:
            present_classes.add(g["cls"])
        for p in r["preds"]:
            present_classes.add(p["cls"])
    ordered_classes = [c for c in ALL_CLASSES_11 if c in present_classes] + sorted(
        present_classes - set(ALL_CLASSES_11)
    )

    class_thresholds: Dict[str, float] = {}
    per_class_report: Dict[str, Dict[str, Any]] = {}

    for cls_name in ordered_classes:
        t_rec = recalls_cfg.get(cls_name, 0.90)
        solved = solve_class_conf_threshold(
            calib_recs,
            cls_name,
            target_recall=t_rec,
            iou_thresh=iou_thresh,
            min_conf=min_conf,
            max_conf=max_conf,
            digit_min_precision=digit_min_precision,
            default_conf=default_conf,
        )
        thresh = solved["threshold"]
        class_thresholds[cls_name] = thresh

        calib_eval = evaluate_class_at_threshold(
            calib_recs, cls_name, thresh, iou_thresh=iou_thresh, target_image_height=target_image_height
        )
        val_eval = evaluate_class_at_threshold(
            val_recs, cls_name, thresh, iou_thresh=iou_thresh, target_image_height=target_image_height
        )

        per_class_report[cls_name] = {
            "threshold": thresh,
            "target_recall": t_rec,
            "achievable": solved["achievable"],
            "note": solved["note"],
            "calib": {
                "gt": calib_eval["total_gt"],
                "tp": calib_eval["tp"],
                "fp": calib_eval["fp"],
                "fn": calib_eval["fn"],
                "recall": calib_eval["recall"],
                "precision": calib_eval["precision"],
            },
            "val": {
                "gt": val_eval["total_gt"],
                "tp": val_eval["tp"],
                "fp": val_eval["fp"],
                "fn": val_eval["fn"],
                "recall": val_eval["recall"],
                "precision": val_eval["precision"],
            },
            "_calib_residuals": calib_eval["bottom_residuals_px"],
            "_val_residuals": val_eval["bottom_residuals_px"],
        }

    # 求解 3 个安全类的框底向上残差 CRC 外扩像素值，并在验证集上评估实际覆盖率
    crc_pads: Dict[str, float] = {}
    pad_report: Dict[str, Dict[str, Any]] = {}
    for cls_name, param_name in CRC_PAD_PARAM_MAP.items():
        if cls_name in per_class_report:
            cal_res = per_class_report[cls_name]["_calib_residuals"]
            val_res = per_class_report[cls_name]["_val_residuals"]
        else:
            cal_res = []
            val_res = []

        raw_q = conformal_quantile(cal_res, alpha=alpha)
        pad_px = round(min(max_pad_y_px, max(0.0, raw_q)), 2)
        crc_pads[param_name] = pad_px

        if val_res:
            val_arr = np.asarray(val_res, dtype=np.float64)
            val_cov = float(np.mean(val_arr <= pad_px + 1e-9))
            val_mean = float(np.mean(val_arr))
            val_p95 = float(np.percentile(val_arr, 95))
            val_max = float(np.max(val_arr))
        else:
            val_cov = None
            val_mean = 0.0
            val_p95 = 0.0
            val_max = 0.0

        pad_report[param_name] = {
            "cls": cls_name,
            "pad_y_px": pad_px,
            "raw_quantile_px": round(raw_q, 3),
            "alpha": float(alpha),
            "target_coverage": round(1.0 - float(alpha), 4),
            "calib_matched_count": len(cal_res),
            "val_matched_count": len(val_res),
            "val_coverage": val_cov,
            "val_residual_mean_px": round(val_mean, 3),
            "val_residual_p95_px": round(val_p95, 3),
            "val_residual_max_px": round(val_max, 3),
        }

    # 清理内部残差数组，保留干净可序列化报告
    for info in per_class_report.values():
        info.pop("_calib_residuals", None)
        info.pop("_val_residuals", None)

    overrides_json = json.dumps(class_thresholds, ensure_ascii=False, separators=(", ", ": "))

    return {
        "split": {
            "calib_sessions": calib_sessions,
            "val_sessions": val_sessions,
            "calib_frames": len(calib_recs),
            "val_frames": len(val_recs),
            "calib_ratio": float(calib_ratio),
            "seed": int(seed),
            "session_leakage": False,
        },
        "config": {
            "alpha": float(alpha),
            "iou_thresh": float(iou_thresh),
            "target_image_height": float(target_image_height),
        },
        "class_conf_overrides_dict": class_thresholds,
        "class_conf_overrides_json": overrides_json,
        "crc_pads": crc_pads,
        "per_class": per_class_report,
        "pad_validation": pad_report,
    }


def format_yaml_snippet(result: Dict[str, Any]) -> str:
    """生成可直接粘贴进 finals_fork.yaml / finals_roundabout.yaml 的 YAML 片段。"""
    pads = result["crc_pads"]
    overrides_json = result["class_conf_overrides_json"]
    # 用单引号包裹 JSON 字符串，确保 YAML 严格解析为 string（供 bpu_yolov8_digit_node.class_conf_overrides 使用）
    escaped_json = overrides_json.replace("'", "''")
    lines = [
        "# Tier 1-④-a 离线 CRC 标定结果（严格按视频场次切分，可直接粘贴进决赛 YAML）",
        "line_follower_node:",
        "  ros__parameters:",
        f"    crc_no_entry_pad_y_px: {pads['crc_no_entry_pad_y_px']:.2f}",
        f"    crc_cone_pad_y_px: {pads['crc_cone_pad_y_px']:.2f}",
        f"    crc_yellow_pad_y_px: {pads['crc_yellow_pad_y_px']:.2f}",
        "",
        "bpu_yolov8_digit_node:",
        "  ros__parameters:",
        f"    class_conf_overrides: '{escaped_json}'",
    ]
    return "\n".join(lines)


def print_report(result: Dict[str, Any]) -> None:
    sp = result["split"]
    print("=" * 78)
    print("AIC 2026 Tier 1-④-a 离线 CRC 置信度阈值与框底残差标定报告")
    print("=" * 78)
    print(
        f"场次切分 (无泄漏): calib={len(sp['calib_sessions'])} sessions ({sp['calib_frames']} frames) | "
        f"val={len(sp['val_sessions'])} sessions ({sp['val_frames']} frames)"
    )
    print(f"  calib_sessions: {sp['calib_sessions']}")
    print(f"  val_sessions  : {sp['val_sessions']}")
    print("-" * 78)
    print(
        f"{'Class':<18} {'Thresh':>7} {'TgtR':>6} | "
        f"{'CalR':>6} {'CalP':>6} {'CalGT':>5} | "
        f"{'ValR':>6} {'ValP':>6} {'ValGT':>5}  Note"
    )
    print("-" * 78)
    for cls_name, info in result["per_class"].items():
        c = info["calib"]
        v = info["val"]
        print(
            f"{cls_name:<18} {info['threshold']:7.4f} {info['target_recall']:6.2f} | "
            f"{c['recall']:6.3f} {c['precision']:6.3f} {c['gt']:5d} | "
            f"{v['recall']:6.3f} {v['precision']:6.3f} {v['gt']:5d}  {info['note']}"
        )
    print("-" * 78)
    print("框底向上残差 CRC 外扩（目标覆盖率 >= {:.1f}%）:".format((1.0 - result["config"]["alpha"]) * 100.0))
    for param_name, pinfo in result["pad_validation"].items():
        cov_str = f"{pinfo['val_coverage'] * 100.0:5.1f}%" if pinfo["val_coverage"] is not None else "  N/A "
        print(
            f"  {param_name:<24}: {pinfo['pad_y_px']:6.2f} px "
            f"(cal_n={pinfo['calib_matched_count']:3d}, val_n={pinfo['val_matched_count']:3d}, "
            f"val_cov={cov_str}, val_p95={pinfo['val_residual_p95_px']:.2f}px, val_max={pinfo['val_residual_max_px']:.2f}px)"
        )
    print("=" * 78)
    print("\n可直接复制到 finals_fork.yaml / finals_roundabout.yaml 的片段：\n")
    print(format_yaml_snippet(result))
    print()


def _load_class_names(classes_file: Optional[str], num_classes: int) -> Sequence[str]:
    if classes_file:
        lines = [ln.strip() for ln in Path(classes_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"类别列表文件为空: {classes_file}")
        return lines
    return DIGITS8_CLASSES if num_classes == 8 else ALL_CLASSES_11


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="AIC 2026 Tier 1-④-a 离线 CRC 置信度阈值与框底残差标定脚本（零 ROS2 依赖）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", default=None, help="真实校验集 JSON 文件路径")
    ap.add_argument("--gt-dir", default=None, help="可选 YOLO txt 模式：真值标签目录")
    ap.add_argument("--pred-dir", default=None, help="可选 YOLO txt 模式：预测标签目录（含第 6 列 conf）")
    ap.add_argument("--classes", type=int, choices=[8, 11], default=11, help="YOLO txt 模式的类别数")
    ap.add_argument("--classes-file", default=None, help="可选自定义类别列表文件（如 out_finals11.list）")
    ap.add_argument("--img-w", type=int, default=DEFAULT_IMAGE_WIDTH, help="YOLO txt 模式的图像宽度")
    ap.add_argument("--img-h", type=int, default=DEFAULT_IMAGE_HEIGHT, help="YOLO txt 模式及残差归算的目标图像高度")
    ap.add_argument("--session-regex", default=None, help="YOLO txt 模式下从相对路径提取 session_id 的正则")
    ap.add_argument("--calib-ratio", type=float, default=0.6, help="按 session_id 划分校准集的比例")
    ap.add_argument("--seed", type=int, default=2026, help="场次切分随机种子")
    ap.add_argument("--alpha", type=float, default=0.05, help="框底残差 CRC 显著性水平（目标覆盖率 = 1 - alpha）")
    ap.add_argument("--iou-thresh", type=float, default=0.45, help="真值/预测框 1-to-1 匹配 IoU 阈值")
    ap.add_argument("--digit-min-precision", type=float, default=DEFAULT_DIGIT_MIN_PRECISION, help="数字类 1..8 最低精确率目标")
    ap.add_argument("--max-pad-y-px", type=float, default=40.0, help="框底外扩像素硬上限")
    ap.add_argument("--out-yaml", default=None, help="可选：保存可直接粘贴的 YAML 片段路径")
    ap.add_argument("--out-json", default=None, help="可选：保存完整 JSON 标定报告路径")
    ap.add_argument(
        "--allow-synthetic-for-test",
        action="store_true",
        help="仅限自动化测试使用：跳过路径含 'synth' 的 Sim-to-Real 合成数据拦截",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.input:
        records = load_records_json(args.input, allow_synthetic=args.allow_synthetic_for_test)
    elif args.gt_dir and args.pred_dir:
        cls_names = _load_class_names(args.classes_file, args.classes)
        records = load_records_yolo_dirs(
            gt_dir=args.gt_dir,
            pred_dir=args.pred_dir,
            class_names=cls_names,
            image_width=args.img_w,
            image_height=args.img_h,
            session_regex=args.session_regex,
            allow_synthetic=args.allow_synthetic_for_test,
        )
    else:
        print(
            "[ERROR] 请指定 --input <val_records.json> 或同时指定 --gt-dir <dir> --pred-dir <dir>",
            file=sys.stderr,
        )
        return 2

    res = calibrate_crc(
        records=records,
        calib_ratio=args.calib_ratio,
        seed=args.seed,
        alpha=args.alpha,
        iou_thresh=args.iou_thresh,
        digit_min_precision=args.digit_min_precision,
        max_pad_y_px=args.max_pad_y_px,
        target_image_height=float(args.img_h),
    )
    print_report(res)

    if args.out_yaml:
        yp = Path(args.out_yaml)
        yp.parent.mkdir(parents=True, exist_ok=True)
        yp.write_text(format_yaml_snippet(res) + "\n", encoding="utf-8")
        print(f"[OUT] YAML 片段已保存: {yp}")

    if args.out_json:
        jp = Path(args.out_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OUT] 标定报告 JSON 已保存: {jp}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
