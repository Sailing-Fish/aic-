#!/usr/bin/env python3
# coding=utf-8

from collections import deque
import hashlib
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    dnn = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, RegionOfInterest
    from ai_msgs.msg import PerceptionTargets, Target, Roi
except ImportError:
    rclpy = None
    Node = object
    HistoryPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    Image = None
    RegionOfInterest = None
    PerceptionTargets = None
    Target = None
    Roi = None

# 模型与类别文件默认放在脚本同级的 mod1 目录下。
# 用 __file__ 推导，避免把工作区绝对路径写死（工作区改名后仍可用）。
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mod1')
NMS_CLASS_OFFSET_PX = 4096.0


def bgr2nv12_fast(image):
    """
    优化的内存连续转换：BGR -> NV12
    """
    h, w = image.shape[:2]
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)

    y = yuv420p[:h, :]
    u = yuv420p[h:h + h // 4, :].reshape((h // 2, w // 2))
    v = yuv420p[h + h // 4:, :].reshape((h // 2, w // 2))

    uv = np.empty((h // 2, w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v

    return np.vstack((y, uv))


def letterbox(img, target_shape=(640, 640), color=(114, 114, 114)):
    """
    等比例缩放 + 填充。
    保持原始图像宽高比例，避免地面大倾角数字被强制拉伸。
    """
    h, w = img.shape[:2]
    tw, th = target_shape

    r = min(tw / w, th / h)
    nw = int(round(w * r))
    nh = int(round(h * r))

    pad_w = (tw - nw) / 2.0
    pad_h = (th - nh) / 2.0

    if (w, h) != (nw, nh):
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    padded_img = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return padded_img, r, (pad_w, pad_h)


def prepare_model_input(orig_frame, model_w, model_h, roi_crop_top_px=0):
    """
    准备 BPU 模型输入图像与逆映射参数：
    - roi_crop_top_px <= 0: 保持经典 letterbox 等比例填充。
    - roi_crop_top_px > 0 : 裁剪地平线以上无关顶部区域 [roi_top:orig_h, :] 后垂直拉伸到 (model_w, model_h)，
      对远距离地面贴纸实现 ~2x 垂直去压扁而不引入额外 BPU 耗时。
    返回: (input_frame, scale_x, scale_y, pad_w, pad_h, offset_y)
    """
    orig_h, orig_w = orig_frame.shape[:2]
    crop_top = int(roi_crop_top_px or 0)
    if crop_top <= 0:
        input_frame, r, (pad_w, pad_h) = letterbox(orig_frame, (model_w, model_h))
        return input_frame, float(r), float(r), float(pad_w), float(pad_h), 0.0

    roi_top = max(0, min(crop_top, orig_h - 32))
    crop = orig_frame[roi_top:orig_h, :]
    roi_h = max(1, orig_h - roi_top)
    if crop.shape[1] != model_w or crop.shape[0] != model_h:
        input_frame = cv2.resize(crop, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    else:
        input_frame = crop
    scale_x = float(model_w) / float(max(1, orig_w))
    scale_y = float(model_h) / float(roi_h)
    return input_frame, scale_x, scale_y, 0.0, 0.0, float(roi_top)


def load_class_list(cls_list_path):
    """
    从文本文件读取类别列表（每行一类，忽略空行）。
    """
    with open(cls_list_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def infer_model_num_classes(bpu_model):
    """
    从 BPU 模型输出张量形状中自动推断分类头通道数 C（排除 DFL 回归头 64 通道）。
    若模型对象不暴露 outputs 元数据（例如测试桩），返回 None。
    """
    outputs = getattr(bpu_model, 'outputs', None)
    if not outputs:
        return None

    cls_channels = []
    for out in outputs:
        props = getattr(out, 'properties', None)
        shape = getattr(props, 'shape', None)
        if shape is None or len(shape) != 4:
            continue
        dims = [int(d) for d in shape]
        # YOLOv8 BPU 输出一般为 NHWC (1, H, W, C) 或 NCHW (1, C, H, W)
        # 其中 H == W ∈ {80, 40, 20}，C 为 64 (reg) 或 num_classes (cls)
        if dims[1] == dims[2] and dims[3] != dims[1]:
            ch = dims[3]
        elif dims[2] == dims[3] and dims[1] != dims[2]:
            ch = dims[1]
        else:
            ch = dims[-1]
        if ch != 64:
            cls_channels.append(ch)

    if len(cls_channels) == 3 and len(set(cls_channels)) == 1:
        return int(cls_channels[0])
    return None


def validate_model_classes(bpu_model, classes):
    """
    动态校验类别文件数量与 BPU 模型输出头通道数是否一致。
    支持任意类别数（如旧 8 类数字模型、决赛 11 类模型等），不再硬编码 == 8。
    """
    if not classes:
        raise RuntimeError('类别列表为空，无法初始化 YOLOv8 检测节点')

    model_cls = infer_model_num_classes(bpu_model)
    if model_cls is not None and model_cls != len(classes):
        raise RuntimeError(
            f'模型类别数不匹配: BPU 模型输出 {model_cls} 类，'
            f'但类别文件包含 {len(classes)} 类 ({classes})'
        )
    return len(classes)


def parse_class_conf_overrides(overrides_spec, classes, default_conf, logger=None):
    """
    解析每类置信度阈值覆盖配置（JSON 字符串或 dict），返回长度为 len(classes) 的 float32 数组。
    支持用类别名（如 "red_no_entry"）或类别下标字符串（如 "9"）指定阈值。
    """
    base = float(np.clip(float(default_conf), 1e-4, 1.0 - 1e-4))
    thresholds = np.full(len(classes), base, dtype=np.float32)
    if overrides_spec is None:
        return thresholds

    parsed = None
    if isinstance(overrides_spec, dict):
        parsed = overrides_spec
    elif isinstance(overrides_spec, str):
        raw = overrides_spec.strip()
        if not raw:
            return thresholds
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                parsed = obj
            elif logger is not None:
                logger(f'class_conf_overrides 必须为 JSON 对象，已忽略: {raw}')
        except Exception as exc:
            if logger is not None:
                logger(f'解析 class_conf_overrides 失败，已回退默认阈值: {exc}')
            return thresholds

    if not parsed:
        return thresholds

    name_to_idx = {name: idx for idx, name in enumerate(classes)}
    for key, val in parsed.items():
        try:
            thr = float(np.clip(float(val), 1e-4, 1.0 - 1e-4))
        except (TypeError, ValueError):
            if logger is not None:
                logger(f'无效的类别阈值 {key}={val}，已忽略')
            continue

        key_str = str(key).strip()
        if key_str in name_to_idx:
            thresholds[name_to_idx[key_str]] = thr
        elif key_str.isdigit() and 0 <= int(key_str) < len(classes):
            thresholds[int(key_str)] = thr
        elif logger is not None:
            logger(f'class_conf_overrides 包含未知类别 "{key_str}"，已忽略')

    return thresholds


class SimpleBoxTracker:
    """
    轻量级跨帧 IoU + 质心距离目标跟踪器（纯 Python/NumPy）：
    - 仅在相同类别内做一对一贪心匹配，防止不同类别互抢 track_id；
    - 优先按 IoU >= iou_thresh 匹配，高速移动小框回退按质心像素距离匹配；
    - 赋予单调递增持久化整数 track_id，并在连续丢失 > max_missed_frames 帧后回收。
    """

    def __init__(self, iou_thresh=0.30, max_center_dist_px=80.0, max_missed_frames=5):
        self.iou_thresh = float(max(0.0, iou_thresh))
        self.max_center_dist_px = float(max(1.0, max_center_dist_px))
        self.max_missed_frames = int(max(0, max_missed_frames))
        self._next_track_id = 1
        self._tracks = {}

    @staticmethod
    def _box_iou(a, b):
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

    def reset(self):
        self._tracks.clear()
        self._next_track_id = 1

    def update(self, detections):
        """
        更新跟踪状态并原地/浅拷贝返回带持久化 track_id 与 track_hits 的检测列表。
        """
        out_dets = [dict(d) for d in detections]

        # 按类别分组匹配
        det_indices_by_cls = {}
        for idx, det in enumerate(out_dets):
            det_indices_by_cls.setdefault(det['type'], []).append(idx)

        matched_track_ids = set()
        matched_det_indices = set()

        for cls_name, det_indices in det_indices_by_cls.items():
            cls_tracks = [
                (tid, trk) for tid, trk in self._tracks.items() if trk['type'] == cls_name
            ]
            if not cls_tracks:
                continue

            # 构建候选匹配边 (score, tid, det_idx)
            # IoU 达标边得分 = 2.0 + iou（必高于纯质心回退边 [0, 1]）
            candidates = []
            for tid, trk in cls_tracks:
                t_box = (trk['bx'], trk['by'], trk['bw'], trk['bh'])
                tcx, tcy = trk['cx'], trk['cy']
                for det_idx in det_indices:
                    det = out_dets[det_idx]
                    d_box = (det['bx'], det['by'], det['bw'], det['bh'])
                    iou = self._box_iou(t_box, d_box)
                    if iou >= self.iou_thresh:
                        candidates.append((2.0 + iou, tid, det_idx))
                    else:
                        dist = math.hypot(det['cx'] - tcx, det['cy'] - tcy)
                        if dist <= self.max_center_dist_px:
                            score = 1.0 - (dist / self.max_center_dist_px)
                            candidates.append((score, tid, det_idx))

            candidates.sort(key=lambda item: item[0], reverse=True)
            for _score, tid, det_idx in candidates:
                if tid in matched_track_ids or det_idx in matched_det_indices:
                    continue
                matched_track_ids.add(tid)
                matched_det_indices.add(det_idx)

                det = out_dets[det_idx]
                trk = self._tracks[tid]
                trk['bx'] = det['bx']
                trk['by'] = det['by']
                trk['bw'] = det['bw']
                trk['bh'] = det['bh']
                trk['cx'] = det['cx']
                trk['cy'] = det['cy']
                trk['missed'] = 0
                trk['hits'] += 1

                det['track_id'] = int(tid)
                det['track_hits'] = int(trk['hits'])

        # 未匹配的检测生成新轨迹
        for idx, det in enumerate(out_dets):
            if idx in matched_det_indices:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = {
                'type': det['type'],
                'bx': det['bx'],
                'by': det['by'],
                'bw': det['bw'],
                'bh': det['bh'],
                'cx': det['cx'],
                'cy': det['cy'],
                'missed': 0,
                'hits': 1,
            }
            det['track_id'] = int(tid)
            det['track_hits'] = 1

        # 未匹配的旧轨迹累加 missed 并清理过期轨迹
        expired = []
        for tid, trk in self._tracks.items():
            if tid not in matched_track_ids:
                trk['missed'] += 1
                if trk['missed'] > self.max_missed_frames:
                    expired.append(tid)
        for tid in expired:
            del self._tracks[tid]

        return out_dets


class StageLatencyTracker:
    """
    统计各阶段耗时（ms）的滑动窗口 P50 / P95：
    - preprocess: 图像解码 + letterbox + BGR->NV12
    - infer:      BPU forward 推理
    - postprocess:反量化 + YOLOv8 预过滤与 DFL 解码
    - nms:        Per-class NMS + 跨帧跟踪 + ai_msgs/PerceptionTargets 组装
    - draw:       OpenCV 调试图绘制（跳过时为 0）
    - publish:    话题发布耗时（检测结果 + 可选调试图）
    - total:      单帧总处理耗时
    - e2e:        从图像 header.stamp 到检测消息发布完成的端到端延迟
    """
    STAGE_NAMES = (
        'preprocess',
        'infer',
        'postprocess',
        'nms',
        'draw',
        'publish',
        'total',
        'e2e',
    )

    def __init__(self, window_size=120):
        self.window_size = max(10, int(window_size))
        self.samples = {
            name: deque(maxlen=self.window_size) for name in self.STAGE_NAMES
        }

    def record(self, timings_ms):
        for name in self.STAGE_NAMES:
            val = timings_ms.get(name)
            if val is not None and val >= 0.0:
                self.samples[name].append(float(val))

    def percentiles(self, name):
        buf = self.samples.get(name)
        if not buf:
            return 0.0, 0.0
        arr = np.asarray(buf, dtype=np.float64)
        p50, p95 = np.percentile(arr, [50.0, 95.0])
        return float(p50), float(p95)

    def summary_dict(self):
        out = {}
        for name in self.STAGE_NAMES:
            out[name] = self.percentiles(name)
        return out

    def format_report(self, proc_fps, rx_fps, dropped_frames, debug_pub_frames, dropped_total=None):
        stats = self.summary_dict()
        parts = [
            f'proc_fps={proc_fps:.1f}',
            f'rx_fps={rx_fps:.1f}',
            f'dropped={dropped_frames}',
        ]
        if dropped_total is not None:
            parts.append(f'dropped_total={dropped_total}')
        parts.append(f'dbg_pub={debug_pub_frames}')
        for name in self.STAGE_NAMES:
            p50, p95 = stats[name]
            parts.append(f'{name}_p50={p50:.2f}ms {name}_p95={p95:.2f}ms')
        return '[DEC] yolo perf ' + ' '.join(parts)


class LatestFrameSlot:
    """
    单槽位最新帧缓存：
    - 订阅回调仅做 O(1) 写入并唤醒推理线程，不阻塞 ROS Executor；
    - 当推理速度慢于相机帧率时，自动覆盖未处理的旧帧并计数 dropped_frames，
      始终优先处理最新到达的一帧。
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest_msg = None
        self._latest_recv_ts = 0.0
        self.rx_count = 0
        self.dropped_count = 0

    def put(self, msg, recv_ts=None):
        if recv_ts is None:
            recv_ts = time.perf_counter()
        with self._lock:
            self.rx_count += 1
            if self._latest_msg is not None:
                self.dropped_count += 1
            self._latest_msg = msg
            self._latest_recv_ts = recv_ts
            self._event.set()

    def take(self, timeout=0.1):
        if not self._event.wait(timeout=timeout):
            return None, 0.0
        with self._lock:
            msg = self._latest_msg
            recv_ts = self._latest_recv_ts
            self._latest_msg = None
            self._event.clear()
            return msg, recv_ts

    def wake(self):
        self._event.set()


def decode_yolov8_outputs(
    outputs,
    grids,
    dfl_weights,
    num_classes,
    conf_thresh,
    pad_w,
    pad_h,
    scale_r,
    orig_w,
    orig_h,
    class_conf_thresholds=None,
    scale_x=None,
    scale_y=None,
    offset_y=0.0,
):
    """
    YOLOv8 BPU 输出反量化、Pre-sigmoid Logit 预过滤与 DFL 解码。
    - 先在 raw logit 空间用 min_logit = log(min_conf / (1 - min_conf)) 过滤掉绝大部分背景 anchor，
      仅对候选 anchor 计算 sigmoid 与 DFL，显著降低 RDK X5 CPU 后处理耗时。
    - 支持任意类别数与每类独立置信度阈值 (class_conf_thresholds)。
    返回: (all_boxes, all_scores, all_cls_ids, status_ok, cls_head_cnt, reg_head_cnt)
    """
    tensors = []
    for out in outputs:
        data = out.buffer.astype(np.float32)
        # 统一成 NHWC，类别/回归通道固定在最后一轴，便于按通道反量化
        if len(data.shape) == 4 and (data.shape[1] == num_classes or data.shape[1] == 64):
            if data.shape[-1] != num_classes and data.shape[-1] != 64:
                data = np.transpose(data, (0, 2, 3, 1))

        if hasattr(out.properties, 'scale_data') and len(out.properties.scale_data) > 0:
            scale = np.asarray(out.properties.scale_data, dtype=np.float32)
            if len(scale) == 1:
                data *= float(scale[0])
            elif data.ndim == 4 and len(scale) == data.shape[-1]:
                # BPU 可为每个类别/每个 DFL bin 单独给 scale，(C,) 必须广播到通道轴
                data *= scale.reshape(1, 1, 1, -1)
            else:
                data *= scale
        tensors.append(data)

    cls_heads = sorted(
        [t for t in tensors if t.shape[-1] == num_classes],
        key=lambda x: x.shape[1],
        reverse=True,
    )
    reg_heads = sorted(
        [t for t in tensors if t.shape[-1] == 64],
        key=lambda x: x.shape[1],
        reverse=True,
    )

    if len(cls_heads) != 3 or len(reg_heads) != 3:
        return [], [], [], False, len(cls_heads), len(reg_heads)

    if class_conf_thresholds is not None:
        cls_thresh_arr = np.asarray(class_conf_thresholds, dtype=np.float32).reshape(-1)
        if len(cls_thresh_arr) != num_classes:
            cls_thresh_arr = np.full(num_classes, float(conf_thresh), dtype=np.float32)
    else:
        cls_thresh_arr = np.full(num_classes, float(conf_thresh), dtype=np.float32)

    cls_thresh_arr = np.clip(cls_thresh_arr, 1e-6, 1.0 - 1e-6)
    min_conf = float(np.min(cls_thresh_arr))
    min_logit = math.log(min_conf / (1.0 - min_conf))

    all_boxes, all_scores, all_cls_ids = [], [], []

    for (grid, stride), cls_feat, reg_feat in zip(grids, cls_heads, reg_heads):
        cls_feat_flat = cls_feat.reshape(-1, num_classes)

        # Pre-sigmoid logit filter: skip exp() on >99% background anchors
        max_logits = np.max(cls_feat_flat, axis=1)
        cand_indices = np.flatnonzero(max_logits > min_logit)
        if len(cand_indices) == 0:
            continue

        cand_logits = np.clip(cls_feat_flat[cand_indices], -30.0, 30.0)
        cand_p = 1.0 / (1.0 + np.exp(-cand_logits))

        pass_mask = cand_p > cls_thresh_arr
        keep_local = np.any(pass_mask, axis=1)
        if not np.any(keep_local):
            continue

        valid_indices = cand_indices[keep_local]
        valid_p = cand_p[keep_local]
        valid_pass_mask = pass_mask[keep_local]

        masked_p = np.where(valid_pass_mask, valid_p, -1.0)
        valid_cls_ids = np.argmax(masked_p, axis=1)
        valid_scores = masked_p[np.arange(len(valid_cls_ids)), valid_cls_ids]
        valid_grid = grid[valid_indices]

        # DFL
        valid_reg = reg_feat.reshape(-1, 4, 16)[valid_indices]
        exp_reg = np.exp(valid_reg - np.max(valid_reg, axis=-1, keepdims=True))
        reg_p = exp_reg / np.sum(exp_reg, axis=-1, keepdims=True)
        dfl = np.dot(reg_p, dfl_weights)

        # 恢复到640×640 letterbox坐标
        x1 = (valid_grid[:, 0] - dfl[:, 0]) * stride
        y1 = (valid_grid[:, 1] - dfl[:, 1]) * stride
        x2 = (valid_grid[:, 0] + dfl[:, 2]) * stride
        y2 = (valid_grid[:, 1] + dfl[:, 3]) * stride

        # 映射回原始图像坐标（兼容等比例 letterbox 与 ROI 顶部裁剪垂直去压扁）
        sx = float(scale_r) if scale_x is None else float(scale_x)
        sy = float(scale_r) if scale_y is None else float(scale_y)
        oy = float(offset_y)
        x1 = np.clip((x1 - pad_w) / sx, 0, orig_w)
        y1 = np.clip((y1 - pad_h) / sy + oy, 0, orig_h)
        x2 = np.clip((x2 - pad_w) / sx, 0, orig_w)
        y2 = np.clip((y2 - pad_h) / sy + oy, 0, orig_h)

        all_boxes.append(np.stack([x1, y1, x2 - x1, y2 - y1], axis=1))
        all_scores.append(valid_scores)
        all_cls_ids.append(valid_cls_ids)

    return all_boxes, all_scores, all_cls_ids, True, 3, 3


def run_nms_filter(
    all_boxes,
    all_scores,
    all_cls_ids,
    classes,
    conf_thresh,
    iou_thresh,
    top_k,
    min_box_size,
    orig_w,
    orig_h,
    class_agnostic_nms=False,
    class_conf_thresholds=None,
):
    """
    执行 Per-class NMS（默认）或 class-agnostic NMS 并过滤过小检测框。
    Per-class NMS 通过按 cls_id * 4096 偏移坐标实现，确保相邻双数字（10~15cm）
    或靠近禁行区的路锥不会跨类别互相抑制。
    """
    detections = []
    if len(all_boxes) == 0:
        return detections

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    cls_ids = np.concatenate(all_cls_ids, axis=0)

    if class_agnostic_nms:
        nms_boxes = boxes
    else:
        offsets = cls_ids.astype(np.float32) * NMS_CLASS_OFFSET_PX
        nms_boxes = boxes.copy()
        nms_boxes[:, 0] += offsets
        nms_boxes[:, 1] += offsets

    nms_score_thresh = float(conf_thresh)
    if class_conf_thresholds is not None and len(class_conf_thresholds) > 0:
        nms_score_thresh = min(nms_score_thresh, float(np.min(class_conf_thresholds)))

    indices = cv2.dnn.NMSBoxes(
        nms_boxes.tolist(), scores.tolist(), nms_score_thresh, float(iou_thresh)
    )

    if len(indices) == 0:
        return detections

    indices = np.array(indices).flatten()
    if len(indices) > top_k:
        indices = indices[:top_k]

    for det_index, i in enumerate(indices):
        bx, by, bw, bh = boxes[i]

        if bw < min_box_size or bh < min_box_size:
            continue

        conf = float(scores[i])
        cid = int(cls_ids[i])
        raw_name = classes[cid] if cid < len(classes) else str(cid)

        cx, cy = float(bx + bw / 2.0), float(by + bh / 2.0)

        x_offset = int(max(0, round(bx)))
        y_offset = int(max(0, round(by)))
        box_width = int(max(1, round(bw)))
        box_height = int(max(1, round(bh)))

        if x_offset < orig_w:
            box_width = min(box_width, orig_w - x_offset)
        if y_offset < orig_h:
            box_height = min(box_height, orig_h - y_offset)

        detections.append({
            'track_id': int(det_index),
            'type': raw_name,
            'cls_id': cid,
            'confidence': conf,
            'bx': float(bx),
            'by': float(by),
            'bw': float(bw),
            'bh': float(bh),
            'cx': cx,
            'cy': cy,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'box_width': box_width,
            'box_height': box_height,
        })

    return detections


def _overlay_color_for_type(raw_name):
    if raw_name == 'red_no_entry':
        return (0, 0, 255)
    if raw_name == 'red_cone':
        return (0, 128, 255)
    if raw_name == 'yellow_stop_line':
        return (0, 255, 255)
    return (0, 255, 0)


def draw_debug_overlay(orig_frame, detections, current_fps):
    """
    在需要发布调试图时绘制检测框、持久化 track_id 与 FPS 信息。
    """
    for det in detections:
        bx, by, bw, bh = det['bx'], det['by'], det['bw'], det['bh']
        cx, cy = det['cx'], det['cy']
        raw_name = det['type']
        conf = det['confidence']
        tid = det.get('track_id', 0)
        color = _overlay_color_for_type(raw_name)

        draw_x1, draw_y1 = int(bx), int(by)
        draw_x2, draw_y2 = int(bx + bw), int(by + bh)

        cv2.rectangle(orig_frame, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)
        cv2.circle(orig_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        label = f'{raw_name}#{tid} {conf:.2f}'
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lbl_y = max(draw_y1, th + baseline + 2)

        cv2.rectangle(
            orig_frame,
            (draw_x1, lbl_y - th - baseline - 2),
            (draw_x1 + tw + 2, lbl_y),
            color,
            -1,
        )
        cv2.putText(
            orig_frame,
            label,
            (draw_x1 + 1, lbl_y - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    fps_text = f'FPS: {current_fps:.1f} | Dets: {len(detections)}'
    cv2.putText(
        orig_frame,
        fps_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


class BPUYoloV8DigitNode(Node):
    def __init__(self, bpu_model_override=None, start_worker=True):
        super().__init__('bpu_yolov8_digit_node')
        self.bridge = CvBridge() if CvBridge is not None else None

        # =====================================================
        # 1. YOLO 参数、Per-class NMS、跨帧跟踪与性能统计参数
        # =====================================================
        self.declare_parameter('conf_thresh', 0.25)
        self.declare_parameter('iou_thresh', 0.55)
        self.declare_parameter('nms_top_k', 20)
        self.declare_parameter('min_box_size', 6)
        self.declare_parameter('roi_crop_top_px', 0)
        self.declare_parameter('class_agnostic_nms', False)
        self.declare_parameter('class_conf_overrides', '{}')

        self.declare_parameter('enable_tracker', True)
        self.declare_parameter('tracker_iou_thresh', 0.30)
        self.declare_parameter('tracker_max_center_dist_px', 80.0)
        self.declare_parameter('tracker_max_missed_frames', 5)

        env_pub_dbg = os.environ.get('YOLO_PUBLISH_DEBUG_IMAGE', '1').strip().lower()
        default_pub_dbg = env_pub_dbg not in ('0', 'false', 'no', 'off')
        default_dbg_fps = float(os.environ.get('YOLO_DEBUG_IMAGE_MAX_FPS', '6.0'))
        default_stats_sec = float(os.environ.get('YOLO_STATS_LOG_INTERVAL_SEC', '5.0'))

        self.declare_parameter('publish_debug_image', default_pub_dbg)
        self.declare_parameter('debug_image_max_fps', default_dbg_fps)
        self.declare_parameter('async_latest_frame', True)
        self.declare_parameter('stats_window_size', 120)
        self.declare_parameter('stats_log_interval_sec', default_stats_sec)

        # =====================================================
        # 2. 类别文件（动态支持任意类别数：8 类旧模型 / 11 类决赛模型）
        # =====================================================
        self.declare_parameter('model_dir', DEFAULT_MODEL_DIR)
        self.declare_parameter('cls_list_path', '')
        self.declare_parameter('model_path', '')

        model_dir = self.get_parameter('model_dir').value or DEFAULT_MODEL_DIR
        cls_list_path = self.get_parameter('cls_list_path').value or os.path.join(
            model_dir, 'out.list'
        )
        model_path = self.get_parameter('model_path').value or os.path.join(
            model_dir, 'best_modified_yolov8.bin'
        )

        if os.path.exists(cls_list_path):
            self.classes = load_class_list(cls_list_path)
            self.get_logger().info(f'加载 {len(self.classes)} 类 ({cls_list_path}): {self.classes}')
        else:
            self.classes = ['1', '2', '3', '4', '5', '6', '7', '8']
            self.get_logger().warn(f'类别文件不存在: {cls_list_path}，使用预设8类数字标签')

        # =====================================================
        # 3. 加载 X5 BPU 模型并动态校验类别通道数
        # =====================================================
        if bpu_model_override is not None:
            self.bpu_model = bpu_model_override
        else:
            if not os.path.exists(model_path):
                raise RuntimeError(f'模型文件不存在: {model_path}')
            if dnn is None:
                raise RuntimeError('未检测到 hobot_dnn (pyeasy_dnn)，无法加载 BPU 模型')

            model_size = os.path.getsize(model_path)
            h = hashlib.sha256()
            with open(model_path, 'rb') as mf:
                for chunk in iter(lambda: mf.read(65536), b''):
                    h.update(chunk)
            self.get_logger().info(
                f'模型文件: {model_path} (size={model_size}B, sha256={h.hexdigest()[:12]})'
            )
            self.bpu_model = dnn.load(model_path)[0]

        validate_model_classes(self.bpu_model, self.classes)
        roi_crop_top_px = int(self.get_parameter('roi_crop_top_px').value)
        self.get_logger().info(
            f'RDK X5 BPU YOLOv8 视觉检测节点就绪 '
            f'(num_classes={len(self.classes)}, roi_crop_top_px={roi_crop_top_px})'
        )

        # =====================================================
        # 4. YOLOv8 模型参数 & 跟踪器
        # =====================================================
        self.model_w = 640
        self.model_h = 640
        self.strides = [8, 16, 32]
        self.grids = self._make_grid(self.model_w, self.model_h)
        self.dfl_weights = np.arange(16, dtype=np.float32)

        self.tracker = SimpleBoxTracker(
            iou_thresh=float(self.get_parameter('tracker_iou_thresh').value),
            max_center_dist_px=float(self.get_parameter('tracker_max_center_dist_px').value),
            max_missed_frames=int(self.get_parameter('tracker_max_missed_frames').value),
        )
        self._cached_overrides_raw = None
        self._cached_conf_thresh = None
        self._cached_cls_thresholds = None

        # =====================================================
        # 5. FPS 与各阶段耗时统计
        # =====================================================
        self.frame_count = 0
        self.debug_pub_count = 0
        self.start_time = time.perf_counter()
        self.current_fps = 0.0
        self._last_debug_pub_ts = 0.0
        self._last_stats_log_ts = time.perf_counter()
        self._last_stats_frame_count = 0
        self._last_stats_rx_count = 0
        self._last_stats_dropped_count = 0

        window_size = int(self.get_parameter('stats_window_size').value)
        self.latency_tracker = StageLatencyTracker(window_size=window_size)

        # =====================================================
        # 6. 最新帧单槽位缓冲与异步推理工作线程
        # =====================================================
        self.frame_slot = LatestFrameSlot()
        self._stop_worker = False
        self._worker_thread = None

        # =====================================================
        # 7. ROS2 接口（保持原有话题与消息类型不变）
        #    图像订阅使用 BEST_EFFORT + depth=1，避免高帧率大图在 FastDDS
        #    RELIABLE 模式下触发 2s HEARTBEAT/GAP 重传卡死。
        # =====================================================
        img_sub_qos = (
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            if QoSProfile is not None
            else 1
        )
        self.sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, img_sub_qos
        )

        self.det_pub = self.create_publisher(
            PerceptionTargets, '/hobot_dnn_detection', 1
        )

        self.img_pub = self.create_publisher(
            Image, '/yolo_detection_image', 2
        )

        pub_dbg = bool(self.get_parameter('publish_debug_image').value)
        dbg_fps = float(self.get_parameter('debug_image_max_fps').value)
        async_mode = bool(self.get_parameter('async_latest_frame').value)
        class_agnostic = bool(self.get_parameter('class_agnostic_nms').value)
        enable_tracker = bool(self.get_parameter('enable_tracker').value)

        self.get_logger().info('订阅图像: /image_raw')
        self.get_logger().info('发布检测结果: /hobot_dnn_detection [ai_msgs/msg/PerceptionTargets]')
        self.get_logger().info(
            f'发布检测图像: /yolo_detection_image (enabled={pub_dbg}, max_fps={dbg_fps:.1f}, '
            f'async_latest={async_mode}, class_agnostic_nms={class_agnostic}, tracker={enable_tracker})'
        )

        if start_worker and async_mode:
            self._worker_thread = threading.Thread(
                target=self._worker_loop, name='bpu_yolov8_worker', daemon=True
            )
            self._worker_thread.start()

    def _get_class_conf_thresholds(self, conf_thresh):
        raw = str(self.get_parameter('class_conf_overrides').value or '{}')
        if raw != self._cached_overrides_raw or conf_thresh != self._cached_conf_thresh:
            self._cached_cls_thresholds = parse_class_conf_overrides(
                raw,
                self.classes,
                conf_thresh,
                logger=lambda m: self.get_logger().warn(m),
            )
            self._cached_overrides_raw = raw
            self._cached_conf_thresh = conf_thresh
        return self._cached_cls_thresholds

    def _make_grid(self, w, h):
        grids = []
        for s in self.strides:
            gw = w // s
            gh = h // s
            xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
            grid = np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32) + 0.5
            grids.append((grid, s))
        return grids

    def _should_publish_debug_image(self, now_ts):
        publish_debug = bool(self.get_parameter('publish_debug_image').value)
        if not publish_debug:
            return False
        max_fps = float(self.get_parameter('debug_image_max_fps').value)
        if max_fps <= 0.0:
            return False
        min_interval = 1.0 / max_fps
        if (now_ts - self._last_debug_pub_ts) < min_interval:
            return False
        return True

    def _compute_e2e_latency_ms(self, msg):
        """
        根据图像 header.stamp 与当前 ROS 时钟计算端到端延迟（ms）。
        若 stamp 未设置（全 0），返回 None。
        """
        stamp = getattr(msg.header, 'stamp', None)
        if stamp is None:
            return None
        sec = int(getattr(stamp, 'sec', 0))
        nanosec = int(getattr(stamp, 'nanosec', 0))
        if sec == 0 and nanosec == 0:
            return None
        try:
            now_ns = self.get_clock().now().nanoseconds
            msg_ns = sec * 1_000_000_000 + nanosec
            diff_ms = (now_ns - msg_ns) / 1e6
            if 0.0 <= diff_ms <= 60_000.0:
                return float(diff_ms)
        except Exception:
            pass
        return None

    def image_callback(self, msg):
        """
        ROS 订阅回调：
        - 默认异步最新帧模式下，仅将最新消息放入单槽位缓存并立即返回，避免阻塞 Executor；
        - 若未处理的旧帧仍在槽位中，自动覆盖并计入 dropped_count。
        """
        recv_ts = time.perf_counter()
        if bool(self.get_parameter('async_latest_frame').value) and self._worker_thread is not None:
            self.frame_slot.put(msg, recv_ts)
        else:
            self.frame_slot.rx_count += 1
            self._process_frame(msg, recv_ts)

    def _worker_loop(self):
        while not self._stop_worker:
            msg, recv_ts = self.frame_slot.take(timeout=0.1)
            if msg is None or self._stop_worker:
                continue
            try:
                self._process_frame(msg, recv_ts)
            except Exception as e:
                self.get_logger().error(f'YOLO工作线程处理异常: {e}')

    def _process_frame(self, msg, recv_ts=None):
        t_frame_start = time.perf_counter()

        conf_thresh = float(self.get_parameter('conf_thresh').value)
        iou_thresh = float(self.get_parameter('iou_thresh').value)
        top_k = int(self.get_parameter('nms_top_k').value)
        min_box_size = float(self.get_parameter('min_box_size').value)
        class_agnostic_nms = bool(self.get_parameter('class_agnostic_nms').value)
        enable_tracker = bool(self.get_parameter('enable_tracker').value)
        cls_thresholds = self._get_class_conf_thresholds(conf_thresh)

        # =====================================================
        # 1 & 2. 预处理：ROS Image -> OpenCV -> Letterbox -> NV12
        # =====================================================
        t0 = time.perf_counter()
        try:
            orig_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
            return

        orig_h, orig_w = orig_frame.shape[:2]
        roi_crop_top_px = int(self.get_parameter('roi_crop_top_px').value)
        input_frame, scale_x, scale_y, pad_w, pad_h, offset_y = prepare_model_input(
            orig_frame, self.model_w, self.model_h, roi_crop_top_px=roi_crop_top_px
        )
        nv12 = bgr2nv12_fast(input_frame)
        t_preprocess_end = time.perf_counter()

        # =====================================================
        # 3. X5 BPU 推理
        # =====================================================
        try:
            outputs = self.bpu_model.forward(nv12)
        except Exception as e:
            self.get_logger().error(f'BPU推理失败: {e}')
            return
        t_infer_end = time.perf_counter()

        # =====================================================
        # 4. 后处理：反量化 + Pre-sigmoid 预过滤 + YOLOv8 DFL 解码
        # =====================================================
        num_classes = len(self.classes)
        all_boxes, all_scores, all_cls_ids, ok, cls_cnt, reg_cnt = decode_yolov8_outputs(
            outputs=outputs,
            grids=self.grids,
            dfl_weights=self.dfl_weights,
            num_classes=num_classes,
            conf_thresh=conf_thresh,
            pad_w=pad_w,
            pad_h=pad_h,
            scale_r=scale_x,
            orig_w=orig_w,
            orig_h=orig_h,
            class_conf_thresholds=cls_thresholds,
            scale_x=scale_x,
            scale_y=scale_y,
            offset_y=offset_y,
        )
        t_postprocess_end = time.perf_counter()

        if not ok:
            self.get_logger().warn(
                f'YOLOv8输出头数量异常: cls={cls_cnt}, reg={reg_cnt} (expected num_classes={num_classes})'
            )
            return

        # =====================================================
        # 5. Per-class NMS + 跨帧跟踪 + ai_msgs/PerceptionTargets 封装
        # =====================================================
        detections = run_nms_filter(
            all_boxes=all_boxes,
            all_scores=all_scores,
            all_cls_ids=all_cls_ids,
            classes=self.classes,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            top_k=top_k,
            min_box_size=min_box_size,
            orig_w=orig_w,
            orig_h=orig_h,
            class_agnostic_nms=class_agnostic_nms,
            class_conf_thresholds=cls_thresholds,
        )

        if enable_tracker:
            detections = self.tracker.update(detections)

        det_msg = PerceptionTargets()
        det_msg.header = msg.header
        det_msg.fps = int(round(self.current_fps))

        for det in detections:
            target = Target()
            target.type = det['type']
            target.track_id = int(det['track_id'])

            roi = Roi()
            roi.type = det['type']
            roi.confidence = det['confidence']

            rect = RegionOfInterest()
            rect.x_offset = det['x_offset']
            rect.y_offset = det['y_offset']
            rect.width = det['box_width']
            rect.height = det['box_height']
            rect.do_rectify = False
            roi.rect = rect

            target.rois.append(roi)
            det_msg.targets.append(target)

        t_nms_end = time.perf_counter()

        # =====================================================
        # 6. 优先发布 /hobot_dnn_detection（不受调试图绘制与编码阻塞）
        # =====================================================
        t_pub_det_start = time.perf_counter()
        self.det_pub.publish(det_msg)
        t_pub_det_end = time.perf_counter()
        e2e_ms = self._compute_e2e_latency_ms(msg)

        # =====================================================
        # 7. 按需绘制并限频发布 /yolo_detection_image
        # =====================================================
        draw_ms = 0.0
        pub_img_ms = 0.0
        if self._should_publish_debug_image(t_pub_det_end):
            t_draw_start = time.perf_counter()
            draw_debug_overlay(orig_frame, detections, self.current_fps)
            t_draw_end = time.perf_counter()
            draw_ms = (t_draw_end - t_draw_start) * 1000.0

            t_pub_img_start = time.perf_counter()
            out_img_msg = self.bridge.cv2_to_imgmsg(orig_frame, encoding='bgr8')
            out_img_msg.header = msg.header
            self.img_pub.publish(out_img_msg)
            t_pub_img_end = time.perf_counter()
            pub_img_ms = (t_pub_img_end - t_pub_img_start) * 1000.0

            self._last_debug_pub_ts = t_pub_img_end
            self.debug_pub_count += 1

        t_frame_end = time.perf_counter()

        # =====================================================
        # 8. 记录各阶段耗时与 FPS 统计
        # =====================================================
        preprocess_ms = (t_preprocess_end - t0) * 1000.0
        infer_ms = (t_infer_end - t_preprocess_end) * 1000.0
        postprocess_ms = (t_postprocess_end - t_infer_end) * 1000.0
        nms_ms = (t_nms_end - t_postprocess_end) * 1000.0
        publish_ms = (t_pub_det_end - t_pub_det_start) * 1000.0 + pub_img_ms
        total_ms = (t_frame_end - t_frame_start) * 1000.0

        self.latency_tracker.record({
            'preprocess': preprocess_ms,
            'infer': infer_ms,
            'postprocess': postprocess_ms,
            'nms': nms_ms,
            'draw': draw_ms,
            'publish': publish_ms,
            'total': total_ms,
            'e2e': e2e_ms,
        })

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = t_frame_end - self.start_time
            if elapsed > 0:
                self.current_fps = 30.0 / elapsed
            self.start_time = t_frame_end

        stats_interval = float(self.get_parameter('stats_log_interval_sec').value)
        if stats_interval > 0.0 and (t_frame_end - self._last_stats_log_ts) >= stats_interval:
            dt = t_frame_end - self._last_stats_log_ts
            proc_delta = self.frame_count - self._last_stats_frame_count
            rx_delta = self.frame_slot.rx_count - self._last_stats_rx_count
            dropped_delta = self.frame_slot.dropped_count - self._last_stats_dropped_count
            proc_fps = proc_delta / dt if dt > 0 else 0.0
            rx_fps = rx_delta / dt if dt > 0 else 0.0

            report = self.latency_tracker.format_report(
                proc_fps=proc_fps,
                rx_fps=rx_fps,
                dropped_frames=dropped_delta,
                debug_pub_frames=self.debug_pub_count,
                dropped_total=self.frame_slot.dropped_count,
            )
            self.get_logger().info(report)

            self._last_stats_log_ts = t_frame_end
            self._last_stats_frame_count = self.frame_count
            self._last_stats_rx_count = self.frame_slot.rx_count
            self._last_stats_dropped_count = self.frame_slot.dropped_count

    def destroy_node(self):
        self._stop_worker = True
        self.frame_slot.wake()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = BPUYoloV8DigitNode()
    except Exception as e:
        # 构造失败必须让进程退出，否则节点看起来"活着"但从不发布检测结果。
        print(f'[bpu_yolov8_node] 初始化失败: {e}', file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        # Ctrl+C 时 rclpy 的信号处理器已经做过 shutdown，这里只需静默退出。
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
