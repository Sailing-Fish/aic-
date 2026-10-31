#!/usr/bin/env python3
# coding=utf-8

from collections import deque
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
    from sensor_msgs.msg import Image, RegionOfInterest
    from ai_msgs.msg import PerceptionTargets, Target, Roi
except ImportError:
    rclpy = None
    Node = object
    Image = None
    RegionOfInterest = None
    PerceptionTargets = None
    Target = None
    Roi = None

# 模型与类别文件默认放在脚本同级的 mod1 目录下。
# 用 __file__ 推导，避免把工作区绝对路径写死（工作区改名后仍可用）。
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mod1')


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


class StageLatencyTracker:
    """
    统计各阶段耗时（ms）的滑动窗口 P50 / P95：
    - preprocess: 图像解码 + letterbox + BGR->NV12
    - infer:      BPU forward 推理
    - postprocess:反量化 + YOLOv8 DFL 解码与坐标映射
    - nms:        NMS + ai_msgs/PerceptionTargets 组装
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

    def format_report(self, proc_fps, rx_fps, dropped_frames, debug_pub_frames):
        stats = self.summary_dict()
        parts = [
            f'proc_fps={proc_fps:.1f}',
            f'rx_fps={rx_fps:.1f}',
            f'dropped={dropped_frames}',
            f'dbg_pub={debug_pub_frames}',
        ]
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
):
    """
    YOLOv8 BPU 输出反量化与 DFL 解码（保持原有解码数学逻辑与坐标反映射完全不变）。
    返回: (all_boxes, all_scores, all_cls_ids, status_ok, cls_head_cnt, reg_head_cnt)
    """
    tensors = []
    for out in outputs:
        data = out.buffer.astype(np.float32)
        if hasattr(out.properties, 'scale_data') and len(out.properties.scale_data) > 0:
            scale = np.array(out.properties.scale_data, dtype=np.float32)
            data *= scale[0] if len(scale) == 1 else scale

        if len(data.shape) == 4 and (data.shape[1] == num_classes or data.shape[1] == 64):
            data = np.transpose(data, (0, 2, 3, 1))
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

    all_boxes, all_scores, all_cls_ids = [], [], []

    for (grid, stride), cls_feat, reg_feat in zip(grids, cls_heads, reg_heads):
        cls_feat_flat = cls_feat.reshape(-1, num_classes)
        cls_p = 1.0 / (1.0 + np.exp(-np.clip(cls_feat_flat, -30.0, 30.0)))

        max_scores = np.max(cls_p, axis=1)
        mask = max_scores > conf_thresh

        if not np.any(mask):
            continue

        valid_cls_ids = np.argmax(cls_p[mask], axis=1)
        valid_scores = max_scores[mask]
        valid_grid = grid[mask]

        # DFL
        valid_reg = reg_feat.reshape(-1, 4, 16)[mask]
        exp_reg = np.exp(valid_reg - np.max(valid_reg, axis=-1, keepdims=True))
        reg_p = exp_reg / np.sum(exp_reg, axis=-1, keepdims=True)
        dfl = np.dot(reg_p, dfl_weights)

        # 恢复到640×640 letterbox坐标
        x1 = (valid_grid[:, 0] - dfl[:, 0]) * stride
        y1 = (valid_grid[:, 1] - dfl[:, 1]) * stride
        x2 = (valid_grid[:, 0] + dfl[:, 2]) * stride
        y2 = (valid_grid[:, 1] + dfl[:, 3]) * stride

        # 映射回原始图像坐标
        x1 = np.clip((x1 - pad_w) / scale_r, 0, orig_w)
        y1 = np.clip((y1 - pad_h) / scale_r, 0, orig_h)
        x2 = np.clip((x2 - pad_w) / scale_r, 0, orig_w)
        y2 = np.clip((y2 - pad_h) / scale_r, 0, orig_h)

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
):
    """
    执行 NMS 并过滤过小检测框，返回结构化检测列表供消息封装和可选绘图复用。
    不改变任何原有阈值、坐标截断与类别映射语义。
    """
    detections = []
    if len(all_boxes) == 0:
        return detections

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    cls_ids = np.concatenate(all_cls_ids, axis=0)

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), float(conf_thresh), float(iou_thresh)
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


def draw_debug_overlay(orig_frame, detections, current_fps):
    """
    在需要发布调试图时绘制检测框与 FPS 信息。
    """
    color = (0, 255, 0)
    for det in detections:
        bx, by, bw, bh = det['bx'], det['by'], det['bw'], det['bh']
        cx, cy = det['cx'], det['cy']
        raw_name = det['type']
        conf = det['confidence']

        draw_x1, draw_y1 = int(bx), int(by)
        draw_x2, draw_y2 = int(bx + bw), int(by + bh)

        cv2.rectangle(orig_frame, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)
        cv2.circle(orig_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        label = f'{raw_name} {conf:.2f}'
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

    fps_text = f'FPS: {current_fps:.1f} | Digits: {len(detections)}'
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
        # 1. YOLO 参数与调试/性能统计参数
        # =====================================================
        self.declare_parameter('conf_thresh', 0.25)
        self.declare_parameter('iou_thresh', 0.55)
        self.declare_parameter('nms_top_k', 20)
        self.declare_parameter('min_box_size', 6)

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
        # 2. 类别文件
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
            with open(cls_list_path, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f.readlines() if line.strip()]
            self.get_logger().info(f'加载 {len(self.classes)} 类: {self.classes}')
        else:
            self.classes = ['1', '2', '3', '4', '5', '6', '7', '8']
            self.get_logger().warn(f'类别文件不存在: {cls_list_path}，使用预设8类数字标签')

        if len(self.classes) != 8:
            raise RuntimeError(
                f'类别数量错误: 当前 out.list 有 {len(self.classes)} 类，新模型应只有 8 类（1~8）'
            )

        # =====================================================
        # 3. 加载 X5 BPU 模型
        # =====================================================
        if bpu_model_override is not None:
            self.bpu_model = bpu_model_override
        else:
            if not os.path.exists(model_path):
                raise RuntimeError(f'模型文件不存在: {model_path}')
            if dnn is None:
                raise RuntimeError('未检测到 hobot_dnn (pyeasy_dnn)，无法加载 BPU 模型')

            self.get_logger().info(f'模型文件: {model_path}')
            self.bpu_model = dnn.load(model_path)[0]

        self.get_logger().info('RDK X5 BPU YOLOv8 视觉检测节点就绪')

        # =====================================================
        # 4. YOLOv8 模型参数
        # =====================================================
        self.model_w = 640
        self.model_h = 640
        self.strides = [8, 16, 32]
        self.grids = self._make_grid(self.model_w, self.model_h)
        self.dfl_weights = np.arange(16, dtype=np.float32)

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
        # =====================================================
        self.sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 1
        )

        self.det_pub = self.create_publisher(
            PerceptionTargets, '/hobot_dnn_detection', 1
        )

        self.img_pub = self.create_publisher(
            Image, '/yolo_detection_image', 1
        )

        pub_dbg = bool(self.get_parameter('publish_debug_image').value)
        dbg_fps = float(self.get_parameter('debug_image_max_fps').value)
        async_mode = bool(self.get_parameter('async_latest_frame').value)

        self.get_logger().info('订阅图像: /image_raw')
        self.get_logger().info('发布检测结果: /hobot_dnn_detection [ai_msgs/msg/PerceptionTargets]')
        self.get_logger().info(
            f'发布检测图像: /yolo_detection_image (enabled={pub_dbg}, max_fps={dbg_fps:.1f}, async_latest={async_mode})'
        )

        if start_worker and async_mode:
            self._worker_thread = threading.Thread(
                target=self._worker_loop, name='bpu_yolov8_worker', daemon=True
            )
            self._worker_thread.start()

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
        input_frame, r, (pad_w, pad_h) = letterbox(orig_frame, (self.model_w, self.model_h))
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
        # 4. 后处理：反量化 + YOLOv8 DFL 解码
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
            scale_r=r,
            orig_w=orig_w,
            orig_h=orig_h,
        )
        t_postprocess_end = time.perf_counter()

        if not ok:
            self.get_logger().warn(
                f'YOLOv8输出头数量异常: cls={cls_cnt}, reg={reg_cnt}'
            )
            return

        # =====================================================
        # 5. NMS + ai_msgs/PerceptionTargets 消息封装
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
        )

        det_msg = PerceptionTargets()
        det_msg.header = msg.header
        det_msg.fps = int(round(self.current_fps))

        for det in detections:
            target = Target()
            target.type = det['type']
            target.track_id = det['track_id']

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
            proc_fps = proc_delta / dt if dt > 0 else 0.0
            rx_fps = rx_delta / dt if dt > 0 else 0.0

            report = self.latency_tracker.format_report(
                proc_fps=proc_fps,
                rx_fps=rx_fps,
                dropped_frames=self.frame_slot.dropped_count,
                debug_pub_frames=self.debug_pub_count,
            )
            self.get_logger().info(report)

            self._last_stats_log_ts = t_frame_end
            self._last_stats_frame_count = self.frame_count
            self._last_stats_rx_count = self.frame_slot.rx_count

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
