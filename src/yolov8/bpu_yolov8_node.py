#!/usr/bin/env python3
# coding=utf-8

import os
import sys
import time

import cv2
import numpy as np
from cv_bridge import CvBridge
from hobot_dnn import pyeasy_dnn as dnn

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, RegionOfInterest

# ============================================================
# 修改：
# 原来使用 vision_msgs/Detection2DArray
# 现在直接发布 ai_msgs/PerceptionTargets
# 与 detection_turn_controller.cpp 完全对接
# ============================================================
from ai_msgs.msg import PerceptionTargets, Target, Roi

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


class BPUYoloV8DigitNode(Node):
    def __init__(self):
        super().__init__('bpu_yolov8_digit_node')
        self.bridge = CvBridge()

        # =====================================================
        # 1. YOLO 参数
        # =====================================================
        self.declare_parameter('conf_thresh', 0.25)
        self.declare_parameter('iou_thresh', 0.55)
        self.declare_parameter('nms_top_k', 20)
        self.declare_parameter('min_box_size', 6)

        # =====================================================
        # 2. 类别文件
        # =====================================================
        self.declare_parameter('model_dir', DEFAULT_MODEL_DIR)
        self.declare_parameter('cls_list_path', '')
        self.declare_parameter('model_path', '')

        model_dir = self.get_parameter('model_dir').value or DEFAULT_MODEL_DIR
        cls_list_path = self.get_parameter('cls_list_path').value or os.path.join(
            model_dir, 'out.list')
        model_path = self.get_parameter('model_path').value or os.path.join(
            model_dir, 'best_modified_yolov8.bin')

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
        if not os.path.exists(model_path):
            raise RuntimeError(f'模型文件不存在: {model_path}')

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
        # 5. FPS统计
        # =====================================================
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0

        # =====================================================
        # 6. ROS2接口
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

        self.get_logger().info('订阅图像: /image_raw')
        self.get_logger().info('发布检测结果: /hobot_dnn_detection [ai_msgs/msg/PerceptionTargets]')
        self.get_logger().info('发布检测图像: /yolo_detection_image')

    def _make_grid(self, w, h):
        grids = []
        for s in self.strides:
            gw = w // s
            gh = h // s
            xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
            grid = np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32) + 0.5
            grids.append((grid, s))
        return grids

    def image_callback(self, msg):
        conf_thresh = self.get_parameter('conf_thresh').value
        iou_thresh = self.get_parameter('iou_thresh').value
        top_k = self.get_parameter('nms_top_k').value
        min_box_size = self.get_parameter('min_box_size').value

        # =====================================================
        # 1. ROS Image -> OpenCV
        # =====================================================
        try:
            orig_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
            return

        orig_h, orig_w = orig_frame.shape[:2]

        # =====================================================
        # 2. Letterbox & BGR -> NV12
        # =====================================================
        input_frame, r, (pad_w, pad_h) = letterbox(orig_frame, (self.model_w, self.model_h))
        nv12 = bgr2nv12_fast(input_frame)

        # =====================================================
        # 4. X5 BPU推理
        # =====================================================
        try:
            outputs = self.bpu_model.forward(nv12)
        except Exception as e:
            self.get_logger().error(f'BPU推理失败: {e}')
            return

        # =====================================================
        # 5. 反量化 + 输出维度整理
        # =====================================================
        tensors = []
        num_classes = len(self.classes)

        for out in outputs:
            data = out.buffer.astype(np.float32)
            if hasattr(out.properties, 'scale_data') and len(out.properties.scale_data) > 0:
                scale = np.array(out.properties.scale_data, dtype=np.float32)
                data *= scale[0] if len(scale) == 1 else scale

            if len(data.shape) == 4 and (data.shape[1] == num_classes or data.shape[1] == 64):
                data = np.transpose(data, (0, 2, 3, 1))
            tensors.append(data)

        cls_heads = sorted([t for t in tensors if t.shape[-1] == num_classes], key=lambda x: x.shape[1], reverse=True)
        reg_heads = sorted([t for t in tensors if t.shape[-1] == 64], key=lambda x: x.shape[1], reverse=True)

        if len(cls_heads) != 3 or len(reg_heads) != 3:
            self.get_logger().warn(f'YOLOv8输出头数量异常: cls={len(cls_heads)}, reg={len(reg_heads)}')
            return

        # =====================================================
        # 6. YOLOv8 DFL解析
        # =====================================================
        all_boxes, all_scores, all_cls_ids = [], [], []

        for (grid, stride), cls_feat, reg_feat in zip(self.grids, cls_heads, reg_heads):
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
            dfl = np.dot(reg_p, self.dfl_weights)

            # 恢复到640×640 letterbox坐标
            x1 = (valid_grid[:, 0] - dfl[:, 0]) * stride
            y1 = (valid_grid[:, 1] - dfl[:, 1]) * stride
            x2 = (valid_grid[:, 0] + dfl[:, 2]) * stride
            y2 = (valid_grid[:, 1] + dfl[:, 3]) * stride

            # 映射回原始图像坐标
            x1 = np.clip((x1 - pad_w) / r, 0, orig_w)
            y1 = np.clip((y1 - pad_h) / r, 0, orig_h)
            x2 = np.clip((x2 - pad_w) / r, 0, orig_w)
            y2 = np.clip((y2 - pad_h) / r, 0, orig_h)

            all_boxes.append(np.stack([x1, y1, x2 - x1, y2 - y1], axis=1))
            all_scores.append(valid_scores)
            all_cls_ids.append(valid_cls_ids)

        # =====================================================
        # 7. 创建 ai_msgs/PerceptionTargets
        # =====================================================
        det_msg = PerceptionTargets()
        det_msg.header = msg.header
        det_msg.fps = int(round(self.current_fps))

        # =====================================================
        # 8. NMS + ai_msgs消息封装
        # =====================================================
        if len(all_boxes) > 0:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            cls_ids = np.concatenate(all_cls_ids, axis=0)

            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), float(conf_thresh), float(iou_thresh))

            if len(indices) > 0:
                indices = np.array(indices).flatten()
                if len(indices) > top_k:
                    indices = indices[:top_k]

                for det_index, i in enumerate(indices):
                    bx, by, bw, bh = boxes[i]

                    if bw < min_box_size or bh < min_box_size:
                        continue

                    conf = float(scores[i])
                    cid = int(cls_ids[i])
                    raw_name = self.classes[cid] if cid < len(self.classes) else str(cid)

                    cx, cy = float(bx + bw / 2.0), float(by + bh / 2.0)

                    # A. ai_msgs Target
                    target = Target()
                    target.type = raw_name
                    target.track_id = int(det_index)

                    # B. ai_msgs Roi
                    roi = Roi()
                    roi.type = raw_name
                    roi.confidence = float(conf)

                    # C. sensor_msgs/RegionOfInterest
                    rect = RegionOfInterest()
                    x_offset = int(max(0, round(bx)))
                    y_offset = int(max(0, round(by)))
                    box_width = int(max(1, round(bw)))
                    box_height = int(max(1, round(bh)))

                    if x_offset < orig_w:
                        box_width = min(box_width, orig_w - x_offset)
                    if y_offset < orig_h:
                        box_height = min(box_height, orig_h - y_offset)

                    rect.x_offset, rect.y_offset = x_offset, y_offset
                    rect.width, rect.height = box_width, box_height
                    rect.do_rectify = False
                    roi.rect = rect

                    target.rois.append(roi)
                    det_msg.targets.append(target)

                    # D. OpenCV可视化
                    color = (0, 255, 0)
                    draw_x1, draw_y1 = int(bx), int(by)
                    draw_x2, draw_y2 = int(bx + bw), int(by + bh)

                    cv2.rectangle(orig_frame, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)
                    cv2.circle(orig_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

                    label = f'{raw_name} {conf:.2f}'
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    lbl_y = max(draw_y1, th + baseline + 2)

                    cv2.rectangle(orig_frame, (draw_x1, lbl_y - th - baseline - 2), (draw_x1 + tw + 2, lbl_y), color, -1)
                    cv2.putText(orig_frame, label, (draw_x1 + 1, lbl_y - baseline - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # =====================================================
        # 9. 发布消息与图像
        # =====================================================
        self.det_pub.publish(det_msg)

        fps_text = f'FPS: {self.current_fps:.1f} | Digits: {len(det_msg.targets)}'
        cv2.putText(orig_frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        out_img_msg = self.bridge.cv2_to_imgmsg(orig_frame, encoding='bgr8')
        out_img_msg.header = msg.header
        self.img_pub.publish(out_img_msg)

        # =====================================================
        # 11. FPS统计
        # =====================================================
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now = time.time()
            elapsed = now - self.start_time
            if elapsed > 0:
                self.current_fps = 30.0 / elapsed
            self.start_time = now


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
