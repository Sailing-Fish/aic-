#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
import os
import sys
import time

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
except ImportError:
    rclpy = None
    Node = object
    Image = None
    CvBridge = None


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')

        # 声明并获取 ROS 2 参数
        self.declare_parameter('video_device', '/dev/video0')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('framerate', 30.0)
        self.declare_parameter('topic_name', '/image_raw')
        self.declare_parameter('qos_depth', 1)
        self.declare_parameter('v4l2_buffer_size', 1)
        default_stats_sec = float(os.environ.get('CAMERA_STATS_LOG_INTERVAL_SEC', '5.0'))
        self.declare_parameter('stats_log_interval_sec', default_stats_sec)

        device = self.get_parameter('video_device').get_parameter_value().string_value
        width = self.get_parameter('image_width').get_parameter_value().integer_value
        height = self.get_parameter('image_height').get_parameter_value().integer_value
        framerate = self.get_parameter('framerate').get_parameter_value().double_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        qos_depth = max(1, int(self.get_parameter('qos_depth').get_parameter_value().integer_value))
        v4l2_buf_size = max(1, int(self.get_parameter('v4l2_buffer_size').get_parameter_value().integer_value))
        self.stats_log_interval_sec = float(
            self.get_parameter('stats_log_interval_sec').get_parameter_value().double_value
        )

        # 创建 ROS 2 图像话题发布者（默认队列深度 1，避免旧帧堆积）
        self.publisher_ = self.create_publisher(Image, topic_name, qos_depth)
        self.bridge = CvBridge()

        # 初始化 OpenCV 摄像头捕获
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f"无法打开设备 {device}，请检查连接及权限！")
            sys.exit(1)

        # 减小 V4L2 底层缓冲深度，避免读取到滞后的旧帧
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, v4l2_buf_size)
        # 设置摄像头编解码格式为 MJPEG (保证 640x480 / 1080P 30 帧)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, framerate)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_buf = int(self.cap.get(cv2.CAP_PROP_BUFFERSIZE))

        self.get_logger().info("摄像头已成功启动!")
        self.get_logger().info(f"设备路径  : {device}")
        self.get_logger().info(f"分辨率    : {actual_w}x{actual_h}")
        self.get_logger().info(f"目标帧率  : {actual_fps} FPS")
        self.get_logger().info(f"发布话题  : {topic_name} (qos_depth={qos_depth}, v4l2_buf={actual_buf})")

        # 帧率与耗时统计
        self.frame_count = 0
        self.fail_count = 0
        self._last_stats_ts = time.perf_counter()
        self._last_stats_frames = 0
        self._read_ms_window = deque(maxlen=150)
        self._pub_ms_window = deque(maxlen=150)

        # 创建定时器，循环采集并发布图像
        timer_period = 1.0 / max(1.0, framerate)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        t0 = time.perf_counter()
        ret, frame = self.cap.read()
        t_read = time.perf_counter()
        if not ret or frame is None:
            self.fail_count += 1
            self.get_logger().warn("未能成功读取摄像头图像帧！")
            return

        # 将 OpenCV 的 BGR8 图像转为 ROS 2 sensor_msgs/Image 消息
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"

        # 发布话题
        self.publisher_.publish(msg)
        t_pub = time.perf_counter()

        self.frame_count += 1
        self._read_ms_window.append((t_read - t0) * 1000.0)
        self._pub_ms_window.append((t_pub - t_read) * 1000.0)

        if self.stats_log_interval_sec > 0.0 and (t_pub - self._last_stats_ts) >= self.stats_log_interval_sec:
            dt = t_pub - self._last_stats_ts
            measured_fps = (self.frame_count - self._last_stats_frames) / dt if dt > 0 else 0.0
            read_arr = np.asarray(self._read_ms_window, dtype=np.float64)
            pub_arr = np.asarray(self._pub_ms_window, dtype=np.float64)
            read_p50, read_p95 = np.percentile(read_arr, [50.0, 95.0]) if len(read_arr) else (0.0, 0.0)
            pub_p50, pub_p95 = np.percentile(pub_arr, [50.0, 95.0]) if len(pub_arr) else (0.0, 0.0)

            self.get_logger().info(
                f"[DEC] camera perf fps={measured_fps:.1f} "
                f"read_p50={read_p50:.2f}ms read_p95={read_p95:.2f}ms "
                f"pub_p50={pub_p50:.2f}ms pub_p95={pub_p95:.2f}ms "
                f"fails={self.fail_count}"
            )
            self._last_stats_ts = t_pub
            self._last_stats_frames = self.frame_count

    def destroy_node(self):
        # 节点退出时释放摄像头资源
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("收到中断信号，程序退出...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
