#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        
        # 声明并获取 ROS 2 参数
        self.declare_parameter('video_device', '/dev/video0')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('framerate', 30.0)
        self.declare_parameter('topic_name', '/image_raw')

        device = self.get_parameter('video_device').get_parameter_value().string_value
        width = self.get_parameter('image_width').get_parameter_value().integer_value
        height = self.get_parameter('image_height').get_parameter_value().integer_value
        framerate = self.get_parameter('framerate').get_parameter_value().double_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value

        # 创建 ROS 2 图像话题发布者
        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()

        # 初始化 OpenCV 摄像头捕获
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f"无法打开设备 {device}，请检查连接及权限！")
            sys.exit(1)

        # 设置摄像头编解码格式为 MJPEG (保证 1080P 30 帧)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, framerate)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.get_logger().info(f"摄像头已成功启动!")
        self.get_logger().info(f"设备路径  : {device}")
        self.get_logger().info(f"分辨率    : {actual_w}x{actual_h}")
        self.get_logger().info(f"目标帧率  : {actual_fps} FPS")
        self.get_logger().info(f"发布话题  : {topic_name}")

        # 创建定时器，循环采集并发布图像
        timer_period = 1.0 / framerate
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn("未能成功读取摄像头图像帧！")
            return

        # 将 OpenCV 的 BGR8 图像转为 ROS 2 sensor_msgs/Image 消息
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"

        # 发布话题
        self.publisher_.publish(msg)

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
