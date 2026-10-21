#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
失控兜底（deadman）。

背景：底盘下位机固件没有指令超时保护——主控不再发 /cmd_vel 时，
车会保持最后一条速度指令一直跑。line_follower 崩溃 → /cmd_vel 静默 → 车直冲。

策略：监听三个**上游**速度话题，全部静默超过 threshold 就以 20 Hz 向
/cmd_vel 发零速；任一上游恢复发布，立即停止兜底。

注意订阅的是上游而不是 /cmd_vel 本身，否则会收到自己发的零速形成自锁。
"""

import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

UPSTREAM_TOPICS = [
    '/cmd_vel_follower',
    '/cmd_vel_detection',
    '/cmd_vel_stop',
]


class SafetyNode(Node):
    def __init__(self):
        super().__init__('safety_node')

        self.declare_parameter('silence_timeout', 0.30)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('require_seen', True)

        self.silence_timeout = float(self.get_parameter('silence_timeout').value)
        rate = float(self.get_parameter('publish_rate').value)
        self.require_seen = bool(self.get_parameter('require_seen').value)

        self.last_seen = {t: None for t in UPSTREAM_TOPICS}
        self.seen_any = False
        self.braking = False

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        for topic in UPSTREAM_TOPICS:
            self.create_subscription(
                Twist, topic,
                lambda msg, t=topic: self._on_upstream(t, msg), 10)

        self.timer = self.create_timer(1.0 / max(1.0, rate), self._tick)

        self.get_logger().info(
            f'safety_node armed: upstream={UPSTREAM_TOPICS} '
            f'silence_timeout={self.silence_timeout:.2f}s rate={rate:.0f}Hz')

    def _on_upstream(self, topic, _msg):
        self.last_seen[topic] = self.get_clock().now()
        self.seen_any = True

    def _tick(self):
        if self.require_seen and not self.seen_any:
            return

        now = self.get_clock().now()
        newest = None
        newest_src = None
        for topic, stamp in self.last_seen.items():
            if stamp is None:
                continue
            if newest is None or stamp > newest:
                newest = stamp
                newest_src = topic

        if newest is None:
            silence = float('inf')
        else:
            silence = (now - newest).nanoseconds / 1e9

        if silence > self.silence_timeout:
            if not self.braking:
                self.braking = True
                self.get_logger().info(
                    f'[DEC] safety deadman_on silence={silence:.2f}s')
            self._publish_zero()
        else:
            if self.braking:
                self.braking = False
                self.get_logger().info(
                    f'[DEC] safety deadman_off last_src={newest_src} '
                    f'silence={silence:.2f}s')

    def _publish_zero(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
