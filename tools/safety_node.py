#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
失控兜底（deadman）。

背景：底盘下位机固件没有指令超时保护——主控不再发 /cmd_vel 时，
车会保持最后一条速度指令一直跑。
  1) line_follower 崩溃 → 上游速度话题静默 → 车直冲。
  2) cmd_mux_node 单独崩溃/卡死 → line_follower 仍在发 /cmd_vel_follower，
     但 /cmd_vel 已无人转发 → 车同样直冲！

策略：
  - 监听两个上游速度话题（/cmd_vel_follower, /cmd_vel_stop）
  - 同时监听 cmd_mux_node 的 20Hz 心跳话题（/cmd_mux/heartbeat）与外部 /cmd_vel 活动
  - 只要上游全部静默超过 silence_timeout，OR mux 心跳超时超过 mux_timeout_sec
    （或上游已活跃超过 mux_startup_grace_sec 却始终未收到 mux 心跳），
    立即以 20 Hz 向 /cmd_vel 发零速；故障恢复后立即停止兜底。

注意：不把自身发布的零速 /cmd_vel 当作 mux 存活依据，避免自锁。
"""

import sys

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool
except ImportError:
    rclpy = None
    Node = object
    Twist = None
    Bool = None

UPSTREAM_TOPICS = [
    '/cmd_vel_follower',
    '/cmd_vel_stop',
]


class SafetyWatchdogLogic:
    """
    Pure-Python deterministic state machine for deadman safety braking.
    Testable without ROS2 runtime.
    """

    def __init__(
        self,
        upstream_topics=None,
        silence_timeout=0.50,
        mux_timeout_sec=0.35,
        mux_startup_grace_sec=3.0,
        require_seen=True,
        require_mux_heartbeat=True,
    ):
        self.upstream_topics = list(upstream_topics or UPSTREAM_TOPICS)
        self.silence_timeout = float(max(0.01, silence_timeout))
        self.mux_timeout_sec = float(max(0.01, mux_timeout_sec))
        self.mux_startup_grace_sec = float(max(0.0, mux_startup_grace_sec))
        self.require_seen = bool(require_seen)
        self.require_mux_heartbeat = bool(require_mux_heartbeat)

        self.last_seen = {t: None for t in self.upstream_topics}
        self.seen_any = False
        self.first_upstream_ts = None

        self.mux_seen = False
        self.last_mux_hb_ts = None

        self.external_cmd_vel_seen = False
        self.last_external_cmd_vel_ts = None

        self.braking = False
        self.brake_reason = ''

    def on_upstream(self, topic, now_sec):
        now_sec = float(now_sec)
        self.last_seen[topic] = now_sec
        if not self.seen_any:
            self.seen_any = True
            self.first_upstream_ts = now_sec

    def on_mux_heartbeat(self, now_sec, alive=True):
        if not alive:
            return
        now_sec = float(now_sec)
        self.mux_seen = True
        self.last_mux_hb_ts = now_sec

    def on_cmd_vel_echo(self, now_sec, linear_x, angular_z):
        """
        Observe /cmd_vel traffic purely to detect active chassis motion.
        Ignores zero-velocity messages while braking so self-published zeros
        can never mask a dead cmd_mux_node.
        """
        is_nonzero = (abs(float(linear_x)) > 1e-6) or (abs(float(angular_z)) > 1e-6)
        if self.braking and not is_nonzero:
            return
        if is_nonzero:
            self.external_cmd_vel_seen = True
            self.last_external_cmd_vel_ts = float(now_sec)

    def evaluate(self, now_sec):
        """
        Evaluate deadman condition at timestamp now_sec (seconds).
        Returns dict:
          {
            'should_brake': bool,
            'transition': None | 'deadman_on' | 'deadman_off',
            'reason': str,
            'upstream_silence': float,
            'mux_silence': float,
            'newest_src': str | None,
          }
        """
        now_sec = float(now_sec)

        if self.require_seen and not (self.seen_any or self.mux_seen or self.external_cmd_vel_seen):
            return {
                'should_brake': False,
                'transition': None,
                'reason': 'idle',
                'upstream_silence': float('inf'),
                'mux_silence': float('inf'),
                'newest_src': None,
            }

        newest = None
        newest_src = None
        for topic, stamp in self.last_seen.items():
            if stamp is None:
                continue
            if newest is None or stamp > newest:
                newest = stamp
                newest_src = topic

        upstream_silence = float('inf') if newest is None else max(0.0, now_sec - newest)
        mux_silence = (
            float('inf')
            if self.last_mux_hb_ts is None
            else max(0.0, now_sec - self.last_mux_hb_ts)
        )

        reasons = []
        if self.seen_any and upstream_silence > self.silence_timeout:
            reasons.append('upstream_silence')
        elif (not self.seen_any) and (not self.require_seen) and upstream_silence > self.silence_timeout:
            reasons.append('upstream_silence')

        if self.require_mux_heartbeat:
            if self.mux_seen:
                if mux_silence > self.mux_timeout_sec:
                    reasons.append('mux_heartbeat_timeout')
            elif self.seen_any and self.first_upstream_ts is not None:
                if (now_sec - self.first_upstream_ts) > self.mux_startup_grace_sec:
                    reasons.append('mux_heartbeat_missing')
            elif self.external_cmd_vel_seen and self.last_external_cmd_vel_ts is not None:
                if (now_sec - self.last_external_cmd_vel_ts) > self.mux_timeout_sec:
                    reasons.append('cmd_vel_stale_without_mux')

        should_brake = len(reasons) > 0
        reason_str = '+'.join(reasons) if reasons else 'ok'

        transition = None
        if should_brake and not self.braking:
            self.braking = True
            self.brake_reason = reason_str
            transition = 'deadman_on'
        elif should_brake and self.braking:
            self.brake_reason = reason_str
        elif (not should_brake) and self.braking:
            self.braking = False
            self.brake_reason = ''
            transition = 'deadman_off'

        return {
            'should_brake': should_brake,
            'transition': transition,
            'reason': reason_str,
            'upstream_silence': upstream_silence,
            'mux_silence': mux_silence,
            'newest_src': newest_src,
        }


class SafetyNode(Node):
    def __init__(self):
        super().__init__('safety_node')

        self.declare_parameter('silence_timeout', 0.50)
        self.declare_parameter('mux_timeout_sec', 0.35)
        self.declare_parameter('mux_startup_grace_sec', 3.0)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('require_seen', True)
        self.declare_parameter('require_mux_heartbeat', True)
        self.declare_parameter('mux_heartbeat_topic', '/cmd_mux/heartbeat')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        silence_timeout = float(self.get_parameter('silence_timeout').value)
        mux_timeout_sec = float(self.get_parameter('mux_timeout_sec').value)
        mux_startup_grace_sec = float(self.get_parameter('mux_startup_grace_sec').value)
        rate = float(self.get_parameter('publish_rate').value)
        require_seen = bool(self.get_parameter('require_seen').value)
        require_mux_heartbeat = bool(self.get_parameter('require_mux_heartbeat').value)
        mux_heartbeat_topic = str(self.get_parameter('mux_heartbeat_topic').value)
        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)

        self.logic = SafetyWatchdogLogic(
            upstream_topics=UPSTREAM_TOPICS,
            silence_timeout=silence_timeout,
            mux_timeout_sec=mux_timeout_sec,
            mux_startup_grace_sec=mux_startup_grace_sec,
            require_seen=require_seen,
            require_mux_heartbeat=require_mux_heartbeat,
        )

        self.pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        for topic in UPSTREAM_TOPICS:
            self.create_subscription(
                Twist,
                topic,
                lambda msg, t=topic: self._on_upstream(t, msg),
                10,
            )

        if Bool is not None and mux_heartbeat_topic:
            self.create_subscription(
                Bool,
                mux_heartbeat_topic,
                self._on_mux_heartbeat,
                10,
            )

        self.create_subscription(
            Twist,
            cmd_vel_topic,
            self._on_cmd_vel_echo,
            10,
        )

        self.timer = self.create_timer(1.0 / max(1.0, rate), self._tick)

        self.get_logger().info(
            f'safety_node armed: upstream={UPSTREAM_TOPICS} '
            f'mux_hb={mux_heartbeat_topic} silence_timeout={silence_timeout:.2f}s '
            f'mux_timeout={mux_timeout_sec:.2f}s rate={rate:.0f}Hz'
        )

    def _now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _on_upstream(self, topic, _msg):
        self.logic.on_upstream(topic, self._now_sec())

    def _on_mux_heartbeat(self, msg):
        self.logic.on_mux_heartbeat(self._now_sec(), alive=bool(msg.data))

    def _on_cmd_vel_echo(self, msg):
        self.logic.on_cmd_vel_echo(
            self._now_sec(),
            msg.linear.x,
            msg.angular.z,
        )

    def _tick(self):
        res = self.logic.evaluate(self._now_sec())
        if res['transition'] == 'deadman_on':
            self.get_logger().info(
                f"[DEC] safety deadman_on reason={res['reason']} "
                f"upstream_silence={res['upstream_silence']:.2f}s "
                f"mux_silence={res['mux_silence']:.2f}s"
            )
        elif res['transition'] == 'deadman_off':
            self.get_logger().info(
                f"[DEC] safety deadman_off last_src={res['newest_src']} "
                f"upstream_silence={res['upstream_silence']:.2f}s "
                f"mux_silence={res['mux_silence']:.2f}s"
            )

        if res['should_brake']:
            self._publish_zero()

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
