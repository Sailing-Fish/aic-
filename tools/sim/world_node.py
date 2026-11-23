#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仿真世界节点：订阅 /cmd_vel，积分车辆位姿，发布 /odom 与 /image_raw。

只替换"世界"，不碰巡线算法。现有 line_follower_node / cmd_mux_node 一行不改，
只要把它们的输入接到这里，就能在虚拟赛道上跑。

动力学按实测标定：
    L        = 139.5 mm   轴距
    tau_speed= 0.61 s     速度一阶响应
    tau_steer= 0.26 s     转向一阶响应
    max_steer= 21 deg     舵机极限（实测饱和点）

支持光照扰动与决赛道具压力测试：
    --gain 0.85 --color-cast warm --glare 0.4 --blur 0.8
    --random-yellow 2 --cones 3 --no-entry 2

用法：
    python3 tools/sim/world_node.py --track tools/sim/tracks/fork_s.json
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import TrackRenderer, parse_color_cast, point_to_polyline_distance_mm  # noqa: E402

try:
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Empty
    from tf2_ros import TransformBroadcaster
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    Node = object  # type: ignore[assignment,misc]


def yaw_to_quat(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class OffTrackTracker:
    """纯计算离道监测器：按车辆中心到赛道中心线垂直距离 vs 半车道宽累计出界时长。"""

    def __init__(self, centerline_mm, lane_width_mm: float = 430.0, margin_mm: float = 0.0):
        self.centerline_mm = np.asarray(centerline_mm, dtype=np.float64)
        self.half_lane_mm = 0.5 * float(lane_width_mm) + float(margin_mm)
        self.off_track_s = 0.0
        self.last_dist_mm = 0.0
        self.is_off_track = False

    def reset(self) -> None:
        self.off_track_s = 0.0
        self.last_dist_mm = 0.0
        self.is_off_track = False

    def update(self, x_mm: float, y_mm: float, dt: float) -> float:
        dist = point_to_polyline_distance_mm(x_mm, y_mm, self.centerline_mm)
        self.last_dist_mm = dist
        self.is_off_track = dist > self.half_lane_mm
        if self.is_off_track and dt > 0.0:
            self.off_track_s += dt
        return self.off_track_s


class SimWorld(Node):
    def __init__(self, args):
        if not HAS_ROS:
            raise RuntimeError("ROS2 (rclpy) 未安装，无法启动 SimWorld 节点")
        super().__init__("sim_world")
        self.renderer = TrackRenderer(
            args.track,
            random_yellow=args.random_yellow,
            cones=args.cones,
            no_entry=args.no_entry,
            prop_seed=args.prop_seed,
        )
        meta = self.renderer.meta
        self.track = meta["name"]
        self.render_blur = float(args.blur)
        self.render_gain = float(args.gain)
        self.render_color_cast = parse_color_cast(args.color_cast)
        self.render_glare = float(args.glare)

        self.L = args.wheelbase_mm / 1000.0
        self.tau_v = args.tau_speed
        self.tau_d = args.tau_steer
        self.max_steer = math.radians(args.max_steer_deg)

        cl = meta["centerline_mm"]
        if args.init_x is None:
            x0, y0 = cl[0]
            x1, y1 = cl[min(2, len(cl) - 1)]
            yaw0 = math.atan2(y1 - y0, x1 - x0)
        else:
            x0, y0, yaw0 = args.init_x, args.init_y, math.radians(args.init_yaw_deg)

        self.x = float(x0)          # mm
        self.y = float(y0)
        self.yaw = float(yaw0)
        self.v = 0.0                # m/s
        self.delta = 0.0            # rad, 前轮转角
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.s = 0.0                # 累计里程 m
        self.off_tracker = OffTrackTracker(
            cl, lane_width_mm=float(meta.get("lane_width_mm", 430.0))
        )
        self.off_track = 0.0        # 累计出界时长 s（实时计算）
        self._frame = 0
        # 起始位姿快照，供 /sim/reset 复位（串行扫参时复用同一个世界节点）
        self.x0, self.y0, self.yaw0 = self.x, self.y, self.yaw

        self.bridge = CvBridge()
        self.pub_odom = self.create_publisher(Odometry, args.odom_topic, 10)
        self.pub_img = self.create_publisher(Image, args.image_topic, 10)
        self.pub_gt = self.create_publisher(Twist, "/sim/gt", 10)
        self.tf = TransformBroadcaster(self)
        # 必须保存引用：不赋值给成员时订阅对象可能被回收，导致完全收不到消息
        self.sub_cmd = self.create_subscription(
            Twist, args.cmd_topic, self.on_cmd, 10)
        self.sub_reset = self.create_subscription(
            Empty, "/sim/reset", self.on_reset, 10)

        self.dt = 1.0 / args.rate
        self.timer = self.create_timer(self.dt, self.on_timer)
        self.get_logger().info(
            f"sim_world up: track={self.track} start=({self.x:.0f},{self.y:.0f})mm "
            f"yaw={math.degrees(self.yaw):.1f}deg L={self.L*1000:.1f}mm "
            f"rate={args.rate}Hz gain={self.render_gain:.2f} glare={self.render_glare:.2f} "
            f"props=(Y+{args.random_yellow},C={args.cones},NE={args.no_entry})")

    def on_cmd(self, msg):
        self.v_cmd = float(msg.linear.x)
        self.w_cmd = float(msg.angular.z)

    def on_reset(self, msg):
        """把车放回起点，清空里程与出界计数。串行扫参时每组之前调用。"""
        self.x, self.y, self.yaw = self.x0, self.y0, self.yaw0
        self.v = self.delta = self.v_cmd = self.w_cmd = 0.0
        self.s = 0.0
        self.off_tracker.reset()
        self.off_track = 0.0
        self._frame = 0
        self.get_logger().info("[sim] reset")

    def on_timer(self):
        dt = self.dt
        # 一阶滞后：离散化 alpha = dt/(tau+dt)
        self.v += dt / (self.tau_v + dt) * (self.v_cmd - self.v)
        # 由 ω 反解舵角，低速时限幅避免发散
        if abs(self.v) > 0.02:
            d_cmd = math.atan(self.w_cmd * self.L / self.v)
        else:
            d_cmd = math.copysign(self.max_steer, self.w_cmd) if self.w_cmd else 0.0
        d_cmd = max(-self.max_steer, min(self.max_steer, d_cmd))
        self.delta += dt / (self.tau_d + dt) * (d_cmd - self.delta)

        # 自行车模型积分（x/y 用 mm）
        v_mm = self.v * 1000.0
        self.x += v_mm * math.cos(self.yaw) * dt
        self.y += v_mm * math.sin(self.yaw) * dt
        self.yaw += self.v * math.tan(self.delta) / self.L * dt
        self.s += abs(self.v) * dt

        # 修复：实时计算车辆到赛道中心线垂直距离，超过半车道宽累加出界时长
        self.off_track = self.off_tracker.update(self.x, self.y, dt)

        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = yaw_to_quat(self.yaw)

        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = "odom"
        od.child_frame_id = "base_footprint"
        od.pose.pose.position.x = self.x / 1000.0
        od.pose.pose.position.y = self.y / 1000.0
        od.pose.pose.orientation.z = qz
        od.pose.pose.orientation.w = qw
        od.twist.twist.linear.x = self.v
        od.twist.twist.angular.z = self.v * math.tan(self.delta) / self.L
        self.pub_odom.publish(od)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_footprint"
        tf.transform.translation.x = od.pose.pose.position.x
        tf.transform.translation.y = od.pose.pose.position.y
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf.sendTransform(tf)

        gt = Twist()
        gt.linear.x = self.s
        gt.linear.y = self.off_tracker.last_dist_mm / 1000.0
        gt.linear.z = self.off_track
        gt.angular.z = self.yaw
        self.pub_gt.publish(gt)

        img = self.renderer.render(
            self.x, self.y, self.yaw,
            blur=self.render_blur,
            gain=self.render_gain,
            color_cast=self.render_color_cast,
            glare=self.render_glare,
        )
        im = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        im.header.stamp = stamp
        im.header.frame_id = "camera_link"
        try:
            self.pub_img.publish(im)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"[sim] image publish failed: {exc}")

        # 每 30 帧打一行，扫参脚本直接解析
        self._frame += 1
        if self._frame % 30 == 0:
            self.get_logger().info(
                f"[sim] s={self.s:.2f} off={self.off_track:.2f} "
                f"lat={self.off_tracker.last_dist_mm:.0f}mm "
                f"v={self.v:.3f} yaw={math.degrees(self.yaw):.1f}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tracks", "fork_s.json"))
    ap.add_argument("--rate", type=float, default=30.0)
    # 默认走仿真专用话题，避免和真底盘抢 /cmd_vel
    ap.add_argument("--cmd-topic", default="/sim/cmd_vel")
    ap.add_argument("--odom-topic", default="/sim/odom")
    ap.add_argument("--image-topic", default="/image_raw")
    ap.add_argument("--wheelbase-mm", type=float, default=139.5)
    ap.add_argument("--tau-speed", type=float, default=0.61)
    ap.add_argument("--tau-steer", type=float, default=0.26)
    ap.add_argument("--max-steer-deg", type=float, default=21.0)
    ap.add_argument("--init-x", type=float, default=None)
    ap.add_argument("--init-y", type=float, default=0.0)
    ap.add_argument("--init-yaw-deg", type=float, default=0.0)
    # 光照与画质扰动
    ap.add_argument("--gain", type=float, default=1.0, help="亮度增益，如 0.75 或 1.25")
    ap.add_argument("--color-cast", default="none",
                    help="偏色/色温：none / warm / cool / yellowed 或 'b,g,r' 三元组")
    ap.add_argument("--glare", type=float, default=0.0, help="定向眩光强度 0.0..1.0")
    ap.add_argument("--blur", type=float, default=0.0, help="高斯模糊 sigma")
    # 决赛道具注入
    ap.add_argument("--random-yellow", type=int, default=0, help="额外注入随机黄线数量")
    ap.add_argument("--cones", type=int, default=0, help="随机注入红锥数量")
    ap.add_argument("--no-entry", type=int, default=0, help="随机注入空心红禁行区数量")
    ap.add_argument("--prop-seed", type=int, default=42, help="道具随机种子")
    return ap


def main():
    args = build_arg_parser().parse_args()
    rclpy.init()
    node = SimWorld(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"[sim] spin exited abnormally: {exc}", flush=True)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
