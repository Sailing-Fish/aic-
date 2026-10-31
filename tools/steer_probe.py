#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两速同角速度转向探针。

目的：判断底盘的 `/cmd_vel.angular.z` 到底是
    A) 角度映射：同一个 angular.z 在高速产生更大的 yaw rate（w 约随 v 翻倍）
    B) 角速度闭环：同一个 angular.z 基本产生同样的 yaw rate（w 不随 v 变）

这决定视觉前馈 `steer_ff_use_v` 要不要开：
    A 角度映射 -> false
    B 角速度闭环 -> true（前馈乘 v/0.45）

用法：
    python3 tools/steer_probe.py --dry-run
    python3 tools/steer_probe.py --speeds 0.20 0.40 --yaw 0.30

安全：
    - 只直发 /cmd_vel；巡线/仲裁/兜底必须先停，底盘驱动保持；
    - 车会走弧线，需要至少 2.5 x 2.5 m 的空地；
    - 超距、odom 超时、低压、Ctrl+C 都会发零速；
    - 每个速度之间提示把车推回起点。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

WS = Path(__file__).resolve().parents[1]
LOG_ROOT = WS / "tools" / "logs"
L = 0.1395  # 实测轴距 m


@dataclass
class ProbeResult:
    v_cmd: float
    yaw_cmd: float
    v_ss: float = 0.0
    w_ss: float = 0.0
    delta_deg: float = 0.0
    ratio: float = 0.0
    voltage: float = float("nan")
    status: str = "OK"
    note: str = ""


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def run_real(args: argparse.Namespace) -> int:
    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from std_msgs.msg import Float32
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 导入 ROS 失败：{exc}", file=sys.stderr)
        print("        请先执行： source /opt/tros/humble/setup.bash", file=sys.stderr)
        return 2

    class ProbeNode(Node):
        def __init__(self) -> None:
            super().__init__("steer_probe")
            self.pub = None
            self.v = 0.0
            self.w = 0.0
            self.x = 0.0
            self.y = 0.0
            self.path = 0.0
            self.last_x: Optional[float] = None
            self.last_y: Optional[float] = None
            self.voltage = float("nan")
            self.voltage_seen = False
            self.last_odom = 0.0
            self.odom_seen = False
            self.create_subscription(Odometry, args.odom_topic, self._odom, 20)
            self.create_subscription(Float32, args.voltage_topic, self._volt, 10)

        def _odom(self, msg) -> None:
            self.odom_seen = True
            self.last_odom = time.monotonic()
            self.v = abs(float(msg.twist.twist.linear.x))
            self.w = float(msg.twist.twist.angular.z)
            self.x = float(msg.pose.pose.position.x)
            self.y = float(msg.pose.pose.position.y)
            if self.last_x is not None and self.last_y is not None:
                step = math.hypot(self.x - self.last_x, self.y - self.last_y)
                if math.isfinite(step) and step < 1.0:
                    self.path += step
            self.last_x, self.last_y = self.x, self.y

        def _volt(self, msg) -> None:
            self.voltage = float(msg.data)
            self.voltage_seen = True

        def create_pub(self) -> None:
            self.pub = self.create_publisher(Twist, args.cmd_topic, 10)

        def send(self, v: float, w: float) -> None:
            if self.pub is None:
                return
            m = Twist()
            m.linear.x = float(v)
            m.angular.z = float(w)
            self.pub.publish(m)

        def others(self) -> List[str]:
            out = []
            for info in self.get_publishers_info_by_topic(args.cmd_topic):
                if info.node_name != self.get_name():
                    out.append(f"{info.node_namespace.rstrip('/')}/{info.node_name}")
            return out

    rclpy.init(args=None)
    node = ProbeNode()

    def spin(sec: float = 0.002) -> None:
        rclpy.spin_once(node, timeout_sec=sec)

    def hold(v: float, w: float, sec: float,
             samples: Optional[list] = None) -> None:
        end = time.monotonic() + sec
        while time.monotonic() < end:
            node.send(v, w)
            spin(0.001)
            if samples is not None:
                samples.append((time.monotonic(), node.v, node.w))
            time.sleep(0.002)

    def zero(sec: float = 0.5) -> None:
        hold(0.0, 0.0, sec)

    def max_path_ok(start_path: float) -> bool:
        return node.path - start_path <= args.max_distance

    def run_one(v_cmd: float, yaw_cmd: float) -> ProbeResult:
        res = ProbeResult(v_cmd=v_cmd, yaw_cmd=yaw_cmd)
        zero(args.settle)
        if node.v > 0.03:
            res.status = "ABORT_NOT_STILL"
            res.note = f"起步前 v={node.v:.3f}"
            return res
        start_path = node.path
        if node.voltage_seen and node.voltage < args.min_voltage:
            res.status = "ABORT_LOW_VOLTAGE"
            res.note = f"{node.voltage:.2f} V"
            return res

        # 先直行，让速度稳定
        hold(v_cmd, 0.0, args.straight)
        if not max_path_ok(start_path):
            res.status = "ABORT_DISTANCE"
            zero(0.3)
            return res

        # 再保持同一角速度
        samples = []
        hold(v_cmd, yaw_cmd, args.duration, samples)
        zero(0.4)
        res.voltage = node.voltage

        if not samples:
            res.status = "ABORT_NO_SAMPLE"
            return res
        t_end = samples[-1][0]
        tail = [(v, w) for t, v, w in samples if t >= t_end - args.steady_window]
        if len(tail) < 5:
            res.status = "ABORT_NO_STEADY"
            return res
        res.v_ss = sum(v for v, _ in tail) / len(tail)
        res.w_ss = sum(w for _, w in tail) / len(tail)
        if res.v_ss > 0.03:
            res.delta_deg = math.degrees(math.atan(L * res.w_ss / res.v_ss))
        if abs(yaw_cmd) > 1e-6:
            res.ratio = res.w_ss / yaw_cmd
        if not max_path_ok(start_path):
            res.status = "ABORT_DISTANCE"
        if node.voltage_seen and node.voltage < args.min_voltage:
            res.status = "ABORT_LOW_VOLTAGE"
        return res

    out_dir = Path(args.out) if args.out else LOG_ROOT / f"steer_probe_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.csv"
    results: List[ProbeResult] = []

    try:
        # 发现 + 发布者检查
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            spin(0.02)
        others = node.others()
        if others:
            print("[ERROR] /cmd_vel 已有发布者，探针会和它们抢控制：")
            for o in others:
                print(f"        - {o}")
            print("        请先停掉巡线/仲裁/兜底，保留底盘驱动。")
            return 2
        node.create_pub()
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            spin(0.02)
        if node.others():
            print("[ERROR] /cmd_vel 仍有其他发布者：", node.others())
            return 2

        t_end = time.monotonic() + args.wait_timeout
        while time.monotonic() < t_end:
            spin(0.02)
            if node.odom_seen and time.monotonic() - node.last_odom < 0.5:
                break
        if not node.odom_seen:
            print(f"[ERROR] {args.odom_topic} 没有数据。")
            return 2
        if node.voltage_seen and node.voltage < args.min_voltage:
            print(f"[ERROR] 电压 {node.voltage:.2f} V < {args.min_voltage:.1f} V。")
            return 2

        print(f"[OK] odom 就绪，电压 "
              f"{node.voltage:.2f} V" if node.voltage_seen else "[OK] odom 就绪")
        print(f"[INFO] 输出目录：{out_dir}")
        print("[SAFE] 车会走弧线，确认前方和左侧/右侧都有 >=2.5 m 空间。")
        if not args.yes:
            for i in range(5, 0, -1):
                print(f"       {i} 秒后开始...", end="\r", flush=True)
                time.sleep(1.0)
            print()

        with open(summary, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["v_cmd", "yaw_cmd", "v_ss", "w_ss",
                        "delta_deg", "ratio_w", "voltage", "status", "note"])
            f.flush()
            for v_cmd in args.speeds:
                # 每次都必须复位：探针会让车走弧线，第二次不能从上次终点接着跑。
                try:
                    input(f"[{v_cmd:.2f} m/s, yaw={args.yaw:.2f}] "
                          "把车推回起点、车头摆正，按 Enter 开始...")
                except EOFError:
                    print("\n[STOP] 没有交互输入。")
                    break
                res = run_one(v_cmd, args.yaw)
                results.append(res)
                w.writerow([round(res.v_cmd, 3), round(res.yaw_cmd, 3),
                            round(res.v_ss, 4), round(res.w_ss, 4),
                            round(res.delta_deg, 2), round(res.ratio, 3),
                            round(res.voltage, 3) if math.isfinite(res.voltage) else "",
                            res.status, res.note])
                f.flush()
                print(f"  -> {res.status:20s} v_ss={res.v_ss:.3f} "
                      f"w_ss={res.w_ss:+.3f} delta={res.delta_deg:+.1f}° "
                      f"w/w_cmd={res.ratio:+.2f}"
                      + (f" ({res.note})" if res.note else ""))

        print("\n" + "=" * 78)
        print(f"{'v_cmd':>6} {'v_ss':>6} {'w_cmd':>7} {'w_ss':>7} "
              f"{'delta':>7} {'w/w_cmd':>8} {'status':>10}")
        print("-" * 78)
        for r in results:
            print(f"{r.v_cmd:6.2f} {r.v_ss:6.3f} {r.yaw_cmd:7.2f} "
                  f"{r.w_ss:7.3f} {r.delta_deg:6.1f}° {r.ratio:8.2f} "
                  f"{r.status:>10}")

        ok = [r for r in results if r.status == "OK" and r.v_ss > 0.05]
        if len(ok) >= 2:
            ok.sort(key=lambda r: r.v_ss)
            lo, hi = ok[0], ok[-1]
            if abs(lo.w_ss) > 1e-3:
                ratio = abs(hi.w_ss / lo.w_ss)
                print(f"\n判断：v={lo.v_ss:.2f} 时 w={lo.w_ss:+.3f}，"
                      f"v={hi.v_ss:.2f} 时 w={hi.w_ss:+.3f}，比值 {ratio:.2f}")
                if ratio > 1.5:
                    print("  -> 更像“角度映射”：同一 angular.z 在高速产生更大 yaw rate。")
                    print("     前馈参数保持 steer_ff_use_v=false。")
                elif ratio < 1.2:
                    print("  -> 更像“角速度闭环”：yaw rate 基本不随速度变。")
                    print("     前馈参数设 steer_ff_use_v=true（乘 v/0.45）。")
                else:
                    print("  -> 介于两者之间，先按角度映射试；A/B 时两种都试一下。")
        print(f"[out] {summary}")
        return 0 if results and all(r.status == "OK" for r in results) else 1
    finally:
        try:
            zero(0.5)
        except Exception:  # noqa: BLE001
            pass
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


def print_dry(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("两速同角速度转向探针（dry-run，不发指令）")
    print("=" * 72)
    print(f"线速度档 : {', '.join(f'{v:.2f}' for v in args.speeds)} m/s")
    print(f"角速度   : {args.yaw:+.2f} rad/s（固定）")
    print(f"直行稳定 : {args.straight:.2f} s")
    print(f"转向保持 : {args.duration:.2f} s，取最后 {args.steady_window:.2f} s 稳态")
    print(f"安全阈值 : 电压 < {args.min_voltage:.1f} V / 路径 > {args.max_distance:.2f} m")
    print("\n判断逻辑：")
    print("  同一 angular.z，高速 yaw rate 约翻倍 -> 角度映射 -> steer_ff_use_v=false")
    print("  同一 angular.z，yaw rate 基本不变   -> 角速度闭环 -> steer_ff_use_v=true")
    print("\n跑之前：停掉巡线/仲裁/兜底，保留底盘；车放空地，前方/侧方 >=2.5 m。")
    print("正式命令：")
    print("  source /opt/tros/humble/setup.bash")
    print("  python3 tools/steer_probe.py --speeds "
          + " ".join(f"{v:.2f}" for v in args.speeds)
          + f" --yaw {args.yaw:.2f}")
    print("=" * 72)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="两速同角速度转向探针：判断固件是角度映射还是角速度闭环",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.20, 0.40],
                    help="线速度档，建议 0.20 和 0.40")
    ap.add_argument("--yaw", type=float, default=0.30, help="固定 angular.z")
    ap.add_argument("--straight", type=float, default=0.8, help="转向前的直行稳定时间 s")
    ap.add_argument("--duration", type=float, default=2.0, help="转向保持时间 s")
    ap.add_argument("--steady-window", type=float, default=0.8,
                    help="取最后多少秒算稳态")
    ap.add_argument("--settle", type=float, default=0.5, help="每次开始前静止确认 s")
    ap.add_argument("--max-distance", type=float, default=2.0, help="单次最大路径 m")
    ap.add_argument("--min-voltage", type=float, default=11.0, help="低压中止 V")
    ap.add_argument("--wait-timeout", type=float, default=8.0, help="等 odom 的超时 s")
    ap.add_argument("--cmd-topic", default="/cmd_vel")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--voltage-topic", default="/PowerVoltage")
    ap.add_argument("--out", default=None, help="输出目录")
    ap.add_argument("--yes", action="store_true", help="跳过初始 5 s 倒计时")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不发指令")
    args = ap.parse_args(argv)
    if len(args.speeds) < 2:
        ap.error("至少给两个速度档，才能判断随速度的变化")
    if any(v <= 0 or v > 0.8 for v in args.speeds):
        ap.error("--speeds 必须在 (0, 0.8] m/s")
    if abs(args.yaw) < 0.05:
        ap.error("--yaw 太小，测不出转向")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print_dry(args)
        return 0
    try:
        return run_real(args)
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C，已发零速退出。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
