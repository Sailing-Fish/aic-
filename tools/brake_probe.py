#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""负速度（反向推力）刹车探针。

目的：回答一个问题——下位机接不接受**负的 linear.x**，能不能靠反拖快速停车？

背景：turn_on_robot 的 Send_Velocity() 只是把 linear.x*1000 塞进有符号 16 位发串口，
ROS 层不做任何截断。所以"发负速度"在协议上是通的，但**固件会不会执行**只有实车能验证：
    A) 接受并反转  -> 反向推力刹车可用
    B) 当 0 处理    -> 等同滑行，刹车距离不变
    C) 符号取反     -> 负速度变成前进（危险，必须知道，绝不能用于刹车）

测试分两段：
  1) 静态探针：车静止时发 -probe_speed 一小会儿，看 odom 速度是否真的变负。
     A/B/C 在这里就能定下来，车只移动几厘米，最安全。
  2) 对比刹车：加速到 V 后分别发 0（滑行基准）和 -B（反向推力），
     比较降到 stop-speed 的**时间与距离**，并记录反向过冲（v_min、倒车距离）。

用法（先 source /opt/tros/humble/setup.bash）：
    # 只看时序，不发任何指令
    python3 tools/brake_probe.py --dry-run

    # 只做静态探针，确认负速度到底会不会动（车后留 0.3 m 空）
    python3 tools/brake_probe.py --speeds 0.30 --brakes 0.0 --probe-only --yes

    # 实车完整对比：0=滑行基准，0.45/0.60=反向推力
    python3 tools/brake_probe.py --speeds 0.44 0.60 --brakes 0.0 0.45 0.60 --reps 2

    # 急停调参：0.6 m/s 从 3 个撤回阈值里找「停住不后退」的点
    python3 tools/brake_probe.py --speeds 0.60 --brakes 0.0 0.45 0.60 \
        --release-speeds 0.08 0.15 0.22 --reps 1

急停（停住不后退）：
    - 关键参数是"反拖撤回阈值" --release-speeds：发 -B 反拖，速度降到该值立刻改发 0。
      阈值太低会顶成倒车，太高又浪费刹车距离，所以脚本一次扫几个值；
    - summary 里直接给每组的判定：倒车 <= --max-rollback 且 v_min >= --max-rollback-v
      记为 OK，并按速度推荐一组「停住且距离最短」的参数。

底盘：
    - /odom 没数据时自动拉起 turn_on_wheeltec_robot.launch.py（只起底盘，不拉巡线/相机），
      已在跑就复用；退出时只收自己拉起的那个。日志写 out_dir/base.log；
    - 加 --no-start-base 可关闭（自己先起好底盘时用）。

安全：
    - 只直发 /cmd_vel，不经过 line_follower / cmd_mux / safety_node；
    - 启动时检查 /cmd_vel 是否还有其他发布者，有就拒绝运行；
    - 反向指令**只发到 v 降到 --release-speeds 就撤回**，不会一直顶着倒车；
    - odom 速度掉到 -max-reverse-speed 以下、倒车超过 --max-back-distance、
      前进超距、odom 超时、低压、Ctrl+C，全部立即发 0 并中止；
    - 静态探针要求车后 >= 0.3 m 净空；动态段速度越高要的直线越长，0.6 m/s 建议 >= 3 m。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

WS = Path(__file__).resolve().parents[1]
LOG_ROOT = WS / "tools" / "logs"
PROBE_MOVE_EPS = 0.01   # 探针里判定“真的位移了”的阈值 m
DEFAULT_SPEEDS = [0.44, 0.50]
DEFAULT_BRAKES = [0.0, 0.25, 0.45]      # 0 = 滑行基准，>0 = 反向推力大小
SUMMARY_FIELDS = [
    "v_cmd", "brake", "release", "rep", "v_at_brake", "t_rev", "t_zero", "d_zero",
    "a_dec", "v_min", "d_reverse", "t_stop", "d_forward",
    "voltage", "status",
]


# ---------------------------------------------------------------- 数据

@dataclass
class Sample:
    t: float          # 相对"发刹车指令"的秒数（ACCEL 段为负）
    phase: str
    v_cmd: float
    v: float          # 带符号！这是本脚本和 speed_step 的关键区别
    yaw: float
    x: float
    y: float
    path: float       # 累计路径长度（绝对值）
    voltage: float


@dataclass
class Result:
    v_cmd: float
    brake: float
    release: float
    rep: int
    v_at_brake: float = 0.0
    t_rev: Optional[float] = None      # 负速度真实持续了多久
    t_zero: Optional[float] = None
    d_zero: Optional[float] = None
    a_dec: Optional[float] = None
    v_min: float = 0.0
    d_reverse: float = 0.0
    t_stop: Optional[float] = None
    d_forward: Optional[float] = None
    voltage: float = float("nan")
    status: str = "OK"
    note: str = ""
    samples: List[Sample] = field(default_factory=list, repr=False)

    def row(self) -> dict:
        r = {}
        for k in SUMMARY_FIELDS:
            v = getattr(self, k)
            if isinstance(v, float) and math.isnan(v):
                v = ""
            elif isinstance(v, float):
                v = round(v, 4)
            r[k] = v
        return r


@dataclass
class ProbeResult:
    status: str = "SKIP"
    v_min: float = 0.0
    v_peak: float = 0.0
    moved: float = 0.0
    note: str = ""


# ---------------------------------------------------------------- 小工具

def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def fmt(v: Optional[float], nd: int = 3, suffix: str = "") -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}{suffix}"


def crossing_level(points: List[Tuple[float, float, float]],
                   level: float) -> Optional[Tuple[float, float]]:
    """points=(t, v, proj)，找 v 第一次降到 level 以下的时刻/位置，线性插值。

    用 level（默认 --stop-speed）而不是严格的 0：滑行时 odom 速度是指数衰减，
    未必真的出现 <=0 的样本，卡 0 会让滑行基准整组没有数据。
    """
    prev = None
    for t, v, p in points:
        if prev is not None and prev[1] > level >= v:
            pv, pp = prev[1], prev[2]
            dv = pv - v
            f = ((pv - level) / dv) if abs(dv) > 1e-9 else 1.0
            f = max(0.0, min(1.0, f))
            return prev[0] + f * (t - prev[0]), pp + f * (p - pp)
        prev = (t, v, p)
    return None


# ---------------------------------------------------------------- 底盘

def spawn_base(log_path: Path):
    """拉起底盘 launch，输出重定向到日志；独立 session 便于整组收掉。"""
    log = open(log_path, "w", encoding="utf-8")
    cmd = (
        "source /opt/tros/humble/setup.bash >/dev/null 2>&1; "
        f"source '{WS}/install/setup.bash' >/dev/null 2>&1; "
        "exec ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py"
    )
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return proc, log


def stop_base(proc, log) -> None:
    """只收自己拉起的底盘：先 SIGINT，5 s 不退就整组 KILL。"""
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass
    if log is not None:
        try:
            log.close()
        except OSError:
            pass


def base_process_running() -> bool:
    try:
        return subprocess.run(
            ["pgrep", "-f",
             "install/turn_on_wheeltec_robot/lib/turn_on_wheeltec_robot/wheeltec_robot_node"],
            stdout=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False


def log_tail(path: Path, n: int = 8) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join("        " + x for x in lines[-n:])


# ---------------------------------------------------------------- 实车节点

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

    class BrakeNode(Node):
        def __init__(self) -> None:
            super().__init__("brake_probe")
            self.pub = None
            self.v = 0.0           # 带符号
            self.yaw = 0.0
            self.x = 0.0
            self.y = 0.0
            self.path = 0.0
            self.last_x: Optional[float] = None
            self.last_y: Optional[float] = None
            self.voltage = float("nan")
            self.voltage_seen = False
            self.last_odom = 0.0
            self.odom_seen = False
            self.create_subscription(Odometry, args.odom_topic, self._on_odom, 20)
            self.create_subscription(Float32, args.voltage_topic, self._on_voltage, 10)

        def _on_odom(self, msg) -> None:
            self.odom_seen = True
            self.last_odom = time.monotonic()
            self.v = float(msg.twist.twist.linear.x)   # 不取 abs：要看方向
            q = msg.pose.pose.orientation
            self.x = float(msg.pose.pose.position.x)
            self.y = float(msg.pose.pose.position.y)
            self.yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            if self.last_x is not None and self.last_y is not None:
                step = math.hypot(self.x - self.last_x, self.y - self.last_y)
                if math.isfinite(step) and step < 1.0:
                    self.path += step
            self.last_x, self.last_y = self.x, self.y

        def _on_voltage(self, msg) -> None:
            self.voltage = float(msg.data)
            self.voltage_seen = True

        def create_cmd_publisher(self):
            self.pub = self.create_publisher(Twist, args.cmd_topic, 10)

        def publish(self, v: float, w: float = 0.0) -> None:
            if self.pub is None:
                return
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(w)
            self.pub.publish(msg)

        def other_publishers(self) -> List[str]:
            out = []
            for info in self.get_publishers_info_by_topic(args.cmd_topic):
                if info.node_name != self.get_name():
                    out.append(f"{info.node_namespace.rstrip('/')}/{info.node_name}")
            return out

    rclpy.init(args=None)
    node = BrakeNode()

    def spin_once(sec: float = 0.002) -> None:
        rclpy.spin_once(node, timeout_sec=sec)

    def drain(sec: float, cmd: float = 0.0) -> None:
        end = time.monotonic() + sec
        node.publish(cmd)
        while time.monotonic() < end:
            spin_once(0.002)
            node.publish(cmd)
            time.sleep(0.002)

    def stop_and_hold(sec: float = 0.5) -> None:
        drain(sec, 0.0)

    def wait_odom(sec: float) -> bool:
        end = time.monotonic() + sec
        while time.monotonic() < end:
            spin_once(0.02)
            if node.odom_seen and time.monotonic() - node.last_odom < 0.5:
                return True
        return False

    def proj_of(x: float, y: float, x0: float, y0: float, yaw0: float) -> float:
        return (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)

    def safety_violation(start_yaw: float, x0: float, y0: float,
                         forward_limit: float, back_limit: float) -> Optional[str]:
        now = time.monotonic()
        if not node.odom_seen or now - node.last_odom > args.odom_timeout:
            return "ABORT_ODOM_TIMEOUT"
        if node.voltage_seen and node.voltage < args.min_voltage:
            return "ABORT_LOW_VOLTAGE"
        if abs(wrap_angle(node.yaw - start_yaw)) > math.radians(args.yaw_limit_deg):
            return "ABORT_YAW_DRIFT"
        p = proj_of(node.x, node.y, x0, y0, start_yaw)
        if p > forward_limit:
            return "ABORT_DISTANCE"
        if p < -back_limit:
            return "ABORT_REVERSE_DISTANCE"
        return None

    # ------------------------------------------------------------ 静态探针
    def run_probe() -> ProbeResult:
        res = ProbeResult()
        drain(args.settle, 0.0)
        if abs(node.v) > 0.03:
            res.status = "ABORT_NOT_STILL"
            res.note = f"起步前 v={node.v:+.3f} m/s"
            return res

        x0, y0, yaw0 = node.x, node.y, node.yaw
        t_end = time.monotonic() + args.probe_time
        status: Optional[str] = None
        while time.monotonic() < t_end:
            status = safety_violation(yaw0, x0, y0,
                                      args.max_back_distance, args.max_back_distance)
            if status:
                break
            if node.v < -args.max_reverse_speed:
                status = "ABORT_REVERSE_OVERSPEED"
                break
            node.publish(-args.probe_speed, 0.0)
            spin_once(0.001)
            res.v_min = min(res.v_min, node.v)
            res.v_peak = max(res.v_peak, node.v)
            time.sleep(0.002)
        # 回零段也要继续跟踪：固件是软停车，速度峰值可能出现在撤令之后
        t_end = time.monotonic() + 0.5
        while time.monotonic() < t_end:
            node.publish(0.0)
            spin_once(0.001)
            res.v_min = min(res.v_min, node.v)
            res.v_peak = max(res.v_peak, node.v)
            time.sleep(0.002)
        res.moved = proj_of(node.x, node.y, x0, y0, yaw0)

        if status == "ABORT_REVERSE_OVERSPEED":
            # 倒得比预期还快：说明负速度不但被接受，还挺“有劲”
            res.status = "REVERSE_OK"
            res.note = (f"负速度被接受且很猛（超过 {args.max_reverse_speed:.2f} m/s，"
                        "已立即发 0 中止）")
            return res
        if status:
            res.status = status
            res.note = "探针触发安全中止"
            return res
        if res.v_peak >= args.probe_v_threshold or res.moved >= PROBE_MOVE_EPS:
            res.status = "SIGN_INVERTED"
            res.note = f"负指令变成前进 {res.v_peak:.3f} m/s（绝不能用于刹车）"
        elif res.v_min <= -args.probe_v_threshold:
            res.status = "REVERSE_OK"
            res.note = f"负速度被接受，实测倒车 {abs(res.v_min):.3f} m/s"
        elif res.moved <= -PROBE_MOVE_EPS:
            # 位置是同一路速度积分出来的，位移为负就说明真的倒车了，
            # 只是低速段有死区/软启动，0.08 这种小指令只跑出很小的速度。
            ratio = abs(res.v_min) / args.probe_speed if args.probe_speed else 0.0
            res.status = "REVERSE_SLOW"
            res.note = (f"位置后移 {-res.moved*100:.1f} cm，负速度被接受但很弱："
                        f"指令 -{args.probe_speed:.2f} 只测到 {abs(res.v_min):.3f} m/s"
                        f"（{ratio*100:.0f}%）")
        else:
            res.status = "CLAMPED"
            res.note = "负速度被当下位机忽略（等同发 0：没位移、速度没变负）"
        return res

    # ------------------------------------------------------------ 单次对比
    def run_rep(v_cmd: float, brake: float, release: float, rep: int) -> Result:
        res = Result(v_cmd=v_cmd, brake=brake, release=release, rep=rep)
        samples: List[Sample] = []
        period = 1.0 / max(1.0, args.rate)

        # 1) 静止确认
        drain(args.settle, 0.0)
        if abs(node.v) > 0.03:
            res.status = "ABORT_NOT_STILL"
            res.note = f"起步前 v={node.v:+.3f} m/s"
            return res

        x0, y0, yaw0 = node.x, node.y, node.yaw
        path0 = node.path
        cmd_t = time.monotonic()
        phase = "ACCEL"
        brake_t: Optional[float] = None
        status: Optional[str] = None
        released = False
        still_since: Optional[float] = None
        stop_t: Optional[float] = None
        v_at_brake = 0.0

        def record(ph: str) -> None:
            samples.append(Sample(
                t=time.monotonic() - cmd_t,
                phase=ph,
                v_cmd=v_cmd if ph == "ACCEL" else (-brake if ph == "REVERSE" else 0.0),
                v=node.v,
                yaw=node.yaw,
                x=node.x,
                y=node.y,
                path=node.path - path0,
                voltage=node.voltage,
            ))

        def steer() -> float:
            if not args.heading_hold:
                return 0.0
            w = -args.heading_hold * wrap_angle(node.yaw - yaw0)
            lim = args.heading_hold_max
            return max(-lim, min(lim, w))

        def tick() -> None:
            nonlocal next_tick
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

        next_tick = cmd_t

        # 2) 加速到目标比例
        while True:
            status = safety_violation(yaw0, x0, y0,
                                      args.max_distance, args.max_back_distance)
            if status:
                break
            if time.monotonic() - cmd_t > args.timeout_reach:
                status = "ABORT_REACH_TIMEOUT"
                break
            node.publish(v_cmd, steer())
            spin_once(0.001)
            record(phase)
            if node.v >= args.reach_ratio * v_cmd:
                break
            tick()

        # 3) 可选匀速段
        if status is None and args.hold > 0.0:
            phase = "HOLD"
            hold_end = time.monotonic() + args.hold
            while time.monotonic() < hold_end:
                status = safety_violation(yaw0, x0, y0,
                                          args.max_distance, args.max_back_distance)
                if status:
                    break
                node.publish(v_cmd, steer())
                spin_once(0.001)
                record(phase)
                tick()

        # 4) 刹车：brake=0 发 0 滑行；brake>0 发 -brake 反拖，降到 release 就撤回
        if status is None:
            brake_t = time.monotonic()
            v_at_brake = node.v
            released = brake <= 0.0
            phase = "COAST" if released else "REVERSE"
            deadline = brake_t + args.timeout_brake
            while True:
                now = time.monotonic()
                status = safety_violation(yaw0, x0, y0,
                                          args.max_distance, args.max_back_distance)
                if status:
                    break
                if now > deadline:
                    status = "ABORT_BRAKE_TIMEOUT"
                    break
                if node.v < -args.max_reverse_speed:
                    status = "ABORT_REVERSE_OVERSPEED"
                    break

                if not released and node.v <= release:
                    released = True
                    phase = "HOLD_ZERO"
                    res.t_rev = now - brake_t
                    node.publish(0.0)
                if released:
                    node.publish(0.0)
                else:
                    # 刹车段不再纠偏：倒车时角速度符号语义会翻，宁可直着停
                    node.publish(-brake, 0.0)
                spin_once(0.001)
                record(phase)

                if abs(node.v) < args.stop_speed:
                    if still_since is None:
                        still_since = now
                    elif now - still_since >= args.still_hold:
                        stop_t = now
                        break
                else:
                    still_since = None
                tick()

        # 5) 收尾
        stop_and_hold(0.3)

        # 6) 指标
        proj = lambda s: proj_of(s.x, s.y, x0, y0, yaw0)  # noqa: E731
        res.v_at_brake = v_at_brake
        if brake_t is not None:
            bt = brake_t - cmd_t
            seg = [s for s in samples if s.t >= bt - 1e-9]
            res.v_min = min((s.v for s in seg), default=0.0)
            if seg:
                res.d_forward = max(proj(s) for s in seg)
                z = crossing_level([(s.t - bt, s.v, proj(s)) for s in seg],
                                   args.stop_speed)
                if z:
                    res.t_zero, res.d_zero = z
                    if res.t_zero > 1e-3:
                        res.a_dec = v_at_brake / res.t_zero
                    end_proj = proj(seg[-1])
                    res.d_reverse = max(0.0, (res.d_forward or 0.0) - end_proj)
            if stop_t is not None:
                res.t_stop = stop_t - brake_t
        res.voltage = node.voltage
        res.status = status or "OK"
        if res.status != "OK" and not res.note:
            res.note = {
                "ABORT_ODOM_TIMEOUT": "odom 超时",
                "ABORT_LOW_VOLTAGE": f"电压 {node.voltage:.2f} V < {args.min_voltage:.1f} V",
                "ABORT_YAW_DRIFT": f"偏航漂移 > {args.yaw_limit_deg:.1f}°",
                "ABORT_DISTANCE": f"前进 > {args.max_distance:.2f} m",
                "ABORT_REVERSE_DISTANCE": f"倒车 > {args.max_back_distance:.2f} m",
                "ABORT_REACH_TIMEOUT": "加速超时",
                "ABORT_BRAKE_TIMEOUT": "刹车超时，未停稳",
                "ABORT_REVERSE_OVERSPEED": f"倒车速度超过 {args.max_reverse_speed:.2f} m/s",
            }.get(res.status, "")
        if res.status == "ABORT_YAW_DRIFT":
            drift = math.degrees(abs(wrap_angle(node.yaw - yaw0)))
            res.note = (f"偏航漂移 {drift:.1f}° > {args.yaw_limit_deg:.1f}°"
                        "（车在画弧：加大 --yaw-limit-deg 或开 --heading-hold）")
        res.samples = samples
        node.publish(0.0)
        return res

    # ------------------------------------------------------------ 主流程
    out_dir = Path(args.out) if args.out else LOG_ROOT / f"brake_probe_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    meta = {
        "started_iso": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "workspace": str(WS),
    }
    results: List[Result] = []
    probe = ProbeResult()
    base_proc = None
    base_log = None

    try:
        # 0) 发布者检查
        drain(1.0, 0.0)
        others = node.other_publishers()
        if others:
            print("[ERROR] /cmd_vel 已有发布者，测试会和他们抢控制：")
            for o in others:
                print(f"        - {o}")
            print("        请先停掉巡线/仲裁/兜底（底盘驱动保持运行）：")
            print("        如果 tools/start.sh 在跑，先在它的终端 Ctrl+C")
            print("        pkill -INT -f line_follower_with_detection")
            print("        pkill -INT -f safety_node.py")
            return 2

        node.create_cmd_publisher()
        drain(1.0, 0.0)
        others = node.other_publishers()
        if others:
            print("[ERROR] /cmd_vel 仍有其他发布者：")
            for o in others:
                print(f"        - {o}")
            return 2

        # 1) 等 odom；没有就顺手把底盘拉起来（只拉底盘，不碰巡线/相机）
        if not wait_odom(2.5):
            base_log_path = out_dir / "base.log"
            if not args.start_base:
                print(f"[WARN] {args.odom_topic} 还没出数（--no-start-base，只等不启动）")
            elif base_process_running():
                print("[INFO] 底盘进程已在运行，等它把串口打开...")
            else:
                base_proc, base_log = spawn_base(base_log_path)
                print(f"[INFO] {args.odom_topic} 没数据，已自动拉起底盘 "
                      f"pid={base_proc.pid}（日志 {base_log_path}）")
            if not wait_odom(args.base_timeout):
                print(f"[ERROR] {args.odom_topic} 仍然没有数据：底盘驱动没起来，"
                      "或串口 /dev/wheeltec_controller 没打开。")
                tail = log_tail(base_log_path)
                if tail:
                    print("        底盘日志末尾：")
                    print(tail)
                print("        手动启动底盘（只起底盘，不会拉起巡线）：")
                print("          source /opt/tros/humble/setup.bash")
                print(f"          source {WS}/install/setup.bash")
                print("          ros2 launch turn_on_wheeltec_robot "
                      "turn_on_wheeltec_robot.launch.py")
                print("        另开终端确认： ros2 topic hz /odom   （应有 20+ Hz）")
                print("        若 /dev/wheeltec_controller 不存在： ls -l /dev/ttyACM*，"
                      "查 USB 线和底盘电源。")
                return 2
            print("[OK] 底盘就绪")
        if not node.voltage_seen:
            t_end = time.monotonic() + 2.0
            while time.monotonic() < t_end and not node.voltage_seen:
                spin_once(0.02)
        if not node.voltage_seen:
            print(f"[WARN] {args.voltage_topic} 没有数据，本次不检查低压。")
        elif node.voltage < args.min_voltage:
            print(f"[ERROR] 电压 {node.voltage:.2f} V < {args.min_voltage:.1f} V，先充电。")
            return 2

        print(f"[OK] odom 已就绪，电压 {node.voltage:.2f} V" if node.voltage_seen
              else "[OK] odom 已就绪")
        print(f"[INFO] 输出目录：{out_dir}")

        # 2) 静态探针
        if args.probe:
            print("\n[STEP 1/2] 静态探针：车静止，发 "
                  f"{-args.probe_speed:.2f} m/s 共 {args.probe_time:.1f} s，"
                  "看 odom 速度会不会变负。")
            print("[SAFE] 确认车后 >= 0.3 m 空、车头方向无人。")
            if not args.yes:
                try:
                    input("        准备好后按 Enter 开始探针...")
                except EOFError:
                    print("\n[STOP] 没有交互输入，结束。")
                    return 0
            probe = run_probe()
            print(f"  -> {probe.status:16s} v_min={probe.v_min:+.3f} m/s "
                  f"位移={probe.moved*100:+.1f} cm  ({probe.note})")
            if probe.status in ("ABORT_ODOM_TIMEOUT", "ABORT_LOW_VOLTAGE",
                                "ABORT_NOT_STILL", "ABORT_REVERSE_DISTANCE"):
                print("[ERROR] 探针中止，先排查底盘再继续。")
                return 3
            if args.probe_only:
                print_probe_verdict(probe)
                meta["probe"] = vars(probe)
                (out_dir / "meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
                return 0
            if probe.status == "SIGN_INVERTED":
                print("[ERROR] 负速度变成了前进——绝不能用它刹车，后续动态测试取消。")
                return 3
            if probe.status == "CLAMPED":
                print("[WARN] 负速度大概率无效，反拖档预计和滑行基准一样；"
                      "跑 1 组确认后可以 Ctrl+C 提前结束。")
            elif probe.status == "REVERSE_SLOW":
                print("[INFO] 负速度有效但低速段很弱（正常现象：小指令进死区）。"
                      "动态段会直接用 -0.25 / -0.45 量真实减速度。")
        else:
            print("[WARN] --no-probe：跳过静态探针，直接对比刹车"
                  "（负速度行为未知，第一组务必低速、小反向）")

        # 3) 动态对比
        runs = len(args.speeds) * len(args.brakes) * args.reps
        print(f"\n[STEP 2/2] 对比刹车：{len(args.speeds)} 速度 x "
              f"{len(args.brakes)} 刹车档 x {args.reps} 次 = {runs} 组，"
              f"预计前进 {runs * 1.2:.1f} m。")
        print("[SAFE] 确认：车在直线起点、前方 >= 2.5 m 净空、车头方向无人。")
        if not args.yes:
            for i in range(5, 0, -1):
                print(f"       {i} 秒后开始...", end="\r", flush=True)
                time.sleep(1.0)
            print()

        with open(summary_path, "w", newline="", encoding="utf-8") as fsum:
            writer = csv.DictWriter(fsum, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            fsum.flush()

            # 滑行基准（brake=0）与撤回阈值无关，只跑一组，别重复触发
            plan = []
            for v_cmd in args.speeds:
                for brake in args.brakes:
                    releases = [0.0] if brake <= 0.0 else list(args.release_speeds)
                    for release in releases:
                        plan.append((v_cmd, brake, release))

            stop_all = False
            for v_cmd, brake, release in plan:
                if stop_all:
                    break
                for rep in range(1, args.reps + 1):
                    others = node.other_publishers()
                    if others:
                        print("[ERROR] 检测到新的 /cmd_vel 发布者，中止：")
                        for o in others:
                            print(f"        - {o}")
                        stop_all = True
                        break
                    if not args.yes:
                        tag = "滑行" if brake <= 0.0 else f"-{brake:.2f}/撤回{release:.2f}"
                        try:
                            input(f"[V={v_cmd:.2f} {tag} "
                                  f"第 {rep}/{args.reps} 次] "
                                  "把车推回起点，按 Enter 开始...")
                        except EOFError:
                            print("\n[STOP] 没有交互输入，结束。")
                            stop_all = True
                            break
                    res = run_rep(v_cmd, brake, release, rep)
                    raw = out_dir / f"raw_{v_cmd:.2f}_{brake:.2f}_{release:.2f}_{rep}.csv"
                    with open(raw, "w", newline="", encoding="utf-8") as fraw:
                        wraw = csv.DictWriter(fraw, fieldnames=[
                            "t", "phase", "v_cmd", "v", "yaw", "x", "y",
                            "path", "voltage"])
                        wraw.writeheader()
                        for s in res.samples:
                            wraw.writerow({
                                "t": round(s.t, 4), "phase": s.phase,
                                "v_cmd": round(s.v_cmd, 4),
                                "v": round(s.v, 4),
                                "yaw": round(s.yaw, 4),
                                "x": round(s.x, 4), "y": round(s.y, 4),
                                "path": round(s.path, 4),
                                "voltage": round(s.voltage, 3)
                                if math.isfinite(s.voltage) else "",
                            })
                    results.append(res)
                    writer.writerow(res.row())
                    fsum.flush()
                    print(f"  -> {res.status:24s} "
                          f"rel={release:4.2f} "
                          f"反拖={fmt(res.t_rev, 2, 's'):>6} "
                          f"v={fmt(res.v_at_brake, 2):>5} "
                          f"t0={fmt(res.t_zero, 3, 's'):>7} "
                          f"d0={fmt(res.d_zero, 3, 'm'):>7} "
                          f"a={fmt(res.a_dec, 2, 'm/s2'):>10} "
                          f"vmin={res.v_min:+.3f} "
                          f"倒车={res.d_reverse*100:4.1f}cm"
                          + (f"  ({res.note})" if res.note else ""))
                    if res.status != "OK":
                        stop_all = True
                        break

        print_probe_verdict(probe)
        print_summary(results, out_dir, args)
        meta["probe"] = vars(probe)
        meta["results"] = [r.row() for r in results]
        (out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0 if results and all(r.status == "OK" for r in results) else 1
    finally:
        try:
            stop_and_hold(0.5)
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
        if base_proc is not None:
            try:
                stop_base(base_proc, base_log)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------- 汇总

PROBE_TEXT = {
    "REVERSE_OK": "负速度被下位机接受，能真的倒着走 -> 反向推力刹车可行",
    "CLAMPED": "负速度被下位机忽略，效果等同发 0 -> 反向推力刹车不可用",
    "SIGN_INVERTED": "负速度被当成前进 -> 危险，绝不能用于刹车",
    "REVERSE_SLOW": "负速度被接受（车确实在倒），但低速段死区/软启动明显，实际速度远小于指令",
    "ABORT_NOT_STILL": "探针前车没停稳",
    "ABORT_ODOM_TIMEOUT": "探针中 odom 断了",
    "ABORT_LOW_VOLTAGE": "探针中电压过低",
    "ABORT_REVERSE_DISTANCE": "探针倒车超限",
    "SKIP": "未做静态探针",
}


def print_probe_verdict(probe: ProbeResult) -> None:
    print("\n" + "=" * 76)
    print("静态探针结论")
    print("-" * 76)
    print(f"  状态   : {probe.status}")
    print(f"  含义   : {PROBE_TEXT.get(probe.status, probe.status)}")
    print(f"  实测   : v_min={probe.v_min:+.3f} m/s, v_peak={probe.v_peak:+.3f} m/s, "
          f"位移={probe.moved*100:+.1f} cm")
    print("=" * 76)


def print_summary(results: List[Result], out_dir: Path, args) -> None:
    ok = [r for r in results if r.status == "OK" and r.d_zero is not None]
    tol_d = args.max_rollback
    tol_v = args.max_rollback_v

    def still(group) -> bool:
        """急停判据：停住之后不后退。"""
        rev = sum(x.d_reverse for x in group) / len(group)
        vmin = sum(x.v_min for x in group) / len(group)
        return rev <= tol_d and vmin >= -tol_v

    print("\n" + "=" * 106)
    print(f"{'v_cmd':>6} {'brake':>6} {'rel':>5} {'n':>3} {'v_at':>6} {'t_rev':>6} "
          f"{'t_zero':>7} {'d_zero':>7} {'a_dec':>7} {'v_min':>7} {'倒车cm':>7} "
          f"{'判定':>4} {'缩短':>6}")
    print("-" * 113)

    base = {}
    for v in sorted({r.v_cmd for r in results}):
        sub0 = [r for r in ok if abs(r.v_cmd - v) < 1e-6 and r.brake <= 0.0]
        base[v] = median([r.d_zero for r in sub0])

    for v in sorted({r.v_cmd for r in results}):
        for b in sorted({r.brake for r in results if abs(r.v_cmd - v) < 1e-6}):
            rels = sorted({r.release for r in results
                           if abs(r.v_cmd - v) < 1e-6 and abs(r.brake - b) < 1e-9})
            for rel in rels:
                sub = [r for r in ok if abs(r.v_cmd - v) < 1e-6
                       and abs(r.brake - b) < 1e-9 and abs(r.release - rel) < 1e-9]
                if not sub:
                    print(f"{v:6.2f} {b:6.2f} {rel:5.2f} {0:3d}   全部失败/中止")
                    continue
                avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
                d = avg([r.d_zero for r in sub])
                gain = ""
                if base.get(v) and b > 0.0:
                    gain = f"{(1 - d / base[v]) * 100:+.0f}%"
                trev = avg([r.t_rev for r in sub if r.t_rev is not None]) \
                    if any(r.t_rev is not None for r in sub) else None
                print(f"{v:6.2f} {b:6.2f} {rel:5.2f} {len(sub):3d} "
                      f"{avg([r.v_at_brake for r in sub]):6.2f} "
                      f"{(f'{trev:.3f}' if trev is not None else '-'):>6} "
                      f"{avg([r.t_zero for r in sub]):7.3f} "
                      f"{d:7.3f} "
                      f"{avg([r.a_dec or 0 for r in sub]):7.2f} "
                      f"{avg([r.v_min for r in sub]):+7.3f} "
                      f"{avg([r.d_reverse for r in sub])*100:7.1f} "
                      f"{'OK' if still(sub) else '过冲':>4} "
                      f"{gain:>6}")
    print("=" * 113)
    print("注：t_rev = 负速度实际持续时长（发 -B 到 v 降到 release 阈值）。")
    print("    t_zero/d_zero = 从发刹车指令到速度降到 --stop-speed 的时间/距离。")
    print(f"    判定 OK = 停住不后退（倒车 <= {tol_d*100:.0f} cm 且 "
          f"v_min >= -{tol_v:.2f} m/s）。")

    if not ok:
        print("没有一组 OK，先排查 odom/电压/是否还有别的 /cmd_vel 发布者。")
        print(f"[out] {out_dir / 'summary.csv'}")
        return

    lines = []
    for v in sorted({r.v_cmd for r in ok}):
        b0 = base.get(v)
        if not b0:
            continue
        cands = []
        for b in sorted({r.brake for r in ok if abs(r.v_cmd - v) < 1e-6 and r.brake > 0.0}):
            for rel in sorted({r.release for r in ok
                               if abs(r.v_cmd - v) < 1e-6 and abs(r.brake - b) < 1e-9}):
                sub = [r for r in ok if abs(r.v_cmd - v) < 1e-6
                       and abs(r.brake - b) < 1e-9 and abs(r.release - rel) < 1e-9]
                if not sub:
                    continue
                d = sum(r.d_zero for r in sub) / len(sub)
                rev = sum(r.d_reverse for r in sub) / len(sub)
                vmin = sum(r.v_min for r in sub) / len(sub)
                cands.append((d, b, rel, rev, vmin))
        if not cands:
            continue
        good = [c for c in cands if c[3] <= tol_d and c[4] >= -tol_v]
        if good:
            d, b, rel, rev, vmin = min(good)
            lines.append(f"  V={v:.2f}: -{b:.2f} 反拖，v 降到 {rel:.2f} 撤回 -> 停住 "
                         f"{d:.3f} m（滑行 {b0:.3f} m，省 {(1-d/b0)*100:.0f}%），"
                         f"倒车 {rev*100:.1f} cm，v_min {vmin:+.3f} m/s")
        else:
            d, b, rel, rev, vmin = min(cands)
            lines.append(f"  V={v:.2f}: 没有一组做到「停住不后退」"
                         f"（最好 -{b:.2f}/rel={rel:.2f} 仍倒 {rev*100:.1f} cm、"
                         f"v_min {vmin:+.3f}）；把 --release-speeds 调到 0.15/0.20/0.25 再扫")
    if lines:
        print("\n急停建议：")
        for s in lines:
            print(s)
    print(f"[out] {out_dir / 'summary.csv'}")


# ---------------------------------------------------------------- dry-run

def print_dry_run(args: argparse.Namespace) -> None:
    print("=" * 76)
    print("负速度刹车探针（dry-run，只打印，不发任何 /cmd_vel）")
    print("=" * 76)
    print("静态探针 : " + ("关闭" if not args.probe else
                          f"发 -{args.probe_speed:.2f} m/s 持续 {args.probe_time:.1f} s，"
                          f"判定阈值 {args.probe_v_threshold:.3f} m/s"))
    print(f"速度档   : {', '.join(f'{v:.2f}' for v in args.speeds)} m/s")
    print(f"刹车档   : " + ", ".join(
        ("0（滑行基准）" if b == 0 else f"-{b:.2f}") for b in args.brakes))
    n_runs = 0
    for b in args.brakes:
        n_runs += args.reps * (1 if b <= 0 else len(args.release_speeds))
    n_runs *= len(args.speeds)
    print(f"每档次数 : {args.reps}  共 {n_runs} 组"
          f"（滑行基准每组速度只跑 1 组）")
    print(f"加速判定 : odom 达到 {args.reach_ratio*100:.0f}% 目标速度后开始刹")
    print("反向撤回 : v 降到 "
          + " / ".join(f"{r:.2f}" for r in args.release_speeds)
          + " m/s 立即改发 0（逐个扫，找停住且不后退的点）")
    print(f"急停判据 : 倒车 <= {args.max_rollback*100:.0f} cm 且 "
          f"v_min >= -{args.max_rollback_v:.2f} m/s")
    print(f"停稳判定 : |v| < {args.stop_speed:.2f} m/s 持续 {args.still_hold:.2f} s")
    print(f"安全阈值 : 倒车速度 > {args.max_reverse_speed:.2f} / 前进 > "
          f"{args.max_distance:.2f} m / 倒车 > {args.max_back_distance:.2f} m / "
          f"电压 < {args.min_voltage:.1f} V")
    print(f"记录     : {args.rate:.0f} Hz，odom 速度**带符号**")
    print("底盘     : " + ("/odom 没数据时自动拉起底盘 launch（已在跑就复用）"
                          if args.start_base else "不自动启动，需要自己先起底盘"))
    print("\n单次时序：")
    print("  0.0 s   发 0，静止确认")
    print("  ...     发 +V 加速")
    print(f"  ...     v >= {args.reach_ratio:.2f}*V（或匀速 {args.hold:.1f} s 后）")
    print("  ...     brake=0 发 0；brake>0 发 -brake")
    print("  ...     反拖档在 v 降到 release 阈值时改发 0（每个阈值各跑一组）")
    print(f"  ...     |v| < {args.stop_speed:.2f} 持续 {args.still_hold:.2f} s 结束")
    print("\n跑之前必须做：")
    print("  0) 如果 tools/start.sh 在跑，先在它的终端 Ctrl+C")
    print("  1) pkill -INT -f line_follower_with_detection")
    print("     pkill -INT -f safety_node.py")
    print("  2) 底盘没起也没关系：脚本会自动拉起来（日志 base.log）；"
          "也可用 --no-start-base 自己先起")
    print("  3) 车放直线起点，前方 >= 2.5 m；静态探针要车后 >= 0.3 m")
    print("  4) 先只跑探针： --probe-only --yes --speeds 0.30 --brakes 0.0")
    print("\n正式命令：")
    print("  source /opt/tros/humble/setup.bash")
    print("  python3 tools/brake_probe.py --speeds "
          + " ".join(f"{v:.2f}" for v in args.speeds)
          + " --brakes " + " ".join(f"{b:.2f}" for b in args.brakes)
          + " --release-speeds " + " ".join(f"{r:.2f}" for r in args.release_speeds)
          + f" --reps {args.reps}")
    print("=" * 76)


# ---------------------------------------------------------------- CLI

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="负速度刹车探针：验证下位机是否接受负 linear.x，并对比滑行/反拖刹车距离",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS,
                    help="刹车前的速度档，m/s")
    ap.add_argument("--brakes", type=float, nargs="+", default=DEFAULT_BRAKES,
                    help="刹车强度：0=发 0 滑行，>0=发 -brake 反拖")
    ap.add_argument("--reps", type=int, default=2, help="每档次数")
    ap.add_argument("--reach-ratio", type=float, default=0.92,
                    help="达到目标速度的多少比例就开始刹")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="达到速度后先匀速多久再刹（0=立刻刹）")
    ap.add_argument("--probe", action=argparse.BooleanOptionalAction, default=True,
                    help="先做静态负速度探针")
    ap.add_argument("--probe-only", action="store_true", help="只做静态探针")
    ap.add_argument("--probe-speed", type=float, default=0.08,
                    help="静态探针发的负速度大小 m/s")
    ap.add_argument("--probe-time", type=float, default=0.8, help="静态探针持续时间 s")
    ap.add_argument("--probe-v-threshold", type=float, default=0.02,
                    help="判定 odom 速度真的变负的阈值 m/s")
    ap.add_argument("--release-speeds", type=float, nargs="+", default=[0.05],
                    help="反拖撤回阈值：v 降到该值立刻改发 0。可一次扫多个，"
                         "太低会顶成倒车，太高浪费刹车距离")
    ap.add_argument("--max-rollback", type=float, default=0.02,
                    help="判定「停住不后退」允许的倒车距离 m")
    ap.add_argument("--max-rollback-v", type=float, default=0.05,
                    help="判定「停住不后退」允许的反向速度峰值 m/s")
    ap.add_argument("--max-reverse-speed", type=float, default=0.20,
                    help="允许的最大倒车速度，超过立即中止（调急停时别设太小，"
                         "否则过冲数据会被中止掉）")
    ap.add_argument("--max-back-distance", type=float, default=0.20,
                    help="允许的最大倒车位移，超过立即中止")
    ap.add_argument("--max-distance", type=float, default=2.0,
                    help="单次允许的最大前进位移，超过立即中止")
    ap.add_argument("--settle", type=float, default=0.5, help="起步前静止确认时长 s")
    ap.add_argument("--rate", type=float, default=30.0, help="记录频率 Hz")
    ap.add_argument("--stop-speed", type=float, default=0.02, help="停稳速度阈值 m/s")
    ap.add_argument("--still-hold", type=float, default=0.3, help="停稳保持时长 s")
    ap.add_argument("--timeout-reach", type=float, default=2.5, help="加速超时 s")
    ap.add_argument("--timeout-brake", type=float, default=4.0, help="刹车超时 s")
    ap.add_argument("--min-voltage", type=float, default=11.0, help="低压中止阈值 V")
    ap.add_argument("--odom-timeout", type=float, default=0.3, help="odom 超时 s")
    ap.add_argument("--yaw-limit-deg", type=float, default=8.0,
                    help="相对起点的偏航漂移上限")
    ap.add_argument("--wait-timeout", type=float, default=8.0, help="等 odom 的超时 s")
    ap.add_argument("--start-base", action=argparse.BooleanOptionalAction, default=True,
                    help="/odom 没数据时自动拉起底盘 launch（已在跑就复用，退出时只收自己拉起的）")
    ap.add_argument("--base-timeout", type=float, default=20.0,
                    help="等底盘串口打开/出 odom 的超时 s")
    ap.add_argument("--cmd-topic", default="/cmd_vel")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--voltage-topic", default="/PowerVoltage")
    ap.add_argument("--heading-hold", type=float, default=0.0, metavar="K",
                    help="加速段偏航保持增益：angular.z=-K*(yaw-yaw0)，0=关闭")
    ap.add_argument("--heading-hold-max", type=float, default=0.15,
                    help="偏航保持输出上限 rad/s")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认 tools/logs/brake_probe_<时间>）")
    ap.add_argument("--yes", action="store_true", help="跳过倒计时和每次的 Enter 确认")
    ap.add_argument("--dry-run", action="store_true", help="只打印时序，不发指令")
    args = ap.parse_args(argv)

    if not args.speeds:
        ap.error("--speeds 不能为空")
    if any(v <= 0 or v > 1.0 for v in args.speeds):
        ap.error("--speeds 必须在 (0, 1.0] m/s")
    if any(b < 0 or b > 0.8 for b in args.brakes):
        ap.error("--brakes 必须在 [0, 0.8] m/s（0=滑行，>0=反拖大小）")
    if args.reps < 1:
        ap.error("--reps 必须 >= 1")
    if not (0.5 <= args.reach_ratio <= 1.0):
        ap.error("--reach-ratio 必须在 0.5–1.0")
    if any(rel <= args.stop_speed for rel in args.release_speeds):
        ap.error("--release-speeds 必须都 > --stop-speed，否则反拖会顶到倒车")
    if len(args.release_speeds) > 4:
        ap.error("--release-speeds 最多 4 个，否则组数太多、要来回推车十几次")
    if args.probe_speed > 0.30:
        ap.error("--probe-speed 太大：倒车位移约 = 速度 x 时间，别超过 0.30")
    if args.max_distance <= 0.3:
        ap.error("--max-distance 太小")
    if args.probe and args.probe_time < 0.2:
        ap.error("--probe-time 太短，测不出速度响应")
    # 不自动补滑行基准：只想单独测一组反拖时，多跑一组反而是负担
    if not any(b == 0.0 for b in args.brakes):
        print("[INFO] --brakes 里没有 0（滑行基准）：不自动补跑，"
              "表格里“缩短”列会是空的")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print_dry_run(args)
        return 0
    try:
        return run_real(args)
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C，已发零速退出。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
