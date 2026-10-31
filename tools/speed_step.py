#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实车速度阶跃 + 立刻制动测试。

目的：回答两个问题
    1) 从静止加速到 V，再立刻发 0，直到停稳，一共走多远、花多久？
    2) 在 0.5 / 0.8 / 1.2 m 的可用距离内，最高能以多少速度安全停住？

用法（先 source /opt/tros/humble/setup.bash）：
    # 只看时序，不发任何指令
    python3 tools/speed_step.py --dry-run

    # 实车：先停掉巡线/仲裁/兜底，只保留底盘驱动
    python3 tools/speed_step.py \
        --speeds 0.44 0.47 0.50 0.53 0.55 --reps 3 \
        --reach-ratio 0.90 --max-distance 2.0

    # 每跑完一次用卷尺量实际位移并当场录入（标定 odom 比例）
    python3 tools/speed_step.py --speeds 0.50 --reps 3 --ask-tape

    # 所有 reps 都停在同一根卷尺终点线上，直接给同一个实测值
    python3 tools/speed_step.py --speeds 0.50 --reps 3 --tape-distance 1.82

摄像头：
    - 默认随测试一起拉起 src/camera/camera.py，确认 /image_raw 出图才开始；
      已有相机在跑就直接复用，不会重复打开 /dev/video0。日志写 out_dir/camera.log；
    - 退出时（含异常/Ctrl+C）把这个子进程一起收掉；
    - 加 --no-camera 可关闭。

走直线：
    - 摄像头本身只发图像，让车走直的是上游视觉闭环；测试独占 /cmd_vel 时它没法参与。
    - 舵机中位有偏移时，纯 angular.z=0 会让车慢慢画弧。用 --heading-hold K
      按偏航误差反向补一点角速度，把车压在直线上（默认关闭）。

安全：
    - 只直接发 /cmd_vel，不经过 line_follower / cmd_mux；
    - 启动时检查 /cmd_vel 是否还有其他发布者，有就拒绝运行；
    - 超距、odom 超时、电压过低、偏航漂移过大都会立即发 0 并中止；
    - Ctrl+C 连续发 0 约 0.5 s 再退出；
    - 测试需要至少 2.5–3 m 直线净空，车头方向不要站人。
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
CAMERA_SCRIPT = WS / "src" / "camera" / "camera.py"
DEFAULT_SPEEDS = [0.44, 0.47, 0.50, 0.53, 0.55]
SUMMARY_FIELDS = [
    "v_cmd", "rep", "v_peak", "v_at_brake", "v_ss",
    "t90", "d90", "t95", "d95",
    "t_stop", "d_stop", "t_total", "d_total",
    "a_dec_eff", "a_dec_fit", "t_lag",
    "voltage", "odom_scale", "tape_distance", "status",
]


# ---------------------------------------------------------------- 数据

@dataclass
class Sample:
    t: float          # 相对“发目标速度”的秒数
    phase: str
    v_cmd: float
    v: float
    yaw: float
    x: float
    y: float
    d_path: float     # 从本次起步开始累计的路径长度
    voltage: float


@dataclass
class RepResult:
    v_cmd: float
    rep: int
    v_peak: float = 0.0
    v_at_brake: float = 0.0
    v_ss: float = 0.0
    t90: Optional[float] = None
    d90: Optional[float] = None
    t95: Optional[float] = None
    d95: Optional[float] = None
    t_stop: Optional[float] = None
    d_stop: Optional[float] = None
    t_total: Optional[float] = None
    d_total: Optional[float] = None
    a_dec_eff: Optional[float] = None
    a_dec_fit: Optional[float] = None
    t_lag: Optional[float] = None
    voltage: float = float("nan")
    odom_scale: Optional[float] = None
    tape_distance: Optional[float] = None
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


# ---------------------------------------------------------------- 小工具

def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def linear_fit(xs: List[float], ys: List[float]) -> Optional[float]:
    """最小二乘斜率；样本不足或 x 无变化返回 None。"""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-9:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def first_crossing(samples: List[Sample], threshold: float) -> Optional[Tuple[float, float]]:
    """找 v 第一次 >= threshold 的时刻/路径；相邻两点线性插值。"""
    for i, s in enumerate(samples):
        if s.v < threshold:
            continue
        if i == 0:
            return s.t, s.d_path
        p = samples[i - 1]
        dv = s.v - p.v
        f = (threshold - p.v) / dv if abs(dv) > 1e-9 else 0.0
        f = max(0.0, min(1.0, f))
        return p.t + f * (s.t - p.t), p.d_path + f * (s.d_path - p.d_path)
    return None


def fmt(v: Optional[float], nd: int = 3, suffix: str = "") -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}{suffix}"


def stop_speed_for_distance(distance: float, a_dec: float, t_lag: float,
                            margin: float = 0.10) -> float:
    """给定可用距离 D，求还能停住的最大初速度。

        D = v*t_lag + v^2/(2a) + margin
        v = -a*t_lag + sqrt((a*t_lag)^2 + 2a(D-margin))
    """
    usable = distance - margin
    if usable <= 0.0 or a_dec <= 0.0:
        return 0.0
    return -a_dec * t_lag + math.sqrt((a_dec * t_lag) ** 2 + 2.0 * a_dec * usable)


def speed_from_rest_for_distance(distance: float, a_dec: float, t_lag: float,
                                 tau_acc: float, margin: float = 0.10) -> float:
    """从静止起步，到黄线距离 D，能加速到并停住的最大速度。

    含加速段：一阶响应到 v 约走 v*tau_acc，刹车段 v*t_lag + v^2/(2a)。
        v^2/(2a) + (tau_acc+t_lag)v = D - margin
    """
    usable = distance - margin
    if usable <= 0.0 or a_dec <= 0.0:
        return 0.0
    b = max(0.0, tau_acc) + max(0.0, t_lag)
    return -a_dec * b + math.sqrt((a_dec * b) ** 2 + 2.0 * a_dec * usable)


# ---------------------------------------------------------------- 摄像头

def spawn_camera(log_path: Path):
    """按 tools/start.sh 的方式拉起 camera.py，输出重定向到日志文件。"""
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(CAMERA_SCRIPT)],
        cwd=str(CAMERA_SCRIPT.parent),
        stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return proc, log


def stop_camera(proc, log) -> None:
    """SIGINT 让 camera.py 走 finally 释放 /dev/video0，3 s 不退就 KILL。"""
    if proc is not None and proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass
    if log is not None:
        try:
            log.close()
        except OSError:
            pass


def camera_process_running() -> bool:
    try:
        return subprocess.run(["pgrep", "-f", "camera.py"],
                              stdout=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False


def camera_log_tail(path: Path, n: int = 6) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join("        " + s for s in lines[-n:])


# ---------------------------------------------------------------- 实车节点

def run_real(args: argparse.Namespace) -> int:
    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float32
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 导入 ROS 失败：{exc}", file=sys.stderr)
        print("        请先执行： source /opt/tros/humble/setup.bash", file=sys.stderr)
        return 2

    class SpeedStepNode(Node):
        def __init__(self) -> None:
            super().__init__("speed_step_test")
            self.pub = None
            self.v = 0.0
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
            self.img_count = 0
            self.last_img = 0.0
            self.create_subscription(Odometry, args.odom_topic, self._on_odom, 20)
            self.create_subscription(Float32, args.voltage_topic, self._on_voltage, 10)
            self.create_subscription(Image, args.image_topic, self._on_image, 5)

        def _on_odom(self, msg) -> None:
            self.odom_seen = True
            self.last_odom = time.monotonic()
            self.v = abs(float(msg.twist.twist.linear.x))
            q = msg.pose.pose.orientation
            self.x = float(msg.pose.pose.position.x)
            self.y = float(msg.pose.pose.position.y)
            self.yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            if self.last_x is not None and self.last_y is not None:
                step = math.hypot(self.x - self.last_x, self.y - self.last_y)
                if math.isfinite(step) and step < 1.0:  # 单帧跳变不计
                    self.path += step
            self.last_x, self.last_y = self.x, self.y

        def _on_voltage(self, msg) -> None:
            self.voltage = float(msg.data)
            self.voltage_seen = True

        def _on_image(self, msg) -> None:
            # 不解码，只用来确认相机真的在出图
            self.img_count += 1
            self.last_img = time.monotonic()

        def image_alive(self, timeout: float = 1.0) -> bool:
            return self.last_img > 0.0 and time.monotonic() - self.last_img < timeout

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
    node = SpeedStepNode()
    results: List[RepResult] = []

    def spin_once(sec: float = 0.002) -> None:
        rclpy.spin_once(node, timeout_sec=sec)

    def drain(sec: float, cmd: float = 0.0) -> None:
        """保持发 cmd 一段时间，同时继续处理回调。"""
        end = time.monotonic() + sec
        node.publish(cmd)
        while time.monotonic() < end:
            spin_once(0.002)
            node.publish(cmd)
            time.sleep(0.002)

    def stop_and_hold(sec: float = 0.5) -> None:
        drain(sec, 0.0)

    def safety_violation(start_yaw: float, start_path: float) -> Optional[str]:
        now = time.monotonic()
        if not node.odom_seen or now - node.last_odom > args.odom_timeout:
            return "ABORT_ODOM_TIMEOUT"
        if node.voltage_seen and node.voltage < args.min_voltage:
            return "ABORT_LOW_VOLTAGE"
        if abs(wrap_angle(node.yaw - start_yaw)) > math.radians(args.yaw_limit_deg):
            return "ABORT_YAW_DRIFT"
        if node.path - start_path > args.max_distance:
            return "ABORT_DISTANCE"
        return None

    def run_rep(v_cmd: float, rep: int) -> RepResult:
        res = RepResult(v_cmd=v_cmd, rep=rep)
        samples: List[Sample] = []

        # 1) 静止确认
        drain(args.settle, 0.0)
        if node.v > 0.03:
            res.status = "ABORT_NOT_STILL"
            res.note = f"起步前速度 {node.v:.3f} m/s > 0.03"
            return res

        start_x, start_y = node.x, node.y
        start_yaw = node.yaw
        path0 = node.path
        cmd_t = time.monotonic()
        phase = "ACCEL"
        status: Optional[str] = None
        brake_t: Optional[float] = None
        v_at_brake = 0.0
        still_since: Optional[float] = None
        stop_done_t: Optional[float] = None
        period = 1.0 / max(1.0, args.rate)
        next_tick = cmd_t

        def record(ph: str) -> None:
            samples.append(Sample(
                t=time.monotonic() - cmd_t,
                phase=ph,
                v_cmd=v_cmd if ph in ("ACCEL", "HOLD") else 0.0,
                v=node.v,
                yaw=node.yaw,
                x=node.x,
                y=node.y,
                d_path=node.path - path0,
                voltage=node.voltage,
            ))

        def steer() -> float:
            """偏航保持：抵消舵机中位偏移造成的画弧，把车压在直线上。"""
            if not args.heading_hold:
                return 0.0
            w = -args.heading_hold * wrap_angle(node.yaw - start_yaw)
            lim = args.heading_hold_max
            return max(-lim, min(lim, w))

        # 2) 加速到目标比例
        reach_t: Optional[float] = None
        while True:
            now = time.monotonic()
            status = safety_violation(start_yaw, path0)
            if status:
                break
            if now - cmd_t > args.timeout_reach:
                status = "ABORT_REACH_TIMEOUT"
                break
            node.publish(v_cmd, steer())
            spin_once(0.001)
            record(phase)
            if node.v >= args.reach_ratio * v_cmd:
                reach_t = time.monotonic()
                break
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

        # 3) 可选匀速段，测稳态
        if status is None and reach_t is not None and args.hold > 0.0:
            phase = "HOLD"
            hold_end = time.monotonic() + args.hold
            while time.monotonic() < hold_end:
                status = safety_violation(start_yaw, path0)
                if status:
                    break
                node.publish(v_cmd, steer())
                spin_once(0.001)
                record(phase)
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()

        # 4) 立刻发 0，直到停稳
        if status is None:
            phase = "BRAKE"
            brake_t = time.monotonic()
            v_at_brake = node.v
            path_brake = node.path
            node.publish(0.0)
            while True:
                now = time.monotonic()
                status = safety_violation(start_yaw, path0)
                if status:
                    break
                if now - brake_t > args.timeout_stop:
                    status = "ABORT_STOP_TIMEOUT"
                    break
                node.publish(0.0)
                spin_once(0.001)
                record(phase)
                if node.v < args.stop_speed:
                    if still_since is None:
                        still_since = now
                    elif now - still_since >= args.still_hold:
                        stop_done_t = now
                        break
                else:
                    still_since = None
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()

        # 5) 收尾：确保零速，再采一小段
        stop_and_hold(0.3)
        final_t = time.monotonic()
        final_path = node.path - path0
        final_net = math.hypot(node.x - start_x, node.y - start_y)

        if samples:
            res.v_peak = max(s.v for s in samples)
        res.v_at_brake = v_at_brake
        if brake_t is not None:
            res.t_stop = (stop_done_t - brake_t) if stop_done_t else None
            if stop_done_t:
                # 取停车时刻之前的路径点
                d_at_stop = None
                for s in samples:
                    if s.t <= stop_done_t - cmd_t + 1e-6:
                        d_at_stop = s.d_path
                if d_at_stop is not None:
                    res.d_stop = max(0.0, d_at_stop - (final_path - final_net))
        if reach_t is not None:
            c90 = first_crossing(samples, args.reach_ratio * v_cmd)
            if c90:
                res.t90, res.d90 = c90
            c95 = first_crossing(samples, 0.95 * v_cmd)
            if c95:
                res.t95, res.d95 = c95
        res.t_total = stop_done_t - cmd_t if stop_done_t else final_t - cmd_t
        res.d_total = final_path
        if args.hold > 0:
            tail = [s.v for s in samples if s.phase == "HOLD"]
            if tail:
                res.v_ss = sum(tail) / len(tail)
        if res.v_ss <= 0.0:
            res.v_ss = v_at_brake
        if res.t_stop and res.t_stop > 0 and v_at_brake > 0:
            res.a_dec_eff = v_at_brake / res.t_stop

        # 减速段线性拟合：v = v0 - a*t
        brake_samples = [s for s in samples
                         if s.phase == "BRAKE" and s.t >= (brake_t - cmd_t if brake_t else 0)]
        if len(brake_samples) >= 3:
            slope = linear_fit([s.t - (brake_t - cmd_t) for s in brake_samples],
                               [s.v for s in brake_samples])
            if slope is not None and slope < 0:
                res.a_dec_fit = -slope
        # 发 0 到速度开始明显下降的滞后：第一次低于 95% v_at_brake
        if brake_t is not None and v_at_brake > 0.02:
            for s in brake_samples:
                if s.v < 0.95 * v_at_brake:
                    res.t_lag = s.t - (brake_t - cmd_t)
                    break

        res.voltage = node.voltage
        res.status = status or "OK"
        if res.status != "OK" and not res.note:
            res.note = {
                "ABORT_ODOM_TIMEOUT": "odom 超时",
                "ABORT_LOW_VOLTAGE": f"电压 {node.voltage:.2f} V < {args.min_voltage:.1f} V",
                "ABORT_YAW_DRIFT": f"偏航漂移 > {args.yaw_limit_deg:.1f}°",
                "ABORT_DISTANCE": f"路径 > {args.max_distance:.2f} m",
                "ABORT_REACH_TIMEOUT": "加速超时，未到目标比例",
                "ABORT_STOP_TIMEOUT": "减速超时，未停稳",
            }.get(res.status, "")
        res.samples = samples
        node.publish(0.0)

        # 卷尺标定：先发 0 再阻塞等人量，量完回车录入
        res.tape_distance = args.tape_distance or None
        if args.ask_tape:
            try:
                txt = input("  卷尺实测位移/m（回车跳过）: ").strip()
            except EOFError:
                txt = ""
            if txt:
                try:
                    res.tape_distance = float(txt)
                except ValueError:
                    print("  [WARN] 不是数字，本次不记卷尺值")
        if res.tape_distance and res.d_total:
            res.odom_scale = res.d_total / res.tape_distance
        return res

    out_dir = Path(args.out) if args.out else LOG_ROOT / f"speed_step_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    meta = {
        "started_iso": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "workspace": str(WS),
    }

    cam_log_path = out_dir / "camera.log"
    cam_proc = None
    cam_log = None

    try:
        # 0) 摄像头：视觉链路是这辆车能走直线的前提，默认随测试一起拉起
        if args.camera:
            # 先等一会儿 ROS 发现，避免把已经在跑的相机又拉一份抢 /dev/video0
            t_end = time.monotonic() + 1.0
            while time.monotonic() < t_end:
                spin_once(0.02)
            if node.count_publishers(args.image_topic) > 0 or camera_process_running():
                print(f"[INFO] {args.image_topic} 已有相机在跑，直接复用")
            else:
                if not CAMERA_SCRIPT.exists():
                    print(f"[ERROR] 找不到 {CAMERA_SCRIPT}")
                    return 2
                cam_proc, cam_log = spawn_camera(cam_log_path)
                print(f"[INFO] 已启动摄像头 pid={cam_proc.pid}（日志 {cam_log_path}）")

            t_end = time.monotonic() + args.camera_timeout
            while time.monotonic() < t_end:
                spin_once(0.02)
                if node.image_alive():
                    break
            if not node.image_alive():
                print(f"[ERROR] {args.camera_timeout:.0f}s 内 {args.image_topic} 没有出图，"
                      "不开相机车会画弧，拒绝开始。")
                tail = camera_log_tail(cam_log_path)
                if tail:
                    print("        相机日志末尾：")
                    print(tail)
                print("        检查 /dev/video0、USB 线；或显式加 --no-camera 跳过。")
                return 3
            print(f"[OK] 相机出图正常（已收 {node.img_count} 帧）")
        else:
            print("[WARN] --no-camera：没有视觉闭环，舵机中位有偏就会画弧；"
                  "车不直时加 --heading-hold 0.8")

        # 再给发现一点时间，检查 /cmd_vel 是否已有发布者
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            spin_once(0.02)
        others = node.other_publishers()
        if others:
            print("[ERROR] /cmd_vel 已有发布者，测试会和他们抢控制：")
            for o in others:
                print(f"        - {o}")
            print("        请先停掉巡线/仲裁/兜底（底盘驱动保持运行），例如：")
            print("        如果 tools/start.sh 在跑，先在它的终端 Ctrl+C")
            print("        pkill -INT -f line_follower_with_detection")
            print("        pkill -INT -f safety_node.py")
            return 2

        node.create_cmd_publisher()
        # 等 publisher 发现，再确认没有别的发布者
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            spin_once(0.02)
        others = node.other_publishers()
        if others:
            print("[ERROR] /cmd_vel 仍有其他发布者：")
            for o in others:
                print(f"        - {o}")
            return 2

        # 等 odom
        t_end = time.monotonic() + args.wait_timeout
        while time.monotonic() < t_end:
            spin_once(0.02)
            if node.odom_seen and time.monotonic() - node.last_odom < 0.5:
                break
        if not node.odom_seen:
            print(f"[ERROR] {args.odom_topic} 没有数据，检查底盘驱动/串口。")
            return 2
        if not node.voltage_seen:
            print(f"[WARN] {args.voltage_topic} 没有数据，本次不检查低压。")
        elif node.voltage < args.min_voltage:
            print(f"[ERROR] 电压 {node.voltage:.2f} V < {args.min_voltage:.1f} V，先充电。")
            return 2

        print(f"[OK] odom 已就绪，电压 {node.voltage:.2f} V" if node.voltage_seen
              else "[OK] odom 已就绪")
        print(f"[INFO] 输出目录：{out_dir}")
        print("[SAFE] 确认：车在直线起点、前方 ≥2.5 m 净空、车头方向无人。")

        if not args.yes:
            for i in range(5, 0, -1):
                print(f"       {i} 秒后开始...", end="\r", flush=True)
                time.sleep(1.0)
            print()

        with open(summary_path, "w", newline="", encoding="utf-8") as fsum:
            writer = csv.DictWriter(fsum, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            fsum.flush()

            stop_all = False
            for v_cmd in args.speeds:
                for rep in range(1, args.reps + 1):
                    # 每次开始前再查一次，防止有人中途把巡线拉起来
                    others = node.other_publishers()
                    if others:
                        print("[ERROR] 检测到新的 /cmd_vel 发布者，中止：")
                        for o in others:
                            print(f"        - {o}")
                        stop_all = True
                        break
                    if not args.yes:
                        try:
                            input(f"[{v_cmd:.2f} m/s 第 {rep}/{args.reps} 次] "
                                  "把车推回起点，按 Enter 开始...")
                        except EOFError:
                            print("\n[STOP] 没有交互输入，结束。")
                            stop_all = True
                            break
                    res = run_rep(v_cmd, rep)
                    raw_path = out_dir / f"raw_{v_cmd:.2f}_{rep}.csv"
                    with open(raw_path, "w", newline="", encoding="utf-8") as fraw:
                        wraw = csv.DictWriter(fraw, fieldnames=[
                            "t", "phase", "v_cmd", "v", "yaw", "x", "y",
                            "d_path", "voltage"])
                        wraw.writeheader()
                        for s in res.samples:
                            wraw.writerow({
                                "t": round(s.t, 4), "phase": s.phase,
                                "v_cmd": round(s.v_cmd, 4),
                                "v": round(s.v, 4), "yaw": round(s.yaw, 4),
                                "x": round(s.x, 4), "y": round(s.y, 4),
                                "d_path": round(s.d_path, 4),
                                "voltage": round(s.voltage, 3)
                                if math.isfinite(s.voltage) else "",
                            })
                    results.append(res)
                    writer.writerow(res.row())
                    fsum.flush()
                    print(f"  -> {res.status:20s} "
                          f"t_total={fmt(res.t_total, 2, 's'):>8} "
                          f"d_total={fmt(res.d_total, 3, 'm'):>8} "
                          f"t_stop={fmt(res.t_stop, 2, 's'):>7} "
                          f"d_stop={fmt(res.d_stop, 3, 'm'):>7} "
                          f"a_dec={fmt(res.a_dec_eff or res.a_dec_fit, 2, 'm/s2'):>10}"
                          + (f"  odom/卷尺={res.odom_scale:.3f}"
                             if res.odom_scale else "")
                          + (f"  ({res.note})" if res.note else ""))
                    if res.status != "OK":
                        stop_all = True
                        break
                if stop_all:
                    break

        print_summary(results, out_dir)
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
        stop_camera(cam_proc, cam_log)


# ---------------------------------------------------------------- 汇总

def print_summary(results: List[RepResult], out_dir: Path) -> None:
    ok = [r for r in results if r.status == "OK"]
    print("\n" + "=" * 92)
    print(f"{'v_cmd':>6} {'n_ok':>5} {'v_ss':>6} {'t90':>6} {'d90':>6} "
          f"{'t_stop':>7} {'d_stop':>7} {'d_total':>8} {'a_dec':>7}")
    print("-" * 92)
    for v in sorted({r.v_cmd for r in results}):
        sub = [r for r in ok if abs(r.v_cmd - v) < 1e-6]
        if not sub:
            print(f"{v:6.2f} {0:5d}   全部失败/中止")
            continue
        avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
        print(f"{v:6.2f} {len(sub):5d} "
              f"{avg([r.v_ss for r in sub]):6.2f} "
              f"{avg([(r.t90 or 0) for r in sub]):6.2f} "
              f"{avg([(r.d90 or 0) for r in sub]):6.3f} "
              f"{avg([(r.t_stop or 0) for r in sub]):7.2f} "
              f"{avg([(r.d_stop or 0) for r in sub]):7.3f} "
              f"{avg([(r.d_total or 0) for r in sub]):8.3f} "
              f"{avg([(r.a_dec_eff or r.a_dec_fit or 0) for r in sub]):7.2f}")
    print("=" * 92)

    if not ok:
        print("没有一组 OK，先检查 odom / 电压 / 是否还有别的 /cmd_vel 发布者。")
        print(f"[out] {out_dir / 'summary.csv'}")
        return

    a_vals = [r.a_dec_fit or r.a_dec_eff for r in ok
              if (r.a_dec_fit or r.a_dec_eff)]
    lag_vals = [r.t_lag for r in ok if r.t_lag is not None]
    a_dec = sorted(a_vals)[len(a_vals) // 2] if a_vals else 0.5
    t_lag = sorted(lag_vals)[len(lag_vals) // 2] if lag_vals else 0.2
    vmax = max(r.v_cmd for r in results)

    # 从 d90 反推一阶加速的时间常数：到 90% 的距离 = V*tau*(-ln(0.1)-0.9)
    tau_vals = []
    for r in ok:
        if r.d90 and r.v_cmd > 0:
            tau_vals.append(r.d90 / (r.v_cmd * (2.302585 - 0.9)))
    tau_acc = sorted(tau_vals)[len(tau_vals) // 2] if tau_vals else 0.61
    tau_acc = min(max(tau_acc, 0.20), 2.0)

    print(f"\n用于估算的 a_dec={a_dec:.2f} m/s², t_lag={t_lag:.2f} s, "
          f"tau_acc={tau_acc:.2f} s, margin=0.10 m")
    print("场景 A：已经在速度 v，前方 D 内能停住（只看刹车段）")
    print(f"{'可用距离 D':>10} {'建议最高速度':>14}  (不超过已测最高 {vmax:.2f})")
    print("-" * 46)
    for D in (0.5, 0.8, 1.2):
        v = stop_speed_for_distance(D, a_dec, t_lag, margin=0.10)
        print(f"{D:10.1f} {min(v, vmax):14.2f}")

    print("\n场景 B：从静止起步，到黄线距离 D，能加速到并停住（含加速段）")
    print(f"{'黄线距离 D':>10} {'建议最高速度':>14}  (不超过已测最高 {vmax:.2f})")
    print("-" * 46)
    for D in (0.5, 0.8, 1.2):
        v = speed_from_rest_for_distance(D, a_dec, t_lag, tau_acc, margin=0.10)
        print(f"{D:10.1f} {min(v, vmax):14.2f}")

    # 卷尺比例
    tapes = [(r.d_total, r.tape_distance) for r in ok
             if r.tape_distance and r.d_total]
    if tapes:
        scale = sum(d / t for d, t in tapes) / len(tapes)
        print(f"\nodom/卷尺 比例：{scale:.3f}（odom 虚高 {100*(scale-1):+.1f}%）")
    print(f"[out] {out_dir / 'summary.csv'}")


# ---------------------------------------------------------------- dry-run

def print_dry_run(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("速度阶跃-制动测试（dry-run，只打印，不发任何 /cmd_vel）")
    print("=" * 72)
    print(f"速度档      : {', '.join(f'{v:.2f}' for v in args.speeds)} m/s")
    print(f"每档次数    : {args.reps}")
    print(f"目标判定    : odom 速度达到 {args.reach_ratio*100:.0f}% 后立刻发 0")
    print(f"匀速保持    : {args.hold:.2f} s（0 = 不保持，直接刹）")
    print(f"最大距离    : {args.max_distance:.2f} m（超过立即中止）")
    print(f"停稳判定    : |v| < {args.stop_speed:.2f} m/s 持续 {args.still_hold:.2f} s")
    print(f"安全阈值    : 电压 < {args.min_voltage:.1f} V / odom 超时 "
          f"{args.odom_timeout:.2f} s / 偏航 > {args.yaw_limit_deg:.1f}°")
    print(f"卷尺标定    : " + ("每次跑完询问实测位移" if args.ask_tape
                              else (f"固定 {args.tape_distance:.3f} m" if args.tape_distance
                                    else "关闭（summary 的 tape_distance 为空）")))
    print(f"摄像头      : " + (f"测试期间保持出图（{args.image_topic}，"
                              f"等待 {args.camera_timeout:.0f}s）" if args.camera
                              else "关闭（--no-camera）"))
    print(f"走直线      : " + (f"--heading-hold {args.heading_hold:.2f} "
                              f"(上限 ±{args.heading_hold_max:.2f} rad/s)"
                              if args.heading_hold else "纯 angular.z=0，不纠偏"))
    print("\n单次时序：")
    print("  0.0 s   发 0，静止确认 0.5 s")
    print("  0.5 s   发 V，30 Hz 记录")
    print(f"  ...     v >= {args.reach_ratio:.2f}*V 或 {args.timeout_reach:.1f} s 超时")
    print(f"  ...     可选匀速 {args.hold:.2f} s")
    print("  ...     立刻发 0")
    print(f"  ...     |v| < {args.stop_speed:.2f} 持续 {args.still_hold:.2f} s，或 "
          f"{args.timeout_stop:.1f} s 超时")
    print("\n跑之前必须做：")
    print("  0) 如果 tools/start.sh 在跑，先在它的终端 Ctrl+C，")
    print("     否则看门狗会把巡线重新拉起来，和测试抢 /cmd_vel。")
    print("  1) 停掉巡线/仲裁/兜底，但保持底盘驱动：")
    print("     pkill -INT -f line_follower_with_detection")
    print("     pkill -INT -f safety_node.py")
    print("  2) 确认底盘还在： ros2 topic hz /odom")
    print("  3) 车放在直线起点，前方 >= 2.5 m 净空")
    print("  4) 先跑一次低速： python3 tools/speed_step.py --speeds 0.30 --reps 1")
    print("\n正式命令：")
    print("  source /opt/tros/humble/setup.bash")
    print("  python3 tools/speed_step.py --speeds "
          + " ".join(f"{v:.2f}" for v in args.speeds)
          + f" --reps {args.reps}")
    print("=" * 72)


# ---------------------------------------------------------------- CLI

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="实车速度阶跃-制动测试：0→V→0，测位移/时间/减速度",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS,
                    help="目标速度档，m/s")
    ap.add_argument("--reps", type=int, default=3, help="每档次数")
    ap.add_argument("--reach-ratio", type=float, default=0.90,
                    help="达到目标速度的多少比例就立刻刹")
    ap.add_argument("--max-distance", type=float, default=2.0,
                    help="单次允许的最大路径，超过立即中止")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="达到目标后先匀速多久再刹（测稳态用）")
    ap.add_argument("--settle", type=float, default=0.5, help="起步前静止确认时长")
    ap.add_argument("--rate", type=float, default=30.0, help="命令/采样频率 Hz")
    ap.add_argument("--stop-speed", type=float, default=0.02, help="停稳速度阈值 m/s")
    ap.add_argument("--still-hold", type=float, default=0.3, help="停稳保持时长 s")
    ap.add_argument("--timeout-reach", type=float, default=2.5, help="加速超时 s")
    ap.add_argument("--timeout-stop", type=float, default=3.0, help="减速超时 s")
    ap.add_argument("--min-voltage", type=float, default=11.0, help="低压中止阈值 V")
    ap.add_argument("--odom-timeout", type=float, default=0.3, help="odom 超时 s")
    ap.add_argument("--yaw-limit-deg", type=float, default=8.0,
                    help="相对起点的偏航漂移上限")
    ap.add_argument("--wait-timeout", type=float, default=8.0, help="等待 odom 的超时 s")
    ap.add_argument("--cmd-topic", default="/cmd_vel")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--voltage-topic", default="/PowerVoltage")
    ap.add_argument("--camera", action=argparse.BooleanOptionalAction, default=True,
                    help="测试期间保持摄像头出图（已在跑就复用，退出时只收自己拉起的）")
    ap.add_argument("--image-topic", default="/image_raw", help="相机图像话题")
    ap.add_argument("--camera-timeout", type=float, default=8.0,
                    help="等相机出图的超时 s，超时拒绝开始")
    ap.add_argument("--heading-hold", type=float, default=0.0, metavar="K",
                    help="偏航保持增益：angular.z=-K*(yaw-yaw0)，0=关闭")
    ap.add_argument("--heading-hold-max", type=float, default=0.15,
                    help="偏航保持输出上限 rad/s")
    ap.add_argument("--out", default=None, help="输出目录（默认 tools/logs/speed_step_<时间>）")
    ap.add_argument("--tape-distance", type=float, default=0.0,
                    help="卷尺实测位移（m）；所有 reps 共用同一个终点标记时填")
    ap.add_argument("--ask-tape", action="store_true",
                    help="每次结束后询问卷尺实测位移")
    ap.add_argument("--yes", action="store_true", help="跳过 5 s 倒计时和 Enter 确认")
    ap.add_argument("--dry-run", action="store_true", help="只打印时序，不发指令")
    args = ap.parse_args(argv)

    if not args.speeds:
        ap.error("--speeds 不能为空")
    if any(v <= 0 or v > 1.2 for v in args.speeds):
        ap.error("--speeds 必须在 (0, 1.2] m/s")
    if args.reps < 1:
        ap.error("--reps 必须 >= 1")
    if not (0.5 <= args.reach_ratio <= 1.0):
        ap.error("--reach-ratio 必须在 0.5–1.0")
    if args.max_distance <= 0.2:
        ap.error("--max-distance 太小")
    if args.reach_ratio >= 0.999:
        print("[WARN] 一阶速度响应不会精确到 100%；reach-ratio=1.0 只会在 "
              "--timeout-reach 后刹，测到的是超时点。建议 0.95–0.98。")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # 卷尺值只在所有 reps 共用同一终点时才有意义；ask-tape 优先
    if args.ask_tape:
        args.tape_distance = 0.0
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
