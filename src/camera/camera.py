#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool
    from cv_bridge import CvBridge
except ImportError:
    rclpy = None
    Node = object
    Image = None
    Bool = None
    CvBridge = None


VALID_AE_AWB_MODES = ('auto', 'lock_on_start', 'manual_fixed')
_CTRL_VALUE_RE = re.compile(r'^\s*([A-Za-z0-9_]+)\s*:\s*(-?\d+)\s*$')


class CameraV4L2Policy:
    """
    Decoupled V4L2 camera control policy testable without camera hardware.

    Supported ae_awb_mode values:
      - "auto": keep UVC auto-exposure and auto-white-balance enabled.
      - "lock_on_start": allow auto-adaptation while idle; on rising edge of
        start_cmd read current exposure/WB/gain via v4l2-ctl (capping exposure
        if exposure_time_absolute > 0) and lock them; on falling edge unlock
        back to auto if unlock_on_stop is True.
      - "manual_fixed": lock manual exposure/WB/gain on startup using configured
        fixed parameter values.

    All v4l2-ctl calls fail gracefully (missing binary, unsupported control,
    timeout, non-zero exit) and never raise exceptions into the capture loop.
    """

    def __init__(
        self,
        device='/dev/video0',
        power_line_frequency=1,
        ae_awb_mode='auto',
        exposure_time_absolute=0,
        white_balance_temperature=0,
        gain=-1,
        backlight_compensation=0,
        exposure_dynamic_framerate=0,
        unlock_on_stop=True,
        cmd_timeout_sec=0.5,
        runner=None,
        logger=None,
    ):
        self.device = str(device)
        self.power_line_frequency = int(power_line_frequency)
        mode = str(ae_awb_mode or 'auto').strip().lower()
        if mode not in VALID_AE_AWB_MODES:
            mode = 'auto'
        self.ae_awb_mode = mode
        self.exposure_time_absolute = int(exposure_time_absolute)
        self.white_balance_temperature = int(white_balance_temperature)
        self.gain = int(gain)
        self.backlight_compensation = int(backlight_compensation)
        self.exposure_dynamic_framerate = int(exposure_dynamic_framerate)
        self.unlock_on_stop = bool(unlock_on_stop)
        self.cmd_timeout_sec = float(max(0.05, cmd_timeout_sec))
        self._runner = runner if runner is not None else subprocess.run
        self._logger = logger

        self.v4l2_available = True
        self.locked = False
        self.last_start_state = False
        self.locked_values = {}

    def _log(self, msg):
        if self._logger is not None:
            self._logger(msg)

    def _run_v4l2(self, args):
        if not self.v4l2_available:
            return False, '', 'v4l2-ctl unavailable'
        cmd = ['v4l2-ctl', '-d', self.device] + list(args)
        try:
            proc = self._runner(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.cmd_timeout_sec,
                check=False,
            )
        except FileNotFoundError:
            self.v4l2_available = False
            self._log(
                f'[DEC] camera v4l2 unavailable device={self.device} '
                f'reason=v4l2-ctl_not_found'
            )
            return False, '', 'v4l2-ctl not found'
        except subprocess.TimeoutExpired:
            self._log(
                f'[DEC] camera v4l2 timeout device={self.device} args={" ".join(args)}'
            )
            return False, '', 'timeout'
        except OSError as exc:
            self._log(
                f'[DEC] camera v4l2 os_error device={self.device} err={exc}'
            )
            return False, '', str(exc)

        rc = getattr(proc, 'returncode', 1)
        stdout = getattr(proc, 'stdout', '') or ''
        stderr = getattr(proc, 'stderr', '') or ''
        return (rc == 0), stdout, stderr

    @staticmethod
    def parse_ctrl_output(stdout):
        parsed = {}
        for line in (stdout or '').splitlines():
            m = _CTRL_VALUE_RE.match(line)
            if m:
                parsed[m.group(1)] = int(m.group(2))
        return parsed

    def set_ctrl(self, name, value, aliases=(), optional=False):
        """Try setting a V4L2 control by primary name, then fallback aliases."""
        candidates = (name,) + tuple(aliases)
        last_err = ''
        for ctrl_name in candidates:
            ok, _, err = self._run_v4l2(['--set-ctrl', f'{ctrl_name}={int(value)}'])
            if ok:
                return True, ctrl_name
            last_err = (err or '').strip()
            if not self.v4l2_available:
                return False, ctrl_name
        event = 'unsupported_optional' if optional else 'set_fail'
        self._log(
            f'[DEC] camera v4l2 {event} ctrl={name} val={int(value)} '
            f'err={last_err or "unsupported"}'
        )
        return False, name

    def get_ctrl(self, name, aliases=()):
        """Try reading an integer V4L2 control value; returns None if unsupported."""
        candidates = (name,) + tuple(aliases)
        for ctrl_name in candidates:
            ok, out, _ = self._run_v4l2(['--get-ctrl', ctrl_name])
            if ok:
                parsed = self.parse_ctrl_output(out)
                if ctrl_name in parsed:
                    return parsed[ctrl_name]
                if len(parsed) == 1:
                    return next(iter(parsed.values()))
            if not self.v4l2_available:
                return None
        return None

    def _set_auto_ae_awb(self):
        ae_ok, _ = self.set_ctrl('auto_exposure', 3, aliases=('exposure_auto',))
        if self.exposure_dynamic_framerate in (0, 1):
            self.set_ctrl(
                'exposure_dynamic_framerate',
                self.exposure_dynamic_framerate,
                aliases=('exposure_auto_priority',),
                optional=True,
            )
        awb_ok, _ = self.set_ctrl(
            'white_balance_automatic', 1, aliases=('white_balance_temperature_auto',)
        )
        self.locked = False
        self.locked_values = {}
        return ae_ok, awb_ok

    def _lock_manual_values(self, exposure=None, wb_temp=None, gain=None):
        applied = {}
        ae_ok, _ = self.set_ctrl('auto_exposure', 1, aliases=('exposure_auto',))
        applied['auto_exposure_manual'] = ae_ok
        if exposure is not None and int(exposure) > 0:
            ok, _ = self.set_ctrl(
                'exposure_time_absolute', int(exposure), aliases=('exposure_absolute',)
            )
            if ok:
                applied['exposure_time_absolute'] = int(exposure)

        awb_ok, _ = self.set_ctrl(
            'white_balance_automatic', 0, aliases=('white_balance_temperature_auto',)
        )
        applied['white_balance_manual'] = awb_ok
        if wb_temp is not None and int(wb_temp) > 0:
            ok, _ = self.set_ctrl('white_balance_temperature', int(wb_temp))
            if ok:
                applied['white_balance_temperature'] = int(wb_temp)

        if gain is not None and int(gain) >= 0:
            ok, _ = self.set_ctrl('gain', int(gain))
            if ok:
                applied['gain'] = int(gain)

        self.locked = bool(ae_ok or awb_ok)
        self.locked_values = applied
        return applied

    def apply_startup(self):
        """Apply anti-flicker, backlight compensation, and initial AE/AWB mode."""
        summary = {'mode': self.ae_awb_mode, 'device': self.device}
        if self.power_line_frequency in (0, 1, 2):
            ok, _ = self.set_ctrl('power_line_frequency', self.power_line_frequency)
            summary['power_line_frequency'] = self.power_line_frequency if ok else None

        if self.backlight_compensation >= 0:
            ok, _ = self.set_ctrl('backlight_compensation', self.backlight_compensation)
            summary['backlight_compensation'] = self.backlight_compensation if ok else None

        if self.exposure_dynamic_framerate in (0, 1):
            ok, _ = self.set_ctrl(
                'exposure_dynamic_framerate',
                self.exposure_dynamic_framerate,
                aliases=('exposure_auto_priority',),
                optional=True,
            )
            summary['exposure_dynamic_framerate'] = (
                self.exposure_dynamic_framerate if ok else None
            )

        if self.ae_awb_mode == 'manual_fixed':
            applied = self._lock_manual_values(
                exposure=self.exposure_time_absolute if self.exposure_time_absolute > 0 else None,
                wb_temp=self.white_balance_temperature if self.white_balance_temperature > 0 else None,
                gain=self.gain if self.gain >= 0 else None,
            )
            summary['applied'] = applied
        else:
            ae_ok, awb_ok = self._set_auto_ae_awb()
            summary['auto_ae'] = ae_ok
            summary['auto_awb'] = awb_ok
            if self.gain >= 0:
                ok, _ = self.set_ctrl('gain', self.gain)
                summary['gain'] = self.gain if ok else None

        self._log(
            f'[DEC] camera v4l2 startup mode={self.ae_awb_mode} '
            f'plf={self.power_line_frequency} blc={self.backlight_compensation} '
            f'locked={self.locked} details={summary}'
        )
        return summary

    def on_start_cmd(self, started):
        """
        Handle start_cmd state transitions.
        Returns action string: 'locked', 'unlocked', or 'noop'.
        """
        started = bool(started)
        prev = self.last_start_state
        self.last_start_state = started

        if self.ae_awb_mode != 'lock_on_start':
            return 'noop'

        if started and not prev:
            cur_exp = self.get_ctrl('exposure_time_absolute', aliases=('exposure_absolute',))
            cur_wb = self.get_ctrl('white_balance_temperature')
            cur_gain = self.get_ctrl('gain')

            target_exp = cur_exp
            if self.exposure_time_absolute > 0:
                if target_exp is None:
                    target_exp = self.exposure_time_absolute
                else:
                    target_exp = min(target_exp, self.exposure_time_absolute)

            target_wb = cur_wb if cur_wb is not None else (
                self.white_balance_temperature if self.white_balance_temperature > 0 else None
            )
            target_gain = cur_gain if cur_gain is not None else (
                self.gain if self.gain >= 0 else None
            )

            applied = self._lock_manual_values(
                exposure=target_exp,
                wb_temp=target_wb,
                gain=target_gain,
            )
            self._log(
                f'[DEC] camera v4l2 lock_on_start read=(exp={cur_exp},wb={cur_wb},gain={cur_gain}) '
                f'target=(exp={target_exp},wb={target_wb},gain={target_gain}) '
                f'locked={self.locked} applied={applied}'
            )
            return 'locked'

        if (not started) and prev and self.unlock_on_stop:
            ae_ok, awb_ok = self._set_auto_ae_awb()
            self._log(
                f'[DEC] camera v4l2 unlock_on_stop auto_ae={ae_ok} auto_awb={awb_ok}'
            )
            return 'unlocked'

        return 'noop'


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
        # Double-buffering (2) prevents V4L2 single-buffer SOF starvation (which halves 30fps -> 15fps)
        # while keeping capture latency to at most 1 frame.
        self.declare_parameter('v4l2_buffer_size', 2)
        default_stats_sec = float(os.environ.get('CAMERA_STATS_LOG_INTERVAL_SEC', '5.0'))
        self.declare_parameter('stats_log_interval_sec', default_stats_sec)

        # V4L2 曝光/白平衡/抗频闪策略参数（解耦可切）
        self.declare_parameter('power_line_frequency', 1)
        self.declare_parameter('ae_awb_mode', 'auto')
        self.declare_parameter('exposure_time_absolute', 0)
        self.declare_parameter('white_balance_temperature', 0)
        self.declare_parameter('gain', -1)
        self.declare_parameter('backlight_compensation', 0)
        self.declare_parameter('exposure_dynamic_framerate', 0)
        self.declare_parameter('unlock_on_stop', True)
        self.declare_parameter('start_cmd_topic', 'line_follower/start_cmd')

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

        power_line_frequency = int(self.get_parameter('power_line_frequency').value)
        ae_awb_mode = str(self.get_parameter('ae_awb_mode').value)
        exposure_time_absolute = int(self.get_parameter('exposure_time_absolute').value)
        white_balance_temperature = int(self.get_parameter('white_balance_temperature').value)
        gain = int(self.get_parameter('gain').value)
        backlight_compensation = int(self.get_parameter('backlight_compensation').value)
        exposure_dynamic_framerate = int(self.get_parameter('exposure_dynamic_framerate').value)
        unlock_on_stop = bool(self.get_parameter('unlock_on_stop').value)
        start_cmd_topic = str(self.get_parameter('start_cmd_topic').value)

        # 创建 ROS 2 图像话题发布者（默认队列深度 1，避免旧帧堆积）
        self.publisher_ = self.create_publisher(Image, topic_name, qos_depth)
        self.bridge = CvBridge()

        # 初始化 OpenCV 摄像头捕获
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f"无法打开设备 {device}，请检查连接及权限！")
            sys.exit(1)

        # 先协商 MJPEG + 分辨率 + 帧率，再设置 V4L2 缓冲深度（2 帧双缓冲避免 30fps 腰斩到 15fps）
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, framerate)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, v4l2_buf_size)

        # 应用 V4L2 抗频闪与 AE/AWB 策略（打开流后应用，避免被设备初始化重置）
        self.v4l2_policy = CameraV4L2Policy(
            device=device,
            power_line_frequency=power_line_frequency,
            ae_awb_mode=ae_awb_mode,
            exposure_time_absolute=exposure_time_absolute,
            white_balance_temperature=white_balance_temperature,
            gain=gain,
            backlight_compensation=backlight_compensation,
            exposure_dynamic_framerate=exposure_dynamic_framerate,
            unlock_on_stop=unlock_on_stop,
            logger=lambda m: self.get_logger().info(m),
        )
        self.v4l2_policy.apply_startup()

        # 订阅发车指令用于 lock_on_start 模式下在起跑瞬间锁曝光/白平衡
        self.start_sub_ = None
        if Bool is not None and start_cmd_topic:
            self.start_sub_ = self.create_subscription(
                Bool, start_cmd_topic, self._on_start_cmd, 10
            )

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_buf = int(self.cap.get(cv2.CAP_PROP_BUFFERSIZE))
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = ''.join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))

        self.get_logger().info("摄像头已成功启动!")
        self.get_logger().info(f"设备路径  : {device}")
        self.get_logger().info(f"分辨率    : {actual_w}x{actual_h} ({actual_fourcc})")
        self.get_logger().info(f"目标帧率  : {actual_fps} FPS")
        self.get_logger().info(f"发布话题  : {topic_name} (qos_depth={qos_depth}, v4l2_buf={actual_buf})")
        self.get_logger().info(
            f"V4L2策略  : mode={self.v4l2_policy.ae_awb_mode} "
            f"plf={power_line_frequency} exp={exposure_time_absolute} "
            f"wb={white_balance_temperature} gain={gain} blc={backlight_compensation} "
            f"dyn_fps={exposure_dynamic_framerate} start_topic={start_cmd_topic}"
        )

        # 帧率与耗时统计
        self.frame_count = 0
        self.fail_count = 0
        self._last_stats_ts = time.perf_counter()
        self._last_stats_frames = 0
        self._read_ms_window = deque(maxlen=150)
        self._pub_ms_window = deque(maxlen=150)

        # 创建定时器，循环采集并发布图像（略快于硬件帧间隔，由 V4L2 read() 硬件节拍精准同步）
        timer_period = 0.85 / max(1.0, framerate)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def _on_start_cmd(self, msg):
        try:
            self.v4l2_policy.on_start_cmd(bool(msg.data))
        except Exception as exc:
            self.get_logger().warn(f'[DEC] camera v4l2 start_cb_err={exc}')

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
