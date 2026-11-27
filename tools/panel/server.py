#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web 面板后端（零第三方依赖，只用标准库 + rclpy/cv2/yaml）。

监听 127.0.0.1:8080（被占则 8081…8089），通过 SSH 隧道访问。

设计要点（见 tools/DESIGN.md）：
  - 面板不参与任何控制链路，崩了不影响车
  - 发车后所有写接口返回 423（赛中只读，规则禁止手动遥控）
  - 不做发车按钮：start_toggle_node 以 10Hz 持续发布，会覆盖面板发的值（R05）
  - MJPEG 硬限 6fps，把 CPU 留给 30fps 的巡线循环（R09）
"""

import argparse
import datetime
import json
import math
import os
import queue
import re
import sys
import threading
import time
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import yaml

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, Float32, String
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    rclpy = None
    Node = object
    HistoryPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    CvBridge = None
    Twist = None
    Image = None
    Bool = None
    Float32 = None
    String = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import params_schema  # noqa: E402

# ---------------------------------------------------------------- 双层闭环刹车 & 畅通直道物理包络提速扩展 Schema
BRAKE_SPEED_DEFAULTS = {
    # Dual-Layer Closed-Loop Brake
    'enable_sqrt_brake_envelope': True,
    'enable_closed_loop_brake': True,
    'brake_damping_gain': 0.85,
    'brake_zero_cutoff_mps': 0.025,
    # Clear-Path Physics-Envelope Adaptive Speed
    'enable_clear_path_speed': True,
    'clear_straight_speed': 0.62,
    'clear_preview_dist_m': 0.95,
    'max_accel_mps2': 0.90,
    'max_decel_mps2': 2.20,
    'active_decel_kp': 0.45,
}

BRAKE_SPEED_SCHEMA_GROUPS = [
    {
        'tier': 'calibration',
        'section': 'line_follower_node',
        'group': '双层闭环刹车（平方根减速包络 + 阻尼反拖零倒车）',
        'desc': '与电池电压前馈及 1 参摩擦自适应联动；闭环阻尼按实时 odom 速度反拖并触底零速截断，杜绝倒车过冲',
        'default_open': True,
        'items': [
            ('enable_sqrt_brake_envelope', '启用平方根物理减速包络', 'bool', 0, 1, 1,
             '进入黄线预减速区后按 v=√(2·a_eff·Δd) 连续平顺收敛至停点；关闭回退线性斜坡'),
            ('enable_closed_loop_brake', '启用闭环比例阻尼反拖刹车', 'bool', 0, 1, 1,
             '按实时 odom 速度线性阻尼反拖，降到截止阈值立即切 0.0 杜绝倒车过冲'),
            ('brake_damping_gain', '闭环反拖阻尼增益 (1/s)', 'float', 0.2, 2.0, 0.05,
             '反拖命令 v_brake = -clamp(gain * v_odom, 0, yellow_brake_speed)；默认 0.85'),
            ('brake_zero_cutoff_mps', '反拖零速截断阈值 (m/s)', 'float', 0.005, 0.08, 0.005,
             '当 odom 速度 <= 该阈值时反拖立即归零，防止顶成倒车（默认 0.025 m/s）'),
        ],
    },
    {
        'tier': 'calibration',
        'section': 'line_follower_node',
        'group': '畅通直道物理包络提速（Clear-Path 自适应巡航与斜率限幅）',
        'desc': '仅在无障碍/高置信/小曲率/居中直道段平滑提速，并受运动学安全刹停速帽与非对称加减速斜率约束',
        'default_open': True,
        'items': [
            ('enable_clear_path_speed', '启用畅通直道物理包络提速', 'bool', 0, 1, 1,
             '开：畅通直道按 clear_path_score 平滑提速至 clear_straight_speed，遇弯/障立即受安全速帽回落'),
            ('clear_straight_speed', '畅通直道提速上限 (m/s)', 'float', 0.35, 0.95, 0.01,
             'Run 1 稳健保底推荐 0.62 m/s，Run 2 冲刺可调至 0.78–0.85 m/s'),
            ('clear_preview_dist_m', '畅通前瞻安全视距 (m)', 'float', 0.45, 1.50, 0.05,
             '用于计算运动学安全速帽 v_safe_cap 的前瞻停距（默认 0.95 m）'),
            ('max_accel_mps2', '最大平滑加速度 (m/s²)', 'float', 0.3, 2.5, 0.05,
             '直道起步与出弯提速的加速度斜率上限，防抬头/打滑（默认 0.90 m/s²）'),
            ('max_decel_mps2', '最大平滑减速度 (m/s²)', 'float', 0.8, 4.0, 0.1,
             '入弯与遇障减速的减速度斜率上限（非对称快降慢升，默认 2.20 m/s²）'),
            ('active_decel_kp', '主动超速减速比例增益', 'float', 0.0, 1.2, 0.05,
             '实际 odom 车速高于安全速帽时的主动减速率增益（巡线中恒 >= 0，默认 0.45）'),
        ],
    },
]

SPEED_BRAKE_PRESETS = [
    {
        'id': 'run1_safe',
        'label': 'Run 1 稳健保底（双层刹车 + 0.62 畅通直道）',
        'desc': '推荐初赛/Run 1：闭环零倒车刹车 + 0.62 m/s 畅通直道包络',
        'values': {
            'line_follower_node': dict(BRAKE_SPEED_DEFAULTS),
        },
    },
    {
        'id': 'run2_fast',
        'label': 'Run 2 畅通提速（0.78 m/s 直道冲刺）',
        'desc': '在 Run 1 跑通基础上提高直道上限与加减速响应',
        'values': {
            'line_follower_node': {
                'enable_sqrt_brake_envelope': True,
                'enable_closed_loop_brake': True,
                'brake_damping_gain': 0.95,
                'brake_zero_cutoff_mps': 0.025,
                'enable_clear_path_speed': True,
                'clear_straight_speed': 0.78,
                'clear_preview_dist_m': 1.05,
                'max_accel_mps2': 1.20,
                'max_decel_mps2': 2.60,
                'active_decel_kp': 0.55,
            },
        },
    },
    {
        'id': 'legacy_compat',
        'label': '兼容旧版（关闭双层刹车与直道提速）',
        'desc': '全部回退到旧版线性斜坡刹车与恒定 target_speed 巡航',
        'values': {
            'line_follower_node': {
                'enable_sqrt_brake_envelope': False,
                'enable_closed_loop_brake': False,
                'enable_clear_path_speed': False,
            },
        },
    },
]


def get_full_schema():
    """合并 params_schema.SCHEMA 与双层闭环刹车/畅通直道提速参数分组（去重且不改变既有顺序与 Tier）。"""
    base = list(params_schema.SCHEMA)
    existing = {
        (g['section'], item[0])
        for g in base
        for item in g.get('items', ())
    }
    extra_groups = []
    for g in BRAKE_SPEED_SCHEMA_GROUPS:
        sec = g['section']
        filtered_items = [it for it in g['items'] if (sec, it[0]) not in existing]
        if filtered_items:
            g_copy = dict(g)
            g_copy['items'] = filtered_items
            extra_groups.append(g_copy)

    if not extra_groups:
        return base

    # 插入到第一个 calibration 分组（Tier 1 开关）之后，既醒目又不影响 finals_core 与 B10 组内顺序
    insert_at = len(base)
    for idx, g in enumerate(base):
        if g.get('tier') == 'calibration':
            insert_at = idx + 1
            break
    return base[:insert_at] + extra_groups + base[insert_at:]


def find_param_spec(section, key):
    spec = params_schema.find(section, key)
    if spec is not None:
        return spec
    for g in BRAKE_SPEED_SCHEMA_GROUPS:
        if g['section'] != section:
            continue
        for item in g['items']:
            if item[0] == key:
                return item
    return None

WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
TOOLS = os.path.join(WS, 'tools')
STATE_DIR = '/tmp/rdkx5'
LOGROOT = os.path.join(TOOLS, 'logs')
CONFIG_DIR = os.path.join(TOOLS, 'config')
PANEL_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(WS, 'dataset_capture', 'raw_videos')
DEBUG_RUNS_DIR = os.path.join(WS, 'dataset_capture', 'debug_runs')

MAP_FILES = ('finals_fork', 'finals_roundabout')
DEFAULT_MAP = 'finals_fork'


def _resolve_map_name(name):
    if name in MAP_FILES:
        return name
    if name == 'roundabout':
        return 'finals_roundabout'
    return DEFAULT_MAP

STREAM_FPS = 6.0
STREAM_PERIOD = 1.0 / STREAM_FPS
JPEG_QUALITY = 70
FRAME_STALE_SEC = 3.0
VOLTAGE_LOW_THRESHOLD = 11.6
RECORD_DEFAULT_FPS = 15.0
RECORD_MAX_DURATION_SEC = 120.0
ALLOWED_LOG_FILES = frozenset({'follow', 'base', 'camera', 'yolo', 'start', 'panel', 'safety', 'collector'})
_TAG_SAFE_RE = re.compile(r'[^a-zA-Z0-9_-]+')


def _sanitize_tag(raw, fallback):
    cleaned = _TAG_SAFE_RE.sub('_', str(raw or '').strip()).strip('_-').lower()
    return (cleaned[:24] or fallback)


class RawVideoRecorder:
    """无 ROS 依赖的异步原始画面录制器（录 /image_raw，不带叠加框，供 YOLO 训练）。

    设计原则：
      - 空闲时 is_active=False，ROS 回调首行直接 return，零 CPU/零内存开销
      - 录制时按目标帧率（默认 15fps）节流并入有界队列，后台写盘线程编码，不阻塞巡线
      - 自动限制最长时长（默认 120s），单段 30-45s 仅 ~2-4MB，便于直接 git push 上 GitHub
    """

    def __init__(self, capture_dir=None, debug_dir=None):
        self.capture_dir = capture_dir or CAPTURE_DIR
        self.debug_dir = debug_dir or DEBUG_RUNS_DIR
        self._lock = threading.Lock()
        self.is_active = False
        self.source = 'raw'       # 'raw' (/image_raw) | 'debug' (debug_image + synchronized log bundle)
        self._queue = None
        self._thread = None
        self._filename = ''
        self._filepath = ''
        self._fps = RECORD_DEFAULT_FPS
        self._max_duration = RECORD_MAX_DURATION_SEC
        self._t0 = 0.0
        self._last_enq_ts = 0.0
        self._frames_written = 0
        self._frames_dropped = 0
        self._last_error = ''
        self._frame_stamps = []   # (frame_idx, t_rel_s, wall_epoch)
        self._log_dir = None
        self._follow_off = 0
        self._events_off = 0

    def start(self, scene='mixed_full', lighting='normal',
              fps=RECORD_DEFAULT_FPS, max_duration_sec=RECORD_MAX_DURATION_SEC,
              source='raw', log_dir=None):
        src = 'debug' if str(source or '').strip().lower() == 'debug' else 'raw'
        with self._lock:
            if self.is_active:
                return False, f'正在录制 {self._filename}，请先停止'
            target_dir = self.debug_dir if src == 'debug' else self.capture_dir
            os.makedirs(target_dir, exist_ok=True)
            s_tag = _sanitize_tag(scene, 'run' if src == 'debug' else 'clip')
            l_tag = _sanitize_tag(lighting, 'normal')
            fps_val = max(2.0, min(30.0, float(fps or RECORD_DEFAULT_FPS)))
            max_dur = max(5.0, min(300.0, float(max_duration_sec or RECORD_MAX_DURATION_SEC)))
            ts_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            if src == 'debug':
                fname = f'{ts_str}_{s_tag}_debug.mp4'
            else:
                fname = f'{ts_str}_{s_tag}_{l_tag}.mp4'
            fpath = os.path.join(target_dir, fname)

            self.source = src
            self._filename = fname
            self._filepath = fpath
            self._fps = fps_val
            self._max_duration = max_dur
            self._t0 = time.time()
            self._last_enq_ts = 0.0
            self._frames_written = 0
            self._frames_dropped = 0
            self._last_error = ''
            self._frame_stamps = []
            self._log_dir = log_dir
            self._follow_off = 0
            self._events_off = 0
            if src == 'debug' and log_dir:
                try:
                    self._follow_off = os.path.getsize(os.path.join(log_dir, 'follow.log'))
                except OSError:
                    self._follow_off = 0
                try:
                    self._events_off = os.path.getsize(os.path.join(log_dir, 'events.csv'))
                except OSError:
                    self._events_off = 0

            self._queue = queue.Queue(maxsize=64)
            self.is_active = True
            self._thread = threading.Thread(
                target=self._writer_loop, args=(self._queue, fpath, fps_val), daemon=True
            )
            self._thread.start()

        print(f'[DEC] panel record_start src={src} file={fname} fps={fps_val:.1f}', flush=True)
        return True, fname

    def push_frame(self, frame, now=None):
        if not self.is_active or frame is None:
            return False
        ts = time.time() if now is None else float(now)
        auto_stop = False
        q = None
        is_debug = False
        t_rel = 0.0
        with self._lock:
            if not self.is_active:
                return False
            t_rel = max(0.0, ts - self._t0)
            if t_rel >= self._max_duration:
                auto_stop = True
            else:
                min_dt = 1.0 / self._fps
                if self._last_enq_ts > 0.0 and (ts - self._last_enq_ts) < (min_dt * 0.92):
                    return False
                self._last_enq_ts = ts
                q = self._queue
                is_debug = (self.source == 'debug')

        if auto_stop:
            threading.Thread(target=self.stop, daemon=True).start()
            return False

        img = frame.copy()
        if is_debug:
            # 烧录与 follow.log / events.csv 精确对齐的相对时间与墙钟时间
            with self._lock:
                f_idx = len(self._frame_stamps)
                self._frame_stamps.append((f_idx, round(t_rel, 3), round(ts, 3)))
            hms = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            stamp = f't={t_rel:05.2f}s #{f_idx:04d} {hms}'
            cv2.rectangle(img, (max(0, img.shape[1] - 258), 2), (img.shape[1] - 2, 24), (10, 14, 20), -1)
            cv2.putText(
                img, stamp, (max(4, img.shape[1] - 252), 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 235, 255), 1, cv2.LINE_AA
            )

        try:
            q.put_nowait(img)
            return True
        except queue.Full:
            with self._lock:
                self._frames_dropped += 1
            return False

    def _writer_loop(self, q, init_path, fps_val):
        writer = None
        actual_path = init_path
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if writer is None:
                    h, w = item.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    w_try = cv2.VideoWriter(actual_path, fourcc, fps_val, (w, h))
                    if not w_try.isOpened():
                        # 极少数精简 OpenCV 缺 mp4v 编码器时自动退回 MJPG .avi
                        actual_path = os.path.splitext(init_path)[0] + '.avi'
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        w_try = cv2.VideoWriter(actual_path, fourcc, fps_val, (w, h))
                        with self._lock:
                            self._filepath = actual_path
                            self._filename = os.path.basename(actual_path)
                    if not w_try.isOpened():
                        with self._lock:
                            self._last_error = 'VideoWriter 打开失败（缺少编码器）'
                        continue
                    writer = w_try
                writer.write(item)
                with self._lock:
                    self._frames_written += 1
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_error = f'{type(exc).__name__}: {exc}'
        finally:
            if writer is not None:
                writer.release()

    def stop(self):
        with self._lock:
            if not self.is_active:
                return False, {'error': '当前未在录制'}
            self.is_active = False
            q = self._queue
            th = self._thread
            t0 = self._t0
            self._queue = None
            self._thread = None

        if q is not None:
            try:
                q.put(None, timeout=1.0)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                q.put(None)
        if th is not None:
            th.join(timeout=5.0)

        with self._lock:
            src = self.source
            fname = self._filename
            fpath = self._filepath
            frames = self._frames_written
            dropped = self._frames_dropped
            err = self._last_error
            elapsed = max(0.0, time.time() - t0) if t0 > 0 else 0.0
            frame_stamps = list(self._frame_stamps)
            log_dir = self._log_dir
            follow_off = self._follow_off
            events_off = self._events_off

        if frames == 0:
            try:
                if fpath and os.path.isfile(fpath):
                    os.remove(fpath)
            except OSError:
                pass
            topic_hint = 'line_follower/debug_image' if src == 'debug' else '/image_raw'
            msg = err or f'未收到 {topic_hint} 画面（请确认相机/巡线节点已运行），未生成空文件'
            return False, {'error': msg, 'frames': 0}

        out_path = fpath
        out_name = fname

        # 故障复盘模式：把带时间戳的调试图视频 + 同步窗口内的 follow.log / events.csv / params / 帧时间戳
        # 一并打成单个紧凑 .zip（约 2~4MB），既可一键下载也可直接 git push！
        if src == 'debug':
            zip_path = os.path.splitext(fpath)[0] + '.zip'
            zip_name = os.path.basename(zip_path)
            try:
                with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.write(fpath, arcname=os.path.basename(fpath))
                    # 帧级精确时间戳表（视频帧号 <-> 相对秒 <-> 墙钟 epoch）
                    ts_lines = ['frame_idx,t_rel_s,wall_epoch\n']
                    for fi, tr, we in frame_stamps:
                        ts_lines.append(f'{fi},{tr:.3f},{we:.3f}\n')
                    zf.writestr('frame_timestamps.csv', ''.join(ts_lines))

                    meta_obj = {
                        'created_iso': datetime.datetime.now().isoformat(timespec='seconds'),
                        'source': 'debug',
                        't0_epoch': round(t0, 3),
                        'duration_sec': round(elapsed, 2),
                        'frames': frames,
                        'dropped': dropped,
                        'log_dir': log_dir or '',
                    }
                    zf.writestr('meta.json', json.dumps(meta_obj, ensure_ascii=False, indent=2))

                    if log_dir and os.path.isdir(log_dir):
                        def _read_slice(p, off):
                            if not os.path.isfile(p):
                                return b''
                            sz = os.path.getsize(p)
                            start_o = off if (0 <= off <= sz) else 0
                            with open(p, 'rb') as rf:
                                rf.seek(start_o)
                                return rf.read(min(4 * 1024 * 1024, max(0, sz - start_o)))

                        f_bytes = _read_slice(os.path.join(log_dir, 'follow.log'), follow_off)
                        if f_bytes:
                            zf.writestr('follow_slice.log', f_bytes)
                        e_path = os.path.join(log_dir, 'events.csv')
                        if os.path.isfile(e_path):
                            with open(e_path, 'rb') as ef:
                                header = ef.readline()
                            e_slice = _read_slice(e_path, events_off)
                            if events_off > 0 and header and not e_slice.startswith(header):
                                e_slice = header + e_slice
                            zf.writestr('events_slice.csv', e_slice)
                        snap = os.path.join(log_dir, 'params_snapshot.yaml')
                        if os.path.isfile(snap):
                            zf.write(snap, arcname='params_snapshot.yaml')
                try:
                    os.remove(fpath)
                except OSError:
                    pass
                out_path = zip_path
                out_name = zip_name
            except Exception as exc:  # noqa: BLE001
                print(f'[panel] debug zip bundle warning: {exc}', flush=True)

        size_mb = 0.0
        try:
            size_mb = round(os.path.getsize(out_path) / (1024.0 * 1024.0), 2)
        except OSError:
            pass
        try:
            rel_path = os.path.relpath(out_path, WS).replace('\\', '/')
        except ValueError:
            rel_path = out_path.replace('\\', '/')
        print(
            f'[DEC] panel record_stop src={src} file={out_name} frames={frames} '
            f'dur={elapsed:.1f}s size_mb={size_mb:.2f} dropped={dropped}',
            flush=True,
        )
        return True, {
            'source': src,
            'filename': out_name,
            'rel_path': rel_path,
            'frames': frames,
            'dropped': dropped,
            'duration_sec': round(elapsed, 1),
            'size_mb': size_mb,
        }

    def list_files(self, limit=30):
        out = []
        with self._lock:
            active_name = self._filename if self.is_active else ''
        scan_dirs = [
            ('debug', self.debug_dir, ('.zip', '.mp4', '.avi')),
            ('raw', self.capture_dir, ('.mp4', '.avi')),
        ]
        for kind, folder, exts in scan_dirs:
            if not os.path.isdir(folder):
                continue
            try:
                for name in os.listdir(folder):
                    if not name.lower().endswith(exts):
                        continue
                    if name == active_name:
                        continue
                    fp = os.path.join(folder, name)
                    if not os.path.isfile(fp):
                        continue
                    st = os.stat(fp)
                    out.append({
                        'kind': kind,
                        'name': name,
                        'size_mb': round(st.st_size / (1024.0 * 1024.0), 2),
                        'mtime': st.st_mtime,
                        'mtime_str': datetime.datetime.fromtimestamp(st.st_mtime).strftime('%m-%d %H:%M:%S'),
                    })
            except OSError:
                continue
        out.sort(key=lambda x: x['mtime'], reverse=True)
        return out[:limit]

    def resolve_safe_path(self, filename):
        base = os.path.basename(str(filename or ''))
        if not base or base != filename or not base.lower().endswith(('.mp4', '.avi', '.zip')):
            return None
        for folder in (self.debug_dir, self.capture_dir):
            fp = os.path.join(folder, base)
            if os.path.isfile(fp):
                return fp
        return None

    def delete_file(self, filename):
        with self._lock:
            if self.is_active and filename == self._filename:
                return False, '该文件正在录制中，无法删除'
        fp = self.resolve_safe_path(filename)
        if not fp:
            return False, '文件不存在'
        try:
            os.remove(fp)
        except OSError as exc:
            return False, str(exc)
        print(f'[DEC] panel record_delete file={filename}', flush=True)
        return True, f'已删除 {filename}'

    def status(self):
        with self._lock:
            active = self.is_active
            src = self.source
            fname = self._filename
            frames = self._frames_written
            dropped = self._frames_dropped
            fps = self._fps
            max_dur = self._max_duration
            elapsed = round(max(0.0, time.time() - self._t0), 1) if (active and self._t0 > 0) else 0.0
            fpath = self._filepath
        cur_mb = 0.0
        if active and fpath:
            try:
                cur_mb = round(os.path.getsize(fpath) / (1024.0 * 1024.0), 2)
            except OSError:
                cur_mb = 0.0
        return {
            'active': active,
            'source': src if active else '',
            'filename': fname if active else '',
            'frames': frames if active else 0,
            'dropped': dropped if active else 0,
            'elapsed_sec': elapsed,
            'size_mb': cur_mb,
            'fps': fps,
            'max_duration_sec': max_dur,
            'dir_rel': 'dataset_capture/debug_runs' if (active and src == 'debug') else 'dataset_capture/raw_videos',
            'files': self.list_files(),
        }


RECORDER = RawVideoRecorder()


# ---------------------------------------------------------------- 共享状态
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = {'debug': None, 'yolo': None}
        self.frame_ts = {'debug': 0.0, 'yolo': 0.0}
        self.frame_seq = {'debug': 0, 'yolo': 0}
        self.started = False
        self.start_cmd = False   # start_toggle 最近一次发布的原始值（不认沿）
        self.locked = False       # 参数是否锁定：只在"真正发车"的上升沿置位
        self.track_state = 'unknown'
        self.yellow_count = 0
        self.yellow_permanent = 4
        self.timer_start = None
        self.timer_stop = None
        self.last_yellow_ts = 0.0
        self.final_stopped = False
        self.round = 1
        self.restart = None       # {'name','t0','off'} 最近一次面板发起的重启
        self.segment = {}         # line_follower/segment 最近一次上报：seg/n/count/dist
        self.voltage = None       # /PowerVoltage 最近一次读数 (V)
        self.voltage_ts = 0.0     # 上次收到电压的时间戳


SH = Shared()


def reset_round_locked():
    """重置本轮统计。只动面板自己的数字，不碰车（调用方需持锁）。"""
    SH.timer_start = None
    SH.timer_stop = None
    SH.yellow_count = 0
    SH.last_yellow_ts = 0.0
    SH.final_stopped = False
    SH.track_state = 'unknown'
    SH.segment = {}
    SH.round += 1


def apply_segment_message(raw_text, now=None):
    """解析 line_follower/segment 消息；显式 final=1 时立即停表（B6）。"""
    info = {}
    for tok in str(raw_text or '').split():
        if '=' not in tok:
            continue
        k, v = tok.split('=', 1)
        try:
            info[k] = float(v)
        except ValueError:
            info[k] = v
    if 'seg' in info or 'mode' in info or 'final' in info:
        ts = time.time() if now is None else float(now)
        is_final = False
        if 'final' in info:
            try:
                is_final = int(round(float(info['final']))) == 1
            except (TypeError, ValueError):
                is_final = str(info['final']).strip().lower() in ('1', 'true')
        with SH.lock:
            SH.segment = info
            if is_final and SH.timer_start is not None and not SH.final_stopped and SH.timer_stop is None:
                SH.timer_stop = ts
                SH.final_stopped = True
    return info


# ---------------------------------------------------------------- ROS 节点
class PanelNode(Node):
    def __init__(self):
        super().__init__('panel_node')
        self.bridge = CvBridge()
        self._last_started = False
        self._last_debug_jpeg_ts = 0.0

        img_qos = (
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            if QoSProfile is not None
            else 5
        )
        self.create_subscription(Image, 'line_follower/debug_image', self._on_debug, img_qos)
        self.create_subscription(Image, '/yolo_detection_image', self._on_yolo, img_qos)
        self.create_subscription(Image, '/image_raw', self._on_raw_image, img_qos)
        self.create_subscription(Bool, 'line_follower/start_cmd', self._on_start, 10)
        self.create_subscription(String, 'line_follower/track_state', self._on_state, 10)
        # 当前状态与显式终点标志：mode=finals s=.. count=.. branch=.. final=0/1
        self.create_subscription(String, 'line_follower/segment', self._on_segment, 10)
        self.create_subscription(Float32, '/PowerVoltage', self._on_voltage, 10)
        self.estop_pub = self.create_publisher(Twist, '/cmd_vel_stop', 10)

        self.get_logger().info('panel_node ready')

    def _encode(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def _on_debug(self, msg):
        now = time.time()
        need_rec = RECORDER.is_active and RECORDER.source == 'debug'
        need_jpeg = (now - self._last_debug_jpeg_ts) >= STREAM_PERIOD
        if not need_rec and not need_jpeg:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if need_rec:
            RECORDER.push_frame(frame, now=now)
        if not need_jpeg:
            return
        self._last_debug_jpeg_ts = now
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with SH.lock:
                SH.frames['debug'] = buf.tobytes()
                SH.frame_ts['debug'] = now
                SH.frame_seq['debug'] += 1

    def _on_yolo(self, msg):
        data = self._encode(msg)
        if data:
            with SH.lock:
                SH.frames['yolo'] = data
                SH.frame_ts['yolo'] = time.time()
                SH.frame_seq['yolo'] += 1

    def _on_raw_image(self, msg):
        # 空闲或处于调试复盘模式时直接返回，不做任何图像拷贝或解码
        if not RECORDER.is_active or RECORDER.source != 'raw':
            return
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        RECORDER.push_frame(frame)

    def _on_start(self, msg):
        new_val = bool(msg.data)

        # 原始值单独存一份：判断"重启安不安全"要用它。
        # start_toggle 以 10Hz 重复发布同一个值，这里只认跳变沿。
        with SH.lock:
            SH.start_cmd = new_val

        if new_val == self._last_started:
            return
        self._last_started = new_val

        with SH.lock:
            if new_val:
                # 上升沿 = 发车 → 新的一轮。
                # 必须把 last_yellow_ts / track_state / segment 也清掉：
                # 上一轮永久停车后 state / final=1 会滞留，
                # 不清的话新一圈刚开始就会异常状态残留。
                SH.timer_start = time.time()
                SH.timer_stop = None
                SH.final_stopped = False
                SH.yellow_count = 0
                SH.last_yellow_ts = 0.0
                SH.track_state = 'unknown'
                SH.segment = {}
                SH.round += 1
                SH.started = True
                SH.locked = True
            else:
                SH.started = False
                SH.locked = False

        _write_state_file('started', '1' if new_val else '0')

    def _on_state(self, msg):
        state = msg.data
        with SH.lock:
            prev = SH.track_state
            SH.track_state = state
            if state == 'yellow_line_detected' and prev != 'yellow_line_detected':
                SH.yellow_count += 1
                SH.last_yellow_ts = time.time()

    def _on_segment(self, msg):
        """节点上报：finals "mode=finals s=3.20 count=1 branch=-1 final=0"。"""
        apply_segment_message(msg.data)

    def _on_voltage(self, msg):
        val = float(msg.data)
        with SH.lock:
            SH.voltage = val
            SH.voltage_ts = time.time()

    def estop(self, frames=30, rate=20.0):
        msg = Twist()
        for _ in range(frames):
            self.estop_pub.publish(msg)
            time.sleep(1.0 / rate)


def _write_state_file(name, value):
    try:
        with open(os.path.join(STATE_DIR, name), 'w', encoding='utf-8') as f:
            f.write(value)
    except OSError:
        pass


NODE = None


# ---------------------------------------------------------------- 数据读取
def read_status():
    path = os.path.join(STATE_DIR, 'status.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'global': 'down', 'procs': {}, 'started': False, 'run_id': None}


def run_dir():
    st = read_status()
    rid = st.get('run_id')
    if rid:
        p = os.path.join(LOGROOT, rid)
        if os.path.isdir(p):
            return p
    latest = os.path.join(LOGROOT, 'latest')
    if os.path.isdir(latest):
        return os.path.realpath(latest)
    return None


def read_seg(name):
    """读 wrap.sh 写的 <name>.seg：本次进程的 pid / 日志起点 / 启动时间。

    有了它就不用猜"这一行是哪个进程打的"，也不用在长日志里往回翻 30KB。
    """
    path = os.path.join(STATE_DIR, f'{name}.seg')
    out = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                k, _, v = line.strip().partition('=')
                if k:
                    out[k] = v
    except OSError:
        return {}
    for k in ('pid', 'log_off', 'started_epoch'):
        try:
            out[k] = int(out.get(k, 0))
        except (TypeError, ValueError):
            out[k] = 0
    return out


def read_effective_params():
    """解析 follow.log 里节点启动时打印的「[DEC] param effective ...」行。

    只看本次进程的日志段（seg.log_off 之后）。拿整份文件或"最后 30KB"都会
    被上一次启动的行干扰，这正是"改了参数到底生效没有"说不清的原因。
    """
    rd = run_dir()
    if not rd:
        return {}
    path = os.path.join(rd, 'follow.log')
    if not os.path.isfile(path):
        return {}
    seg = read_seg('follow')
    try:
        size = os.path.getsize(path)
        start = int(seg.get('log_off') or 0)
        # start == size 表示新进程还没输出，必须返回 {}（等它上报），
        # 不能退回读旧内容，否则面板会拿上一代进程的参数冒充“已生效”。
        if not (0 <= start <= size):
            start = max(0, size - 65536)
        with open(path, 'rb') as f:
            f.seek(start)
            # param effective 在节点启动后几秒内就打出来了，读前 64KB 足够
            text = f.read(65536).decode('utf-8', errors='replace')
    except OSError:
        return {}

    # 节点启动时是"全局一行 + 每段一行"，必须合并而不是取最后一行，
    # 否则面板只能看到最后一段的生效值，切到段1/2/3 就判不出是否待重启。
    result = {}
    for line in text.splitlines():
        if 'param effective' not in line:
            continue
        for tok in line.split('param effective', 1)[1].split():
            if '=' in tok:
                k, v = tok.split('=', 1)
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
    return result


READY_PAT = {
    'follow': b'param effective',
    'panel': b'[panel] http://',
    'base': b'serial port opened',
    'camera': '摄像头已成功启动'.encode('utf-8'),
    'yolo': '视觉检测节点就绪'.encode('utf-8'),
}


def restart_state():
    """面板发起的重启进行到哪一步：queued / killing / starting / ready / timeout。

    依据是日志里本次进程的分界线（wrap.sh 写的 [DEC] proc start），
    不是 status.json —— 看门狗在等新进程就绪时不会刷新 status，会误判。
    """
    with SH.lock:
        r = dict(SH.restart) if SH.restart else None
    if not r:
        return None

    name, off = r['name'], int(r.get('off') or 0)
    elapsed = time.time() - r['t0']
    rd = run_dir()
    chunk = b''
    if rd:
        path = os.path.join(rd, f'{name}.log')
        try:
            size = os.path.getsize(path)
            if size > off:
                with open(path, 'rb') as f:
                    f.seek(off)
                    chunk = f.read(min(524288, size - off))
        except OSError:
            chunk = b''

    started = b'[DEC] proc start name=' in chunk
    pat = READY_PAT.get(name)
    ready = (pat in chunk) if pat else started

    if ready:
        state = 'ready'
    elif elapsed > 30:
        state = 'timeout'
    elif started:
        state = 'starting'
    elif elapsed > 2:
        state = 'killing'
    else:
        state = 'queued'
    return {'name': name, 'state': state, 'elapsed': round(elapsed, 1)}


def read_recent_events(limit=60):
    rd = run_dir()
    if not rd:
        return []
    path = os.path.join(rd, 'events.csv')
    if not os.path.isfile(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            rows = f.readlines()
    except OSError:
        return []
    out = []
    for line in rows[1:][-limit:]:
        parts = line.rstrip('\r\n').split(',', 5)
        if len(parts) >= 6:
            out.append({'t': parts[1], 'source': parts[2], 'area': parts[3],
                        'event': parts[4], 'kv': parts[5]})
    return out


def _parse_kv_tokens(kv_str):
    out = {}
    for tok in str(kv_str or '').replace(',', ' ').split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            k = k.strip()
            if k:
                out[k] = v.strip()
    return out


def _parse_metric_num(raw, as_int=False, digits=2):
    if raw is None:
        return None
    s = str(raw).strip().rstrip(',;')
    if '->' in s:
        s = s.split('->', 1)[1].strip()
    for suf in ('m/s²', 'm/s2', 'mps2', 'm/s', 'mps', 'ms'):
        if s.endswith(suf):
            s = s[:-len(suf)].strip()
            break
    try:
        val = float(s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return int(round(val)) if as_int else round(val, digits)


DEC_TELEMETRY_KEY_MAP = {
    'v_safe_cap': 'v_safe_cap_mps',
    'v_safe_cap_mps': 'v_safe_cap_mps',
    'vsafe': 'v_safe_cap_mps',
    'clear_score': 'clear_path_score',
    'clear_path_score': 'clear_path_score',
    'clr_score': 'clear_path_score',
    'brake_cmd': 'brake_cmd_mps',
    'brake_cmd_mps': 'brake_cmd_mps',
    'v_brake': 'brake_cmd_mps',
    'a_eff': 'effective_brake_decel_mps2',
    'effective_brake_decel_mps2': 'effective_brake_decel_mps2',
    'brake_a_eff': 'effective_brake_decel_mps2',
}


def parse_dec_telemetry_line(line):
    """解析单条 [DEC] 或 kv 字符串中的双层刹车 / 畅通直道提速遥测字段。

    支持字段映射：
      - v_safe_cap / v_safe_cap_mps -> v_safe_cap_mps
      - clear_score / clear_path_score -> clear_path_score
      - brake_cmd / brake_cmd_mps -> brake_cmd_mps
      - a_eff / effective_brake_decel_mps2 -> effective_brake_decel_mps2
    """
    text = str(line or '').strip()
    if not text:
        return {}
    if '[DEC]' in text:
        text = text.split('[DEC]', 1)[1]
    kv_map = _parse_kv_tokens(text)
    parsed = {}
    for raw_k, raw_v in kv_map.items():
        canon = DEC_TELEMETRY_KEY_MAP.get(raw_k)
        if not canon:
            continue
        num = _parse_metric_num(raw_v, digits=3)
        if num is not None:
            parsed[canon] = num
    return parsed


def extract_brake_speed_telemetry(lines, require_dec_tag=True):
    """从日志行序列提取最新的双层刹车 / 畅通直道遥测快照（缺失字段保持 None）。"""
    out = {
        'v_safe_cap_mps': None,
        'clear_path_score': None,
        'brake_cmd_mps': None,
        'effective_brake_decel_mps2': None,
        'last_t': '',
    }
    for ln in lines or ():
        s = str(ln or '')
        if require_dec_tag and '[DEC]' not in s:
            continue
        parsed = parse_dec_telemetry_line(s)
        if not parsed:
            continue
        out.update(parsed)
    return out


def _read_follow_dec_telemetry(rd, tail_bytes=32768):
    """从当前运行轮次的 follow.log 尾部解析最新 [DEC] 刹车/提速遥测。"""
    if not rd:
        return {}
    path = os.path.join(rd, 'follow.log')
    if not os.path.isfile(path):
        return {}
    try:
        size = os.path.getsize(path)
        seg = read_seg('follow')
        seg_off = int(seg.get('log_off') or 0)
        start = seg_off if (0 <= seg_off <= size) else 0
        if size - start > tail_bytes:
            start = size - tail_bytes
        if start >= size:
            return {}
        with open(path, 'rb') as f:
            f.seek(start)
            text = f.read(tail_bytes).decode('utf-8', errors='replace')
    except OSError:
        return {}
    parsed = extract_brake_speed_telemetry(text.splitlines(), require_dec_tag=True)
    return {k: v for k, v in parsed.items() if k != 'last_t' and v is not None}


def _read_run_elapsed_sec(rd):
    """返回相对本次 start.sh 启动的已运行秒数（与 events.csv 的 t_rel 对齐）。"""
    if not rd:
        return None
    meta = os.path.join(rd, 'meta.json')
    try:
        with open(meta, 'r', encoding='utf-8') as f:
            dt = datetime.datetime.fromisoformat(json.load(f)['started_iso'])
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return max(0.0, (datetime.datetime.now() - dt).total_seconds())
    except Exception:
        return None


def read_diagnostics(scan_lines=400, milestone_limit=40):
    """从 events.csv 与 follow.log 剥离高频性能心跳，提取健康指标、Tier 1、双层刹车/提速遥测与关键里程碑。"""
    perf = {
        'cam_fps': None, 'cam_read_p95': None, 'cam_fails': None,
        'yolo_fps': None, 'yolo_dropped': None, 'yolo_e2e_p95': None,
    }
    red_hint = {'active': False, 'src': '', 'd': None, 't': ''}
    tier1 = {
        'cbf_count': 0,
        'cbf_last_src': '',
        'cbf_last_clr': None,
        'cbf_last_t': '',
        'cbf_hard_stop': False,
        'brake_updates': 0,
        'brake_scale': None,
        'brake_a_eff': None,
        'brake_vbat': None,
        'brake_last_t': '',
    }
    brake_speed = {
        'v_safe_cap_mps': None,
        'clear_path_score': None,
        'brake_cmd_mps': None,
        'effective_brake_decel_mps2': None,
        'last_t': '',
    }
    milestones = []
    safety_active = False

    rd = run_dir()
    if not rd:
        return {
            'perf': perf, 'milestones': milestones, 'red_hint': red_hint,
            'tier1': tier1, 'brake_speed': brake_speed, 'safety_active': safety_active,
        }
    path = os.path.join(rd, 'events.csv')
    rows = []
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                rows = f.readlines()
        except OSError:
            rows = []

    for line in rows[1:][-scan_lines:]:
        parts = line.rstrip('\r\n').split(',', 5)
        if len(parts) < 6:
            continue
        t_str, source, area, event, kv = parts[1], parts[2], parts[3], parts[4], parts[5]

        bs_parsed = parse_dec_telemetry_line(kv)
        if bs_parsed:
            brake_speed.update(bs_parsed)
            brake_speed['last_t'] = t_str

        if area == 'camera' and event == 'perf':
            kv_map = _parse_kv_tokens(kv)
            if 'fps' in kv_map:
                perf['cam_fps'] = _parse_metric_num(kv_map.get('fps'))
            if 'read_p95' in kv_map:
                perf['cam_read_p95'] = _parse_metric_num(kv_map.get('read_p95'))
            if 'fails' in kv_map:
                perf['cam_fails'] = _parse_metric_num(kv_map.get('fails'), as_int=True)
            continue

        if area == 'yolo' and event == 'perf':
            kv_map = _parse_kv_tokens(kv)
            fps_tok = kv_map.get('proc_fps') if 'proc_fps' in kv_map else kv_map.get('fps')
            if fps_tok is not None:
                perf['yolo_fps'] = _parse_metric_num(fps_tok)
            if 'dropped' in kv_map:
                perf['yolo_dropped'] = _parse_metric_num(kv_map.get('dropped'), as_int=True)
            if 'e2e_p95' in kv_map:
                perf['yolo_e2e_p95'] = _parse_metric_num(kv_map.get('e2e_p95'))
            continue

        if area == 'red_src':
            kv_map = _parse_kv_tokens(kv)
            if event == 'detect':
                red_hint = {
                    'active': True,
                    'src': kv_map.get('src', ''),
                    'd': _parse_metric_num(kv_map.get('d')),
                    't': t_str,
                }
            elif event == 'clear':
                red_hint = {'active': False, 'src': '', 'd': None, 't': t_str}
            continue

        if area == 'cbf':
            kv_map = _parse_kv_tokens(kv)
            prev_src = tier1['cbf_last_src']
            src_val = kv_map.get('src', '')
            v_tok = kv_map.get('v', '')
            v_after = _parse_metric_num(v_tok.split('->', 1)[1]) if '->' in v_tok else None
            hard_stop = bool(src_val == 'invalid_cmd' or (v_after is not None and v_after <= 0.01))
            tier1['cbf_count'] += 1
            tier1['cbf_last_src'] = src_val
            tier1['cbf_last_clr'] = _parse_metric_num(kv_map.get('clr'))
            tier1['cbf_last_t'] = t_str
            tier1['cbf_hard_stop'] = hard_stop
            # 只在首次介入、来源变化或紧急降零速时写入里程碑，防止 2Hz 日志刷满队列
            if tier1['cbf_count'] == 1 or src_val != prev_src or hard_stop:
                milestones.append({
                    't': t_str, 'source': source, 'area': area, 'event': event, 'kv': kv,
                })
            continue

        if area == 'brake_adapt':
            kv_map = _parse_kv_tokens(kv)
            scale_tok = kv_map.get('scale', '')
            scale_after = _parse_metric_num(
                scale_tok.split('->', 1)[1] if '->' in scale_tok else scale_tok
            )
            tier1['brake_updates'] += 1
            if scale_after is not None:
                tier1['brake_scale'] = scale_after
            tier1['brake_a_eff'] = _parse_metric_num(kv_map.get('a_eff'))
            tier1['brake_vbat'] = _parse_metric_num(kv_map.get('vbat'))
            tier1['brake_last_t'] = t_str
            milestones.append({
                't': t_str, 'source': source, 'area': area, 'event': event, 'kv': kv,
            })
            continue

        if area == 'safety':
            if event == 'deadman_on':
                safety_active = True
            elif event == 'deadman_off':
                safety_active = False

        if area == 'ctrl':
            continue
        if area == 'state' and event == 'change' and 'to=lost' not in kv:
            continue
        if area == 'mux' and event == 'change' and 'active=stop' not in kv:
            continue

        milestones.append({
            't': t_str, 'source': source, 'area': area, 'event': event, 'kv': kv,
        })

    # 补充/刷新来自 follow.log 最新 [DEC] 行的实时刹车与提速遥测
    follow_bs = _read_follow_dec_telemetry(rd)
    if follow_bs:
        brake_speed.update(follow_bs)

    return {
        'perf': perf,
        'milestones': milestones[-milestone_limit:],
        'red_hint': red_hint,
        'tier1': tier1,
        'brake_speed': brake_speed,
        'safety_active': safety_active,
    }


def load_params(map_name):
    if map_name not in MAP_FILES:
        raise FileNotFoundError(os.path.join(CONFIG_DIR, f'{map_name}.yaml'))
    path = os.path.join(CONFIG_DIR, f'{map_name}.yaml')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    groups = []
    for g in get_full_schema():
        if g.get('maps') and map_name not in g['maps']:
            continue
        sec = g['section']
        cur = (data.get(sec) or {}).get('ros__parameters') or {}
        items = []
        for key, label, typ, lo, hi, step, helptext in g['items']:
            if key in cur:
                raw_val = cur[key]
            elif sec == 'line_follower_node' and key in BRAKE_SPEED_DEFAULTS:
                raw_val = BRAKE_SPEED_DEFAULTS[key]
            else:
                continue
            item_obj = {
                'key': key, 'label': label, 'type': typ,
                'value': str(raw_val) if typ == 'enum' else raw_val,
                'help': helptext,
            }
            if typ == 'enum':
                item_obj['options'] = list(lo)
            else:
                item_obj['min'] = lo
                item_obj['max'] = hi
                item_obj['step'] = step
            items.append(item_obj)
        if items:
            grp = {
                'tier': g.get('tier', 'calibration'),
                'section': sec,
                'group': g['group'],
                'items': items,
            }
            if g.get('desc'):
                grp['desc'] = g['desc']
            if 'default_open' in g:
                grp['default_open'] = bool(g['default_open'])
            groups.append(grp)
    return {
        'map': map_name,
        'tiers': params_schema.TIERS,
        'groups': groups,
        'presets': SPEED_BRAKE_PRESETS,
    }


def save_params(map_name, values):
    """values: {section: {key: value}}"""
    path = os.path.join(CONFIG_DIR, f'{map_name}.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    # 备份必须是“改之前”的内容，所以先深拷贝再动手（踩过一次坑）
    import copy
    backup = copy.deepcopy(data)

    changed = []
    for section, kv in values.items():
        params = (data.get(section) or {}).get('ros__parameters')
        if params is None:
            continue
        for key, val in kv.items():
            spec = find_param_spec(section, key)
            if spec is None:
                continue
            _, _, typ, lo, hi, _step, _help = spec
            if typ == 'bool':
                val = bool(val)
            elif typ == 'enum':
                val = str(val)
                if val not in lo:
                    return False, f'{key}={val} 不在可选项 {lo} 中'
            elif typ == 'int':
                if isinstance(val, bool):
                    return False, f'{key} 数值无效'
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    return False, f'{key} 数值无效'
                if not math.isfinite(fval):
                    return False, f'{key} 数值无效'
                val = int(round(fval))
            else:
                if isinstance(val, bool):
                    return False, f'{key} 数值无效'
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    return False, f'{key} 数值无效'
                if not math.isfinite(val):
                    return False, f'{key} 数值无效'
            if typ in ('int', 'float') and not (lo <= val <= hi):
                return False, f'{key}={val} 超出范围 [{lo}, {hi}]'
            if params.get(key) != val:
                changed.append(f'{key}:{params.get(key)}->{val}')
            params[key] = val

    if not changed:
        return True, 'no change'

    with open(path + '.bak', 'w', encoding='utf-8') as f:
        yaml.safe_dump(backup, f, allow_unicode=True, sort_keys=False)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        f.flush()
    os.replace(tmp_path, path)

    print(f'[DEC] panel param_write map={map_name} ' + ' '.join(changed), flush=True)
    return True, '; '.join(changed)


def apply_preset(map_name, preset_id):
    """按预置档位（run1_safe / run2_fast / legacy_compat）安全批量写入参数。"""
    target = next((p for p in SPEED_BRAKE_PRESETS if p['id'] == str(preset_id or '').strip()), None)
    if target is None:
        valid = [p['id'] for p in SPEED_BRAKE_PRESETS]
        return False, f'未知预设 {preset_id}（可选: {valid}）'
    return save_params(map_name, target['values'])


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = 'rdkx5-panel/1.0'

    def log_message(self, fmt, *args):
        pass  # 别刷屏

    # ---------- helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json({'ok': False, 'error': msg}, code)

    def _body(self):
        n = int(self.headers.get('Content-Length', '0') or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode('utf-8'))

    def _is_running(self):
        # 这里判的是"参数是否锁定"，不是"发车信号是否为真"。
        # 急停/重置之后允许改参数，所以用 locked 而不是 started。
        with SH.lock:
            return SH.locked

    # ---------- routes
    def do_GET(self):
        path = self.path.split('?')[0]
        query = {}
        if '?' in self.path:
            for kv in self.path.split('?', 1)[1].split('&'):
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    query[urllib.parse.unquote_plus(k)] = urllib.parse.unquote_plus(v)

        try:
            if path == '/':
                return self._index()
            if path.startswith('/assets/'):
                return self._asset(path[len('/assets/'):])
            if path == '/api/status':
                return self._status()
            if path == '/api/stream/debug':
                return self._stream('debug')
            if path == '/api/stream/yolo':
                return self._stream('yolo')
            if path == '/api/logs':
                return self._logs(query)
            if path == '/api/params':
                return self._json(load_params(_resolve_map_name(query.get('map'))))
            if path == '/api/record/status':
                return self._json({'ok': True, 'recording': RECORDER.status()})
            if path.startswith('/api/record/download/'):
                fname = urllib.parse.unquote(path[len('/api/record/download/'):])
                return self._download_recording(fname)
            return self._err(404, 'not found')
        except FileNotFoundError as e:
            return self._err(404, f'file not found: {e}')
        except BrokenPipeError:
            return
        except Exception as e:  # noqa: BLE001
            return self._err(500, f'{type(e).__name__}: {e}')

    def do_POST(self):
        path = self.path.split('?')[0]
        try:
            if path == '/api/params':
                if self._is_running():
                    return self._err(423, '已发车，参数锁定（赛规禁止赛中调车）')
                body = self._body()
                ok, msg = save_params(_resolve_map_name(body.get('map')), body.get('values', {}))
                return self._json({'ok': ok, 'msg': msg}, 200 if ok else 400)

            if path == '/api/params/preset':
                if self._is_running():
                    return self._err(423, '已发车，参数锁定（赛规禁止赛中调车）')
                body = self._body()
                ok, msg = apply_preset(_resolve_map_name(body.get('map')), body.get('preset'))
                return self._json({'ok': ok, 'msg': msg}, 200 if ok else 400)

            if path.startswith('/api/restart/'):
                if self._is_running():
                    return self._err(423, '已发车，禁止重启节点')
                name = path.rsplit('/', 1)[-1]
                allowed = {'base', 'camera', 'yolo', 'follow', 'panel', 'safety', 'collector'}
                if name not in allowed:
                    return self._err(400, f'未知进程 {name}')
                # 发车信号还是开的时候重启 follow，新节点一订阅到 true 就会立刻开动。
                # 必须先真的停车（⑤窗口按空格），不能靠急停绕过去。
                with SH.lock:
                    live = SH.start_cmd
                if live:
                    return self._err(
                        409, '⑤发车信号仍为开：新进程会立刻开动。请先在⑤窗口按空格停车，再重启')

                off = 0
                rd = run_dir()
                if rd:
                    try:
                        off = os.path.getsize(os.path.join(rd, f'{name}.log'))
                    except OSError:
                        off = 0
                with SH.lock:
                    SH.restart = {'name': name, 't0': time.time(), 'off': off}
                with open(os.path.join(STATE_DIR, 'cmd.restart'), 'w', encoding='utf-8') as f:
                    f.write(name)
                print(f'[DEC] panel restart name={name}', flush=True)
                return self._json({'ok': True, 'msg': f'已请求重启 {name}，正在等新进程就绪…',
                                   'restart': restart_state()})

            if path == '/api/round/reset':
                # 只重置面板统计，不动车：车跑出赛道后，把它搬回起点、按空格，
                # 这里清零是为了让计时从 0 重新计。
                # 同时解除参数锁定——重置说明这一轮已经结束了。
                with SH.lock:
                    reset_round_locked()
                    SH.started = False
                    SH.locked = False
                _write_state_file('started', '0')
                print('[DEC] panel round_reset', flush=True)
                return self._json({'ok': True, 'msg': '本轮已清零，参数已解锁'})

            if path == '/api/estop':
                if NODE is None:
                    return self._err(503, 'ROS 节点未就绪，无法发送急停指令')
                threading.Thread(target=NODE.estop, daemon=True).start()
                with SH.lock:
                    SH.started = False
                    SH.locked = False
                _write_state_file('started', '0')
                print('[DEC] panel estop', flush=True)
                return self._json({'ok': True, 'msg': '已发送 30 帧零速'})

            if path == '/api/record/start':
                body = self._body()
                src = 'debug' if str(body.get('source', 'raw')).lower() == 'debug' else 'raw'
                if src == 'raw' and self._is_running():
                    return self._err(423, '发车运行中禁止原始训练集录制（故障复盘请切「调试图+同步日志」）')
                ok, info = RECORDER.start(
                    scene=body.get('scene', 'run' if src == 'debug' else 'mixed_full'),
                    lighting=body.get('lighting', 'normal'),
                    fps=body.get('fps', 12.0 if src == 'debug' else RECORD_DEFAULT_FPS),
                    max_duration_sec=body.get('max_duration_sec', RECORD_MAX_DURATION_SEC),
                    source=src,
                    log_dir=run_dir() if src == 'debug' else None,
                )
                if not ok:
                    return self._err(409, info)
                label = '故障复盘（调试图+同步日志）' if src == 'debug' else '原始画面 /image_raw'
                return self._json({'ok': True, 'filename': info, 'msg': f'开始录制 {label}：{info}',
                                   'recording': RECORDER.status()})

            if path == '/api/record/stop':
                ok, info = RECORDER.stop()
                if not ok:
                    return self._json({'ok': False, 'error': info.get('error', '停止失败'),
                                       'recording': RECORDER.status()}, 400)
                return self._json({'ok': True, 'result': info,
                                   'msg': f"录制完成：{info['filename']}（{info['frames']} 帧，{info['size_mb']:.2f} MB）",
                                   'recording': RECORDER.status()})

            if path == '/api/record/delete':
                if self._is_running():
                    return self._err(423, '已发车，禁止删除文件')
                body = self._body()
                ok, msg = RECORDER.delete_file(body.get('filename', ''))
                return self._json({'ok': ok, 'msg': msg, 'recording': RECORDER.status()},
                                  200 if ok else 400)

            return self._err(404, 'not found')
        except Exception as e:  # noqa: BLE001
            return self._err(500, f'{type(e).__name__}: {e}')

    # ---------- handlers
    def _asset(self, rel_name):
        safe = os.path.basename(urllib.parse.unquote(rel_name))
        if not safe or safe != rel_name or not safe.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.svg')):
            return self._err(404, 'asset not found')
        p = os.path.join(PANEL_DIR, 'assets', safe)
        if not os.path.isfile(p):
            return self._err(404, 'asset not found')
        ext = os.path.splitext(safe)[1].lower()
        ctype = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml',
        }.get(ext, 'application/octet-stream')
        with open(p, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'public, max-age=86400')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _index(self):
        p = os.path.join(PANEL_DIR, 'index.html')
        with open(p, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _status(self):
        st = read_status()
        now = time.time()
        with SH.lock:
            started = SH.started
            locked = SH.locked
            live = SH.start_cmd
            state = SH.track_state
            yc = SH.yellow_count
            yp = SH.yellow_permanent
            ts, te = SH.timer_start, SH.timer_stop
            rnd = SH.round
            seg_info = dict(SH.segment)
            frames_ok = {
                k: bool(SH.frames[k] is not None and (now - SH.frame_ts[k]) < FRAME_STALE_SEC)
                for k in ('debug', 'yolo')
            }
            frame_age = {
                k: (round(now - SH.frame_ts[k], 2) if SH.frame_ts[k] > 0 else None)
                for k in ('debug', 'yolo')
            }
            volt = SH.voltage
            volt_ts = SH.voltage_ts

        if ts is None:
            elapsed = 0.0
        elif te is None:
            elapsed = time.time() - ts
        else:
            elapsed = te - ts

        # “YAML 比进程新”是比参数值对比更可靠的判据：
        # 只要保存过且晚于 follow 进程启动时间，就一定有参数没生效。
        seg = read_seg('follow')
        started_epoch = int(seg.get('started_epoch') or 0)
        map_name = _resolve_map_name(st.get('params_file'))
        try:
            ymtime = os.stat(os.path.join(CONFIG_DIR, f'{map_name}.yaml')).st_mtime
        except OSError:
            ymtime = 0.0
        yaml_stale = bool(started_epoch and ymtime > started_epoch + 1)
        diag = read_diagnostics()
        run_elapsed = _read_run_elapsed_sec(run_dir())
        brake_speed = dict(diag.get('brake_speed') or {})
        for raw_k, canon in DEC_TELEMETRY_KEY_MAP.items():
            if brake_speed.get(canon) is None and raw_k in seg_info:
                num = _parse_metric_num(seg_info.get(raw_k), digits=3)
                if num is not None:
                    brake_speed[canon] = num

        self._json({
            'ok': True,
            'global': st.get('global', 'down'),
            'run_id': st.get('run_id'),
            'params_file': st.get('params_file'),
            'procs': st.get('procs', {}),
            'started': started,
            'start_cmd': live,
            'locked': locked,
            'track_state': state,
            'yellow': {'count': yc, 'permanent': yp},
            'segment': seg_info,
            'timer': {'elapsed': round(elapsed, 2), 'running': ts is not None and te is None,
                      'final': te is not None, 'armed': ts is not None},
            'round': rnd,
            'follow': {'started_epoch': started_epoch,
                       'started_iso': seg.get('started_iso', ''),
                       'log_off': int(seg.get('log_off') or 0)},
            'yaml': {'file': f'{map_name}.yaml', 'mtime': ymtime, 'stale': yaml_stale},
            'restart': restart_state(),
            'effective': read_effective_params(),
            'events': read_recent_events(60),
            'perf': diag['perf'],
            'milestones': diag['milestones'],
            'red_hint': diag['red_hint'],
            'tier1': diag['tier1'],
            'brake_speed': brake_speed,
            'safety_active': bool(diag.get('safety_active', False)),
            'run_elapsed_sec': round(run_elapsed, 2) if run_elapsed is not None else None,
            'frames': frames_ok,
            'frame_age': frame_age,
            'voltage': {
                'value': round(volt, 2) if volt is not None else None,
                'low': bool(volt is not None and volt < VOLTAGE_LOW_THRESHOLD),
                'stale': bool(volt_ts == 0.0 or (time.time() - volt_ts) > 5.0),
                'threshold': VOLTAGE_LOW_THRESHOLD,
            },
            'recording': RECORDER.status(),
            'panel_port': PORT,
        })

    def _download_recording(self, filename):
        fp = RECORDER.resolve_safe_path(filename)
        if not fp:
            return self._err(404, f'文件不存在：{filename}')
        low = fp.lower()
        ctype = 'application/zip' if low.endswith('.zip') else (
            'video/mp4' if low.endswith('.mp4') else 'video/x-msvideo'
        )
        size = os.path.getsize(fp)
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(size))
        self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(fp)}"')
        self.end_headers()
        with open(fp, 'rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _stream(self, name):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        last_seq = -1
        try:
            while True:
                with SH.lock:
                    data = SH.frames.get(name)
                    seq = SH.frame_seq.get(name, 0)
                if data and seq != last_seq:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(data)}\r\n\r\n'.encode())
                    self.wfile.write(data)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                    last_seq = seq
                time.sleep(STREAM_PERIOD)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _logs(self, query):
        rd = run_dir()
        if rd is None:
            return self._json({'lines': [], 'offset': 0, 'eof': True})
        fname = query.get('f', 'follow')
        if fname not in ALLOWED_LOG_FILES:
            return self._err(400, f'未知日志 {fname}')
        only_dec = query.get('dec', '0') == '1'
        try:
            offset = max(0, int(query.get('offset', '0') or 0))
        except (TypeError, ValueError):
            offset = 0
        path = os.path.join(rd, f'{fname}.log')
        if not os.path.isfile(path):
            return self._json({'lines': [], 'offset': 0, 'eof': True})

        size = os.path.getsize(path)
        if offset > size:      # 文件被截断/换了一轮
            offset = 0
        with open(path, 'rb') as f:
            f.seek(offset)
            raw = f.read()

        # 偏移必须按字节算。之前是 decode(errors='replace') 后再 encode 求长度，
        # 碰到中文半行时字节数对不上，会漏行或重复行——日志就“不可信”了。
        cut = raw.rfind(b'\n')
        if cut < 0:
            complete = b''
        else:
            complete = raw[:cut + 1]
        offset += len(complete)
        chunk = complete.decode('utf-8', errors='replace')
        lines = chunk.split('\n')
        if lines and lines[-1] == '':
            lines.pop()
        lines = [ln.rstrip('\r') for ln in lines]

        if only_dec:
            lines = [ln for ln in lines if '[DEC]' in ln]

        return self._json({'lines': lines, 'offset': offset, 'eof': False})


# ---------------------------------------------------------------- 端口
PORT = 0


def bind_server(host, start_port, tries=10):
    global PORT
    for p in range(start_port, start_port + tries):
        try:
            srv = ThreadingHTTPServer((host, p), Handler)
            PORT = srv.server_address[1]
            return srv
        except OSError:
            continue
    raise RuntimeError(f'{start_port}-{start_port + tries - 1} 全部被占用')


def main():
    global NODE
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--preview', action='store_true', help='离线预览模式（不初始化 ROS2）')
    args = ap.parse_args()

    os.makedirs(STATE_DIR, exist_ok=True)

    preview_mode = bool(args.preview or not HAS_ROS)
    if not preview_mode:
        rclpy.init()
        NODE = PanelNode()
        threading.Thread(target=rclpy.spin, args=(NODE,), daemon=True).start()

    srv = bind_server(args.host, args.port)
    _write_state_file('panel_port', str(PORT))
    _write_state_file('started', '0')
    mode_hint = '(离线预览模式，无 ROS2)' if preview_mode else '(发车后只读)'
    print(f'[panel] http://{args.host}:{PORT}/  {mode_hint}', flush=True)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if RECORDER.is_active:
            RECORDER.stop()
        srv.server_close()
        if NODE is not None:
            NODE.destroy_node()
        if HAS_ROS and rclpy is not None and rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
