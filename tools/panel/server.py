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
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import yaml
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import params_schema  # noqa: E402

WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
TOOLS = os.path.join(WS, 'tools')
STATE_DIR = '/tmp/rdkx5'
LOGROOT = os.path.join(TOOLS, 'logs')
CONFIG_DIR = os.path.join(TOOLS, 'config')
PANEL_DIR = os.path.dirname(os.path.abspath(__file__))

STREAM_FPS = 6.0
STREAM_PERIOD = 1.0 / STREAM_FPS
JPEG_QUALITY = 70


# ---------------------------------------------------------------- 共享状态
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = {'debug': None, 'yolo': None}
        self.started = False
        self.locked = False       # 参数是否锁定：只在"真正发车"的上升沿置位
        self.track_state = 'unknown'
        self.yellow_count = 0
        self.yellow_permanent = 4
        self.timer_start = None
        self.timer_stop = None
        self.last_yellow_ts = 0.0
        self.final_stopped = False
        self.round = 1


SH = Shared()


def reset_round_locked():
    """重置本轮统计。只动面板自己的数字，不碰车（调用方需持锁）。"""
    SH.timer_start = None
    SH.timer_stop = None
    SH.yellow_count = 0
    SH.last_yellow_ts = 0.0
    SH.final_stopped = False
    SH.track_state = 'unknown'
    SH.round += 1


# ---------------------------------------------------------------- ROS 节点
class PanelNode(Node):
    def __init__(self):
        super().__init__('panel_node')
        self.bridge = CvBridge()
        self._last_started = False

        self.create_subscription(Image, 'line_follower/debug_image', self._on_debug, 5)
        self.create_subscription(Image, '/yolo_detection_image', self._on_yolo, 5)
        self.create_subscription(Bool, 'line_follower/start_cmd', self._on_start, 10)
        self.create_subscription(String, 'line_follower/track_state', self._on_state, 10)
        self.estop_pub = self.create_publisher(Twist, '/cmd_vel_stop', 10)

        self.create_timer(0.2, self._tick)
        self.get_logger().info('panel_node ready')

    def _encode(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def _on_debug(self, msg):
        data = self._encode(msg)
        if data:
            with SH.lock:
                SH.frames['debug'] = data

    def _on_yolo(self, msg):
        data = self._encode(msg)
        if data:
            with SH.lock:
                SH.frames['yolo'] = data

    def _on_start(self, msg):
        new_val = bool(msg.data)

        # start_toggle 以 10Hz 重复发布同一个值，这里只认跳变沿。
        # 否则：急停/重置把面板解成"未发车"后，下一条重复的 true 会立刻把它锁回去。
        if new_val == self._last_started:
            return
        self._last_started = new_val

        with SH.lock:
            if new_val:
                # 上升沿 = 发车 → 新的一轮。
                # 必须把 last_yellow_ts / track_state 也清掉：
                # 上一轮永久停车后 state 会一直停在 yellow_line_detected，
                # 不清的话计时器会在新一圈刚开始就立刻被判定"终点已到"而冻结。
                SH.timer_start = time.time()
                SH.timer_stop = None
                SH.final_stopped = False
                SH.yellow_count = 0
                SH.last_yellow_ts = 0.0
                SH.track_state = 'unknown'
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

    def _tick(self):
        # 永久停车的判据：黄线状态持续 >2s（临时停车只有 yellow_stop_duration≈1s）
        with SH.lock:
            if (SH.track_state == 'yellow_line_detected'
                    and SH.last_yellow_ts > 0
                    and not SH.final_stopped
                    and time.time() - SH.last_yellow_ts > 2.0
                    and SH.timer_start is not None
                    and SH.last_yellow_ts >= SH.timer_start   # 只认本轮之后的黄线
                    and SH.timer_stop is None):
                SH.timer_stop = SH.last_yellow_ts
                SH.final_stopped = True

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


def read_effective_params():
    """解析 follow.log 里节点启动时打印的「[DEC] param effective ...」行。

    这是"真正生效的值"，和 YAML 里的值对比就能知道要不要重启节点。
    """
    rd = run_dir()
    if not rd:
        return {}
    path = os.path.join(rd, 'follow.log')
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 30000))
            text = f.read().decode('utf-8', errors='replace')
    except OSError:
        return {}

    result = {}
    for line in text.splitlines():
        if 'param effective' not in line:
            continue
        result = {}
        for tok in line.split('param effective', 1)[1].split():
            if '=' in tok:
                k, v = tok.split('=', 1)
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
    return result


def read_recent_events(limit=10):
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
        parts = line.rstrip('\n').split(',')
        if len(parts) >= 6:
            out.append({'t': parts[1], 'source': parts[2], 'area': parts[3],
                        'event': parts[4], 'kv': parts[5]})
    return out


def load_params(map_name):
    path = os.path.join(CONFIG_DIR, f'{map_name}.yaml')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    groups = []
    for g in params_schema.SCHEMA:
        sec = g['section']
        cur = (data.get(sec) or {}).get('ros__parameters') or {}
        items = []
        for key, label, typ, lo, hi, step, helptext in g['items']:
            if key not in cur:
                continue
            items.append({
                'key': key, 'label': label, 'type': typ,
                'min': lo, 'max': hi, 'step': step,
                'value': cur[key], 'help': helptext,
            })
        if items:
            groups.append({'section': sec, 'group': g['group'], 'items': items})
    return {'map': map_name, 'groups': groups}


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
            spec = params_schema.find(section, key)
            if spec is None:
                continue
            _, _, typ, lo, hi, _step, _help = spec
            if typ == 'bool':
                val = bool(val)
            elif typ == 'int':
                val = int(round(float(val)))
            else:
                val = float(val)
            if typ in ('int', 'float') and not (lo <= val <= hi):
                return False, f'{key}={val} 超出范围 [{lo}, {hi}]'
            if params.get(key) != val:
                changed.append(f'{key}:{params.get(key)}->{val}')
            params[key] = val

    if not changed:
        return True, 'no change'

    with open(path + '.bak', 'w', encoding='utf-8') as f:
        yaml.safe_dump(backup, f, allow_unicode=True, sort_keys=False)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print(f'[DEC] panel param_write map={map_name} ' + ' '.join(changed), flush=True)
    return True, '; '.join(changed)


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
                    query[k] = v

        try:
            if path == '/':
                return self._index()
            if path == '/api/status':
                return self._status()
            if path == '/api/stream/debug':
                return self._stream('debug')
            if path == '/api/stream/yolo':
                return self._stream('yolo')
            if path == '/api/logs':
                return self._logs(query)
            if path == '/api/params':
                return self._json(load_params(query.get('map', 'fork')))
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
                ok, msg = save_params(body.get('map', 'fork'), body.get('values', {}))
                return self._json({'ok': ok, 'msg': msg}, 200 if ok else 400)

            if path.startswith('/api/restart/'):
                if self._is_running():
                    return self._err(423, '已发车，禁止重启节点')
                name = path.rsplit('/', 1)[-1]
                allowed = {'base', 'camera', 'yolo', 'follow', 'panel', 'safety', 'collector'}
                if name not in allowed:
                    return self._err(400, f'未知进程 {name}')
                with open(os.path.join(STATE_DIR, 'cmd.restart'), 'w', encoding='utf-8') as f:
                    f.write(name)
                print(f'[DEC] panel restart name={name}', flush=True)
                return self._json({'ok': True, 'msg': f'已请求重启 {name}'})

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
                threading.Thread(target=NODE.estop, daemon=True).start()
                with SH.lock:
                    SH.started = False
                    SH.locked = False
                _write_state_file('started', '0')
                print('[DEC] panel estop', flush=True)
                return self._json({'ok': True, 'msg': '已发送 30 帧零速'})

            return self._err(404, 'not found')
        except Exception as e:  # noqa: BLE001
            return self._err(500, f'{type(e).__name__}: {e}')

    # ---------- handlers
    def _index(self):
        p = os.path.join(PANEL_DIR, 'index.html')
        with open(p, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _status(self):
        st = read_status()
        with SH.lock:
            started = SH.started
            locked = SH.locked
            state = SH.track_state
            yc = SH.yellow_count
            yp = SH.yellow_permanent
            ts, te = SH.timer_start, SH.timer_stop
            rnd = SH.round
            frames_ok = {k: (v is not None) for k, v in SH.frames.items()}

        if ts is None:
            elapsed = 0.0
        elif te is None:
            elapsed = time.time() - ts
        else:
            elapsed = te - ts

        self._json({
            'ok': True,
            'global': st.get('global', 'down'),
            'run_id': st.get('run_id'),
            'params_file': st.get('params_file'),
            'procs': st.get('procs', {}),
            'started': started,
            'locked': locked,
            'track_state': state,
            'yellow': {'count': yc, 'permanent': yp},
            'timer': {'elapsed': round(elapsed, 2), 'running': ts is not None and te is None,
                      'final': te is not None, 'armed': ts is not None},
            'round': rnd,
            'effective': read_effective_params(),
            'events': read_recent_events(10),
            'frames': frames_ok,
            'panel_port': PORT,
        })

    def _stream(self, name):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            while True:
                with SH.lock:
                    data = SH.frames.get(name)
                if data:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(data)}\r\n\r\n'.encode())
                    self.wfile.write(data)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                time.sleep(STREAM_PERIOD)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _logs(self, query):
        rd = run_dir()
        if rd is None:
            return self._json({'lines': [], 'offset': 0, 'eof': True})
        fname = query.get('f', 'follow')
        only_dec = query.get('dec', '0') == '1'
        offset = int(query.get('offset', '0') or 0)
        path = os.path.join(rd, f'{fname}.log')
        if not os.path.isfile(path):
            return self._json({'lines': [], 'offset': 0, 'eof': True})

        size = os.path.getsize(path)
        if offset > size:      # 文件被截断/换了一轮
            offset = 0
        with open(path, 'rb') as f:
            f.seek(offset)
            raw = f.read()
        chunk = raw.decode('utf-8', errors='replace')
        lines = chunk.splitlines()
        if lines and not chunk.endswith('\n'):
            # 最后一行可能不完整，只推进到它的起点，下次再读
            offset += len(raw) - len(lines[-1].encode('utf-8'))
            lines = lines[:-1]
        else:
            offset += len(raw)

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
            PORT = p
            return srv
        except OSError:
            continue
    raise RuntimeError(f'{start_port}-{start_port + tries - 1} 全部被占用')


def main():
    global NODE
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    args = ap.parse_args()

    os.makedirs(STATE_DIR, exist_ok=True)

    rclpy.init()
    NODE = PanelNode()
    threading.Thread(target=rclpy.spin, args=(NODE,), daemon=True).start()

    # 黄线总数从参数文件取，供面板显示 n/N
    try:
        for g in load_params('fork')['groups']:
            for it in g['items']:
                if it['key'] == 'yellow_permanent_count':
                    SH.yellow_permanent = int(it['value'])
    except Exception:
        pass

    srv = bind_server(args.host, args.port)
    _write_state_file('panel_port', str(PORT))
    _write_state_file('started', '0')
    print(f'[panel] http://{args.host}:{PORT}/  (发车后只读)', flush=True)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        NODE.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
