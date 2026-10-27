#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并发扫 target_speed：每个速度跑一个独立 ROS_DOMAIN_ID 的闭环，互不干扰。

    python3 tools/sim/sweep.py                      # 默认 0.35 0.40 0.45 0.50
    python3 tools/sim/sweep.py --speeds 0.30 0.40   # 自定义
    python3 tools/sim/sweep.py --track roundabout

判定一档"通过"需要同时满足：
    - 4 根黄线都触发（final_stop）
    - 累计出界时长 < 1.0 s
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
TRACK = "fork_s"
SPEEDS = [0.35, 0.40, 0.45, 0.50]
TIMEOUT = 120.0
results = {}
lock = threading.Lock()


def make_yaml(v, base):
    txt = Path(base).read_text(encoding="utf-8")
    txt = re.sub(r"(target_speed:\s*)[0-9.]+", rf"\g<1>{v}", txt)
    txt = re.sub(r"(max_speed:\s*)[0-9.]+", rf"\g<1>{v}", txt)
    txt = re.sub(r"(output_topic:\s*)/cmd_vel", r"\g<1>/sim/cmd_vel", txt)
    p = Path(f"/tmp/sweep_{v:.2f}.yaml")
    p.write_text(txt, encoding="utf-8")
    return p


def parse_follow(path):
    txt = Path(path).read_text(errors="ignore") if Path(path).exists() else ""
    t0 = None
    m = re.search(r"\[(\d+\.\d+)\].*Start signal changed: false -> true", txt)
    if m:
        t0 = float(m.group(1))
    stops = [float(x) for x in re.findall(r"\[(\d+\.\d+)\].*yellow stop_begin", txt)]
    finals = [float(x) for x in re.findall(r"\[(\d+\.\d+)\].*yellow final_stop", txt)]
    return t0, stops, (finals[-1] if finals else None)


def parse_world(path):
    txt = Path(path).read_text(errors="ignore") if Path(path).exists() else ""
    offs = [float(x) for x in re.findall(r"\[sim\] s=[0-9.-]+ off=([0-9.]+)", txt)]
    ss = [float(x) for x in re.findall(r"\[sim\] s=([0-9.]+)", txt)]
    return (max(offs) if offs else 0.0), (max(ss) if ss else 0.0)


def kill_tree(proc):
    if proc is None:
        return
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            pass
        time.sleep(1.2)


def run_one(v, idx, track, base_yaml):
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(40 + idx)
    yaml = make_yaml(v, base_yaml)
    wlog = f"/tmp/sweep_{v:.2f}_world.log"
    flog = f"/tmp/sweep_{v:.2f}_follow.log"

    world = fol = None
    try:
        fw = open(wlog, "w")
        world = subprocess.Popen(
            [sys.executable, "-u", str(WS / "tools/sim/world_node.py"),
             "--track", str(WS / f"tools/sim/tracks/{track}.json")],
            env=env, stdout=fw, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(2.5)

        ff = open(flog, "w")
        fol = subprocess.Popen(
            ["ros2", "launch", "line_follower",
             "line_follower_with_detection.launch.py",
             f"params_file:={yaml}", "cmd_vel_topic:=/sim/cmd_vel"],
            env=env, stdout=ff, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(9.0)

        subprocess.run(
            ["ros2", "topic", "pub", "-r", "10", "--times", "20",
             "/line_follower/start_cmd", "std_msgs/msg/Bool", "{data: true}"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)

        t_begin = time.time()
        final = t0 = None
        while time.time() - t_begin < TIMEOUT:
            t0, stops, final = parse_follow(flog)
            if final:
                break
            time.sleep(0.5)
        off, smax = parse_world(wlog)
        lap = (final - t0) if (final and t0) else None
        ok = bool(final) and off < 1.0
        with lock:
            results[v] = {"lap": lap, "stops": len(stops), "off": off,
                          "smax": smax, "ok": ok}
    except Exception as e:
        with lock:
            results[v] = {"lap": None, "stops": 0, "off": 0.0, "smax": 0.0,
                          "ok": False, "err": str(e)}
    finally:
        kill_tree(fol)
        kill_tree(world)


def main():
    global TRACK
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=float, nargs="+", default=SPEEDS)
    ap.add_argument("--track", default=TRACK)
    ap.add_argument("--yaml", default=str(WS / "tools/config/fork.yaml"))
    args = ap.parse_args()
    TRACK = args.track

    leftover = subprocess.run(["pgrep", "-af", "world_node.py|line_follower_node"],
                              capture_output=True, text=True).stdout.strip()
    if leftover:
        print("[ERROR] 已有仿真/巡线进程在跑，先停掉：")
        print(leftover)
        print("  pkill -f world_node.py; pkill -INT -f line_follower")
        return 1

    print(f"并发 {len(args.speeds)} 组：{args.speeds}  track={args.track}")
    print("每个速度一个独立 ROS_DOMAIN_ID，互不干扰。\n")

    threads = []
    for i, v in enumerate(args.speeds):
        t = threading.Thread(target=run_one, args=(v, i, args.track, args.yaml))
        t.start()
        threads.append(t)
        time.sleep(1.5)   # 错开启动，避免 4 组同时抢 CPU
    for t in threads:
        t.join()

    print("\n" + "=" * 66)
    print(f"{'speed':>6} {'圈速(s)':>9} {'停车':>5} {'出界(s)':>8} {'里程(m)':>8} {'判定':>6}")
    print("-" * 66)
    best = None
    for v in args.speeds:
        r = results.get(v, {})
        lap = r.get("lap")
        lap_s = f"{lap:.2f}" if lap else "TIMEOUT"
        verdict = "PASS" if r.get("ok") else "FAIL"
        if r.get("ok") and lap and (best is None or lap < best[1]):
            best = (v, lap)
        print(f"{v:6.2f} {lap_s:>9} {r.get('stops',0):5d} "
              f"{r.get('off',0.0):8.2f} {r.get('smax',0.0):8.2f} {verdict:>6}")
    print("=" * 66)
    if best:
        print(f"最快通过档: {best[0]:.2f} m/s -> {best[1]:.2f} s")
    else:
        print("没有一档通过，检查参数或赛道")
    print("\n日志: /tmp/sweep_<speed>_world.log / _follow.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
