#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""串行坐标下降扫参：一次只动一个维度，出问题立刻 kill 进入下一组。

为什么不是全网格：5 个参数各 5 档 = 3125 组 × 45 s ≈ 39 小时，不可行。
坐标下降把搜索压到 ~20 组（约 15 分钟），且每一步都能解释。

流程：
    阶段1  speed 0.26→0.40  找当前控制参数下的速度边界
    阶段2  kp    在最优速度下扫
    阶段3  damp
    阶段4  yaw
    阶段5  用调好的控制参数回头再试更高速度（控制变好后边界会上移）

快速失败（不等超时）：
    - 出界累计 > 1.0 s
    - 里程 5 s 不增长（丢线卡死）
    - 单组超过 75 s 未到终点

用法：
    python3 tools/sim/search.py
    python3 tools/sim/search.py --track roundabout
"""
import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
WL = "/tmp/search_world.log"
FL = "/tmp/search_follow.log"
BASE = {"speed": 0.26, "kp": 2.45, "damp": 0.65, "yaw": 1.5, "stop": 1.0}
MAX_RUN = 75.0
MAX_OFF = 1.0
STUCK_WIN = 5.0
STUCK_EPS = 0.05

history = []


def set_in_section(txt, key, val, section="line_follower_node"):
    lines = txt.split("\n")
    cur = None
    out = []
    for ln in lines:
        if ln and not ln[0].isspace() and ln.rstrip().endswith(":"):
            cur = ln.strip()[:-1]
        if cur == section and re.match(rf"\s*{key}:\s*-?[0-9.]+", ln):
            ln = re.sub(rf"({key}:\s*)-?[0-9.]+", rf"\g<1>{val}", ln)
        out.append(ln)
    return "\n".join(out)


def make_yaml(cfg, base, tag):
    txt = Path(base).read_text(encoding="utf-8")
    for k, v in (("target_speed", cfg["speed"]), ("max_speed", cfg["speed"]),
                 ("kp", cfg["kp"]), ("speed_gain_damping", cfg["damp"]),
                 ("max_yaw_rate", cfg["yaw"]),
                 ("yellow_stop_duration", cfg["stop"])):
        txt = set_in_section(txt, k, v)
    txt = re.sub(r"(output_topic:\s*)/cmd_vel", r"\g<1>/sim/cmd_vel", txt)
    p = Path(f"/tmp/search_{tag}.yaml")
    p.write_text(txt, encoding="utf-8")
    return p


def read_from(path, offset):
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", "ignore")


def parse_follow(txt):
    t0 = None
    m = re.search(r"\[(\d+\.\d+)\].*Start signal changed: false -> true", txt)
    if m:
        t0 = float(m.group(1))
    finals = re.findall(r"\[(\d+\.\d+)\].*yellow final_stop", txt)
    stops = len(re.findall(r"yellow stop_begin", txt))
    return t0, stops, (float(finals[-1]) if finals else None)


def parse_world(txt):
    offs = [float(x) for x in re.findall(r"\[sim\] s=[0-9.-]+ off=([0-9.]+)", txt)]
    ss = [float(x) for x in re.findall(r"\[sim\] s=([0-9.]+)", txt)]
    return (max(offs) if offs else 0.0), (max(ss) if ss else 0.0)


def kill_launch(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except Exception:
        pass
    time.sleep(1.0)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass


def run_one(cfg, base_yaml, track, idx):
    tag = f"{idx:02d}_{cfg['speed']:.2f}_{cfg['kp']:.2f}_{cfg['damp']:.2f}_{cfg['yaw']:.2f}"
    yaml = make_yaml(cfg, base_yaml, tag)

    # 世界节点常驻，先复位到起点。
    # --wait-matching-subscriptions 0：订阅没建立时不阻塞等待
    try:
        subprocess.run(["ros2", "topic", "pub", "--once",
                        "--wait-matching-subscriptions", "0",
                        "/sim/reset", "std_msgs/msg/Empty", "{}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=8)
    except subprocess.TimeoutExpired:
        return {**cfg, "lap": None, "why": "reset_fail", "off": 0.0, "s": 0.0}
    time.sleep(0.5)
    w_off = Path(WL).stat().st_size if Path(WL).exists() else 0

    fol = None
    try:
        ff = open(FL, "w")
        fol = subprocess.Popen(
            ["ros2", "launch", "line_follower",
             "line_follower_with_detection.launch.py",
             f"params_file:={yaml}", "cmd_vel_topic:=/sim/cmd_vel"],
            stdout=ff, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(9.0)
        if "Cmd mux" not in Path(FL).read_text(errors="ignore"):
            return {**cfg, "lap": None, "why": "launch_fail", "off": 0.0, "s": 0.0}

        subprocess.run(
            ["ros2", "topic", "pub", "-r", "10", "--times", "20",
             "/line_follower/start_cmd", "std_msgs/msg/Bool", "{data: true}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)

        t_begin = time.time()
        prev_s, prev_t = 0.0, t_begin
        while True:
            el = time.time() - t_begin
            ftxt = Path(FL).read_text(errors="ignore")
            t0, nstop, final = parse_follow(ftxt)
            off, smax = parse_world(read_from(WL, w_off))

            if final and t0:
                return {**cfg, "lap": round(final - t0, 2), "why": "done",
                        "off": round(off, 2), "s": round(smax, 2)}
            if off > MAX_OFF:
                return {**cfg, "lap": None, "why": f"off_track({off:.1f}s)",
                        "off": round(off, 2), "s": round(smax, 2)}
            if el > MAX_RUN:
                return {**cfg, "lap": None, "why": "timeout",
                        "off": round(off, 2), "s": round(smax, 2)}
            # 里程连续 STUCK_WIN 秒不增长才判卡死（黄线停车 1 s 不会误判）
            if smax - prev_s > STUCK_EPS:
                prev_s, prev_t = smax, time.time()
            elif time.time() - prev_t > STUCK_WIN:
                return {**cfg, "lap": None, "why": "stuck",
                        "off": round(off, 2), "s": round(smax, 2)}
            time.sleep(0.4)
    finally:
        kill_launch(fol)


def fmt(cfg):
    return (f"speed={cfg['speed']:.2f} kp={cfg['kp']:.2f} "
            f"damp={cfg['damp']:.2f} yaw={cfg['yaw']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="fork_s")
    ap.add_argument("--yaml", default=str(WS / "tools/config/fork.yaml"))
    ap.add_argument("--start", type=float, default=0.26)
    args = ap.parse_args()

    if subprocess.run(["pgrep", "-af", "world_node.py|line_follower_node"],
                      capture_output=True, text=True).stdout.strip():
        print("[ERROR] 已有仿真/巡线进程，先停干净再跑")
        return 1

    # 世界节点常驻：只启一次，后面每组用 /sim/reset 复位
    wlog = open(WL, "w")
    world = subprocess.Popen(
        [sys.executable, "-u", str(WS / "tools/sim/world_node.py"),
         "--track", str(WS / f"tools/sim/tracks/{args.track}.json")],
        stdout=wlog, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3.0)
    print(f"[world] 常驻启动，赛道 {args.track}\n", flush=True)

    best = dict(BASE, speed=args.start)
    idx = 0
    t_all = time.time()

    def step(params, label):
        nonlocal idx, best
        idx += 1
        t0 = time.time()
        r = run_one(params, args.yaml, args.track, idx)
        history.append(r)
        dt = time.time() - t0
        lap = f"{r['lap']:.2f}s" if r.get("lap") else r.get("why", "?")
        print(f"  [{idx:02d}] {fmt(params):48s} -> {lap:20s} "
              f"off={r.get('off', 0):.1f} s={r.get('s', 0):.1f} ({dt:.0f}s)",
              flush=True)
        return r

    try:
        # 阶段 1：速度边界
        print("阶段 1/5  找速度边界（其余沿用当前参数）", flush=True)
        speeds = [round(args.start + 0.02 * i, 2) for i in range(8)]
        best_lap, best_speed = None, None
        for v in speeds:
            r = step(dict(best, speed=v), "speed")
            if r.get("lap") and (best_lap is None or r["lap"] < best_lap):
                best_lap, best_speed = r["lap"], v
            elif r.get("why") in ("off_track", "stuck", "timeout") and best_speed:
                print(f"       速度 {v:.2f} 失败（{r['why']}），边界约 {best_speed:.2f}\n",
                      flush=True)
                break
        if best_speed:
            best["speed"] = best_speed
        print(f"  => 速度定为 {best['speed']:.2f}（{best_lap}s）\n", flush=True)

        # 阶段 2-4：控制参数
        for key, vals, name in (("kp", [2.0, 2.45, 2.9, 3.3], "kp"),
                                ("damp", [0.35, 0.50, 0.65], "damping"),
                                ("yaw", [1.00, 1.25, 1.50], "max_yaw_rate")):
            print(f"阶段  {name}", flush=True)
            cur = best[key]
            cand = [cur] + [v for v in vals if abs(v - cur) > 1e-6]
            bestv, bestl = cur, (best_lap if key == "kp" else None)
            for v in cand:
                r = step(dict(best, **{key: v}), key)
                if r.get("lap") and (bestl is None or r["lap"] < bestl):
                    bestl, bestv = r["lap"], v
            best[key] = bestv
            print(f"  => {name} 定为 {bestv}（{bestl}s）\n", flush=True)

        # 阶段 5：控制变好后，回头再试更高速度
        print("阶段 5/5  用调好的控制参数重试更高速度", flush=True)
        base_v = best["speed"]
        for v in (round(base_v + 0.02, 2), round(base_v + 0.04, 2)):
            r = step(dict(best, speed=v), "speed")
            if r.get("lap") and (best_lap is None or r["lap"] < best_lap):
                best_lap, best["speed"] = r["lap"], v
            else:
                print(f"       {v:.2f} 失败（{r.get('why')}），停在这里", flush=True)
                break
    finally:
        try:
            os.killpg(os.getpgid(world.pid), signal.SIGKILL)
        except Exception:
            pass
        wlog.close()

    print("\n" + "=" * 74)
    print("最终推荐")
    print("=" * 74)
    print(f"  {fmt(best)}")
    print(f"  预期圈速 {best_lap:.2f} s" if best_lap else "  无通过组合")
    print(f"  总耗时 {(time.time() - t_all) / 60:.1f} 分钟，共 {len(history)} 组")

    out = WS / "tools/sim/profiles" / f"search_{args.track}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["speed", "kp", "damp", "yaw", "stop",
                                          "lap", "why", "off", "s"])
        w.writeheader()
        for r in history:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
