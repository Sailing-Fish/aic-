#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多参数并发搜索：目标是在不丢线/不出界的前提下把圈速压到最低。

为什么不只扫车速：kp(转向增益)、speed_gain_damping(误差降速)、max_yaw_rate(角速度上限)
与车速强耦合。同一套 kp 在 0.3 稳、在 0.45 会震荡；同一套 damping 在低速无所谓、
高速会把弯道速度压死。只扫车速等于用错误的控制参数去评价速度。

搜索空间（基于物理约束裁剪）：
    target_speed / max_speed   0.36 – 0.48   上限来自实测 Rmin 0.36m 与视觉延迟
    kp                         2.0  – 3.5    低于 2 切弯，高于 3.5 高速易震荡
    speed_gain_damping         0.30 – 0.70   0.65 时误差 0.6 只剩 61% 速度，弯道被压死
    max_yaw_rate               1.0  – 1.5    0.4m/s 时物理上限约 1.04–1.16，1.5 必然饱和
    yellow_stop_duration       固定 0.85     影响总时间、不影响稳定性，单独算

用法：
    python3 tools/sim/tune.py                    # 默认 16 组，4 并发
    python3 tools/sim/tune.py --n 12 --jobs 3
    python3 tools/sim/tune.py --track roundabout
"""
import argparse
import csv
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
TIMEOUT = 110.0
results = {}
lock = threading.Lock()


def set_in_section(txt, key, val, section="line_follower_node"):
    """只改指定节点段里的参数，避免误改 detection_turn_controller 的同名项。"""
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
    p = Path(f"/tmp/tune_{tag}.yaml")
    p.write_text(txt, encoding="utf-8")
    return p


def parse_follow(path):
    txt = Path(path).read_text(errors="ignore") if Path(path).exists() else ""
    m = re.search(r"\[(\d+\.\d+)\].*Start signal changed: false -> true", txt)
    t0 = float(m.group(1)) if m else None
    stops = re.findall(r"yellow stop_begin", txt)
    finals = re.findall(r"\[(\d+\.\d+)\].*yellow final_stop", txt)
    return t0, len(stops), (float(finals[-1]) if finals else None)


def parse_world(path):
    """返回 (最大出界秒数, 最大里程, 实测帧率)。

    帧率从每 30 帧一行的 [sim] 日志间隔反推：间隔 >1.11s 即低于 27Hz，
    说明这组被 CPU 竞争拖慢，圈速不可与其他批比较。
    """
    txt = Path(path).read_text(errors="ignore") if Path(path).exists() else ""
    offs = [float(x) for x in re.findall(r"\[sim\] s=[0-9.-]+ off=([0-9.]+)", txt)]
    ss = [float(x) for x in re.findall(r"\[sim\] s=([0-9.]+)", txt)]
    ts = [float(x) for x in
          re.findall(r"\[(\d+\.\d+)\] \[sim_world\]: \[sim\] s=", txt)]
    fps = 0.0
    if len(ts) >= 3:
        gaps = [b - a for a, b in zip(ts, ts[1:]) if 0.2 < b - a < 6.0]
        if gaps:
            fps = 30.0 / (sum(gaps) / len(gaps))
    return (max(offs) if offs else 0.0), (max(ss) if ss else 0.0), fps


def kill_tree(proc):
    if proc is None:
        return
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            pass
        time.sleep(1.0)


def run_one(cfg, idx, track, base_yaml):
    tag = f"{cfg['speed']:.2f}_{cfg['kp']:.2f}_{cfg['damp']:.2f}_{cfg['yaw']:.2f}"
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(50 + idx)
    yaml = make_yaml(cfg, base_yaml, tag)
    wlog, flog = f"/tmp/tune_{tag}_world.log", f"/tmp/tune_{tag}_follow.log"
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
        nstop = 0
        while time.time() - t_begin < TIMEOUT:
            t0, nstop, final = parse_follow(flog)
            if final:
                break
            time.sleep(0.5)
        off, smax = parse_world(wlog)
        lap = (final - t0) if (final and t0) else None
        ok = bool(final) and off < 1.0
        with lock:
            results[tag] = {**cfg, "lap": lap, "stops": nstop, "off": off,
                            "smax": smax, "ok": ok}
    except Exception as e:
        with lock:
            results[tag] = {**cfg, "lap": None, "stops": 0, "off": 0.0,
                            "smax": 0.0, "ok": False, "err": str(e)}
    finally:
        kill_tree(fol)
        kill_tree(world)


def build_configs(n_random, seed=42):
    base = {"speed": 0.40, "kp": 2.45, "damp": 0.65, "yaw": 1.5, "stop": 0.85}
    cfgs = [dict(base, speed=s) for s in (0.40, 0.45)]
    rng = random.Random(seed)
    for _ in range(n_random):
        cfgs.append({
            "speed": round(rng.uniform(0.36, 0.48), 2),
            "kp": round(rng.uniform(2.0, 3.5), 2),
            "damp": round(rng.uniform(0.30, 0.70), 2),
            "yaw": round(rng.uniform(1.0, 1.5), 2),
            "stop": 0.85,
        })
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14, help="随机组数（另加 2 组基准）")
    ap.add_argument("--jobs", type=int, default=4, help="并发数，8 核建议不超过 4")
    ap.add_argument("--track", default="fork_s")
    ap.add_argument("--yaml", default=str(WS / "tools/config/fork.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    leftover = subprocess.run(["pgrep", "-af", "world_node.py|line_follower_node"],
                              capture_output=True, text=True).stdout.strip()
    if leftover:
        print("[ERROR] 已有仿真/巡线进程，先停：")
        print(leftover)
        return 1

    cfgs = build_configs(args.n, args.seed)
    total = len(cfgs)
    print(f"共 {total} 组，{args.jobs} 并发，赛道 {args.track}，预计 "
          f"{int((total + args.jobs - 1) / args.jobs * 60 / 60)}–"
          f"{int((total + args.jobs - 1) / args.jobs * 70 / 60) + 2} 分钟\n")

    t_all = time.time()
    for b in range(0, total, args.jobs):
        batch = cfgs[b:b + args.jobs]
        threads = []
        for i, cfg in enumerate(batch):
            t = threading.Thread(target=run_one, args=(cfg, i, args.track, args.yaml))
            t.start()
            threads.append(t)
            time.sleep(1.5)
        for t in threads:
            t.join()
        print(f"  batch {b // args.jobs + 1}/{(total + args.jobs - 1) // args.jobs} 完成 "
              f"({time.time() - t_all:.0f}s)", flush=True)

    rows = list(results.values())
    rows.sort(key=lambda r: (not r["ok"], r["lap"] if r["lap"] else 9e9))
    print("\n" + "=" * 88)
    print(f"{'speed':>6} {'kp':>5} {'damp':>5} {'yaw':>5} {'圈速':>8} "
          f"{'停车':>5} {'出界':>6} {'里程':>6} {'fps':>5} {'判定':>7}")
    print("-" * 92)
    for r in rows:
        lap = f"{r['lap']:.2f}" if r["lap"] else "TIMEOUT"
        print(f"{r['speed']:6.2f} {r['kp']:5.2f} {r['damp']:5.2f} {r['yaw']:5.2f} "
              f"{lap:>8} {r['stops']:5d} {r['off']:6.2f} {r['smax']:6.2f} "
              f"{r.get('fps', 0):5.1f} {'PASS' if r['ok'] else 'FAIL':>7}")
    print("=" * 88)

    passed = [r for r in rows if r["ok"]]
    if passed:
        best = passed[0]
        print(f"\n最快通过: speed={best['speed']} kp={best['kp']} "
              f"damp={best['damp']} yaw={best['yaw']} -> {best['lap']:.2f}s")
        # 平台区：找 speed 差 <=0.02、kp 差 <=0.25、damp 差 <=0.05 的邻居
        neigh = [r for r in passed if r is not best
                 and abs(r["speed"] - best["speed"]) <= 0.02
                 and abs(r["kp"] - best["kp"]) <= 0.25
                 and abs(r["damp"] - best["damp"]) <= 0.05]
        if neigh:
            avg = sum(r["lap"] for r in neigh) / len(neigh)
            print(f"  邻近 {len(neigh)} 组平均 {avg:.2f}s "
                  f"(差 {abs(avg - best['lap']):.2f}s) -> "
                  f"{'平台区，可信' if abs(avg - best['lap']) < 2.0 else '孤立尖峰，慎用'}")
        else:
            print("  无邻近样本，无法判断是平台还是孤立尖峰，建议加密该区域")
    else:
        print("\n没有一组通过，需要放宽搜索范围或检查赛道")

    out = WS / "tools/sim/profiles" / f"tune_{args.track}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["speed", "kp", "damp", "yaw", "stop",
                                          "lap", "stops", "off", "smax", "ok"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
