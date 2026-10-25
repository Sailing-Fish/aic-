#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由赛道曲率算时间最优速度剖面。

约束：
    1) 曲率限速  v <= sqrt(a_lat / |kappa|)
    2) 最高速    v <= v_max
    3) 黄线处    v = 0（停车）
    4) 加减速    |dv/dt| <= a_acc / a_dec

做法是经典的前向-后向扫描：先按曲率给上限，再从起点向前传播"能加速到多少"，
从终点向后传播"必须提前减到多少"。得到的 v(s) 在剩余约束下时间最短。

用法：
    python3 tools/sim/profile.py --track tools/sim/tracks/fork_s.json
    python3 tools/sim/profile.py --track ... --vmax 0.6 --alat 2.0 --stop 0.35
"""
import argparse
import json
import os

import numpy as np


def build_profile(s, kappa, yellow_s, v_max, a_lat, a_acc, a_dec,
                  stop_duration, stop_count, const_speed=None):
    """返回 (v, t_move, t_total)。

    停车区间（黄线前后各 0.12m）的速度取 0，且其时间不计入积分——
    实际停车时间由 stop_duration 单独累加，否则 0 速段会被当成爬行。
    """
    if const_speed is None:
        v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-4)))
    else:
        v = np.full(len(s), float(const_speed))
    ds = np.diff(s)

    stopped = np.zeros(len(s), bool)
    for ys in yellow_s[:stop_count]:
        i0 = int(np.searchsorted(s, ys - 0.12))
        i1 = int(np.searchsorted(s, ys + 0.12))
        v[max(i0, 0):max(i1, i0 + 1)] = 0.0
        stopped[max(i0, 0):max(i1, i0 + 1)] = True

    # 前向：受加速能力限制
    for i in range(len(v) - 1):
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2.0 * a_acc * ds[i]))
    # 后向：受减速能力限制
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_dec * ds[i]))

    v_avg = np.maximum(0.5 * (v[:-1] + v[1:]), 0.05)
    moving = ~(stopped[:-1] | stopped[1:])
    t_move = float(np.sum(ds[moving] / v_avg[moving]))
    t_total = t_move + stop_count * stop_duration
    return v, t_move, t_total


def constant_speed_time(s, v_const, yellow_s, stop_duration, stop_count):
    t_move = float(s[-1] / v_const)
    return t_move, t_move + stop_count * stop_duration


def summarize(s, v, name):
    """按直道/弯道分段汇总，给实车一个可读的速度表。"""
    kappa_ok = True
    segs = []
    cur = 0
    for i in range(1, len(v)):
        if abs(v[i] - v[i - 1]) > 0.08 and i - cur > 5:
            segs.append((s[cur], s[i], float(np.median(v[cur:i]))))
            cur = i
    segs.append((s[cur], s[-1], float(np.median(v[cur:]))))
    # 合并相近速度的段
    merged = []
    for a, b, vv in segs:
        if merged and abs(merged[-1][2] - vv) < 0.05 and a - merged[-1][1] < 0.6:
            merged[-1] = (merged[-1][0], b, 0.5 * (merged[-1][2] + vv))
        else:
            merged.append((a, b, vv))
    print(f"\n[{name}] 分段速度表:")
    print(f"  {'s_start':>8} {'s_end':>8} {'len_m':>7} {'v_target':>9}")
    for a, b, vv in merged:
        print(f"  {a:8.2f} {b:8.2f} {b-a:7.2f} {vv:9.2f}")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--vmax", type=float, default=0.8)
    ap.add_argument("--alat", type=float, default=1.5, help="横向加速度上限 m/s^2")
    ap.add_argument("--aacc", type=float, default=0.25, help="实测加速 0.15->0.4 约 1s")
    ap.add_argument("--adec", type=float, default=0.40, help="实测减速 0.4->0 约 1s")
    ap.add_argument("--stop", type=float, default=1.0, help="单次黄线停车时长 s")
    ap.add_argument("--stops", type=int, default=3, help="中间停车次数（不含终点）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.track, "r", encoding="utf-8") as f:
        d = json.load(f)
    s = np.array(d["s_m"])
    kappa = np.array(d["kappa_per_m"])
    yellow = d["yellow_lines_s_m"]

    v, t_move, t_total = build_profile(
        s, kappa, yellow, args.vmax, args.alat, args.aacc, args.adec,
        args.stop, args.stops)

    print(f"track={d['name']}  L={s[-1]:.2f}m  Rmin={1/np.abs(kappa).max():.2f}m  "
          f"yellow={[round(x,2) for x in yellow]}")
    print(f"params: vmax={args.vmax} a_lat={args.alat} a_acc={args.aacc} "
          f"a_dec={args.adec} stop={args.stop}s x{args.stops}")

    for vc in (0.3, 0.4, 0.5):
        tm, tt = constant_speed_time(s, vc, yellow, args.stop, args.stops)
        print(f"  恒速 {vc:.1f} m/s -> {tt:6.2f} s")

    print(f"  最优剖面      -> {t_total:6.2f} s  (行驶 {t_move:.2f} + 停车 "
          f"{args.stops*args.stop:.2f})")

    segs = summarize(s, v, "optimal")

    # 停稳即走：把 stop 从 1.0 降到 0.35 的收益
    _, _, t_short = build_profile(s, kappa, yellow, args.vmax, args.alat,
                                  args.aacc, args.adec, 0.35, args.stops)
    print(f"\n停车 1.0s -> 0.35s: {t_total:.2f} -> {t_short:.2f} s "
          f"(省 {t_total - t_short:.2f} s)")

    out = args.out or os.path.splitext(args.track)[0].replace(
        "/tracks/", "/profiles/") + f"_vmax{args.vmax:.2f}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "track": d["name"],
            "params": vars(args),
            "s_m": [round(float(x), 4) for x in s],
            "v_target": [round(float(x), 4) for x in v],
            "kappa_per_m": [round(float(x), 5) for x in kappa],
            "t_move_s": round(t_move, 3),
            "t_total_s": round(t_total, 3),
            "segments": [{"s0": round(a, 3), "s1": round(b, 3),
                          "v": round(vv, 3)} for a, b, vv in segs],
        }, f, ensure_ascii=False, indent=1)
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
