#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[DEC] 决策事件汇总器（单写者）。

tail 本次运行目录下的所有 *.log，把 `[DEC] ...` 行拆成结构化记录写入
events.csv，供赛后统计与面板取数。

行格式约定：
    [DEC] <area> <event> [k=v ...]

容错：若第二个 token 形如 k=v（说明没写 event），则 event 记为 "state"，
      该 token 一并当作 kv 处理。

CSV 列：ts_iso, t_rel, source, area, event, kv
  - t_rel = 相对本次运行启动的秒数，便于与视频时间轴对齐
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime


def load_start_time(run_dir):
    meta = os.path.join(run_dir, 'meta.json')
    try:
        with open(meta, 'r', encoding='utf-8') as f:
            dt = datetime.fromisoformat(json.load(f)['started_iso'])
        # meta.json 里是带时区的 ISO（date -Iseconds），统一转成本地 naive，
        # 否则和 datetime.now() 相减会抛 TypeError
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return dt
    except Exception:
        return datetime.now()


def parse_dec(line):
    """'[DEC] yellow stop_begin count=1/4' -> ('yellow','stop_begin','count=1/4')"""
    body = line.split('[DEC]', 1)[1].strip()
    if not body:
        return None
    parts = body.split()
    area = parts[0]
    if len(parts) == 1:
        return area, 'event', ''
    if '=' in parts[1]:
        return area, 'state', ' '.join(parts[1:])
    event = parts[1]
    kv = ' '.join(parts[2:])
    return area, event, kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--interval', type=float, default=0.5)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f'run dir not found: {run_dir}', file=sys.stderr)
        return 2

    csv_path = os.path.join(run_dir, 'events.csv')
    start = load_start_time(run_dir)

    new_file = not os.path.exists(csv_path)
    fh = open(csv_path, 'a', encoding='utf-8', newline='')
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(['ts_iso', 't_rel', 'source', 'area', 'event', 'kv'])
        fh.flush()

    offsets = {}
    self_name = os.path.basename(__file__).replace('.py', '.log')

    print(f'[collector] watching {run_dir}, csv={csv_path}')

    try:
        while True:
            for fn in sorted(os.listdir(run_dir)):
                if not fn.endswith('.log') or fn == self_name:
                    continue
                path = os.path.join(run_dir, fn)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                off = offsets.get(path, 0)
                if size <= off:
                    continue
                try:
                    # 二进制读，按字节记偏移；日志里有中文，文本模式 seek 不可靠
                    with open(path, 'rb') as f:
                        f.seek(off)
                        raw = f.read()
                except OSError:
                    continue
                offsets[path] = off + len(raw)
                chunk = raw.decode('utf-8', errors='replace')

                source = fn[:-4]
                for line in chunk.splitlines():
                    if '[DEC]' not in line:
                        continue
                    parsed = parse_dec(line)
                    if not parsed:
                        continue
                    area, event, kv = parsed
                    now = datetime.now()
                    t_rel = (now - start).total_seconds()
                    writer.writerow([
                        now.isoformat(timespec='milliseconds'),
                        f'{t_rel:.3f}',
                        source, area, event, kv,
                    ])
                fh.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
