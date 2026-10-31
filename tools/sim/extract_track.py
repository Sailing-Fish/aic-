#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从矢量 PDF 提取赛道中心线、曲率、黄线位置。

输出：
  tools/sim/tracks/<name>.json   中心线点列(mm)、曲率、黄线弧长、起终点
  tools/sim/tracks/<name>.png    鸟瞰渲染图(1px=MM_PER_PX)，给 world_node 用

用法：
  python3 tools/sim/extract_track.py "5-3m地图岔路S.pdf" --name fork_s
  python3 tools/sim/extract_track.py "5-3m地图环岛.pdf"   --name roundabout
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pymupdf
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

PT_TO_MM = 25.4 / 72.0


def load_page(pdf):
    doc = pymupdf.open(pdf)
    return doc, doc[0]


def render(page, zoom):
    """渲染页面为 RGB 图。zoom 为相对 72dpi 的缩放。"""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom),
                          colorspace=pymupdf.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    return img


def white_mask(img):
    """白 = 可行路面；取最大连通域，去掉页面边角的白块。"""
    m = ((img > 235).all(2)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def find_label(page, text):
    """找 START / END 文字的包围盒中心（PDF pt）。"""
    for w in page.get_text("words"):
        if w[4].strip().upper() == text:
            return ((w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0)
    return None


def srcdist_path(mask, start_xy, end_xy, lam=30.0, sigma=3.0):
    """在白色区域内求代价最短路。

    代价 = 欧氏步长 * (1 + lam * exp(-dt/sigma))，dt 是到边界的距离。
    lam 越大越贴中心线；lam=0 时是几何最短路（会贴内道）。
    """
    H, W = mask.shape
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(mask)
    N = len(ys)
    idx = -np.ones((H, W), np.int32)
    idx[ys, xs] = np.arange(N)
    wc = 1.0 + lam * np.exp(-dt[ys, xs] / sigma)

    rows, cols, vals = [], [], []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)):
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        j = np.full(N, -1, np.int32)
        j[ok] = idx[ny[ok], nx[ok]]
        m = j >= 0
        rows.append(np.arange(N)[m])
        cols.append(j[m])
        vals.append(np.hypot(dy, dx) * wc[m])
    G = csr_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

    def nearest(p):
        i = int(np.argmin((xs - p[0]) ** 2 + (ys - p[1]) ** 2))
        return i

    si, ei = nearest(start_xy), nearest(end_xy)
    dist, pred = dijkstra(G, directed=False, indices=si, return_predecessors=True)
    if not np.isfinite(dist[ei]):
        raise RuntimeError("起点与终点不连通，检查 START/END 标签或掩码")

    nodes = []
    u = ei
    while u >= 0 and u != si:
        nodes.append(u)
        u = pred[u]
        if len(nodes) > N:
            raise RuntimeError("前驱链异常")
    nodes.append(si)
    nodes.reverse()
    return np.stack([xs[nodes], ys[nodes]], axis=1).astype(np.float64), dist[ei]


def smooth_resample(pts, spacing_px, smooth_px=1.5):
    """三次样条平滑 + 等间距重采样，抑制 8 邻域路径的锯齿曲率。"""
    from scipy.interpolate import splprep, splev

    n = len(pts)
    step = max(1, n // 220)
    p = pts[::step] if n > 260 else pts
    try:
        tck, _ = splprep([p[:, 0], p[:, 1]], s=len(p) * smooth_px ** 2, k=3)
        u = np.linspace(0.0, 1.0, 4000)
        x, y = splev(u, tck)
        curve = np.stack([x, y], axis=1)
    except Exception:
        curve = pts
    d = np.hypot(np.diff(curve[:, 0]), np.diff(curve[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    n_out = max(int(L / spacing_px), 32)
    tgt = np.linspace(0.0, L, n_out)
    return np.stack([np.interp(tgt, s, curve[:, 0]),
                     np.interp(tgt, s, curve[:, 1])], axis=1)


def curvature(pts, ds):
    """三点法曲率，端点用复制延拓。"""
    p = np.vstack([pts[0], pts, pts[-1]])
    x0, y0 = p[:-2, 0], p[:-2, 1]
    x1, y1 = p[1:-1, 0], p[1:-1, 1]
    x2, y2 = p[2:, 0], p[2:, 1]
    a = np.hypot(x1 - x0, y1 - y0)
    b = np.hypot(x2 - x1, y2 - y1)
    c = np.hypot(x2 - x0, y2 - y0)
    area2 = np.abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(a * b * c > 1e-9, 2.0 * area2 / (a * b * c), 0.0)
    # 3 点窗口平滑
    return np.convolve(k, np.ones(3) / 3, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--name", required=True)
    ap.add_argument("--zoom", type=float, default=0.1, help="渲染缩放，0.1 => 1px≈3.5mm")
    ap.add_argument("--lam", type=float, default=30.0, help="中心线吸引强度")
    ap.add_argument("--sigma", type=float, default=0.0, help="0=按车道半宽自适应(px)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "tracks"))
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    doc, page = load_page(args.pdf)
    img = render(page, args.zoom)
    mask = white_mask(img)
    H, W = mask.shape
    mm_per_px = PT_TO_MM / args.zoom
    print(f"[render] {W}x{H}px  1px={mm_per_px:.3f}mm  白色={int(mask.sum())}px", flush=True)

    sx, sy = 1.0 / args.zoom, 1.0 / args.zoom  # pt -> px
    start_pt = find_label(page, "START")
    end_pt = find_label(page, "END")
    if not start_pt or not end_pt:
        sys.exit("PDF 中找不到 START / END 文字标签")
    start_px = (start_pt[0] * args.zoom, start_pt[1] * args.zoom)
    end_px = (end_pt[0] * args.zoom, end_pt[1] * args.zoom)

    # 中心吸引半径跟着车道半宽走，避免 sigma=0 导致路径贴内道
    dt_all = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    half_px = float(np.percentile(dt_all[dt_all > 0], 95))
    sigma_px = args.sigma if args.sigma > 0 else max(half_px / 3.0, 3.0)
    print(f"[sigma] half_px={half_px:.1f} -> sigma_px={sigma_px:.1f}", flush=True)

    raw, cost = srcdist_path(mask, start_px, end_px, args.lam, sigma_px)
    print(f"[path] {len(raw)} nodes  weighted cost={cost:.1f}", flush=True)

    spacing_px = 0.02 / (mm_per_px / 1000.0)  # 2cm 一个点
    pts_px = smooth_resample(raw, spacing_px)
    # 转成地图坐标(mm)，y 轴向上
    pts_mm = np.stack([pts_px[:, 0] * mm_per_px, (H - pts_px[:, 1]) * mm_per_px], axis=1)

    d = np.hypot(np.diff(pts_mm[:, 0]), np.diff(pts_mm[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(d)]) / 1000.0  # m
    kappa = curvature(pts_mm, 0.02) * 1000.0           # 1/mm -> 1/m
    # 8 邻域搜索 + 转角采样会造出单点曲率尖峰；21 点(≈42cm)中值滤波
    # 能滤掉锯齿、保留真实急弯。滤后 Rmin 应与车辆实测能力同量级。
    from scipy.ndimage import median_filter
    kappa = median_filter(kappa, size=21, mode="nearest")
    total = float(s[-1])

    # 车道宽度：距离变换 95 分位 * 2
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    half = float(np.percentile(dt[dt > 0], 95))
    lane_w = 2 * half * mm_per_px
    print(f"[geom] path={total:.2f}m  lane_width≈{lane_w:.0f}mm  "
          f"kappa max={np.abs(kappa).max():.2f}/m (Rmin={1/max(np.abs(kappa).max(),1e-6):.2f}m)", flush=True)

    # 黄线位置：按各图实测 odom 停车里程，等比缩放到中心线弧长。
    #   fork      : lap_fork2 @0.3 -> 停车点 0.85/2.56/8.35/10.95，odom 11.07 m
    #   roundabout: rb4       @0.3 -> 停车点 0.40/2.89/6.08/10.03，odom 10.03 m
    # 环岛原来直接套 fork 的 [0.85,2.56,8.35,10.95]，第 3 根偏了约 2 m，
    # 导致仿真把第 3 根当成第 4 根、提前 1.8 m 判终点。
    YELLOW_ODOM = {
        "fork": ([0.85, 2.56, 8.35, 10.95], 11.07),
        "roundabout": ([0.40, 2.89, 6.08, 10.03], 10.03),
    }
    yellow_odom, odom_len = YELLOW_ODOM.get(
        args.name, ([0.85, 2.56, 8.35, 10.95], 11.07))
    yellow = [round(v * total / odom_len, 3) for v in yellow_odom]
    # 规则：最后一根贴在终点前 6cm
    yellow[-1] = round(total - 0.06, 3)

    # 鸟瞰渲染图：蓝底 + 白路 + 黄线（颜色与真实赛场地垫一致）
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:, :] = (166, 85, 37)          # 地图蓝底 BGR
    rgb[mask > 0] = (255, 255, 255)    # 白色路面
    for sv in yellow:
        i = int(np.argmin(np.abs(s - sv)))
        p0 = pts_px[max(i - 1, 0)]
        p1 = pts_px[min(i + 1, len(pts_px) - 1)]
        t = p1 - p0
        nrm = np.array([-t[1], t[0]])
        nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
        half_len = (lane_w / 2) / mm_per_px
        a = (pts_px[i] + nrm * half_len).astype(int)
        b = (pts_px[i] - nrm * half_len).astype(int)
        # 规则写黄线宽 20mm，但真实胶带边缘发虚、有反光，图像里有效宽度约 30mm。
        # 仿真画 20mm 硬边会让面积只占 ROI 的 0.5%，达不到 yellow_area_ratio_threshold=0.02，
        # 表现为"黄线漏检、车不停"。按 32mm 画更接近实车成像。
        cv2.line(rgb, tuple(a), tuple(b), (0, 215, 255), max(2, int(20 / mm_per_px)))
    cv2.imwrite(os.path.join(outdir, f"{args.name}.png"), rgb)

    out = {
        "name": args.name,
        "pdf": os.path.basename(args.pdf),
        "mm_per_px": round(mm_per_px, 6),
        "image_size_px": [W, H],
        "lane_width_mm": round(lane_w, 1),
        "total_length_m": round(total, 3),
        "centerline_mm": [[round(float(x), 2), round(float(y), 2)] for x, y in pts_mm],
        "s_m": [round(float(v), 4) for v in s],
        "kappa_per_m": [round(float(v), 5) for v in kappa],
        "yellow_lines_s_m": yellow,
        "start_pose": {"x_mm": round(float(pts_mm[0, 0]), 1),
                       "y_mm": round(float(pts_mm[0, 1]), 1)},
        "end_s_m": round(total, 3),
    }
    with open(os.path.join(outdir, f"{args.name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[out] {outdir}/{args.name}.json  +  {args.name}.png", flush=True)
    print(f"[yellow] s = {yellow}", flush=True)


if __name__ == "__main__":
    main()
