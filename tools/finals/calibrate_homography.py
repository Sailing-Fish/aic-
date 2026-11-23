#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""10 分钟 IPM 地面单应标定工具 (4+ 点透视标定)。

坐标系约定（与 C++ GroundProjector 严格一致）：
    原点 (d_bumper_m = 0.0, y_lat_m = 0.0) = 车头泡沫防撞条中心在地面上的投影
    +d_bumper_m : 向前（米）
    +y_lat_m    : 向左（米，ROS 标准右手系）

使用方式：
  1) 交互点选模式（从静态截图或摄像头抓拍）：
     python3 tools/finals/calibrate_homography.py --image snapshot.png
     python3 tools/finals/calibrate_homography.py --device /dev/video0 --save-snap snap.png

  2) 无头 / 自动化模式（从 JSON 或 CSV 读取对应点）：
     python3 tools/finals/calibrate_homography.py --points calib_points.json --out homography.yaml

JSON 点集格式示例：
  [
    {"u": 220, "v": 420, "d_bumper_m": 0.30, "y_lat_m":  0.215},
    {"u": 420, "v": 420, "d_bumper_m": 0.30, "y_lat_m": -0.215},
    {"u": 265, "v": 270, "d_bumper_m": 1.00, "y_lat_m":  0.215},
    {"u": 375, "v": 270, "d_bumper_m": 1.00, "y_lat_m": -0.215}
  ]

CSV 点集格式示例（带表头）：
  u,v,d_bumper_m,y_lat_m
  220,420,0.30,0.215
  ...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# 推荐的地面标定靶点（米）：430mm 赛道左右边界 × 距离 0.30m / 0.60m / 1.00m
DEFAULT_TARGET_POINTS_M: List[Tuple[float, float]] = [
    (0.30, 0.215),   # 近左（距车头 30cm，左边界 +21.5cm）
    (0.30, -0.215),  # 近右（距车头 30cm，右边界 -21.5cm）
    (0.60, 0.215),   # 中左（距车头 60cm，左边界 +21.5cm）
    (0.60, -0.215),  # 中右（距车头 60cm，右边界 -21.5cm）
    (1.00, 0.215),   # 远左（距车头 100cm，左边界 +21.5cm）
    (1.00, -0.215),  # 远右（距车头 100cm，右边界 -21.5cm）
]


def pixel_to_ground(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    """将像素坐标 (u, v) 通过 3x3 单应矩阵映射到车头地面系 (d_bumper_m, y_lat_m)。"""
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    vec = H @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    if abs(vec[2]) < 1e-12:
        raise ValueError(f"Homography projection degenerate at ({u}, {v}): w={vec[2]}")
    return float(vec[0] / vec[2]), float(vec[1] / vec[2])


def ground_to_pixel(H_inv: np.ndarray, d_bumper_m: float, y_lat_m: float) -> Tuple[float, float]:
    """将地面坐标 (d_bumper_m, y_lat_m) 通过逆单应矩阵映射回像素 (u, v)。"""
    H_inv = np.asarray(H_inv, dtype=np.float64).reshape(3, 3)
    vec = H_inv @ np.array([float(d_bumper_m), float(y_lat_m), 1.0], dtype=np.float64)
    if abs(vec[2]) < 1e-12:
        raise ValueError(
            f"Inverse homography projection degenerate at ({d_bumper_m}, {y_lat_m}): w={vec[2]}"
        )
    return float(vec[0] / vec[2]), float(vec[1] / vec[2])


def solve_homography(
    img_pts: Sequence[Sequence[float]],
    ground_pts: Sequence[Sequence[float]],
    use_ransac: bool = False,
    ransac_thresh_m: float = 0.02,
) -> Dict[str, Any]:
    """由 >=4 组 (u, v) <-> (d_bumper_m, y_lat_m) 对应点求解 3x3 单应并评估误差。

    Returns:
        dict containing:
            H: 3x3 ndarray (float64)
            H_inv: 3x3 ndarray (float64)
            H_flat: list of 9 floats (row-major, ready for YAML)
            det: float determinant of H
            cond: float condition number of H
            errors_mm: list of per-point ground Euclidean errors in mm
            mean_err_mm: float
            max_err_mm: float
            reproj_px_errors: list of per-point inverse pixel errors in px
            mean_px_err: float
            max_px_err: float
    """
    src = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(ground_pts, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] < 4 or dst.shape[0] < 4:
        raise ValueError(f"至少需要 4 组对应点，当前只有 {src.shape[0]} 组")
    if src.shape != dst.shape:
        raise ValueError(f"图像点数 {src.shape[0]} 与地面点数 {dst.shape[0]} 不一致")

    method = cv2.RANSAC if (use_ransac and len(src) > 4) else 0
    H, mask = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=ransac_thresh_m)
    if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise RuntimeError("cv2.findHomography 求解失败，请检查点是否共线或重复")

    # 归一化使 H[2, 2] == 1.0（若接近 0 则按 Frobenius 范数归一化）
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    det = float(np.linalg.det(H))
    if abs(det) < 1e-12:
        raise RuntimeError(f"单应矩阵奇异不可逆 (det={det:.3e})，请检查标定点分布")

    cond = float(np.linalg.cond(H))
    H_inv = np.linalg.inv(H)
    if not np.all(np.isfinite(H_inv)):
        raise RuntimeError("逆单应矩阵含 NaN/Inf")

    errors_mm: List[float] = []
    reproj_px_errors: List[float] = []
    for (u, v), (d_gt, y_gt) in zip(src, dst):
        d_pred, y_pred = pixel_to_ground(H, u, v)
        err_m = math.hypot(d_pred - d_gt, y_pred - y_gt)
        errors_mm.append(err_m * 1000.0)

        u_pred, v_pred = ground_to_pixel(H_inv, d_gt, y_gt)
        reproj_px_errors.append(math.hypot(u_pred - u, v_pred - v))

    inliers = (
        [int(x) for x in mask.ravel().tolist()]
        if mask is not None
        else [1] * len(src)
    )

    return {
        "H": H,
        "H_inv": H_inv,
        "H_flat": [float(x) for x in H.ravel().tolist()],
        "det": det,
        "cond": cond,
        "inliers": inliers,
        "errors_mm": errors_mm,
        "mean_err_mm": float(np.mean(errors_mm)),
        "max_err_mm": float(np.max(errors_mm)),
        "reproj_px_errors": reproj_px_errors,
        "mean_px_err": float(np.mean(reproj_px_errors)),
        "max_px_err": float(np.max(reproj_px_errors)),
    }


def _fmt_yaml_float(val: float, precision: int = 8) -> str:
    """将浮点数格式化为合法 YAML 浮点字面量（保证 ROS2 double_array 不误判为 int）。"""
    s = f"{float(val):.{precision}g}"
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s


def format_yaml_snippet(H: np.ndarray, precision: int = 8) -> str:
    """生成可直接粘贴到 line_follower_node ros__parameters 的 YAML 片段。"""
    vals = [round(float(x), precision) for x in np.asarray(H, dtype=np.float64).ravel().tolist()]
    row0 = ", ".join(_fmt_yaml_float(v, precision) for v in vals[0:3])
    row1 = ", ".join(_fmt_yaml_float(v, precision) for v in vals[3:6])
    row2 = ", ".join(_fmt_yaml_float(v, precision) for v in vals[6:9])
    lines = [
        "# IPM 地面单应标定结果（原点 = 车头泡沫防撞条中心，+d 向前 m，+y 向左 m）",
        "use_homography: true",
        "homography_matrix: [",
        f"  {row0},",
        f"  {row1},",
        f"  {row2}",
        "]",
    ]
    return "\n".join(lines)


def load_points_file(path: str | Path) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """读取 JSON 或 CSV 格式的标定对应点。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"标定点文件不存在: {p}")

    ext = p.suffix.lower()
    img_pts: List[Tuple[float, float]] = []
    ground_pts: List[Tuple[float, float]] = []

    if ext == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        rows = raw["points"] if isinstance(raw, dict) and "points" in raw else raw
        if not isinstance(rows, list):
            raise ValueError("JSON 必须为数组或包含 'points' 数组的字典")
        for item in rows:
            u = float(item["u"])
            v = float(item["v"])
            d = float(item.get("d_bumper_m", item.get("d", item.get("x"))))
            y = float(item.get("y_lat_m", item.get("y")))
            img_pts.append((u, v))
            ground_pts.append((d, y))
    elif ext == ".csv":
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or not row.get("u"):
                    continue
                u = float(row["u"])
                v = float(row["v"])
                d = float(row.get("d_bumper_m") or row.get("d") or row.get("x"))
                y = float(row.get("y_lat_m") or row.get("y"))
                img_pts.append((u, v))
                ground_pts.append((d, y))
    else:
        raise ValueError(f"不支持的点集扩展名 {ext}（仅支持 .json / .csv）")

    return img_pts, ground_pts


def capture_snapshot(device: str, width: int, height: int, warmup: int = 15) -> np.ndarray:
    """从摄像头抓拍一帧用于标定。"""
    dev: Any = int(device) if device.isdigit() else device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2 if os.name != "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {device}")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = None
        for _ in range(max(1, warmup)):
            ok, frm = cap.read()
            if ok and frm is not None:
                frame = frm
        if frame is None:
            raise RuntimeError(f"摄像头 {device} 抓拍失败")
        return frame
    finally:
        cap.release()


def interactive_pick(
    image: np.ndarray,
    targets: Sequence[Tuple[float, float]] = DEFAULT_TARGET_POINTS_M,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """OpenCV 窗口交互点击 >=4 个标定靶点。"""
    img_pts: List[Tuple[float, float]] = []
    ground_pts: List[Tuple[float, float]] = []
    win = "AIC IPM Homography Calibration (Left-click=pick, u=undo, Enter=solve, q=quit)"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = len(img_pts)
            if idx < len(targets):
                d_m, y_m = targets[idx]
                print(
                    f"  [点 #{idx + 1}] 像素=({x}, {y}) -> 预设地面=({d_m:.3f}m, {y_m:+.3f}m)"
                )
            else:
                raw = input(
                    f"  [点 #{idx + 1}] 像素=({x}, {y}) 请输入地面坐标 'd_bumper_m y_lat_m': "
                ).strip()
                parts = raw.replace(",", " ").split()
                d_m, y_m = float(parts[0]), float(parts[1])
            img_pts.append((float(x), float(y)))
            ground_pts.append((float(d_m), float(y_m)))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print("操作说明：")
    print("  - 按顺序左键点击以下地面点（原点 = 车头泡沫防撞条中心，+d 向前，+y 向左）：")
    for i, (d_m, y_m) in enumerate(targets):
        print(f"      #{i + 1}: d={d_m:.3f} m, y={y_m:+.3f} m")
    print("  - 至少点 4 个点；按 'u' 撤销上一个点，按 Enter / Space 求解，按 'q' / Esc 退出。")

    while True:
        vis = image.copy()
        for i, ((u, v), (d_m, y_m)) in enumerate(zip(img_pts, ground_pts)):
            cv2.circle(vis, (int(u), int(v)), 5, (0, 255, 0), -1)
            cv2.putText(
                vis,
                f"#{i + 1}({d_m:.2f},{y_m:+.2f})",
                (int(u) + 6, int(v) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        nxt = len(img_pts)
        hint = (
            f"Next #{nxt + 1}: d={targets[nxt][0]:.2f}m, y={targets[nxt][1]:+.2f}m"
            if nxt < len(targets)
            else f"Picked {nxt} pts (Enter to solve, click to add custom)"
        )
        cv2.putText(vis, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            sys.exit(1)
        elif key == ord("u") and img_pts:
            img_pts.pop()
            ground_pts.pop()
            print("  已撤销上一个点")
        elif key in (13, 10, 32):
            if len(img_pts) >= 4:
                break
            print(f"  当前只有 {len(img_pts)} 个点，至少需要 4 个点！")

    cv2.destroyAllWindows()
    return img_pts, ground_pts


def print_report(
    img_pts: Sequence[Tuple[float, float]],
    ground_pts: Sequence[Tuple[float, float]],
    res: Dict[str, Any],
) -> None:
    print("=" * 72)
    print("IPM 地面单应标定报告（原点 = 车头防撞条中心，+d 向前 m，+y 向左 m）")
    print("=" * 72)
    print(f"{'#':>2} {'u(px)':>8} {'v(px)':>8} {'d_gt(m)':>9} {'y_gt(m)':>9} "
          f"{'err(mm)':>9} {'reproj(px)':>11}")
    print("-" * 72)
    for i, ((u, v), (d, y), emm, epx) in enumerate(
        zip(img_pts, ground_pts, res["errors_mm"], res["reproj_px_errors"]), start=1
    ):
        print(f"{i:2d} {u:8.1f} {v:8.1f} {d:9.4f} {y:+9.4f} {emm:9.2f} {epx:11.2f}")
    print("-" * 72)
    print(f"平均地面误差 : {res['mean_err_mm']:.2f} mm  | 最大地面误差 : {res['max_err_mm']:.2f} mm")
    print(f"平均像素误差 : {res['mean_px_err']:.2f} px  | 最大像素误差 : {res['max_px_err']:.2f} px")
    print(f"矩阵可逆性   : det={res['det']:.4e}  cond={res['cond']:.2e}  (OK)")
    print("=" * 72)
    print("\n可直接复制到 YAML 的配置块：\n")
    print(format_yaml_snippet(res["H"]))
    print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="AIC 2026 决赛 IPM 地面单应标定工具 (4+ 点)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--image", default=None, help="标定静态截图路径（交互点选）")
    ap.add_argument("--device", default=None, help="摄像头设备（如 /dev/video0 或 0）")
    ap.add_argument("--save-snap", default=None, help="抓拍图像保存路径")
    ap.add_argument("--width", type=int, default=640, help="抓拍宽度")
    ap.add_argument("--height", type=int, default=480, help="抓拍高度")
    ap.add_argument("--points", default=None, help="无头模式输入对应点文件 (.json 或 .csv)")
    ap.add_argument("--save-points", default=None, help="保存交互点选的对应点到 JSON")
    ap.add_argument("--ransac", action="store_true", help="点数 >4 时使用 RANSAC 抗离群点")
    ap.add_argument("--out", default=None, help="可选：把 YAML 片段写入文件")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.points:
        img_pts, ground_pts = load_points_file(args.points)
    else:
        if args.image:
            img = (
                cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
                if os.path.isfile(args.image)
                else None
            )
            if img is None:
                print(f"[ERROR] 无法读取图像: {args.image}", file=sys.stderr)
                return 2
        elif args.device is not None:
            img = capture_snapshot(args.device, args.width, args.height)
            if args.save_snap:
                Path(args.save_snap).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(args.save_snap, img)
                print(f"[INFO] 已保存抓拍图: {args.save_snap}")
        else:
            print("[ERROR] 请指定 --points <file> 或 --image <img.png> 或 --device /dev/video0",
                  file=sys.stderr)
            return 2
        img_pts, ground_pts = interactive_pick(img)
        if args.save_points:
            Path(args.save_points).parent.mkdir(parents=True, exist_ok=True)
            payload = [
                {"u": u, "v": v, "d_bumper_m": d, "y_lat_m": y}
                for (u, v), (d, y) in zip(img_pts, ground_pts)
            ]
            Path(args.save_points).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[INFO] 已保存对应点: {args.save_points}")

    res = solve_homography(img_pts, ground_pts, use_ransac=args.ransac)
    print_report(img_pts, ground_pts, res)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(format_yaml_snippet(res["H"]) + "\n", encoding="utf-8")
        print(f"[OUT] 已写入 YAML 片段: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
