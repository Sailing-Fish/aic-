#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AIC 2026 决赛 8 类数字（1..8）YOLOv8 一键训练与 ONNX 导出脚本。

安全硬约束（不可覆盖）：
    - fliplr=0.0, flipud=0.0（严禁水平/垂直翻转，避免 2/5/6 等数字语义破坏）
    - degrees=10.0（严格 <= 15°）
    - close_mosaic=15, hsv_h=0.015, hsv_s=0.6, hsv_v=0.5

训练完成后自动：
    1. 加载 best.pt 在验证集上评估并打印 mAP50 / mAP50-95；
    2. 导出 best.onnx（opset=11/12, simplify=True, imgsz=640），适配 RDK X5 Horizon OpenExplorer 转 .bin；
    3. 可选生成验证集预测拼图 (--val-pred-out) 与实拍测试拼图 (--real-images + --real-pred-out)。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

# 硬编码安全超参数（严禁开启任何翻转增强）
FORCED_TRAIN_AUG_KWARGS = {
    "fliplr": 0.0,
    "flipud": 0.0,
    "degrees": 10.0,
    "close_mosaic": 15,
    "hsv_h": 0.015,
    "hsv_s": 0.6,
    "hsv_v": 0.5,
}

LOCAL_PRETRAINED_FALLBACKS = (
    Path("yolov8n.pt"),
    Path(r"d:\ultralytics-main\ultralytics-main\yolov8n.pt"),
)


def resolve_model_path(model_arg: str) -> str:
    """优先使用用户指定路径或本地预置 yolov8n.pt 权重。"""
    p = Path(model_arg)
    if p.is_file():
        return str(p)
    if model_arg == "yolov8n.pt":
        for cand in LOCAL_PRETRAINED_FALLBACKS:
            if cand.is_file():
                return str(cand)
    return model_arg


def build_prediction_collage(
    model,
    image_paths: Sequence[Path],
    out_path: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "0",
    max_images: int = 9,
    cols: int = 3,
    cell_size: tuple[int, int] = (640, 480),
) -> Path:
    """对给定图片列表运行 YOLO predict 并拼成网格对比图保存。"""
    selected = list(image_paths)[:max_images]
    if not selected:
        raise ValueError("image_paths is empty; cannot build prediction collage")

    cells: List[np.ndarray] = []
    cw, ch = cell_size
    for img_p in selected:
        results = model.predict(
            source=str(img_p),
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        plotted = results[0].plot() if results else cv2.imdecode(np.fromfile(str(img_p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if plotted is None:
            plotted = np.zeros((ch, cw, 3), dtype=np.uint8)
        else:
            # 保持比例缩放并填充黑边到统一格子大小，方便拼接横竖不同比例的实拍图
            ih, iw = plotted.shape[:2]
            scale = min(cw / max(1, iw), ch / max(1, ih))
            nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
            resized = cv2.resize(plotted, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas_cell = np.full((ch, cw, 3), 24, dtype=np.uint8)
            ox = (cw - nw) // 2
            oy = (ch - nh) // 2
            canvas_cell[oy:oy + nh, ox:ox + nw] = resized
            plotted = canvas_cell
        cv2.putText(plotted, img_p.name[:32], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cells.append(plotted)

    n = len(cells)
    cols_eff = min(cols, n)
    rows_eff = int(math.ceil(n / cols_eff))
    while len(cells) < rows_eff * cols_eff:
        cells.append(np.full((ch, cw, 3), 24, dtype=np.uint8))

    grid = np.vstack([np.hstack(cells[r * cols_eff:(r + 1) * cols_eff]) for r in range(rows_eff)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", grid)
    if ok:
        buf.tofile(str(out_path))
    return out_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="AIC 2026 决赛 8 类数字 YOLOv8 训练与 ONNX 导出脚本（严禁翻转）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", required=True, help="dataset.yaml 路径")
    ap.add_argument("--model", default="yolov8n.pt", help="起始预训练权重（默认自动查找 yolov8n.pt）")
    ap.add_argument("--epochs", type=int, default=70, help="训练轮数")
    ap.add_argument("--batch", type=int, default=16, help="Batch size（RTX 5060 8GB 推荐 16 或 32）")
    ap.add_argument("--imgsz", type=int, default=640, help="训练与导出输入分辨率（须与 BPU 640x640 一致）")
    ap.add_argument("--device", default="0", help="训练设备（例如 '0' 或 'cpu'）")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader 工作进程数")
    ap.add_argument("--project", default="runs/digits8", help="Ultralytics 输出根目录")
    ap.add_argument("--name", default="yolov8n_digits8", help="本次实验名称")
    ap.add_argument("--opset", type=int, choices=[11, 12], default=11, help="ONNX opset 版本（RDK X5 推荐 11 或 12）")
    ap.add_argument("--val-pred-out", default=None, help="可选：保存验证集预测拼图路径")
    ap.add_argument("--real-images", nargs="*", default=None, help="可选：实拍验证图路径列表")
    ap.add_argument("--real-pred-out", default=None, help="可选：实拍图预测拼图输出路径")
    ap.add_argument("--conf", type=float, default=0.25, help="预测可视化置信度阈值")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import yaml
    from ultralytics import YOLO

    data_yaml = Path(args.data).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset.yaml 不存在: {data_yaml}")

    model_path = resolve_model_path(args.model)
    print(f"[TRAIN] data={data_yaml}")
    print(f"[TRAIN] base_model={model_path} epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} device={args.device}")
    print(f"[TRAIN] forced_aug={FORCED_TRAIN_AUG_KWARGS}")

    model = YOLO(model_path)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        **FORCED_TRAIN_AUG_KWARGS,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name)).resolve()
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        best_pt = save_dir / "weights" / "last.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"训练结束但未找到权重文件: {save_dir / 'weights'}")

    best_model = YOLO(str(best_pt))
    val_metrics = best_model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device, verbose=False)
    box_metrics = getattr(val_metrics, "box", None)
    map50 = float(getattr(box_metrics, "map50", 0.0)) if box_metrics is not None else 0.0
    map50_95 = float(getattr(box_metrics, "map", 0.0)) if box_metrics is not None else 0.0
    mp = float(getattr(box_metrics, "mp", 0.0)) if box_metrics is not None else 0.0
    mr = float(getattr(box_metrics, "mr", 0.0)) if box_metrics is not None else 0.0

    onnx_exported = best_model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
    )
    best_onnx = Path(str(onnx_exported)).resolve()

    if args.val_pred_out:
        ds_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        ds_root = Path(ds_cfg.get("path", data_yaml.parent))
        val_dir = ds_root / str(ds_cfg.get("val", "images/val"))
        val_imgs = sorted([p for p in val_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if val_imgs:
            step = max(1, len(val_imgs) // 9)
            sample_vals = val_imgs[::step][:9]
            vp_out = build_prediction_collage(
                best_model, sample_vals, Path(args.val_pred_out),
                imgsz=args.imgsz, conf=args.conf, device=args.device,
            )
            print(f"[VAL_PRED] saved: {vp_out}")

    if args.real_images and args.real_pred_out:
        real_paths = [Path(p) for p in args.real_images if Path(p).is_file()]
        if real_paths:
            rp_out = build_prediction_collage(
                best_model, real_paths, Path(args.real_pred_out),
                imgsz=args.imgsz, conf=args.conf, device=args.device, max_images=len(real_paths), cols=2,
            )
            print(f"[REAL_PRED] saved: {rp_out}")

    calib_hint = data_yaml.parent / "calib_list.txt"
    print("\n================ 训练与导出汇总 ================")
    print(f"  Precision   : {mp:.4f}")
    print(f"  Recall      : {mr:.4f}")
    print(f"  mAP50       : {map50:.4f}")
    print(f"  mAP50-95    : {map50_95:.4f}")
    print(f"  best.pt     : {best_pt}")
    print(f"  best.onnx   : {best_onnx} (opset={args.opset}, simplify=True, imgsz={args.imgsz})")
    print("---------------- RDK X5 转 .bin 下一步 ----------------")
    print(f"  1. 将 {best_onnx.name} 与校准集列表 {calib_hint} 拷贝到 Horizon OpenExplorer Docker (bayes-e / x5)。")
    print("  2. 在 OE config yaml 中配置 input_type_rt: nv12, input_type_train: rgb, norm_type: data_scale, scale_value: 0.00392156862745098。")
    print("  3. 若需匹配 RDK X5 bpu_yolov8_node.py 的 6 路解耦输出头，按模型修改脚本/OE 去掉末级 DFL+Concat 解码后执行 hb_mapper makertbin。")
    print("  4. 将生成的 .bin 部署到 src/yolov8/mod1/best_modified_yolov8.bin（对应 out.list 1..8）。")
    print("======================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
