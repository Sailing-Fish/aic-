#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIC 2026 决赛 8 类数字 YOLOv8 -> RDK X5 BPU (bayes-e) INT8 一键转换、耗时测算与精度对齐门控脚本：
1. 将 Ultralytics best.pt 裁剪末级 DFL+Concat 解码，导出 6 路解耦输出头 NHWC ONNX
   [(1,80,80,8), (1,80,80,64), (1,40,40,8), (1,40,40,64), (1,20,20,8), (1,20,20,64)]；
2. 在本机 Horizon OpenExplorer Docker (`openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8`) 中执行：
   - hb_mapper checker
   - hb_mapper makertbin (nv12 -> rgb, scale=1/255, cal_data=实车对齐640x640图片)
   - hb_model_modifier 自动定位并剥离末尾 6 个 HzDequantize 节点（输出 INT32 + per-channel scale_data，完美契合 bpu_yolov8_node.py）
   - hb_perf 实测单帧 BPU 耗时 (ms / FPS / DDR bytes)
3. 用 HB_ONNXRuntime 在 Docker 内对真实留出集 / 实车 debug 盲测帧 / 压测集对比
   original_float_model.onnx (FP32, [0..255]) 与 quantized_model.onnx (INT8 NHWC, int16-128)：
   - 核对 6 路输出头 Cosine 相似度
   - 直接调用 bpu_yolov8_node.decode_yolov8_outputs + run_nms_filter（含 roi_crop_top_px 反映射）生成 INT8 真实识别对比拼图。
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "yolov8"))
import bpu_yolov8_node as yolo_mod  # noqa: E402

DOCKER_IMAGE = "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8"


def export_6head_nhwc_onnx(pt_path: Path, onnx_out: Path, imgsz: int = 640, opset: int = 11) -> Path:
    import onnx
    import torch
    from ultralytics import YOLO

    yolo = YOLO(str(pt_path))
    model = yolo.model.eval().cpu()
    # 融合 Conv+BN 以获得最优 X5 BPU 算子折叠
    if hasattr(model, "fuse"):
        model.fuse()

    det = model.model[-1]
    orig_forward = det.forward

    def bpu_6head_forward(self, x):
        outs = []
        for i in range(self.nl):
            cls_i = self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous()
            reg_i = self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous()
            outs.append(cls_i)
            outs.append(reg_i)
        return tuple(outs)

    det.forward = types.MethodType(bpu_6head_forward, det)
    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)
    raw_onnx = onnx_out.with_suffix(".raw.onnx")
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    output_names = ["cls_80", "reg_80", "cls_40", "reg_40", "cls_20", "reg_20"]
    with torch.no_grad():
        y = model(dummy)
        shapes = [tuple(t.shape) for t in y]
        print(f"[EXPORT] Verified PyTorch 6-head NHWC output shapes: {shapes}")
        torch.onnx.export(
            model,
            dummy,
            str(raw_onnx),
            opset_version=opset,
            input_names=["images"],
            output_names=output_names,
            do_constant_folding=True,
            dynamo=False,
        )
    det.forward = orig_forward

    # 简化图结构并清除无关元数据，便于 OpenExplorer 编译
    try:
        from onnxsim import simplify

        model_proto = onnx.load(str(raw_onnx))
        simp_proto, ok = simplify(model_proto)
        if ok:
            onnx.save(simp_proto, str(onnx_out))
            raw_onnx.unlink(missing_ok=True)
        else:
            shutil.move(str(raw_onnx), str(onnx_out))
    except Exception as exc:
        print(f"[EXPORT] onnxsim fallback to unsimplified ONNX: {exc}")
        shutil.move(str(raw_onnx), str(onnx_out))

    m_chk = onnx.load(str(onnx_out))
    onnx.checker.check_model(m_chk)
    out_shapes = [
        tuple(d.dim_value for d in o.type.tensor_type.shape.dim)
        for o in m_chk.graph.output
    ]
    print(f"[EXPORT] Final 6-head NHWC ONNX: {onnx_out} ({onnx_out.stat().st_size / 1e6:.2f} MB) outputs={out_shapes}")
    expected = [(1, 80, 80, 8), (1, 80, 80, 64), (1, 40, 40, 8), (1, 40, 40, 64), (1, 20, 20, 8), (1, 20, 20, 64)]
    if out_shapes != expected:
        raise RuntimeError(f"ONNX output shapes mismatch! expected={expected}, got={out_shapes}")
    return onnx_out


def write_oe_config_and_docker_script(work_dir: Path, onnx_filename: str, imgsz: int = 640) -> Tuple[Path, Path]:
    yaml_path = work_dir / "yolov8_digits8_v4_x5_nhwc.yaml"
    yaml_path.write_text(
        f"model_parameters:\n"
        f"  onnx_model: './{onnx_filename}'\n"
        f"  march: 'bayes-e'\n"
        f"  layer_out_dump: False\n"
        f"  working_dir: './model_output_nhwc'\n"
        f"  output_model_file_prefix: 'best_modified_yolov8'\n"
        f"\n"
        f"input_parameters:\n"
        f"  input_name: 'images'\n"
        f"  input_type_rt: 'nv12'\n"
        f"  input_type_train: 'rgb'\n"
        f"  input_layout_train: 'NCHW'\n"
        f"  input_shape: '1x3x{imgsz}x{imgsz}'\n"
        f"  norm_type: 'data_scale'\n"
        f"  scale_value: 0.00392156862745098\n"
        f"\n"
        f"calibration_parameters:\n"
        f"  cal_data_dir: './cal_data'\n"
        f"  cal_data_type: 'float32'\n"
        f"  preprocess_on: True\n"
        f"  calibration_type: 'default'\n"
        f"\n"
        f"compiler_parameters:\n"
        f"  compile_mode: 'latency'\n"
        f"  debug: False\n"
        f"  optimize_level: 'O3'\n",
        encoding="utf-8",
    )

    docker_py = work_dir / "run_oe_compile_and_verify.py"
    docker_py.write_text(
        '''#!/usr/bin/env python3
import glob, json, os, re, subprocess, sys
import cv2
import numpy as np
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.hbdtort import runtime_pb2

work = "/data"
out_dir = "/data/model_output_nhwc"

def run(cmd, cwd=work):
    print("CMD:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if r.stdout:
        print(r.stdout[-3000:])
    if r.stderr:
        print(r.stderr[-1500:], file=sys.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}")
    return r.stdout + "\\n" + r.stderr

print("=== [1/4] hb_mapper checker ===")
run(["hb_mapper", "checker", "--model-type", "onnx", "--march", "bayes-e", "--model", "./''' + onnx_filename + '''"])

print("=== [2/4] hb_mapper makertbin ===")
run(["hb_mapper", "makertbin", "--config", "./yolov8_digits8_v4_x5_nhwc.yaml", "--model-type", "onnx"])

print("=== [3/4] hb_model_modifier (strip trailing HzDequantize nodes) ===")
probe = subprocess.run(["hb_model_modifier", "best_modified_yolov8.bin"], capture_output=True, text=True, cwd=out_dir)
txt = probe.stdout + "\\n" + probe.stderr
m = re.search(r"Nodes that can be deleted:\\s*(\\[.*?\\])", txt, flags=re.S)
if not m:
    raise RuntimeError(f"Failed to discover deletable Dequantize nodes: {txt}")
nodes = eval(m.group(1))
deq_nodes = [n for n in nodes if "Dequantize" in n]
print("Deletable nodes:", nodes)
print("Removing Dequantize:", deq_nodes)
if len(deq_nodes) != 6:
    print(f"WARNING: expected 6 Dequantize nodes, got {len(deq_nodes)}: {deq_nodes}")
cmd_mod = ["hb_model_modifier", "best_modified_yolov8.bin"]
for n in deq_nodes:
    cmd_mod += ["-r", n]
run(cmd_mod, cwd=out_dir)

mod_bin = os.path.join(out_dir, "best_modified_yolov8_modified.bin")
with open(mod_bin, "rb") as f:
    mp = runtime_pb2.ModelProto()
    mp.ParseFromString(f.read())
print("Modified .bin outputs:")
for out in mp.graphs[0].output:
    print(" ", " | ".join(l.strip() for l in str(out).splitlines() if l.strip()))

print("=== [4/4] hb_perf latency & DDR report ===")
perf_txt = run(["hb_perf", "best_modified_yolov8_modified.bin"], cwd=out_dir)
perf_info = {}
for line in perf_txt.splitlines():
    if "FPS=" in line or "latency" in line:
        print("PERF:", line)
        m_fps = re.search(r"FPS=([0-9.]+)", line)
        m_lat = re.search(r"latency\\s*=\\s*([0-9.]+)\\s*us", line)
        m_ddr = re.search(r"DDR\\s*=\\s*([0-9]+)\\s*bytes", line)
        if m_fps:
            perf_info["fps"] = float(m_fps.group(1))
        if m_lat:
            perf_info["latency_us"] = float(m_lat.group(1))
            perf_info["latency_ms"] = float(m_lat.group(1)) / 1000.0
        if m_ddr:
            perf_info["ddr_bytes"] = int(m_ddr.group(1))

# Parity check FP32 original_float_model.onnx vs INT8 quantized_model.onnx on verify_inputs/*.png
float_sess = HB_ONNXRuntime(model_file=os.path.join(out_dir, "best_modified_yolov8_original_float_model.onnx"))
quant_sess = HB_ONNXRuntime(model_file=os.path.join(out_dir, "best_modified_yolov8_quantized_model.onnx"))
f_in = float_sess.get_inputs()[0].name
q_in = quant_sess.get_inputs()[0].name
f_outs = [o.name for o in float_sess.get_outputs()]
q_outs = [o.name for o in quant_sess.get_outputs()]

verify_imgs = sorted(glob.glob("/data/verify_inputs/*.png"))
cos_summary = {i: [] for i in range(len(q_outs))}
saved_q_outputs = {}

def cosine(a, b):
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return float(np.dot(af, bf) / denom) if denom > 1e-12 else 0.0

for idx, vp in enumerate(verify_imgs):
    bgr = cv2.imread(vp)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # CRITICAL: original_float_model.onnx already has data_scale=1/255 baked inside -> feed [0, 255] float32!
    inp_f32 = np.expand_dims(np.transpose(rgb.astype(np.float32), (2, 0, 1)), 0)
    # CRITICAL: quantized_model.onnx expects int8 NHWC (rgb - 128)
    inp_i8 = np.expand_dims((rgb.astype(np.int16) - 128).astype(np.int8), 0)

    fo = float_sess.run(f_outs, {f_in: inp_f32})
    qo = quant_sess.run(q_outs, {q_in: inp_i8})

    for hi, (fa, qa) in enumerate(zip(fo, qo)):
        # Match layout (float_model may output NHWC or NCHW)
        if fa.shape != qa.shape and fa.ndim == 4 and fa.shape[1] == qa.shape[-1]:
            fa = np.transpose(fa, (0, 2, 3, 1))
        cos_summary[hi].append(cosine(fa, qa))

    # Dump INT8 outputs to .npz for exact host decoding via bpu_yolov8_node.py
    np.savez_compressed(os.path.join("/data/verify_outputs", os.path.basename(vp)[:-4] + ".npz"), *qo)

head_cos = {q_outs[i]: float(np.mean(cos_summary[i])) for i in range(len(q_outs)) if cos_summary[i]}
print("PARITY_COSINE:", json.dumps(head_cos, indent=2))
with open(os.path.join(work, "compile_summary.json"), "w") as jf:
    json.dump({"perf": perf_info, "head_cosine": head_cos}, jf, indent=2)
print("DOCKER_ALL_DONE")
''',
        encoding="utf-8",
    )
    return yaml_path, docker_py


def stage_verify_inputs(
    work_dir: Path,
    roi_crop_top_px: int,
    extra_images: Sequence[Path],
) -> List[Tuple[str, Path, np.ndarray]]:
    vin = work_dir / "verify_inputs"
    vout = work_dir / "verify_outputs"
    shutil.rmtree(vin, ignore_errors=True)
    shutil.rmtree(vout, ignore_errors=True)
    vin.mkdir(parents=True, exist_ok=True)
    vout.mkdir(parents=True, exist_ok=True)

    items: List[Tuple[str, Path, np.ndarray]] = []
    for p in extra_images:
        if not p.is_file():
            continue
        orig = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if orig is None:
            continue
        # 若本身是未裁剪的 640x480 实拍原图则按 roi_crop_top_px 裁；若已是 640x640 (如 stress 测试图) 则直接送入
        if orig.shape[:2] == (480, 640):
            inp, _, _, _, _, _ = yolo_mod.prepare_model_input(orig, 640, 640, roi_crop_top_px=roi_crop_top_px)
        else:
            inp = cv2.resize(orig, (640, 640)) if orig.shape[:2] != (640, 640) else orig
        stem = f"{len(items):02d}_{p.stem[:28]}"
        out_png = vin / f"{stem}.png"
        ok, buf = cv2.imencode(".png", inp)
        if ok:
            buf.tofile(str(out_png))
            items.append((stem, p, orig))
    print(f"[VERIFY_STAGE] staged {len(items)} images into {vin}")
    return items


def decode_and_render_int8_collage(
    work_dir: Path,
    staged: List[Tuple[str, Path, np.ndarray]],
    out_collage: Path,
    roi_crop_top_px: int = 160,
    conf_thresh: float = 0.25,
) -> None:
    classes = list(yolo_mod.load_class_list(str(REPO_ROOT / "src" / "yolov8" / "mod1" / "out.list")))
    grids = []
    for s in (8, 16, 32):
        gw, gh = 640 // s, 640 // s
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        grid = np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32) + 0.5
        grids.append((grid, s))
    dfl_w = np.arange(16, dtype=np.float32)

    cells: List[np.ndarray] = []
    print("\n=== INT8 Q-ONNX BPU-Node Decode Results ===")
    for stem, src_p, orig in staged:
        npz_p = work_dir / "verify_outputs" / f"{stem}.npz"
        if not npz_p.is_file():
            continue
        data = np.load(npz_p)
        fake_outs = [
            types.SimpleNamespace(
                buffer=data[k],
                properties=types.SimpleNamespace(shape=data[k].shape, scale_data=[1.0]),
            )
            for k in data.files
        ]
        oh, ow = orig.shape[:2]
        eff_roi = roi_crop_top_px if (oh == 480 and ow == 640) else 0
        _, sx, sy, pw, ph, oy = yolo_mod.prepare_model_input(orig, 640, 640, roi_crop_top_px=eff_roi)
        boxes, scores, cls_ids, ok, _, _ = yolo_mod.decode_yolov8_outputs(
            outputs=fake_outs,
            grids=grids,
            dfl_weights=dfl_w,
            num_classes=len(classes),
            conf_thresh=conf_thresh,
            pad_w=pw,
            pad_h=ph,
            scale_r=sx,
            orig_w=ow,
            orig_h=oh,
            scale_x=sx,
            scale_y=sy,
            offset_y=oy,
        )
        dets = yolo_mod.run_nms_filter(
            boxes, scores, cls_ids, classes, conf_thresh, 0.55, 20, 6, ow, oh, class_agnostic_nms=False
        )
        summary = ", ".join(f"{d['type']}:{d['confidence']:.2f}" for d in dets) if dets else "NONE"
        print(f"  {src_p.name:40s} -> {summary}")

        vis = orig.copy()
        if eff_roi > 0:
            cv2.line(vis, (0, eff_roi), (ow, eff_roi), (0, 220, 220), 1)
        for d in dets:
            x1, y1 = int(round(d["bx"])), int(round(d["by"]))
            x2, y2 = int(round(d["bx"] + d["bw"])), int(round(d["by"] + d["bh"]))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 240, 60), 2)
            cv2.putText(
                vis, f"{d['type']} {d['confidence']:.2f}", (max(2, x1), max(16, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 240, 60), 2, cv2.LINE_AA
            )
        vis_rs = cv2.resize(vis, (480, 360 if oh == 480 else 480))
        if vis_rs.shape[0] != 360:
            vis_rs = cv2.resize(vis, (480, 360))
        cv2.putText(vis_rs, src_p.name[:34], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        cells.append(vis_rs)

    if cells:
        cols = 3
        rows = int(math.ceil(len(cells) / cols))
        while len(cells) < rows * cols:
            cells.append(np.zeros((360, 480, 3), dtype=np.uint8))
        grid = np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)])
        out_collage.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_collage), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"[INT8_COLLAGE] saved: {out_collage}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="YOLOv8 best.pt -> 6-head NHWC ONNX -> RDK X5 BPU .bin & INT8 Parity")
    ap.add_argument("--pt", required=True, help="训练好的 best.pt 路径")
    ap.add_argument("--work-dir", default="dataset_work/x5_convert_v4", help="转换工作目录（含 cal_data/）")
    ap.add_argument("--roi-crop-top", type=int, default=160, help="与训练对齐的顶部裁剪像素")
    ap.add_argument("--verify-images", nargs="*", default=[], help="额外实拍/盲测图片列表（自动裁 ROI 并画 INT8 对比拼图）")
    ap.add_argument("--deploy-bin", default=None, help="可选：将生成的 modified.bin 复制到目标路径（如 src/yolov8/mod1/best_modified_yolov8.bin）")
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--imgsz", type=int, default=640)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pt_path = Path(args.pt).resolve()
    work_dir = Path(args.work_dir).resolve()
    cal_dir = work_dir / "cal_data"
    if not pt_path.is_file():
        raise FileNotFoundError(f"best.pt 不存在: {pt_path}")
    if not cal_dir.is_dir() or not list(cal_dir.glob("*.png")):
        raise FileNotFoundError(f"校准集目录为空: {cal_dir}")

    onnx_name = f"{pt_path.parent.parent.name}_bpu6head_nhwc.onnx"
    onnx_out = work_dir / onnx_name
    export_6head_nhwc_onnx(pt_path, onnx_out, imgsz=args.imgsz, opset=args.opset)
    write_oe_config_and_docker_script(work_dir, onnx_name, imgsz=args.imgsz)

    staged = stage_verify_inputs(
        work_dir,
        roi_crop_top_px=args.roi_crop_top,
        extra_images=[Path(p).resolve() for p in args.verify_images],
    )

    mount_src = str(work_dir).replace("\\", "/")
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{mount_src}:/data",
        DOCKER_IMAGE,
        "python3", "/data/run_oe_compile_and_verify.py",
    ]
    print("[DOCKER] running:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"Docker X5 BPU compilation failed with code {r.returncode}")

    decode_and_render_int8_collage(
        work_dir,
        staged,
        work_dir / "int8_real_verify.jpg",
        roi_crop_top_px=args.roi_crop_top,
    )

    mod_bin = work_dir / "model_output_nhwc" / "best_modified_yolov8_modified.bin"
    if args.deploy_bin and mod_bin.is_file():
        dst = Path(args.deploy_bin).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mod_bin, dst)
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()[:12]
        print(f"[DEPLOY] copied {mod_bin.name} -> {dst} ({dst.stat().st_size / 1e6:.2f} MB, sha256={sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
