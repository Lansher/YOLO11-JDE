#!/usr/bin/env python3
"""
一键复现：与「表：Jetson 等部署对比」常见四行一致——
  PyTorch FP32、ONNX Runtime FP32、TensorRT FP32、TensorRT FP16（predict 单图测速）。

在仓库根目录执行：
  python scripts/benchmark_deploy.py --modes all
  python scripts/benchmark_deploy.py --modes pt_fp32,onnx_fp32,trt_fp32,trt_fp16

另支持：pt_fp16、trt_int8（需 data yaml）。

TensorRT 导出必须在 NVIDIA GPU + TensorRT 环境下进行；ONNX FP32 可在 CPU 上导出。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from bootstrap import ensure_repo_root

ensure_repo_root()

import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS


_REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JDE 部署测速：PT/ONNX/TensorRT 等 predict 测速")
    p.add_argument(
        "--model",
        type=str,
        default="YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt",
        help="相对仓库根目录的 .pt 权重路径",
    )
    p.add_argument(
        "--data",
        type=str,
        default=str(_REPO_ROOT / "ultralytics/cfg/datasets/coco8.yaml"),
        help="TensorRT INT8 校准用数据集 yaml（需含 val 图像，建议 ≥300 张更佳）",
    )
    p.add_argument("--imgsz", type=int, default=1280, help="推理边长（与训练一致）")
    p.add_argument("--warmup", type=int, default=30, help="预热次数")
    p.add_argument("--iters", type=int, default=100, help="计时时推理次数")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备：cpu / 0 / cuda:0；默认自动（有 CUDA 则用 0）",
    )
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="测速图像；默认使用 ultralytics 自带 bus.jpg",
    )
    p.add_argument(
        "--modes",
        type=str,
        default="pt_fp32,onnx_fp32,trt_fp32,trt_fp16",
        help="逗号分隔：pt_fp32,onnx_fp32,trt_fp32,trt_fp16,pt_fp16,trt_int8；all=前四项（与常见填表一致）",
    )
    p.add_argument(
        "--force-export",
        action="store_true",
        help="始终重新导出 TensorRT；默认若 tensorrt_exports/<tag>/*.engine 已存在则跳过导出以节省时间",
    )
    return p.parse_args()


def resolve_device(arg_device: str | None) -> str:
    if arg_device is not None:
        return arg_device
    return "0" if torch.cuda.is_available() else "cpu"


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_predict(
    model: YOLO,
    source: str,
    imgsz: int,
    half: bool,
    device: str,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    """返回 (平均延迟 ms, FPS)。"""
    for _ in range(warmup):
        model.predict(source=source, imgsz=imgsz, half=half, verbose=False, device=device)
    cuda_sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.predict(source=source, imgsz=imgsz, half=half, verbose=False, device=device)
    cuda_sync()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    ms = elapsed / iters * 1000.0
    fps = iters / elapsed
    return ms, fps


def staged_weights(model_path: Path, tag: str, subdir: str = "tensorrt_exports") -> Path:
    """复制权重到 <subdir>/<tag>/，避免多次 export 覆盖同名产物。"""
    out_dir = _REPO_ROOT / subdir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / model_path.name
    shutil.copy2(model_path, dst)
    return dst


def export_trt(
    pt_staged: Path,
    *,
    half: bool,
    int8: bool,
    data: str | None,
    imgsz: int,
    device: str,
) -> Path:
    m = YOLO(str(pt_staged), task="jde")
    kwargs = dict(
        format="engine",
        imgsz=imgsz,
        batch=1,
        half=half,
        int8=int8,
        device=device,
        dynamic=True if int8 else False,
        simplify=True,
    )
    if int8:
        kwargs["data"] = data
    out = m.export(**kwargs)
    return Path(out)


def export_onnx_fp32(pt_staged: Path, imgsz: int, device: str | None) -> Path:
    """导出 FP32 ONNX；device=None 时由 Ultralytics 默认选择。"""
    m = YOLO(str(pt_staged), task="jde")
    kwargs = dict(
        format="onnx",
        imgsz=imgsz,
        batch=1,
        half=False,
        dynamic=False,
        simplify=True,
    )
    if device is not None:
        kwargs["device"] = device
    out = m.export(**kwargs)
    return Path(out)


def main() -> None:
    args = parse_args()
    model_path = (_REPO_ROOT / args.model).resolve()
    if not model_path.is_file():
        print(f"[错误] 找不到权重: {model_path}", file=sys.stderr)
        raise SystemExit(1)

    data_yaml = args.data
    if not Path(data_yaml).is_file():
        p = _REPO_ROOT / data_yaml
        if p.is_file():
            data_yaml = str(p)
        else:
            print(f"[错误] 找不到 data yaml: {args.data}", file=sys.stderr)
            raise SystemExit(1)

    source = args.source or str(ASSETS / "bus.jpg")
    if not Path(source).is_file():
        print(f"[错误] 找不到测速图像: {source}", file=sys.stderr)
        raise SystemExit(1)

    device = resolve_device(args.device)
    modes = args.modes.strip().lower()
    if modes == "all":
        want = {"pt_fp32", "onnx_fp32", "trt_fp32", "trt_fp16"}
    else:
        want = {x.strip() for x in modes.split(",") if x.strip()}

    print(f"权重: {model_path}")
    print(f"测速图: {source}")
    print(f"imgsz={args.imgsz}, warmup={args.warmup}, iters={args.iters}, device={device}")
    print(f"modes={sorted(want)}")
    print()

    rows: list[tuple[str, str, str, str, str]] = []

    # --- PyTorch FP32 ---
    if "pt_fp32" in want:
        try:
            m = YOLO(str(model_path), task="jde")
            ms, fps = bench_predict(
                m, source, args.imgsz, half=False, device=device, warmup=args.warmup, iters=args.iters
            )
            rows.append(("PyTorch (.pt)", "FP32", "predict(..., half=False)", f"{ms:.3f}", f"{fps:.2f}"))
            print(f"[OK] PyTorch FP32  平均延迟={ms:.3f} ms  FPS={fps:.2f}")
        except Exception as e:
            rows.append(("PyTorch (.pt)", "FP32", "—", "失败", str(e)))
            print(f"[FAIL] PyTorch FP32: {e}")

    # --- ONNX Runtime FP32（导出 .onnx 后以 YOLO 封装测速，与部署常用路径一致）---
    if "onnx_fp32" in want:
        try:
            staged = staged_weights(model_path, "fp32", subdir="onnx_exports")
            onnx_path = staged.with_suffix(".onnx")
            export_dev = None if device == "cpu" else device
            if not args.force_export and onnx_path.is_file():
                print(f"[reuse] 使用已有 onnx: {onnx_path}")
            else:
                onnx_path = export_onnx_fp32(staged, args.imgsz, device=export_dev)
            # 注：ONNX 当前由 AutoBackend 推断任务；显式 task=jde 可能与部分版本后处理不兼容
            m = YOLO(str(onnx_path))
            bench_dev = device
            ms, fps = bench_predict(
                m, source, args.imgsz, half=False, device=bench_dev, warmup=args.warmup, iters=args.iters
            )
            rows.append(("ONNX Runtime (.onnx)", "FP32", "export onnx FP32", f"{ms:.3f}", f"{fps:.2f}"))
            print(f"[OK] ONNX FP32  平均延迟={ms:.3f} ms  FPS={fps:.2f}  onnx={onnx_path}")
        except Exception as e:
            rows.append(("ONNX Runtime (.onnx)", "FP32", "—", "失败", str(e)))
            print(f"[FAIL] ONNX FP32: {e}")

    # --- TensorRT FP32 ---
    if "trt_fp32" in want:
        if not torch.cuda.is_available():
            rows.append(("TensorRT (.engine)", "FP32", "export FP32", "跳过", "无 CUDA"))
            print("[SKIP] TensorRT FP32：当前无 CUDA")
        else:
            try:
                staged = staged_weights(model_path, "fp32")
                engine_path = staged.with_suffix(".engine")
                if not args.force_export and engine_path.is_file():
                    print(f"[reuse] 使用已有 engine: {engine_path}")
                else:
                    engine_path = export_trt(
                        staged, half=False, int8=False, data=None, imgsz=args.imgsz, device="0"
                    )
                m = YOLO(str(engine_path))
                ms, fps = bench_predict(
                    m, source, args.imgsz, half=False, device="0", warmup=args.warmup, iters=args.iters
                )
                rows.append(("TensorRT (.engine)", "FP32", "export(..., half=False)", f"{ms:.3f}", f"{fps:.2f}"))
                print(f"[OK] TensorRT FP32  平均延迟={ms:.3f} ms  FPS={fps:.2f}  engine={engine_path}")
            except Exception as e:
                rows.append(("TensorRT (.engine)", "FP32", "—", "失败", str(e)))
                print(f"[FAIL] TensorRT FP32: {e}")

    # --- PyTorch FP16（需 CUDA）---
    if "pt_fp16" in want:
        if not torch.cuda.is_available():
            rows.append(("PyTorch (.pt)", "FP16", "half=True", "跳过", "无 CUDA"))
            print("[SKIP] PyTorch FP16：当前无 CUDA")
        else:
            try:
                m = YOLO(str(model_path), task="jde")
                dev = "0" if device == "cpu" else device
                ms, fps = bench_predict(
                    m, source, args.imgsz, half=True, device=dev, warmup=args.warmup, iters=args.iters
                )
                rows.append(("PyTorch (.pt)", "FP16", "predict(..., half=True)", f"{ms:.3f}", f"{fps:.2f}"))
                print(f"[OK] PyTorch FP16  平均延迟={ms:.3f} ms  FPS={fps:.2f}")
            except Exception as e:
                rows.append(("PyTorch (.pt)", "FP16", "—", "失败", str(e)))
                print(f"[FAIL] PyTorch FP16: {e}")

    # --- TensorRT FP16 ---
    if "trt_fp16" in want:
        if not torch.cuda.is_available():
            rows.append(("TensorRT (.engine)", "FP16", "export half=True", "跳过", "无 CUDA"))
            print("[SKIP] TensorRT FP16：当前无 CUDA")
        else:
            try:
                staged = staged_weights(model_path, "fp16")
                engine_path = staged.with_suffix(".engine")
                if not args.force_export and engine_path.is_file():
                    print(f"[reuse] 使用已有 engine: {engine_path}")
                else:
                    engine_path = export_trt(
                        staged, half=True, int8=False, data=None, imgsz=args.imgsz, device="0"
                    )
                m = YOLO(str(engine_path))
                ms, fps = bench_predict(
                    m, source, args.imgsz, half=False, device="0", warmup=args.warmup, iters=args.iters
                )
                rows.append(("TensorRT (.engine)", "FP16", "export(..., half=True)", f"{ms:.3f}", f"{fps:.2f}"))
                print(f"[OK] TensorRT FP16  平均延迟={ms:.3f} ms  FPS={fps:.2f}  engine={engine_path}")
            except Exception as e:
                rows.append(("TensorRT (.engine)", "FP16", "—", "失败", str(e)))
                print(f"[FAIL] TensorRT FP16: {e}")

    # --- TensorRT INT8 ---
    if "trt_int8" in want:
        if not torch.cuda.is_available():
            rows.append(("TensorRT (.engine)", "INT8", "export int8=True", "跳过", "无 CUDA"))
            print("[SKIP] TensorRT INT8：当前无 CUDA")
        else:
            try:
                staged = staged_weights(model_path, "int8")
                engine_path = staged.with_suffix(".engine")
                if not args.force_export and engine_path.is_file():
                    print(f"[reuse] 使用已有 engine: {engine_path}")
                else:
                    engine_path = export_trt(
                        staged,
                        half=False,
                        int8=True,
                        data=data_yaml,
                        imgsz=args.imgsz,
                        device="0",
                    )
                m = YOLO(str(engine_path))
                ms, fps = bench_predict(
                    m, source, args.imgsz, half=False, device="0", warmup=args.warmup, iters=args.iters
                )
                rows.append(("TensorRT (.engine)", "INT8", f"data={Path(data_yaml).name}", f"{ms:.3f}", f"{fps:.2f}"))
                print(f"[OK] TensorRT INT8  平均延迟={ms:.3f} ms  FPS={fps:.2f}  engine={engine_path}")
            except Exception as e:
                rows.append(("TensorRT (.engine)", "INT8", "—", "失败", str(e)))
                print(f"[FAIL] TensorRT INT8: {e}")

    print()
    print("### 汇总（predict 单图，不含 track 关联开销）")
    print("| 部署 | 精度 | 说明 | 平均延迟 (ms) | FPS |")
    print("|------|------|------|---------------|-----|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")


if __name__ == "__main__":
    main()
