#!/usr/bin/env python3
"""
一键复现：与「表：Jetson 等部署对比」常见四行一致——
  PyTorch FP32、ONNX Runtime FP32、TensorRT FP32、TensorRT FP16（predict 单图测速）。

在仓库根目录执行：
  python scripts/benchmark_deploy.py --modes all
  python scripts/benchmark_deploy.py --modes pt_fp32,onnx_fp32,trt_fp32,trt_fp16,trt_int8

另支持：pt_fp16、trt_int8（需 data yaml）。

TensorRT 导出必须在 NVIDIA GPU + TensorRT 环境下进行；ONNX FP32 可在 CPU 上导出。
"""
from __future__ import annotations

import argparse
import gc
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


def _is_jetson_device() -> bool:
    return Path("/etc/nv_tegra_release").is_file()


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
    p.add_argument("--imgsz", type=int, default=640, help="推理边长（与训练一致）")
    p.add_argument("--warmup", type=int, default=15, help="预热次数")
    p.add_argument("--iters", type=int, default=50, help="计时时推理次数")
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
    p.add_argument(
        "--trt-workspace",
        type=float,
        default=None,
        help="TensorRT builder workspace（GB）。默认 Jetson=0.125、其它=1；仍 OOM 可试 0.05",
    )
    return p.parse_args()


def resolve_device(arg_device: str | None) -> str:
    if arg_device is not None:
        return arg_device
    return "0" if torch.cuda.is_available() else "cpu"


def warn_if_jetson_without_cuda() -> None:
    """Jetson 上若误装 PyPI 的 torch+cpu，则无法 GPU 推理；打印一次性说明。"""
    if torch.cuda.is_available():
        return
    if not Path("/etc/nv_tegra_release").is_file():
        return
    v = getattr(torch.version, "cuda", None)
    print(
        "[提示] 本机为 Jetson，但 torch 无可用 CUDA（cuda.is_available()=False"
        + (f", torch.version.cuda={v!r}" if v else "")
        + "）。\n"
        "      请卸载 PyPI 的 CPU 版 torch，改装与 JetPack 匹配的 NVIDIA 官方 wheel，\n"
        "      见: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html\n"
        "      安装后通常需: export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH\n"
    )


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def free_cuda_memory() -> None:
    """TensorRT 建引擎前尽量释放显存，避免 tactic 阶段 OOM/段错误。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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


def _export_tag(kind: str, imgsz: int) -> str:
    """区分不同 imgsz 的导出目录，避免误用旧分辨率下的 .onnx/.engine。"""
    return f"{kind}_i{imgsz}"


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
    workspace_gb: float,
) -> Path:
    free_cuda_memory()
    m = YOLO(str(pt_staged), task="jde")
    # workspace 过大时 Jetson 在 tactic 选型阶段易显存差数 MB 即崩溃（段错误）
    kwargs = dict(
        format="engine",
        imgsz=imgsz,
        batch=1,
        half=half,
        int8=int8,
        device=device,
        dynamic=True if int8 else False,
        simplify=True,
        workspace=workspace_gb,
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

    source = args.source or str("/home/jetson/workspace/YOLO11-JDE/heat_ori.png")
    if not Path(source).is_file():
        print(f"[错误] 找不到测速图像: {source}", file=sys.stderr)
        raise SystemExit(1)

    device = resolve_device(args.device)
    warn_if_jetson_without_cuda()
    dlow = str(device).lower()
    if not torch.cuda.is_available() and dlow != "cpu" and not dlow.startswith("cpu"):
        print(
            f"[错误] 已指定 device={device!r}，但 torch.cuda.is_available() 为 False，无法使用 GPU。",
            file=sys.stderr,
        )
        raise SystemExit(2)
    modes = args.modes.strip().lower()
    if modes == "all":
        want = {"pt_fp32", "onnx_fp32", "trt_fp32", "trt_fp16"}
    else:
        want = {x.strip() for x in modes.split(",") if x.strip()}

    trt_ws = args.trt_workspace
    if trt_ws is None:
        # Jetson 大分辨率 YOLO：0.25 仍可能在 tactic 阶段差数百 MB；默认 0.125，不足再 --trt-workspace 0.05
        trt_ws = 0.125 if _is_jetson_device() else 1.0

    print(f"权重: {model_path}")
    print(f"测速图: {source}")
    print(f"imgsz={args.imgsz}, warmup={args.warmup}, iters={args.iters}, device={device}")
    if _is_jetson_device() and args.imgsz >= 960:
        print(
            "[提示] Jetson 上 imgsz≥960 时，YOLO11s-JDE + TensorRT FP16 单帧常为数十 ms；"
            "约 50 FPS（~20 ms/帧）通常需 imgsz≤640、更小骨干（如 n）、或 INT8（需校准数据集），"
            "并确认已用最大功耗/时钟（如 nvpmodel + jetson_clocks）。"
        )
    if want & {"trt_fp32", "trt_fp16", "trt_int8"}:
        print(
            f"TensorRT workspace={trt_ws} GB；Jetson 上 FP32/FP16 在**子进程**内建引擎（无 PyTorch 占显存），"
            "见 ultralytics/engine/trt_isolated_builder.py 与 docs/问题修复记录 §4"
        )
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
        finally:
            free_cuda_memory()

    # --- ONNX Runtime FP32（导出 .onnx 后以 YOLO 封装测速，与部署常用路径一致）---
    if "onnx_fp32" in want:
        try:
            staged = staged_weights(model_path, _export_tag("fp32", args.imgsz), subdir="onnx_exports")
            onnx_path = staged.with_suffix(".onnx")
            export_dev = None if device == "cpu" else device
            if not args.force_export and onnx_path.is_file():
                print(f"[reuse] 使用已有 onnx: {onnx_path}")
            else:
                onnx_path = export_onnx_fp32(staged, args.imgsz, device=export_dev)
            # 必须为 jde：否则走后处理 detect 路径，nc 会把嵌入维当成分类，候选暴涨，NMS 极慢/超时
            m = YOLO(str(onnx_path), task="jde")
            bench_dev = device
            ms, fps = bench_predict(
                m, source, args.imgsz, half=False, device=bench_dev, warmup=args.warmup, iters=args.iters
            )
            rows.append(("ONNX Runtime (.onnx)", "FP32", "export onnx FP32", f"{ms:.3f}", f"{fps:.2f}"))
            print(f"[OK] ONNX FP32  平均延迟={ms:.3f} ms  FPS={fps:.2f}  onnx={onnx_path}")
        except Exception as e:
            rows.append(("ONNX Runtime (.onnx)", "FP32", "—", "失败", str(e)))
            print(f"[FAIL] ONNX FP32: {e}")
        finally:
            free_cuda_memory()

    # --- TensorRT FP32 ---
    if "trt_fp32" in want:
        if not torch.cuda.is_available():
            rows.append(("TensorRT (.engine)", "FP32", "export FP32", "跳过", "无 CUDA"))
            print("[SKIP] TensorRT FP32：当前无 CUDA")
        else:
            try:
                staged = staged_weights(model_path, _export_tag("fp32", args.imgsz))
                engine_path = staged.with_suffix(".engine")
                if not args.force_export and engine_path.is_file():
                    print(f"[reuse] 使用已有 engine: {engine_path}")
                else:
                    engine_path = export_trt(
                        staged,
                        half=False,
                        int8=False,
                        data=None,
                        imgsz=args.imgsz,
                        device="0",
                        workspace_gb=trt_ws,
                    )
                m = YOLO(str(engine_path), task="jde")
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
                staged = staged_weights(model_path, _export_tag("fp16", args.imgsz))
                engine_path = staged.with_suffix(".engine")
                if not args.force_export and engine_path.is_file():
                    print(f"[reuse] 使用已有 engine: {engine_path}")
                else:
                    engine_path = export_trt(
                        staged,
                        half=True,
                        int8=False,
                        data=None,
                        imgsz=args.imgsz,
                        device="0",
                        workspace_gb=trt_ws,
                    )
                m = YOLO(str(engine_path), task="jde")
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
                staged = staged_weights(model_path, _export_tag("int8", args.imgsz))
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
                        workspace_gb=trt_ws,
                    )
                m = YOLO(str(engine_path), task="jde")
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
