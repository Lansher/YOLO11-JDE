#!/usr/bin/env python3
# Copyright AGPL-3.0 — standalone TensorRT build (no torch import) for memory-starved devices.
"""Build a YOLO/Ultralytics .engine from ONNX in a clean process (no PyTorch on GPU).

Run as a script path, not as ``python -m ultralytics...``, so ``ultralytics`` is never imported
and CUDA memory is not held by the training stack.

.. code-block:: bash

   python ultralytics/engine/trt_isolated_builder.py --onnx model.onnx --engine model.engine \\
       --meta-json meta.json --workspace-gb 0.125
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _build(
    *,
    f_onnx: str,
    f_engine: str,
    meta_path: str,
    workspace_gb: float,
    fp16: bool,
    jetson_restrict_tac: bool,
    verbose: bool,
) -> None:
    # Orin/Ampere 上 TF32 可能在 tactic 试探时多占显存；须在 import tensorrt（进而加载 CUDA/cuBLAS）之前设置
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    import tensorrt as trt  # noqa: PLC0415 — load only in child, after CLI parse

    onnx_p = Path(f_onnx)
    if not onnx_p.is_file():
        raise SystemExit(f"ONNX not found: {f_onnx}")
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    meta_s = json.dumps(meta)
    if len(meta_s) > 2**31 - 1:
        raise SystemExit("metadata JSON too large")

    is_trt10 = int(trt.__version__.split(".")[0]) >= 10
    workspace = int(workspace_gb * (1 << 30))

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    if is_trt10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    else:
        config.max_workspace_size = workspace

    if jetson_restrict_tac:
        try:
            TS = getattr(trt, "TacticSource", None)
            if TS is not None:
                tac = (1 << int(TS.CUBLAS)) | (1 << int(TS.CUBLAS_LT))
                config.set_tactic_sources(tac)
        except Exception:
            pass

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_p)):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise SystemExit("ONNX parse failed")

    half = fp16 and builder.platform_has_fast_fp16
    if half:
        config.set_flag(trt.BuilderFlag.FP16)
    # 关闭 TF32，降低部分卷积 tactic 试探时的显存峰值（略慢，Jetson 上常值得）
    try:
        if hasattr(trt.BuilderFlag, "TF32"):
            config.clear_flag(trt.BuilderFlag.TF32)
    except Exception:
        pass

    build = builder.build_serialized_network if is_trt10 else builder.build_engine
    out = Path(f_engine)
    out.parent.mkdir(parents=True, exist_ok=True)
    with build(network, config) as engine, open(out, "wb") as t:
        if engine is None:
            raise SystemExit("TensorRT build_engine failed")
        t.write(len(meta_s).to_bytes(4, byteorder="little", signed=True))
        t.write(meta_s.encode("utf-8"))
        t.write(engine if is_trt10 else engine.serialize())


def main() -> None:
    ap = argparse.ArgumentParser(description="TensorRT engine build without PyTorch")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--meta-json", required=True, help="Ultralytics metadata dict as JSON")
    ap.add_argument("--workspace-gb", type=float, default=0.125)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-jetson-tac", action="store_true", help="Allow all tactic sources (needs more VRAM)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    _build(
        f_onnx=args.onnx,
        f_engine=args.engine,
        meta_path=args.meta_json,
        workspace_gb=args.workspace_gb,
        fp16=args.fp16,
        jetson_restrict_tac=not args.no_jetson_tac,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
