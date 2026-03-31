#!/usr/bin/env python3
"""
JDE 训练脚本（EMA 版本，GPU-only）。

参考：scripts/train_with_tracks.py
变更点：
  - 模型结构使用：ultralytics/cfg/models/11/yolo11-jde-ema.yaml
  - 仍然保持 device=[0]，并在无 CUDA 时直接报错
  - 预训练权重用于初始化：如果因为结构/层索引变化导致部分 key 不匹配，
    Ultralytics 会跳过不匹配的参数（EMA 层本身通常没有预训练权重）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from bootstrap import ensure_repo_root

ensure_repo_root()

import torch
from ultralytics import YOLO

# 可选：Comet（没有也不影响训练）
try:
    import comet_ml

    from ultralytics.utils import SETTINGS

    SETTINGS["comet"] = True
    try:
        comet_ml.init()
    except Exception:
        SETTINGS["comet"] = False
except ModuleNotFoundError:
    from ultralytics.utils import SETTINGS

    SETTINGS["comet"] = False


# ======= 线程/环境设置（保持和 train_with_tracks.py 接近）=======
import os

N_THREADS = "32"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA。为确保 GPU-only，请使用 CUDA 版 torch 并确认 torch.cuda.is_available()=True。")

    device = [0]
    ly_root = Path(__file__).resolve().parents[1]  # YOLO11-JDE

    # 数据集：默认沿用你 train_with_tracks.py 的 MOT20 track_id 数据
    # 如果你想改成 VisDrone pedestrian-only track_id 数据，只需把 data 改成：
    #   data=str(ly_root / "datasets" / "visdrone_jde_ped" / "data.yaml")
    # 注意：MOT20_YOLO_label 在工程根目录的 datasets 下，而不是 YOLO11-JDE/datasets
    data = str(ly_root.parent / "datasets" / "MOT20_YOLO_label" / "MOT20_label.yaml")

    # ======= 预训练权重与模型配置 =======
    pre_trained_model = str(ly_root / "YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt")
    cfg_ema = str(ly_root / "ultralytics" / "cfg" / "models" / "11" / "yolo11-jde-ema.yaml")

    # 先用 EMA YAML 构建，再加载预训练权重（非严格加载：不匹配 key 会被跳过）
    model = YOLO(cfg_ema, task="jde")
    try:
        model = model.load(pre_trained_model)
    except Exception:
        # 如果结构差异导致无法加载全部权重，也允许从 EMA 结构初始化继续训练
        print("⚠️ 预训练权重加载失败（可能是部分层不匹配）。将继续使用 EMA 结构随机初始化。")

    # ========== 训练配置 ==========
    epochs = 50
    batch = 8

    model.train(
        project="reid_xps",
        name=f"CH-jde-EMA-{batch}b-{epochs}e_WITH_TRACKS_1280px_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        data=data,
        epochs=epochs,
        batch=batch,
        device=device,
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        freeze=None,
        imgsz=1280,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        clr=1.5,
        close_mosaic=0,
        patience=50,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        optimizer="AdamW",
        momentum=0.937,
        weight_decay=0.0005,
        tracker="jdetracker_mot20_optimized.yaml",
        save=True,
        save_json=True,
        plots=True,
        verbose=True,
        cache=False,
        amp=True,
        deterministic=False,
    )


if __name__ == "__main__":
    main()

