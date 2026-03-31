#!/usr/bin/env python3
"""
使用 VisDrone pedestrian-only + track_id 标签训练 JDE（只跑 GPU）。

参考：scripts/train_with_tracks.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from bootstrap import ensure_repo_root

ensure_repo_root()

try:
    import torch
except ImportError as e:
    raise RuntimeError(
        "import torch 失败，通常是 CUDA 版 torch 与系统 CUDA 库不匹配导致的。"
        "你可以用如下方式绕过（示例，路径按你的 env 调整）：\n"
        "  LD_PRELOAD=/home/bns/anaconda3/envs/yolo11jde/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12 "
        "python scripts/train_visdrone_ped_with_tracks.py"
    ) from e

from ultralytics import YOLO

# 可选：Comet（没有也能训练）
try:
    import comet_ml

    from ultralytics.utils import SETTINGS

    SETTINGS["comet"] = True
    try:
        comet_ml.init()
    except Exception:
        # API key 未配置等情况不阻断训练
        pass
except ModuleNotFoundError:
    from ultralytics.utils import SETTINGS

    SETTINGS["comet"] = False


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA。为确保 GPU-only，请使用 CUDA 版 PyTorch 并确保 import torch 成功。")

    ly_root = Path(__file__).resolve().parents[2]
    data_yaml = ly_root / "datasets" / "visdrone_jde_ped" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"找不到数据配置：{data_yaml}\n"
            "请先运行：python scripts/prepare_visdrone_ped_jde.py"
        )

    # 训练配置（默认与 train_with_tracks.py 保持一致）
    epochs = 50
    batch = 8
    imgsz = 1280

    device = [0]

    pre_trained_model = ly_root / "YOLO11-JDE" / "YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt"
    model = YOLO("yolo11s-jde.yaml", task="jde").load(str(pre_trained_model))

    model.train(
        project="reid_xps",
        name=f"VISDRONE-ped-jde-{batch}b-{epochs}e_WITH_TRACKS_{imgsz}px_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        device=device,
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        freeze=None,
        imgsz=imgsz,
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

