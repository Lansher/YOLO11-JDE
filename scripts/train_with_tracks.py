from bootstrap import ensure_repo_root

ensure_repo_root()

"""
使用包含track ID的数据集训练JDE模型
这是提升ReID性能的关键步骤
"""

import comet_ml
from ultralytics import YOLO
import torch
from datetime import datetime
from functools import partial

"""
import os
# Set number of threads
N_THREADS = '32'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
os.environ['MKL_NUM_THREADS'] = N_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = N_THREADS
"""

# Initialize COMET logger
from ultralytics.utils import SETTINGS
SETTINGS['comet'] = True
comet_ml.init()

from tracker.evaluation.mot_callback import mot_eval

# 使用预训练模型
pre_trained_model = '/home/bns/sevenT/ly/YOLO11-JDE/YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt'
model = YOLO('yolo11s-jde.yaml', task='jde').load(pre_trained_model)

# ========== 训练配置 ==========
epochs = 50
batch = 8  # 根据GPU内存调整，如果内存充足可以增加到16

# 可选：启用MOT评估回调（如果内存充足）
# 注意：MOT评估会消耗额外内存，如果遇到内存不足可以禁用
# model.add_callback("on_val_end", partial(mot_eval, period=epochs))

if not torch.cuda.is_available():
    raise RuntimeError("当前环境未检测到可用 CUDA。为确保 GPU-only，请先修复 torch CUDA 依赖。")

device = [0]

model.train(
    project='reid_xps',
    name=f'CH-jde-{batch}b-{epochs}e_WITH_TRACKS_1280px' + '_' + datetime.now().strftime('%Y%m%d-%H%M%S'),

    # ========== 使用包含track ID的数据集 ==========
    # 使用convert_mot20_gt_to_yolo.py转换的数据集（包含track ID）
    data='/home/bns/sevenT/ly/datasets/MOT20_YOLO_label/MOT20_label.yaml',
    # 如果使用旧的数据集路径，使用：
    # data='/root/autodl-tmp/MOT20_YOLO_GT/MOT20_GT.yaml',
    # 如果仍使用原数据集（无track ID），使用：
    # data='MOT20.yaml',
    
    epochs=epochs,
    batch=batch,
    device=device,
    
    # ========== 学习率优化 ==========
    lr0=0.001,  # 降低初始学习率，适合fine-tuning
    lrf=0.01,   # 最终学习率比例
    cos_lr=True,  # 使用余弦学习率调度
    
    # ========== 冻结策略 ==========
    freeze=None,  # 全模型微调，充分利用track ID信息学习ReID特征
    
    # ========== 数据增强 ==========
    imgsz=1280,
    mosaic=1.0,  # 保持mosaic增强
    mixup=0.1,   # 添加mixup增强，提高泛化能力
    hsv_h=0.015,  # HSV色调增强
    hsv_s=0.7,    # HSV饱和度增强
    hsv_v=0.4,    # HSV明度增强
    degrees=5.0,  # 轻微旋转增强
    translate=0.1,  # 平移增强
    scale=0.5,    # 缩放增强
    fliplr=0.5,  # 水平翻转
    
    # ========== 损失权重优化 ==========
    box=7.5,     # 边界框损失权重（保持默认）
    cls=0.5,     # 分类损失权重（保持默认）
    dfl=1.5,     # DFL损失权重（保持默认）
    clr=1.5,     # 增加ReID损失权重，因为现在有track ID信息，可以更好地学习ReID特征
    
    # ========== 训练策略 ==========
    close_mosaic=0,     # Always 0 for JDE
    patience=50,        # 增加early stopping patience，给模型更多机会
    warmup_epochs=5.0,  # 增加warmup轮数，更稳定的训练开始
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # ========== 优化器 ==========
    optimizer='AdamW',  # 使用AdamW优化器，通常比SGD在fine-tuning时表现更好
    momentum=0.937,     # SGD momentum（如果使用SGD）
    weight_decay=0.0005,  # 权重衰减
    
    # ========== Tracker配置 ==========
    tracker='jdetracker_mot20_optimized.yaml',  # 使用优化后的tracker配置
    
    save=True,
    save_json=True,
    plots=True,
    verbose=True,
    cache=False,
    amp=True,  # 启用混合精度训练，加速训练并可能提高稳定性
    deterministic=False,
)
