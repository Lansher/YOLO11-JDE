from bootstrap import ensure_repo_root

ensure_repo_root()

import comet_ml
from ultralytics import YOLO
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

# Initialize COMET logger, first log done in notebook where API key was asked, now seems to be saved in .comet.config
from ultralytics.utils import SETTINGS
SETTINGS['comet'] = True  # set True to log using Comet.ml
comet_ml.init()

from tracker.evaluation.mot_callback import mot_eval

pre_trained_model = '/root/autodl-tmp/YOLO11-JDE/YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt'
# Initialize model and load matching weights
model = YOLO('yolo11s-jde.yaml', task='jde').load(pre_trained_model)

# ========== 优化配置 ==========
epochs = 50  # 增加训练轮数：从10增加到50，给模型更多学习时间
batch = 8  # 如果GPU内存允许，可以尝试增加到24或32

# 暂时禁用mot_eval回调以避免验证后内存不足
# model.add_callback("on_val_end", partial(mot_eval, period=epochs))   # Evaluate every X epochs
model.train(
    project='reid_xps',
    name=f'CH-jde-{batch}b-{epochs}e_OPTIMIZED_TBHS_m075_1280px' + '_' + datetime.now().strftime('%Y%m%d-%H%M%S'),

    # data='MOT20.yaml',
    data='/root/autodl-tmp/MOT20_YOLO_label/MOT20_label.yaml',
    epochs=epochs,
    batch=batch,
    device=[0],
    
    # ========== 学习率优化 ==========
    lr0=0.001,  # 降低初始学习率：从默认0.01降到0.001，更稳定的fine-tuning
    lrf=0.01,   # 最终学习率比例
    cos_lr=True,  # 启用余弦学习率调度：更平滑的学习率衰减
    
    # ========== 冻结策略优化 ==========
    # 方案1：减少冻结层数，允许backbone微调（推荐）
    freeze=None,  # 不冻结，允许全模型微调（如果内存足够）
    # 方案2：如果内存不足，可以尝试只冻结前几层
    # freeze=5,  # 只冻结前5层，允许更多层参与学习
    
    # ========== 数据增强优化 ==========
    imgsz=1280,
    mosaic=1.0,  # 保持mosaic增强
    mixup=0.1,   # 添加mixup增强：提高泛化能力
    copy_paste=0.0,  # JDE任务通常不使用copy_paste
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
    clr=1.0,     # 对比学习损失权重（可以尝试增加到1.5-2.0以加强ReID学习）
    
    # ========== 训练策略优化 ==========
    close_mosaic=0,     # Always 0 for JDE
    patience=50,        # 增加early stopping patience：从25增加到50，给模型更多机会
    warmup_epochs=5.0,  # 增加warmup轮数：从3增加到5，更稳定的训练开始
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # ========== 优化器设置 ==========
    optimizer='AdamW',  # 使用AdamW优化器：通常比SGD在fine-tuning时表现更好
    momentum=0.937,     # SGD momentum（如果使用SGD）
    weight_decay=0.0005,  # 权重衰减
    
    tracker='jdetracker.yaml',  # Tracker config file with ReID activated

    save=True,
    save_json=True,
    plots=True,
    verbose=True,
    cache=False,
    amp=True,  # 启用混合精度训练：加速训练并可能提高稳定性
    deterministic=False,
    
    # ========== 其他优化 ==========
    multi_scale=False,  # 多尺度训练会增加训练时间，但可能提高mAP
    # multi_scale=True,  # 如果时间允许，可以启用多尺度训练
)
