from bootstrap import ensure_repo_root

ensure_repo_root()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ultralytics import YOLO
from tracker.evaluation.mot_callback import mot_eval

model = YOLO('/root/autodl-tmp/YOLO11-JDE/YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt', task="jde")
model.add_callback("on_val_start", mot_eval)  # 已启用，支持 MOT20

model.val(
    project='reid_xps',
    name=f'MOT20-test',
    data='/root/autodl-tmp/MOT20_YOLO/MOT20.yaml',  # 使用MOT20_YOLO目录下的配置文件
    imgsz=640,
    device='cpu',  # 使用CPU，因为没有可用的CUDA设备
    #max_det=150,
    tracker='yolojdetracker.yaml',
    half=False,
    amp=False,
)
