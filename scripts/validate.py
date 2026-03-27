from bootstrap import ensure_repo_root

ensure_repo_root()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ultralytics import YOLO
from tracker.evaluation.mot_callback import mot_eval

model = YOLO('/root/autodl-tmp/reid_xps/CH-jde-8b-50e_OPTIMIZED_TBHS_m075_1280px_20260127-162020/weights/best.pt', task="jde")
model.add_callback("on_val_start", mot_eval)

model.val(
    project='reid_xps',
    name=f'MOT20-test',
    data='/root/autodl-tmp/MOT20_YOLO_label/MOT20_label.yaml',
    imgsz=640,
    device='0',  # 使用CPU，因为没有可用的CUDA设备
    #max_det=150,
    tracker='yolojdetracker.yaml',
    half=False,
    amp=False,
)
