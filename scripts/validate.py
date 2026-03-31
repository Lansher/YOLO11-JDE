from bootstrap import ensure_repo_root

ensure_repo_root()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ultralytics import YOLO
from tracker.evaluation.mot_callback import mot_eval

model = YOLO('/home/bns/sevenT/ly/YOLO11-JDE/reid_xps/CH-jde-8b-50e_WITH_TRACKS_1280px_20260331-152925/weights/best.pt', task="jde")
model.add_callback("on_val_start", mot_eval)

model.val(
    project='reid_xps',
    name=f'MOT20-test',
    data='/home/bns/sevenT/ly/datasets/MOT20_YOLO_label/MOT20_label.yaml',
    imgsz=1280,
    device='0',  
    #max_det=150,
    tracker='yolojdetracker.yaml',
    half=False,
    amp=False,
    split='test',
)
