from bootstrap import ensure_repo_root

ensure_repo_root()

from ultralytics import YOLO
import torch
from datetime import datetime
from functools import partial

"""
import os
# Set number of threads
N_THREADS = '8'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
os.environ['MKL_NUM_THREADS'] = N_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = N_THREADS
"""

# Initialize COMET logger (optional)
try:
    import comet_ml

    # first log done in notebook where API key was asked, now seems to be saved in .comet.config
    from ultralytics.utils import SETTINGS

    SETTINGS["comet"] = True
    comet_ml.init()
except ModuleNotFoundError:
    from ultralytics.utils import SETTINGS

    SETTINGS["comet"] = False

from tracker.evaluation.mot_callback import mot_eval

pre_trained_model = '/root/autodl-tmp/YOLO11-JDE/YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt'
# Initialize model and load matching weights
model = YOLO('yolo11s-jde.yaml', task='jde').load(pre_trained_model)

epochs = 10
batch = 8

# Ultralytics 的 device 参数需要与当前环境匹配：
# - 当前环境无 CUDA 时，device=[0] 会报 Invalid CUDA
# - 有 CUDA 时默认用第 0 张卡
device = [0] if torch.cuda.is_available() else "cpu"

model.add_callback("on_val_end", partial(mot_eval, period=epochs))   # Evaluate every X epochs
model.train(
    project='reid_xps',
    name=f'CH-jde-{batch}b-{epochs}e_TBHS_m075_1280px' + '_' + datetime.now().strftime('%Y%m%d-%H%M%S'),

    data='/root/autodl-tmp/MOT20_YOLO_label/MOT20_GT.yaml',
    epochs=epochs,
    batch=batch,
    device=device,
    # bbox_erase=0.1,
    imgsz=1280,
    # clr=0.5,
    # Freeze layers up to N-1. 24 trains only Re-ID branch, 23 trains only heads, 11 freezes backbone
    freeze=11,

    close_mosaic=0,     # Always 0 for JDE
    patience=25,
    tracker='jdetracker.yaml',  # Tracker config file with ReID activated

    save=True,
    save_json=True,
    plots=True,
    verbose=True,
    cache=False,
    amp=False,
    deterministic=False,
)
