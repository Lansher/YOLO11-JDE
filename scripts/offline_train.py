"""后台启动训练，日志写入仓库根目录下的 `logs/`。"""
import os
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TRAIN = _ROOT / "scripts" / "train.py"
_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / f"train_{datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.txt"

cmd = f"nohup sh -c '{sys.executable} {_TRAIN} > {_LOG_FILE} 2>&1' &"
os.system(cmd)
