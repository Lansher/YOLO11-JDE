#!/usr/bin/env python3
"""兼容入口：转发到 `scripts/train.py`（请在仓库根目录执行）。"""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "scripts" / "train.py"
    raise SystemExit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))


if __name__ == "__main__":
    main()
