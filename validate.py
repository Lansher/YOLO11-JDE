#!/usr/bin/env python3
"""兼容入口：转发到 `scripts/validate.py`。"""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "scripts" / "validate.py"
    raise SystemExit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))


if __name__ == "__main__":
    main()
