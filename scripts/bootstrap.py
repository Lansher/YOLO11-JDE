"""将仓库根目录加入 sys.path，便于从 `scripts/` 运行时导入 `tracker` 等本地包。"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_repo_root() -> None:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
