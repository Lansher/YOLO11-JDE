"""将仓库根目录加入 sys.path，便于从 `scripts/` 运行时导入 `tracker` 等本地包。"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _numpy_compat_np_bool() -> None:
    """NumPy 2.0 起移除 np.bool；TensorRT 等旧绑定仍可能访问 np.bool。"""
    import numpy as np

    if not hasattr(np, "bool"):
        try:
            np.bool = np.bool_  # type: ignore[attr-defined,misc]
        except AttributeError:
            np.bool = bool  # type: ignore[attr-defined,misc]


def ensure_repo_root() -> None:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _numpy_compat_np_bool()
