#!/usr/bin/env python3
"""
准备 VisDrone pedestrian-only 的 JDE 训练数据集（保留 track_id）。

输入：
  - datasets/visdrone/images/train|val
  - datasets/visdrone/labels/train|val
    标签格式：class_id x_center y_center w h track_id（共 6 列）

输出：
  - datasets/visdrone_jde_ped/
    - images/train|val  (通过 symlink 指向原 images)
    - labels/train|val  (过滤 class_id==0 后的标签，保留 track_id)
    - data.yaml          (nc=1, names={0: pedestrian})
"""

from __future__ import annotations

from pathlib import Path
import os
import sys


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_symlink(target: Path, link: Path) -> None:
    """
    创建 symlink 并尽量做到可重复运行：
    - link 不存在 -> 创建
    - link 存在且指向正确 target -> 跳过
    - 否则 -> 报错
    """
    if link.exists() or link.is_symlink():
        if link.is_symlink() and os.path.realpath(link) == str(target.resolve()):
            return
        raise RuntimeError(f"目标路径已存在且无法复用 symlink：{link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def _filter_labels(src_label_dir: Path, dst_label_dir: Path) -> None:
    _ensure_dir(dst_label_dir)

    label_files = sorted(src_label_dir.glob("*.txt"))
    if not label_files:
        raise RuntimeError(f"未在 {src_label_dir} 找到任何 .txt 标签文件")

    for lf in label_files:
        out_lines: list[str] = []
        with lf.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 6:
                    # 不符合格式就跳过，避免训练因单文件异常中断
                    continue
                class_id = int(float(parts[0]))  # 兼容偶发的科学计数/浮点
                x_c, y_c, w, h = parts[1], parts[2], parts[3], parts[4]
                track_id = int(float(parts[5]))

                # pedestrian-only：VisDrone 的 pedestrian 通常是 class_id=0
                if class_id != 0:
                    continue

                out_lines.append(f"0 {float(x_c):.6f} {float(y_c):.6f} {float(w):.6f} {float(h):.6f} {track_id}")

        out_path = dst_label_dir / lf.name
        with out_path.open("w", encoding="utf-8") as wf:
            wf.write("\n".join(out_lines))


def main() -> None:
    ly_root = Path(__file__).resolve().parents[2]
    src_root = ly_root / "datasets" / "visdrone"
    dst_root = ly_root / "datasets" / "visdrone_jde_ped"

    splits = ["train", "val"]
    src_images = src_root / "images"
    src_labels = src_root / "labels"

    dst_images = dst_root / "images"
    dst_labels = dst_root / "labels"

    # 基础校验
    for sp in splits:
        if not (src_images / sp).exists():
            raise RuntimeError(f"源 images 不存在：{src_images / sp}")
        if not (src_labels / sp).exists():
            raise RuntimeError(f"源 labels 不存在：{src_labels / sp}")

    # 创建目标目录
    _ensure_dir(dst_root)
    for sp in splits:
        _ensure_dir(dst_labels / sp)

        # images 用 symlink，避免复制大文件
        _safe_symlink(src_images / sp, dst_images / sp)

    # 过滤标签（仅保留 class_id==0）
    for sp in splits:
        _filter_labels(src_labels / sp, dst_labels / sp)

    # data.yaml
    data_yaml = dst_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "# VisDrone (pedestrian-only) prepared for JDE training",
                f"path: {dst_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "nc: 1",
                "names:",
                "  0: pedestrian",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"✓ Done. Prepared dataset: {dst_root}")
    print(f"✓ data.yaml: {data_yaml}")


if __name__ == "__main__":
    main()

