"""
用 Ultralytics Solutions 的 Heatmap 对视频生成「累积热力」：模型跟踪框在时间上反复出现的区域越亮，
经 colormap 后高值偏红（近似「注意力/关注」在场景中的空间分布；不是 Grad-CAM）。

用法示例：
  cd YOLO11-JDE
  python scripts/heatmap_video.py \\
    --source video/your.mp4 \\
    --output video_outputs/heatmap_out.avi \\
    --model YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt \\
    --tracker ultralytics/cfg/trackers/jdetracker.yaml \\
    --imgsz 608 1088
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ultralytics import YOLO, solutions  # noqa: E402


def _patch_extract_tracks_for_track_kwargs(heatmap: solutions.Heatmap) -> None:
    """为 BaseSolution.extract_tracks 增加 imgsz / tracker / verbose 等，便于 JDE 权重与分辨率。"""

    def extract_tracks(self, im0):  # type: ignore[no-untyped-def]
        track_kw: dict = {"persist": True, "verbose": False, "classes": self.CFG["classes"]}
        if self.CFG.get("imgsz") is not None:
            track_kw["imgsz"] = self.CFG["imgsz"]
        if self.CFG.get("tracker"):
            track_kw["tracker"] = self.CFG["tracker"]

        self.tracks = self.model.track(source=im0, **track_kw)
        self.track_data = self.tracks[0].obb or self.tracks[0].boxes

        if self.track_data and self.track_data.id is not None:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
        else:
            self.boxes, self.clss, self.track_ids = [], [], []

    heatmap.extract_tracks = types.MethodType(extract_tracks, heatmap)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="输入视频路径")
    p.add_argument("--output", type=Path, default=_ROOT / "video_outputs" / "heatmap_output.avi")
    p.add_argument(
        "--model",
        type=Path,
        default=_ROOT / "YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt",
        help="YOLO 权重（JDE 需与本仓库 ultralytics 一致）",
    )
    p.add_argument(
        "--tracker",
        type=str,
        default=str(_ROOT / "ultralytics/cfg/trackers/jdetracker.yaml"),
        help="跟踪器配置（JDE 一般用 jdetracker.yaml）",
    )
    p.add_argument(
        "--colormap",
        type=int,
        default=cv2.COLORMAP_JET,
        help="OpenCV colormap 常量，默认 JET：低值偏蓝、高值偏红",
    )
    p.add_argument("--imgsz", type=int, nargs=2, default=[608, 1088], metavar=("H", "W"))
    p.add_argument("--show", action="store_true", help="是否弹窗预览")
    p.add_argument(
        "--task",
        type=str,
        default="jde",
        help="YOLO task（JDE 权重用 jde；通用检测可用 detect 并换 yolo11n.pt）",
    )
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.source))
    assert cap.isOpened(), f"无法打开视频: {args.source}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = max(1.0, float(fps))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))

    heatmap = solutions.Heatmap(
        show=args.show,
        model=str(args.model),
        colormap=args.colormap,
        imgsz=tuple(args.imgsz),
        tracker=args.tracker,
    )
    # BaseSolution 默认 YOLO(model) 不带 task；JDE 权重需显式指定
    heatmap.model = YOLO(str(args.model), task=args.task)
    heatmap.names = heatmap.model.names
    _patch_extract_tracks_for_track_kwargs(heatmap)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        out = heatmap.generate_heatmap(im0)
        video_writer.write(out)

    cap.release()
    video_writer.release()
    if args.show:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print(f"已保存: {args.output.resolve()}")


if __name__ == "__main__":
    main()
