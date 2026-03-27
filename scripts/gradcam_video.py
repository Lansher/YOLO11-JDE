from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Grad-CAM video for YOLO11-JDE model.")
    p.add_argument("--source", type=Path, required=True, help="Input video path.")
    p.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "video_outputs" / "gradcam_output.avi",
        help="Output video path.",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=_ROOT / "YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt",
        help="JDE checkpoint path.",
    )
    p.add_argument("--imgsz", type=int, nargs=2, default=[608, 1088], metavar=("H", "W"))
    p.add_argument("--target-layer", type=int, default=16, help="Index in model.model to hook for CAM.")
    p.add_argument(
        "--score-index",
        type=int,
        default=4,
        help="Channel index in prediction tensor to optimize (JDE default obj/conf channel is usually 4).",
    )
    p.add_argument("--topk", type=int, default=20, help="Average top-k scores before backward.")
    p.add_argument("--alpha", type=float, default=0.45, help="Heatmap blend ratio.")
    p.add_argument("--colormap", type=int, default=cv2.COLORMAP_JET, help="OpenCV colormap id.")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-frames", type=int, default=0, help="0 means process full video.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = YOLO(str(args.model), task="jde").model.to(device).eval()
    modules = dict(model.model.named_children())
    layer_key = str(args.target_layer)
    if layer_key not in modules:
        raise ValueError(f"target layer {args.target_layer} not found. Available: {list(modules.keys())}")
    target_module = modules[layer_key]

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_, __, output):
        activations.clear()
        activations.append(output if isinstance(output, torch.Tensor) else output[0])

    def bwd_hook(_, grad_input, grad_output):
        del grad_input
        gradients.clear()
        gradients.append(grad_output[0])

    h1 = target_module.register_forward_hook(fwd_hook)
    h2 = target_module.register_full_backward_hook(bwd_hook)

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (out_w, out_h),
    )

    max_frames = args.max_frames if args.max_frames > 0 else total
    idx = 0

    try:
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if max_frames and idx >= max_frames:
                break

            in_h, in_w = args.imgsz
            resized = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            inp.requires_grad_(True)

            model.zero_grad(set_to_none=True)
            pred, _ = model(inp)
            scores = pred[0, args.score_index, :]
            k = max(1, min(args.topk, scores.numel()))
            y = torch.topk(scores, k=k).values.mean()
            y.backward()

            if not activations or not gradients:
                cam_small = np.zeros((in_h, in_w), dtype=np.float32)
            else:
                acts = activations[0]
                grads = gradients[0]
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = (weights * acts).sum(dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = F.interpolate(cam, size=(out_h, out_w), mode="bilinear", align_corners=False)
                cam = cam[0, 0].detach().cpu().numpy()
                cam -= cam.min()
                cam /= (cam.max() + 1e-6)
                cam_small = cam.astype(np.float32)

            heat_u8 = np.uint8(np.clip(cam_small * 255.0, 0, 255))
            heat_bgr = cv2.applyColorMap(heat_u8, args.colormap)
            overlay = cv2.addWeighted(frame_bgr, 1.0 - args.alpha, heat_bgr, args.alpha, 0.0)
            cv2.putText(
                overlay,
                f"Grad-CAM score={float(y.detach().cpu()):.4f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(overlay)
            idx += 1
            if idx % 25 == 0:
                print(f"processed {idx}/{total if total > 0 else '?'} frames")
    finally:
        h1.remove()
        h2.remove()
        cap.release()
        writer.release()

    print(f"Saved Grad-CAM video: {args.output.resolve()}")


if __name__ == "__main__":
    main()
