#!/usr/bin/env python3
"""
brake_ped_road.py - Pedestrian-on-Road Brake Decision Pipeline

Combines two models:
    1. YOLO11-Large on QNN/HTP (NPU) — pedestrian detection
    2. YOLOPv2 on CPU (TorchScript) — drivable area (road) segmentation

Logic: If any pedestrian bounding box has pixel-level overlap with the
YOLOPv2 road mask, issue BRAKE. Otherwise, KEEP DRIVING.

Each output frame is annotated with:
    - Green semi-transparent road mask overlay
    - Pedestrian bounding boxes (red if on road, green if off road)
    - Large centered text banner: "BRAKE" (red) or "KEEP DRIVING" (green)

Usage:
    source ~/npu_setup.sh
    python3 scripts/brake_ped_road.py \
        --source ~/datasets/kitti_videos/2011_09_26_drive_0046.mp4 \
        --max-frames 30 \
        --output ~/annotated_outputs/brake_ped_road_0046.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure YOLOPv2 repo root is on sys.path for utils imports
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
YOLO11_MODEL_PATH = Path("/home/saferide/weights/libyolo11l.so")
YOLOPV2_WEIGHTS = Path(__file__).resolve().parent.parent / "data" / "weights" / "yolopv2.pt"

YOLO_INPUT_SIZE = 640
PERSON_CLASS_ID = 0
PERSON_CONF_MIN = 0.25
NMS_IOU_THRESH = 0.45

# Overlay styling
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# =========================================================================
# YOLO11 QNN — pedestrian detection (from fused_pipeline.py)
# =========================================================================

def init_qnn_yolo(model_path: Path):
    """Initialize QNN YOLO model on HTP."""
    from qai_appbuilder import (
        LogLevel, PerfProfile, ProfilingLevel, QNNConfig, QNNContext, Runtime,
    )

    class QnnYolo(QNNContext):
        def Inference(self, input_data):
            inputs = list(input_data) if isinstance(input_data, (list, tuple)) else [input_data]
            return super().Inference(inputs)

    sdk_root = Path(os.environ.get(
        "QNN_SDK_ROOT",
        os.environ.get("QAIRT_SDK_ROOT", os.environ.get("SNPE_ROOT", "/opt/qairt")),
    ))
    lib_dir = sdk_root / "lib/aarch64-oe-linux-gcc11.2"
    if not lib_dir.exists():
        print(f"ERROR: QNN SDK lib dir not found: {lib_dir}. Source npu_setup.sh first.")
        sys.exit(1)

    QNNConfig.Config(str(lib_dir), Runtime.HTP, LogLevel.WARN, ProfilingLevel.BASIC)
    yolo = QnnYolo("yolo_brake", str(model_path))
    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
    print(f"YOLO11L loaded on HTP: {model_path.name}")
    return yolo


def letterbox_frame(frame_bgr: np.ndarray, input_size: int = 640):
    """Letterbox BGR frame to (input_size, input_size), return HWC float32 RGB [0,1]."""
    orig_h, orig_w = frame_bgr.shape[:2]
    scale = min(input_size / float(orig_w), input_size / float(orig_h))
    nw, nh = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), 128, dtype=np.uint8)
    dw = (input_size - nw) // 2
    dh = (input_size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw, :] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb), (orig_w, orig_h)


def run_yolo_inference(model, input_nhwc: np.ndarray):
    """Run QNN inference; returns (boxes [4,N], scores [80,N])."""
    try:
        outputs = model.Inference(input_nhwc)
    except Exception:
        x_nchw = np.transpose(input_nhwc[0], (2, 0, 1))[np.newaxis, ...]
        outputs = model.Inference(np.ascontiguousarray(x_nchw))

    outs = list(outputs.values()) if isinstance(outputs, dict) else list(outputs)
    tensors = [np.squeeze(np.asarray(o, dtype=np.float32)) for o in outs]

    if len(tensors) == 1:
        t = tensors[0]
        n = t.size
        if n % 84 == 0:
            N = n // 84
            combined = t.reshape(84, N) if t.shape[0] == 84 else t.reshape(N, 84).T
            return combined[:4], combined[4:]
        if n % 85 == 0:
            N = n // 85
            combined = t.reshape(85, N) if t.shape[0] == 85 else t.reshape(N, 85).T
            return combined[:4], combined[5:]

    boxes_t = scores_t = None
    for t in tensors:
        if t.size == 4 * 8400:
            boxes_t = t
        elif t.size == 80 * 8400:
            scores_t = t
    if boxes_t is None or scores_t is None and len(tensors) >= 2:
        a, b = tensors[0], tensors[1]
        if a.size < b.size:
            boxes_t, scores_t = a, b
        else:
            scores_t, boxes_t = a, b

    return boxes_t.reshape(4, -1), scores_t.reshape(80, -1)


def _is_xyxy(box_coords):
    if len(box_coords) < 2:
        return False
    check = (box_coords[:, 2] > box_coords[:, 0]) & (box_coords[:, 3] > box_coords[:, 1])
    return check.mean() > 0.95


def _xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


def postprocess_yolo(boxes_4n, scores_80n, orig_size, conf_thresh, iou_thresh):
    """Returns list of pedestrian dicts [{x1, y1, x2, y2, conf}, ...]."""
    combined = np.concatenate([boxes_4n, scores_80n], axis=0).T  # [N, 84]
    box_coords = combined[:, :4]
    class_scores = combined[:, 4:]

    if _is_xyxy(box_coords):
        box_coords = _xyxy_to_cxcywh(box_coords)

    # Filter to person class only
    person_scores = class_scores[:, PERSON_CLASS_ID]
    mask = person_scores > conf_thresh
    if not mask.any():
        return []

    filt_boxes = box_coords[mask]
    filt_scores = person_scores[mask]

    # NMS
    keep = _nms(filt_boxes, filt_scores, iou_thresh)
    if not keep:
        return []
    filt_boxes = filt_boxes[keep]
    filt_scores = filt_scores[keep]

    # Rescale to original image coords
    orig_w, orig_h = orig_size
    scale = min(YOLO_INPUT_SIZE / orig_w, YOLO_INPUT_SIZE / orig_h)
    pad_w = (YOLO_INPUT_SIZE - orig_w * scale) / 2
    pad_h = (YOLO_INPUT_SIZE - orig_h * scale) / 2

    filt_boxes[:, 0] = (filt_boxes[:, 0] - pad_w) / scale
    filt_boxes[:, 1] = (filt_boxes[:, 1] - pad_h) / scale
    filt_boxes[:, 2] /= scale
    filt_boxes[:, 3] /= scale

    peds = []
    for i in range(len(filt_boxes)):
        cx, cy, w, h = filt_boxes[i]
        x1 = max(0, cx - w / 2)
        y1 = max(0, cy - h / 2)
        x2 = min(orig_w, cx + w / 2)
        y2 = min(orig_h, cy + h / 2)
        peds.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(filt_scores[i])})
    return peds


def _nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2; y1 = cy - h / 2; x2 = cx + w / 2; y2 = cy + h / 2
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


# =========================================================================
# YOLOPv2 — road segmentation (from run_amsterdam_test.py)
# =========================================================================

def init_yolopv2(weights_path: Path, device):
    """Load YOLOPv2 TorchScript model."""
    model = torch.jit.load(str(weights_path), map_location='cpu')
    model = model.to(device).eval()
    print(f"YOLOPv2 loaded on {device}: {weights_path.name}")
    return model


def yolopv2_preprocess(frame_bgr: np.ndarray, img_size: int = 640, stride: int = 32):
    """Resize and pad frame for YOLOPv2, return (tensor [1,3,H,W], orig_shape)."""
    orig_h, orig_w = frame_bgr.shape[:2]
    # Compute new shape preserving aspect ratio
    r = min(img_size / orig_h, img_size / orig_w)
    new_h, new_w = int(orig_h * r), int(orig_w * r)
    # Pad to stride multiple
    pad_h = (stride - new_h % stride) % stride
    pad_w = (stride - new_w % stride) % stride
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Pad bottom and right
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # HWC BGR -> CHW RGB float32
    img = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor, (orig_h, orig_w), (new_h + pad_h, new_w + pad_w)


def yolopv2_road_mask(model, frame_bgr: np.ndarray, device, img_size: int = 640):
    """Run YOLOPv2 and return binary road mask at original frame resolution."""
    from utils.utils import split_for_trace_model, driving_area_mask

    tensor, (orig_h, orig_w), (proc_h, proc_w) = yolopv2_preprocess(frame_bgr, img_size)
    tensor = tensor.to(device).float()

    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(tensor)

    da_mask = driving_area_mask(seg)  # H x W uint8, 0 or 1

    # Crop padding then resize to original
    r = min(img_size / orig_h, img_size / orig_w)
    new_h, new_w = int(orig_h * r), int(orig_w * r)
    da_mask_cropped = da_mask[:new_h, :new_w]
    road_mask = cv2.resize(da_mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return road_mask


# =========================================================================
# Brake decision (from brake_overlay.py)
# =========================================================================

def ped_intersects_road(ped: Dict, road_mask: np.ndarray) -> bool:
    """Check if any pixel within the pedestrian bbox overlaps the road mask."""
    img_h, img_w = road_mask.shape[:2]
    x1 = int(max(0, ped["x1"]))
    y1 = int(max(0, ped["y1"]))
    x2 = int(min(img_w, ped["x2"]))
    y2 = int(min(img_h, ped["y2"]))
    if x2 <= x1 or y2 <= y1:
        return False
    roi = road_mask[y1:y2, x1:x2]
    return bool(np.any(roi > 0))


# =========================================================================
# Annotation drawing
# =========================================================================

def draw_annotated_frame(
    frame: np.ndarray,
    road_mask: np.ndarray,
    pedestrians: List[Dict],
    brake: bool,
    reasons: List[str],
) -> np.ndarray:
    """Draw road mask, pedestrian boxes, and brake/keep-driving banner."""
    out = frame.copy()
    img_h, img_w = out.shape[:2]

    # --- Road mask overlay (green, semi-transparent) ---
    if road_mask is not None and np.any(road_mask > 0):
        green_overlay = out.copy()
        green_overlay[road_mask > 0] = (0, 200, 0)
        cv2.addWeighted(green_overlay, 0.35, out, 0.65, 0, out)

    # --- Pedestrian bounding boxes ---
    for ped in pedestrians:
        on_road = ped.get("on_road", False)
        color = (0, 0, 255) if on_road else (0, 255, 0)  # red if on road, green if off
        x1, y1, x2, y2 = int(ped["x1"]), int(ped["y1"]), int(ped["x2"]), int(ped["y2"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ped {ped['conf']:.2f}" + (" ON ROAD" if on_road else "")
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    _FONT, 0.5, color, 1, cv2.LINE_AA)

    # --- Banner ---
    main_text = "BRAKE" if brake else "KEEP DRIVING"
    main_color = (0, 0, 255) if brake else (0, 200, 0)
    font_scale = 2.0
    thickness = 5

    (tw, th), baseline = cv2.getTextSize(main_text, _FONT, font_scale, thickness)
    banner_h = th + baseline + 40
    banner_w = tw + 60

    bx1 = (img_w - banner_w) // 2
    bx2 = bx1 + banner_w
    by1 = 0
    by2 = banner_h

    overlay = out.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    text_x = (img_w - tw) // 2
    text_y = 20 + th
    cv2.putText(out, main_text, (text_x, text_y),
                _FONT, font_scale, main_color, thickness, cv2.LINE_AA)

    # Reason lines
    if reasons:
        cursor_y = by2 + 20
        for line in reasons:
            cv2.putText(out, line, (10, cursor_y),
                        _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cursor_y += 20

    return out


# =========================================================================
# Main pipeline
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Pedestrian-on-Road Brake Decision Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default=os.path.expanduser("~/annotated_outputs/brake_ped_road.mp4"))
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0=all)")
    parser.add_argument("--yolo-model", type=str, default=str(YOLO11_MODEL_PATH))
    parser.add_argument("--yolopv2-weights", type=str, default=str(YOLOPV2_WEIGHTS))
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=PERSON_CONF_MIN)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # --- Init models ---
    yolo_model = init_qnn_yolo(Path(args.yolo_model))
    cpu_device = torch.device("cpu")
    yolopv2_model = init_yolopv2(Path(args.yolopv2_weights), cpu_device)

    # Warmup YOLOPv2
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 384, 640).to(cpu_device)
        yolopv2_model(dummy)
    print("YOLOPv2 warmup done.\n")

    # --- Open video ---
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames if args.max_frames > 0 else total_frames

    vid_writer = None
    frame_idx = 0
    brake_count = 0

    print(f"Source: {args.source}")
    print(f"Processing up to {max_frames} frames at {fps} fps\n")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # --- YOLO11 pedestrian detection (NPU) ---
        yolo_input, orig_size = letterbox_frame(frame, YOLO_INPUT_SIZE)
        yolo_input_batch = np.expand_dims(yolo_input, axis=0)
        boxes_4n, scores_80n = run_yolo_inference(yolo_model, yolo_input_batch)
        pedestrians = postprocess_yolo(boxes_4n, scores_80n, orig_size,
                                       args.conf_thres, NMS_IOU_THRESH)

        # --- YOLOPv2 road segmentation (CPU) ---
        road_mask = yolopv2_road_mask(yolopv2_model, frame, cpu_device, args.img_size)

        # --- Brake decision ---
        brake = False
        reasons = []
        for ped in pedestrians:
            on_road = ped_intersects_road(ped, road_mask)
            ped["on_road"] = on_road
            if on_road:
                brake = True
                reasons.append(f"pedestrian in road region (conf={ped['conf']:.2f})")

        if not brake:
            reasons = ["no pedestrian on road"]

        if brake:
            brake_count += 1

        # --- Annotate ---
        annotated = draw_annotated_frame(frame, road_mask, pedestrians, brake, reasons)

        # --- Write video ---
        if vid_writer is None:
            h, w = annotated.shape[:2]
            vid_writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        vid_writer.write(annotated)
        frame_idx += 1

        t1 = time.perf_counter()
        cmd = "BRAKE" if brake else "KEEP DRIVING"
        print(f"  Frame {frame_idx:3d}/{max_frames}  {cmd:14s}  peds={len(pedestrians)}  "
              f"time={((t1-t0)*1000):.0f}ms  {reasons[0] if reasons else ''}")

    cap.release()
    if vid_writer:
        vid_writer.release()

    print(f"\nDone! {frame_idx} frames processed.")
    print(f"Brake issued on {brake_count}/{frame_idx} frames")
    print(f"Saved (mp4v): {args.output}")

    # Re-encode to H.264 for browser compatibility
    h264_output = args.output.replace(".mp4", "_h264.mp4")
    ret = os.system(f'ffmpeg -y -i "{args.output}" -c:v libx264 -preset medium '
                    f'-crf 18 -pix_fmt yuv420p -movflags +faststart "{h264_output}" 2>/dev/null')
    if ret == 0:
        os.replace(h264_output, args.output)
        print(f"Re-encoded to H.264: {args.output}")
    else:
        print(f"Warning: H.264 re-encode failed, keeping mp4v version")


if __name__ == "__main__":
    main()
