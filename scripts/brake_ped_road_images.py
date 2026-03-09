#!/usr/bin/env python3
"""
brake_ped_road_images.py - Full NPU Brake Decision Pipeline on individual images.

Combines two QNN/HTP models (both .so on NPU):
    1. YOLOv11-Large — pedestrian detection
    2. YOLOPv2 — drivable area segmentation + lane line detection

Logic: If any pedestrian bounding box overlaps the YOLOPv2 road mask → BRAKE.

Each output frame is annotated with:
    - Green semi-transparent road mask overlay
    - Yellow lane line overlay
    - Pedestrian bounding boxes (red if on road, green if off road)
    - Large centered text banner: "BRAKE" (red) or "KEEP DRIVING" (green)

Usage:
    source ~/npu_setup.sh
    python3 scripts/brake_ped_road_images.py \
        --images img1.png img2.png ... \
        --output-dir ~/annotated_outputs/brake_ped_road_npu
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YOLO_INPUT_SIZE = 640
PERSON_CLASS_ID = 0
PERSON_CONF_MIN = 0.25
NMS_IOU_THRESH = 0.45

YOLOPV2_INPUT_H = 384
YOLOPV2_INPUT_W = 640

DEFAULT_YOLO11_MODEL = Path(os.path.expanduser("~/models/libyolo11l.so"))
DEFAULT_YOLOPV2_MODEL = Path(os.path.expanduser("~/models/yolopv2_qnn/libyolopv2.so"))

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# =========================================================================
# QNN initialization (shared config, two model contexts)
# =========================================================================

_qnn_configured = False


def _ensure_qnn_config():
    """Configure QNN runtime once for all models."""
    global _qnn_configured
    if _qnn_configured:
        return
    from qai_appbuilder import LogLevel, PerfProfile, ProfilingLevel, QNNConfig, Runtime

    sdk_root = Path(os.environ.get(
        "QNN_SDK_ROOT",
        os.environ.get("QAIRT_SDK_ROOT", os.environ.get("SNPE_ROOT", "/opt/qairt")),
    ))
    lib_dir = sdk_root / "lib/aarch64-oe-linux-gcc11.2"
    if not lib_dir.exists():
        print(f"ERROR: QNN SDK lib dir not found: {lib_dir}. Source npu_setup.sh first.")
        sys.exit(1)

    QNNConfig.Config(str(lib_dir), Runtime.HTP, LogLevel.WARN, ProfilingLevel.BASIC)
    _qnn_configured = True


def init_qnn_yolo(model_path: Path):
    """Load YOLOv11 .so on HTP for pedestrian detection."""
    _ensure_qnn_config()
    from qai_appbuilder import PerfProfile, QNNContext

    class QnnYolo(QNNContext):
        def Inference(self, input_data):
            inputs = list(input_data) if isinstance(input_data, (list, tuple)) else [input_data]
            return super().Inference(inputs)

    model = QnnYolo("yolo11_brake", str(model_path))
    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
    print(f"YOLOv11L loaded on HTP: {model_path.name}", flush=True)
    return model


def init_qnn_yolopv2(model_path: Path):
    """Load YOLOPv2 .so on HTP for road segmentation + lane lines."""
    _ensure_qnn_config()
    from qai_appbuilder import QNNContext

    class QnnYolopv2(QNNContext):
        def Inference(self, input_data):
            inputs = list(input_data) if isinstance(input_data, (list, tuple)) else [input_data]
            return super().Inference(inputs)

    model = QnnYolopv2("yolopv2_brake", str(model_path))
    print(f"YOLOPv2 loaded on HTP: {model_path.name}", flush=True)
    return model


# =========================================================================
# YOLOv11 — pedestrian detection
# =========================================================================

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


def postprocess_yolo(boxes_4n, scores_80n, orig_size, conf_thresh, iou_thresh):
    """Returns list of pedestrian dicts [{x1, y1, x2, y2, conf}, ...]."""
    combined = np.concatenate([boxes_4n, scores_80n], axis=0).T
    box_coords = combined[:, :4]
    class_scores = combined[:, 4:]

    if _is_xyxy(box_coords):
        box_coords = _xyxy_to_cxcywh(box_coords)

    person_scores = class_scores[:, PERSON_CLASS_ID]
    mask = person_scores > conf_thresh
    if not mask.any():
        return []

    filt_boxes = box_coords[mask]
    filt_scores = person_scores[mask]

    keep = _nms(filt_boxes, filt_scores, iou_thresh)
    if not keep:
        return []
    filt_boxes = filt_boxes[keep]
    filt_scores = filt_scores[keep]

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


# =========================================================================
# YOLOPv2 — road segmentation + lane lines (NPU)
# =========================================================================

def preprocess_yolopv2(frame_bgr: np.ndarray):
    """Resize frame to 384x640, convert to NHWC RGB float32 [0,1]."""
    resized = cv2.resize(frame_bgr, (YOLOPV2_INPUT_W, YOLOPV2_INPUT_H),
                         interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(rgb, axis=0).copy()


def run_yolopv2_inference(model, input_nhwc: np.ndarray):
    """Run QNN inference, return (da_seg, ll_seg) tensors."""
    outputs = model.Inference(input_nhwc)
    outs = list(outputs.values()) if isinstance(outputs, dict) else list(outputs)
    tensors = [np.asarray(o, dtype=np.float32).flatten() for o in outs]

    da_seg = None
    ll_seg = None
    da_size = YOLOPV2_INPUT_H * YOLOPV2_INPUT_W * 2
    ll_size = YOLOPV2_INPUT_H * YOLOPV2_INPUT_W * 1

    for t in tensors:
        if t.size == da_size:
            da_seg = t.reshape(1, YOLOPV2_INPUT_H, YOLOPV2_INPUT_W, 2)
        elif t.size == ll_size:
            ll_seg = t.reshape(1, YOLOPV2_INPUT_H, YOLOPV2_INPUT_W, 1)

    return da_seg, ll_seg


def postprocess_masks(da_seg, ll_seg, orig_h, orig_w, morph_close_kernel: int = 25):
    """Convert raw QNN outputs to binary road + lane masks at original resolution."""
    if da_seg is not None:
        da_mask = np.argmax(da_seg[0], axis=-1).astype(np.uint8)
        road_mask = cv2.resize(da_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        road_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if morph_close_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (morph_close_kernel, morph_close_kernel))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

    if ll_seg is not None:
        ll_prob = ll_seg[0, :, :, 0]
        ll_mask = (ll_prob > 0.5).astype(np.uint8)
        lane_mask = cv2.resize(ll_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        lane_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    return road_mask, lane_mask


# =========================================================================
# Brake decision
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
    lane_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw road mask, lane lines, pedestrian boxes, and brake/keep-driving banner."""
    out = frame.copy()
    img_h, img_w = out.shape[:2]

    # --- Road mask overlay (green, semi-transparent) ---
    if road_mask is not None and np.any(road_mask > 0):
        green_overlay = out.copy()
        green_overlay[road_mask > 0] = (0, 200, 0)
        cv2.addWeighted(green_overlay, 0.35, out, 0.65, 0, out)

    # --- Lane line overlay (yellow, semi-transparent) ---
    if lane_mask is not None and np.any(lane_mask > 0):
        lane_overlay = out.copy()
        lane_overlay[lane_mask > 0] = (0, 255, 255)
        cv2.addWeighted(lane_overlay, 0.6, out, 0.4, 0, out)

    # --- Pedestrian bounding boxes ---
    for ped in pedestrians:
        on_road = ped.get("on_road", False)
        color = (0, 0, 255) if on_road else (0, 255, 0)
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
    parser = argparse.ArgumentParser(
        description="Full NPU Brake Decision: YOLOv11L (ped detection) + YOLOPv2 (road seg)")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--yolo-model", type=str, default=str(DEFAULT_YOLO11_MODEL))
    parser.add_argument("--yolopv2-model", type=str, default=str(DEFAULT_YOLOPV2_MODEL))
    parser.add_argument("--conf-thres", type=float, default=PERSON_CONF_MIN)
    parser.add_argument("--morph-close-kernel", type=int, default=25,
                        help="Kernel size for morphological closing on road mask (0=disable)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Init both models on NPU ---
    yolo_model = init_qnn_yolo(Path(args.yolo_model))
    yolopv2_model = init_qnn_yolopv2(Path(args.yolopv2_model))
    print("Both models loaded on HTP.\n", flush=True)

    for img_path in args.images:
        img_name = Path(img_path).name
        print(f"Processing {img_name}...", flush=True)

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  ERROR: Cannot read {img_path}, skipping")
            continue

        orig_h, orig_w = frame.shape[:2]
        t0 = time.perf_counter()

        # --- YOLOv11 pedestrian detection (NPU) ---
        yolo_input, orig_size = letterbox_frame(frame, YOLO_INPUT_SIZE)
        yolo_input_batch = np.expand_dims(yolo_input, axis=0)
        boxes_4n, scores_80n = run_yolo_inference(yolo_model, yolo_input_batch)
        pedestrians = postprocess_yolo(boxes_4n, scores_80n, orig_size,
                                       args.conf_thres, NMS_IOU_THRESH)

        # --- YOLOPv2 road segmentation + lane lines (NPU) ---
        yolopv2_input = preprocess_yolopv2(frame)
        da_seg, ll_seg = run_yolopv2_inference(yolopv2_model, yolopv2_input)
        road_mask, lane_mask = postprocess_masks(da_seg, ll_seg, orig_h, orig_w,
                                                 morph_close_kernel=args.morph_close_kernel)

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

        # --- Annotate & save ---
        annotated = draw_annotated_frame(frame, road_mask, pedestrians, brake, reasons,
                                         lane_mask=lane_mask)
        out_path = os.path.join(args.output_dir, f"annotated_{img_name}")
        cv2.imwrite(out_path, annotated)

        t1 = time.perf_counter()
        cmd = "BRAKE" if brake else "KEEP DRIVING"
        print(f"  {cmd} | peds={len(pedestrians)} | {(t1-t0)*1000:.0f}ms | saved: {out_path}",
              flush=True)

    print(f"\nDone! {len(args.images)} images processed.", flush=True)
    print(f"Annotated images saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
