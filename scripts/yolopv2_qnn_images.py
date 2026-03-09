#!/usr/bin/env python3
"""
yolopv2_qnn_images.py - Run YOLOPv2 on QNN/HTP (NPU) for road segmentation + lane lines.

Loads the YOLOPv2 .so model compiled for QNN and runs inference on individual
images, producing annotated outputs with road mask and lane line overlays.

The QNN model outputs:
    - 3 detection head tensors (not used here — we use YOLOv11 for detection)
    - Drivable area segmentation: [1, 384, 640, 2] (BHWC, 2-class)
    - Lane line mask: [1, 384, 640, 1] (BHWC, binary)

Usage:
    source ~/npu_setup.sh
    python3 scripts/yolopv2_qnn_images.py \
        --images img1.png img2.png ... \
        --output-dir ~/annotated_outputs/yolopv2_qnn_test \
        --yolopv2-model ~/models/yolopv2_qnn/libyolopv2.so
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YOLOPV2_INPUT_H = 384
YOLOPV2_INPUT_W = 640
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# QNN model initialization (same pattern as fused_pipeline.py)
# ---------------------------------------------------------------------------

def init_qnn_yolopv2(model_path: Path):
    """Initialize YOLOPv2 QNN model on HTP."""
    from qai_appbuilder import (
        LogLevel, PerfProfile, ProfilingLevel, QNNConfig, QNNContext, Runtime,
    )

    class QnnYolopv2(QNNContext):
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
    model = QnnYolopv2("yolopv2_qnn", str(model_path))
    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
    print(f"YOLOPv2 QNN loaded on HTP: {model_path.name}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(frame_bgr: np.ndarray):
    """Resize frame to 384x640, convert to NHWC RGB float32 [0,1]."""
    resized = cv2.resize(frame_bgr, (YOLOPV2_INPUT_W, YOLOPV2_INPUT_H),
                         interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # QNN expects NHWC: [1, 384, 640, 3]
    return np.expand_dims(rgb, axis=0).copy()


# ---------------------------------------------------------------------------
# Inference + postprocessing
# ---------------------------------------------------------------------------

def run_yolopv2_inference(model, input_nhwc: np.ndarray):
    """Run QNN inference, return (da_seg [1,384,640,2], ll_seg [1,384,640,1]).

    The model outputs 5 tensors:
        - 3 detection heads (ignored)
        - drivable area seg [1,384,640,2]
        - lane line seg [1,384,640,1]
    We identify them by shape.
    """
    outputs = model.Inference(input_nhwc)
    outs = list(outputs.values()) if isinstance(outputs, dict) else list(outputs)
    tensors = [np.asarray(o, dtype=np.float32).flatten() for o in outs]

    da_seg = None
    ll_seg = None

    da_size = YOLOPV2_INPUT_H * YOLOPV2_INPUT_W * 2   # 491520
    ll_size = YOLOPV2_INPUT_H * YOLOPV2_INPUT_W * 1   # 245760

    for t in tensors:
        if t.size == da_size:
            da_seg = t.reshape(1, YOLOPV2_INPUT_H, YOLOPV2_INPUT_W, 2)
        elif t.size == ll_size:
            ll_seg = t.reshape(1, YOLOPV2_INPUT_H, YOLOPV2_INPUT_W, 1)

    return da_seg, ll_seg


def postprocess_masks(da_seg, ll_seg, orig_h, orig_w):
    """Convert raw QNN outputs to binary masks at original resolution.

    da_seg: [1, 384, 640, 2] — BHWC, 2-class logits
    ll_seg: [1, 384, 640, 1] — BHWC, lane probability
    """
    # Drivable area: argmax over last dim (class 1 = drivable)
    if da_seg is not None:
        da_mask = np.argmax(da_seg[0], axis=-1).astype(np.uint8)  # [384, 640]
        road_mask = cv2.resize(da_mask, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    else:
        road_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    # Lane line: threshold at 0.5
    if ll_seg is not None:
        ll_prob = ll_seg[0, :, :, 0]  # [384, 640]
        ll_mask = (ll_prob > 0.5).astype(np.uint8)
        lane_mask = cv2.resize(ll_mask, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    else:
        lane_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    return road_mask, lane_mask


# ---------------------------------------------------------------------------
# Annotation drawing
# ---------------------------------------------------------------------------

def draw_annotated_frame(
    frame: np.ndarray,
    road_mask: np.ndarray,
    lane_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw road mask and lane line overlays."""
    out = frame.copy()

    # Road mask overlay (green, semi-transparent)
    if road_mask is not None and np.any(road_mask > 0):
        green_overlay = out.copy()
        green_overlay[road_mask > 0] = (0, 200, 0)
        cv2.addWeighted(green_overlay, 0.35, out, 0.65, 0, out)

    # Lane line overlay (yellow, semi-transparent)
    if lane_mask is not None and np.any(lane_mask > 0):
        lane_overlay = out.copy()
        lane_overlay[lane_mask > 0] = (0, 255, 255)
        cv2.addWeighted(lane_overlay, 0.6, out, 0.4, 0, out)

    # Info text
    road_pct = np.count_nonzero(road_mask) / road_mask.size * 100 if road_mask is not None else 0
    lane_pct = np.count_nonzero(lane_mask) / lane_mask.size * 100 if lane_mask is not None else 0
    info = f"Road: {road_pct:.1f}%  |  Lanes: {lane_pct:.1f}%  |  YOLOPv2 QNN (NPU)"
    cv2.putText(out, info, (10, out.shape[0] - 15),
                _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLOPv2 QNN road segmentation on images")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--yolopv2-model", type=str,
                        default=os.path.expanduser("~/models/yolopv2_qnn/libyolopv2.so"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Init model
    model = init_qnn_yolopv2(Path(args.yolopv2_model))

    for img_path in args.images:
        img_name = Path(img_path).name
        print(f"Processing {img_name}...", flush=True)

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  ERROR: Cannot read {img_path}, skipping")
            continue

        orig_h, orig_w = frame.shape[:2]
        t0 = time.perf_counter()

        # Preprocess
        input_nhwc = preprocess_frame(frame)

        # Inference on NPU
        da_seg, ll_seg = run_yolopv2_inference(model, input_nhwc)

        # Postprocess to masks
        road_mask, lane_mask = postprocess_masks(da_seg, ll_seg, orig_h, orig_w)

        # Annotate
        annotated = draw_annotated_frame(frame, road_mask, lane_mask)

        # Save
        out_path = os.path.join(args.output_dir, f"annotated_{img_name}")
        cv2.imwrite(out_path, annotated)

        t1 = time.perf_counter()
        road_pct = np.count_nonzero(road_mask) / road_mask.size * 100
        print(f"  road={road_pct:.1f}% | {(t1-t0)*1000:.0f}ms | saved: {out_path}", flush=True)

    print(f"\nDone! {len(args.images)} images processed.", flush=True)
    print(f"Annotated images saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
