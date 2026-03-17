#!/usr/bin/env python3
"""
brake_ped_road_images.py - Brake Decision Pipeline with Depth Estimation.

Combines three models:
    1. YOLO v8 (KITTI-trained) — pedestrian/object detection
    2. ZoeDepth (Depth-Anything small ViT) — metric depth estimation (meters)
    3. YOLOPv2 (QNN/HTP on NPU) — drivable area segmentation + lane line detection

Logic:
    If any pedestrian is detected at < 30 meters AND its bounding box overlaps
    the YOLOPv2 road segmentation mask → BRAKE.  Otherwise → KEEP DRIVING.

Each output frame is annotated with:
    - Green semi-transparent road mask overlay
    - Yellow lane line overlay
    - Cyan extended-road-fill overlay
    - Pedestrian bounding boxes with distance labels
      (red if on road AND < 30m, orange if < 30m but off road, green otherwise)
    - Large centered text banner: "BRAKE" (red) or "KEEP DRIVING" (green)

Usage:
    source ~/npu_setup.sh
    python3 scripts/brake_ped_road_images.py \
        --images img1.png img2.png ... \
        --output-dir ~/annotated_outputs/brake_depth_ped_road
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
BRAKE_DISTANCE_THRESHOLD = 30.0  # meters

YOLOPV2_INPUT_H = 384
YOLOPV2_INPUT_W = 640

ZOEDEPTH_REPO = Path(os.path.expanduser("~/zoedepth-depth-estimation"))
YOLO_MODEL_PATH = ZOEDEPTH_REPO / "checkpoints" / "yolo_best.pt"
DEPTH_MODEL_PATH = f"local::{ZOEDEPTH_REPO / 'checkpoints' / 'zoedepth-depthanything-smallvit-10epochs_best.pt'}"
DEPTH_VIT_TYPE = "small"
DEPTH_STRATEGY = "bbox_median"

DEFAULT_YOLOPV2_MODEL = Path(os.path.expanduser("~/models/yolopv2_qnn/libyolopv2.so"))

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# =========================================================================
# QNN initialization (YOLOPv2 on NPU)
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
# YOLO v8 — pedestrian detection (PyTorch / ultralytics)
# =========================================================================

def init_yolo_detection(model_path: Path):
    """Load YOLO v8 detection model (KITTI-trained)."""
    sys.path.insert(0, str(ZOEDEPTH_REPO))
    from ultralytics import YOLO
    model = YOLO(model=str(model_path), task="detect")
    print(f"YOLO detection model loaded: {model_path.name}", flush=True)
    return model


def run_yolo_detection(model, frame_bgr: np.ndarray) -> List[Dict]:
    """Run YOLO detection, return list of pedestrian dicts with xyxy coords."""
    from PIL import Image
    # KITTI class names from the zoedepth repo
    KITTI_CLASS_NAMES = [
        "Car", "Pedestrian", "Van", "Cyclist", "Truck",
        "Misc", "Tram", "Person_sitting", "DontCare",
    ]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    results = model(pil_img, verbose=False)
    img_result = results[0].cpu()
    boxes = img_result.boxes

    detections = []
    for i, cls_idx in enumerate(boxes.cls.int()):
        idx = cls_idx.item()
        if idx >= len(KITTI_CLASS_NAMES):
            continue
        class_name = KITTI_CLASS_NAMES[idx]
        xyxy = boxes.xyxy[i].tolist()
        conf = float(boxes.conf[i])
        detections.append({
            "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
            "class_name": class_name,
            "conf": conf,
            "xyxy": xyxy,
        })
    return detections


# =========================================================================
# ZoeDepth — metric depth estimation (PyTorch)
# =========================================================================

def init_depth_model(pretrained_resource: str, vit_type: str):
    """Load ZoeDepth metric depth model.

    The internal DINOv2 backbone uses torch.hub.load with a relative path,
    so we temporarily chdir into the zoedepth repo root during model init.
    """
    sys.path.insert(0, str(ZOEDEPTH_REPO))
    from distance_estimation.depth_prediction.predict_depth_metric import load_depth_model

    prev_cwd = os.getcwd()
    os.chdir(str(ZOEDEPTH_REPO))
    try:
        model = load_depth_model(pretrained_resource=pretrained_resource,
                                 vit_encoder_type=vit_type)
    finally:
        os.chdir(prev_cwd)
    print(f"ZoeDepth metric depth model loaded (ViT-{vit_type})", flush=True)
    return model


def run_depth_estimation(model, frame_bgr: np.ndarray) -> np.ndarray:
    """Run depth estimation, return depth map in meters at original resolution."""
    from PIL import Image
    sys.path.insert(0, str(ZOEDEPTH_REPO))
    from distance_estimation.depth_prediction.predict_depth_metric import process_image

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    depth_map = process_image(model=model, image=pil_img)
    return depth_map  # shape (H, W), values in meters


def get_pedestrian_distance(depth_map: np.ndarray, xyxy: List[float],
                            strategy: str = "bbox_median") -> float:
    """Extract distance in meters for a detection bounding box."""
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = depth_map.shape[:2]
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return float("inf")

    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return float("inf")

    region_type, method = strategy.split("_")
    if region_type == "center":
        ch = region.shape[0]
        cw = region.shape[1]
        region = region[ch // 4: ch * 3 // 4, cw // 4: cw * 3 // 4]
        if region.size == 0:
            return float("inf")

    if method == "mean":
        return float(np.mean(region))
    elif method == "median":
        return float(np.median(region))
    elif method == "min":
        return float(np.min(region))
    elif method == "percentile":
        return float(np.percentile(region, 25))
    return float(np.median(region))


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


def postprocess_masks(da_seg, ll_seg, orig_h, orig_w, morph_close_kernel: int = 30):
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


def extend_road_mask_between_lanes(road_mask: np.ndarray, lane_mask: np.ndarray):
    """Extend road mask using adjacent lane lines to fill occlusion gaps."""
    road_ext = road_mask.copy()
    lane_ext = lane_mask.copy() if lane_mask is not None else None

    if lane_mask is None or not np.any(lane_mask):
        return road_ext, lane_ext

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        lane_mask, connectivity=8
    )

    MIN_AREA = 500
    segments = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue
        x_left = stats[i, cv2.CC_STAT_LEFT]
        x_right = x_left + stats[i, cv2.CC_STAT_WIDTH] - 1
        y_top = stats[i, cv2.CC_STAT_TOP]
        y_bottom = y_top + stats[i, cv2.CC_STAT_HEIGHT] - 1
        seg_pixels = np.where(labels == i)
        bottom_quarter_y = y_bottom - max(1, (y_bottom - y_top) // 4)
        bottom_mask = seg_pixels[0] >= bottom_quarter_y
        median_x_bottom = int(np.median(seg_pixels[1][bottom_mask])) if np.any(bottom_mask) else int(centroids[i][0])
        bottom_row_mask = seg_pixels[0] == y_bottom
        if np.any(bottom_row_mask):
            bottom_xs = seg_pixels[1][bottom_row_mask]
            bottom_width = int(bottom_xs.max() - bottom_xs.min()) + 1
        else:
            bottom_width = stats[i, cv2.CC_STAT_WIDTH]
        segments.append({
            "label": i, "area": area,
            "x_center": centroids[i][0],
            "x_left": x_left, "x_right": x_right,
            "y_top": y_top, "y_bottom": y_bottom,
            "height": y_bottom - y_top + 1,
            "median_x_bottom": median_x_bottom,
            "bottom_width": bottom_width,
        })

    if len(segments) < 2:
        return road_ext, lane_ext

    segments.sort(key=lambda s: s["x_center"])
    MIN_HEIGHT_DIFF_RATIO = 0.15

    for i in range(len(segments) - 1):
        seg_l = segments[i]
        seg_r = segments[i + 1]

        between = False
        for j in range(len(segments)):
            if j == i or j == i + 1:
                continue
            if seg_l["x_center"] < segments[j]["x_center"] < seg_r["x_center"]:
                between = True
                break
        if between:
            continue

        h_l = seg_l["height"]
        h_r = seg_r["height"]
        max_h = max(h_l, h_r)
        diff = abs(h_l - h_r)
        if diff < max_h * MIN_HEIGHT_DIFF_RATIO:
            continue

        target_bottom = max(seg_l["y_bottom"], seg_r["y_bottom"])
        fill_top = min(seg_l["y_top"], seg_r["y_top"])
        fill_x_left = seg_l["x_right"]
        fill_x_right = seg_r["x_left"]
        if fill_x_right <= fill_x_left:
            continue

        road_ext[fill_top:target_bottom + 1, fill_x_left:fill_x_right + 1] = 1

        if lane_ext is not None:
            shorter = seg_l if h_l < h_r else seg_r
            shorter_bottom = shorter["y_bottom"]
            if shorter_bottom < target_bottom:
                cx = shorter["median_x_bottom"]
                half_w = max(2, shorter["bottom_width"] // 2)
                x1 = max(0, cx - half_w)
                x2 = min(lane_ext.shape[1] - 1, cx + half_w)
                lane_ext[shorter_bottom:target_bottom + 1, x1:x2 + 1] = 1

    return road_ext, lane_ext


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
    detections: List[Dict],
    brake: bool,
    reasons: List[str],
    lane_mask: Optional[np.ndarray] = None,
    road_mask_extended: Optional[np.ndarray] = None,
    lane_mask_extended: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw road mask, lane lines, detection boxes with distance, and brake banner."""
    out = frame.copy()
    img_h, img_w = out.shape[:2]

    # --- Road mask overlay (green, semi-transparent) ---
    if road_mask is not None and np.any(road_mask > 0):
        green_overlay = out.copy()
        green_overlay[road_mask > 0] = (0, 200, 0)
        cv2.addWeighted(green_overlay, 0.35, out, 0.65, 0, out)

    # --- Extended road fill overlay (cyan) ---
    if road_mask_extended is not None and road_mask is not None:
        ext_only = (road_mask_extended > 0) & (road_mask == 0)
        if np.any(ext_only):
            cyan_overlay = out.copy()
            cyan_overlay[ext_only] = (200, 200, 0)
            cv2.addWeighted(cyan_overlay, 0.45, out, 0.55, 0, out)

    # --- Lane line overlay (yellow) ---
    if lane_mask is not None and np.any(lane_mask > 0):
        lane_overlay = out.copy()
        lane_overlay[lane_mask > 0] = (0, 255, 255)
        cv2.addWeighted(lane_overlay, 0.6, out, 0.4, 0, out)

    # --- Extended lane line overlay (magenta) ---
    if lane_mask_extended is not None and lane_mask is not None:
        lane_ext_only = (lane_mask_extended > 0) & (lane_mask == 0)
        if np.any(lane_ext_only):
            magenta_overlay = out.copy()
            magenta_overlay[lane_ext_only] = (255, 0, 255)
            cv2.addWeighted(magenta_overlay, 0.7, out, 0.3, 0, out)

    # --- Detection bounding boxes ---
    for det in detections:
        class_name = det.get("class_name", "")
        is_ped = class_name in ("Pedestrian", "Cyclist", "Person_sitting")
        on_road = det.get("on_road", False)
        distance = det.get("distance", float("inf"))
        in_range = distance < BRAKE_DISTANCE_THRESHOLD

        if is_ped and on_road and in_range:
            color = (0, 0, 255)      # Red — brake trigger
            thickness = 3
        elif is_ped and in_range:
            color = (0, 165, 255)    # Orange — close but off road
            thickness = 2
        elif is_ped:
            color = (0, 255, 0)      # Green — far pedestrian
            thickness = 2
        else:
            color = (255, 180, 0)    # Blue — non-pedestrian
            thickness = 1

        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        dist_str = f"{distance:.1f}m" if distance < 200 else "?"
        label = f"{class_name} {det['conf']:.2f} {dist_str}"
        if is_ped and on_road and in_range:
            label += " ON ROAD"

        # Label background
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
        cv2.rectangle(out, (x1, max(y1 - th - 8, 0)), (x1 + tw + 4, max(y1 - 2, 0)), color, -1)
        cv2.putText(out, label, (x1 + 2, max(y1 - 4, 12)),
                    _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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
        description="Brake Decision: YOLO (detection) + ZoeDepth (distance) + YOLOPv2 (road seg)")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--yolopv2-model", type=str, default=str(DEFAULT_YOLOPV2_MODEL))
    parser.add_argument("--yolo-model", type=str, default=str(YOLO_MODEL_PATH))
    parser.add_argument("--depth-model", type=str, default=DEPTH_MODEL_PATH)
    parser.add_argument("--depth-vit-type", type=str, default=DEPTH_VIT_TYPE,
                        choices=["small", "large"])
    parser.add_argument("--depth-strategy", type=str, default=DEPTH_STRATEGY,
                        choices=["bbox_mean", "bbox_median", "bbox_min", "bbox_percentile",
                                 "center_mean", "center_median", "center_min", "center_percentile"])
    parser.add_argument("--brake-distance", type=float, default=BRAKE_DISTANCE_THRESHOLD,
                        help="Brake if pedestrian closer than this (meters)")
    parser.add_argument("--morph-close-kernel", type=int, default=30,
                        help="Kernel size for morphological closing on road mask (0=disable)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    brake_dist = args.brake_distance

    # --- Init models ---
    yolo_model = init_yolo_detection(Path(args.yolo_model))
    depth_model = init_depth_model(args.depth_model, args.depth_vit_type)
    yolopv2_model = init_qnn_yolopv2(Path(args.yolopv2_model))
    print("All three models loaded.\n", flush=True)

    for img_path in args.images:
        img_name = Path(img_path).name
        print(f"Processing {img_name}...", flush=True)

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  ERROR: Cannot read {img_path}, skipping")
            continue

        orig_h, orig_w = frame.shape[:2]
        t0 = time.perf_counter()

        # --- YOLO detection (all classes) ---
        detections = run_yolo_detection(yolo_model, frame)

        # --- ZoeDepth metric depth estimation ---
        depth_map = run_depth_estimation(depth_model, frame)

        # Assign distance to each detection
        for det in detections:
            det["distance"] = get_pedestrian_distance(
                depth_map, det["xyxy"], strategy=args.depth_strategy)

        # --- YOLOPv2 road segmentation + lane lines (NPU) ---
        yolopv2_input = preprocess_yolopv2(frame)
        da_seg, ll_seg = run_yolopv2_inference(yolopv2_model, yolopv2_input)
        road_mask, lane_mask = postprocess_masks(da_seg, ll_seg, orig_h, orig_w,
                                                 morph_close_kernel=args.morph_close_kernel)

        # --- Extend road mask using lane lines ---
        road_mask_extended, lane_mask_extended = extend_road_mask_between_lanes(road_mask, lane_mask)

        # --- Brake decision ---
        # Only pedestrians/cyclists/person_sitting trigger braking
        brake = False
        reasons = []
        ped_classes = {"Pedestrian", "Cyclist", "Person_sitting"}

        for det in detections:
            on_road = ped_intersects_road(det, road_mask_extended)
            det["on_road"] = on_road

            if det["class_name"] in ped_classes:
                dist = det["distance"]
                if on_road and dist < brake_dist:
                    brake = True
                    reasons.append(
                        f"{det['class_name'].lower()} at {dist:.1f}m on road "
                        f"(conf={det['conf']:.2f})")

        if not brake:
            reasons = [f"no pedestrian on road within {brake_dist:.0f}m"]

        # --- Annotate & save ---
        annotated = draw_annotated_frame(frame, road_mask, detections, brake, reasons,
                                         lane_mask=lane_mask,
                                         road_mask_extended=road_mask_extended,
                                         lane_mask_extended=lane_mask_extended)
        out_path = os.path.join(args.output_dir, f"annotated_{img_name}")
        cv2.imwrite(out_path, annotated)

        t1 = time.perf_counter()
        cmd = "BRAKE" if brake else "KEEP DRIVING"
        n_peds = sum(1 for d in detections if d["class_name"] in ped_classes)
        n_on_road = sum(1 for d in detections
                        if d["class_name"] in ped_classes
                        and d.get("on_road") and d["distance"] < brake_dist)
        print(f"  {cmd} | total_det={len(detections)} peds={n_peds} "
              f"peds_on_road_<{brake_dist:.0f}m={n_on_road} | "
              f"{(t1-t0)*1000:.0f}ms | saved: {out_path}",
              flush=True)

    print(f"\nDone! {len(args.images)} images processed.", flush=True)
    print(f"Annotated images saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
