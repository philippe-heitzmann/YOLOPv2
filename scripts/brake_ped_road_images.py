#!/usr/bin/env python3
"""
brake_ped_road_images.py - Pedestrian-on-Road Brake Decision on individual images.

Adapted from brake_ped_road.py to process a list of images instead of video.
Outputs annotated images to a directory.

Usage:
    source ~/npu_setup.sh
    python3 scripts/brake_ped_road_images.py \
        --images img1.png img2.png ... \
        --output-dir ~/annotated_outputs/brake_ped_road_kitti
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

# Ensure YOLOPv2 repo root is on sys.path for utils imports
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Import all shared logic from the original script
# ---------------------------------------------------------------------------
from scripts.brake_ped_road import (
    PERSON_CONF_MIN,
    NMS_IOU_THRESH,
    YOLO_INPUT_SIZE,
    YOLOPV2_WEIGHTS,
    draw_annotated_frame,
    init_qnn_yolo,
    init_yolopv2,
    letterbox_frame,
    ped_intersects_road,
    postprocess_yolo,
    run_yolo_inference,
    yolopv2_road_mask,
)

YOLO11_MODEL_PATH = Path(os.path.expanduser("~/models/libyolo11l.so"))


def main():
    parser = argparse.ArgumentParser(description="Brake decision on individual images")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for annotated images")
    parser.add_argument("--yolo-model", type=str, default=str(YOLO11_MODEL_PATH))
    parser.add_argument("--yolopv2-weights", type=str, default=str(YOLOPV2_WEIGHTS))
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=PERSON_CONF_MIN)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Init models ---
    yolo_model = init_qnn_yolo(Path(args.yolo_model))
    cpu_device = torch.device("cpu")
    yolopv2_model = init_yolopv2(Path(args.yolopv2_weights), cpu_device)

    # Warmup YOLOPv2
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 384, 640).to(cpu_device)
        yolopv2_model(dummy)
    print("YOLOPv2 warmup done.\n")

    for img_path in args.images:
        img_name = Path(img_path).name
        print(f"Processing {img_name}...")

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  ERROR: Cannot read {img_path}, skipping")
            continue

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

        # --- Annotate & save ---
        annotated = draw_annotated_frame(frame, road_mask, pedestrians, brake, reasons)
        out_path = os.path.join(args.output_dir, f"annotated_{img_name}")
        cv2.imwrite(out_path, annotated)

        t1 = time.perf_counter()
        cmd = "BRAKE" if brake else "KEEP DRIVING"
        print(f"  {cmd} | peds={len(pedestrians)} | {(t1-t0)*1000:.0f}ms | saved: {out_path}")

    print(f"\nDone! {len(args.images)} images processed.")
    print(f"Annotated images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
