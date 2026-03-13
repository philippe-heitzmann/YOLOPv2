#!/usr/bin/env python3
"""
Visualize YOLOPv2 QNN CPU inference outputs.

Overlays drivable area segmentation and lane line segmentation on the
original test images and saves annotated PNGs.

Note: QNN outputs are in NHWC layout (the QNN runtime converts NCHW->NHWC
internally during model compilation).
"""

import os
import numpy as np
from PIL import Image

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_DIR, "qnn_cpu_results")
INFERENCE_DIR = os.path.join(RESULTS_DIR, "inference_output")
INPUT_LIST = os.path.join(RESULTS_DIR, "raw_inputs", "input_list.txt")
TEST_IMG_DIR = os.path.expanduser(
    "~/datasets/eurocity/ECP/day/img/val/amsterdam"
)
OUTPUT_DIR = os.path.join(RESULTS_DIR, "annotated")

INPUT_H, INPUT_W = 384, 640


def create_overlay(orig_img, drivable_nhwc, lane_nhwc):
    """Create annotated image with drivable area (green) and lane lines (red)."""
    img_resized = orig_img.resize((INPUT_W, INPUT_H), Image.BILINEAR)
    overlay = np.array(img_resized, dtype=np.float32)

    # QNN outputs are NHWC:
    #   drivable: [1, 384, 640, 2] -> argmax over last dim, class 1 = drivable
    #   lane:     [1, 384, 640, 1] -> threshold sigmoid
    drivable = drivable_nhwc[0].argmax(axis=-1)  # [384, 640]
    lane = (lane_nhwc[0, :, :, 0] > 0.5).astype(np.float32)  # [384, 640]

    # Green overlay for drivable area
    alpha = 0.35
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255.0
    drivable_3d = np.stack([drivable] * 3, axis=-1)
    overlay = overlay * (1 - alpha * drivable_3d) + green * (alpha * drivable_3d)

    # Red overlay for lane lines
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255.0
    lane_3d = np.stack([lane] * 3, axis=-1)
    lane_alpha = 0.7
    overlay = overlay * (1 - lane_alpha * lane_3d) + red * (lane_alpha * lane_3d)

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_LIST) as f:
        input_files = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(input_files)} images...\n")

    for idx, raw_path in enumerate(input_files):
        basename = os.path.splitext(os.path.basename(raw_path))[0]
        result_dir = os.path.join(INFERENCE_DIR, f"Result_{idx}")

        if not os.path.isdir(result_dir):
            print(f"  [{idx}] {basename}: Result dir missing, skipping")
            continue

        # Load outputs in NHWC layout
        drivable_raw = np.fromfile(
            os.path.join(result_dir, "_677.raw"), dtype=np.float32
        )
        drivable = drivable_raw.reshape(1, INPUT_H, INPUT_W, 2)

        lane_raw = np.fromfile(
            os.path.join(result_dir, "_759.raw"), dtype=np.float32
        )
        lane = lane_raw.reshape(1, INPUT_H, INPUT_W, 1)

        # Find original image
        orig_path = os.path.join(TEST_IMG_DIR, f"{basename}.png")
        if not os.path.exists(orig_path):
            print(f"  [{idx}] {basename}: Original not found, reconstructing from raw")
            raw_data = np.fromfile(raw_path, dtype=np.float32).reshape(INPUT_H, INPUT_W, 3)
            orig_img = Image.fromarray((raw_data * 255).astype(np.uint8))
        else:
            orig_img = Image.open(orig_path).convert("RGB")

        annotated = create_overlay(orig_img, drivable, lane)
        out_path = os.path.join(OUTPUT_DIR, f"{basename}_annotated.png")
        annotated.save(out_path)

        drivable_pct = (drivable[0].argmax(axis=-1).sum() / (INPUT_H * INPUT_W)) * 100
        lane_pct = ((lane[0, :, :, 0] > 0.5).sum() / (INPUT_H * INPUT_W)) * 100
        print(f"  [{idx}] {basename}: drivable={drivable_pct:.1f}%, lanes={lane_pct:.1f}% -> {out_path}")

    print(f"\nAnnotated images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
