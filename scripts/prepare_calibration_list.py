#!/usr/bin/env python3
"""
Prepare calibration data for QNN quantization.

Resizes images from calibration_data to 384x640, converts to raw float32
NHWC format, and generates an input_list.txt for qnn-onnx-converter.
"""

import os
import sys
import glob
import numpy as np
from PIL import Image

CALIB_SRC = "/home/hotdog/datasets/calibration_data"
CALIB_DST = "/home/hotdog/YOLOPv2/calibration_raw"
INPUT_H, INPUT_W = 384, 640


def main():
    os.makedirs(CALIB_DST, exist_ok=True)

    images = sorted(glob.glob(os.path.join(CALIB_SRC, "*.png")))
    if not images:
        print(f"No PNG images found in {CALIB_SRC}")
        sys.exit(1)

    raw_paths = []
    for img_path in images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        raw_path = os.path.join(CALIB_DST, f"{basename}.raw")

        img = Image.open(img_path).convert("RGB")
        img = img.resize((INPUT_W, INPUT_H), Image.BILINEAR)

        # Normalize to [0, 1] float32, keep HWC layout (QNN expects NHWC)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr.tofile(raw_path)
        raw_paths.append(raw_path)

    list_path = os.path.join(CALIB_DST, "input_list.txt")
    with open(list_path, "w") as f:
        for p in raw_paths:
            f.write(p + "\n")

    print(f"Prepared {len(raw_paths)} calibration images -> {CALIB_DST}")
    print(f"Input list: {list_path}")


if __name__ == "__main__":
    main()
