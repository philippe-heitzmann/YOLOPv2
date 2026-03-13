#!/usr/bin/env python3
"""
Run YOLOPv2 QNN model on CPU using qnn-net-run, then visualize results.

Picks 5 random Amsterdam test images, preprocesses them to 384x640 raw
format, runs inference via qnn-net-run with the QNN CPU backend, and
saves output visualizations.
"""

import os
import sys
import glob
import random
import subprocess
import shutil
import numpy as np
from PIL import Image

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QNN_BUILD_DIR = os.path.join(REPO_DIR, "qnn_build")
QNN_SDK_ROOT = "/opt/qcom/aistack/qairt/2.41.0.251128"

TEST_IMG_DIR = os.path.expanduser(
    "~/datasets/eurocity/ECP/day/img/val/amsterdam"
)
OUTPUT_DIR = os.path.join(REPO_DIR, "qnn_cpu_results")

INPUT_H, INPUT_W = 384, 640
NUM_IMAGES = 5
SEED = 42


def preprocess_image(img_path, raw_path):
    """Resize image to 384x640, normalize to float32 HWC (NHWC), save as .raw."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((INPUT_W, INPUT_H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Keep HWC layout — QNN expects NHWC
    arr.tofile(raw_path)
    return img


def run_qnn_net_run(input_list_path, output_dir):
    """Run qnn-net-run with CPU backend."""
    model_lib = os.path.join(QNN_BUILD_DIR, "libyolopv2.so")
    qnn_net_run = os.path.join(
        QNN_SDK_ROOT, "bin", "x86_64-linux-clang", "qnn-net-run"
    )
    qnn_cpu_backend = os.path.join(
        QNN_SDK_ROOT, "lib", "x86_64-linux-clang", "libQnnCpu.so"
    )

    if not os.path.exists(model_lib):
        print(f"ERROR: Model library not found: {model_lib}")
        print("Run scripts/yolopv2_to_qnn.sh first.")
        sys.exit(1)

    cmd = [
        qnn_net_run,
        "--model", model_lib,
        "--backend", qnn_cpu_backend,
        "--input_list", input_list_path,
        "--output_dir", output_dir,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"qnn-net-run failed (exit {result.returncode}):")
        print(result.stderr)
        sys.exit(1)


def main():
    random.seed(SEED)

    # Find test images
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
    if len(test_images) == 0:
        print(f"No images found in {TEST_IMG_DIR}")
        sys.exit(1)

    selected = random.sample(test_images, min(NUM_IMAGES, len(test_images)))
    print(f"Selected {len(selected)} test images:")
    for p in selected:
        print(f"  {os.path.basename(p)}")

    # Prepare directories
    raw_dir = os.path.join(OUTPUT_DIR, "raw_inputs")
    inference_dir = os.path.join(OUTPUT_DIR, "inference_output")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(inference_dir, exist_ok=True)

    # Preprocess
    raw_paths = []
    original_images = {}
    for img_path in selected:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        raw_path = os.path.join(raw_dir, f"{basename}.raw")
        orig_img = preprocess_image(img_path, raw_path)
        raw_paths.append(raw_path)
        original_images[basename] = (img_path, orig_img)

    # Write input list
    input_list_path = os.path.join(raw_dir, "input_list.txt")
    with open(input_list_path, "w") as f:
        for p in raw_paths:
            f.write(p + "\n")

    print(f"\nPreprocessed {len(raw_paths)} images to {raw_dir}")

    # Run inference
    print("\n======== Running QNN CPU inference ========")
    run_qnn_net_run(input_list_path, inference_dir)

    # List outputs
    print("\n======== Inference outputs ========")
    for root, dirs, files in os.walk(inference_dir):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            print(f"  {os.path.relpath(fpath, inference_dir)}  ({size} bytes)")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
