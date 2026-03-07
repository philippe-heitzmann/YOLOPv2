import os
import sys
import time
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.utils import (
    time_synchronized, select_device, scale_coords, non_max_suppression,
    split_for_trace_model, driving_area_mask, lane_line_mask,
    plot_one_box, show_seg_result, AverageMeter, LoadImages
)

# Config
SRC_DIR = os.path.expanduser("~/datasets/ecp_amsterdam_test/images/val")
OUT_DIR = os.path.expanduser("~/yolopv2_amsterdam_test_0305")
WEIGHTS = "data/weights/yolopv2.pt"
IMG_SIZE = 640
CONF_THRES = 0.3
IOU_THRES = 0.45
NUM_IMAGES = 30

# Select 30 random images
all_images = sorted([f for f in os.listdir(SRC_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
random.seed(42)
selected = random.sample(all_images, NUM_IMAGES)
print(f"Selected {len(selected)} random images")

# Create temp dir with selected images (symlinks)
tmp_dir = "/tmp/yolopv2_amsterdam_30"
os.makedirs(tmp_dir, exist_ok=True)
for f in selected:
    src = os.path.join(SRC_DIR, f)
    dst = os.path.join(tmp_dir, f)
    if not os.path.exists(dst):
        os.symlink(src, dst)

# Load model on CPU
device = select_device('cpu')
model = torch.jit.load(WEIGHTS, map_location='cpu')
model = model.to(device)
model.eval()

# Dataloader
stride = 32
dataset = LoadImages(tmp_dir, img_size=IMG_SIZE, stride=stride)

os.makedirs(OUT_DIR, exist_ok=True)

inf_times = []
nms_times = []
total_times = []

print(f"\nRunning inference on {NUM_IMAGES} images (CPU)...\n")

with torch.no_grad():
    # Warmup
    model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device))

    for idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        t_start = time.time()

        img_tensor = torch.from_numpy(img).to(device).float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img_tensor)
        t2 = time_synchronized()

        # Post-process
        pred = split_for_trace_model(pred, anchor_grid)

        t3 = time_synchronized()
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Draw results
        for i, det in enumerate(pred):
            im0 = im0s.copy()
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    plot_one_box(xyxy, im0, line_thickness=3)

            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

            # Save
            fname = Path(path).name
            save_path = os.path.join(OUT_DIR, fname)
            cv2.imwrite(save_path, im0)

        t_end = time.time()

        inf_time = t2 - t1
        nms_time = t4 - t3
        total_time = t_end - t_start

        inf_times.append(inf_time)
        nms_times.append(nms_time)
        total_times.append(total_time)

        print(f"[{idx+1:2d}/{NUM_IMAGES}] {Path(path).name}  "
              f"inf: {inf_time*1000:.1f}ms  nms: {nms_time*1000:.1f}ms  "
              f"total: {total_time*1000:.1f}ms")

# Summary
avg_inf = np.mean(inf_times) * 1000
avg_nms = np.mean(nms_times) * 1000
avg_total = np.mean(total_times) * 1000
fps = 1000.0 / avg_total

print(f"\n{'='*60}")
print(f"Results Summary ({NUM_IMAGES} images on CPU)")
print(f"{'='*60}")
print(f"Avg inference time:  {avg_inf:.1f} ms")
print(f"Avg NMS time:        {avg_nms:.1f} ms")
print(f"Avg total time:      {avg_total:.1f} ms")
print(f"FPS (end-to-end):    {fps:.2f}")
print(f"FPS (inference only): {1000.0/avg_inf:.2f}")
print(f"Output saved to:     {OUT_DIR}")
print(f"{'='*60}")

# Cleanup temp dir
shutil.rmtree(tmp_dir)
