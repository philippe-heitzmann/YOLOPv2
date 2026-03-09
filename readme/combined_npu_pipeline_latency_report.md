# Combined NPU Pipeline Latency Report — YOLOv11L + YOLOPv2

## Setup

| Property | Value |
|----------|-------|
| Platform | QCS6490 (Hexagon DSP v68, NPU/HTP) |
| YOLOv11L model | `libyolo11l.so` — pedestrian detection, INT8 |
| YOLOPv2 model | `libyolopv2.so` — road segmentation + lane lines, INT8 |
| Backend | Both models on QNN/HTP (NPU) |
| Dataset | ECP Amsterdam test set (`~/datasets/ecp_amsterdam_test/images/val/`) |
| Total images | 179 |
| Morph close kernel | 25x25 (elliptical, on road mask) |
| Script | `scripts/brake_ped_road_images.py` |

## End-to-End Pipeline Latency

| Metric | Value |
|--------|-------|
| **Mean latency** | **173.0 ms** |
| **Median latency** | **173.0 ms** |
| **Std deviation** | 4.3 ms |
| **Min latency** | 157.2 ms |
| **Max latency** | 188.0 ms |
| **P95 latency** | 179.7 ms |
| **P99 latency** | 183.0 ms |
| **Mean FPS** | **5.8 fps** |
| **Median FPS** | **5.8 fps** |

## Per-Component Breakdown

| Component | Mean (ms) | Median (ms) | P95 (ms) | % of Total |
|-----------|-----------|-------------|----------|------------|
| YOLOv11L ped detection (NPU) | 82.5 | 83.0 | 87.0 | 47.7% |
| YOLOPv2 road seg + lane (NPU) | 41.0 | 40.8 | 43.9 | 23.7% |
| Postprocessing (NMS + morph close + brake) | 49.5 | 49.1 | 52.3 | 28.6% |

## Before vs After NPU Migration

| Configuration | Mean Latency | FPS | Speedup |
|--------------|-------------|-----|---------|
| YOLOv11 NPU + YOLOPv2 **CPU** (old) | ~1,830 ms | 0.55 | 1x |
| YOLOv11 NPU + YOLOPv2 **NPU** (new) | **173 ms** | **5.8** | **10.6x** |

Moving YOLOPv2 from CPU TorchScript (1,712ms) to NPU QNN INT8 (41ms) eliminated the main bottleneck, yielding a **10.6x end-to-end speedup**.

## Brake Decision Summary

| Metric | Count |
|--------|-------|
| Total images | 179 |
| BRAKE decisions | 25 (14.0%) |
| KEEP DRIVING decisions | 154 (86.0%) |

## Optimization Opportunities

The postprocessing stage (49.5ms, 28.6% of total) is now the second-largest component. This includes:
- YOLO NMS and box decoding (~1-2ms)
- YOLOPv2 argmax + resize + morph close (~1-2ms)
- **Preprocessing** for both models (~45ms): two separate resize + color convert operations

Potential optimizations:
1. **Shared preprocessing**: resize once, reuse across models (saves ~20ms)
2. **Batch HTP scheduling**: if QNN supports concurrent context execution
3. **Smaller YOLOv11 variant**: YOLOv11-Nano would reduce the 82ms detection to ~15-20ms at some accuracy cost

## Output Locations

- Latency stats JSON: `~/annotated_outputs/combined_npu_latency_stats.json`
- Annotated sample images: `gs://model_outputs_saferide/brake_ped_road_npu_combined_0308/`
