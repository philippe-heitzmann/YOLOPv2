# YOLOPv2 Latency Report — KITTI drive_0046 (CPU)

**Date:** 2026-03-07
**Video:** `2011_09_26_drive_0046.mp4` (first 3 seconds, 30 frames)
**Dataset:** KITTI Raw 2011_09_26, "Person" category — pedestrians walking on a residential street

---

## Hardware

| Parameter | Value |
|---|---|
| Architecture | aarch64 (ARM 64-bit) |
| Big cores | 4x Cortex-A78 @ up to 2707 MHz |
| Little cores | 4x Cortex-A55 @ up to 1958 MHz |
| Threads per core | 1 |
| RAM | 7.2 GB |
| OS | Linux 6.6.52 (PREEMPT) |

## Model & Input Configuration

| Parameter | Value |
|---|---|
| Model | YOLOPv2 (TorchScript traced) |
| PyTorch version | 2.9.1+cpu |
| Precision | FP32 (CPU, no half) |
| Model input tensor | `[1, 3, 384, 640]` |
| Original frame resolution | 1242 x 376 |
| `--img-size` | 640 |
| Stride | 32 |
| Confidence threshold | 0.3 |
| IoU threshold (NMS) | 0.45 |

**Note:** The 1242x376 KITTI frames are letterboxed/resized to 384x640 to fit the model's 640px input while preserving aspect ratio (stride-aligned to 32).

## Factors Affecting Latency

1. **CPU-only inference on ARM** — No GPU/NPU acceleration. The Cortex-A78 big cores handle the bulk of computation, but lack the throughput of even a low-end GPU.
2. **FP32 precision** — The model runs full 32-bit float. FP16 or INT8 quantization would reduce latency but requires CUDA or a compatible accelerator.
3. **TorchScript trace overhead** — The `split_for_trace_model` post-processing step adds ~10-16ms per frame due to JIT trace compatibility workarounds.
4. **Input resolution** — At 640px input, the effective tensor is [1,3,384,640]. Smaller `--img-size` (e.g., 320) would reduce latency roughly quadratically but hurt detection accuracy.
5. **Number of detections** — NMS time scales with detection count, though it's negligible here (1-5ms) with only 2-3 detections per frame.
6. **Multi-task heads** — YOLOPv2 runs three output heads simultaneously (object detection, drivable area segmentation, lane line segmentation), which increases compute vs. a detection-only model.
7. **Thread count** — PyTorch defaults to using all available cores. On big.LITTLE ARM, scheduling across heterogeneous cores can introduce variance.

## Latency Distribution (30 frames)

### Summary Statistics

| Metric | Inference (ms) | NMS (ms) | Total Pipeline (ms) |
|---|---|---|---|
| **Mean** | 1711.5 | 1.8 | 1734.7 |
| **Std Dev** | 23.0 | 0.8 | 24.4 |
| **Min** | 1659.9 | 1.2 | 1681.4 |
| **Max** | 1784.4 | 5.3 | 1819.6 |
| **Median** | 1712.7 | 1.6 | 1734.3 |
| **P95** | 1749.1 | 3.3 | 1771.4 |
| **P99** | 1774.1 | 4.7 | 1805.6 |
| **Effective FPS** | — | — | **0.58** |

### Per-Frame Breakdown

| Frame | Inference (ms) | Split/Trace (ms) | NMS (ms) | Seg Post (ms) | Total (ms) | Detections |
|---|---|---|---|---|---|---|
| 1 | 1784.4 | 15.9 | 5.3 | 13.9 | 1819.6 | 3 |
| 2 | 1724.4 | 9.6 | 1.6 | 11.7 | 1747.4 | 3 |
| 3 | 1676.2 | 12.0 | 1.7 | 10.0 | 1699.8 | 3 |
| 4 | 1707.2 | 9.9 | 1.5 | 10.4 | 1728.9 | 3 |
| 5 | 1730.4 | 12.8 | 1.4 | 10.7 | 1754.8 | 3 |
| 6 | 1696.2 | 13.8 | 3.3 | 11.3 | 1724.5 | 3 |
| 7 | 1693.7 | 12.2 | 1.7 | 10.5 | 1718.0 | 2 |
| 8 | 1708.8 | 10.2 | 1.5 | 11.0 | 1731.6 | 2 |
| 9 | 1720.6 | 10.9 | 1.6 | 10.5 | 1743.6 | 2 |
| 10 | 1691.4 | 10.3 | 1.8 | 10.8 | 1714.2 | 2 |
| 11 | 1674.8 | 10.5 | 1.2 | 10.1 | 1696.5 | 2 |
| 12 | 1711.0 | 9.6 | 1.7 | 10.1 | 1732.4 | 2 |
| 13 | 1701.4 | 9.6 | 1.4 | 9.9 | 1722.2 | 2 |
| 14 | 1703.1 | 13.5 | 2.1 | 11.6 | 1730.2 | 2 |
| 15 | 1718.8 | 10.8 | 1.6 | 9.9 | 1741.0 | 2 |
| 16 | 1715.7 | 7.8 | 1.5 | 9.5 | 1734.6 | 2 |
| 17 | 1712.7 | 10.7 | 1.7 | 10.8 | 1735.8 | 2 |
| 18 | 1744.0 | 11.1 | 1.9 | 9.8 | 1766.8 | 2 |
| 19 | 1659.9 | 10.3 | 1.7 | 9.5 | 1681.4 | 2 |
| 20 | 1713.1 | 9.1 | 1.4 | 9.6 | 1733.1 | 2 |
| 21 | 1703.2 | 9.3 | 1.4 | 9.9 | 1723.7 | 2 |
| 22 | 1706.5 | 15.5 | 1.8 | 10.5 | 1734.3 | 2 |
| 23 | 1721.5 | 10.3 | 1.5 | 10.5 | 1743.8 | 2 |
| 24 | 1717.5 | 9.3 | 1.5 | 9.5 | 1737.9 | 2 |
| 25 | 1725.0 | 8.8 | 1.5 | 9.3 | 1744.6 | 2 |
| 26 | 1703.7 | 16.3 | 1.7 | 9.7 | 1731.4 | 2 |
| 27 | 1753.4 | 12.1 | 1.8 | 10.5 | 1777.7 | 2 |
| 28 | 1719.8 | 8.7 | 1.7 | 9.3 | 1739.5 | 2 |
| 29 | 1707.0 | 11.3 | 1.5 | 10.1 | 1729.9 | 2 |
| 30 | 1700.5 | 10.7 | 1.2 | 9.5 | 1721.7 | 2 |

### Observations

- **Inference dominates** — The neural network forward pass accounts for ~98.7% of total pipeline time. NMS and segmentation post-processing are negligible (~1.3% combined).
- **Stable latency** — Standard deviation is only 23ms (1.3% of mean), indicating consistent performance with no thermal throttling or scheduling outliers across the 30-frame run.
- **Frame 1 is warmest** — The first frame (1819ms) is the slowest despite a 3-iteration warmup, likely due to initial memory allocation and cache population.
- **Detection count dropped** — Frames 1-6 detect 3 objects, frames 7-30 detect 2, as the camera approaches and one pedestrian exits the frame.
- **0.58 FPS on CPU** — Far below real-time (10 FPS source). GPU or NPU acceleration, model quantization, or a lighter architecture would be needed for real-time on this platform.
