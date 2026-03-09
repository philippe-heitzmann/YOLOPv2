# YOLOPv2 QNN (NPU) Latency Report — ECP Amsterdam Test Set

## Setup

| Property | Value |
|----------|-------|
| Model | YOLOPv2 INT8 quantized `.so` (QNN/HTP) |
| Model file | `libyolopv2.so` (39 MB) |
| Backend | Qualcomm HTP (NPU) via `qai_appbuilder` |
| Platform | QCS6490 (Hexagon DSP v68) |
| Input size | 384 x 640 (NHWC, float32 [0,1]) |
| Quantization | INT8 (per-channel, entropy calibration) |
| Dataset | ECP Amsterdam test set (`~/datasets/ecp_amsterdam_test/images/val/`) |
| Total images | 179 |
| Script | `scripts/yolopv2_qnn_images.py` |

## Latency Results

| Metric | Value |
|--------|-------|
| **Mean latency** | **51.4 ms** |
| **Median latency** | **51.6 ms** |
| **Std deviation** | 2.7 ms |
| **Min latency** | 43.1 ms |
| **Max latency** | 70.8 ms |
| **P95 latency** | 54.5 ms |
| **P99 latency** | 55.6 ms |
| **Warmup (1st frame)** | 53.3 ms |
| **Mean FPS** | **19.5 fps** |
| **Median FPS** | **19.4 fps** |

## CPU vs NPU Comparison

| Backend | Mean Latency | FPS | Speedup |
|---------|-------------|-----|---------|
| CPU (TorchScript FP32) | 1,712 ms | 0.58 | 1x |
| **NPU (QNN INT8)** | **51.4 ms** | **19.5** | **33x** |

The NPU delivers a **33x speedup** over the CPU TorchScript baseline, bringing YOLOPv2 from unusable (0.58 fps) to real-time capable (19.5 fps).

## Notes

- Latency includes preprocessing (resize + color convert) and postprocessing (argmax + resize to original resolution), not just inference
- The first frame latency (53ms) is comparable to subsequent frames — the HTP context is pre-warmed during model loading
- Standard deviation of 2.7ms indicates very stable inference times across all 179 frames
- Road mask coverage varied from 0% (no visible road surface) to ~20% depending on the scene
- The `free(): invalid pointer` crash at process exit is a known QNN teardown issue and does not affect inference correctness

## Output Locations

- Annotated images: `~/annotated_outputs/yolopv2_qnn_amsterdam_full/`
- Latency stats JSON: `~/annotated_outputs/yolopv2_qnn_amsterdam_full/latency_stats.json`
- GCS: `gs://model_outputs_saferide/yolopv2_qnn_npu_amsterdam_0308/`
