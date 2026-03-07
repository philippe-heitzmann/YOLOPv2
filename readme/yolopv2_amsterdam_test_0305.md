# YOLOPv2 Amsterdam Test - 2025-03-05

## Overview

YOLOPv2 inference was run on **30 randomly selected images** from the ECP Amsterdam test dataset (`~/datasets/ecp_amsterdam_test/images/val/`). The annotated output images (with bounding boxes, drivable area segmentation, and lane line segmentation) are saved to:

```
~/yolopv2_amsterdam_test_0305/
```

## Hardware

| Spec       | Value                                        |
|------------|----------------------------------------------|
| Platform   | Quectel QuecPi Alpha (aarch64)               |
| CPU        | ARM Cortex-A55 (CPU part 0xd05), 8 cores     |
| RAM        | 7.2 GB                                        |
| OS         | Debian 13 (trixie), Linux 6.6.52             |
| PyTorch    | 2.9.1+cpu                                     |

## Execution Backend: CPU

The model ran entirely on the **CPU**. No NPU (Neural Processing Unit) was detected or used on this platform:

- No NPU device nodes found (`/dev/npu*`, `/dev/rknpu*`, etc.)
- No NPU SDK libraries installed (no RKNN, QNN, SNPE, or similar packages)
- PyTorch was built for CPU only (`torch 2.9.1+cpu`), with no CUDA or NPU backend available

The YOLOPv2 TorchScript model (`.pt`) was loaded with `map_location='cpu'` and executed using standard PyTorch CPU inference on the ARM Cortex-A55 cores.

## Latency Results (30 images)

| Metric                    | Value         |
|---------------------------|---------------|
| Avg inference time        | 1699.2 ms     |
| Avg NMS time              | 2.5 ms        |
| Avg total time per frame  | 1802.9 ms     |
| FPS (end-to-end)          | 0.55          |
| FPS (inference only)      | 0.59          |

### Breakdown

- **Inference** dominates the total time (~94%). The Cortex-A55 is a power-efficient core not designed for heavy neural network workloads, so ~1.7s per frame is expected.
- **NMS** is negligible at ~2.5 ms per frame.
- **Pre/post-processing** (image loading, resize, letterbox, segmentation mask generation, drawing, and saving) accounts for the remaining ~100 ms.

## Model Details

- **Model**: YOLOPv2 (TorchScript, `data/weights/yolopv2.pt`, 150 MB)
- **Input size**: 640x640
- **Confidence threshold**: 0.3
- **IOU threshold**: 0.45
- **Tasks**: Object detection + drivable area segmentation + lane line segmentation

## Conclusion

At ~0.55 FPS on the Cortex-A55 CPU, the model is far from real-time on this platform. To achieve real-time performance, the model would need to be deployed on an NPU or GPU-accelerated backend (e.g., via ONNX Runtime, RKNN, QNN, or TensorRT), or a lighter model variant would be needed.
