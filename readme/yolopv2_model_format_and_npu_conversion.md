# YOLOPv2 — Model Format & NPU Conversion Feasibility

## Model Format

The YOLOPv2 weights file (`data/weights/yolopv2.pt`, 150 MB) is a **TorchScript Traced Model** — not a regular PyTorch state_dict. It was saved via `torch.jit.trace()` and is loaded with `torch.jit.load()`.

This means the model is a self-contained serialized computation graph (stored as a ZIP archive with bytecode + tensor data). No separate model definition class is needed to load it.

## Model Specifications

| Property | Value |
|----------|-------|
| Format | TorchScript (traced) |
| Parameters | **38,952,443 (39.0M)** |
| Total layers | 442 |
| Backbone | ELAN (Efficient Layer Aggregation Network) |
| Input shape | `[1, 3, 384, 640]` (batch, RGB, height, width) |
| Input dtype | float32, normalized [0.0, 1.0] |
| File size | 150 MB |

### Outputs (3 heads, multi-task)

| Output | Shape | Description |
|--------|-------|-------------|
| Detection | `[pred, anchor_grid]` — 3 scale tensors of shape [1, 255, H, W] | 80-class COCO object detection (255 = 3 anchors x 85 values) |
| Drivable area segmentation | `[1, 2, 384, 640]` | 2-class (background vs. drivable) per-pixel logits |
| Lane line detection | `[1, 1, 384, 640]` | Binary lane probability per pixel |

## Current Performance

From our latency benchmarks on this platform (QCS6490 ARM aarch64):

| Backend | Latency | FPS |
|---------|---------|-----|
| **CPU (current)** | **~1712 ms/frame** | **0.58 fps** |
| V100 GPU (paper) | ~11 ms/frame | 91 fps |

The CPU inference at 1.7 seconds per frame is the main bottleneck in the brake pipeline (YOLOv11 on NPU takes ~70ms).

## Can We Convert YOLOPv2 to Run on NPU (QNN/HTP)?

**Yes, it is feasible.** The conversion pipeline would follow the same approach used for the YOLOv11 model (`libyolo11l.so`). Here is how it would work and what changes are needed.

### Conversion Pipeline

The script below (used for YOLOv11n) shows the standard QNN conversion flow. The same pipeline applies to YOLOPv2 with modifications:

```
PyTorch (.pt) → ONNX (.onnx) → QNN C++ (.cpp/.bin) → Shared library (.so)
```

**Step-by-step:**

1. **TorchScript → ONNX export**
   - Use `torch.onnx.export()` on the traced model
   - Must specify the 3 output nodes (detection, segmentation, lane line)
   - Fixed input shape: `[1, 3, 384, 640]`

2. **ONNX → QNN conversion** (`qnn-onnx-converter`)
   - Same tool as for YOLOv11
   - Need to specify `--out_node` for the segmentation and lane outputs
   - INT8 quantization with calibration data (BDD100K or KITTI driving images)
   - Per-channel quantization + entropy calibration (same flags as YOLOv11 script)

3. **QNN → .so compilation** (`qnn-model-lib-generator`)
   - Cross-compile for aarch64 target
   - Produces `libyolopv2.so` loadable via `qai_appbuilder`

4. **Context binary generation** (`qnn-context-binary-generator`)
   - Optional: pre-compiles the HTP execution graph for faster first-inference

### Key Differences vs. YOLOv11 Conversion

| Aspect | YOLOv11 | YOLOPv2 |
|--------|---------|---------|
| Parameters | ~25M (Large) | 39M |
| Input size | 640x640 | 384x640 |
| Output nodes | 2 (boxes + scores) | 3 (detection + seg + lane) |
| Source format | PyTorch → ONNX | TorchScript → ONNX |
| Architecture | Standard YOLO | ELAN backbone + multi-head |
| Quantization concern | Low (detection only) | **Medium** — segmentation masks are sensitive to quantization noise |

### Challenges and Risks

1. **Multi-head output**: YOLOPv2 has 3 output branches. The `qnn-onnx-converter` needs all 3 output node names specified correctly. These must be identified from the ONNX graph (using Netron or `onnx` Python API).

2. **Segmentation quality under INT8**: The drivable area segmentation produces soft logits that get argmax'd. INT8 quantization may reduce mask quality (noisy edges, missed regions). This needs careful validation — the 93.2% mIoU on FP32 may drop several points.

3. **Model size on HTP**: At 39M parameters, the model is larger than YOLOv11L. HTP memory constraints on QCS6490 need to be verified. The model may need to be split across HTP + CPU if it doesn't fit.

4. **ONNX export of TorchScript**: Exporting a `torch.jit` traced model to ONNX can be tricky. The `split_for_trace_model()` workaround in the current code suggests there are already trace-related quirks. The ONNX export may need custom handling for the multi-output structure.

5. **Calibration data**: INT8 quantization requires representative calibration images (typically 100-500 images). We'd use KITTI or BDD100K driving images resized to 384x640.

### Expected Performance Improvement

Based on the YOLOv11 results on this same NPU:

| Metric | CPU (current) | NPU (estimated) |
|--------|---------------|------------------|
| Latency | 1712 ms | **80-150 ms** (10-20x speedup) |
| FPS | 0.58 | **7-12 fps** |

This estimate is based on:
- YOLOv11L (25M params) runs at ~70ms on HTP
- YOLOPv2 (39M params) is ~1.6x larger → proportional increase
- Multi-head output adds some overhead vs. detection-only
- INT8 quantization on HTP is very efficient for conv-heavy architectures

### What Would the Conversion Process Look Like

The adapted conversion script would be:

```bash
#!/bin/bash
# yolopv2_to_qnn.sh

# Step 1: TorchScript → ONNX
python3 -c "
import torch
model = torch.jit.load('data/weights/yolopv2.pt', map_location='cpu')
model.eval()
dummy = torch.randn(1, 3, 384, 640)
torch.onnx.export(model, dummy, 'yolopv2.onnx',
                  input_names=['images'],
                  output_names=['det_out', 'da_seg', 'll_seg'],
                  opset_version=12)
"

# Step 2: ONNX → QNN (with INT8 quantization)
# NOTE: output node names must be verified from the ONNX graph
DA_SEG_NODE='<seg_output_node_name>'   # identify via Netron
LL_SEG_NODE='<ll_output_node_name>'    # identify via Netron

qnn-onnx-converter \
    --input_network yolopv2.onnx \
    --input_dim "images" 1,3,384,640 \
    --output_path yolopv2.cpp \
    --out_node ${DA_SEG_NODE} --out_node ${LL_SEG_NODE} \
    --input_list calibration_images_list.txt \
    --use_per_channel_quantization \
    --act_quantizer_calibration entropy \
    --param_quantizer_calibration entropy \
    --act_bitwidth 8 \
    --weights_bitwidth 8 \
    --bias_bitwidth 32

# Step 3: Compile to .so
# (same cross-compilation as YOLOv11 script)
```

### Recommendation

The conversion is **feasible and highly recommended** given the current 1.7s CPU latency is the pipeline bottleneck. The main work items are:

1. Export TorchScript model to ONNX (verify all 3 output heads export correctly)
2. Identify exact output node names from the ONNX graph
3. Prepare calibration image list (384x640 driving images)
4. Run `qnn-onnx-converter` with INT8 quantization
5. Validate segmentation mask quality (mIoU) after quantization
6. Cross-compile to `.so` and integrate into `brake_ped_road.py`

If segmentation quality degrades too much under INT8, a fallback is to use **FP16 quantization** (`--act_bitwidth 16 --weights_bitwidth 16`) which preserves more precision at the cost of ~2x latency vs INT8 (still ~5-8x faster than CPU).
