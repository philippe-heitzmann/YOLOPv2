# YOLOPv2 QNN Conversion: NCHW vs NHWC Layout Issue

## Problem

After converting YOLOPv2 from ONNX to QNN (INT8 quantized) and running inference on CPU via `qnn-net-run`, the drivable area and lane line segmentation masks showed **repeating artifact patterns scattered across the image** instead of coherent road regions. No mask region aligned with the actual road surface.

## Root Cause

**Data layout mismatch between ONNX (NCHW) and QNN (NHWC).**

The QNN toolchain internally converts models from NCHW to NHWC layout during compilation. This means:

1. **Inputs to `qnn-net-run` must be in NHWC (HWC) format**, not NCHW (CHW).
2. **Outputs from `qnn-net-run` are in NHWC format**, not NCHW.

The original scripts were:
- Saving input raw files as CHW (transposing HWC -> CHW before writing)
- Reading output raw files as NCHW `[1, C, H, W]` instead of NHWC `[1, H, W, C]`

Both mismatches caused the data to be spatially scrambled — the channel interleaving pattern produced the characteristic "repeating small regions" visual artifact.

### Why This Produces Repeating Patterns

When CHW data is interpreted as HWC (or vice versa), each spatial position reads values from wrong channels/positions in a periodic pattern. For a `[1, 2, 384, 640]` tensor reshaped as `[1, 384, 640, 2]`:
- Row 0 of the "NCHW interpretation" actually reads across both channels and spatial positions
- This creates a tiled/striped artifact with period related to the channel count

## Fix

### Input preprocessing (`prepare_calibration_list.py`, `run_yolopv2_qnn_cpu.py`)

```python
# BEFORE (wrong — NCHW):
arr = np.array(img, dtype=np.float32) / 255.0
arr = arr.transpose(2, 0, 1)  # HWC -> CHW
arr.tofile(raw_path)

# AFTER (correct — NHWC):
arr = np.array(img, dtype=np.float32) / 255.0
arr.tofile(raw_path)  # keep HWC layout
```

### Output reading (`visualize_qnn_outputs.py`)

```python
# BEFORE (wrong — NCHW):
drivable = raw.reshape(1, 2, 384, 640)
mask = drivable[0].argmax(axis=0)

# AFTER (correct — NHWC):
drivable = raw.reshape(1, 384, 640, 2)
mask = drivable[0].argmax(axis=-1)
```

## Verification

Compared QNN quantized output channel statistics against PyTorch FP32 reference for the same image (`amsterdam_01241`):

| Metric | PyTorch FP32 | QNN INT8 (NHWC) | QNN INT8 (NCHW, wrong) |
|--------|-------------|-----------------|----------------------|
| ch0 mean | 0.9827 | 0.9724 | 0.4810 |
| ch1 mean | 0.0044 | 0.0163 | 0.4629 |
| Drivable % | 1.3% | 1.8% | 9.7% |

The NHWC interpretation closely matches PyTorch reference. The NCHW interpretation shows ~50/50 channel means (scrambled data) and inflated drivable coverage.

## Key Takeaway

When using the QNN toolchain (`qnn-onnx-converter` + `qnn-net-run`):
- The converter handles NCHW->NHWC transpose internally during model compilation
- All raw I/O for `qnn-net-run` (and calibration data for the converter) must be in **NHWC** layout
- This applies regardless of the original ONNX model's layout convention
