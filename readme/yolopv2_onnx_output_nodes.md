# YOLOPv2 ONNX Output Node Names

Output nodes extracted from `yolopv2.onnx` (exported with input shape `[1, 3, 384, 640]`, opset 11).

## 3 Primary Output Branches

| Branch | Node Name | Shape | Description |
|--------|-----------|-------|-------------|
| Detection | `760` | `[]` (scalar/dynamic) | Object detection output |
| Drivable Area Segmentation | `677` | `[1, 2, 384, 640]` | 2-class drivable area mask |
| Lane Line Segmentation | `759` | `[1, 1, 384, 640]` | Lane line mask |

## Additional Outputs (Detection Head Internals)

These are intermediate anchor grid outputs from the 3 detection head scales. They may not be needed for QNN conversion but are present in the ONNX graph:

| Node Name | Shape |
|-----------|-------|
| `769` | `[1, 3, 1, 1, 2]` |
| `770` | `[1, 3, 1, 1, 2]` |
| `771` | `[1, 3, 1, 1, 2]` |

## QNN Conversion Command (3 main outputs)

```bash
qnn-onnx-converter --input_network yolopv2.onnx \
                   --input_dim "images" 1,3,384,640 \
                   --out_node "760" \
                   --out_node "677" \
                   --out_node "759" \
                   --output_path yolopv2.cpp
```

## How to Reproduce

```bash
source ~/qairt_py310/bin/activate
python3 scripts/inspect_onnx_outputs.py --pt ~/weights/yolopv2.pt --export-onnx yolopv2.onnx
```
