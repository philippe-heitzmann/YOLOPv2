#!/usr/bin/env python3
"""
Inspect YOLOPv2 ONNX model output layer names.

YOLOPv2 has 3 output branches (detection, drivable area segmentation,
lane line segmentation). The qnn-onnx-converter needs all 3 output node
names specified via --out_node flags.

Usage:
    # If you already have an ONNX file:
    python3 scripts/inspect_onnx_outputs.py --onnx path/to/yolopv2.onnx

    # If you only have the .pt (TorchScript) file, export to ONNX first:
    python3 scripts/inspect_onnx_outputs.py --pt path/to/yolopv2.pt --export-onnx yolopv2.onnx
"""

import argparse
import sys


def export_torchscript_to_onnx(pt_path, onnx_path, input_h=384, input_w=640):
    """Export a TorchScript traced model to ONNX."""
    import torch

    print(f"Loading TorchScript model from {pt_path} ...")
    model = torch.jit.load(pt_path, map_location="cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, input_h, input_w)
    print(f"Exporting to ONNX with input shape [1, 3, {input_h}, {input_w}] ...")

    # Use legacy exporter for TorchScript models (torch 2.x new exporter
    # does not support ScriptModule)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["images"],
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"ONNX model saved to {onnx_path}\n")
    return onnx_path


def inspect_onnx(onnx_path):
    """Load an ONNX model and print all output node names and shapes."""
    import onnx

    print(f"Loading ONNX model from {onnx_path} ...")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model is valid.\n")

    graph = model.graph

    # Print inputs
    print("=" * 60)
    print("INPUTS:")
    print("=" * 60)
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        print(f"  Name:  {inp.name}")
        print(f"  Shape: {shape}")
        print()

    # Print outputs
    print("=" * 60)
    print("OUTPUTS (use these as --out_node for qnn-onnx-converter):")
    print("=" * 60)
    for i, out in enumerate(graph.output):
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        print(f"  [{i}] Name:  {out.name}")
        print(f"      Shape: {shape}")
        print()

    # Print qnn-onnx-converter command snippet
    output_names = [out.name for out in graph.output]
    print("=" * 60)
    print("QNN CONVERSION COMMAND SNIPPET:")
    print("=" * 60)
    out_flags = " ".join(f'--out_node "{name}"' for name in output_names)
    print(f"  qnn-onnx-converter --input_network {onnx_path} \\")
    print(f'                     --input_dim "images" 1,3,384,640 \\')
    print(f"                     {out_flags} \\")
    print(f"                     --output_path yolopv2.cpp")
    print()

    return output_names


def main():
    parser = argparse.ArgumentParser(
        description="Inspect YOLOPv2 ONNX output layer names for QNN conversion"
    )
    parser.add_argument("--onnx", type=str, help="Path to existing ONNX model")
    parser.add_argument("--pt", type=str, help="Path to TorchScript .pt model")
    parser.add_argument(
        "--export-onnx", type=str, default="yolopv2.onnx",
        help="Output path for ONNX export (used with --pt)",
    )
    parser.add_argument(
        "--input-height", type=int, default=384,
        help="Input height for ONNX export (default: 384)",
    )
    parser.add_argument(
        "--input-width", type=int, default=640,
        help="Input width for ONNX export (default: 640)",
    )
    args = parser.parse_args()

    if not args.onnx and not args.pt:
        parser.error("Provide either --onnx or --pt")

    onnx_path = args.onnx
    if args.pt:
        onnx_path = export_torchscript_to_onnx(
            args.pt, args.export_onnx, args.input_height, args.input_width
        )

    inspect_onnx(onnx_path)


if __name__ == "__main__":
    main()
