#!/bin/bash
#
# Convert YOLOPv2 ONNX model to QNN format for NPU execution.
# Adapted from ~/Data/ym/yolo11n_qnn_opset11/yolo11n_to_qnn.sh
#
# Output nodes:
#   /105/0_2/Mul_output_0  [1, 255, 48, 80]   detection head scale 0
#   /105/1_2/Mul_output_0  [1, 255, 24, 40]   detection head scale 1
#   /105/2_2/Mul_output_0  [1, 255, 12, 20]   detection head scale 2
#   677                    [1, 2, 384, 640]   drivable area segmentation
#   759                    [1, 1, 384, 640]   lane line segmentation
#
# Note: 769/770/771 are initializers (constant anchor grids), not computed outputs.
#

set -e

CWD=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

ONNX_MODEL="${REPO_DIR}/yolopv2.onnx"
WEIGHTS="${HOME}/weights/yolopv2.pt"
CALIB_LIST="${REPO_DIR}/calibration_raw/input_list.txt"

# Output node names from ONNX graph
# Detection head (3 scales)
DET_HEAD_0='/105/0_2/Mul_output_0'    # [1, 255, 48, 80]
DET_HEAD_1='/105/1_2/Mul_output_0'    # [1, 255, 24, 40]
DET_HEAD_2='/105/2_2/Mul_output_0'    # [1, 255, 12, 20]
# Segmentation heads
DRIVABLE_SEG='677'                     # [1, 2, 384, 640]
LANE_SEG='759'                         # [1, 1, 384, 640]

# Activate environment
source ~/qairt_py310/bin/activate
source /opt/qcom/aistack/qairt/2.41.0.251128/bin/envsetup.sh

# Step 0: Export ONNX if not present
if [ ! -e "${ONNX_MODEL}" ]; then
  echo "======== Exporting TorchScript to ONNX ==========="
  python3 "${SCRIPT_DIR}/inspect_onnx_outputs.py" \
    --pt "${WEIGHTS}" --export-onnx "${ONNX_MODEL}"
fi

# Step 1: Prepare calibration data if not present
if [ ! -e "${CALIB_LIST}" ]; then
  echo "======== Preparing calibration data ==========="
  python3 "${SCRIPT_DIR}/prepare_calibration_list.py"
fi

if [ ! -e "${CALIB_LIST}" ]; then
  echo "Calibration data preparation failed..."
  exit 1
fi

# Step 2: Create working directory
WORK_DIR="${REPO_DIR}/qnn_build"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Step 3: ONNX to QNN conversion with INT8 quantization
if [ -e "${ONNX_MODEL}" ]; then
  echo "======== Converting ONNX to QNN ==========="
  qnn-onnx-converter \
    --input_network "${ONNX_MODEL}" \
    --input_dim "images" 1,3,384,640 \
    --output_path yolopv2.cpp \
    --out_node "${DET_HEAD_0}" \
    --out_node "${DET_HEAD_1}" \
    --out_node "${DET_HEAD_2}" \
    --out_node "${DRIVABLE_SEG}" \
    --out_node "${LANE_SEG}" \
    --input_list "${CALIB_LIST}" \
    --use_per_channel_quantization \
    --act_quantizer_calibration entropy \
    --param_quantizer_calibration entropy \
    --act_bitwidth 8 \
    --weights_bitwidth 8 \
    --bias_bitwidth 32 \
    --quantizer_log qlog
else
  echo "ONNX model not found at ${ONNX_MODEL}"
  exit 1
fi

# Step 4: Verify conversion outputs
if [ -e yolopv2.bin ] && [ -e yolopv2.cpp ] && [ -e yolopv2_net.json ]; then
  echo "======== ONNX to QNN conversion completed ==========="
else
  echo "ONNX to QNN conversion failed..."
  exit 1
fi

# Step 5: Generate x86 model library for CPU testing
echo "======== Generating x86 CPU model library ==========="
QNN_TARGET_ARCH_X86="x86_64-linux-clang"
python3 "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator" \
  -c "yolopv2.cpp" -b "yolopv2.bin" -o model_libs -t "${QNN_TARGET_ARCH_X86}"

if [ -e "model_libs/${QNN_TARGET_ARCH_X86}/libyolopv2.so" ]; then
  echo "======== x86 CPU model library generated ==========="
  cp "model_libs/${QNN_TARGET_ARCH_X86}/libyolopv2.so" .
else
  echo "Failed to generate x86 CPU model library..."
  exit 1
fi

echo ""
echo "======== BUILD COMPLETE ==========="
echo "Working directory: ${WORK_DIR}"
echo "Model library:     ${WORK_DIR}/libyolopv2.so"
echo "Model config:      ${WORK_DIR}/yolopv2_net.json"
echo ""
echo "To run CPU inference, use:"
echo "  python3 ${SCRIPT_DIR}/run_yolopv2_qnn_cpu.py"
