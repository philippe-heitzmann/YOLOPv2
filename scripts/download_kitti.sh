#!/bin/bash
#
# Download select KITTI 2011_09_26 raw sequences (synced+rectified)
# that contain pedestrians and cars in urban environments.
# Image sequences are converted to .mp4 videos using ffmpeg.
#
# Usage: ./scripts/download_kitti.sh
#

set -e

DEST_DIR="$HOME/datasets/kitti_raw"
VIDEO_DIR="$HOME/datasets/kitti_videos"
BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"

# Selected sequences with pedestrians + cars in urban/city scenes
# Format: drive_id | category | description
SEQUENCES=(
    "2011_09_26_drive_0005"   # City - urban drive with vehicles and pedestrians
    "2011_09_26_drive_0009"   # Person - pedestrians on road
    "2011_09_26_drive_0011"   # City - urban traffic
    "2011_09_26_drive_0013"   # City - urban scene, commonly used for demos
    "2011_09_26_drive_0014"   # City - pedestrians crossing
    "2011_09_26_drive_0017"   # City - urban intersection
    "2011_09_26_drive_0046"   # Person - pedestrians walking
    "2011_09_26_drive_0056"   # City - busy urban scene
    "2011_09_26_drive_0091"   # City - longer urban sequence
)

mkdir -p "$DEST_DIR"
mkdir -p "$VIDEO_DIR"

echo "============================================"
echo " KITTI Raw Data Downloader"
echo " Destination: $DEST_DIR"
echo " Videos:      $VIDEO_DIR"
echo "============================================"
echo ""

# Download calibration file for 2011_09_26
CALIB_URL="${BASE_URL}/2011_09_26_calib.zip"
CALIB_ZIP="${DEST_DIR}/2011_09_26_calib.zip"
if [ ! -d "${DEST_DIR}/2011_09_26" ] || [ ! -f "${DEST_DIR}/2011_09_26/calib_cam_to_cam.txt" ]; then
    echo "[1/$(( ${#SEQUENCES[@]} + 1 ))] Downloading calibration data..."
    wget -c -q --show-progress -O "$CALIB_ZIP" "$CALIB_URL"
    unzip -o -q "$CALIB_ZIP" -d "$DEST_DIR"
    rm -f "$CALIB_ZIP"
    echo "  -> Calibration data extracted."
else
    echo "[1/$(( ${#SEQUENCES[@]} + 1 ))] Calibration data already exists, skipping."
fi

# Download each sequence
for i in "${!SEQUENCES[@]}"; do
    SEQ="${SEQUENCES[$i]}"
    STEP=$(( i + 2 ))
    TOTAL=$(( ${#SEQUENCES[@]} + 1 ))

    ZIP_NAME="${SEQ}_sync.zip"
    ZIP_PATH="${DEST_DIR}/${ZIP_NAME}"
    SEQ_URL="${BASE_URL}/${SEQ}/${ZIP_NAME}"

    # Check if already extracted (look for image_02 folder with images)
    EXTRACTED_DIR="${DEST_DIR}/2011_09_26/${SEQ}_sync/image_02/data"
    if [ -d "$EXTRACTED_DIR" ] && [ "$(ls -A "$EXTRACTED_DIR" 2>/dev/null)" ]; then
        echo "[${STEP}/${TOTAL}] ${SEQ} already downloaded, skipping."
    else
        echo "[${STEP}/${TOTAL}] Downloading ${SEQ}..."
        wget -c -q --show-progress -O "$ZIP_PATH" "$SEQ_URL"
        echo "  -> Extracting..."
        unzip -o -q "$ZIP_PATH" -d "$DEST_DIR"
        rm -f "$ZIP_PATH"
        echo "  -> Done."
    fi
done

echo ""
echo "============================================"
echo " Converting image sequences to MP4 videos"
echo "============================================"
echo ""

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: ffmpeg not found. Skipping video conversion."
    echo "Install ffmpeg and re-run, or use the image sequences directly."
    echo "  sudo apt install ffmpeg"
    echo ""
    echo "Image sequences are at: ${DEST_DIR}/2011_09_26/<sequence>/image_02/data/"
    exit 0
fi

for SEQ in "${SEQUENCES[@]}"; do
    IMG_DIR="${DEST_DIR}/2011_09_26/${SEQ}_sync/image_02/data"
    OUT_VIDEO="${VIDEO_DIR}/${SEQ}.mp4"

    if [ -f "$OUT_VIDEO" ]; then
        echo "  ${SEQ}.mp4 already exists, skipping."
        continue
    fi

    if [ ! -d "$IMG_DIR" ]; then
        echo "  WARNING: Image dir not found for ${SEQ}, skipping."
        continue
    fi

    NUM_FRAMES=$(ls "$IMG_DIR"/*.png 2>/dev/null | wc -l)
    echo "  Converting ${SEQ} (${NUM_FRAMES} frames) -> ${SEQ}.mp4"

    ffmpeg -y -framerate 10 -pattern_type glob -i "${IMG_DIR}/*.png" \
        -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
        "$OUT_VIDEO" 2>/dev/null

    echo "  -> Saved: ${OUT_VIDEO}"
done

echo ""
echo "============================================"
echo " Done!"
echo "============================================"
echo ""
echo "Raw image sequences: ${DEST_DIR}/2011_09_26/"
echo "MP4 videos:          ${VIDEO_DIR}/"
echo ""
echo "To run YOLOPv2 on a video:"
echo "  python demo.py --source ${VIDEO_DIR}/2011_09_26_drive_0013.mp4"
