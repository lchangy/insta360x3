#!/usr/bin/env bash

# Run only the YOLO26s-depth point-cloud publisher.
# The cubemap left/right image topics must already be published.

set -Eeuo pipefail

WORKSPACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROS_SETUP="/opt/ros/humble/setup.bash"
INSTALL_SETUP="${WORKSPACE_DIR}/install/setup.bash"
LOCAL_PYTHON="${WORKSPACE_DIR}/.venv/bin/python"
SYSTEM_PYTHON="/usr/bin/python3"
# The script can be launched from any directory. Override this line only when
# the project's .venv is not the environment used for ROS/YOLO inference.
export DA360_PYTHON="${DA360_PYTHON:-${LOCAL_PYTHON}}"
CONFIG_FILE="${WORKSPACE_DIR}/insta360_ros_driver/config/yolo26s_depth.yaml"
MODEL_PATH="${YOLO26S_DEPTH_MODEL:-${WORKSPACE_DIR}/yolo26s-depth.pt}"
POINT_STRIDE="${YOLO_POINT_STRIDE:-2}"
IMGSZ="${YOLO_IMGSZ:-768}"
DEVICE="${YOLO_DEVICE:-}"
FRAME_ID="${YOLO_FRAME_ID:-camera_frame}"
DEPTH_MODE="${YOLO_DEPTH_MODE:-range}"
PUBLISH_DEPTH=true

die() {
  echo "[yolo26s-depth] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./start_yolo26s_depth_pointcloud.sh [options]

Subscribes to /cubemap/left/image and /cubemap/right/image, then publishes:
  /yolo26s_depth/left/points
  /yolo26s_depth/right/points

Options:
  --model-path PATH   YOLO26s-depth checkpoint or Ultralytics model name.
  --point-stride N    Keep every Nth pixel in the point cloud (default: 2).
  --imgsz N           YOLO inference image size (default: 768).
  --device DEVICE     Torch device, for example cuda:0 or cpu.
  --frame-id ID       Point-cloud frame id (default: camera_frame).
  --depth-mode MODE   Interpret depth as range or optical_z (default: range).
  --no-depth          Do not publish the intermediate 32FC1 depth images.
  -h, --help          Show this help.

Environment:
  DA360_PYTHON         Python environment containing ROS 2, Torch, OpenCV,
                       NumPy, and the pinned YOLO26 depth support.
  YOLO26S_DEPTH_MODEL  Default model path/name.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      [[ $# -ge 2 ]] || die "--model-path requires a path or model name"
      MODEL_PATH="$2"
      shift 2
      ;;
    --point-stride)
      [[ $# -ge 2 ]] || die "--point-stride requires a positive integer"
      POINT_STRIDE="$2"
      shift 2
      ;;
    --imgsz)
      [[ $# -ge 2 ]] || die "--imgsz requires a positive integer"
      IMGSZ="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || die "--device requires a Torch device"
      DEVICE="$2"
      shift 2
      ;;
    --frame-id)
      [[ $# -ge 2 ]] || die "--frame-id requires a frame id"
      FRAME_ID="$2"
      shift 2
      ;;
    --depth-mode)
      [[ $# -ge 2 ]] || die "--depth-mode requires range or optical_z"
      DEPTH_MODE="$2"
      shift 2
      ;;
    --no-depth)
      PUBLISH_DEPTH=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown option: $1"
      ;;
  esac
done

[[ "$POINT_STRIDE" =~ ^[1-9][0-9]*$ ]] || die "point stride must be a positive integer"
[[ "$IMGSZ" =~ ^[1-9][0-9]*$ ]] || die "imgsz must be a positive integer"
[[ "$DEPTH_MODE" == "range" || "$DEPTH_MODE" == "optical_z" ]] \
  || die "depth mode must be range or optical_z"
[[ -r "$ROS_SETUP" ]] || die "ROS 2 Humble setup not found: $ROS_SETUP"
[[ -r "$INSTALL_SETUP" ]] || die "workspace is not built; missing $INSTALL_SETUP"
[[ -r "$CONFIG_FILE" ]] || die "parameter file not found: $CONFIG_FILE"

cd -- "$WORKSPACE_DIR"
# ROS 2 setup scripts reference optional variables that may be unset. Keep
# nounset enabled for this launcher, but disable it while sourcing ROS files.
set +u
source "$ROS_SETUP"
source "$INSTALL_SETUP"
set -u

# Keep ROS logs inside the workspace so the node does not depend on the
# permissions of ~/.ros (and so the launcher works in restricted environments).
export ROS_LOG_DIR="${ROS_LOG_DIR:-${WORKSPACE_DIR}/.ros/log}"
mkdir -p "$ROS_LOG_DIR"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

if [[ -x "$DA360_PYTHON" ]]; then
  WORKER_PYTHON="$DA360_PYTHON"
elif [[ "$DA360_PYTHON" == "$LOCAL_PYTHON" && -x "$SYSTEM_PYTHON" ]]; then
  WORKER_PYTHON="$SYSTEM_PYTHON"
else
  die "Python executable not found: $DA360_PYTHON"
fi

PACKAGE_PREFIX="$(ros2 pkg prefix insta360_ros_driver 2>/dev/null)" \
  || die "insta360_ros_driver is not available; rebuild this workspace"
NODE_SCRIPT="${PACKAGE_PREFIX}/lib/insta360_ros_driver/yolo26s_depth_pointcloud.py"
[[ -x "$NODE_SCRIPT" ]] || die "publisher script is not installed; rebuild this workspace"

"$WORKER_PYTHON" -c \
  'import cv2, numpy, rclpy, torch, ultralytics; from ultralytics.nn.tasks import DepthModel' \
  >/dev/null 2>&1 \
  || die "Python lacks ROS/Torch/OpenCV or YOLO26 depth support: $WORKER_PYTHON"

if [[ "$MODEL_PATH" != /* && "$MODEL_PATH" == *.pt \
  && -r "${WORKSPACE_DIR}/${MODEL_PATH}" ]]; then
  MODEL_PATH="${WORKSPACE_DIR}/${MODEL_PATH}"
fi

echo "[yolo26s-depth] Python: $WORKER_PYTHON"
echo "[yolo26s-depth] Model: $MODEL_PATH"
echo "[yolo26s-depth] Point stride: $POINT_STRIDE"
echo "[yolo26s-depth] Inference size: $IMGSZ; depth mode: $DEPTH_MODE"
echo "[yolo26s-depth] Waiting for /cubemap/left/image and /cubemap/right/image"

ROS_ARGS=(
  --ros-args
  --params-file "$CONFIG_FILE"
  -p "model_path:=${MODEL_PATH}"
  -p "point_stride:=${POINT_STRIDE}"
  -p "imgsz:=${IMGSZ}"
  -p "depth_mode:=${DEPTH_MODE}"
  -p "frame_id:=${FRAME_ID}"
  -p "publish_depth:=${PUBLISH_DEPTH}"
)
if [[ -n "$DEVICE" ]]; then
  ROS_ARGS+=( -p "device:=${DEVICE}" )
fi

exec "$WORKER_PYTHON" "$NODE_SCRIPT" "${ROS_ARGS[@]}"
