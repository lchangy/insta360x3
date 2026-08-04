#!/usr/bin/env bash

# One-command launcher for the independent UniDepthV2-Small point-cloud node.
#
# This script deliberately starts only unidepthv2_pointcloud.launch.py.  The
# camera, panorama, Cubemap, DA360, and YOLO processes must already be running
# and are never stopped or modified by this script.

# ROS 2 setup scripts reference optional variables before defining them, so
# enable nounset only after both setup files have been sourced.
set -Ee -o pipefail

WORKSPACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# ------------------------- local configuration -------------------------
# Keep all runtime configuration here so normal use is just:
#   ./start_unidepthv2_pointcloud.sh
ROS_SETUP="/opt/ros/humble/setup.bash"
INSTALL_SETUP="${WORKSPACE_DIR}/install/setup.bash"
UNIDEPTH_PYTHON="${WORKSPACE_DIR}/.venv/bin/python"
UNIDEPTH_REPO="${WORKSPACE_DIR}/.cache/unidepth"
UNIDEPTH_GIT_URL="https://github.com/lpiccinelli-eth/unidepth.git"
# Local UniDepthV2-Small checkpoint downloaded by the user.  UniDepth's
# from_pretrained() accepts a local Hugging Face-style directory directly.
MODEL_NAME="/home/orion/Downloads/unidepth"
UNIDEPTH_DEVICE=""
POINT_STRIDE=2
RESOLUTION_LEVEL=5
FACE_FOV_DEG="90.0"
OPEN_RVIZ=true
MODEL_WAIT_SECONDS=300
CUBEMAP_WAIT_SECONDS=20
ROS_LOCALHOST_ONLY_VALUE=1
# ------------------------------------------------------------------------

LAUNCH_PID=""
CLEANING_UP=0

usage() {
  cat <<'EOF'
Usage: ./start_unidepthv2_pointcloud.sh

Starts the independent UniDepthV2-Small node and its RViz configuration.
The existing camera/Cubemap pipeline must already publish:
  /cubemap/front/image
  /cubemap/right/image
  /cubemap/back/image
  /cubemap/left/image

Edit the configuration block at the top of this script to change the Python,
UniDepth checkout, model, device, point stride, or RViz setting.
EOF
}

die() {
  echo "[unidepthv2] ERROR: $*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
[[ $# -eq 0 ]] || die "this launcher has no command-line options; edit its configuration block"

[[ -r "$ROS_SETUP" ]] || die "ROS 2 setup not found: $ROS_SETUP"
[[ -r "$INSTALL_SETUP" ]] || die "workspace is not built; missing $INSTALL_SETUP"
[[ -x "$UNIDEPTH_PYTHON" ]] || die "Python not found: $UNIDEPTH_PYTHON"
command -v git >/dev/null 2>&1 || die "git is required to fetch UniDepth on first run"
command -v timeout >/dev/null 2>&1 || die "timeout is required by this launcher"

source "$ROS_SETUP"
source "$INSTALL_SETUP"
set -u

export PATH="/usr/bin:/bin:/opt/ros/humble/bin:${PATH}"
export PYTHONUNBUFFERED=1
export ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE"
export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
export PYTHONPATH="${UNIDEPTH_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

PACKAGE_PREFIX="$(ros2 pkg prefix insta360_ros_driver 2>/dev/null)" \
  || die "insta360_ros_driver is not installed; rebuild the workspace"
LAUNCH_FILE="${PACKAGE_PREFIX}/share/insta360_ros_driver/launch/unidepthv2_pointcloud.launch.py"
[[ -r "$LAUNCH_FILE" ]] || die "UniDepth launch is not installed; rebuild the workspace"

if [[ ! -f "${UNIDEPTH_REPO}/unidepth/models/unidepthv2/unidepthv2.py" ]]; then
  if [[ -e "$UNIDEPTH_REPO" ]]; then
    die "UniDepth path exists but is not a valid checkout: $UNIDEPTH_REPO"
  fi
  echo "[unidepthv2] first run: cloning UniDepth to $UNIDEPTH_REPO"
  mkdir -p "$(dirname -- "$UNIDEPTH_REPO")"
  git clone --depth 1 "$UNIDEPTH_GIT_URL" "$UNIDEPTH_REPO"
fi

"$UNIDEPTH_PYTHON" -c \
  'import rclpy, torch, numpy, einops, huggingface_hub, timm' \
  || die "Python environment lacks ROS 2/Torch/UniDepth runtime dependencies: $UNIDEPTH_PYTHON"
"$UNIDEPTH_PYTHON" -c \
  'import sys; sys.path.insert(0, sys.argv[1]); from unidepth.models import UniDepthV2' \
  "$UNIDEPTH_REPO" \
  || die "UniDepth cannot be imported from $UNIDEPTH_REPO"

if [[ "$MODEL_NAME" == /* || "$MODEL_NAME" == ./* ]]; then
  [[ -r "$MODEL_NAME/config.json" ]] \
    || die "local UniDepth model is missing config.json: $MODEL_NAME"
  [[ -r "$MODEL_NAME/model.safetensors" || -r "$MODEL_NAME/pytorch_model.bin" ]] \
    || die "local UniDepth model has no model.safetensors or pytorch_model.bin: $MODEL_NAME"
fi

wait_for_topic() {
  local topic="$1"
  local label="$2"
  local timeout_seconds="$3"
  local deadline=$((SECONDS + timeout_seconds))

  while (( SECONDS < deadline )); do
    if [[ -n "$LAUNCH_PID" ]] && ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      die "UniDepth launch exited while waiting for $label"
    fi
    if timeout 2s ros2 topic echo "$topic" --once --field header >/dev/null 2>&1; then
      echo "[unidepthv2] ready: $label ($topic)"
      return 0
    fi
  done
  die "timed out after ${timeout_seconds}s waiting for $label on $topic"
}

wait_for_node() {
  local deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    if [[ -n "$LAUNCH_PID" ]] && ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      die "UniDepth launch exited before the node came up"
    fi
    if ros2 node list 2>/dev/null | grep -qx '/unidepthv2_pointcloud'; then
      echo "[unidepthv2] node is running: /unidepthv2_pointcloud"
      return 0
    fi
    sleep 1
  done
  die "UniDepth node did not appear; check the launch output above"
}

cleanup() {
  local status=$?
  [[ $CLEANING_UP -eq 0 ]] || return
  CLEANING_UP=1
  trap - EXIT INT TERM
  if [[ -n "$LAUNCH_PID" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
    kill -INT "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
  fi
  exit "$status"
}

on_signal() {
  if [[ -n "$LAUNCH_PID" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
    kill -INT "$LAUNCH_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT
trap on_signal INT TERM

echo "[unidepthv2] workspace: $WORKSPACE_DIR"
echo "[unidepthv2] Python: $UNIDEPTH_PYTHON"
echo "[unidepthv2] UniDepth: $UNIDEPTH_REPO"
echo "[unidepthv2] model: $MODEL_NAME"
echo "[unidepthv2] waiting for the existing Cubemap stream"
for face in front right back left; do
  wait_for_topic \
    "/cubemap/${face}/image" \
    "Cubemap ${face} image" \
    "$CUBEMAP_WAIT_SECONDS"
done

LAUNCH_ARGS=(
  "worker_python:=$UNIDEPTH_PYTHON"
  "model_name:=$MODEL_NAME"
  "unidepth_repo:=$UNIDEPTH_REPO"
  "point_stride:=$POINT_STRIDE"
  "resolution_level:=$RESOLUTION_LEVEL"
  "face_fov_deg:=$FACE_FOV_DEG"
  "rviz:=$OPEN_RVIZ"
)
# An empty device means automatic CUDA/CPU selection inside the node.  Do not
# pass the empty ROS launch argument as `device:=`, which launch rejects.
if [[ -n "$UNIDEPTH_DEVICE" ]]; then
  LAUNCH_ARGS+=("device:=$UNIDEPTH_DEVICE")
fi

ros2 launch insta360_ros_driver unidepthv2_pointcloud.launch.py \
  "${LAUNCH_ARGS[@]}" &
LAUNCH_PID=$!

wait_for_node
for face in front right back left; do
  wait_for_topic \
    "/unidepthv2/${face}/points" \
    "UniDepth ${face} point cloud" \
    "$MODEL_WAIT_SECONDS"
done

echo "[unidepthv2] running; press Ctrl-C to stop only UniDepth and its RViz"
set +e
wait "$LAUNCH_PID"
launch_status=$?
set -e
exit "$launch_status"
