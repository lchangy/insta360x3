#!/usr/bin/env bash

# One-repository launcher: Insta360 X3 -> panorama -> DA360 depth/point cloud.

set -Eeo pipefail

WORKSPACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROS_SETUP="/opt/ros/humble/setup.bash"
INSTALL_SETUP="${WORKSPACE_DIR}/install/setup.bash"
DRIVER_SETUP="${WORKSPACE_DIR}/insta360_ros_driver/setup.sh"
SDK_LIB_DIR="${WORKSPACE_DIR}/insta360_ros_driver/lib"
LOCAL_DA360_PYTHON="${WORKSPACE_DIR}/.venv/bin/python"
SYSTEM_PYTHON="/usr/bin/python3"

MODEL_PATH=""
POINT_STRIDE=4
OPEN_RVIZ=true
ENABLE_CUBEMAP=true
CUBEMAP_GUI=true
CUBEMAP_FACE_SIZE=360
CALIBRATION=false
LAUNCH_PID=""
CLEANING_UP=0

usage() {
  cat <<'EOF'
Usage: ./start_pointcloud_pipeline.sh [options]

Starts the complete Insta360 X3 -> panorama -> DA360 point-cloud pipeline.
The DA360 runtime is bundled; download DA360_small.pth as described in README.md.

Options:
  --no-rviz              Do not start RViz.
  --no-cubemap           Do not publish cubemap views.
  --cubemap-no-gui       Publish cubemap topics without opening its window.
  --cubemap-face-size N  Cubemap face width/height (default: 360).
  --calibrate            Replace the C++ panorama node with the live calibration UI.
  --model-path PATH      Override the bundled DA360_small.pth checkpoint.
  --point-stride N       Point-cloud sampling stride (default: 4).
  -h, --help             Show this help.

Environment:
  DA360_PYTHON           Python with rclpy, OpenCV, NumPy, and CUDA Torch.
                         Defaults to ./.venv/bin/python, then /usr/bin/python3.
EOF
}

die() {
  echo "[pointcloud-pipeline] ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-rviz)
      OPEN_RVIZ=false
      shift
      ;;
    --no-cubemap)
      ENABLE_CUBEMAP=false
      CUBEMAP_GUI=false
      shift
      ;;
    --cubemap-no-gui)
      CUBEMAP_GUI=false
      shift
      ;;
    --cubemap-face-size)
      [[ $# -ge 2 ]] || die "--cubemap-face-size requires a positive integer"
      CUBEMAP_FACE_SIZE="$2"
      shift 2
      ;;
    --calibrate)
      CALIBRATION=true
      shift
      ;;
    --model-path)
      [[ $# -ge 2 ]] || die "--model-path requires a path"
      MODEL_PATH="$2"
      shift 2
      ;;
    --point-stride)
      [[ $# -ge 2 ]] || die "--point-stride requires a positive integer"
      POINT_STRIDE="$2"
      shift 2
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
[[ "$CUBEMAP_FACE_SIZE" =~ ^[1-9][0-9]*$ ]] || die "cubemap face size must be a positive integer"
[[ -r "$ROS_SETUP" ]] || die "ROS 2 Humble setup not found: $ROS_SETUP"
[[ -r "$INSTALL_SETUP" ]] || die "workspace is not built; missing $INSTALL_SETUP"

source "$ROS_SETUP"
source "$INSTALL_SETUP"
set -u

export PATH="/usr/bin:/bin:/opt/ros/humble/bin:${PATH}"
export LD_LIBRARY_PATH="${SDK_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

PACKAGE_PREFIX="$(ros2 pkg prefix insta360_ros_driver 2>/dev/null)" \
  || die "insta360_ros_driver is not available; rebuild this workspace"
PACKAGE_SHARE="${PACKAGE_PREFIX}/share/insta360_ros_driver"
RUNTIME_ROOT="${PACKAGE_SHARE}/da360_runtime"
LAUNCH_FILE="${PACKAGE_SHARE}/launch/pointcloud_pipeline.launch.py"

[[ -r "$LAUNCH_FILE" ]] || die "pipeline launch is not installed; rebuild this workspace"
[[ -d "$RUNTIME_ROOT/networks" ]] || die "bundled DA360 runtime is missing: $RUNTIME_ROOT"

if [[ -n "$MODEL_PATH" ]]; then
  if [[ "$MODEL_PATH" != /* ]]; then
    MODEL_PATH="${WORKSPACE_DIR}/${MODEL_PATH}"
  fi
else
  MODEL_PATH="${RUNTIME_ROOT}/checkpoints/DA360_small.pth"
fi
[[ -r "$MODEL_PATH" ]] || die \
  "DA360 checkpoint not found: $MODEL_PATH; download DA360_small.pth as described in README.md"

if [[ -n "${DA360_PYTHON:-}" ]]; then
  WORKER_PYTHON="$DA360_PYTHON"
elif [[ -x "$LOCAL_DA360_PYTHON" ]]; then
  WORKER_PYTHON="$LOCAL_DA360_PYTHON"
else
  WORKER_PYTHON="$SYSTEM_PYTHON"
fi
[[ -x "$WORKER_PYTHON" ]] || die \
  "DA360 Python not found; create ${LOCAL_DA360_PYTHON} or set DA360_PYTHON"
"$WORKER_PYTHON" -c 'import cv2, numpy, rclpy, torch, torchvision' >/dev/null 2>&1 \
  || die "DA360 Python is missing cv2, numpy, rclpy, torch, or torchvision: $WORKER_PYTHON; see DEPLOYMENT.md"

ensure_udev_access() {
  local usb_node=""

  if command -v lsusb >/dev/null 2>&1 && ! lsusb -d 2e1a:0002 >/dev/null 2>&1; then
    die "Insta360 X3 is not detected; power it on, select USB Mode=Android, and reconnect it"
  fi

  if [[ -L /dev/insta || -e /dev/insta ]]; then
    usb_node="$(readlink -f /dev/insta 2>/dev/null || true)"
  fi
  if [[ -z "$usb_node" || ! -e "$usb_node" ]]; then
    [[ -x "$DRIVER_SETUP" ]] || die "USB setup script not found: $DRIVER_SETUP"
    echo "[pointcloud-pipeline] /dev/insta is missing; installing the udev rule"
    "$DRIVER_SETUP"
    usb_node="$(readlink -f /dev/insta 2>/dev/null || true)"
  fi

  [[ -n "$usb_node" && -e "$usb_node" ]] || die "USB node /dev/insta was not created"
  [[ -r "$usb_node" && -w "$usb_node" ]] || die "USB node is not readable/writable: $usb_node"
  echo "[pointcloud-pipeline] USB: $usb_node ($(stat -c '%U:%G %A' "$usb_node"))"
}

collect_pipeline_pids() {
  local pattern
  local -a patterns=(
    "/opt/ros/humble/bin/ros2 launch insta360_ros_driver bringup.launch.xml"
    "/opt/ros/humble/bin/ros2 launch insta360_ros_driver pointcloud_pipeline.launch.py"
    "${PACKAGE_PREFIX}/lib/insta360_ros_driver/insta360_ros_driver"
    "${PACKAGE_PREFIX}/lib/insta360_ros_driver/decoder"
    "${PACKAGE_PREFIX}/lib/insta360_ros_driver/equirectangular_cpp"
    "${PACKAGE_PREFIX}/lib/insta360_ros_driver/calibrate.py"
    "${PACKAGE_PREFIX}/share/insta360_ros_driver/da360_runtime/ros2_da360/ros2_da360/da360_realtime_node.py"
    "${PACKAGE_PREFIX}/share/insta360_ros_driver/da360_runtime/ros2_da360/ros2_da360/da360_inference_worker.py"
    "${PACKAGE_PREFIX}/lib/insta360_ros_driver/ros_cubemap_view.py"
    "tools/ros_cubemap_view.py"
    "rviz2 -d .*da360_realtime.rviz"
  )
  for pattern in "${patterns[@]}"; do
    pgrep -f -- "$pattern" 2>/dev/null || true
  done | awk -v self="$$" '$1 != self' | sort -nu
}

stop_pipeline_processes() {
  local -a pids=()
  local pid
  mapfile -t pids < <(collect_pipeline_pids)
  [[ ${#pids[@]} -gt 0 ]] || return 0

  echo "[pointcloud-pipeline] stopping existing pipeline processes: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in {1..20}; do
    local still_running=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        still_running=1
        break
      fi
    done
    [[ $still_running -eq 0 ]] && return 0
    sleep 0.1
  done
  for pid in "${pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}

wait_for_message() {
  local topic="$1"
  local timeout_seconds="$2"
  local label="$3"
  local deadline=$((SECONDS + timeout_seconds))

  while (( SECONDS < deadline )); do
    if [[ -n "$LAUNCH_PID" ]] && ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      die "launch exited while waiting for $label"
    fi
    if timeout 2s ros2 topic echo "$topic" --once --field header >/dev/null 2>&1; then
      echo "[pointcloud-pipeline] ready: $label ($topic)"
      return 0
    fi
  done
  die "timed out after ${timeout_seconds}s waiting for $label on $topic"
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
  stop_pipeline_processes
  exit "$status"
}

on_signal() {
  if [[ -n "$LAUNCH_PID" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
    kill -INT "$LAUNCH_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT
trap on_signal INT TERM

ensure_udev_access
stop_pipeline_processes

echo "[pointcloud-pipeline] repository: $WORKSPACE_DIR"
echo "[pointcloud-pipeline] bundled runtime: $RUNTIME_ROOT"
echo "[pointcloud-pipeline] model: $MODEL_PATH"
echo "[pointcloud-pipeline] launching camera, panorama, calibration=$CALIBRATION, cubemap=$ENABLE_CUBEMAP, DA360, point cloud, rviz=$OPEN_RVIZ"

ros2 launch insta360_ros_driver pointcloud_pipeline.launch.py \
  worker_python:="$WORKER_PYTHON" \
  model_path:="$MODEL_PATH" \
  point_stride:="$POINT_STRIDE" \
  cubemap:="$ENABLE_CUBEMAP" \
  cubemap_gui:="$CUBEMAP_GUI" \
  cubemap_face_size:="$CUBEMAP_FACE_SIZE" \
  calibration:="$CALIBRATION" \
  rviz:="$OPEN_RVIZ" &
LAUNCH_PID=$!

wait_for_message /equirectangular/image 20 "equirectangular panorama"
if [[ "$ENABLE_CUBEMAP" == true ]]; then
  wait_for_message /cubemap/horizontal/image 20 "Cubemap horizontal mosaic"
fi
wait_for_message /da360/points 90 "DA360 point cloud"

echo "[pointcloud-pipeline] pipeline is running; press Ctrl-C to stop everything"
set +e
wait "$LAUNCH_PID"
launch_status=$?
set -e
exit "$launch_status"
