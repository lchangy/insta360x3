#!/usr/bin/env bash

# Isolated Gazebo semantic-map simulation -> six cube faces -> ERP -> DA360.
# This launcher never stops or starts the real Insta360 pipeline.

set -Eeuo pipefail

WORKSPACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROS_SETUP="/opt/ros/humble/setup.bash"
SEMANTIC_MAPPING_WS="${SEMANTIC_MAPPING_WS:-/home/orion/Desktop/semantic_mapping_ws}"
MAP_DIR="${DA360_SIM_MAP_DIR:-${SEMANTIC_MAPPING_WS}/maps/001}"
WORLD_FILE="${DA360_SIM_WORLD:-${MAP_DIR}/random_start.sdf}"
MODEL_DIR="${SEMANTIC_MAPPING_WS}/src/room_gazebo_worlds/models"
INSTALLED_MODEL_DIR="${SEMANTIC_MAPPING_WS}/install/room_gazebo_worlds/share/room_gazebo_worlds/models"
DA360_PYTHON="${DA360_PYTHON:-${WORKSPACE_DIR}/.venv/bin/python}"
DA360_MODEL_PATH="${DA360_MODEL_PATH:-${WORKSPACE_DIR}/insta360_ros_driver/da360_runtime/checkpoints/DA360_small.pth}"
FACE_SIZE="${DA360_SIM_FACE_SIZE:-360}"
PUBLISH_RATE="${DA360_SIM_RATE:-5}"
POINT_STRIDE="${DA360_SIM_POINT_STRIDE:-4}"
ROS_DOMAIN="${DA360_SIM_ROS_DOMAIN_ID:-42}"
OPEN_GUI="${DA360_SIM_GUI:-false}"
OPEN_RVIZ="${DA360_SIM_RVIZ:-false}"
RUN_DA360=true

RUN_DIR=""
GAZEBO_PID=""
PANORAMA_PID=""
DA360_PID=""
RVIZ_PID=""
declare -a BRIDGE_PIDS=()

usage() {
  cat <<'EOF'
Usage: ./start_da360_gazebo_sim.sh [options]

Uses semantic_mapping's 3-D houseworld and runs an isolated six-face camera
simulation into the bundled DA360 small model.

Options:
  --gui                 Open Gazebo GUI (default: headless server).
  --rviz                Open the DA360 RViz view.
  --no-da360            Only publish the simulated ERP image.
  --map-dir PATH        Semantic map directory (default: semantic_mapping_ws/maps/001).
  --world PATH          SDF world (default: MAP_DIR/random_start.sdf).
  --face-size N         Cube-face size; 360 produces 1440x720 ERP (default: 360).
  --rate HZ             Simulated camera / ERP publish rate (default: 5).
  --point-stride N      DA360 PointCloud2 sampling stride (default: 4).
  --ros-domain-id N     Isolated ROS 2 domain (default: 42).
  -h, --help            Show this help.

Environment overrides:
  SEMANTIC_MAPPING_WS, DA360_PYTHON, DA360_MODEL_PATH, DA360_SIM_*.
EOF
}

die() {
  echo "[da360-gazebo-sim] ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gui) OPEN_GUI=true; shift ;;
    --rviz) OPEN_RVIZ=true; shift ;;
    --no-da360) RUN_DA360=false; shift ;;
    --map-dir) [[ $# -ge 2 ]] || die "--map-dir requires a path"; MAP_DIR="$2"; shift 2 ;;
    --world) [[ $# -ge 2 ]] || die "--world requires a path"; WORLD_FILE="$2"; shift 2 ;;
    --face-size) [[ $# -ge 2 ]] || die "--face-size requires an integer"; FACE_SIZE="$2"; shift 2 ;;
    --rate) [[ $# -ge 2 ]] || die "--rate requires a number"; PUBLISH_RATE="$2"; shift 2 ;;
    --point-stride) [[ $# -ge 2 ]] || die "--point-stride requires an integer"; POINT_STRIDE="$2"; shift 2 ;;
    --ros-domain-id) [[ $# -ge 2 ]] || die "--ros-domain-id requires an integer"; ROS_DOMAIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

[[ "$FACE_SIZE" =~ ^[1-9][0-9]*$ ]] || die "face size must be a positive integer"
[[ "$POINT_STRIDE" =~ ^[1-9][0-9]*$ ]] || die "point stride must be a positive integer"
[[ "$ROS_DOMAIN" =~ ^[0-9]+$ ]] || die "ROS domain id must be an integer"
[[ -r "$ROS_SETUP" ]] || die "ROS setup not found: $ROS_SETUP"
[[ -r "$WORLD_FILE" ]] || die "SDF world not found: $WORLD_FILE"
[[ -d "$MODEL_DIR" ]] || die "semantic_mapping model directory not found: $MODEL_DIR"
[[ -r "$DA360_MODEL_PATH" ]] || die "DA360 checkpoint not found: $DA360_MODEL_PATH"
command -v ign >/dev/null 2>&1 || die "ign gazebo is not installed"
command -v ros2 >/dev/null 2>&1 || die "ROS 2 is not installed"

set +u
source "$ROS_SETUP"
if [[ -r "${SEMANTIC_MAPPING_WS}/install/setup.bash" ]]; then
  source "${SEMANTIC_MAPPING_WS}/install/setup.bash"
fi
set -u

export ROS_DOMAIN_ID="$ROS_DOMAIN"
export ROS_LOCALHOST_ONLY="1"
export GZ_PARTITION="${GZ_PARTITION:-instax3_da360_sim}"
export IGN_PARTITION="${IGN_PARTITION:-$GZ_PARTITION}"
export GZ_SIM_RESOURCE_PATH="${MODEL_DIR}:${INSTALLED_MODEL_DIR}${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"
export IGN_GAZEBO_RESOURCE_PATH="${MODEL_DIR}:${INSTALLED_MODEL_DIR}${IGN_GAZEBO_RESOURCE_PATH:+:${IGN_GAZEBO_RESOURCE_PATH}}"

if [[ ! -x "$DA360_PYTHON" ]]; then
  die "DA360 Python not found: $DA360_PYTHON"
fi
if [[ "$RUN_DA360" == true ]]; then
  "$DA360_PYTHON" -c 'import cv2, numpy, rclpy, torch' >/dev/null 2>&1 \
    || die "DA360 Python lacks cv2, numpy, rclpy, or torch: $DA360_PYTHON"
fi

RUN_DIR="$(mktemp -d /tmp/instax3-da360-gazebo.XXXXXX)"
GENERATED_WORLD="${RUN_DIR}/world.sdf"
GENERATOR="${WORKSPACE_DIR}/insta360_ros_driver/scripts/generate_da360_gazebo_world.py"
PANORAMA_NODE="${WORKSPACE_DIR}/insta360_ros_driver/scripts/gazebo_equirectangular.py"
DA360_NODE="${WORKSPACE_DIR}/insta360_ros_driver/da360_runtime/ros2_da360/ros2_da360/da360_realtime_node.py"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "$RVIZ_PID" "$DA360_PID" "$PANORAMA_PID" "${BRIDGE_PIDS[@]}" "$GAZEBO_PID"; do
    [[ -n "$pid" ]] || continue
    kill -INT "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "$RVIZ_PID" "$DA360_PID" "$PANORAMA_PID" "${BRIDGE_PIDS[@]}" "$GAZEBO_PID"; do
    [[ -n "$pid" ]] || continue
    kill -TERM "$pid" 2>/dev/null || true
  done
  rm -rf "$RUN_DIR"
  exit "$status"
}

on_signal() {
  [[ -n "$GAZEBO_PID" ]] && kill -INT "$GAZEBO_PID" 2>/dev/null || true
}

trap cleanup EXIT
trap on_signal INT TERM

python3 "$GENERATOR" \
  --input-world "$WORLD_FILE" \
  --output-world "$GENERATED_WORLD" \
  --face-size "$FACE_SIZE" \
  --update-rate "$PUBLISH_RATE"

echo "[da360-gazebo-sim] semantic workspace: $SEMANTIC_MAPPING_WS"
echo "[da360-gazebo-sim] map directory: $MAP_DIR"
echo "[da360-gazebo-sim] source world: $WORLD_FILE"
echo "[da360-gazebo-sim] ROS_DOMAIN_ID=$ROS_DOMAIN (isolated from the real pipeline)"
echo "[da360-gazebo-sim] six faces: ${FACE_SIZE}x${FACE_SIZE}; ERP: $((FACE_SIZE * 4))x$((FACE_SIZE * 2))"

if [[ "$OPEN_GUI" == true ]]; then
  ign gazebo -r "$GENERATED_WORLD" >"${RUN_DIR}/gazebo.log" 2>&1 &
else
  ign gazebo -r -s --headless-rendering "$GENERATED_WORLD" >"${RUN_DIR}/gazebo.log" 2>&1 &
fi
GAZEBO_PID=$!
sleep 5
kill -0 "$GAZEBO_PID" 2>/dev/null || {
  tail -80 "${RUN_DIR}/gazebo.log" >&2 || true
  die "Gazebo exited during startup"
}

for face in front right back left top bottom; do
  ros2 run ros_gz_image image_bridge "/da360_sim/cubemap/${face}" \
    >"${RUN_DIR}/bridge_${face}.log" 2>&1 &
  BRIDGE_PIDS+=("$!")
done

PYTHONUNBUFFERED=1 "$DA360_PYTHON" "$PANORAMA_NODE" \
  --ros-args \
  -p face_size:="$FACE_SIZE" \
  -p publish_rate:="$PUBLISH_RATE" \
  -p output_topic:=/equirectangular/image \
  -p frame_id:=camera_frame \
  -p topic_prefix:=/da360_sim/cubemap \
  >"${RUN_DIR}/panorama.log" 2>&1 &
PANORAMA_PID=$!

if [[ "$RUN_DA360" == true ]]; then
  PYTHONUNBUFFERED=1 "$DA360_PYTHON" "$DA360_NODE" \
    --ros-args \
    -p repo_root:="${WORKSPACE_DIR}/insta360_ros_driver/da360_runtime" \
    -p worker_python:="$DA360_PYTHON" \
    -p model_path:="$DA360_MODEL_PATH" \
    -p input_topic:=/equirectangular/image \
    -p depth_topic:=/da360/depth \
    -p pointcloud_topic:=/da360/points \
    -p frame_id:=camera_frame \
    -p point_stride:="$POINT_STRIDE" \
    >"${RUN_DIR}/da360.log" 2>&1 &
  DA360_PID=$!
fi

if [[ "$OPEN_RVIZ" == true ]]; then
  rviz2 -d "${WORKSPACE_DIR}/insta360_ros_driver/rviz/da360_realtime.rviz" \
    >"${RUN_DIR}/rviz.log" 2>&1 &
  RVIZ_PID=$!
fi

wait_for_topic() {
  local topic="$1"
  local label="$2"
  local timeout_seconds="$3"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$PANORAMA_PID" 2>/dev/null; then
      tail -80 "${RUN_DIR}/panorama.log" >&2 || true
      die "panorama node exited while waiting for $label"
    fi
    if timeout 3s ros2 topic echo "$topic" --once --field header >/dev/null 2>&1; then
      echo "[da360-gazebo-sim] ready: $label ($topic)"
      return 0
    fi
  done
  die "timed out waiting for $label on $topic"
}

wait_for_topic /equirectangular/image "1440x720 ERP image" 45
if [[ "$RUN_DA360" == true ]]; then
  wait_for_topic /da360/points "DA360 point cloud" 120
fi

echo "[da360-gazebo-sim] running. Logs: $RUN_DIR"
echo "[da360-gazebo-sim] ROS topics: /equirectangular/image /da360/depth /da360/points"
echo "[da360-gazebo-sim] inspect rates with: ROS_DOMAIN_ID=$ROS_DOMAIN ros2 topic hz /equirectangular/image"
echo "[da360-gazebo-sim] press Ctrl-C to stop only this simulation"

wait "$GAZEBO_PID"
