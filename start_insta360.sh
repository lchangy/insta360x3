#!/usr/bin/env bash

# One-command launcher for the Insta360 X3 ROS2 driver.
#
# Examples:
#   ./start_insta360.sh                         # default repository bringup
#   ./start_insta360.sh --view                  # bringup + rqt raw image view
#   ./start_insta360.sh --compressed-view       # bringup + compressed transport view
#   ./start_insta360.sh --equirectangular       # also publish /equirectangular/image
#   ./start_insta360.sh --equirectangular --surround

# ROS setup scripts use a few optional variables without initializing them.
# Enable nounset only after those setup files have been sourced below.
set -Eeo pipefail

WORKSPACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROS_SETUP="/opt/ros/humble/setup.bash"
INSTALL_SETUP="${WORKSPACE_DIR}/install/setup.bash"
DRIVER_SETUP="${WORKSPACE_DIR}/insta360_ros_driver/setup.sh"
SDK_LIB_DIR="${WORKSPACE_DIR}/insta360_ros_driver/lib"
VIEW_LOG="${WORKSPACE_DIR}/log/insta360_view.log"

OPEN_VIEW=0
OPEN_COMPRESSED_VIEW=0
ENABLE_EQUIRECTANGULAR=0
ENABLE_SURROUND=0
KEEP_EXISTING=0

usage() {
  cat <<'EOF'
Usage: ./start_insta360.sh [options]

Starts one clean instance of the Insta360 ROS2 bringup.

Options:
  --view                 Open rqt_image_view on /dual_fisheye/image.
  --compressed-view      View the H.264 stream after the driver's decoder.
  --equirectangular      Also publish /equirectangular/image.
  --surround             Publish four views; implies --equirectangular.
  --keep-existing        Do not stop existing Insta360 ROS processes.
  -h, --help             Show this help.
EOF
}

die() {
  echo "[insta360] ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --view)
      OPEN_VIEW=1
      shift
      ;;
    --compressed-view)
      OPEN_COMPRESSED_VIEW=1
      shift
      ;;
    --equirectangular)
      ENABLE_EQUIRECTANGULAR=1
      shift
      ;;
    --surround)
      ENABLE_SURROUND=1
      ENABLE_EQUIRECTANGULAR=1
      shift
      ;;
    --keep-existing)
      KEEP_EXISTING=1
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

[[ -r "$ROS_SETUP" ]] || die "ROS 2 Humble setup not found: $ROS_SETUP"
[[ -r "$INSTALL_SETUP" ]] || die "workspace is not built; missing $INSTALL_SETUP"

# The SDK/driver are C++, but sourcing the system ROS environment also keeps
# ros2 from accidentally selecting a Conda Python during launcher discovery.
source "$ROS_SETUP"
source "$INSTALL_SETUP"
set -u
export PATH="/usr/bin:/bin:/opt/ros/humble/bin:${PATH}"
export LD_LIBRARY_PATH="${SDK_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

ensure_udev_access() {
  local usb_node=""

  if command -v lsusb >/dev/null 2>&1 && ! lsusb -d 2e1a:0002 >/dev/null 2>&1; then
    die "Insta360 X3 (2e1a:0002) is not enumerated. Turn on the camera, select USB Mode=Android, and reconnect it."
  fi

  if [[ -L /dev/insta || -e /dev/insta ]]; then
    usb_node="$(readlink -f /dev/insta 2>/dev/null || true)"
  fi

  if [[ -z "$usb_node" || ! -e "$usb_node" ]]; then
    [[ -x "$DRIVER_SETUP" ]] || die "USB setup script not found: $DRIVER_SETUP"
    echo "[insta360] /dev/insta is missing; installing the udev rule (sudo may ask for your password)..."
    "$DRIVER_SETUP"
    usb_node="$(readlink -f /dev/insta 2>/dev/null || true)"
  fi

  [[ -n "$usb_node" && -e "$usb_node" ]] || die "USB node /dev/insta was not created"
  [[ -r "$usb_node" && -w "$usb_node" ]] || die "USB node is not readable/writable: $usb_node"
  echo "[insta360] USB: $usb_node ($(stat -c '%U:%G %A' "$usb_node"))"
}

collect_pids() {
  local pattern pid
  local -a patterns=(
    "/opt/ros/humble/bin/ros2 launch insta360_ros_driver bringup.launch.xml"
    "${WORKSPACE_DIR}/install/insta360_ros_driver/lib/insta360_ros_driver/insta360_ros_driver"
    "${WORKSPACE_DIR}/install/insta360_ros_driver/lib/insta360_ros_driver/decoder"
    "${WORKSPACE_DIR}/install/insta360_ros_driver/lib/insta360_ros_driver/equirectangular_cpp"
    "${WORKSPACE_DIR}/install/insta360_ros_driver/lib/insta360_ros_driver/surround_views"
    "${WORKSPACE_DIR}/install/insta360_ros_driver/share/insta360_ros_driver/config/imu_filter.yaml"
    "/opt/ros/humble/lib/rqt_image_view/rqt_image_view /dual_fisheye/image"
    "/opt/ros/humble/lib/rqt_image_view/rqt_image_view /dual_fisheye/compressed_view"
    "/opt/ros/humble/lib/rqt_image_view/rqt_image_view --clear-config /dual_fisheye/image"
    "/opt/ros/humble/lib/rqt_image_view/rqt_image_view --clear-config /dual_fisheye/compressed_view"
    "/opt/ros/humble/lib/image_transport/republish compressed raw"
  )
  for pattern in "${patterns[@]}"; do
    while read -r pid; do
      [[ -n "$pid" && "$pid" != "$$" ]] && echo "$pid"
    done < <(pgrep -f -- "$pattern" || true)
  done | sort -nu
}

stop_existing() {
  local -a pids=()
  local pid
  mapfile -t pids < <(collect_pids)
  if [[ ${#pids[@]} -eq 0 ]]; then
    return
  fi

  echo "[insta360] stopping existing driver processes: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in {1..20}; do
    local still_running=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        still_running=1
        break
      fi
    done
    [[ $still_running -eq 0 ]] && return
    sleep 0.1
  done

  for pid in "${pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}

start_rqt_view() {
  mkdir -p "${WORKSPACE_DIR}/log"
  echo "[insta360] opening rqt_image_view on /dual_fisheye/image"
  nohup /usr/bin/python3 /opt/ros/humble/lib/rqt_image_view/rqt_image_view \
    --clear-config \
    /dual_fisheye/image >"$VIEW_LOG" 2>&1 < /dev/null &
}

start_compressed_view() {
  mkdir -p "${WORKSPACE_DIR}/log"
  echo "[insta360] opening decoded view of the H.264 compressed stream on /dual_fisheye/image"
  # /dual_fisheye/image/compressed contains H.264 bytes (format=h264), not
  # JPEG/PNG. The package's decoder node must produce sensor_msgs/Image first.
  nohup /usr/bin/python3 /opt/ros/humble/lib/rqt_image_view/rqt_image_view \
    --clear-config \
    /dual_fisheye/image >>"$VIEW_LOG" 2>&1 < /dev/null &
}

ensure_udev_access
if [[ $KEEP_EXISTING -eq 0 ]]; then
  stop_existing
fi

launch_args=("bringup.launch.xml")
if [[ $ENABLE_EQUIRECTANGULAR -eq 1 ]]; then
  launch_args+=("equirectangular:=true")
fi
if [[ $ENABLE_SURROUND -eq 1 ]]; then
  launch_args+=("surround:=true")
fi

if [[ $OPEN_COMPRESSED_VIEW -eq 1 ]]; then
  start_compressed_view
elif [[ $OPEN_VIEW -eq 1 ]]; then
  start_rqt_view
fi

echo "[insta360] workspace: $WORKSPACE_DIR"
echo "[insta360] launching: ros2 launch insta360_ros_driver ${launch_args[*]}"
exec ros2 launch insta360_ros_driver "${launch_args[@]}"
