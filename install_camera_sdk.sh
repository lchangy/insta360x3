#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PACKAGE_DIR="${ROOT_DIR}/insta360_ros_driver"

usage() {
  echo "Usage: ./install_camera_sdk.sh /path/to/extracted/CameraSDK" >&2
}

[[ $# -eq 1 ]] || { usage; exit 2; }
SDK_SOURCE="$(realpath -e -- "$1")"
[[ -d "$SDK_SOURCE" ]] || { echo "SDK directory not found: $1" >&2; exit 1; }

SDK_LIBRARY="$(find "$SDK_SOURCE" -type f -name 'libCameraSDK.so' -print -quit)"
CAMERA_HEADER="$(find "$SDK_SOURCE" -type f -path '*/camera/camera.h' -print -quit)"
STREAM_HEADER="$(find "$SDK_SOURCE" -type f -path '*/stream/stream_delegate.h' -print -quit)"

[[ -n "$SDK_LIBRARY" ]] || { echo "libCameraSDK.so was not found under $SDK_SOURCE" >&2; exit 1; }
[[ -n "$CAMERA_HEADER" ]] || { echo "camera/camera.h was not found under $SDK_SOURCE" >&2; exit 1; }
[[ -n "$STREAM_HEADER" ]] || { echo "stream/stream_delegate.h was not found under $SDK_SOURCE" >&2; exit 1; }

CAMERA_INCLUDE_DIR="$(dirname -- "$CAMERA_HEADER")"
STREAM_INCLUDE_DIR="$(dirname -- "$STREAM_HEADER")"

mkdir -p "${PACKAGE_DIR}/lib" "${PACKAGE_DIR}/include/camera" "${PACKAGE_DIR}/include/stream"
install -m 0755 "$SDK_LIBRARY" "${PACKAGE_DIR}/lib/libCameraSDK.so"
install -m 0644 "${CAMERA_INCLUDE_DIR}"/*.h "${PACKAGE_DIR}/include/camera/"
install -m 0644 "${STREAM_INCLUDE_DIR}"/*.h "${PACKAGE_DIR}/include/stream/"

echo "CameraSDK installed into ${PACKAGE_DIR}"
echo "Next: source /opt/ros/humble/setup.bash && colcon build --symlink-install --packages-select insta360_ros_driver"

