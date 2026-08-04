# Third-party notices

This repository combines components with different licenses. The repository as
a whole must not be described as if every component were covered only by the
ROS driver's Apache-2.0 license.

## ROS driver

The upstream `insta360_ros_driver` source is licensed under Apache License 2.0.
The license text is stored at `insta360_ros_driver/LICENSE.txt`.

## DA360 runtime and model weights

The bundled DA360 runtime is derived from the Insta360 Research Team DA360
project. Its MIT license is stored at
`insta360_ros_driver/da360_runtime/LICENSE`.

DA360 model weights are not distributed in this repository. Users download
them directly from the official DA360 project and must follow its terms.

The runtime includes model architecture code derived from Depth Anything V2 /
DINOv2. Keep the upstream copyright headers and verify their upstream license
requirements before redistributing a modified runtime.

## Cubemap projection

The Cubemap projection was adapted from DAP material. Its CC BY-NC 4.0 license
is stored at `insta360_ros_driver/third_party_licenses/DAP_LICENSE`. This license
restricts use to non-commercial purposes and requires attribution and change
notices.

## Ultralytics YOLO26 depth

The YOLO26s-depth runtime is provided by the pinned upstream Ultralytics
source revision in `pyproject.toml` / `uv.lock` and is licensed under AGPL-3.0.
The `yolo26s-depth.pt` model weights are not distributed in this repository;
Ultralytics resolves them separately at runtime. Review the upstream license
and distribution terms before deploying the combined application.

## Insta360 CameraSDK

`libCameraSDK.so` and the CameraSDK headers are proprietary SDK artifacts and
are intentionally excluded by the root `.gitignore`. Users must obtain them
from Insta360 and comply with their own SDK agreement. Do not redistribute
these files unless you have explicit permission.
