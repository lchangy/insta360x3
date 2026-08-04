# Insta360 X3 到 DA360 / YOLO26s-depth 点云

本仓库已经内置 DA360 small 的最小推理运行时、Cubemap 节点和 YOLO26s-depth
点云节点。DA360 权重需按 [DEPLOYMENT.md](DEPLOYMENT.md) 从官方地址下载；
YOLO26s-depth 使用 Ultralytics 的 `yolo26s-depth.pt`，首次加载时会按模型名解析
本地缓存或下载，也可以传入自己的 `.pt` 路径。

## 构建

```bash
cd /path/to/instax3
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select insta360_ros_driver
```

DA360 推理需要带 CUDA PyTorch、OpenCV、NumPy 和 ROS 2 Python 包的解释器。
启动脚本优先使用本仓库的 `.venv/bin/python`，也可以显式指定：

```bash
export DA360_PYTHON=/path/to/venv/bin/python
```

如果启用 YOLO，还需要同一个解释器包含 `ultralytics`，因为 YOLO26s-depth 会由这条
启动流程拉起；默认关闭 YOLO 时不需要加载它。
`pyproject.toml` 已将 Ultralytics 固定到包含 `DepthModel` 的上游提交；新环境请先
执行 `uv sync --frozen`，不要只安装 PyPI 上的旧版 `ultralytics`。

## 一键启动

打开 Insta360 X3，将 USB 模式设为 **Android**，然后运行：

```bash
./start_pointcloud_pipeline.sh
```

流程发布以下话题：

- `/equirectangular/image`：1440×720 等距柱状全景图。
- `/da360/depth`：`32FC1` 相对深度图。
- `/da360/points`：带 RGB 的球面 `PointCloud2`。
- `/cubemap/front/image`、`right`、`back`、`left`：四个 360×360 视角。
- `/cubemap/horizontal/image`：`FRONT | RIGHT / BACK | LEFT` 拼图。
- `/yolo26s_depth/left/depth`、`/right/depth`：YOLO26s-depth 的 `32FC1` 米制深度图。
- `/yolo26s_depth/left/points`、`/right/points`：左/右 Cubemap 的带 RGB `PointCloud2`，
  两者都已经投影到 `camera_frame`，可与 `/da360/points` 叠加。

默认启动 RViz 和 DA360，使用内置 `DA360_small.pth`，DA360
`point_stride=4`。YOLO 默认关闭，需要时显式加 `--yolo-depth`；可选参数：

```bash
./start_pointcloud_pipeline.sh --no-rviz
./start_pointcloud_pipeline.sh --cubemap-no-gui
./start_pointcloud_pipeline.sh --no-cubemap
./start_pointcloud_pipeline.sh --cubemap-face-size 512
./start_pointcloud_pipeline.sh --da360-point-stride 2
./start_pointcloud_pipeline.sh --model-path /path/to/checkpoint.pth
./start_pointcloud_pipeline.sh --yolo-depth
./start_pointcloud_pipeline.sh --yolo-model-path /path/to/yolo26s-depth.pt
./start_pointcloud_pipeline.sh --yolo-point-stride 4
./start_pointcloud_pipeline.sh --no-yolo-depth
```

DA360 的 `point_stride` 也可以直接传给 launch：`point_stride:=1` 输出最密，
`point_stride:=2` 是较密的折中，默认 `4` 负载最低。步长越小，点云消息和 RViz 负载越大。

YOLO 节点的完整参数在
`insta360_ros_driver/config/yolo26s_depth.yaml`；其中 `depth_mode=range` 按官方
“相机到表面的米制距离”解释深度。若使用的是光轴 Z 深度，可改为
`depth_mode=optical_z`。

## 实时校准

校准模式会用带滑块界面的 Python 全景节点替换 C++ 全景节点，Cubemap、深度和
点云仍会继续运行：

```bash
./start_pointcloud_pipeline.sh --calibrate
```

`Back CX`、`Back CY` 和 `Back Radius` 滑块分别对应 `back_cx_offset`、
`back_cy_offset` 和 `back_radius_scale`，只调整后镜头映射，不再影响前镜头。

调整滑块后按 `s` 将参数写回本仓库的
`insta360_ros_driver/config/equirectangular.yaml`，按 `q` 或
Ctrl-C 关闭整条流程。

Ctrl-C 会统一关闭相机驱动、解码、全景转换、DA360/YOLO worker 和 RViz。
