# UniDepthV2-Small 四面 Cubemap 点云

新增的 `unidepthv2_pointcloud` 是独立 ROS 2 节点，只订阅已经存在的四个
Cubemap 图像话题 `/cubemap/front/image`、`right`、`back`、`left`，不会启动、修改或替换原来的相机、全景、DA360 或 YOLO 流程。

## 启动

如果现有流程已经在发布 Cubemap，直接一键启动：

```bash
./start_unidepthv2_pointcloud.sh
```

脚本会自动使用本仓库 `.venv/bin/python`，UniDepth checkout 位于
`.cache/unidepth`，当前直接使用已下载的本地模型 `/home/orion/Downloads/unidepth`，不会再次下载权重。
路径、模型和推理参数都在脚本顶部配置。

也可以手动使用 launch：

先让现有流程发布 Cubemap 话题，然后在另一个终端运行：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch insta360_ros_driver unidepthv2_pointcloud.launch.py \
  worker_python:=/path/to/python-with-unidepth
```

`worker_python` 需要同时包含 ROS 2 Python、CUDA PyTorch 和 UniDepth。使用本地模型目录时，
将该目录传给 `model_name`，例如：

```bash
ros2 launch insta360_ros_driver unidepthv2_pointcloud.launch.py \
  worker_python:=/path/to/python \
  unidepth_repo:=/path/to/unidepth \
  model_name:=/home/orion/Downloads/unidepth
```

默认还会打开独立 RViz。只启动节点时使用 `rviz:=false`，或用 `device:=cpu` 进行 CPU 调试。

## 输出话题

- `/unidepthv2/front/points`、`/right/points`、`/back/points`、`/left/points`：四个 Cubemap 点云。
- `/unidepthv2/front/depth`、`/right/depth`、`/back/depth`、`/left/depth`：可选 `32FC1` 深度图。

四个点云都带 RGB，并发布在 `camera_frame` 下，内置 RViz 配置为
`insta360_ros_driver/rviz/unidepthv2_pointcloud.rviz`。`point_stride` 默认是 2；显存或实时性不足时可调大，
需要更高输出分辨率时可提高 `resolution_level`（有效范围 0--9）。

UniDepth 的单面坐标会按当前 Cubemap 投影转换到共同坐标系：前面朝 `+Z`、右面朝 `+X`、
后面朝 `-Z`、左面朝 `-X`，竖直方向为 `+Y` 向上。这样四个点云可在 RViz 中直接叠加观察。
