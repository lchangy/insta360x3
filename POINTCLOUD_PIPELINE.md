# Insta360 X3 到 DA360 点云

本仓库已经内置 DA360 small 的最小推理运行时和 Cubemap 节点，运行时不依赖其他
源码仓库。模型权重需按 [DEPLOYMENT.md](DEPLOYMENT.md) 从 DA360 官方地址下载。

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

默认启动 RViz，并使用内置 `DA360_small.pth`、`point_stride=4`。可选参数：

```bash
./start_pointcloud_pipeline.sh --no-rviz
./start_pointcloud_pipeline.sh --cubemap-no-gui
./start_pointcloud_pipeline.sh --no-cubemap
./start_pointcloud_pipeline.sh --cubemap-face-size 512
./start_pointcloud_pipeline.sh --point-stride 2
./start_pointcloud_pipeline.sh --model-path /path/to/checkpoint.pth
```

## 实时校准

校准模式会用带滑块界面的 Python 全景节点替换 C++ 全景节点，Cubemap、深度和
点云仍会继续运行：

```bash
./start_pointcloud_pipeline.sh --calibrate
```

调整滑块后按 `s` 将参数写回本仓库的
`insta360_ros_driver/config/equirectangular.yaml`，按 `q` 或
Ctrl-C 关闭整条流程。

Ctrl-C 会统一关闭相机驱动、解码、全景转换、DA360 worker 和 RViz。
