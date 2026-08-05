# Insta360 X3 ROS 2 全景深度与点云

本项目在 Ubuntu 22.04 / ROS 2 Humble 上把 Insta360 X3 实时视频处理为
实时管线默认输出 1036×518 全景图（可选恢复 1440×720）、按需 Cubemap 四视角、DA360 small 相对深度，以及可选的 YOLO26s-depth
左/右视角米制深度点云，并用 RViz 叠加显示；UniDepthV2-Small 四面点云作为独立节点运行。
相机、全景、Cubemap、深度模型和启动文件都由同一个仓库管理。

```text
Insta360 X3 -> H.264 双鱼眼 -> 解码 -> 等距柱状全景
                                      ├-> Cubemap -> 可选 YOLO26s-depth 左/右 -> PointCloud2
                                      └-> DA360 深度 -> PointCloud2 -> RViz
                                      └-> 独立 UniDepthV2-Small 四面 -> PointCloud2 -> RViz
```

主要功能：

1. **相机采集**：通过 USB 连接 Insta360 X3，实时发布双鱼眼视频和 IMU 数据。
2. **视频解码**：将相机输出的 H.264 视频解码为 ROS 2 图像话题。
3. **全景转换**：把双鱼眼画面转换为等距柱状全景图；实时点云管线默认使用 1036×518，以降低 ROS 图像带宽，必要时可用 `--equirect-size 1440x720` 恢复全分辨率。
4. **全景深度估计**：使用 DA360 small 从全景图生成相对深度图。
5. **Cubemap 深度估计**：可选使用 Ultralytics YOLO26s-depth 分别处理左、右 Cubemap 图像，生成米制深度。
6. **点云生成**：将 DA360、左 Cubemap、右 Cubemap 深度转换为带 RGB 的 `PointCloud2`，并在 RViz 同时显示。
7. **Cubemap 输出**：生成 FRONT、RIGHT、BACK、LEFT 四个视角及组合预览图。
8. **可视化**：自动启动 RViz 显示四路点云，也支持关闭 RViz 或窗口后台运行。
9. **实时校准**：通过滑块分别调整前后镜头中心、后镜头半径、裁剪、平移和旋转，并直接保存校准参数。
10. **独立 UniDepth**：`start_unidepthv2_pointcloud.sh` 只启动 UniDepthV2-Small 四面点云和独立 RViz，不干扰原流程。
11. **统一管理**：原流程脚本管理相机、DA360 和可选 YOLO；UniDepth 单独管理。
12. **环境锁定**：使用 `pyproject.toml` 和 `uv.lock` 固定 Python/CUDA 依赖版本。

数据流：

```mermaid
flowchart LR
    CAM[Insta360 X3] -->|USB H.264| DRIVER[相机驱动]
    CAM -->|IMU| IMU_RAW["/imu/data_raw"]
    DRIVER --> COMPRESSED["/dual_fisheye/image/compressed"]
    COMPRESSED --> DECODER[H.264 解码]
    DECODER --> FISHEYE["/dual_fisheye/image"]
    FISHEYE --> PANORAMA[全景转换或校准]
    PANORAMA --> EQUIRECT["/equirectangular/image"]
    EQUIRECT --> DA360[DA360 small]
    DA360 --> DEPTH["/da360/depth"]
    DA360 --> POINTS["/da360/points"]
    POINTS --> RVIZ[RViz]
    EQUIRECT --> CUBEMAP[Cubemap 转换]
    CUBEMAP --> FACES[前/右/后/左视角]
    FACES --> YOLO[YOLO26s-depth 左/右]
    YOLO --> YOLO_POINTS["/yolo26s_depth/{left,right}/points"]
    YOLO_POINTS --> RVIZ
    CUBEMAP --> MOSAIC["/cubemap/horizontal/image"]
    IMU_RAW --> FILTER[Madgwick 滤波]
    FILTER --> IMU["/imu/data"]
```

## 一、配置环境

需要 Ubuntu 22.04、ROS 2 Humble、Python 3.10、Insta360 X3 CameraSDK、支持
PyTorch 的 NVIDIA GPU 和 Ultralytics。CameraSDK、DA360 和 YOLO26s-depth 权重
需要分别从官方来源下载或由 Ultralytics 按模型名缓存。

```bash
git clone https://github.com/lchangy/insta360x3.git
cd instax3

./install_camera_sdk.sh /path/to/extracted/CameraSDK

mkdir -p insta360_ros_driver/da360_runtime/checkpoints
# 从下方 DA360 官方模型地址下载 DA360_small.pth，并放入上述目录

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python /usr/bin/python3 --system-site-packages
uv sync --frozen

source /opt/ros/humble/setup.bash
rosdep install --from-paths insta360_ros_driver --ignore-src -r -y
colcon build --symlink-install --packages-select insta360_ros_driver
./insta360_ros_driver/setup.sh
```

DA360 small 模型不上传到本仓库。请从
[DA360 官方 Google Drive](https://drive.google.com/drive/folders/1FMLWZfJ_IPKOa_cEbVqrq8_BRkl3oB_2?usp=drive_link)
下载 `DA360_small.pth`，保存为：

```text
insta360_ros_driver/da360_runtime/checkpoints/DA360_small.pth
```

当前验证文件的 SHA-256：

```text
cba5dfeeb2199b4a7089a98ce08c7506c1e5ea12b22c3e4ad51cbdb15150dd74
```

参考仓库：

1. [Insta360 desktop CameraSDK](https://github.com/insta360develop/desktop-camerasdk-cpp)：相机连接与视频流 SDK。
2. [ai4ce/insta360_ros_driver](https://github.com/ai4ce/insta360_ros_driver)：ROS 2 相机驱动基础。
3. [Insta360 Research DA360](https://github.com/Insta360-Research-Team/DA360)：全景深度模型、运行时代码和权重来源。
4. [Insta360 Research DAP](https://github.com/Insta360-Research-Team/DAP)：全景/Cubemap 相关实现参考。
5. [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)：DA360 网络基础实现之一。

相机需要插入 microSD 卡、切换到双镜头模式，并把 USB 模式设置为 **Android**。
完整的系统依赖和逐步说明见 [DEPLOYMENT.md](DEPLOYMENT.md)。第三方授权说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 二、启动方式与模式

| 模式 | 命令 |
| --- | --- |
| 原流程：相机、全景、Cubemap、DA360、IMU、RViz（YOLO 默认关闭） | `./start_pointcloud_pipeline.sh` |
| 原流程加 YOLO26s-depth | `./start_pointcloud_pipeline.sh --yolo-depth` |
| 独立 UniDepthV2-Small 四面点云 | `./start_unidepthv2_pointcloud.sh` |
| 实时校准，其他模块继续运行 | `./start_pointcloud_pipeline.sh --calibrate` |
| 不启动 RViz | `./start_pointcloud_pipeline.sh --no-rviz` |
| 发布 Cubemap 但不打开窗口 | `./start_pointcloud_pipeline.sh --cubemap-no-gui` |
| 最高性能模式并记录 Jetson 状态 | `./start_pointcloud_pipeline.sh --max-performance --no-rviz` |
| 完全关闭 Cubemap | `./start_pointcloud_pipeline.sh --no-cubemap` |
| 低界面负载校准 | `./start_pointcloud_pipeline.sh --calibrate --cubemap-no-gui --no-rviz` |

校准模式启动命令：

```bash
cd /path/to/instax3
./start_pointcloud_pipeline.sh --calibrate
```

该命令会打开校准滑块窗口，同时继续运行 Cubemap、DA360 点云和 RViz。拖动滑块
调整中心、裁剪、平移和旋转参数。其中 `back_cx_offset`、`back_cy_offset` 和
`back_radius_scale` 只影响后镜头；在校准图像窗口按 `s` 保存到
`insta360_ros_driver/config/equirectangular.yaml`，按 `q` 或在启动终端按
`Ctrl+C` 关闭整条流程。
![alt text](img_v3_02147_2ee6d122-b689-47af-9080-67e9f3985d9g.jpg)
如果校准时界面负载较高，可以关闭 Cubemap 窗口和 RViz：

```bash
./start_pointcloud_pipeline.sh --calibrate --cubemap-no-gui --no-rviz
```

其他参数：

```bash
./start_pointcloud_pipeline.sh --cubemap-face-size 512
./start_pointcloud_pipeline.sh --cubemap-max-fps 15
./start_pointcloud_pipeline.sh --da360-point-stride 2
DA360_PROFILE=1 DA360_CUDA_GRAPH=auto ./start_pointcloud_pipeline.sh --max-performance --no-rviz
./start_pointcloud_pipeline.sh --model-path /path/to/model.pth
./start_pointcloud_pipeline.sh --yolo-depth
./start_pointcloud_pipeline.sh --yolo-model-path /path/to/yolo26s-depth.pt
./start_pointcloud_pipeline.sh --no-yolo-depth
./start_pointcloud_pipeline.sh --help
```

主要输出话题：

```text
/equirectangular/image
/da360/depth
/da360/points
/cubemap/front/image
/cubemap/right/image
/cubemap/back/image
/cubemap/left/image
/cubemap/horizontal/image
/yolo26s_depth/left/depth
/yolo26s_depth/right/depth
/yolo26s_depth/left/points
/yolo26s_depth/right/points
/unidepthv2/{front,right,back,left}/depth
/unidepthv2/{front,right,back,left}/points
/imu/data_raw
/imu/data
```

所有点云的 frame_id 默认是 camera_frame，并统一采用 ROS 机体坐标约定：
x 向前、y 向左、z 向上。因此在 RViz 中蓝色 z 轴对应现实世界的竖直方向，
DA360、YOLO26s-depth 和 UniDepthV2 点云可以直接叠加。

校准窗口中按 `s` 保存参数，按 `q` 退出。所有模式都可在启动终端按 `Ctrl+C`
统一停止整条流程。
