# Insta360 X3 全景点云部署与使用

## 一、配置环境

已验证环境为 Ubuntu 22.04、ROS 2 Humble、Python 3.10、NVIDIA GPU，以及
Insta360 X3。相机需要插入 microSD 卡、切换到双镜头模式，并把 USB 模式设为
**Android**。

1. 克隆仓库：

   ```bash
   sudo apt update
   sudo apt install -y git
   git clone https://github.com/lchangy/insta360x3.git
   cd instax3
   ```

   从 [DA360 官方 Google Drive](https://drive.google.com/drive/folders/1FMLWZfJ_IPKOa_cEbVqrq8_BRkl3oB_2?usp=drive_link)
   下载 `DA360_small.pth`，放到：

   ```text
   insta360_ros_driver/da360_runtime/checkpoints/DA360_small.pth
   ```

   验证模型：

   ```bash
   echo "cba5dfeeb2199b4a7089a98ce08c7506c1e5ea12b22c3e4ad51cbdb15150dd74  insta360_ros_driver/da360_runtime/checkpoints/DA360_small.pth" | sha256sum -c -
   ```

2. 安装 ROS 2 Humble 后，安装编译和运行依赖：

   ```bash
   sudo apt install -y \
     python3-colcon-common-extensions python3-rosdep python3-venv python3-pip \
     ros-humble-camera-info-manager ros-humble-cv-bridge \
     ros-humble-image-transport ros-humble-imu-tools ros-humble-rviz2 \
     libusb-1.0-0-dev libudev-dev libcurl4-openssl-dev pkg-config \
     libopencv-dev ffmpeg

   source /opt/ros/humble/setup.bash
   rosdep install --from-paths insta360_ros_driver --ignore-src -r -y
   ```

   如果 rosdep 从未初始化，先运行一次：

   ```bash
   sudo rosdep init
   rosdep update
   ```

3. 从 Insta360 官方申请并解压 Linux x86_64 CameraSDK，然后安装到工程：

   ```bash
   ./install_camera_sdk.sh /path/to/extracted/CameraSDK
   ```

   公共仓库默认不包含 `libCameraSDK.so` 和 SDK 头文件，因为它们需要遵守
   Insta360 SDK 的单独授权协议。

4. 使用 uv 按 `uv.lock` 创建 DA360 Python 环境。`--system-site-packages` 不能省略，
   否则虚拟环境找不到 ROS 的 `rclpy` 和消息包：

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv --python /usr/bin/python3 --system-site-packages
   uv sync --frozen
   ```

   `pyproject.toml` 指定 CUDA 12.8 PyTorch 索引，`uv.lock` 固定完整依赖树。依赖
   发生变化时由维护者运行 `uv lock` 更新锁文件；部署机器只运行
   `uv sync --frozen`。验证 GPU：

   ```bash
   .venv/bin/python -c "import rclpy, cv2, torch, ultralytics; from ultralytics.nn.tasks import DepthModel; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
   ```

5. 编译 ROS 包并配置一次 USB 权限：

   ```bash
   source /opt/ros/humble/setup.bash
   colcon build --symlink-install --packages-select insta360_ros_driver
   ./insta360_ros_driver/setup.sh
   ```

   如果脚本把用户加入了 `plugdev` 组，需要注销并重新登录。重新插拔相机后检查：

   ```bash
   lsusb -d 2e1a:0002
   readlink -f /dev/insta
   ```

## 二、启动方式与模式

在仓库根目录运行。启动脚本会自动加载 ROS、检查相机和模型，并统一管理所有进程。

| 模式 | 命令 | 说明 |
| --- | --- | --- |
| 完整模式 | `./start_pointcloud_pipeline.sh` | 相机、全景、Cubemap、DA360 点云、IMU、RViz 启动；YOLO 默认关闭 |
| 启用 YOLO | `./start_pointcloud_pipeline.sh --yolo-depth` | 在原流程中增加左/右 Cubemap YOLO26s-depth |
| 独立 UniDepth | `./start_unidepthv2_pointcloud.sh` | 只启动 UniDepthV2-Small 四面点云和独立 RViz，要求 Cubemap 已运行 |
| 校准模式 | `./start_pointcloud_pipeline.sh --calibrate` | 打开校准滑块；Cubemap、点云和 RViz 仍同时运行 |
| 无 RViz | `./start_pointcloud_pipeline.sh --no-rviz` | 只发布数据，不打开 RViz |
| Cubemap 后台 | `./start_pointcloud_pipeline.sh --cubemap-no-gui` | 发布 Cubemap 话题但不打开四视角窗口 |
| 无 Cubemap | `./start_pointcloud_pipeline.sh --no-cubemap` | 不计算和发布 Cubemap |
| 低界面负载校准 | `./start_pointcloud_pipeline.sh --calibrate --cubemap-no-gui --no-rviz` | 只保留校准窗口，降低 GPU/GUI 压力 |

可调整的数值参数：

```bash
# Cubemap 每个面的尺寸，默认 360
./start_pointcloud_pipeline.sh --cubemap-face-size 512

# 点云采样步长，默认 4；越小越密、负载越高
./start_pointcloud_pipeline.sh --da360-point-stride 2

# 启用 YOLO26s-depth（默认关闭）
./start_pointcloud_pipeline.sh --yolo-depth

# 使用其他 DA360 权重
./start_pointcloud_pipeline.sh --model-path /path/to/model.pth

# 查看全部参数
./start_pointcloud_pipeline.sh --help
```

校准模式中，拖动滑块调整拼接参数，在校准图像窗口按 `s` 保存到
`insta360_ros_driver/config/equirectangular.yaml`，按 `q` 退出。普通模式使用保存后的
参数。

主要输出为：

```text
/equirectangular/image
/da360/depth
/da360/points
/cubemap/front/image
/cubemap/right/image
/cubemap/back/image
/cubemap/left/image
/cubemap/horizontal/image
/imu/data_raw
/imu/data
```

终端出现以下三条信息表示完整链路已经就绪：

```text
ready: equirectangular panorama (/equirectangular/image)
ready: Cubemap horizontal mosaic (/cubemap/horizontal/image)
ready: DA360 point cloud (/da360/points)
```

首次加载模型和 CUDA 预热需要数秒。停止时在启动终端按 `Ctrl+C`，相机、解码、
全景、Cubemap、DA360 和 RViz 会一起关闭。
