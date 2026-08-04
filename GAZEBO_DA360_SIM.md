# Gazebo DA360 仿真

这个入口使用 `semantic_mapping_ws` 的 3D 房间地图，在仿真机器人上临时添加六个 90° 相机：`front/right/back/left/top/bottom`。六张方形图合成为 DA360 使用的 2:1 全景图，再接入 UniDepthV2 small 和 DA360 点云节点。

仿真使用独立的 `ROS_DOMAIN_ID=42`、Gazebo partition 和临时生成的 SDF，不会停止或修改现有 Insta360、YOLO 或真实点云流程。

## 一键启动

在仓库根目录执行：

```bash
./start_da360_gazebo_sim.sh
```

默认使用：

- 场景：`/home/orion/Desktop/semantic_mapping_ws/maps/001/random_start.sdf`
- 六面图：`360x360`
- 全景图：`1440x720`
- DA360 点云采样步长：`4`
- ROS domain：`42`

打开 Gazebo 和 RViz：

```bash
./start_da360_gazebo_sim.sh --gui --rviz
```

只验证六面图到全景图，不加载 DA360：

```bash
./start_da360_gazebo_sim.sh --no-da360
```

## 可调参数

例如降低仿真图像频率、提高点云密度：

```bash
DA360_SIM_RATE=3 DA360_SIM_POINT_STRIDE=2 ./start_da360_gazebo_sim.sh
```

也可以用命令行参数覆盖：`--map-dir`、`--world`、`--face-size`、`--rate`、`--point-stride`、`--ros-domain-id`。

## 话题

六面原始图：

```text
/da360_sim/cubemap/front
/da360_sim/cubemap/right
/da360_sim/cubemap/back
/da360_sim/cubemap/left
/da360_sim/cubemap/top
/da360_sim/cubemap/bottom
```

融合后的全景图和 DA360 输出：

```text
/equirectangular/image
/da360/depth
/da360/points
```

检查图像和点云速率时，在另一个终端也要设置相同 domain：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
ros2 topic hz /equirectangular/image
ros2 topic hz /da360/points
```

如果 Gazebo 输出少量装饰画材质找不到的 warning，但场景、相机和话题仍然启动，这是非致命的资源警告。
