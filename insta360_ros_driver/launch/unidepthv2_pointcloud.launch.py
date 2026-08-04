"""Run only the independent UniDepthV2-Small cubemap point-cloud node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


PACKAGE_NAME = "insta360_ros_driver"


def generate_launch_description():
    package_share = FindPackageShare(PACKAGE_NAME)
    script = PathJoinSubstitution([
        package_share,
        "..",
        "..",
        "lib",
        PACKAGE_NAME,
        "unidepthv2_pointcloud.py",
    ])
    config = PathJoinSubstitution([
        package_share,
        "config",
        "unidepthv2_pointcloud.yaml",
    ])

    worker_python = LaunchConfiguration("worker_python")
    model_name = LaunchConfiguration("model_name")
    unidepth_repo = LaunchConfiguration("unidepth_repo")
    front_image_topic = LaunchConfiguration("front_image_topic")
    front_depth_topic = LaunchConfiguration("front_depth_topic")
    front_pointcloud_topic = LaunchConfiguration("front_pointcloud_topic")
    left_image_topic = LaunchConfiguration("left_image_topic")
    right_image_topic = LaunchConfiguration("right_image_topic")
    back_image_topic = LaunchConfiguration("back_image_topic")
    left_depth_topic = LaunchConfiguration("left_depth_topic")
    right_depth_topic = LaunchConfiguration("right_depth_topic")
    back_depth_topic = LaunchConfiguration("back_depth_topic")
    left_pointcloud_topic = LaunchConfiguration("left_pointcloud_topic")
    right_pointcloud_topic = LaunchConfiguration("right_pointcloud_topic")
    back_pointcloud_topic = LaunchConfiguration("back_pointcloud_topic")
    frame_id = LaunchConfiguration("frame_id")
    point_stride = LaunchConfiguration("point_stride")
    face_fov_deg = LaunchConfiguration("face_fov_deg")
    device = LaunchConfiguration("device")
    resolution_level = LaunchConfiguration("resolution_level")
    interpolation_mode = LaunchConfiguration("interpolation_mode")
    use_rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    common_cmd = [
        worker_python,
        "-u",
        script,
        "--ros-args",
        "--params-file",
        config,
        "-p",
        ["model_name:=", model_name],
        "-p",
        ["unidepth_repo:=", unidepth_repo],
        "-p",
        ["left_image_topic:=", left_image_topic],
        "-p",
        ["right_image_topic:=", right_image_topic],
        "-p",
        ["front_image_topic:=", front_image_topic],
        "-p",
        ["back_image_topic:=", back_image_topic],
        "-p",
        ["left_depth_topic:=", left_depth_topic],
        "-p",
        ["right_depth_topic:=", right_depth_topic],
        "-p",
        ["front_depth_topic:=", front_depth_topic],
        "-p",
        ["back_depth_topic:=", back_depth_topic],
        "-p",
        ["left_pointcloud_topic:=", left_pointcloud_topic],
        "-p",
        ["right_pointcloud_topic:=", right_pointcloud_topic],
        "-p",
        ["front_pointcloud_topic:=", front_pointcloud_topic],
        "-p",
        ["back_pointcloud_topic:=", back_pointcloud_topic],
        "-p",
        ["frame_id:=", frame_id],
        "-p",
        ["point_stride:=", point_stride],
        "-p",
        ["face_fov_deg:=", face_fov_deg],
    ]

    # An empty device means automatic CUDA/CPU selection in the node.  ROS 2
    # rejects an explicit empty override (`-p device:=`), so use two command
    # variants and only include the override when the user supplied one.
    node_auto_device = ExecuteProcess(
        cmd=common_cmd + [
            "-p",
            ["resolution_level:=", resolution_level],
            "-p",
            ["interpolation_mode:=", interpolation_mode],
        ],
        output="screen",
        condition=IfCondition(
            PythonExpression(["'", device, "' == ''"])
        ),
        additional_env={"PYTHONUNBUFFERED": "1"},
    )

    node_explicit_device = ExecuteProcess(
        cmd=common_cmd + [
            "-p",
            ["device:=", device],
            "-p",
            ["resolution_level:=", resolution_level],
            "-p",
            ["interpolation_mode:=", interpolation_mode],
        ],
        output="screen",
        condition=IfCondition(
            PythonExpression(["'", device, "' != ''"])
        ),
        additional_env={"PYTHONUNBUFFERED": "1"},
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="unidepthv2_rviz",
        output="screen",
        condition=IfCondition(use_rviz),
        arguments=["-d", rviz_config],
    )

    arguments = [
        DeclareLaunchArgument(
            "worker_python",
            default_value=EnvironmentVariable(
                "UNIDEPTH_PYTHON",
                default_value="/usr/bin/python3",
            ),
            description="Python with ROS 2, CUDA Torch, and UniDepth installed",
        ),
        DeclareLaunchArgument(
            "model_name",
            default_value="lpiccinelli/unidepth-v2-vits14",
        ),
        DeclareLaunchArgument("unidepth_repo", default_value=""),
        DeclareLaunchArgument("front_image_topic", default_value="/cubemap/front/image"),
        DeclareLaunchArgument("left_image_topic", default_value="/cubemap/left/image"),
        DeclareLaunchArgument("right_image_topic", default_value="/cubemap/right/image"),
        DeclareLaunchArgument("back_image_topic", default_value="/cubemap/back/image"),
        DeclareLaunchArgument("front_depth_topic", default_value="/unidepthv2/front/depth"),
        DeclareLaunchArgument("left_depth_topic", default_value="/unidepthv2/left/depth"),
        DeclareLaunchArgument("right_depth_topic", default_value="/unidepthv2/right/depth"),
        DeclareLaunchArgument("back_depth_topic", default_value="/unidepthv2/back/depth"),
        DeclareLaunchArgument(
            "front_pointcloud_topic",
            default_value="/unidepthv2/front/points",
        ),
        DeclareLaunchArgument(
            "left_pointcloud_topic",
            default_value="/unidepthv2/left/points",
        ),
        DeclareLaunchArgument(
            "right_pointcloud_topic",
            default_value="/unidepthv2/right/points",
        ),
        DeclareLaunchArgument(
            "back_pointcloud_topic",
            default_value="/unidepthv2/back/points",
        ),
        DeclareLaunchArgument("frame_id", default_value="camera_frame"),
        DeclareLaunchArgument("point_stride", default_value="2"),
        DeclareLaunchArgument("face_fov_deg", default_value="90.0"),
        DeclareLaunchArgument("device", default_value=""),
        DeclareLaunchArgument("resolution_level", default_value="5"),
        DeclareLaunchArgument("interpolation_mode", default_value="bilinear"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([
                package_share,
                "rviz",
                "unidepthv2_pointcloud.rviz",
            ]),
        ),
    ]
    return LaunchDescription(arguments + [node_auto_device, node_explicit_device, rviz])
