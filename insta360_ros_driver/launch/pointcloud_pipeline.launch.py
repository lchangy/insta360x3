"""Insta360 X3 -> cubemap -> DA360 and YOLO26s-depth point clouds."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


PACKAGE_NAME = 'insta360_ros_driver'


def _shutdown_when_process_exits(action, label):
    return RegisterEventHandler(
        OnProcessExit(
            target_action=action,
            on_exit=[
                LogInfo(msg=f'Core process exited: {label}; stopping point-cloud pipeline'),
                EmitEvent(event=Shutdown(reason=f'{label} exited')),
            ],
        )
    )


def generate_launch_description():
    package_share = FindPackageShare(PACKAGE_NAME)
    runtime_root = PathJoinSubstitution([package_share, 'da360_runtime'])

    worker_python = LaunchConfiguration('worker_python')
    model_path = LaunchConfiguration('model_path')
    point_stride = LaunchConfiguration('point_stride')
    frame_id = LaunchConfiguration('frame_id')
    input_topic = LaunchConfiguration('input_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic')
    equirectangular_config = LaunchConfiguration('equirectangular_config')
    imu_config = LaunchConfiguration('imu_config')
    use_imu_filter = LaunchConfiguration('imu_filter')
    calibration = LaunchConfiguration('calibration')
    use_cubemap = LaunchConfiguration('cubemap')
    cubemap_gui = LaunchConfiguration('cubemap_gui')
    cubemap_face_size = LaunchConfiguration('cubemap_face_size')
    use_yolo26s_depth = LaunchConfiguration('yolo26s_depth')
    yolo26s_depth_config = LaunchConfiguration('yolo26s_depth_config')
    yolo_model_path = LaunchConfiguration('yolo_model_path')
    yolo_point_stride = LaunchConfiguration('yolo_point_stride')
    yolo_imgsz = LaunchConfiguration('yolo_imgsz')
    yolo_depth_mode = LaunchConfiguration('yolo_depth_mode')
    use_rviz = LaunchConfiguration('rviz')
    rviz_config = LaunchConfiguration('rviz_config')

    camera_driver = Node(
        package=PACKAGE_NAME,
        executable='insta360_ros_driver',
        name='insta360_ros_driver',
        output='screen',
    )

    decoder = Node(
        package=PACKAGE_NAME,
        executable='decoder',
        name='image_decoder',
        output='screen',
        parameters=[{
            'compressed_topic': '/dual_fisheye/image/compressed',
            'uncompressed_topic': '/dual_fisheye/image',
            'skip_frame': 0,
            'i_frame_only': False,
        }],
    )

    equirectangular = Node(
        package=PACKAGE_NAME,
        executable='equirectangular_cpp',
        name='equirectangular_node',
        output='screen',
        condition=UnlessCondition(calibration),
        parameters=[equirectangular_config],
    )

    calibration_script = PathJoinSubstitution([
        package_share,
        '..',
        '..',
        'lib',
        PACKAGE_NAME,
        'calibrate.py',
    ])
    calibration_node = ExecuteProcess(
        cmd=[
            worker_python,
            '-u',
            calibration_script,
            '--ros-args',
            '--params-file', equirectangular_config,
            '-p', ['config_path:=', equirectangular_config],
        ],
        output='screen',
        condition=IfCondition(calibration),
        additional_env={'PYTHONUNBUFFERED': '1'},
    )

    imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        condition=IfCondition(use_imu_filter),
        parameters=[imu_config],
    )

    da360_node_script = PathJoinSubstitution([
        runtime_root,
        'ros2_da360',
        'ros2_da360',
        'da360_realtime_node.py',
    ])
    da360 = ExecuteProcess(
        cmd=[
            worker_python,
            '-u',
            da360_node_script,
            '--ros-args',
            '-p', ['repo_root:=', runtime_root],
            '-p', ['worker_python:=', worker_python],
            '-p', ['model_path:=', model_path],
            '-p', ['input_topic:=', input_topic],
            '-p', ['depth_topic:=', depth_topic],
            '-p', ['pointcloud_topic:=', pointcloud_topic],
            '-p', ['frame_id:=', frame_id],
            '-p', ['point_stride:=', point_stride],
        ],
        cwd=runtime_root,
        output='screen',
        additional_env={'PYTHONUNBUFFERED': '1'},
    )

    cubemap = Node(
        package=PACKAGE_NAME,
        executable='ros_cubemap_view.py',
        name='cubemap_view',
        output='screen',
        condition=IfCondition(use_cubemap),
        arguments=[
            '--topic', input_topic,
            '--face-size', cubemap_face_size,
            '--gui', cubemap_gui,
        ],
    )

    yolo26s_depth_script = PathJoinSubstitution([
        package_share,
        '..',
        '..',
        'lib',
        PACKAGE_NAME,
        'yolo26s_depth_pointcloud.py',
    ])
    yolo26s_depth = ExecuteProcess(
        cmd=[
            worker_python,
            '-u',
            yolo26s_depth_script,
            '--ros-args',
            '--params-file', yolo26s_depth_config,
            '-p', ['model_path:=', yolo_model_path],
            '-p', ['point_stride:=', yolo_point_stride],
            '-p', ['imgsz:=', yolo_imgsz],
            '-p', ['depth_mode:=', yolo_depth_mode],
        ],
        output='screen',
        condition=IfCondition(use_yolo26s_depth),
        additional_env={'PYTHONUNBUFFERED': '1'},
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        output='screen',
        condition=IfCondition(use_rviz),
        arguments=['-d', rviz_config],
    )

    arguments = [
        DeclareLaunchArgument(
            'worker_python',
            default_value=EnvironmentVariable(
                'DA360_PYTHON',
                default_value='/usr/bin/python3',
            ),
            description='Python interpreter containing CUDA Torch and OpenCV',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=PathJoinSubstitution([
                runtime_root,
                'checkpoints',
                'DA360_small.pth',
            ]),
        ),
        DeclareLaunchArgument('point_stride', default_value='4'),
        DeclareLaunchArgument('frame_id', default_value='camera_frame'),
        DeclareLaunchArgument('input_topic', default_value='/equirectangular/image'),
        DeclareLaunchArgument('depth_topic', default_value='/da360/depth'),
        DeclareLaunchArgument('pointcloud_topic', default_value='/da360/points'),
        DeclareLaunchArgument(
            'equirectangular_config',
            default_value=PathJoinSubstitution([package_share, 'config', 'equirectangular.yaml']),
        ),
        DeclareLaunchArgument(
            'imu_config',
            default_value=PathJoinSubstitution([package_share, 'config', 'imu_filter.yaml']),
        ),
        DeclareLaunchArgument('imu_filter', default_value='true'),
        DeclareLaunchArgument('calibration', default_value='false'),
        DeclareLaunchArgument('cubemap', default_value='true'),
        DeclareLaunchArgument('cubemap_gui', default_value='true'),
        DeclareLaunchArgument('cubemap_face_size', default_value='360'),
        DeclareLaunchArgument('yolo26s_depth', default_value='false'),
        DeclareLaunchArgument(
            'yolo26s_depth_config',
            default_value=PathJoinSubstitution([
                package_share,
                'config',
                'yolo26s_depth.yaml',
            ]),
        ),
        DeclareLaunchArgument('yolo_model_path', default_value='yolo26s-depth.pt'),
        DeclareLaunchArgument('yolo_point_stride', default_value='2'),
        DeclareLaunchArgument('yolo_imgsz', default_value='768'),
        DeclareLaunchArgument('yolo_depth_mode', default_value='range'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([package_share, 'rviz', 'da360_realtime.rviz']),
        ),
    ]

    core_actions = [
        camera_driver,
        decoder,
        equirectangular,
        calibration_node,
        da360,
        cubemap,
        yolo26s_depth,
    ]
    shutdown_handlers = [
        _shutdown_when_process_exits(camera_driver, 'Insta360 camera driver'),
        _shutdown_when_process_exits(decoder, 'H.264 decoder'),
        _shutdown_when_process_exits(equirectangular, 'equirectangular converter'),
        _shutdown_when_process_exits(calibration_node, 'equirectangular calibration'),
        _shutdown_when_process_exits(da360, 'DA360 depth worker'),
        _shutdown_when_process_exits(cubemap, 'Cubemap converter'),
        _shutdown_when_process_exits(yolo26s_depth, 'YOLO26s-depth point-cloud worker'),
    ]

    return LaunchDescription(
        arguments
        + core_actions
        + [imu_filter, rviz]
        + shutdown_handlers
    )
