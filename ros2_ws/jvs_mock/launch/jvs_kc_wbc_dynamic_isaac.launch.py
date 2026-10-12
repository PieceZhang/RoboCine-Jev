"""Isaac scene with physical-time kinematic obstacles and live dual LiDAR."""

from pathlib import Path
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from jvs_mock.trajectory_scene import trajectory_spawn



def _value(context, name):
    return LaunchConfiguration(name).perform(context)


def _start_isaac(context, share):
    domain = int(_value(context, "ros_domain_id"))
    if not 0 <= domain <= 232:
        raise ValueError("ros_domain_id must be in [0, 232]")
    stage = str(Path(_value(context, "stage_view")).expanduser().resolve())
    if not Path(stage).is_file():
        raise FileNotFoundError(f"stage_view does not exist: {stage}")
    spawn = trajectory_spawn(_value(context, 'trajectory_file'), {
        key: float(_value(context, argument)) for key, argument in
        [('x', 'x'), ('z', 'z'), ('yaw', 'yaw'), ('height', 'spawn_height')]})

    return [
        SetEnvironmentVariable("ROS_DOMAIN_ID", str(domain)),
        LogInfo(msg=f"KC-WBC scene ROS_DOMAIN_ID={domain}; use the same ros_domain_id for trajfollow"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(share, "launch", "jvs_mock_dynamic_simview.launch.py")
            ),
            launch_arguments={
                "isaac_python": _value(context, "isaac_python"),
                "simview_stage": stage,
                "headless": _value(context, "headless"),
                "start_command_adapter": "false",
                "start_trajectory_follower": "false",
                "trajectory_file": "",
                "reset_service": _value(context, "reset_service"),
                "obstacle_trajectory_file": _value(context, "obstacle_trajectory_file"),
                "dynamic_audit_file": _value(context, "dynamic_audit_file"),
                "record_dir": _value(context, "record_dir"),
                "record_fps": "10.0",
                "record_width": "960",
                "record_height": "720",
                "physics_hz": _value(context, "physics_hz"),
                **{name: _value(context, name) for name in (
                    'solver_type', 'solver_velocity_iterations', 'arm_drive_stiffness',
                    'arm_drive_damping', 'wheel_drive_damping', 'actuator_armature',
                    'arm_drive_type',
                    'command_filter_hz',
                    'physics_feedback_hz',
                    'camera_motion_audit_file')},
                "real_time_factor": _value(context, "real_time_factor"),
                "enable_lidar": _value(context, "enable_lidar"),
                "lidar_config": _value(context, "lidar_config"),
                "lidar_topic": _value(context, "lidar_topic"),
                "lidar_full_scan": _value(context, "lidar_full_scan"),
                "lidar_debug_view": _value(context, "lidar_debug_view"),
                "front_lidar_topic": _value(context, "front_lidar_topic"),
                "front_lidar_config": _value(context, "front_lidar_config"),
                "lidar_motion_bvh": _value(context, "lidar_motion_bvh"),
                "x": str(spawn['x']),
                "z": str(spawn['z']),
                "yaw": str(spawn['yaw']),
                "spawn_height": str(spawn['height']),
                "viewport_scroll_scale": _value(
                    context, "viewport_scroll_scale"
                ),
                "viewport_drag_scale": _value(context, "viewport_drag_scale"),
                "viewport_speed_step": _value(context, "viewport_speed_step"),
            }.items(),
        )
    ]


def generate_launch_description():
    share = get_package_share_directory("jvs_mock")
    default_python = str(Path.home() / "miniconda3/envs/env_isaaclab/bin/python")
    return LaunchDescription([
        DeclareLaunchArgument(
            "ros_domain_id", default_value="226",
            description="Dedicated KC-WBC ROS domain; must match the trajectory launch",
        ),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("obstacle_trajectory_file", default_value=os.path.join(share, "scenes", "dynamic_crossing.json")),
        DeclareLaunchArgument("dynamic_audit_file", default_value=""),
        DeclareLaunchArgument("record_dir", default_value=""),
        DeclareLaunchArgument("physics_hz", default_value="1000.0"),
        DeclareLaunchArgument("solver_type", default_value="PGS"),
        DeclareLaunchArgument("solver_velocity_iterations", default_value="16"),
        DeclareLaunchArgument("arm_drive_stiffness", default_value="200.0"),
        DeclareLaunchArgument("arm_drive_damping", default_value="1000.0"),
        DeclareLaunchArgument("arm_drive_type", default_value="force"),
        DeclareLaunchArgument("command_filter_hz", default_value="0.0"),
        DeclareLaunchArgument("physics_feedback_hz", default_value="50.0"),
        DeclareLaunchArgument("wheel_drive_damping", default_value="20.0"),
        DeclareLaunchArgument("actuator_armature", default_value="0.0"),
        DeclareLaunchArgument("camera_motion_audit_file", default_value=""),
        DeclareLaunchArgument('real_time_factor', default_value='.1'),
        DeclareLaunchArgument("trajectory_file", default_value="",
            description="Read initial_base_pose from this trajectory; does not start tracking"),
        DeclareLaunchArgument(
            "stage_view",
            description="Required Y-up simview USDA/USD scene path",
        ),
        DeclareLaunchArgument("isaac_python", default_value=default_python),
        DeclareLaunchArgument("reset_service", default_value="/jvs/reset_robot"),
        DeclareLaunchArgument("x", default_value="5.0"),
        DeclareLaunchArgument("z", default_value="1.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("spawn_height", default_value="0.002"),
        DeclareLaunchArgument(
            "viewport_scroll_scale",
            default_value="0.05",
            description="Mouse-wheel zoom scale; Isaac Sim default is 1.0",
        ),
        DeclareLaunchArgument(
            "viewport_drag_scale",
            default_value="0.2",
            description=(
                "Mouse-drag scale for pan, orbit, look, and dolly; "
                "Isaac Sim default is 1.0"
            ),
        ),
        DeclareLaunchArgument(
            "viewport_speed_step",
            default_value="1.01",
            description=(
                "RMB+wheel camera-speed multiplier; Isaac Sim default is 1.1"
            ),
        ),
        DeclareLaunchArgument("enable_lidar", default_value="true"),
        DeclareLaunchArgument("lidar_config", default_value="JVS_DynamicNearfield"),
        DeclareLaunchArgument("lidar_topic", default_value="/lidar/points"),
        DeclareLaunchArgument("lidar_motion_bvh", default_value="true"),
        DeclareLaunchArgument("front_lidar_topic", default_value="/lidar/front/points"),
        DeclareLaunchArgument("front_lidar_config", default_value="JVS_DynamicWide"),
        DeclareLaunchArgument("lidar_full_scan", default_value="false"),
        DeclareLaunchArgument("lidar_debug_view", default_value="false",
            description="Draw RTX LiDAR returns in the Isaac Sim viewport"),
        OpaqueFunction(function=_start_isaac, args=[share]),
    ])
