"""Prepare the scene mesh and run the KC-WBC trajectory command source."""

from pathlib import Path
import json
from jvs_mock.trajectory_scene import trajectory_spawn
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_STAGE = (
    "/home/w84384651/Documents/RoboCine-Yuelin/"
    "cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda"
)


def _value(context, name):
    return LaunchConfiguration(name).perform(context)


def _controller(context, share, obstacle_file):
    return Node(
        package="jvs_mock",
        executable="whole_body_controller",
        name="whole_body_controller",
        output="screen",
        additional_env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        parameters=[
            os.path.join(share, "config", "whole_body_controller.yaml"),
            os.path.join(share, "config", "shot_whole_body_controller.yaml"),
            {
                "controller_mode": _value(context, "controller_mode"),
                "mesh_source": _value(context, "mesh_source"),
                "lidar_topics": [_value(context, "lidar_topic"), _value(context, "front_lidar_topic")],
                **{'lidar_' + key: float(_value(context, 'mesh_' + key)) for key in
                   ('map_radius', 'voxel_size', 'minimum_height', 'maximum_height')},
                "lidar_minimum_clearance": float(_value(context, "minimum_clearance")),
                "lidar_timeout_s": float(_value(context, "lidar_timeout_s")),
                "lidar_startup_timeout_s": float(_value(context, "lidar_startup_timeout_s")),
                "obstacle_file": obstacle_file,
                "trajectory_file": _value(context, "trajectory_file"),
                "output_directory": _value(context, "output_directory"),
            },
        ],
    )


def _after_mesh(event, context, share, cache):
    if event.returncode != 0:
        return [
            EmitEvent(
                event=Shutdown(reason="ground-truth USD mesh extraction failed")
            )
        ]
    return [
        LogInfo(msg=f"KC-WBC ground-truth mesh ready: {cache}"),
        _controller(context, share, cache),
    ]


def _start_mesh_preparation(context, share, source, stage):
    spawn = trajectory_spawn(_value(context, 'trajectory_file'), {
        key: float(_value(context, argument)) for key, argument in
        [('x', 'x'), ('z', 'z'), ('yaw', 'yaw'), ('height', 'spawn_height')]})
    if source == "lidar":
        return [LogInfo(msg="KC-WBC waiting for live dual LiDAR observations in odom"),
                _controller(context, share, "")]

    requested = _value(context, "mesh_cache_file").strip()
    cache = str(Path(requested).expanduser().resolve()) if requested else str(
        Path(stage).with_suffix(".kc_mesh.npz")
    )
    extractor = os.path.join(share, "scripts", "extract_usda_collision_mesh.py")
    command = [
        _value(context, "isaac_python"),
        extractor,
        "--stage", stage,
        "--output", cache,
        "--spawn-x", str(spawn['x']),
        "--spawn-z", str(spawn['z']),
        "--spawn-yaw", str(spawn['yaw']),
        "--spawn-height", str(spawn['height']),
        "--map-radius", _value(context, "mesh_map_radius"),
        "--voxel-size", _value(context, "mesh_voxel_size"),
        "--minimum-height", _value(context, "mesh_minimum_height"),
        "--maximum-height", _value(context, "mesh_maximum_height"),
        "--minimum-clearance", _value(context, "minimum_clearance"),
    ]
    if _value(context, "force_mesh_rebuild").lower() in (
        "true", "1", "yes", "on"
    ):
        command.append("--force")
    process = ExecuteProcess(
        name="extract_usda_collision_mesh",
        cmd=command,
        output="screen",
        additional_env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    return [
        process,
        RegisterEventHandler(
            OnProcessExit(
                target_action=process,
                on_exit=lambda event, context: _after_mesh(
                    event, context, share, cache
                ),
            )
        ),
    ]


def _after_reset(event, context, share, source, stage):
    if event.returncode != 0:
        return [
            EmitEvent(
                event=Shutdown(
                    reason="Isaac robot reset or startup-state verification failed"
                )
            )
        ]
    return [
        LogInfo(msg="Isaac startup state verified; preparing KC-WBC"),
        *_start_mesh_preparation(context, share, source, stage),
    ]


def _runtime_actions(context, share):
    domain = int(_value(context, "ros_domain_id"))
    if not 0 <= domain <= 232:
        raise ValueError("ros_domain_id must be in [0, 232]")
    source = _value(context, "mesh_source").strip().lower()
    if source not in ("groundtruth", "lidar"):
        raise ValueError("mesh_source must be 'groundtruth' or 'lidar'")
    stage = str(Path(_value(context, "stage_view")).expanduser().resolve())
    if not Path(stage).is_file():
        raise FileNotFoundError(f"stage_view does not exist: {stage}")

    reset = Node(
        package="jvs_mock",
        executable="isaac_reset_client",
        name="jvs_isaac_reset_client",
        output="screen",
        parameters=[{
            "expected_scene_pose": json.dumps(trajectory_spawn(_value(context, 'trajectory_file'), {
                key: float(_value(context, argument)) for key, argument in
                [('x', 'x'), ('z', 'z'), ('yaw', 'yaw'), ('height', 'spawn_height')]})),
            "reset_service": _value(context, "reset_service"),
            "timeout_s": float(_value(context, "reset_timeout_s")),
            "position_tolerance_rad": float(
                _value(context, "reset_position_tolerance_rad")
            ),
            "joint_velocity_tolerance_radps": float(
                _value(context, "reset_joint_velocity_tolerance_radps")
            ),
            "base_velocity_tolerance": float(
                _value(context, "reset_base_velocity_tolerance")
            ),
            "stable_samples": int(_value(context, "reset_stable_samples")),
        }],
    )
    return [
        SetEnvironmentVariable("ROS_DOMAIN_ID", str(domain)),
        LogInfo(msg=f"KC-WBC trajectory ROS_DOMAIN_ID={domain}; scene launch must use the same domain"),
        reset,
        RegisterEventHandler(
            OnProcessExit(
                target_action=reset,
                on_exit=lambda event, context: _after_reset(
                    event, context, share, source, stage
                ),
            )
        ),
    ]


def generate_launch_description():
    share = get_package_share_directory("jvs_mock")
    default_python = str(Path.home() / "miniconda3/envs/env_isaaclab/bin/python")
    return LaunchDescription([
        DeclareLaunchArgument(
            "ros_domain_id", default_value="176",
            description="Dedicated KC-WBC ROS domain; must match the scene launch",
        ),
        DeclareLaunchArgument("controller_mode", default_value="kc"),
        DeclareLaunchArgument(
            "stage_view",
            default_value=DEFAULT_STAGE,
            description="Scene used to build the KC-WBC collision mesh",
        ),
        DeclareLaunchArgument(
            "mesh_source", default_value="groundtruth", description="groundtruth or lidar"
        ),
        DeclareLaunchArgument(
            "mesh_cache_file",
            default_value="",
            description="empty derives <stage>.kc_mesh.npz",
        ),
        DeclareLaunchArgument("lidar_topic", default_value="/lidar/points"),
        DeclareLaunchArgument("front_lidar_topic", default_value="/lidar/front/points"),
        DeclareLaunchArgument("lidar_timeout_s", default_value="2.0"),
        DeclareLaunchArgument("lidar_startup_timeout_s", default_value="60.0"),
        DeclareLaunchArgument("force_mesh_rebuild", default_value="false"),
        DeclareLaunchArgument("mesh_map_radius", default_value="3.0"),
        DeclareLaunchArgument("mesh_voxel_size", default_value="0.08"),
        DeclareLaunchArgument("mesh_minimum_height", default_value="0.03"),
        DeclareLaunchArgument("mesh_maximum_height", default_value="1.60"),
        DeclareLaunchArgument("minimum_clearance", default_value="0.025"),
        DeclareLaunchArgument("isaac_python", default_value=default_python),
        DeclareLaunchArgument("reset_service", default_value="/jvs/reset_robot"),
        DeclareLaunchArgument("reset_timeout_s", default_value="30.0"),
        DeclareLaunchArgument(
            "reset_position_tolerance_rad", default_value="0.005"
        ),
        DeclareLaunchArgument(
            "reset_joint_velocity_tolerance_radps", default_value="0.05"
        ),
        DeclareLaunchArgument(
            "reset_base_velocity_tolerance", default_value="0.005"
        ),
        DeclareLaunchArgument("reset_stable_samples", default_value="5"),
        DeclareLaunchArgument(
            "trajectory_file",
            default_value=os.path.join(share, "trajectories", "shot_moveout.json"),
        ),
        DeclareLaunchArgument(
            "output_directory", default_value="/tmp/jvs_stage3_groundtruth"
        ),
        DeclareLaunchArgument("x", default_value="5.0"),
        DeclareLaunchArgument("z", default_value="1.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("spawn_height", default_value="0.002"),
        OpaqueFunction(function=_runtime_actions, args=[share]),
    ])
