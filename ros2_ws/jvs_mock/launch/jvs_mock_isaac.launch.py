"""Start the JVS mock with the NVIDIA Isaac Sim backend."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def _has_isaac_sim(executable: Path) -> bool:
    """Return whether a Conda-style Python prefix contains Isaac Sim."""
    if not executable.is_file():
        return False
    prefix = executable.parent.parent
    return any(prefix.glob("lib/python*/site-packages/isaacsim"))


def _default_isaac_python() -> str:
    explicit = os.environ.get("ISAAC_PYTHON")
    if explicit:
        return explicit
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "python"
        if _has_isaac_sim(candidate):
            return str(candidate)
    conventional = (
        Path.home()
        / "miniconda3"
        / "envs"
        / "env_isaaclab"
        / "bin"
        / "python"
    )
    return str(conventional) if _has_isaac_sim(conventional) else "python3"


def generate_launch_description():
    package_share = get_package_share_directory("jvs_mock")
    runner = os.path.join(package_share, "scripts", "jvs_isaac_sim.py")
    bootstrap = os.path.join(package_share, "scripts", "jvs_isaac_bootstrap.py")
    xacro_file = os.path.join(package_share, "urdf", "jvs_mock.urdf.xacro")
    adapter_config = os.path.join(package_share, "config", "isaac_sim.yaml")
    trajectory_launch = os.path.join(package_share, "launch", "jvs_follow_traj.launch.py")

    isaac_python = LaunchConfiguration("isaac_python")
    headless = LaunchConfiguration("headless")
    use_sim_time = LaunchConfiguration("use_sim_time")
    start_trajectory = LaunchConfiguration("start_trajectory_follower")

    isaac = ExecuteProcess(
        name="jvs_isaac_sim",
        cmd=[
            sys.executable,
            bootstrap,
            "--isaac-python", FindExecutable(name=isaac_python),
            "--runner", runner,
            "--",
            "--xacro", xacro_file,
            "--xacro-executable", FindExecutable(name="xacro"),
            "--stage", LaunchConfiguration("stage"),
            "--obstacles", LaunchConfiguration("obstacle_file"),
            "--save-stage", LaunchConfiguration("save_stage"),
            "--headless", headless,
            "--add-ground-plane", LaunchConfiguration("add_ground_plane"),
            "--physics-hz", LaunchConfiguration("physics_hz"),
            "--solver-velocity-iterations", LaunchConfiguration("solver_velocity_iterations"),
            "--arm-drive-stiffness", LaunchConfiguration("arm_drive_stiffness"),
            "--arm-drive-damping", LaunchConfiguration("arm_drive_damping"),
            "--arm-drive-type", LaunchConfiguration("arm_drive_type"),
            "--command-filter-hz", LaunchConfiguration("command_filter_hz"),
            "--physics-feedback-hz", LaunchConfiguration("physics_feedback_hz"),
            "--actuator-armature", LaunchConfiguration("actuator_armature"),
            "--solver-type", LaunchConfiguration("solver_type"),
            "--wheel-drive-damping", LaunchConfiguration("wheel_drive_damping"),
            "--spawn-x", LaunchConfiguration("x"),
            "--spawn-y", LaunchConfiguration("y"),
            "--spawn-z", LaunchConfiguration("z"),
            "--spawn-yaw", LaunchConfiguration("yaw"),
            "--enable-lidar", LaunchConfiguration("enable_lidar"),
            "--lidar-mode", LaunchConfiguration("lidar_mode"),
            "--lidar-config", LaunchConfiguration("lidar_config"),
            "--lidar-variant", LaunchConfiguration("lidar_variant"),
            "--lidar-topic", LaunchConfiguration("lidar_topic"),
            "--lidar-frame-id", LaunchConfiguration("lidar_frame_id"),
            "--lidar-full-scan", LaunchConfiguration("lidar_full_scan"),
            "--lidar-motion-bvh", LaunchConfiguration("lidar_motion_bvh"),
            "--lidar-debug-view", LaunchConfiguration("lidar_debug_view"),
            "--enable-front-lidar", LaunchConfiguration("enable_front_lidar"),
            "--front-lidar-config", LaunchConfiguration("front_lidar_config"),
            "--front-lidar-topic", LaunchConfiguration("front_lidar_topic"),
        ],
        output="screen",
    )

    command_adapter = Node(
        package="jvs_mock",
        executable="isaac_command_adapter",
        name="jvs_isaac_command_adapter",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_command_adapter")),
        parameters=[
            adapter_config,
            {
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                "wheel_radius": ParameterValue(
                    LaunchConfiguration("wheel_radius"), value_type=float
                ),
                "wheel_separation": ParameterValue(
                    LaunchConfiguration("wheel_separation"), value_type=float
                ),
            },
        ],
    )

    trajectory_follower = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(trajectory_launch),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "trajectory_file": LaunchConfiguration("trajectory_file"),
        }.items(),
        condition=IfCondition(start_trajectory),
    )

    return LaunchDescription([
        DeclareLaunchArgument("isaac_python", default_value=_default_isaac_python()),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("stage", default_value=""),
        DeclareLaunchArgument("obstacle_file", default_value=""),
        DeclareLaunchArgument("save_stage", default_value=""),
        DeclareLaunchArgument("add_ground_plane", default_value="true"),
        DeclareLaunchArgument("physics_hz", default_value="1000.0"),
        DeclareLaunchArgument("solver_velocity_iterations", default_value="16"),
        DeclareLaunchArgument("arm_drive_stiffness", default_value="200.0"),
        DeclareLaunchArgument("arm_drive_damping", default_value="1000.0"),
        DeclareLaunchArgument("arm_drive_type", default_value="force"),
        DeclareLaunchArgument("command_filter_hz", default_value="0.0"),
        DeclareLaunchArgument("physics_feedback_hz", default_value="50.0"),
        DeclareLaunchArgument("actuator_armature", default_value="0.0"),
        DeclareLaunchArgument("solver_type", default_value="PGS"),
        DeclareLaunchArgument("wheel_drive_damping", default_value="20.0"),
        DeclareLaunchArgument("wheel_radius", default_value="0.07"),
        DeclareLaunchArgument("wheel_separation", default_value="0.21"),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="0.002"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("enable_lidar", default_value="true"),
        DeclareLaunchArgument("lidar_mode", default_value="both", choices=["front", "both"]),
        DeclareLaunchArgument("lidar_config", default_value="JVS_Nearfield"),
        DeclareLaunchArgument("lidar_variant", default_value=""),
        DeclareLaunchArgument("lidar_topic", default_value="/lidar/points"),
        DeclareLaunchArgument("lidar_frame_id", default_value="lidar_link"),
        DeclareLaunchArgument("lidar_full_scan", default_value="true"),
        DeclareLaunchArgument("lidar_motion_bvh", default_value="false"),
        DeclareLaunchArgument("lidar_debug_view", default_value="false"),
        DeclareLaunchArgument("enable_front_lidar", default_value="true"),
        DeclareLaunchArgument("front_lidar_config", default_value="Example_Rotary"),
        DeclareLaunchArgument("front_lidar_topic", default_value="/lidar/front/points"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("start_trajectory_follower", default_value="false"),
        DeclareLaunchArgument("start_command_adapter", default_value="true"),
        DeclareLaunchArgument(
            "trajectory_file",
            default_value=os.path.join(
                package_share, "trajectories", "default_task_space_trajectory.json"
            ),
        ),
        DeclareLaunchArgument(
            "accept_eula",
            default_value=EnvironmentVariable("OMNI_KIT_ACCEPT_EULA", default_value=""),
            description="Set to yes only after accepting the NVIDIA Omniverse EULA",
        ),
        SetEnvironmentVariable("OMNI_KIT_ACCEPT_EULA", LaunchConfiguration("accept_eula")),
        SetEnvironmentVariable("ROS_DISTRO", "humble"),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        isaac,
        command_adapter,
        trajectory_follower,
        RegisterEventHandler(
            OnProcessExit(
                target_action=isaac,
                on_exit=[EmitEvent(event=Shutdown(reason="Isaac Sim exited"))],
            )
        ),
    ])
