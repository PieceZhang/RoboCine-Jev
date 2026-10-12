"""Launch frl_apartment_0 (Replica, hand-refined scene) in Gazebo Fortress
(gz-sim6, ROS 2 Humble): retimed cinema_traj push-in on the camera_drone +
the walking mannequin subject_01, both teleported via /world/set_pose, plus
the latched Path server.

    ros2 launch cinema_traj_gz_demo frl_apartment_0.launch.py
    ros2 launch cinema_traj_gz_demo frl_apartment_0.launch.py time_scale:=5.0 loop:=true

Both players start on their first timer tick; start them within the same
second (launch does this naturally). The subject walk has a 10 s pause after
0.3 s — the 3.1 s retimed camera shot lands inside the pause, so small
start offsets are visually harmless.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORLD_NAME = "frl_apartment_0"


def generate_launch_description():
    pkg = get_package_share_directory("cinema_traj_gz_demo")
    gz_sim = get_package_share_directory("ros_gz_sim")
    default_traj = os.path.join(pkg, "trajectories", "frl_apartment_0_retimed.json")
    default_subject = os.path.join(pkg, "trajectories", "frl_apartment_0_subject.json")
    world_path = os.path.join(pkg, "worlds", "frl_apartment_0.world")
    traj_file = LaunchConfiguration("traj_file")

    return LaunchDescription([
        DeclareLaunchArgument("traj_file", default_value=default_traj,
                              description="cinema_traj exports JSON (retimed preferred)"),
        DeclareLaunchArgument("time_scale", default_value="10.0",
                              description="playback speed-up for both players"),
        DeclareLaunchArgument("loop", default_value="false"),

        # Gazebo Fortress: gz-sim6, -r = run (unpaused)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gz_sim, "launch", "gz_sim.launch.py")),
            launch_arguments={"gz_args": f"-r {world_path}"}.items()),

        # world set_pose service (gz.msgs.Pose) → ROS 2 SetEntityPose
        Node(package="ros_gz_bridge", executable="parameter_bridge",
             name="set_pose_bridge", output="screen",
             arguments=[f"/world/{WORLD_NAME}/set_pose@ros_gz_interfaces/srv/SetEntityPose"]),

        Node(package="cinema_traj_bridge", executable="traj_server",
             name="cinema_traj_bridge", output="screen",
             parameters=[{"traj_file": traj_file,
                          "frame_id": "map",
                          "path_step": 5}]),

        # camera playback on camera_drone
        Node(package="cinema_traj_gz_demo", executable="path_player",
             name="path_player", output="screen",
             parameters=[{"traj_file": traj_file,
                          "world": WORLD_NAME,
                          "time_scale": LaunchConfiguration("time_scale"),
                          "loop": LaunchConfiguration("loop")}]),

        # mannequin playback on subject_01 (same set_pose mechanism)
        Node(package="cinema_traj_gz_demo", executable="path_player",
             name="subject_player", output="screen",
             parameters=[{"traj_file": default_subject,
                          "entity": "subject_01",
                          "world": WORLD_NAME,
                          "time_scale": LaunchConfiguration("time_scale"),
                          "loop": LaunchConfiguration("loop"),
                          "rate_hz": 20.0}]),
    ])
