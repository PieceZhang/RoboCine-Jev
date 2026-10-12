"""Launch the demo room in Gazebo Fortress (gz-sim6, ROS 2 Humble) with the
retimed cinema_traj trajectory playing back on the camera_drone entity + the
latched Path server.

    ros2 launch cinema_traj_gz_demo demo_room.launch.py
    ros2 launch cinema_traj_gz_demo demo_room.launch.py time_scale:=25.0 loop:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORLD_NAME = "demo_room"


def generate_launch_description():
    pkg = get_package_share_directory("cinema_traj_gz_demo")
    gz_sim = get_package_share_directory("ros_gz_sim")
    default_traj = os.path.join(pkg, "trajectories", "demo_room_retimed.json")
    world_path = os.path.join(pkg, "worlds", "demo_room.world")
    traj_file = LaunchConfiguration("traj_file")

    return LaunchDescription([
        DeclareLaunchArgument("traj_file", default_value=default_traj,
                              description="cinema_traj exports JSON (retimed preferred)"),
        DeclareLaunchArgument("time_scale", default_value="10.0",
                              description="playback speed-up (retimed demo is long)"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("frame_id", default_value="map"),

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
                          "frame_id": LaunchConfiguration("frame_id"),
                          "path_step": 5}]),

        Node(package="cinema_traj_gz_demo", executable="path_player",
             name="path_player", output="screen",
             parameters=[{"traj_file": traj_file,
                          "world": WORLD_NAME,
                          "time_scale": LaunchConfiguration("time_scale"),
                          "loop": LaunchConfiguration("loop")}]),
    ])
