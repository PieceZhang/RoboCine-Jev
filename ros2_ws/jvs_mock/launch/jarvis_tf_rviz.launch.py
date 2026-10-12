"""RViz visualization of the JVS robot TF tree.

Visualizes what the running sim already publishes on /tf — the Isaac
simview runner (or Gazebo ros2_control) broadcasts world → base_link and
the full arm chain (link1..link6 → camera_link → camera_optical_frame)
plus odom → base_link. No sim is started here; keep yours running and do:

    ros2 launch jvs_mock jarvis_tf_rviz.launch.py

The launch also provides the two pieces RViz needs that the sim does not
publish:

* robot_state_publisher — /robot_description for the RobotModel display
  (URDF expanded with sim_backend:=isaac so it matches the simview robot;
  the sim itself already drives the joint TFs, so RSP only supplies the
  description and the fixed tree).
* static_transform_publisher — map → world identity. The cinema_traj
  planner topics (/cam_desired_traj, /cinema_traj_bridge/path_markers)
  use frame "map", which per traj_server's axis convention (Y-up world →
  Z-up ROS) is coordinate-identical to the sim's "world" frame, but no
  node publishes that transform.

Optional args:
    use_markers:=false     hide the planner path/marker overlays
    rviz_config:=<path>    use a different .rviz config
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, UnsetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory("jvs_mock")
    xacro_file = os.path.join(pkg, "urdf", "jvs_mock.urdf.xacro")
    rviz_config = os.path.join(pkg, "config", "jarvis_tf.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_markers = LaunchConfiguration("use_markers")

    robot_description = ParameterValue(
        Command([
            FindExecutable(name="xacro"),
            " ",
            xacro_file,
            " sim_backend:=isaac",
        ]),
        value_type=str,
    )

    return LaunchDescription([
        # Terminals spawned from snap VS Code leak GTK_PATH/GTK_EXE_PREFIX
        # pointing into /snap/code, which makes rviz2's Qt gtk3 platform
        # theme plugin load snap core20 glibc and crash with a
        # "__libc_pthread_init ... GLIBC_PRIVATE" symbol lookup error.
        # Strip the snap variables so rviz2 uses the system GTK.
        UnsetEnvironmentVariable(name="GTK_PATH"),
        UnsetEnvironmentVariable(name="GTK_EXE_PREFIX"),
        UnsetEnvironmentVariable(name="GTK_IM_MODULE_FILE"),
        UnsetEnvironmentVariable(name="GIO_MODULE_DIR"),
        UnsetEnvironmentVariable(name="GSETTINGS_SCHEMA_DIR"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("use_markers", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=rviz_config),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }],
        ),
        # planner topics use frame "map" (Z-up ROS convention); the sim's
        # TF root is "world" with the same axes after traj_server's
        # Y-up → Z-up conversion, so an identity transform links them.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_world_identity",
            arguments=["--x", "0", "--y", "0", "--z", "0",
                       "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                       "--frame-id", "map", "--child-frame-id", "world"],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", LaunchConfiguration("rviz_config")],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ])
