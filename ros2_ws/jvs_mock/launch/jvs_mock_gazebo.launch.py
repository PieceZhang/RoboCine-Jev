"""Start the JVS mock with the Gazebo Fortress backend."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory("jvs_mock")
    gz_sim = get_package_share_directory("ros_gz_sim")
    xacro_file = os.path.join(pkg, "urdf", "jvs_mock.urdf.xacro")
    controllers_file = os.path.join(pkg, "config", "jarvis_controllers.yaml")
    default_world = os.path.join(pkg, "worlds", "empty.world")

    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    world = LaunchConfiguration("world")
    robot_description = ParameterValue(
        Command([
            FindExecutable(name="xacro"),
            " ",
            xacro_file,
            " controllers_file:=",
            controllers_file,
            " lidar_mode:=",
            LaunchConfiguration("lidar_mode"),
        ]),
        value_type=str,
    )

    # Run the server and GUI as separate processes.  Besides avoiding Fortress'
    # combined-launch GUI handshake, this makes a server failure visible to the
    # launch system instead of leaving create waiting for /gazebo/worlds.
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={
            "gz_args": ["-s -r -v 3 ", world],
            "on_exit_shutdown": "true",
        }.items(),
    )
    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": "-g -v 3"}.items(),
        condition=IfCondition(LaunchConfiguration("gui")),
    )
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": use_sim_time}],
    )
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="clock_bridge",
        output="screen",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock]"],
    )
    lidar_points_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="lidar_points_bridge",
        output="screen",
        arguments=[
            "/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        condition=LaunchConfigurationEquals("lidar_mode", "both"),
    )
    front_lidar_points_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="front_lidar_points_bridge",
        output="screen",
        arguments=["/lidar/front/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"],
    )
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_jvs_mock",
        output="screen",
        arguments=[
            "-name", "jvs_mock",
            "-topic", "robot_description",
            "-x", LaunchConfiguration("x"),
            "-y", LaunchConfiguration("y"),
            "-z", LaunchConfiguration("z"),
            "-Y", LaunchConfiguration("yaw"),
        ],
    )

    spawner_common = [
        "--controller-manager", "/controller_manager",
        "--controller-manager-timeout", "60",
    ]
    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="joint_state_broadcaster_spawner",
        output="screen",
        arguments=["joint_state_broadcaster", *spawner_common],
    )
    base_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="base_controller_spawner",
        output="screen",
        arguments=["base_controller", *spawner_common],
    )
    arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="arm_position_controller_spawner",
        output="screen",
        arguments=["arm_position_controller", *spawner_common],
    )
    controller_status = Node(
        package="jvs_mock",
        executable="controller_status",
        name="controller_status",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "timeout_s": 90.0}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("lidar_mode", default_value="both", choices=["front", "both"]),
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Start the Gazebo graphical client",
        ),
        DeclareLaunchArgument(
            "gz_ip",
            default_value="127.0.0.1",
            description="Gazebo Transport interface; loopback is reliable for this local sim",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="0.002"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        # Interface auto-selection can pick a VPN, Docker, or disconnected NIC,
        # preventing local Gazebo processes from discovering one another.
        SetEnvironmentVariable("IGN_IP", LaunchConfiguration("gz_ip")),
        SetEnvironmentVariable("GZ_IP", LaunchConfiguration("gz_ip")),
        gazebo_server,
        gazebo_gui,
        clock_bridge,
        lidar_points_bridge,
        front_lidar_points_bridge,
        state_publisher,
        spawn_robot,
        controller_status,
        RegisterEventHandler(
            OnProcessExit(target_action=spawn_robot, on_exit=[joint_state_spawner])
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_state_spawner,
                on_exit=[base_spawner, arm_spawner],
            )
        ),
    ])
