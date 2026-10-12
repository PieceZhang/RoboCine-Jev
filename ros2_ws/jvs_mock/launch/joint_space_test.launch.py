"""Send slow test commands to the active arm and differential-drive controllers."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("rate_hz", default_value="20.0"),
        DeclareLaunchArgument("period_s", default_value="10.0"),
        DeclareLaunchArgument("amplitude_scale", default_value="1.0"),
        DeclareLaunchArgument("base_command_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("linear_velocity", default_value="0.10"),
        DeclareLaunchArgument("angular_velocity", default_value="0.15"),
        Node(
            package="jvs_mock",
            executable="joint_space_commander",
            name="joint_space_commander",
            output="screen",
            parameters=[{
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "rate_hz": ParameterValue(LaunchConfiguration("rate_hz"), value_type=float),
                "period_s": ParameterValue(LaunchConfiguration("period_s"), value_type=float),
                "amplitude_scale": ParameterValue(
                    LaunchConfiguration("amplitude_scale"), value_type=float
                ),
            }],
        ),
        Node(
            package="jvs_mock",
            executable="base_velocity_commander",
            name="base_velocity_commander",
            output="screen",
            parameters=[{
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "rate_hz": ParameterValue(
                    LaunchConfiguration("base_command_rate_hz"), value_type=float
                ),
                "linear_velocity": ParameterValue(
                    LaunchConfiguration("linear_velocity"), value_type=float
                ),
                "angular_velocity": ParameterValue(
                    LaunchConfiguration("angular_velocity"), value_type=float
                ),
            }],
        ),
    ])
