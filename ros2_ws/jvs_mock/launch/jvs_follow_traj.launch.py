"""Run the relative task-space trajectory follower for the JVS camera arm."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_trajectory = PathJoinSubstitution([
        FindPackageShare("jvs_mock"),
        "trajectories",
        "default_task_space_trajectory.json",
    ])
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("trajectory_file", default_value=default_trajectory),
        DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
        DeclareLaunchArgument(
            "command_topic", default_value="/arm_position_controller/commands"
        ),
        DeclareLaunchArgument(
            "control_rate_hz",
            default_value="0.0",
            description="Command rate; 0 uses sample_rate_hz from the JSON file",
        ),
        DeclareLaunchArgument("start_delay_s", default_value="0.5"),
        DeclareLaunchArgument(
            "log_commands",
            default_value="true",
            description="Print every joint command sent to the arm controller",
        ),
        Node(
            package="jvs_mock",
            executable="jvs_follow_traj",
            name="jvs_follow_traj",
            output="screen",
            parameters=[{
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "trajectory_file": LaunchConfiguration("trajectory_file"),
                "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                "command_topic": LaunchConfiguration("command_topic"),
                "control_rate_hz": ParameterValue(
                    LaunchConfiguration("control_rate_hz"), value_type=float
                ),
                "start_delay_s": ParameterValue(
                    LaunchConfiguration("start_delay_s"), value_type=float
                ),
                "log_commands": ParameterValue(
                    LaunchConfiguration("log_commands"), value_type=bool
                ),
            }],
        ),
    ])
