"""Isaac bringup plus the exclusive whole-body QP command source."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('jvs_mock')
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='true'),
        DeclareLaunchArgument('solver_type', default_value='PGS'),
        DeclareLaunchArgument('trajectory_file', default_value=os.path.join(share, 'trajectories', 'wbc_coordinated.json')),
        DeclareLaunchArgument('output_directory', default_value='/tmp/jvs_wbc'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'jvs_mock_isaac.launch.py')),
            launch_arguments={'headless': LaunchConfiguration('headless'),
                              'start_command_adapter': 'false',
                              'start_trajectory_follower': 'false', 'enable_lidar': 'false',
                              'solver_velocity_iterations': '4',
                              'solver_type': LaunchConfiguration('solver_type')}.items()),
        Node(package='jvs_mock', executable='whole_body_controller', name='whole_body_controller',
             output='screen', parameters=[os.path.join(share, 'config', 'whole_body_controller.yaml'),
                {'trajectory_file': LaunchConfiguration('trajectory_file'),
                 'output_directory': LaunchConfiguration('output_directory')}]),
    ])
