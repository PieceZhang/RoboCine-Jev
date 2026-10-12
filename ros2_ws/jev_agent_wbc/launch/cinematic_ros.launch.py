"""Run on an existing ROS robot/simulator supplying odom, joints, TF and dual LiDAR."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('task_file'),
        DeclareLaunchArgument('output_directory', default_value='/tmp/jev_cinematic_ros'),
        DeclareLaunchArgument('api_url', default_value='http://127.0.0.1:8000/v1/systemone'),
        Node(package='jev_agent_wbc', executable='cinematic_ros', output='screen',
             parameters=[dict(task_file=LaunchConfiguration('task_file'),
                 output_directory=LaunchConfiguration('output_directory'),
                 api_url=LaunchConfiguration('api_url'), use_sim_time=True)],
             additional_env={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}),
    ])
