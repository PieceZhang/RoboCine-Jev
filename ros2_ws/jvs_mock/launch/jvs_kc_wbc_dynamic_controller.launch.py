"""Verify robot reset, then start dynamic KC-WBC using live LiDAR only."""
import json
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from jvs_mock.trajectory_scene import trajectory_spawn


def start(context, share):
    value = lambda name: LaunchConfiguration(name).perform(context)
    domain = int(value('ros_domain_id'))
    if not 0 <= domain <= 232:
        raise ValueError('ros_domain_id must be in [0,232]')
    spawn = trajectory_spawn(value('trajectory_file'), {name: float(value(argument))
        for name, argument in [('x', 'x'), ('z', 'z'), ('yaw', 'yaw'), ('height', 'spawn_height')]})
    reset = Node(package='jvs_mock', executable='isaac_reset_client', output='screen',
        parameters=[dict(expected_scene_pose=json.dumps(spawn), reset_service=value('reset_service'),
            timeout_s=float(value('reset_timeout_s')), position_tolerance_rad=.005,
            joint_velocity_tolerance_radps=.05, base_velocity_tolerance=.005, stable_samples=5)])
    controller = Node(package='jvs_mock', executable='dynamic_whole_body_controller',
        name='whole_body_controller', output='screen',
        additional_env={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}, parameters=[
            os.path.join(share, 'config', 'whole_body_controller.yaml'),
            os.path.join(share, 'config', 'shot_whole_body_controller.yaml'),
            dict(controller_mode='kc', mesh_source='lidar', arm_prepare_s=3.,
                trajectory_file=value('trajectory_file'), output_directory=value('output_directory'),
                lidar_topics=[value('lidar_topic'), value('front_lidar_topic')],
                lidar_map_radius=float(value('mesh_map_radius')), lidar_voxel_size=float(value('mesh_voxel_size')),
                lidar_minimum_height=.03, lidar_maximum_height=1.6, lidar_retention_s=1.5,
                lidar_minimum_clearance=float(value('minimum_clearance')),
                lidar_timeout_s=float(value('lidar_timeout_s')), lidar_startup_timeout_s=60.)])
    def after_reset(event, context):
        return [controller] if event.returncode == 0 else [EmitEvent(event=Shutdown(reason='Isaac reset/state verification failed'))]
    return [SetEnvironmentVariable('ROS_DOMAIN_ID', str(domain)), reset,
            RegisterEventHandler(OnProcessExit(target_action=reset, on_exit=after_reset))]


def generate_launch_description():
    share = get_package_share_directory('jvs_mock')
    defaults = dict(ros_domain_id='226', trajectory_file=os.path.join(share, 'trajectories', 'shot_combo1.json'),
        output_directory='/tmp/jvs_dynamic', stage_view='', reset_service='/jvs/reset_robot',
        reset_timeout_s='180.0', lidar_topic='/lidar/points', front_lidar_topic='/lidar/front/points',
        mesh_map_radius='4.5', mesh_voxel_size='.04', minimum_clearance='.025', lidar_timeout_s='.8',
        x='5.0', z='1.0', yaw='0.0', spawn_height='.002')
    return LaunchDescription([*[DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()],
                              OpaqueFunction(function=start, args=[share])])
