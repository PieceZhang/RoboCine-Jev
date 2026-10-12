"""Independent one-command dynamic demos; default: all numbered combo paths."""
import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration


def start(context, share):
    value = lambda name: LaunchConfiguration(name).perform(context)
    command = [sys.executable, os.path.join(share, 'scripts', 'run_dynamic_demos.py'),
        '--output', value('output_directory'), '--domain', value('ros_domain_id'),
        '--headless', value('headless'), '--demo', value('demo'), '--timeout', value('timeout_s'),
        '--physics-hz', value('physics_hz'), '--stage-view', value('stage_view'),
        '--real-time-factor', value('real_time_factor')]
    for parameter, option in [('trajectory_file', '--trajectory'), ('obstacle_trajectory_file', '--scene'),
                              ]:
        if value(parameter).strip():
            command += [option, value(parameter)]
    if value('record').lower() in ('true', '1', 'yes'):
        command.append('--record')
    process = ExecuteProcess(cmd=command, output='screen', sigterm_timeout='45', sigkill_timeout='25')
    def finished(event, context):
        if event.returncode:
            raise RuntimeError(f'Dynamic demo runner failed (exit {event.returncode}); see {value("output_directory")}')
        return []
    return [RegisterEventHandler(OnProcessExit(target_action=process, on_exit=finished)), process]


def generate_launch_description():
    share = get_package_share_directory('jvs_mock')
    return LaunchDescription([
        DeclareLaunchArgument('trajectory_file', default_value='', description='empty: every numbered combo in order'),
        DeclareLaunchArgument('obstacle_trajectory_file', default_value=''),
        DeclareLaunchArgument('demo', default_value='crossing', description='crossing, overtake, two_movers, or all report presets'),
        DeclareLaunchArgument('stage_view', description='Required USD/USDA scene path; mounted by Isaac with its references'),
        DeclareLaunchArgument('output_directory', default_value='/tmp/jvs_dynamic_demos'),
        DeclareLaunchArgument('ros_domain_id', default_value='226'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('record', default_value='false'),
        DeclareLaunchArgument('physics_hz', default_value='1000.0'),
        DeclareLaunchArgument('real_time_factor', default_value='.1', description='Simulation pacing; independent of trajectory physical seconds'),
        DeclareLaunchArgument('timeout_s', default_value='1800.0'),
        OpaqueFunction(function=start, args=[share]),
    ])
