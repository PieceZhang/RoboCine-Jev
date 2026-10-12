"""Independent short-shot Isaac suite; all paths resolve inside this package."""
from pathlib import Path
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def start(context, share):
    value = lambda key: LaunchConfiguration(key).perform(context)
    command = [sys.executable, str(share/'scripts/jvs_isaac_bootstrap.py'),
        '--isaac-python', value('isaac_python'), '--runner', str(share/'scripts/run_isaac.py'), '--',
        '--case', value('case'), '--mode', value('mode'), '--output', value('output_directory'),
        '--physics-hz', value('physics_hz'), '--pace', value('real_time_factor'),
        '--api-url', value('api_url'), '--headless', 'true']
    if value('record').lower() in ('1', 'true', 'yes'):
        command.append('--record')
    return [ExecuteProcess(cmd=command, output='screen',
        additional_env={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OMNI_KIT_ACCEPT_EULA': 'YES'})]


def generate_launch_description():
    share = Path(get_package_share_directory('jev_agent_wbc'))
    defaults = dict(case='all', mode='jev', output_directory='/tmp/jev_cinematic_isaac',
        physics_hz='1000', real_time_factor='.2', record='true',
        api_url='http://127.0.0.1:8000/v1/systemone',
        isaac_python=str(Path.home()/'miniconda3/envs/env_isaaclab/bin/python'))
    return LaunchDescription([*[DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()],
                              OpaqueFunction(function=start, args=[share])])
