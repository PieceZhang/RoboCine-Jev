#!/usr/bin/env python3
"""Run the full ROS launch against Isaac; stop the owned launch after its report."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=300.)
    parser.add_argument('--ros-domain-id', type=int, default=int(os.environ.get('ROS_DOMAIN_ID','173')),
                        help='isolated DDS domain for this owned test graph')
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--launch', default='jvs_wbc_isaac.launch.py',
                        choices=['jvs_wbc_isaac.launch.py', 'jvs_shot_wbc_isaac.launch.py', 'jvs_kc_wbc_isaac.launch.py'])
    parser.add_argument('--criterion', choices=['sampled', 'tracking'], default='sampled')
    parser.add_argument('--stage-view', type=Path)
    parser.add_argument('--mesh-source', choices=['groundtruth', 'lidar'], default='groundtruth')
    parser.add_argument('--mesh-cache-file', type=Path)
    args = parser.parse_args()
    if not 0 <= args.ros_domain_id <= 232:
        raise ValueError('ROS domain must be in [0,232]')
    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / 'summary.json'
    if summary.exists():
        raise ValueError('use a fresh output directory to avoid accepting a stale report')
    with (args.output / 'launch.log').open('w') as log:
        command = ['ros2', 'launch', 'jvs_mock', args.launch,
            'headless:=true', 'output_directory:='+str(args.output.resolve()),
            'trajectory_file:='+str(args.trajectory.resolve())]
        commands = [command]
        if args.launch == 'jvs_kc_wbc_isaac.launch.py':
            command = ['ros2', 'launch', 'jvs_mock',
                'jvs_kc_wbc_trajfollow.launch.py',
                'output_directory:='+str(args.output.resolve()),
                'trajectory_file:='+str(args.trajectory.resolve()),
                'mesh_source:='+args.mesh_source]
            isaac_command = ['ros2', 'launch', 'jvs_mock', args.launch,
                'headless:=true', 'trajectory_file:='+str(args.trajectory.resolve())]
            # Both KC launch files explicitly set a domain; an environment
            # override alone cannot isolate this test from a running user scene.
            command.append('reset_timeout_s:=180.0')
            command.append('ros_domain_id:='+str(args.ros_domain_id))
            isaac_command.append('ros_domain_id:='+str(args.ros_domain_id))
            if args.stage_view:
                command.append('stage_view:='+str(args.stage_view.resolve()))
                isaac_command.append('stage_view:='+str(args.stage_view.resolve()))
            if args.mesh_cache_file:
                command.append('mesh_cache_file:='+str(args.mesh_cache_file.resolve()))
            commands = [isaac_command, command]
        processes = [subprocess.Popen(item,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ,ROS_DOMAIN_ID=str(args.ros_domain_id)))
            for item in commands]
        result = None
        try:
            start = time.monotonic()
            while (all(process.poll() is None for process in processes)
                   and time.monotonic()-start < args.timeout):
                if summary.exists():
                    try:
                        result = json.loads(summary.read_text())
                    except json.JSONDecodeError:
                        pass
                    if result is not None:
                        (args.output/'tracking_summary.json').write_text(json.dumps(result,indent=2)+'\n')
                        break
                time.sleep(.5)
        finally:
            for process in processes:
                if process.poll() is not None:
                    continue
                # Let launch forward the graceful signal to its children once.
                # Sending it to the group as well can interrupt node cleanup a
                # second time. The forced timeout still owns the entire group.
                process.send_signal(signal.SIGINT)
            for process in processes:
                if process.poll() is not None:
                    continue
                try:
                    process.wait(timeout=30.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        print(json.dumps(result or {'completed': False, 'failure': 'launch exited or timed out without a report'}))
        key = 'tracking_accuracy_pass' if args.criterion == 'tracking' else 'sampled_checks_pass'
        return 0 if result and result.get(key, False) else 2


if __name__ == '__main__':
    raise SystemExit(main())
