#!/usr/bin/env python3
"""Own sequential Isaac + dynamic follower runs, with fresh report directories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from jvs_mock.dynamic_scene import DynamicScene, combo_trajectories


PRESETS = [('crossing', 'shot_combo1.json'), ('overtake', 'shot_combo3.json'),
           ('two_movers', 'shot_combo4.json')]


def stop(process):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=20.)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_one(package, trajectory, scene, output, args):
    DynamicScene(scene)  # fail before starting GPU/ROS processes
    output.mkdir(parents=True, exist_ok=False)
    (output / 'obstacle_scene.json').write_bytes(scene.read_bytes())
    (output / 'trajectory.json').write_bytes(trajectory.read_bytes())
    environment = dict(os.environ, ROS_DOMAIN_ID=str(args.domain), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    common = ['ros_domain_id:=' + str(args.domain), 'trajectory_file:=' + str(trajectory)]
    stage = args.stage_view
    common += ['stage_view:=' + str(stage)]
    scene_command = ['ros2', 'launch', 'jvs_mock', 'jvs_kc_wbc_dynamic_isaac.launch.py', *common,
        'headless:=' + args.headless, 'obstacle_trajectory_file:=' + str(scene),
        'dynamic_audit_file:=' + str(output / 'obstacle_truth.csv'), 'physics_hz:=' + str(args.physics_hz),
        'real_time_factor:=' + str(args.real_time_factor)]
    if args.record:
        scene_command += ['record_dir:=' + str(output / 'isaac_video')]
    follower_command = ['ros2', 'launch', 'jvs_mock', 'jvs_kc_wbc_dynamic_controller.launch.py', *common,
        'output_directory:=' + str(output)]
    processes = []
    start = time.monotonic()
    result = None
    import jvs_mock.dynamic_whole_body_controller as controller_module
    source_root = Path(controller_module.__file__).resolve().parent
    source_paths = [*source_root.glob('dynamic_*.py'), source_root / 'whole_body_controller.py',
                    *[source_root / name for name in ('shot_execution.py', 'shot_predictive_qp.py',
                        'shot_preview_worker.py', 'wbc_trajectory.py', 'wbc_metrics.py')],
                    source_root / 'obstacle_predictive_qp.py', source_root / 'kc_svsdf.py',
                    package / 'scripts/jvs_dynamic_simview_isaac.py', package / 'scripts/isaac_dynamic_obstacles.py',
                    package / 'scripts/isaac_dynamic_scene.py',
                    package / 'scripts/isaac_dynamic_contacts.py',
                    package / 'scripts/run_dynamic_demos.py',
                    package / 'launch/jvs_kc_wbc_dynamic_trajfollow.launch.py',
                    trajectory, scene, stage, package / 'config/whole_body_controller.yaml',
                    package / 'config/shot_whole_body_controller.yaml',
                    package / 'config/JVS_DynamicNearfield.json', package / 'config/JVS_DynamicWide.json',
                    package / 'urdf/jvs_mock.urdf.xacro',
                    package / 'launch/jvs_kc_wbc_dynamic_isaac.launch.py',
                    package / 'launch/jvs_kc_wbc_dynamic_controller.launch.py']
    source_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    metadata = dict(trajectory=str(trajectory), scene=str(scene), stage_view=str(stage), source_sha256=source_hashes,
                    scene_command=scene_command, follower_command=follower_command)
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with (output / 'isaac.log').open('w') as scene_log, (output / 'controller.log').open('w') as controller_log:
        try:
            simulator = subprocess.Popen(scene_command, stdout=scene_log, stderr=subprocess.STDOUT,
                                         env=environment, start_new_session=True)
            processes.append(simulator)
            while simulator.poll() is None and time.monotonic() - start < args.timeout:
                if 'JVS simview ready:' in (output / 'isaac.log').read_text(errors='replace'):
                    break
                time.sleep(.5)
            else:
                raise RuntimeError('Isaac exited or timed out during startup; see isaac.log')
            follower = subprocess.Popen(follower_command, stdout=controller_log, stderr=subprocess.STDOUT,
                                        env=environment, start_new_session=True)
            processes.append(follower)
            while all(process.poll() is None for process in processes) and time.monotonic() - start < args.timeout:
                report = output / 'summary.json'
                if report.is_file():
                    try:
                        result = json.loads(report.read_text())
                    except json.JSONDecodeError:
                        pass
                    if result is not None:
                        # Preserve the completion report before shutdown.
                        (output / 'tracking_summary.json').write_text(json.dumps(result, indent=2) + '\n')
                        break
                time.sleep(.5)
            if result is None:
                raise RuntimeError('controller/simulator exited or timed out without summary')
        except RuntimeError as error:
            result = dict(completed=False, failure=str(error))
            (output / 'tracking_summary.json').write_text(json.dumps(result, indent=2) + '\n')
        finally:
            for process in reversed(processes):
                stop(process)
    print(json.dumps(dict(output=str(output), completed=result.get('completed'), failure=result.get('failure'))), flush=True)
    metadata['sources_unchanged'] = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
                                        for path, value in source_hashes.items())
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, help='empty: run every existing numbered combo')
    parser.add_argument('--scene', type=Path, help='obstacle model/trajectory JSON')
    parser.add_argument('--demo', choices=['crossing', 'overtake', 'two_movers', 'all'], default='crossing')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-view', type=Path, required=True, help='USD/USDA scene loaded by Isaac (including referenced apartment assets)')
    parser.add_argument('--domain', type=int, default=226)
    parser.add_argument('--headless', choices=['true', 'false'], default='true')
    parser.add_argument('--physics-hz', type=float, default=1000.)
    parser.add_argument('--real-time-factor', type=float, default=.1)
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.domain <= 232 or args.timeout <= 0 or args.physics_hz <= 0 or not 0 < args.real_time_factor <= 1.:
        parser.error('invalid domain, timeout or physics frequency')
    args.stage_view = args.stage_view.expanduser().resolve()
    if not args.stage_view.is_file() or args.stage_view.suffix.lower() not in ('.usd', '.usda', '.usdc'):
        parser.error('--stage-view must be an existing USD/USDA/USDC file')
    if args.output.exists():
        parser.error('use a fresh output directory')
    if args.demo == 'all' and (args.scene or args.trajectory):
        parser.error('--demo all uses the three report presets; omit --scene/--trajectory')
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True)
    from ament_index_python.packages import get_package_share_directory
    package = Path(get_package_share_directory('jvs_mock'))
    if args.demo == 'all':
        jobs = [(package / 'trajectories' / trajectory, package / 'scenes' / f'dynamic_{name}.json', name)
                for name, trajectory in PRESETS]
    else:
        trajectories = [args.trajectory.expanduser().resolve()] if args.trajectory else combo_trajectories(package / 'trajectories')
        scene = args.scene.expanduser().resolve() if args.scene else package / 'scenes' / f'dynamic_{args.demo}.json'
        jobs = [(trajectory, scene, trajectory.stem) for trajectory in trajectories]
    results = []
    for trajectory, scene, name in jobs:
        result = run_one(package, trajectory, scene, args.output / name, args)
        results.append(dict(name=name, trajectory=str(trajectory), scene=str(scene), result=result))
        (args.output / 'runs.json').write_text(json.dumps(results, indent=2) + '\n')
    return 0 if all(row['result'].get('completed') and not row['result'].get('failure') for row in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
