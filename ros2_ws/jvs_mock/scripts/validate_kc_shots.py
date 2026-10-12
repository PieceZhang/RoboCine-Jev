#!/usr/bin/env python3
"""Run actual Isaac/ROS shots serially, retaining telemetry and owned-process logs.

Source the ROS workspace first. A new scene is started for each initial pose;
--reuse-scene is for trajectories matching an already running scene. This script
never terminates a scene it did not start. Numeric rollout is not acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def stop(process):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run(options):
    root = Path(__file__).resolve().parents[1]
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, ROS_LOG_DIR=str(output/'ros_logs'),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    reports = []
    for name in options.shots:
        trajectory = (root/'trajectories'/f'{name}.json').resolve()
        document = json.loads(trajectory.read_text())
        folder = output/name
        folder.mkdir(exist_ok=False)
        sources = [*(root/'jvs_mock').glob('*.py'), *(root/'config').glob('*.yaml'),
                   *(root/'launch').glob('*.py'), *(root/'scripts').glob('jvs_*isaac.py')]
        source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(sources)}
        # Archive input before execution; a concurrent catalogue regeneration
        # must not change the reference halfway through an owned test run.
        snapshot = folder/'trajectory.json'
        snapshot.write_bytes(trajectory.read_bytes())
        metadata = dict(trajectory=str(trajectory),
            trajectory_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            initial_base_pose=document.get('initial_base_pose', document.get('scene_context', {}).get('assumed_spawn')),
            stage=str(Path(options.stage).resolve()), ros_domain_id=options.domain,
            stage_sha256=hashlib.sha256(Path(options.stage).read_bytes()).hexdigest(),
            plant='Isaac Sim / PhysX', verification_scope='control_rate_samples_only',
            source_sha256=source_hashes)
        (folder/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        scene = follower = None
        scene_log = follow_log = None
        try:
            common = [f'ros_domain_id:={options.domain}', f'stage_view:={options.stage}',
                      f'trajectory_file:={snapshot}']
            if not options.reuse_scene:
                scene_log = (folder/'scene.log').open('w')
                scene = subprocess.Popen(['ros2', 'launch', 'jvs_mock',
                    'jvs_kc_wbc_isaac.launch.py', 'headless:=true', *common],
                    stdout=scene_log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
                deadline = time.monotonic()+120
                while time.monotonic() < deadline and scene.poll() is None:
                    if 'JVS simview ready:' in (folder/'scene.log').read_text(errors='replace'):
                        break
                    time.sleep(.5)
                else:
                    raise RuntimeError('Isaac scene failed to become ready')
            follow_log = (folder/'launch.log').open('w')
            # Each new spawn gets a correctly fingerprinted, separate mesh.
            cache_name = name.removesuffix('_actions')
            cache_folder = Path(options.mesh_cache_root)/cache_name if options.mesh_cache_root else folder
            cache = ([] if options.reuse_scene else
                     [f'mesh_cache_file:={cache_folder / "scene.kc_mesh.npz"}'])
            follower = subprocess.Popen(['ros2', 'launch', 'jvs_mock',
                'jvs_kc_wbc_trajfollow.launch.py', 'mesh_source:=groundtruth',
                f'output_directory:={folder}', *cache, *common], stdout=follow_log,
                stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            deadline = time.monotonic()+options.timeout
            while time.monotonic() < deadline and follower.poll() is None:
                result = folder/'summary.json'
                if result.exists():
                    try:
                        summary = json.loads(result.read_text())
                    except json.JSONDecodeError:
                        time.sleep(.2)
                        continue
                    (folder/'tracking_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
                    break
                time.sleep(.5)
            else:
                raise RuntimeError('tracking exited without a result or exceeded the timeout')
            metadata['source_changed_during_run'] = [key for key, digest in source_hashes.items()
                if hashlib.sha256((root/key).read_bytes()).hexdigest() != digest]
            (folder/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
            reports.append(dict(name=name, source_changed_during_run=metadata['source_changed_during_run'], **summary))
            print(json.dumps(dict(name=name, completed=summary['completed'],
                approximate_success=summary.get('approximate_success'),
                tracking_accuracy_pass=summary.get('tracking_accuracy_pass'),
                position_rms=summary.get('position_error_rms'),
                position_max=summary.get('position_error_max'),
                orientation_rms=summary.get('orientation_error_rms')), ensure_ascii=False), flush=True)
        except Exception as error:
            reports.append(dict(name=name, completed=False, failure=str(error)))
            print(f'{name}: {error}', flush=True)
        finally:
            stop(follower)
            stop(scene)
            if scene_log:
                scene_log.close()
            if follow_log:
                follow_log.close()
            (output/'validation.json').write_text(json.dumps(reports, indent=2)+'\n')
    required = {'tracking': ['tracking_accuracy_pass'], 'cinematic': ['approximate_success'],
                'both': ['tracking_accuracy_pass', 'approximate_success']}[options.criterion]
    return all(not r.get('source_changed_during_run') and all(r.get(key, False) for key in required)
               for r in reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shots', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', default=str(Path(__file__).resolve().parents[3]/
        'cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda'))
    parser.add_argument('--domain', type=int, default=209)
    parser.add_argument('--timeout', type=float, default=720.)
    parser.add_argument('--reuse-scene', action='store_true')
    parser.add_argument('--criterion', choices=['tracking', 'cinematic', 'both'], default='both',
                        help='Require peak/RMS tracking accuracy, cinematic completion, or both')
    parser.add_argument('--mesh-cache-root', default='',
                        help='Reuse fingerprint-checked <root>/<shot>/scene.kc_mesh.npz caches')
    raise SystemExit(0 if run(parser.parse_args()) else 1)
