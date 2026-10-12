"""Fast kinematic integration harness (explicitly not a PhysX acceptance result)."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .engine import CinematicEngine
from .evaluation import EVALUATION_PROTOCOL, save_json, trajectory_summary
from .reference import OriginalReference
from .scenarios import RuntimeScene, NAMES
from .sensors import RuntimeLidar
from .kc_svsdf import advance
from .dynamic_collision import DynamicCollision
from .whole_body_qp import whole_body_kinematics
from .provenance import source_manifest


def run_case(case, output, mode='jev', api_url=None, pace=1., preview=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene = RuntimeScene(Path(case)/'runtime_scene.json')
    runtime = scene.document
    state = np.asarray(runtime['initial_state'], float)
    z = runtime['base_z']
    original = OriginalReference(Path(case)/'task.json', runtime['subject']['position'])
    save_json(output/'original_task.json', original.task)
    save_json(output/'runtime_scene.json', runtime)
    save_json(output/'evaluation_protocol.json', EVALUATION_PROTOCOL)
    lidar = RuntimeLidar()
    for stamp in np.linspace(-.8, 0., 9):
        observed = lidar.synthetic_scan(scene, state, z, float(stamp))
    engine = CinematicEngine(original, observed, output, mode=mode, api_url=api_url, preview=preview)
    rows, failure = [], ''
    begin = time.monotonic()
    dt = .02
    try:
        for i in range(round(original.duration/dt)+1):
            t = i*dt
            if i % 5 == 0:
                observed = lidar.synthetic_scan(scene, state, z, t)
            command, info = engine.step(state, observed, t, dt, base_z=z, commanded_q=state[3:])
            p, r, _ = whole_body_kinematics(state, z)
            truth = DynamicCollision(scene.truth_collision(t), t, uncertainty_rate=0.)
            clearance = float(truth.clearance(state, z).min())
            row = dict(t=t, state=state.tolist(), command=command.tolist(), camera_position=p.tolist(),
                       camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(), truth_clearance_m=clearance, info=info)
            rows.append(row)
            if i % 50 == 0:
                print(json.dumps(dict(case=runtime['name'], mode=mode, t=t, clearance=clearance,
                    jev_calls=engine.jev.decider.client.calls, installed=engine.installed_plans, tracks=len(observed.get('tracks', [])))), flush=True)
            if t < original.duration:
                state = advance(state, command, dt)
            if pace > 0:
                remaining = (i+1)*dt/pace-(time.monotonic()-begin)
                if remaining > 0:
                    time.sleep(remaining)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        engine._write('run_failed', len(rows)*dt, failure=failure)
    finally:
        summary = trajectory_summary(rows, original, engine.summary(), failure)
        summary.update(backend='kinematic_integration', wall_s=time.monotonic()-begin,
                       observation_source='synthetic_sparse_lidar_raycast', measured_physical_constraints=False)
        save_json(output/'summary.json', summary)
        save_json(output/'telemetry.json', rows)
        engine.close()
    print(json.dumps(dict(output=str(output), completed=summary['completed'], failure=failure,
                          installed=summary['installed_plans'])), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    if not (root/'scenarios').is_dir():
        from ament_index_python.packages import get_package_share_directory
        root = Path(get_package_share_directory('jev_agent_wbc'))
    parser.add_argument('--case', choices=[*NAMES, 'all'], default='orbit')
    parser.add_argument('--scenarios', type=Path, default=root/'scenarios')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['jev', 'strict', 'both'], default='jev')
    parser.add_argument('--api-url', default='http://127.0.0.1:8000/v1/systemone')
    parser.add_argument('--pace', type=float, default=1., help='maximum sim/wall ratio; 0 unpaced for strict tests')
    parser.add_argument('--no-preview', action='store_true', help='ablate the inherited allocation MPC; retain dynamic safety and bounded task planner')
    parser.add_argument('--allocation-preview', action='store_true', help='enable inherited shot-specific base-allocation MPC for ablation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output/'provenance.json', source_manifest(root,
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}))
    cases = NAMES if args.case == 'all' else [args.case]
    results = []
    for name in cases:
        for mode in (['strict', 'jev'] if args.mode == 'both' else [args.mode]):
            result = run_case(args.scenarios/name, args.output/f'{name}_{mode}', mode, args.api_url, args.pace,
                              args.allocation_preview and not args.no_preview)
            results.append(dict(case=name, **result))
            save_json(args.output/'runs.json', results)
    return 0 if all(r['completed'] for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
