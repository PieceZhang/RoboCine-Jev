"""Fast kinematic integration harness (explicitly not a PhysX acceptance result)."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .engine import CinematicEngine
from .candidates import SearchBudget, restore_path, path_hash
from .evaluation import EVALUATION_PROTOCOL, save_json, trajectory_summary
from .reference import OriginalReference
from .scenarios import RuntimeScene, NAMES
from .sensors import RuntimeLidar
from .kc_svsdf import advance
from .dynamic_collision import DynamicCollision
from .whole_body_qp import whole_body_kinematics, Limits
from .provenance import source_manifest
from .camera_intent import POLICY,IntentSpec
from .shot_acceptance import evaluate_shot, attach_acceptance


def run_case(case, output, mode='jev', api_url=None, pace=1., preview=False, budget=None, model='perplexity', disable_wbc_avoidance=False, target_replay=None, planning_wbc_avoidance=False, planner_model="floating_camera", representation_boxes=3, geometry_preferences=True):
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
    # Same explicitly configured pinhole camera as the Isaac measurement rig.
    # This enables geometric projection, not synthetic SAM/visibility evidence.
    fy = .5/np.tan(np.deg2rad(27.5))
    engine = CinematicEngine(original, observed, output, mode=mode, api_url=api_url, model=model,
        subject_model=runtime['subject'],representation_boxes=representation_boxes,geometry_preferences=geometry_preferences,
        planner_model=planner_model, planning_obstacle_avoidance=True if planning_wbc_avoidance else None,
        preview=preview, budget=budget, limits=Limits(obstacle_avoidance=not disable_wbc_avoidance), intrinsics=[fy*288/384, fy, .5, .5])
    schedule, replayed = [], []
    if target_replay is not None:
        if mode != 'strict':
            raise ValueError('target replay must disable fresh semantic decisions with mode=strict')
        source = Path(target_replay)
        if json.loads((source/'original_task.json').read_text()) != original.task or json.loads((source/'runtime_scene.json').read_text()) != runtime:
            raise ValueError('replay task/scene differs from current experiment')
        source_events = [json.loads(line) for line in (source/'events.jsonl').read_text().splitlines()]
        cards = {card['id']: card for event in source_events for card in event.get('cards', [])}
        for event in source_events:
            if event['event'] != 'plan_installed':
                continue
            card = cards[event['candidate_id']]
            path = restore_path(original, card['trajectory'])
            if path_hash(path) != card['id']:
                raise ValueError('replay candidate hash mismatch')
            schedule.append((event['time_s'],path,card))
        engine._write('target_replay_loaded',0.,source=str(source.resolve()),
            schedule=[dict(t=t,candidate_id=c['id']) for t,_,c in schedule],new_model_decisions=False)
    rows, failure, failure_diagnostics = [], '', None
    begin = time.monotonic()
    dt = .02
    try:
        for i in range(round((original.duration+POLICY.terminal_hold_s)/dt)+1):
            t = i*dt
            if i % 5 == 0:
                observed = lidar.synthetic_scan(scene, state, z, t)
            while schedule and schedule[0][0] <= t+1e-9:
                installed_at, path, card = schedule.pop(0)
                engine.active = path
                engine.wbc.set_trajectory(path)
                engine.wbc.input_generation = getattr(engine.wbc, 'input_generation', 0)+1
                engine.wbc.route, engine.wbc.route_map_time = None, -np.inf
                engine.wbc.comfort_cache.clear(); engine.wbc.guide_costs.clear()
                replayed.append(dict(t=t,source_installed_at=installed_at,candidate_id=card['id']))
                engine._write('target_replayed',t,source=str(Path(target_replay).resolve()),
                    candidate_id=card['id'],cards=[card],new_model_decision=False,
                    source_installed_at=installed_at,revalidation='not performed: fixed-target CPU ablation')
            # Exact velocity of this CPU constant-control integration model.
            command, info = engine.step(state, observed, t, dt, base_z=z, commanded_q=state[3:],
                                        measured_control=engine.wbc.previous.copy())
            p, r, _ = whole_body_kinematics(state, z)
            truth = DynamicCollision(scene.truth_collision(t), t, uncertainty_rate=0.)
            clearance = float(truth.clearance(state, z).min())
            row = dict(t=t, state=state.tolist(), command=command.tolist(), camera_position=p.tolist(),
                       camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(), truth_clearance_m=clearance, info=info,
                       runtime_tracks=observed.get('tracks', []), observation_stamp=observed['observation_stamp'])
            rows.append(row)
            if i % 50 == 0:
                print(json.dumps(dict(case=runtime['name'], mode=mode, t=t, clearance=clearance,
                    jev_calls=engine.jev.decider.client.calls, installed=engine.installed_plans, tracks=len(observed.get('tracks', [])))), flush=True)
            if t < original.duration+POLICY.terminal_hold_s:
                state = advance(state, command, dt)
            if pace > 0:
                remaining = (i+1)*dt/pace-(time.monotonic()-begin)
                if remaining > 0:
                    time.sleep(remaining)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        failure_diagnostics=getattr(error,'diagnostics',None)
        engine._write('run_failed', len(rows)*dt, failure=failure,diagnostics=failure_diagnostics)
    finally:
        summary = trajectory_summary(rows, original, engine.summary(), failure)
        summary['failure_diagnostics']=failure_diagnostics
        if engine.camera_history:
            history=engine.camera_history
            measured_intent=IntentSpec(original,runtime['subject'],engine.intrinsics).score(
                np.array([r['t'] for r in history]),[r['camera_position'] for r in history],
                [Rotation.from_quat(r['camera_quaternion_xyzw']).as_matrix() for r in history],
                [r['visual'] for r in history],[r['progress_s'] for r in history])
            summary.update(executed_camera_intent=measured_intent,
                terminal_confirmation_observed_s=max(0.,rows[-1]['t']-original.duration) if rows else 0.,
                intent_success=bool(not failure and summary['safety_pass'] and measured_intent['intent_complete']))
            acceptance=evaluate_shot(original,runtime['subject'],
                [r['t'] for r in history],[r['camera_position'] for r in history],
                [Rotation.from_quat(r['camera_quaternion_xyzw']).as_matrix() for r in history],
                [r['visual'] for r in history],[r['progress_s'] for r in history],
                safety_pass=summary['safety_pass'],execution_ok=not failure,
                intrinsics=engine.intrinsics,motion_end_s=engine.active.duration,
                legacy_intent=measured_intent,
                safety_source='runner_execution_checks; independent_sweep_audit_is_separate')
            attach_acceptance(summary,acceptance)
            save_json(output/'camera_history.json',history)
        summary.update(backend='kinematic_integration', obstacle_avoidance_enabled=not disable_wbc_avoidance, wall_s=time.monotonic()-begin,
                       observation_source='synthetic_sparse_lidar_raycast', measured_physical_constraints=False,
                       camera_model='configured 384x288 pinhole, 55deg vertical FOV; no RGB observations')
        summary['planning_obstacle_avoidance_enabled'] = engine.planning_limits.obstacle_avoidance
        summary.update(target_replay_source=None if target_replay is None else str(Path(target_replay).resolve()),
                       replayed_targets=replayed,
                       target_position_rms_m=float(np.sqrt(np.mean([r['info']['position_error']**2 for r in rows]))) if rows else None)
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
    parser.add_argument('--case', default='orbit', help='authored case folder name, or all catalog cases')
    parser.add_argument('--scenarios', type=Path, default=root/'scenarios')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['jev', 'strict', 'rule', 'both'], default='jev')
    parser.add_argument('--target-replay-root', type=Path, help='replay exact case_jev installed targets, using mode=strict; no new Jev calls')
    parser.add_argument('--disable-wbc-avoidance', action='store_true', help='CPU only: tracking QP with collision observer; no avoidance/brake/escape')
    parser.add_argument('--planning-wbc-avoidance', action='store_true', help='enable avoidance in Agent diagnosis, candidate rollouts and revalidation independently of execution')
    parser.add_argument('--planner-model', choices=['wbc','floating_camera'], default='floating_camera', help='camera intent planning with FK/actuator gates by default; wbc is a legacy comparison; execution is separate')
    parser.add_argument('--representation-boxes',type=int,choices=(1,3,6),default=3)
    parser.add_argument('--geometry-preferences',choices=('on','off'),default='on')
    parser.add_argument('--model', default='perplexity', help='explicit backend model ID')
    parser.add_argument('--api-url', default='http://127.0.0.1:8000/v1/systemone')
    parser.add_argument('--pace', type=float, default=1., help='maximum sim/wall ratio; 0 unpaced for strict tests')
    parser.add_argument('--agent-period', type=float, default=.5, help='minimum event spacing, independent of WBC')
    parser.add_argument('--activation-lead', type=float, default=4., help='exact candidate activation lead; never rebased')
    parser.add_argument('--candidate-proposals', type=int, default=6, help='maximum local numerical proposals')
    parser.add_argument('--candidate-budget', type=int, default=2, help='maximum full counterfactual branches')
    parser.add_argument('--candidate-wall-budget',type=float,default=2.,help='cooperative candidate search budget in wall seconds')
    parser.add_argument('--prediction-horizon', type=float, default=8., help='bounded prediction lease in seconds')
    parser.add_argument('--prediction-step', type=float, default=.1, help='far prediction step, .04 to .1 seconds')
    parser.add_argument('--no-preview', action='store_true', help='ablate the inherited allocation MPC; retain dynamic safety and bounded task planner')
    parser.add_argument('--allocation-preview', action='store_true', help='enable inherited shot-specific base-allocation MPC for ablation')
    args = parser.parse_args()
    if args.target_replay_root and args.mode != "strict":
        parser.error("--target-replay-root requires --mode strict")
    if args.disable_wbc_avoidance and args.allocation_preview:
        parser.error("tracking-only WBC cannot enable allocation preview")
    budget = SearchBudget(max_proposals=args.candidate_proposals,decision_period_s=args.agent_period, activation_lead_s=args.activation_lead,
                          max_rollouts=args.candidate_budget, max_horizon_s=args.prediction_horizon,
                          prediction_step_s=args.prediction_step,candidate_wall_budget_s=args.candidate_wall_budget)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output/'provenance.json', source_manifest(root,
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}))
    cases = NAMES if args.case == 'all' else [args.case]
    for name in cases:
        if (Path(name).name != name or name in ('.', '..') or
                not all((args.scenarios/name/file).is_file() for file in ('task.json', 'runtime_scene.json'))):
            parser.error(f'missing or invalid authored case: {name}')
    results = []
    for name in cases:
        for mode in (['strict', 'jev'] if args.mode == 'both' else [args.mode]):
            result = run_case(args.scenarios/name, args.output/f'{name}_{mode}', mode, args.api_url, args.pace,
                              args.allocation_preview and not args.no_preview, budget, args.model, args.disable_wbc_avoidance,
                              None if args.target_replay_root is None else args.target_replay_root/f"{name}_jev",
                              planning_wbc_avoidance=args.planning_wbc_avoidance, planner_model=args.planner_model,
                              representation_boxes=args.representation_boxes,geometry_preferences=args.geometry_preferences=='on')
            results.append(dict(case=name, **result))
            save_json(args.output/'runs.json', results)
    return 0 if all(r['completed'] for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
