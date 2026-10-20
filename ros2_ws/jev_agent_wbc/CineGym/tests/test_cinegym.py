from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from CineGym import CineGymEnv, Episode, GymConfig, build_command, generate_episode
from CineGym.config import FAMILIES, LEVELS, MOTIONS
from CineGym.episode import fingerprint
from CineGym.evaluate import summarize_runs
from jev_agent_wbc.dynamic_collision import DynamicCollision
from jev_agent_wbc.kc_svsdf import CollisionGeometry
from jev_agent_wbc.scenarios import RuntimeScene
from jev_agent_wbc.whole_body_qp import whole_body_kinematics


def short_episode(crossing=False):
    state = [0., 0., 0., 0., -.9, 2.1, 0., -1.2, 0.]
    p, r, _ = whole_body_kinematics(state, .002)
    rows = [dict(t=t, position=p.tolist(), quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist()) for t in (0., .02, .04)]
    task = dict(reference_camera_trajectory=dict(frame='world_z_up_camera_link', samples=rows),
        shot_phase=[dict(name='hold', motion='hold', start_s=0., end_s=.04)],
        subject=dict(id='subject', description='test subject'), intent='Hold the camera.')
    obj = dict(id='low_01', radius=.1, keyframes=[dict(t=t, position=[x, 0., .13], velocity=v)
        for t, x, v in ((0., -1., [100., 0., 0.]), (.02, 1., [100., 0., 0.]))]) if crossing else dict(
        id='low_01', radius=.1, keyframes=[dict(t=t, position=[-2., -2., .13], velocity=[0., 0., 0.]) for t in (0., .04)])
    scene = dict(name='test_case', initial_state=state, base_z=.002,
        subject=dict(id='subject', position=(p+[1.5, 0., 0.]).tolist(), radius=.13, height=.36),
        obstacles=[obj], pre_roll_motion='constant_initial_velocity')
    return Episode(task, scene, dict(seed=0, family='hold', difficulty='easy', sha256=fingerprint(task, scene)))


def test_seed_reproducibility_split_and_file_integrity(tmp_path):
    config = GymConfig(family='movein', difficulty='easy')
    a, b = generate_episode(7, config), generate_episode(7, config)
    assert a.task == b.task and a.scene == b.scene and a.metadata == b.metadata
    different = generate_episode(7, GymConfig(family='movein', difficulty='easy', split='test'))
    assert a.metadata['sha256'] != different.metadata['sha256']
    folder = a.save(tmp_path/'case')
    assert Episode.load(folder).scene == a.scene
    with pytest.raises(FileExistsError):
        a.save(folder)
    changed = deepcopy(a.scene); changed['obstacles'][0]['radius'] *= 2
    (folder/'runtime_scene.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='fingerprint'):
        Episode.load(folder)


@pytest.mark.parametrize('family', FAMILIES)
def test_families_are_executable_and_all_heights_interfere(family):
    e = generate_episode(200+FAMILIES.index(family), GymConfig(family=family, difficulty='easy',generation_style='legacy'))
    reference, runtime = e.reference, e.runtime
    assert {o['height_level'] for o in e.metadata['audit']['obstacles']} == set(LEVELS)
    assert e.metadata['nominal_tracking_max_m'] <= .045
    assert e.metadata['nominal_orientation_max_rad'] <= .08
    assert all(o['min_surface_clearance_m'] < 0 and o['initial_clearance_m'] >= .12 and
               o['final_clearance_m'] >= .12 for o in e.metadata['audit']['obstacles'])
    assert all(o['motion'] in MOTIONS for o in e.metadata['audit']['obstacles'])
    assert e.metadata['audit']['obstacle_pair_min_clearance_m'] >= .025
    assert reference.duration == e.task['shot_phase'][-1]['end_s']
    stamps = np.linspace(0., reference.duration, 101)
    for obj in e.scene['obstacles']:
        curve = runtime.curves[obj['id']]
        assert np.isfinite(curve(stamps)).all()
        assert np.linalg.norm(curve(0., 1)) < 1e-9
        assert np.linalg.norm(curve(reference.duration, 1)) < 1e-9
    for phase in e.task['shot_phase'][1:]:
        t = phase['start_s']
        assert np.linalg.norm(reference.sample(t-1e-6)[0]-reference.sample(t+1e-6)[0]) < 1e-5


def test_truth_uses_each_radius_and_independent_tracks():
    e = short_episode()
    e.scene['obstacles'].append(deepcopy(e.scene['obstacles'][0]))
    e.scene['obstacles'][1].update(id='high_02', radius=.23)
    runtime = RuntimeScene(e.scene)
    truth = runtime.truth_collision(0.)
    dynamic = DynamicCollision(truth, 0., uncertainty_rate=0.)
    ds = dynamic.clearance_to(np.array(e.scene['initial_state']), dynamic.centers, dynamic.radii, e.scene['base_z'])
    assert np.allclose(ds[0]-ds[1], .13)
    assert len(set(truth['surfel_track_ids'])) == 2
    e.scene['obstacles'][0]['radius'] = 1.
    assert runtime.obstacles(0.)[0]['radius'] == .1


def test_interaction_no_future_obstacles_and_reset():
    with CineGymEnv() as env:
        with pytest.raises(RuntimeError):
            env.step(np.zeros(8))
        obs, _ = env.reset(episode=short_episode())
        assert 'obstacles' not in obs and 'keyframes' not in str(obs.keys())
        assert obs['lidar']['observation_stamp'] <= obs['t']
        with pytest.raises(ValueError):
            env.step(np.ones(8))
        assert env.t == 0.
        obs, reward, done, truncated, info = env.step(np.zeros(8))
        assert not done and not truncated and reward <= 0.
        obs['state'][0] = 100.
        assert env.state[0] == 0.
        _, _, done, truncated, info = env.step(np.zeros(8))
        assert done and not truncated and info['completed'] and not info['collided']
        with pytest.raises(RuntimeError):
            env.step(np.zeros(8))
        assert env.reset(episode=short_episode())[0]['t'] == 0.
    with pytest.raises(RuntimeError):
        env.reset(episode=short_episode())


def test_fast_crossing_detected_inside_control_interval():
    e = short_episode(crossing=True)
    with CineGymEnv() as env:
        env.reset(episode=e)
        assert env._clearance(env.state, .0) > 0.
        assert env._clearance(env.state, .02) > 0.
        _, _, done, _, info = env.step(np.zeros(8))
        assert done and info['collision'] and info['min_clearance_m'] < 0.


def test_both_commands_select_exact_case_and_physical_defaults(tmp_path):
    folder = short_episode().save(tmp_path/'case')
    executable = tmp_path/'isaac_python'; executable.touch()
    cpu = build_command(folder, tmp_path/'cpu', backend='cpu', mode='strict')
    physx = build_command(folder, tmp_path/'physx', backend='physx', isaac_python=executable)
    for command in (cpu, physx):
        assert command[command.index('--case')+1] == folder.name
        assert command[command.index('--pace')+1] == '1'
    assert physx[physx.index('--physics-hz')+1] == '100'
    assert physx[physx.index('--camera-output')+1] == 'execution'
    for extra in (['--pace=.5'], ['--pac', '.5'], ['--case', 'all']):
        with pytest.raises(ValueError):
            build_command(folder, tmp_path/'bad', extra_args=extra)


def test_report_keeps_unknown_and_cpu_physx_separate(tmp_path):
    for i, backend in enumerate(('cpu', 'cpu', 'physx')):
        path = tmp_path/str(i); path.mkdir()
        meta = dict(backend=backend, family='movein', difficulty='easy', seed=i, episode_sha256=str(i))
        (path/'cinegym_run.json').write_text(json.dumps(meta))
        if i == 0:
            folder = path/'case_strict'; folder.mkdir()
            (folder/'summary.json').write_text(json.dumps(dict(completed=True, collision_samples=0, elapsed_shot_s=8.,
                safety_pass=True, shot_acceptance=dict(accepted=True, status='pass'))))
    report = summarize_runs(tmp_path)
    assert report['groups']['cpu/all']['success_rate'] == 1.
    assert report['groups']['cpu/all']['unverified'] == 1
    assert report['groups']['physx/all']['success_rate'] is None


@pytest.mark.parametrize('kwargs', [dict(obstacle_count=2), dict(action_duration_s=(0., 5.)),
    dict(radius_m=(.2, .1)), dict(family='unknown'), dict(max_attempts=0), dict(audit_step_s=float('nan'))])
def test_invalid_sampling_config(kwargs):
    with pytest.raises(ValueError):
        GymConfig(**kwargs)


def test_random_combo_can_be_restricted_to_exactly_two_actions():
    from CineGym.paths import camera_task
    config = GymConfig(family='random_combo', difficulty='medium', combo_actions=2)
    for seed in range(20):
        task, _, _ = camera_task(np.random.default_rng(seed), 'random_combo', config)
        assert len(task['shot_phase']) == 2
        assert task['shot_phase'][0]['end_s'] == task['shot_phase'][1]['start_s']
    with pytest.raises(ValueError, match='combo_actions'):
        GymConfig(combo_actions=1)


def test_global_difficulty_cap_does_not_change_acceptance():
    e=generate_episode(4200,GymConfig(family='movein',difficulty='easy',enforce_global_penetration=True))
    assert -.035-1e-7<=e.metadata['audit']['min_surface_clearance_m']<0.
    assert e.scene['evaluation']==generate_episode(4200,GymConfig(family='movein',difficulty='easy')).scene['evaluation']


def test_network_timeout_is_unverified_until_same_request_is_answered(tmp_path):
    from CineGym.evaluate import unanswered_transport_requests
    journal=tmp_path/'jev_api.jsonl'
    request=dict(model='jev-1.14.0',state='synthetic scene',questions={'q':{}})
    rows=[dict(request=request,status='transport_unverified_retry',error='timed out')]
    journal.write_text('\n'.join(json.dumps(r) for r in rows))
    assert unanswered_transport_requests(tmp_path)==1
    rows.append(dict(request=dict(request,state='different synthetic scene'),response={'answers':{}}))
    journal.write_text('\n'.join(json.dumps(r) for r in rows))
    assert unanswered_transport_requests(tmp_path)==1
    rows.append(dict(request=request,response={'answers':{}}))
    journal.write_text('\n'.join(json.dumps(r) for r in rows))
    assert unanswered_transport_requests(tmp_path)==0


def test_initial_execution_failure_is_not_lost_when_legacy_summary_crashes(tmp_path):
    (tmp_path/'cinegym_run.json').write_text(json.dumps(dict(backend='cpu',family='movein',
        difficulty='easy',seed=1,episode_sha256='known')))
    output=tmp_path/'case_jev';output.mkdir()
    (output/'events.jsonl').write_text(json.dumps(dict(event='run_failed',time_s=0.,
        failure='RuntimeError: TPR hard safety constraints infeasible'))+'\n')
    report=summarize_runs(tmp_path)
    assert report['groups']['cpu/all']['failed']==1
    assert report['runs'][0]['safety_pass'] is None
    # The authorized network policy still overrides a conclusive local failure
    # when the semantic decision needed for this trial was never received.
    (output/'jev_api.jsonl').write_text(json.dumps(dict(status='transport_unverified_retry',
        request={'state':'test synthetic geometry'},error='timed out'))+'\n')
    assert summarize_runs(tmp_path)['groups']['cpu/all']['unverified']==1


@pytest.mark.parametrize('mode',MOTIONS)
def test_natural_lanes_are_c2_with_valid_continuous_motion_bounds(mode):
    from CineGym.obstacles import natural_lane,motion_bounds
    from scipy.interpolate import CubicHermiteSpline
    ts,ps,vs=natural_lane(np.random.default_rng(47),9.,4.,np.array([0.,0.,.7]),
        np.array([1.,0.,0.]),np.array([0.,1.,0.]),.25,mode)
    curve=CubicHermiteSpline(ts,ps,vs)
    eps=1e-9
    assert np.max(abs(curve(ts[1:-1]-eps,2)-curve(ts[1:-1]+eps,2)))<1e-6
    assert np.max(abs(curve([0.,9.],2)))<1e-8
    assert np.linalg.norm(curve(0.,1))>.02
    assert np.allclose(curve(4.),[0.,0.,.7])
    bound=motion_bounds(curve);dense=np.linspace(0.,9.,4001)
    for order,key in [(1,'speed_m_s'),(2,'acceleration_m_s2'),(3,'jerk_m_s3')]:
        assert np.linalg.norm(curve(dense,order),axis=1).max()<=bound[key]+1e-8


@pytest.mark.parametrize('difficulty,conflicts',[('easy',1),('medium',2)])
def test_natural_episodes_validate_shots_and_separate_conflicts_from_passers(difficulty,conflicts):
    e=generate_episode(12000,GymConfig(family='movein',difficulty=difficulty))
    assert e.metadata['nominal_shot_acceptance']['accepted']
    rows=e.metadata['audit']['obstacles']
    assert sum(r['interaction_role']=='conflict' for r in rows)==conflicts
    assert {r['height_level'] for r in rows}==set(LEVELS)
    for r in rows:
        assert r['min_surface_clearance_m']>=-r['global_penetration_cap_m']-1e-7
        if r['interaction_role']=='conflict':
            assert r['min_surface_clearance_m']<0.
            assert r['first_conflict_s'] >= (3. if difficulty=='easy' else 2.5)-1e-8
        else:assert r['min_surface_clearance_m']>=.12
    boxes=[o for o in e.scene['obstacles'] if o.get('shape')=='box']
    assert len(boxes)==2
    assert abs(boxes[0]['size'][2]-boxes[1]['size'][2])>.1
    assert boxes[0]['size'][0]!=boxes[1]['size'][0]
    for box in boxes:
        assert all(abs(k['position'][2]-box['size'][2]/2)<1e-12 and k['velocity']==[0.,0.,0.] for k in box['keyframes'])
    assert e.metadata['avoidance_solvability']['status']=='certified_cpu_timing_witness'
    assert all(max(k['t'] for k in o['keyframes'])>e.reference.duration for o in e.scene['obstacles'])


def test_generated_avoidance_witness_replays_under_actual_cpu_commands():
    from jev_agent_wbc.nominal import nominal_rollout
    from jev_agent_wbc.kc_svsdf import advance
    from CineGym.authoring import WithTerminalHold
    from CineGym.solvability import timing_pool, finish_witness
    e=generate_episode(51000,GymConfig(family='movein',difficulty='easy',split='test',
        require_avoidance_witness=True,max_attempts=48))
    witness=e.metadata['avoidance_solvability']
    assert witness['status']=='certified_cpu_timing_witness'
    times,states,failure=nominal_rollout(WithTerminalHold(e.reference),e.scene['initial_state'],e.scene['base_z'])
    assert not failure
    pool=timing_pool(e.reference,e.scene,times,states,selected_clocks=witness['clock'])
    assert len(pool)==1
    item=pool[0];actual=[np.array(e.scene['initial_state'])]
    for command in item['controls']:actual.append(advance(actual[-1],command,.02))
    np.testing.assert_allclose(actual,item['states'],atol=1e-11)
    accepted=finish_witness(e.reference,e.scene,pool,[0],list(e.runtime.curves.values()))
    assert accepted and accepted['continuous_external_lower_m']>=.025
    assert accepted['continuous_self_lower_m']>=.004 and accepted['shot_accepted']
    # A caller cannot label an unchecked/invalid body trace as certified.
    item['self_lower']=0.
    assert finish_witness(e.reference,e.scene,pool,[0],list(e.runtime.curves.values())) is None


def test_provenance_includes_acceptance_and_continuous_safety_sources():
    from CineGym.episode import source_fingerprints
    hashes=source_fingerprints()
    for name in ('shot_acceptance.py','cinematic_geometry.py','self_collision.py'):
        assert len(hashes['jev_agent_wbc/'+name])==64


def test_explicit_development_interruption_is_not_an_algorithm_failure(tmp_path):
    (tmp_path/'cinegym_run.json').write_text(json.dumps(dict(backend='cpu',family='movein',
        difficulty='easy',seed=1,episode_sha256='known')))
    output=tmp_path/'case_jev';output.mkdir()
    (output/'summary.json').write_text(json.dumps(dict(completed=False,elapsed_shot_s=3.,
        shot_acceptance=dict(status='fail',accepted=False))))
    assert summarize_runs(tmp_path)['groups']['cpu/all']['failed']==1
    (tmp_path/'cinegym_interrupted.json').write_text(json.dumps(dict(reason='development job cancelled after final freeze')))
    report=summarize_runs(tmp_path)
    assert report['groups']['cpu/all']['unverified']==1
    assert report['groups']['cpu/all']['failed']==0
    assert report['runs'][0]['interrupted']


@pytest.mark.parametrize('completed,failure,safety,status',[
    (False,'',True,'unverified'),
    (False,'',None,'unverified'),
    (False,'RuntimeError: hard constraints infeasible',True,'fail'),
    (False,'',False,'fail'),
    (True,'',True,'pass'),
])
def test_wall_timeout_preserves_only_conclusive_backend_outcomes(tmp_path,completed,failure,safety,status):
    (tmp_path/'cinegym_run.json').write_text(json.dumps(dict(backend='cpu',family='movein',
        difficulty='easy',seed=1,episode_sha256='known')))
    (tmp_path/'cinegym_timeout.json').write_text(json.dumps(dict(timed_out=True)))
    output=tmp_path/'case_jev';output.mkdir()
    (output/'summary.json').write_text(json.dumps(dict(completed=completed,elapsed_shot_s=3.,
        failure=failure,safety_pass=safety,collision_samples=0,
        shot_acceptance=dict(status='pass' if completed else 'fail',accepted=completed))))
    row=summarize_runs(tmp_path)['runs'][0]
    assert row['status']==status
    assert row['partial_summary_timeout']==(status=='unverified')
