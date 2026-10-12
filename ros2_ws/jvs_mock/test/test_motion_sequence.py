"""Action-only files preserve camera geometry and compile smooth connections."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jvs_mock.motion_sequence import MinimumJerkBridge, bridge_duration
from jvs_mock.shot_execution import ShotClock
from jvs_mock.wbc_metrics import Metrics
from jvs_mock.wbc_trajectory import CameraTrajectory
from jvs_mock.whole_body_qp import Limits, whole_body_kinematics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from generate_shot_trajectories import motion_only_document


def load(path):
    p, r, _ = whole_body_kinematics(np.zeros(9), .002)
    return CameraTrajectory(path, p, r, np.eye(3))


@pytest.mark.parametrize('index', range(1, 5))
def test_action_files_have_no_connectors_and_identical_action_samples(index):
    full = json.loads((ROOT/f'trajectories/shot_combo{index}.json').read_text())
    sequence = json.loads((ROOT/f'trajectories/shot_combo{index}_actions.json').read_text())
    assert sequence == motion_only_document(full)
    assert 'samples' not in sequence
    assert sequence['duration_s'] == sum(s['duration_s'] for s in sequence['shots'])
    for source, action in zip(full['shots'], sequence['shots']):
        assert not any(key in action for key in ('start_s', 'end_s', 'motion_start_s', 'motion_end_s'))
        assert action['samples'][0]['time_from_start'] == 0.
        assert action['samples'][-1]['time_from_start'] == pytest.approx(action['duration_s'], rel=0., abs=1e-10)
        for sample in action['samples']:
            t = sample['time_from_start']+source['motion_start_s']
            original = min(full['samples'], key=lambda s: abs(s['time_from_start']-t))
            assert sample['pose'] == original['pose']


@pytest.mark.parametrize('index', range(1, 5))
def test_compiled_actions_match_full_geometry_and_are_c2_at_junctions(index):
    full = load(ROOT/f'trajectories/shot_combo{index}.json')
    sequence = load(ROOT/f'trajectories/shot_combo{index}_actions.json')
    np.testing.assert_allclose(sequence.subject, full.subject)
    for before, after in zip(full.shots, sequence.shots):
        duration = before['motion_end_s']-before['motion_start_s']
        for local in np.linspace(0., duration, 31):
            p0, r0, _ = full.sample(before['motion_start_s']+local)
            p1, r1, _ = sequence.sample(after['motion_start_s']+local)
            np.testing.assert_allclose(p0, p1, atol=2e-7)
            assert np.linalg.norm(Rotation.from_matrix(r1@r0.T).as_rotvec()) < 2e-7
            np.testing.assert_allclose(r1[:, 0], (sequence.subject-p1)/np.linalg.norm(sequence.subject-p1), atol=2e-7)
    for _, t, _ in sequence.pieces[:-1]:
        a, b = sequence.sample(t-1e-5), sequence.sample(t+1e-5)
        np.testing.assert_allclose(a[0], b[0], atol=1e-8)
        np.testing.assert_allclose(a[1], b[1], atol=1e-8)
        np.testing.assert_allclose(a[2], b[2], atol=1e-8)
        left_acc = (sequence.sample(t-1e-5)[2]-sequence.sample(t-2e-5)[2])/1e-5
        right_acc = (sequence.sample(t+2e-5)[2]-sequence.sample(t+1e-5)[2])/1e-5
        np.testing.assert_allclose(left_acc, right_acc, atol=2e-5)


def test_connectors_span_arbitrary_endpoint_gaps_with_bounded_speed_and_acceleration():
    first = (np.zeros(3), np.eye(3))
    last = (np.array([.6, .2, -.1]), Rotation.from_euler('xyz', [.1, -.2, 1.]).as_matrix())
    duration = bridge_duration(first, last)
    curve = MinimumJerkBridge(first, last, duration)
    samples = [curve.sample(t) for t in np.linspace(0., duration, 1001)]
    velocities = np.array([s[2] for s in samples])
    acceleration = np.diff(velocities, axis=0)/(duration/1000)
    assert np.max(np.linalg.norm(velocities[:, :3], axis=1)) <= .2+1e-8
    assert np.max(np.linalg.norm(velocities[:, 3:], axis=1)) <= .6+1e-8
    assert np.max(np.linalg.norm(acceleration[:, :3], axis=1)) <= .25+1e-6
    assert np.max(np.linalg.norm(acceleration[:, 3:], axis=1)) <= 1.+1e-6
    for endpoint, pose in zip((samples[0], samples[-1]), (first, last)):
        np.testing.assert_allclose(endpoint[0], pose[0], atol=1e-12)
        np.testing.assert_allclose(endpoint[1], pose[1], atol=1e-12)
        np.testing.assert_allclose(endpoint[2], 0., atol=1e-12)


def test_executor_builds_nonstationary_bridge_without_input_transition_samples(tmp_path):
    document = json.loads((ROOT/'trajectories/shot_combo1_actions.json').read_text())
    document.pop('scene_context')
    document['shots'] = [dict(type='neutral', duration_s=1., samples=[
        dict(time_from_start=0., pose=[0.]*6),
        dict(time_from_start=1., pose=[.1, 0., 0., 0., 0., .1])]),
        dict(type='neutral', duration_s=1., samples=[
        dict(time_from_start=0., pose=[.6, .2, 0., 0., 0., .3]),
        dict(time_from_start=1., pose=[.7, .2, 0., 0., 0., .4])])]
    document['duration_s'] = 2.
    path = tmp_path/'separated_actions.json'
    path.write_text(json.dumps(document))
    trajectory = load(path)
    transition = trajectory.automatic_transitions[0]
    start, end = transition['start_s'], transition['end_s']
    a, b, middle = trajectory.sample(start), trajectory.sample(end), trajectory.sample((start+end)/2)
    assert np.linalg.norm(b[0]-a[0]) > .5
    np.testing.assert_allclose(middle[0], (a[0]+b[0])/2, atol=1e-12)
    assert np.linalg.norm(middle[2][:3]) > 0.
    assert not trajectory.motion_mask((start+end)/2)
    assert trajectory.transition((start+end)/2)['end_s'] == pytest.approx(end)


def test_pair_similarity_requires_both_five_cm_and_matching_controller():
    from analyze_combo_pairs import compare
    first = dict(trajectory='full.json', source_sha256={'controller': 'a'},
                 position_error_rms=.01, position_error_max=.035, orientation_error_rms=.02,
                 tracking_accuracy_pass=True, approximate_success=True, completed=True, failure='',
                 accuracy_scope='shot_motion_intervals_only', scene_sha256='scene',
                 initial_base_pose=dict(x=1., z=2., yaw=.3, height=.002),
                 all_phases_minimum_command_clearance_m=.03)
    second = dict(first, trajectory='actions.json', position_error_rms=.015, position_error_max=.04)
    assert compare(first, second)['similarity_pass']
    second['source_sha256'] = {'controller': 'b'}
    assert not compare(first, second)['similarity_pass']
    second['source_sha256'] = first['source_sha256']
    second['position_error_max'] = .06
    assert not compare(first, second)['similarity_pass']
    second.update(position_error_max=.04, position_error_rms=.022)
    assert not compare(first, second)['similarity_pass']


@pytest.mark.parametrize('invalid', ['incomplete', 'failed', 'wrong_scope', 'missing_clearance',
                                    'unsafe_connector', 'different_scene', 'different_spawn'])
def test_pair_validation_rejects_incomplete_or_noncomparable_evidence(invalid):
    from analyze_combo_pairs import compare
    full = dict(trajectory='full.json', source_sha256={'controller': 'a'},
                position_error_rms=.005, position_error_max=.02, orientation_error_rms=.02,
                completed=True, failure='', tracking_accuracy_pass=True, approximate_success=True,
                accuracy_scope='shot_motion_intervals_only', scene_sha256='scene',
                initial_base_pose=dict(x=1., z=2., yaw=.3, height=.002),
                all_phases_minimum_command_clearance_m=.03)
    actions = deepcopy(full)
    actions['trajectory'] = 'actions.json'
    if invalid == 'incomplete':
        actions['completed'] = False
    elif invalid == 'failed':
        actions['failure'] = 'interrupted'
    elif invalid == 'wrong_scope':
        actions['accuracy_scope'] = 'full_execution'
    elif invalid == 'missing_clearance':
        actions.pop('all_phases_minimum_command_clearance_m')
    elif invalid == 'unsafe_connector':
        actions['all_phases_minimum_command_clearance_m'] = .0249
    elif invalid == 'different_scene':
        actions['scene_sha256'] = 'other-scene'
    else:
        actions['initial_base_pose']['yaw'] += .1
    assert not compare(full, actions)['similarity_pass']


def test_accuracy_clock_reacts_before_five_cm_and_keeps_pose_order():
    clock = ShotClock()
    for _ in range(300):
        old_time, old_rate = clock.time, clock.rate
        time, rate = clock.advance(.02, .04, .01, position_scale=.025, orientation_scale=.05)
        assert time-old_time == pytest.approx(.02*rate)
        assert abs(rate-old_rate) <= .008000001
    assert .15 < rate < .3
    assert rate == pytest.approx(1./(1.+(.04/.025)**2+(.01/.05)**2))


def test_record_inspection_rejects_a_scene_changed_after_execution(tmp_path):
    import hashlib
    from analyze_combo_pairs import inspect
    trajectories = tmp_path/'trajectories'
    trajectories.mkdir()
    trajectory = trajectories/'shot_combo1.json'
    trajectory.write_text('{}')
    stage = tmp_path/'scene.usda'
    stage.write_text('changed scene')
    folder = tmp_path/'results/shot_combo1'
    folder.mkdir(parents=True)
    metadata = dict(trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                    stage=str(stage), stage_sha256=hashlib.sha256(b'original scene').hexdigest())
    for name, document in (('run_metadata.json', metadata), ('reference.json', {}),
                           ('tracking_summary.json', {})):
        (folder/name).write_text(json.dumps(document))
    with pytest.raises(ValueError, match='changed scene'):
        inspect(folder, tmp_path)


def test_preparation_clock_has_no_old_heading_or_reach_switch():
    rates = []
    for heading, reach in ((.35-1e-5, .35-1e-5), (.35+1e-5, .35+1e-5)):
        clock = ShotClock()
        clock.rate = .35
        rates.append(clock.advance(.2, 0., 0., True, heading, reach)[1])
    assert abs(rates[1]-rates[0]) < 1e-4


@pytest.mark.parametrize('motion_error,passes', [(.005, True), (.06, False)])
def test_accuracy_excludes_transition_spikes_but_never_motion_spikes(tmp_path, motion_error, passes):
    trajectory = load(ROOT/'trajectories/shot_combo1.json')
    metrics = Metrics(tmp_path, Limits())
    metrics.trajectory = trajectory
    metrics.position_tolerance = metrics.position_max_tolerance = .05
    metrics.orientation_tolerance = .1
    try:
        for t in np.arange(0., trajectory.duration+.01, .1):
            active = bool(trajectory.motion_mask(t))
            metrics.record(t, .1, np.zeros(9), np.zeros(8), np.zeros(8), np.zeros(2), 0.,
                dict(position_error=motion_error if active else .3, orientation_error=0.,
                     qp_violation=0., solve_s=0.))
        summary = metrics.finish(True)
        assert summary['tracking_accuracy_pass'] is passes
        assert summary['accuracy_scope'] == 'shot_motion_intervals_only'
        assert summary['position_error_max'] == pytest.approx(motion_error)
        assert summary['transition_position_error_max'] == pytest.approx(.3)
        assert summary['full_execution_position_error_max'] == pytest.approx(.3)
        assert summary['motion_samples']+summary['transition_samples'] == summary['samples']
    finally:
        metrics.close()


@pytest.mark.parametrize('mutation', ['duplicate_time', 'bad_duration', 'execution_timestamp', 'top_samples', 'nan_pose'])
def test_invalid_sequences_are_rejected_before_execution(tmp_path, mutation):
    document = deepcopy(json.loads((ROOT/'trajectories/shot_combo1_actions.json').read_text()))
    shot = document['shots'][0]
    if mutation == 'duplicate_time':
        shot['samples'][1]['time_from_start'] = 0.
    elif mutation == 'bad_duration':
        document['duration_s'] += 1.
    elif mutation == 'execution_timestamp':
        shot['start_s'] = 0.
    elif mutation == 'top_samples':
        document['samples'] = []
    else:
        shot['samples'][1]['pose'][0] = float('nan')
    path = tmp_path/'invalid.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load(path)


def test_reanchoring_action_sequence_preserves_shots_and_target(tmp_path):
    document = json.loads((ROOT/'trajectories/shot_combo2_actions.json').read_text())
    document['initial_base_pose'].update(x=3.7, z=1.8, yaw=.55)
    path = tmp_path/'respawn.json'
    path.write_text(json.dumps(document))
    changed = load(path)
    for shot in changed.shots:
        for t in np.linspace(shot['motion_start_s'], shot['motion_end_s'], 15):
            position, rotation, _ = changed.sample(t)
            np.testing.assert_allclose(rotation[:, 0], (changed.subject-position)/np.linalg.norm(changed.subject-position), atol=2e-7)
