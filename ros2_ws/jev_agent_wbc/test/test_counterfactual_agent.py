"""Safety boundary, causal perception and bounded asynchronous policy regressions."""
from concurrent.futures import Future
from dataclasses import replace
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.contracts import Envelope, Validation, digest
from jev_agent_wbc.reference import OriginalReference, DeformedReference
from jev_agent_wbc.perception import (SubjectEvidence, mask_evidence, project_subject,
    camera_lidar_transform, visual_consequences, phase_progress)
from jev_agent_wbc.candidates import (Candidate, Approval, SearchBudget, path_hash,
    verify_candidate, installation_guard, contract_comparison, build_candidates)
from jev_agent_wbc.jev import CandidateDecider, AsyncJev, compact_evidence
from jev_agent_wbc.sam_observer import LatestMaskWorker, rgb_from_message
from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup_case():
    scene = json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    original = OriginalReference(ROOT/'scenarios/movein/task.json', scene['subject']['position'])
    state = np.array(scene['initial_state'])
    observed = dict(format='jvs_lidar_surfels/v1', surfel_centers=np.array([[3., 3., .2]]),
        surfel_radius=.025, surfel_velocities=np.zeros((1, 3)), observation_stamp=0., minimum_clearance_m=.025)
    return original, state, observed, scene['base_z']


def candidate_for(original, state, base_z, start=2.):
    envelope = Envelope(original.sha256, 'proposal', 'test', 0., start, 8., 1.,
                        {'radial_m': [0., .2]}, ('subject_aim', 'fixed_progress'))
    path = DeformedReference(original, envelope, [.1, 0, 0, 0, 0, 0])
    p, r, _ = whole_body_kinematics(state, base_z)
    trace = [dict(t=t, position=p.tolist(), quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
                  state=state.tolist(), joint_margin_rad=.3) for t in (0., 1., 2., 4., 8.)]
    validation = Validation(True, 'test_prediction', cost=0., camera_trace=trace)
    return Candidate(path, validation, visual_consequences(original, trace, intrinsics=[1., 1., .5, .5]), 0., 'framing')


def approval_for(candidate, original):
    return Approval(candidate.id, digest(candidate.card()), path_hash(original), original.sha256, .99, 1)


def test_capture_time_transform_changes_with_arm_configuration():
    wc, wl = np.eye(4), np.eye(4)
    wl[:3, 3] = [1, 0, 0]
    wc[:3, 3] = [0, 2, 0]
    first = camera_lidar_transform(wc, wl)
    wc[:3, :3] = Rotation.from_euler('z', .5).as_matrix()
    second = camera_lidar_transform(wc, wl)
    assert not np.allclose(first, second)
    np.testing.assert_allclose(wc@second, wl, atol=1e-12)


def test_invalid_transform_is_rejected():
    wc = np.eye(4)
    wc[0, 0] = 2
    with pytest.raises(ValueError):
        camera_lidar_transform(wc, np.eye(4))


def test_mask_area_loss_is_not_distance_evidence():
    obs = SubjectEvidence('subject')
    obs.update_geometry(dict(valid=True, position=[2, 0, 0], stamp_s=1., stationary=True))
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:10] = True
    obs.update_visual(mask_evidence(mask, 'subject', 1., identity_valid=True, occluded=True))
    snapshot = obs.snapshot(1.1)
    assert snapshot['visual']['visible_area'] == .125
    assert snapshot['diagnosis'] == 'occlusion'
    assert snapshot['scale_measurement_valid'] is False


@pytest.mark.parametrize('stamp,valid', [(0., False), (2., False), (1., True)])
def test_stale_and_future_visual_stamps_are_unknown(stamp, valid):
    obs = SubjectEvidence('s')
    obs.update_visual(mask_evidence(np.ones((2, 2)), 's', stamp, identity_valid=True))
    assert (obs.snapshot(1.1)['visual']['status'] == 'valid') is valid


def test_invalid_current_does_not_resurrect_last_trusted():
    obs = SubjectEvidence('s')
    obs.update_visual(mask_evidence(np.ones((2, 2)), 's', 1., identity_valid=True))
    obs.update_visual(mask_evidence(None, 's', 1.1, identity_valid=True))
    snapshot = obs.snapshot(1.2)
    assert snapshot['visual']['status'] == 'unknown'
    assert snapshot['last_trusted_visual']['stamp_s'] == 1.
    assert not obs.update_visual(mask_evidence(np.ones((2, 2)), 's', 1., identity_valid=True))
    with pytest.raises(ValueError, match='identity'):
        obs.update_visual(mask_evidence(None, 'other', 1.2))


def test_unsynchronized_or_unassociated_sources_cannot_validate_scale():
    obs = SubjectEvidence('s')
    mask = np.zeros((8, 8), bool)
    mask[2:5, 2:5] = True
    obs.update_visual(mask_evidence(mask, 's', 1., identity_valid=True, occluded=False))
    obs.update_geometry(dict(valid=True, position=[2, 0, 0], stamp_s=1.3, stationary=True))
    assert not obs.snapshot(1.4)['time_aligned']
    assert not obs.snapshot(1.4)['scale_measurement_valid']


def test_projection_respects_original_offcenter_framing_and_pan(setup_case):
    original, state, scene, z = setup_case
    task = original.task
    origin = task['reference_camera_trajectory']['samples'][0]['position']
    for row in task['reference_camera_trajectory']['samples']:
        row['position'] = origin
        row['quaternion_xyzw'] = Rotation.from_euler('z', .02*row['t']).as_quat().tolist()
    task['shot_phase'] = [dict(name='pan', motion='pan', start_s=0., end_s=8.)]
    pan = OriginalReference(task, original.subject)
    p, r, _ = pan.sample(4.)
    measured, nominal = phase_progress(pan, 4., p, r, pan.subject)
    assert measured == pytest.approx(.5)
    assert nominal == pytest.approx(.5)
    uv = project_subject(p, r, pan.subject, [1, 1, .5, .5])
    assert uv is not None and not np.allclose(uv, [.5, .5])
    row = dict(t=4., position=p, quaternion_xyzw=Rotation.from_matrix(r).as_quat(), joint_margin_rad=.3)
    outcome = visual_consequences(pan, [row], intrinsics=[1, 1, .5, .5])
    assert outcome['summary']['composition_error'] == pytest.approx(0.)
    assert outcome['summary']['phase_progress_error'] == pytest.approx(0.)


def test_execution_consequences_are_not_modified_target_consequences(setup_case):
    original, state, scene, z = setup_case
    candidate = candidate_for(original, state, z)
    assert candidate.consequences['source'] == 'predicted_execution_FK_projection'
    assert candidate.consequences['summary']['range_scale_error'] > .05
    assert candidate.consequences['projected_scale'] == 'unknown'
    assert candidate.consequences['occlusion'] == 'unknown'
    no_camera = visual_consequences(original, candidate.validation.camera_trace)
    assert no_camera['summary']['composition_error'] is None


@pytest.mark.parametrize('mutation', ['values', 'activation', 'return', 'consequences'])
def test_approval_does_not_authorize_smaller_retimed_or_altered_candidate(setup_case, mutation):
    original, state, scene, z = setup_case
    candidate = candidate_for(original, state, z)
    approval = approval_for(candidate, original)
    if mutation == 'values':
        candidate.path.values[0] *= .5
    elif mutation == 'activation':
        candidate.path.activation_s += .1
    elif mutation == 'return':
        candidate.path.return_s = 7.
    else:
        candidate.consequences['summary']['range_scale_error'] *= 2
    result = verify_candidate(candidate, approval, original, scene, state, np.zeros(8), .1, z, Limits())
    assert result['error'] == 'approval_binding_mismatch'
    assert result['checks']['target_authorized'] == 'fail'


def test_late_candidate_cannot_be_rebased(setup_case):
    original, state, scene, z = setup_case
    candidate = candidate_for(original, state, z)
    result = verify_candidate(candidate, approval_for(candidate, original), original, scene, state,
                              np.zeros(8), 2., z, Limits())
    assert result['error'] == 'candidate_expired_no_retiming_allowed'
    assert candidate.path.activation_s == 2.


def test_contract_never_hides_next_phase_degradation_with_global_max(setup_case):
    original, state, scene, z = setup_case
    candidate = candidate_for(original, state, z)
    approved = copy.deepcopy(candidate.consequences)
    current = copy.deepcopy(approved)
    phase = list(current['phases'])[-1]
    current['phases'][phase]['view_angle_error_rad'] = approved['phases'][phase]['view_angle_error_rad']+.5
    gates = contract_comparison(approved, current, SearchBudget())
    assert gates[phase+':view_angle_error_rad'] == 'fail'
    assert gates['occlusion'] == 'unknown'


def test_latest_install_guard_rejects_new_obstacle_and_changed_parent(setup_case):
    original, state, scene, z = setup_case
    candidate = candidate_for(original, state, z)
    result = dict(candidate=candidate, approval=approval_for(candidate, original),
                  snapshot_time_s=0., scene=scene, validation=candidate.validation)
    assert installation_guard(result, original, scene, state, .1, SearchBudget()) is None
    changed = {**scene, 'surfel_centers': np.vstack([scene['surfel_centers'], [0, .2, .5]]),
               'surfel_velocities': np.zeros((2, 3))}
    assert installation_guard(result, original, changed, state, .1, SearchBudget()) == 'scene_occupancy_changed'
    assert installation_guard(result, candidate.path, scene, state, .1, SearchBudget()) == 'active_parent_changed'
    moved = state.copy()
    moved[0] += .1
    assert installation_guard(result, original, scene, moved, .1, SearchBudget()) == 'robot_left_verified_state_neighborhood'


def test_candidate_search_has_hard_rollout_budget_and_no_motion_authority(setup_case, monkeypatch):
    from jev_agent_wbc import candidates
    original, state, scene, z = setup_case
    calls = []
    monkeypatch.setattr(candidates, 'rank_candidate', lambda *a, **k: Validation(True, 'rank', cost=0.))
    def validate(path, *args, **kwargs):
        calls.append(path)
        if 'checkpoint' in kwargs:
            kwargs['checkpoint'].update(state=state, controller=SimpleNamespace(previous=np.zeros(8)),
                                        commanded_q=state[3:])
        return candidate_for(original, state, z).validation
    monkeypatch.setattr(candidates, 'validate_rollout', validate)
    result = build_candidates(original, original, 'joint', scene, state, np.zeros(8), 0., z, Limits(),
                              budget=SearchBudget(max_rollouts=3))
    assert len(calls) == 4  # one shared prefix, three counterfactual branches
    assert len(result['candidates']) == 3
    assert all('Unapproved' in c.path.envelope.rationale for c in result['candidates'])
    assert any(np.count_nonzero(c.path.values) == 2 for c in result['candidates'])


def test_jev_two_calls_and_no_chained_same_call_dependency(setup_case):
    original, state, scene, z = setup_case
    card = candidate_for(original, state, z).card()
    class Client:
        def __init__(self): self.calls = []
        def ask(self, evidence, queries):
            self.calls.append(queries)
            return {key: ({'choice': 'framing' if key == 'strategy' else 'c0'}
                if q['type'] == 'choice' else {'noul': .99}) for key, q in queries.items()}
    client = Client()
    decider = CandidateDecider(client)
    assert decider.decide({'stage': 'strategy'})['strategy'] == 'framing'
    assert decider.decide({'stage': 'selection', 'candidates': [card]})['candidate_id'] == card['id']
    assert len(client.calls) == 2
    assert set(client.calls[1]) == {'candidate'}
    assert compact_evidence({'candidates': [card]})['candidates'][0]['trajectory'] == card['trajectory']


def test_explicit_semantic_refusal_cannot_execute(setup_case):
    original, state, scene, z = setup_case
    card = candidate_for(original, state, z).card()
    class Client:
        def ask(self, evidence, queries):
            return {'candidate': {'choice': 'refuse'}}
    with pytest.raises(RuntimeError, match='no_complete_candidate_accepted'):
        CandidateDecider(Client()).decide({'stage': 'selection', 'candidates': [card]})


def test_sam_worker_no_queue_and_preserves_capture_stamp():
    import threading
    release = threading.Event()
    def slow(rgb, stamp):
        release.wait(1.)
        return mask_evidence(rgb[:, :, 0] > 0, 's', stamp)
    worker = LatestMaskWorker(slow)
    try:
        assert worker.submit(np.ones((4, 4, 3)), 1.23)
        started = time.monotonic()
        assert worker.poll() is None
        assert not worker.submit(np.ones((4, 4, 3)), 2.)
        assert time.monotonic()-started < .05
        release.set()
        worker.future.result(timeout=1.)
        time.sleep(.02)  # Delayed polling must not be reported as GPU inference.
        result = worker.poll()
        assert result['stamp_s'] == 1.23
        assert result['dropped_frames'] == 1
        assert result['delivery_wait_s'] >= .018
        assert result['observation_wall_s'] >= result['inference_wall_s']+.018
    finally:
        release.set()
        worker.close()


def test_rgb_adapter_handles_padding_and_bgr():
    message = SimpleNamespace(encoding='bgr8', height=1, width=2, step=8,
                              data=bytes([1, 2, 3, 4, 5, 6, 99, 99]))
    np.testing.assert_array_equal(rgb_from_message(message), [[[3, 2, 1], [6, 5, 4]]])


def test_pending_agent_never_waits_in_control_update(setup_case, tmp_path):
    from jev_agent_wbc.engine import CinematicEngine
    original, state, scene, z = setup_case
    engine = CinematicEngine(original, scene, tmp_path, mode='rule')
    try:
        engine.math_future = Future()  # intentionally never completes
        engine.math_kind = 'candidates'
        started = time.monotonic()
        for i in range(3):
            engine.update(state, scene, i*.02, z)
        assert time.monotonic()-started < .1
        assert engine.active is original
    finally:
        engine.close()


def test_shared_execution_prefix_matches_unshared_rollout(setup_case):
    from jev_agent_wbc.planning import validate_rollout
    original, state, scene, z = setup_case
    candidate = candidate_for(original, state, z, start=.2)
    prefix = {}
    validate_rollout(original, scene, state, np.zeros(8), 0., z, Limits(),
                     horizon=.2, step=.1, checkpoint=prefix, record_trace=True)
    full = validate_rollout(candidate.path, scene, state, np.zeros(8), 0., z, Limits(),
                            horizon=.4, step=.1, record_trace=True)
    branched = validate_rollout(candidate.path, scene, state, np.zeros(8), 0., z, Limits(),
                                horizon=.4, step=.1, record_trace=True, resume=prefix)
    assert full.feasible and branched.feasible
    np.testing.assert_allclose(full.terminal_state, branched.terminal_state, atol=1e-9)
    assert full.original_loss_integral == pytest.approx(branched.original_loss_integral, abs=1e-10)


def test_lost_sensor_support_requires_new_semantic_decision(setup_case):
    original, state, scene, z = setup_case
    candidate = candidate_for(original, state, z)
    approval = replace(approval_for(candidate, original), observation_support={'visual': 'valid'})
    result = verify_candidate(candidate, approval, original, scene, state, np.zeros(8), .1, z, Limits(),
                              evidence={'visual': {'status': 'unknown'}})
    assert result['error'] == 'approved_observation_became_unknown:visual'


def test_growing_obstacle_extent_invalidates_same_centers(setup_case):
    original, state, scene, z = setup_case
    candidate = candidate_for(original, state, z)
    result = dict(candidate=candidate, approval=approval_for(candidate, original),
                  snapshot_time_s=0., scene=scene, validation=candidate.validation)
    changed = {**scene, 'surfel_radius': .2}
    assert installation_guard(result, original, changed, state, .1, SearchBudget()) == 'scene_extent_changed'


def test_subject_update_does_not_redefine_original_visual_target(setup_case):
    original, state, scene, z = setup_case
    p, r, _ = original.sample(0.)
    row = dict(t=0., position=p.tolist(), quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
               joint_margin_rad=.3)
    outcome = visual_consequences(original, [row],
        evidence={'geometry': dict(status='valid', position=(original.subject+[0., .2, 0.]).tolist(), source='lidar')},
        intrinsics=[1., 1., .5, .5])
    assert outcome['summary']['composition_error'] > .01


def test_reference_cache_cannot_be_mutated_by_consumer(setup_case):
    original, state, scene, z = setup_case
    p, r, ff = original.sample(1.)
    expected = p.copy()
    p[:] = 99
    r[:] = 0
    ff[:] = 99
    np.testing.assert_allclose(original.sample(1.)[0], expected)


def test_aligned_sensors_still_require_projection_consistency():
    obs = SubjectEvidence('s')
    mask = np.zeros((100, 100), bool)
    mask[40:60, 40:60] = True
    record = mask_evidence(mask, 's', 1., identity_valid=True, occluded=False)
    record.update(camera_position=[0., 0., 0.], camera_quaternion_xyzw=[0., 0., 0., 1.],
                  intrinsics=[1., 1., .5, .5])
    obs.update_visual(record)
    obs.update_geometry(dict(valid=True, position=[2., 0., 0.], stamp_s=1., stationary=True))
    assert obs.snapshot(1.1)['scale_measurement_valid']
    obs.update_geometry(dict(valid=True, position=[2., 1., 0.], stamp_s=1.05, stationary=True))
    record['stamp_s'] = 1.05
    obs.update_visual(record)
    assert obs.snapshot(1.1)['fusion_consistent'] is False
    assert obs.snapshot(1.1)['diagnosis'] == 'projection_association_mismatch'
    assert not obs.snapshot(1.1)['scale_measurement_valid']


def test_long_shots_have_bounded_recovering_leases_and_rank_after_activation(setup_case, monkeypatch):
    from jev_agent_wbc import candidates
    original, state, scene, z = setup_case
    task = original.task
    for row in task['reference_camera_trajectory']['samples']:
        row['t'] *= 2
    for phase in task['shot_phase']:
        phase['start_s'] *= 2
        phase['end_s'] *= 2
    longer = OriginalReference(task, original.subject)
    starts = []
    def rank(path, scene, robot, previous, t, *args, **kwargs):
        starts.append(t)
        assert t == path.activation_s
        return Validation(True, 'test', cost=0.)
    def validate(path, *args, **kwargs):
        if 'checkpoint' in kwargs:
            kwargs['checkpoint'].update(state=state, controller=SimpleNamespace(previous=np.zeros(8)),
                                        commanded_q=state[3:])
        else:
            assert kwargs['horizon'] == 8.
            assert not kwargs['check_terminal']
        return candidate_for(longer, state, z).validation
    monkeypatch.setattr(candidates, 'rank_candidate', rank)
    monkeypatch.setattr(candidates, 'validate_rollout', validate)
    result = build_candidates(longer, longer, 'framing', scene, state, np.zeros(8), 0., z, Limits())
    assert starts and len(starts) <= 8
    for candidate in result['candidates']:
        assert candidate.path.envelope.end_s == 8.
        assert candidate.path.envelope.terminal == 'recover'
        assert candidate.path.duration == 16.
        np.testing.assert_allclose(candidate.path.parameters(8.), 0.)
        np.testing.assert_allclose(candidate.path.sample(8.)[0], longer.sample(8.)[0])


def test_timestamp_roundoff_is_not_a_future_sensor_reading(setup_case):
    original, state, scene, z = setup_case
    candidate = candidate_for(original, state, z)
    fresh = {**scene, 'observation_stamp': np.nextafter(.1, np.inf)}
    result = dict(candidate=candidate, approval=approval_for(candidate, original),
                  snapshot_time_s=0., scene=scene, validation=candidate.validation)
    assert installation_guard(result, original, fresh, state, .1, SearchBudget()) is None
    future = {**scene, 'observation_stamp': .2}
    assert installation_guard(result, original, future, state, .1, SearchBudget()) == 'scene_stale'


def test_installation_interval_uses_execution_resolution(setup_case):
    from jev_agent_wbc.planning import validate_rollout
    original, state, scene, z = setup_case
    full = validate_rollout(original, scene, state, np.zeros(8), 0., z, Limits(),
                            step=.02, horizon=.4, record_trace=True)
    mixed = validate_rollout(original, scene, state, np.zeros(8), 0., z, Limits(),
                             step=.1, horizon=.4, fine_until_s=.2, record_trace=True)
    full_head = [r for r in full.camera_trace if r['t'] <= .2+1e-8]
    mixed_head = [r for r in mixed.camera_trace if r['t'] <= .2+1e-8]
    assert len(full_head) == len(mixed_head) == 11
    np.testing.assert_allclose([r['state'] for r in full_head], [r['state'] for r in mixed_head], atol=1e-10)


def test_failed_search_is_counted_and_visual_feedback_cannot_starve_events(setup_case, tmp_path):
    from jev_agent_wbc.engine import CinematicEngine
    original, state, scene, z = setup_case
    engine = CinematicEngine(original, scene, tmp_path, mode='rule')
    try:
        future = Future()
        future.set_result(dict(error='no_candidate_passed_physical_validation'))
        engine.math_future, engine.math_kind = future, 'candidates'
        engine.update(state, scene, .1, z)
        assert engine.summary()['validation_attempts'] == 1
        assert engine.summary()['planning_feasibility_rate'] == 0.
        attempted = engine.last_attempt_s
        engine._feedback('visual_unknown', .2, finish_pipeline=False)
        assert engine.last_attempt_s == attempted
    finally:
        engine.close()
