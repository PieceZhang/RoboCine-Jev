"""Numerical equivalence and causal continuation of the execution predictor."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from jev_agent_wbc.jvs_kinematics import forward_kinematics, JOINT_LOWER, JOINT_UPPER, _rotation
from jev_agent_wbc.whole_body_qp import Limits
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.planning import make_wbc

ROOT = Path(__file__).resolve().parents[1]


def test_batched_fk_jacobian_matches_finite_difference():
    rng = np.random.default_rng(17)
    for _ in range(12):
        q = rng.uniform(JOINT_LOWER+.05, JOINT_UPPER-.05)
        p, r, jac = forward_kinematics(q, with_jacobian=True)
        numerical = np.empty((6, 6))
        for j in range(6):
            delta = np.eye(6)[j]*1e-6
            pa, ra = forward_kinematics(q+delta)
            pb, rb = forward_kinematics(q-delta)
            numerical[:, j] = np.r_[(pa-pb)/2e-6, Rotation.from_matrix(ra@rb.T).as_rotvec()/2e-6]
        np.testing.assert_allclose(jac, numerical, atol=5e-9)
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        np.testing.assert_allclose(_rotation(axis, .83), Rotation.from_rotvec(axis*.83).as_matrix(), atol=1e-15)


def test_unrelated_yaw_reserve_does_not_limit_longitudinal_barrier():
    fixture = json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    original = OriginalReference(ROOT/'scenarios/movein/task.json', fixture['subject']['position'])
    state = np.array(fixture['initial_state'])
    scene = dict(format='jvs_lidar_surfels/v1', surfel_centers=np.array([[3., 0., .2]]),
                 surfel_radius=.025, observation_stamp=0., minimum_clearance_m=.025)
    normal = np.zeros((1, 1, 9)); normal[0, 0, 0] = -1.
    limits = Limits(); slow = replace(limits, acceleration=limits.acceleration.copy())
    slow.acceleration[1] *= .01
    first = make_wbc(limits, scene, original).static_barrier_terms(state, np.ones((1, 1)), normal)
    second = make_wbc(slow, scene, original).static_barrier_terms(state, np.ones((1, 1)), normal)
    np.testing.assert_allclose(first[1], second[1], atol=1e-12)


def test_redundancy_pruning_preserves_qp_optimum_and_all_constraints():
    from scipy import sparse
    import osqp
    from jev_agent_wbc.shot_predictive_qp import remove_box_redundancy
    rng = np.random.default_rng(30)
    extra = rng.normal(size=(300, 8))
    lo, hi = np.full(8, -.1), np.full(8, .2)
    # 300 inequalities contain the whole actuator box. The coupled wheel-like
    # row below is active at the optimum and must remain.
    rows = np.vstack([np.eye(8), np.array([1., 1., 0, 0, 0, 0, 0, 0]), extra])
    lower = np.r_[lo, -.3, -np.abs(extra).sum(axis=1)]
    upper = np.r_[hi, .1, np.abs(extra).sum(axis=1)]
    a, l, u = remove_box_redundancy(rows, lower, upper)
    assert len(a) == 9
    answers = []
    for aa, ll, uu in [(rows, lower, upper), (a, l, u)]:
        solver = osqp.OSQP()
        solver.setup(P=sparse.eye(8, format='csc'), q=-np.ones(8),
                     A=sparse.csc_matrix(aa), l=ll, u=uu, verbose=False,
                     eps_abs=1e-9, eps_rel=1e-9, polishing=True)
        result = solver.solve()
        assert result.info.status_val == 1
        answers.append(result.x)
    np.testing.assert_allclose(*answers, atol=2e-8)
    assert np.max(lower-rows@answers[-1]) <= 1e-8
    assert np.max(rows@answers[-1]-upper) <= 1e-8


def test_visibility_duration_is_independent_of_prediction_sample_count():
    from scipy.spatial.transform import Rotation
    from jev_agent_wbc.perception import visual_consequences
    from jev_agent_wbc.candidates import SearchBudget, contract_comparison
    fixture = json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    original = OriginalReference(ROOT/'scenarios/movein/task.json', fixture['subject']['position'])
    def features(step):
        trace = [dict(t=float(t), position=[0., -4., .8],
            quaternion_xyzw=Rotation.identity().as_quat().tolist(), joint_margin_rad=.3)
            for t in np.linspace(0., 1., round(1/step)+1)]
        return visual_consequences(original, trace, intrinsics=[1., 1., .5, .5])
    coarse, fine = features(.1), features(.02)
    assert fine['summary']['center_out_of_frame_samples'] > coarse['summary']['center_out_of_frame_samples']
    assert abs(fine['summary']['center_out_of_frame_duration_s']-1.) < 1e-12
    assert contract_comparison(coarse, fine, SearchBudget())['center_visibility'] == 'pass'


def test_local_search_evidence_matches_observed_clearance_direction():
    from jev_agent_wbc.candidates import local_repair_evidence
    fixture = json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    original = OriginalReference(ROOT/'scenarios/movein/task.json', fixture['subject']['position'])
    state = np.array(fixture['initial_state'])
    scene = dict(format='jvs_lidar_surfels/v1', surfel_centers=np.array([[.7, 0., .8]]),
                 surfel_radius=.025, observation_stamp=0., minimum_clearance_m=.025)
    evidence = local_repair_evidence(original, scene, state, 0., fixture['base_z'])
    radial = {tuple(p['direction']):p['clearance_change_m'] for p in evidence['probes']}
    assert radial[(1, 0, 0)] > 0
    assert radial[(-1, 0, 0)] < 0


def test_unchanged_target_is_not_counted_as_an_installed_repair(tmp_path):
    from concurrent.futures import Future
    import time
    from jev_agent_wbc.engine import CinematicEngine
    from jev_agent_wbc.candidates import Candidate
    from jev_agent_wbc.contracts import Envelope,Validation
    from jev_agent_wbc.reference import DeformedReference
    fixture = json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    original = OriginalReference(ROOT/'scenarios/movein/task.json', fixture['subject']['position'])
    state = np.array(fixture['initial_state'])
    scene = dict(format='jvs_lidar_surfels/v1', surfel_centers=np.array([[3., 3., .2]]),
                 surfel_radius=.025, observation_stamp=0., minimum_clearance_m=.025)
    envelope = Envelope(original.sha256,'noop','test',0.,2.,8.,1.,{})
    candidate = Candidate(DeformedReference(original,envelope,np.zeros(6)),Validation(True,'test'),{},0.,'framing')
    engine = CinematicEngine(original,scene,tmp_path,mode='rule')
    try:
        engine.pending_candidates={candidate.id:candidate}
        future=Future();future.set_result(dict(stage='selection',candidate_id=candidate.id,selected_probability=None))
        engine.jev.future=future
        engine.pipeline_started_wall=engine.jev.submitted_wall=time.monotonic()
        engine.update(state,scene,.1,fixture['base_z'])
        assert engine.active is original and engine.installed_plans == 0
        assert engine.math_future is None
        assert engine.pipeline_outcomes[-1]['outcome']=='continued'
    finally:
        engine.close()


def test_delayed_mask_uses_capture_time_geometry_not_latest_subject_pose():
    from jev_agent_wbc.perception import SubjectEvidence, mask_evidence
    obs = SubjectEvidence('s')
    obs.update_geometry(dict(valid=True, position=[2., 0., 0.], stamp_s=1., stationary=False))
    obs.update_geometry(dict(valid=True, position=[2., 1., 0.], stamp_s=1.4, stationary=False))
    mask = np.zeros((100, 100), bool)
    mask[40:60, 40:60] = True
    visual = mask_evidence(mask, 's', 1., identity_valid=True, occluded=None)
    visual.update(camera_position=[0., 0., 0.], camera_quaternion_xyzw=[0., 0., 0., 1.],
                  intrinsics=[1., 1., .5, .5])
    obs.update_visual(visual)
    snapshot = obs.snapshot(1.45)
    assert snapshot['fusion_consistent'] is True
    assert snapshot['aligned_geometry']['position'] == [2., 0., 0.]
    assert snapshot['geometry']['position'] == [2., 1., 0.]
    assert snapshot['scale_measurement_valid'] is False
    assert obs.snapshot(1.7)['time_aligned'] is False
    obs.update_geometry(dict(valid=False, stamp_s=1.5))
    assert obs.snapshot(1.55)['time_aligned'] is False


def test_delayed_geometry_matching_is_bounded_and_cannot_skip_invalid_sample():
    from jev_agent_wbc.perception import SubjectEvidence, mask_evidence
    obs = SubjectEvidence('s')
    for i in range(100):
        obs.update_geometry(dict(valid=i != 90, position=[2., 0., 0.], stamp_s=i*.01))
    assert len(obs.geometry_history) == 64
    obs.update_visual(mask_evidence(np.ones((2, 2)), 's', .9, identity_valid=True))
    assert obs.snapshot(1.)['time_aligned'] is False


def test_pure_pan_evaluation_uses_rotation_progress_without_retiming_errors():
    from jev_agent_wbc.evaluation import trajectory_summary
    task = json.loads((ROOT/'scenarios/movein/task.json').read_text())
    task['shot_phase'] = [dict(name='pan', motion='pan', start_s=0., end_s=8.)]
    task['reference_camera_trajectory']['samples'] = [dict(t=t, position=[.25, 0., .8],
        quaternion_xyzw=Rotation.from_euler('z', .1*t).as_quat().tolist()) for t in [0., 4., 8.]]
    original = OriginalReference(task, [2., 0., .8])
    for fraction in [1., .5]:
        rows = [dict(t=float(t), camera_position=[.25, 0., .8],
            camera_quaternion_xyzw=Rotation.from_euler('z', fraction*.1*t).as_quat().tolist(),
            truth_clearance_m=1., command=[0.]*8, info=dict(joint_margin_rad=.3))
            for t in np.linspace(0., 8., 81)]
        summary = trajectory_summary(rows, original, {})
        phase = summary['phases'][0]
        assert phase['geometric_progress'] == fraction
        assert phase['completed'] == (fraction == 1.)
        assert abs(phase['end_orientation_error_rad']-(1.-fraction)*.8) < 1e-8
