"""A historical path crossing and a safety-buffer intrusion cannot prove collision."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_dynamic_demos import surface
from audit_counterfactual_collision import base_collision_planes, evaluate, same_runtime_sources
from jvs_mock.dynamic_scene import DynamicScene, model_axis_rotation
from jvs_mock.kc_svsdf import primitive_sdf


@pytest.mark.parametrize('axis,index', [('x', 0), ('y', 1), ('z', 2)])
def test_cylinder_surface_and_solid_agree_on_the_declared_axis(axis, index):
    model = dict(type='cylinder', axis=axis, radius=.035, height=.4)
    points, _ = surface(model, spacing=.01)
    extent = np.max(abs(points), axis=0)
    assert extent[index] == pytest.approx(.2)
    np.testing.assert_allclose(np.delete(extent, index), .035, atol=.001)
    canonical = points @ model_axis_rotation(model)
    np.testing.assert_allclose(primitive_sdf(canonical, 'cylinder', [.035, .4]), 0., atol=1e-12)
    inside = np.zeros(3); inside[index] = .19
    assert primitive_sdf(inside @ model_axis_rotation(model), 'cylinder', [.035, .4]) < 0.


def fixture(tmp_path, stationary=False, positions=None):
    import csv
    baseline = tmp_path / 'baseline'; baseline.mkdir()
    rows = []
    for t in np.arange(0., 2.001, .2):
        rows.append(dict(wall_elapsed_s=t, time_s=t, state_x=0. if stationary else .5*t,
            state_y=0., state_yaw=0., base_z=0., base_rotvec_x=0., base_rotvec_y=0., base_rotvec_z=0.,
            **{f'state_q{i}': 0. for i in range(1, 7)}, phase='tracking', motion_active=1, dt_s=.2))
    with (baseline / 'telemetry.csv').open('w') as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (baseline / 'trajectory.json').write_text(json.dumps(dict(initial_base_pose=dict(x=0., z=0., height=0., yaw=0.))))
    (baseline / 'tracking_summary.json').write_text(json.dumps(dict(completed=True, failure='')))
    (baseline / 'run.json').write_text(json.dumps(dict(source_sha256={})))
    scene = tmp_path / 'scene.json'
    positions = positions or [[.5, 0., .13], [.5, 0., .13]]
    scene.write_text(json.dumps(dict(format='jvs_dynamic_scene/v1', frame='initial_base', obstacles=[
        dict(name='Actor', model=dict(type='sphere', radius=.05), trajectory=[
            dict(time_s=t, position=p) for t, p in zip([0., 2.], positions)])])))
    return baseline, scene


def test_a_simultaneous_solid_overlap_proves_the_nominal_collision(tmp_path):
    baseline, scene = fixture(tmp_path)
    report = evaluate(baseline, scene, 'base')
    assert report['simultaneous_collision_pass']
    assert report['obstacles'][0]['minimum_uninflated_solid_gap_m'] < -.005


def test_one_collision_cannot_satisfy_repeated_encounters(tmp_path):
    baseline, scene = fixture(tmp_path)
    report = evaluate(baseline, scene, 'base', min_encounters=3)
    assert report['obstacles'][0]['qualifying_encounter_count'] == 1
    assert not report['simultaneous_collision_pass']


@pytest.mark.parametrize('demo,body,count', [('crossing', 'arm', 1), ('overtake', 'base', 3)])
def test_configured_scenes_hit_measured_nominal_robot_at_matching_times(tmp_path, demo, body, count):
    import hashlib
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'test/fixtures/nominal_dynamic_encounters'
    case = json.loads((evidence / 'provenance.json').read_text())['cases'][demo]
    trajectory = root / 'trajectories' / case['trajectory']
    # A changed camera trajectory requires fresh nominal measurements.
    assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == case['trajectory_sha256']
    (tmp_path / 'telemetry.csv').write_bytes((evidence / (demo + '.csv')).read_bytes())
    (tmp_path / 'trajectory.json').write_bytes(trajectory.read_bytes())
    (tmp_path / 'run.json').write_text(json.dumps(dict(source_sha256={})))
    report = evaluate(tmp_path, root / 'scenes' / f'dynamic_{demo}.json', body, min_encounters=count)
    assert report['simultaneous_collision_pass']
    assert report['obstacles'][0]['qualifying_encounter_count'] >= count


def test_spatially_intersecting_paths_at_separate_times_do_not_pass(tmp_path):
    baseline, scene = fixture(tmp_path, positions=[[.5, 0., .13], [1.5, 0., .13]])
    report = evaluate(baseline, scene, 'base')
    assert not report['simultaneous_collision_pass']
    assert report['obstacles'][0]['minimum_uninflated_solid_gap_m'] > 0.


def test_safety_buffer_overlap_without_solid_collision_does_not_pass(tmp_path):
    baseline, scene = fixture(tmp_path, stationary=True, positions=[[.22, 0., .13], [.22, 0., .13]])
    report = evaluate(baseline, scene, 'base')
    assert not report['simultaneous_collision_pass']
    assert 0. < report['obstacles'][0]['minimum_uninflated_solid_gap_m'] < .025


def test_invalid_cylinder_axis_is_rejected_before_simulation(tmp_path):
    _, path = fixture(tmp_path)
    document = json.loads(path.read_text())
    document['obstacles'][0]['model'] = dict(type='cylinder', radius=.035, height=.4, axis='diagonal')
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='axis'):
        DynamicScene(path)


def test_isolated_roots_require_complete_matching_runtime_sources():
    files = ['dynamic_whole_body_controller.py', 'whole_body_controller.py',
             'whole_body_controller.yaml', 'shot_whole_body_controller.yaml', 'jvs_dynamic_simview_isaac.py']
    baseline = dict(source_sha256={f'/baseline/{name}': name for name in files})
    actual = dict(source_sha256={f'/isolated/{name}': name for name in files})
    assert same_runtime_sources(actual, baseline)
    actual['source_sha256']['/isolated/jvs_dynamic_simview_isaac.py'] = 'different physics implementation'
    assert not same_runtime_sources(actual, baseline)
    assert not same_runtime_sources(dict(source_sha256={}), dict(source_sha256={}))


def test_base_collision_uses_the_mesh_not_its_circumscribed_circle():
    planes, _ = base_collision_planes()
    angle = np.pi / 32
    point = np.array([.1548 * np.cos(angle), .1548 * np.sin(angle), 0.])
    assert primitive_sdf(point, 'frustum', [.18, .13, .12]) < 0.
    assert np.max(planes[:, :3] @ point + planes[:, 3]) > 0.
