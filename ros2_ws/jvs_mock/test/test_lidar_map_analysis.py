"""CPU-only tests for the audit, independent of Isaac/ROS and user scene assets."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'analyze_lidar_map.py'
SPEC = importlib.util.spec_from_file_location('lidar_map_analysis', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_usd_row_vector_transform_roundtrip():
    matrix = np.array([[0., 1, 0, 0], [-1., 0, 0, 0], [0, 0, 1, 0], [5, -1, .24, 1]])
    points = np.array([[1., 0, 0], [0, 1, .2]])
    world = audit.transform(points, matrix)
    np.testing.assert_allclose(world, [[5, 0, .24], [4, -1, .44]])
    np.testing.assert_allclose(audit.transform(world, np.linalg.inv(matrix)), points, atol=1e-12)


def test_voxelization_retains_real_samples_and_negative_cells():
    points = np.array([[.01, 0, 0], [.07, 0, 0], [-.01, 0, 0], [.09, 0, 0]])
    actual = audit.voxelize(points)
    assert len(actual) == 3
    assert audit.keys(actual) == {(-1, 0, 0), (0, 0, 0), (1, 0, 0)}
    assert all(any(np.array_equal(p, q) for q in points) for p in actual)


def test_empty_capture_is_not_reported_as_complete(tmp_path):
    (tmp_path / 'capture.json').write_text(json.dumps({'scans': []}))
    with pytest.raises(ValueError, match='no scans'):
        audit.analyze(tmp_path, {'points': np.zeros((1, 3))}, tmp_path / 'analysis')


def test_coverage_removes_robot_but_retains_unknown_surfaces(tmp_path, monkeypatch):
    identity = np.eye(4)
    observed = np.array([[1., 0, .1], [1, 0, .8], [.2, 0, .1]])
    np.savez(tmp_path / 'scan.npz', points=observed, world_sensor=identity,
             world_base=identity, base_link_sensor=identity,
             self_boxes=np.array([[[.15, -.05, .05], [.25, .05, .15]]]),
             self_box_frames=np.array([identity]))
    scan = dict(file='scan.npz', pose_index=0, points=3)
    metadata = dict(scans=[scan, scan], poses=[[0, 0, 0]], sensor_paths=['/one'],
                    domain_id=221, config='fixture', note='synthetic test')
    (tmp_path / 'capture.json').write_text(json.dumps(metadata))
    # One low and one arm-height surface are observed; their backs are unknown.
    reference = np.array([[1., 0, .1], [1, 0, .8], [.9, .8, .1], [.9, .8, .8]])
    monkeypatch.setattr(audit, 'plot', lambda *args: None)
    out = tmp_path / 'analysis'
    audit.analyze(tmp_path, {'points': reference}, out)
    metrics = json.loads((out / 'metrics.json').read_text())
    assert metrics['observed_8cm_voxels'] == 2
    assert metrics['scans'][0]['self_removed'] == 1
    assert metrics['scans'][1]['interscan_jaccard'] == 1
    assert metrics['coverage']['entire_scene_navigation_0.035_to_1.60m']['recall_at_8_12_20cm'] == [.5] * 3
    assert metrics['extrinsic_max_abs_variation'] == 0
    assert b'NOT measured returns' in (out / 'unobserved_reference.ply').read_bytes()[:200]


def test_different_sensor_extrinsics_are_not_reported_as_drift(tmp_path, monkeypatch):
    rows = []
    for name, x in [('rear', -.085), ('front', .190)]:
        matrix = np.eye(4)
        matrix[3, 0] = x
        file = name + '.npz'
        np.savez(tmp_path / file, points=np.array([[1., 0, .1], [1., 0, .8]]),
                 world_sensor=matrix, world_base=np.eye(4), base_link_sensor=matrix,
                 self_boxes=np.empty((0, 2, 3)), self_box_frames=np.empty((0, 4, 4)))
        rows.append(dict(file=file, pose_index=0, sensor=name))
    metadata = dict(scans=rows, poses=[[0, 0, 0]], sensor_paths=['/rear', '/front'],
                    domain_id=222, config='fixture', note='test')
    (tmp_path / 'capture.json').write_text(json.dumps(metadata))
    reference = {'points': np.array([[.915, 0, .1], [1.19, 0, .8]])}
    monkeypatch.setattr(audit, 'plot', lambda *args: None)
    audit.analyze(tmp_path, reference, tmp_path / 'analysis')
    report = json.loads((tmp_path / 'analysis/metrics.json').read_text())
    assert report['extrinsic_max_abs_variation'] == 0
    assert set(report['per_sensor_extrinsics']) == {'rear', 'front'}
    assert all(r['interscan_jaccard'] is None for r in report['scans'])
    audit.analyze(tmp_path, reference, tmp_path / 'rear_only', 'rear')
    report = json.loads((tmp_path / 'rear_only/metrics.json').read_text())
    assert set(report['per_sensor_extrinsics']) == {'rear'}
