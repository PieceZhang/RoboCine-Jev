"""The evidence auditor must detect inconsistent masks, identities and TF."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_robot_sensor_coverage import audit


@pytest.fixture
def evidence(tmp_path):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('z', np.pi/2).as_matrix()
    matrix[:3, 3] = [1., 2., .1]
    rgb = tmp_path/'rgb.png'; mask = tmp_path/'mask.png'
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(rgb)
    Image.fromarray(np.zeros((4, 4), np.uint8)).save(mask)
    manifest = dict(lidars=[dict(horizontal_samples=4, elevation_deg=[0.], min_range_m=.05, max_range_m=4.5)])
    pose = dict(index=0, self_mask=mask.name, rgb=rgb.name, rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(),
                camera_self_occlusion_fraction=0., camera_housing_other_body_overlaps=[], lidar_origin_robot_overlaps={})
    detection = dict(raw_scan='scan.npz', stamp_s=.01, targets_world=[[1., 2.7, .1]]*36,
                     detected=[0], missing=list(range(1, 36)), radius_m=.7, target_centre_z_m=.1)
    summary = dict(robot='fixture', posture_count=1, target_detections=1)
    for name, value in [('summary.json', summary), ('sensor_mounts.json', manifest)]:
        (tmp_path/name).write_text(json.dumps(value))
    for name, value in [('postures.jsonl', pose), ('detection.jsonl', detection)]:
        (tmp_path/name).write_text(json.dumps(value)+'\n')
    np.savez_compressed(tmp_path/'scan.npz', points_world=np.array([[1., 2.64, .1]]), sensor=np.array([0]),
                        ray_index=np.array([2]), path_id=np.array([0]), paths=np.array(['/World/Probes/P00']),
                        T_world_lidars=np.array([matrix]), stamp_s=.01)
    return tmp_path


def test_independent_auditor_checks_transformed_sensor_ray(evidence):
    result = audit(evidence)
    assert result['evidence_consistent']
    assert result['checked_raw_points'] == result['checked_target_points'] == 1
    assert result['max_point_to_acquisition_ray_error_m'] < 1e-12
    with np.load(evidence/'scan.npz') as saved:
        data = {k: saved[k] for k in saved.files}
    data['points_world'][0, 0] += .01
    np.savez_compressed(evidence/'scan.npz', **data)
    result = audit(evidence)
    assert not result['evidence_consistent']
    assert any('acquisition-time' in error for error in result['errors'])


def test_mask_or_collision_identity_tampering_is_rejected(evidence):
    Image.fromarray(np.full((4, 4), 255, np.uint8)).save(evidence/'mask.png')
    with np.load(evidence/'scan.npz') as saved:
        data = {k: saved[k] for k in saved.files}
    data['paths'] = np.array(['/World/Probes/P01'])
    np.savez_compressed(evidence/'scan.npz', **data)
    errors = audit(evidence)['errors']
    assert any('mask' in error for error in errors)
    assert any('collision identities' in error for error in errors)


def test_target_inside_minimum_range_is_not_a_valid_detection(evidence):
    with np.load(evidence/'scan.npz') as saved:
        data = {k: saved[k] for k in saved.files}
    data['points_world'][0] = [1., 2.02, .1]
    np.savez_compressed(evidence/'scan.npz', **data)
    result = audit(evidence)
    assert not result['evidence_consistent']
    assert any('collision identities' in error for error in result['errors'])
