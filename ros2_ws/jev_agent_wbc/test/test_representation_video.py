"""Physical-state geometry, causal observations, and encoded film provenance."""
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.representation_video import RepresentationPanel, append_representation_video
from jev_agent_wbc.whole_body_qp import whole_body_kinematics


def recording():
    state = np.array([0., 0., .5, -.5, -.7, 1.5, 0., -.8, 0.])
    base_rotation = Rotation.from_euler('xyz', [.15, -.12, .5]).as_matrix()
    p, r, _ = whole_body_kinematics(state, .025, base_rotation)
    rows = [dict(t=t, state=state.tolist(), base_z=.025, base_rotation=base_rotation.tolist(),
        camera_position=(p+[.4, .2, .1]).tolist(), camera_quaternion_xyzw=[0., 0., 0., 1.],
        actual_camera_position=p.tolist(), actual_camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
        camera_output_source='stabilized', info=dict(tpr_status='normal')) for t in (0., .04, .08)]
    return SimpleNamespace(rows=rows, times=np.array([r['t'] for r in rows]),
        cpu=False, floating=False, events=[], scene={}, summary=dict(backend='isaac_physx'))


def test_physx_representation_uses_tilted_physical_state_not_virtual_camera():
    run = recording()
    panel = RepresentationPanel(run)
    obs = panel.observations[1]
    np.testing.assert_allclose(obs.world_from_mount.rotation, run.rows[1]['base_rotation'], atol=1e-12)
    np.testing.assert_allclose(obs.world_from_camera.translation_m, run.rows[1]['actual_camera_position'], atol=1e-12)
    assert np.linalg.norm(np.asarray(obs.world_from_camera.translation_m)-run.rows[1]['camera_position']) > .4
    assert panel.audit['max_camera_position_error_m'] < 1e-12
    np.testing.assert_allclose(panel.view @ panel.view.T, np.eye(3), atol=1e-12)


def test_geometry_and_observations_are_never_taken_from_a_future_tick():
    from PIL import Image
    run = recording()
    panel = RepresentationPanel(run)
    audit = panel.draw(Image.new('RGB', (1920, 1080)), run, .059)
    assert audit['representation_stamp_s'] == .04
    row = dict(run.rows[1], observation_stamp=.06,
        runtime_tracks=[dict(id=1, age_s=0., center=[1., 1., 1.], velocity=[0., 0., 0.])])
    assert panel.track_relations(row, panel.observations[1]) is None
    row['observation_stamp'] = -.7
    assert panel.track_relations(row, panel.observations[1]) is None


def test_center_distance_is_distinct_from_obstacle_surface_clearance():
    run = recording()
    panel = RepresentationPanel(run)
    obs = panel.observations[1]
    center = obs.world_from_mount.apply([1., 0., 0.])
    row = dict(run.rows[1], observation_stamp=.04,
        runtime_tracks=[dict(id=1, age_s=0., center=center.tolist(), velocity=[0., 0., 0.])])
    relation = panel.track_relations(row, obs)
    expected = min(float(b.distance([1., 0., 0.])) for b in obs.occupancy['mobile_platform'])
    assert relation['distances_m'][0] == pytest.approx(expected)
    assert relation['direction'] == '前方'


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg unavailable')
@pytest.mark.parametrize('time_offset', [0., .02])
def test_fixed_camera_film_adds_geometry_without_changing_source_pixels(tmp_path, time_offset):
    import cv2
    run = recording()
    for row in run.rows:
        row['t'] += time_offset
    (tmp_path/'summary.json').write_text(json.dumps(run.summary))
    (tmp_path/'telemetry.json').write_text(json.dumps(run.rows))
    source = tmp_path/'source.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 25., (64, 48))
    for _ in range(3):
        writer.write(np.full((48, 64, 3), [30, 100, 210], np.uint8))
    writer.release()
    before = source.read_bytes()
    output = tmp_path/'film.mp4'
    report = append_representation_video(source, output, tmp_path)
    assert source.read_bytes() == before
    assert report['representation_panel'] and report['frames'] == 3
    capture = cv2.VideoCapture(str(output))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape == (548, 1920, 3)
    np.testing.assert_allclose(frame[24, 960], [30, 100, 210], atol=12)
    footer = frame[48:]
    # Both module colors survive encoding in the geometry footer.
    assert np.count_nonzero((footer[:,:,0] > 160) & (footer[:,:,0] > footer[:,:,2]+20)) > 100
    assert np.count_nonzero((footer[:,:,2] > 170) & (footer[:,:,2] > footer[:,:,0]+20)) > 100
