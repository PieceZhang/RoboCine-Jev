"""Default camera routing, Agent geometry and actual three-view video content."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.camera_output import CameraOutput, frame_camera_pose, video_camera_source, write_three_camera_video

ROOT = Path(__file__).resolve().parents[1]


def test_default_output_is_stabilized_and_raw_pose_is_retained():
    camera = CameraOutput()
    camera.update(0., np.zeros(3), np.eye(3), np.zeros(3), np.eye(3))
    position, _ = camera.update(.001, np.zeros(3), np.eye(3), np.array([.1, 0., 0.]), np.eye(3))
    assert camera.source == 'stabilized' and position[0] < .001
    assert camera.latest['actual_camera_position'][0] == .1
    np.testing.assert_allclose(camera.pose(.001)[0], position)
    with pytest.raises(ValueError, match='timestamp'):
        camera.pose(.002)


def test_declared_output_cannot_silently_fall_back_to_raw(tmp_path):
    (tmp_path/'execution').mkdir()
    assert video_camera_source(tmp_path) == 'execution'
    (tmp_path/'camera_output.json').write_text(json.dumps(dict(source='stabilized')))
    with pytest.raises(ValueError, match='missing'):
        video_camera_source(tmp_path)
    (tmp_path/'stabilized').mkdir()
    assert video_camera_source(tmp_path) == 'stabilized'


def test_frame_geometry_uses_the_selected_camera():
    frame = dict(camera_output_source='stabilized', actual_camera_position=[1., 2., 3.],
        actual_camera_quaternion_xyzw=[0., 0., 0., 1.], stabilized_camera_position=[4., 5., 6.],
        stabilized_camera_quaternion_xyzw=Rotation.from_euler('z', .4).as_quat().tolist())
    p, r = frame_camera_pose(frame)
    np.testing.assert_allclose(p, [4., 5., 6.])
    np.testing.assert_allclose(r, Rotation.from_euler('z', .4).as_matrix())
    np.testing.assert_allclose(frame_camera_pose(frame, 'execution')[0], [1., 2., 3.])


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg unavailable')
def test_encoded_video_contains_reference_raw_stabilized_in_that_order(tmp_path):
    import cv2
    colors = [(0, 0, 240), (0, 240, 0), (240, 0, 0)]
    frames = [dict(frame=i, t=i*.04) for i in range(3)]
    for view, color in zip(('reference', 'execution', 'stabilized'), colors):
        (tmp_path/view).mkdir()
        for row in frames:
            cv2.imwrite(str(tmp_path/view/f"{row['frame']:05d}.jpg"), np.full((48, 64, 3), color, np.uint8))
    result = write_three_camera_video(tmp_path, frames, 25.)
    assert result['views'] == ['reference', 'execution', 'stabilized'] and result['frames'] == 3
    capture = cv2.VideoCapture(str(tmp_path/'comparison.mp4'))
    try:
        count = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            assert image.shape == (80, 192, 3)
            for column, color in enumerate(colors):
                np.testing.assert_allclose(image[55, column*64+32], color, atol=12)
            count += 1
        assert count == 3
    finally:
        capture.release()


def test_agent_receives_virtual_pose_with_original_physical_state(tmp_path):
    from jev_agent_wbc.engine import CinematicEngine
    from jev_agent_wbc.reference import OriginalReference
    from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics
    runtime = json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    reference = OriginalReference(ROOT/'scenarios/movein/task.json', runtime['subject']['position'])
    state = np.array(runtime['initial_state'])
    scene = dict(format='jvs_lidar_surfels/v1', observation_stamp=0., surfel_centers=[[8., 8., 8.]], surfel_radius=.1,
                 surfel_velocities=[[0., 0., 0.]], minimum_clearance_m=.025, subject_model=runtime['subject'])
    engine = CinematicEngine(reference, scene, tmp_path, mode='strict', limits=Limits(obstacle_avoidance=False))
    try:
        physical_p, physical_r, _ = whole_body_kinematics(state, runtime['base_z'])
        output_p = physical_p+np.array([.03, .02, 0.])
        engine.set_camera_output(0., output_p, physical_r)
        engine.update(state, scene, 0., runtime['base_z'])
        np.testing.assert_allclose(engine.actual_feedback['camera_position'], output_p)
        np.testing.assert_allclose(engine.actual_feedback['state'], state)
        np.testing.assert_allclose(engine.camera_history[-1]['camera_position'], output_p)
        assert engine.actual_feedback['camera_output_source'] == 'stabilized'
        with pytest.raises(ValueError, match='timestamp'):
            engine.update(state, scene, .02, runtime['base_z'])
        engine.set_camera_output(.02, output_p, physical_r, max_age_s=.1)
        engine.update(state, scene, .04, runtime['base_z'])
        engine.update(state, scene, .06, runtime['base_z'])
        assert engine.actual_feedback['stamp_s'] == .02
        assert [row['t'] for row in engine.camera_history] == [0., .02]
    finally:
        engine.close()


def test_visual_scores_use_output_pose_and_keep_the_physical_failure():
    from jev_agent_wbc.evaluation import trajectory_summary
    from jev_agent_wbc.reference import OriginalReference
    runtime = json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    original = OriginalReference(ROOT/'scenarios/movein/task.json', runtime['subject']['position'])
    rows = []
    for t in (0., 4., 8.):
        p, r, _ = original.sample(t)
        q = Rotation.from_matrix(r).as_quat().tolist()
        rows.append(dict(t=t, camera_position=p.tolist(), camera_quaternion_xyzw=q,
            camera_output_source='stabilized', actual_camera_position=(p+[.1, 0., 0.]).tolist(),
            actual_camera_quaternion_xyzw=q, truth_clearance_m=1., command=np.zeros(8).tolist(),
            info=dict(joint_margin_rad=.1)))
    result = trajectory_summary(rows, original, {}, physical_audit=dict(pass_constraints=False))
    assert result['camera_output_source'] == 'stabilized'
    assert result['original_position_rms_m'] == pytest.approx(0.)
    assert result['physical_camera_tracking']['position_rms_m'] == pytest.approx(.1)
    assert result['safety_pass'] is False


def test_sam_evaluation_segments_the_declared_output_and_uses_its_pose(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('camera_sam_evaluation', ROOT/'scripts/evaluate_sam2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    video = tmp_path/'video'
    for view in ('reference', 'execution', 'stabilized'):
        (video/view).mkdir(parents=True)
    (video/'camera_output.json').write_text(json.dumps(dict(source='stabilized')))
    frame = dict(frame=0, t=0., camera_output_source='stabilized', reference_camera_position=[0., 0., 0.],
        actual_camera_position=[0., .5, 0.], actual_camera_quaternion_xyzw=[0., 0., 0., 1.],
        stabilized_camera_position=[0., 0., 0.], stabilized_camera_quaternion_xyzw=[0., 0., 0., 1.],
        subject_track=dict(valid=True, position=[2., 0., 0.], rotation=np.eye(3).tolist(), orientation_valid=True))
    (video/'frames.json').write_text(json.dumps([frame]))
    (tmp_path/'original_task.json').write_text(json.dumps(dict(shot_phase=[dict(name='shot', end_s=1.)])))
    (tmp_path/'summary.json').write_text('{}')
    seen = []
    def segment(predictor, directory, device):
        seen.append(directory.name)
        mask = np.zeros((1, 48, 64), bool)
        mask[:, 12:30, 20:40] = True
        return mask, ['valid']
    monkeypatch.setattr(module, 'segment', segment)
    result = module.evaluate(None, tmp_path, dict(device='cpu'))
    assert seen == ['reference', 'stabilized']
    assert result['camera_output_source'] == 'stabilized'
    assert result['observation_angle_error_rad']['mean'] == pytest.approx(0.)


def test_async_sam_keeps_the_frame_id_from_acquisition():
    import time
    from jev_agent_wbc.sam_observer import LatestMaskWorker
    worker = LatestMaskWorker(lambda rgb, stamp: dict(stamp_s=stamp))
    try:
        metadata = dict(camera_frame='virtual_camera_stabilized')
        assert worker.submit(np.zeros((4, 4, 3), np.uint8), 2.3, metadata)
        metadata['camera_frame'] = 'different_later_camera'
        deadline = time.monotonic()+2.
        result = None
        while result is None and time.monotonic() < deadline:
            result = worker.poll()
            time.sleep(.001)
        assert result['camera_frame'] == 'virtual_camera_stabilized'
        assert result['stamp_s'] == 2.3
    finally:
        worker.close()
