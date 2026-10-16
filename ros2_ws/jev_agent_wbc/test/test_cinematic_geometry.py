"""Endpoint freedom, monotone shot geometry, exact replay and smooth joins."""
import json
from pathlib import Path
import numpy as np
import pytest
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.intent_paths import IntentPath
from jev_agent_wbc.cinematic_geometry import terminal_band,terminal_error,inside_terminal_band,curve_quality
from jev_agent_wbc.candidates import path_hash,path_spec,restore_path

ROOT=Path(__file__).resolve().parents[1]/'scenarios'


@pytest.fixture(params=['orbit','movein','truck_left','truck_right','pedestal_up','pedestal_down'])
def shot(request):
    scene=json.loads((ROOT/request.param/'runtime_scene.json').read_text())
    return OriginalReference(ROOT/request.param/'task.json',scene['subject']['position']),scene['subject']


def test_numerical_endpoint_band_is_nonzero_and_respects_existing_image_limits(shot):
    original,model=shot
    band=terminal_band(original,model)
    offsets=np.array(band['offsets_m'])
    assert len(offsets)==27 and np.max(np.linalg.norm(offsets,axis=1))>.01
    for offset in offsets:
        assert inside_terminal_band(terminal_error(original,np.array(band['reference_position'])+offset,
                                                   model,[.5/np.tan(np.deg2rad(27.5))*.75,.5/np.tan(np.deg2rad(27.5)),.5,.5]))


def test_regular_curve_is_c2_joined_stops_and_keeps_distinct_accepted_endpoint(shot):
    original,model=shot
    offsets=np.array(terminal_band(original,model)['offsets_m'])
    offset=offsets[np.argmax(offsets[:,1])]
    motion=original.phase(0)['motion']
    route='left' if 'pedestal' in motion else 'below'
    path=IntentPath(original,original,1.3,dict(route=route,amplitude_m=.12,extension_s=0.,
        regular_curve=True,endpoint_offset_m=offset.tolist()),model)
    np.testing.assert_allclose(path.sample(.8)[0],original.sample(.8)[0],atol=1e-12)
    np.testing.assert_allclose(path.translation(1.3,1),original.sample(1.3)[2][:3],atol=1e-10)
    np.testing.assert_allclose(path.translation(8.,1),0.,atol=1e-10)
    np.testing.assert_allclose(path.translation(8.,2),0.,atol=1e-9)
    np.testing.assert_allclose(path.sample(8.)[0]-original.sample(8.)[0],offset,atol=1e-10)
    assert curve_quality(path)['direction_pass']
    assert curve_quality(path)['regular_curve_pass']
    restored=restore_path(original,path_spec(path))
    assert path_hash(path)==path_hash(restored)
    for t in np.linspace(0.,8.4,17):
        np.testing.assert_allclose(path.sample(t)[0],restored.sample(t)[0],atol=1e-12)


def test_direction_gate_rejects_smooth_movein_retreat_even_with_exact_endpoint():
    scene=json.loads((ROOT/'movein/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'movein/task.json',scene['subject']['position'])
    path=IntentPath(original,original,1.,dict(route='inner',amplitude_m=1.2,extension_s=0.,regular_curve=True),scene['subject'])
    assert not curve_quality(path)['direction_pass']
    np.testing.assert_allclose(path.sample(8.)[0],original.sample(8.)[0],atol=1e-10)


def test_smoothed_robot_proposal_is_replayable_and_passes_a_new_strict_lift():
    from jev_agent_wbc.kinematic_feasibility import KinematicSearch
    from jev_agent_wbc.whole_body_qp import Limits
    from jev_agent_wbc.task_space_planning import MeasuredCameraPrediction
    from jev_agent_wbc.cinematic_geometry import smooth_lift
    scene=json.loads((ROOT/'movein/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'movein/task.json',scene['subject']['position'])
    state=np.array(scene['initial_state']);z=scene['base_z']
    observed=dict(format='jvs_lidar_surfels/v1',observation_stamp=0.,surfel_centers=[[8.,8.,8.]],
        surfel_radius=.1,surfel_velocities=[[0.,0.,0.]],minimum_clearance_m=.025)
    seed=IntentPath(original,original,.5,dict(route='timing',amplitude_m=0.,extension_s=0.,regular_curve=True),scene['subject'])
    search=KinematicSearch(state,np.zeros(8),0.,z,Limits(),scene=observed,search_config='executable')
    report=search._lift(MeasuredCameraPrediction(seed,state,z,0.,.5),8.4,'path_heading',project_camera=True)
    assert report['feasible']
    fitted=smooth_lift(seed,report)
    assert fitted is not None
    restored=restore_path(original,path_spec(fitted))
    assert path_hash(restored)==path_hash(fitted)
    result=search.lift(MeasuredCameraPrediction(restored,state,z,0.,.5),8.4)
    assert result['feasible'],result['reason']
    assert not result['camera_projected_for_generation']
    assert result['max_position_error_m']<=result['position_tolerance_m']
    assert result['min_body_clearance_m']>=.025
