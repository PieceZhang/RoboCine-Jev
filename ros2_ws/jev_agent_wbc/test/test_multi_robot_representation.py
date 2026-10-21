"""Regression gates for native geometry and immutable cross-robot shot placement."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from jev_agent_wbc.representation import Observation, Pose, Box
from jev_agent_wbc.wbc_physical import MeasuredColliderGeometry
from jev_agent_wbc.multi_robot_geometry import MeasuredGeometryFilter, NativeKinematicModel, shape_clearance
from jev_agent_wbc.multi_robot_scenarios import bind_scenario, timed_path


def observation(z=1.2, yaw=.4):
    return Observation(3., Pose.from_rt(Rotation.from_euler('z', yaw).as_matrix(), [2., 3., .4]),
        Pose((.3, 0., z-.4)), {'mobile_platform': (Box((0., 0., 0.), (.4, .3, .2)),),
                               'camera_manipulator': (Box((.3, 0., .5), (.1, .1, .2)),)})


@pytest.mark.parametrize('case', ['orbit', 'movein'])
@pytest.mark.parametrize('height', [.6, 1.0, 1.6])
def test_gravity_preserving_binding_keeps_challenge_and_clock(case, height):
    original, scene, manifest = bind_scenario(case, observation(height))
    assert original.duration == 8.
    np.testing.assert_allclose(original.sample(0.)[0], observation(height).world_from_camera.translation_m)
    assert manifest['timing_scale'] == manifest['obstacle_size_scale'] == 1.
    assert scene['obstacles'][0]['radius'] == .1
    np.testing.assert_allclose(np.linalg.norm(scene['obstacles'][0]['keyframes'][0]['velocity']), np.hypot(.035, .24))
    assert scene['obstacles'][0]['keyframes'][0]['velocity'][2] == 0.
    task = timed_path(original, observation(height), 9.)
    assert task.trajectory.start_s == 9. and task.trajectory.end_s == 17.
    assert task.observation_id == observation(height).id


def test_explicit_obstacle_height_calibration_preserves_camera_subject_and_horizontal_challenge():
    original,scene,_=bind_scenario('orbit',observation())
    adjusted,moved,binding=bind_scenario('orbit',observation(),obstacle_height_offset_m=.08)
    assert adjusted.task==original.task and scene['subject']==moved['subject']
    assert scene['obstacles'][0]['radius']==moved['obstacles'][0]['radius']==.1
    for a,b in zip(scene['obstacles'][0]['keyframes'],moved['obstacles'][0]['keyframes']):
        np.testing.assert_allclose(np.asarray(b['position'])-a['position'],[0.,0.,.08])
        assert a['velocity']==b['velocity'] and a['t']==b['t']
    assert binding['additional_obstacle_height_offset_m']==.08
    for bad in (.16,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='micro-adjustment'):
            bind_scenario('orbit',observation(),obstacle_height_offset_m=bad)


def test_self_filter_uses_each_acquisition_pose_and_full_rotation():
    geometry = MeasuredColliderGeometry([
        dict(name='a', body='base', part='mobile_platform', center_body=[0., 0., 0.], half_extents_body=[1., .1, .1]),
        dict(name='b', body='arm', part='camera_manipulator', center_body=[0., 0., 0.], half_extents_body=[.1, .1, .1])])
    adapter = MeasuredGeometryFilter(geometry)
    first = {'base': np.eye(4), 'arm': np.eye(4)}
    first['arm'][2, 3] = 2.
    initial = adapter.state(first, np.eye(4))
    later = {k: v.copy() for k, v in first.items()}
    later['base'][:3, :3] = Rotation.from_euler('z', np.pi/2).as_matrix()
    latest = adapter.state(later, np.eye(4))
    assert adapter.distances(initial, [[.5, 0., 0.]])[0, 0] < 0.
    assert adapter.distances(latest, [[.5, 0., 0.]])[0, 0] > .39
    assert adapter.distances(initial, [[.5, 0., 0.]])[0, 0] < 0.


def test_native_camera_and_all_body_fk_agree_for_rotational_and_prismatic_joints():
    home = np.eye(4)
    home[0, 3] = 1.
    model = NativeKinematicModel(dict(base=np.eye(4).tolist(),
        joints=[dict(omega=[0., 0., 1.], velocity=[0., 0., 0.], q0=0., lower=-2., upper=2., speed=1.),
                dict(omega=[0., 0., 0.], velocity=[0., 0., 1.], q0=0., lower=-1., upper=1., speed=.3)],
        homes={'camera': home.tolist()}, chains={'camera': [0, 1]}, shapes=[],
        camera_parent='camera', camera_local=np.eye(4).tolist(), omnidirectional=True))
    state = [.2, .3, .4, .5, .6]
    np.testing.assert_allclose(model.camera(state), model.poses(state)['camera'], atol=1e-12)
    np.testing.assert_allclose(model.camera(state)[:3, 3], [.2+np.cos(.9), .3+np.sin(.9), .6])


def test_clearance_is_surface_distance_and_empty_evidence_is_unknown():
    shapes = [dict(center_world=np.zeros(3), rotation_world=np.eye(3), half_extents_body=np.ones(3))]
    assert shape_clearance(shapes, [[2., 0., 0.]], [.2]) == pytest.approx(.8)
    assert shape_clearance(shapes, [[.5, 0., 0.]], [.2]) == pytest.approx(-.7)
    assert shape_clearance(shapes, [], []) is None


@pytest.mark.parametrize('offset',[0.,.2])
def test_camera_attachment_ranking_preserves_optical_transform_and_excludes_other_chains(offset):
    from types import SimpleNamespace
    from jev_agent_wbc.multi_robot_planning import CameraAttachmentScreen
    from jev_agent_wbc.wbc_protocol import OPTICAL_FROM_SEMANTIC
    home=np.eye(4);home[:3,3]=[.6,.1,.8]
    mount=np.eye(4);mount[:3,:3]=Rotation.from_euler('xyz',[.2,-.1,.4]).as_matrix()
    mount[:3,3]=[.2,-.1,.1]
    shapes=[dict(body='camera',center_body=[.05,.02,0.],half_extents_body=[.1,.15,.2],part='camera_manipulator'),
            dict(body='base',center_body=[0.,0.,0.],half_extents_body=[10.,10.,10.],part='mobile_platform')]
    model=NativeKinematicModel(dict(base=np.eye(4).tolist(),
        joints=[dict(omega=[0.,0.,1.],velocity=[0.,0.,0.],q0=0.,lower=-2.,upper=2.,speed=.5)],
        homes={'camera':home.tolist(),'base':np.eye(4).tolist()},chains={'camera':[0],'base':[]},
        shapes=shapes,camera_parent='camera',camera_local=mount.tolist(),omnidirectional=True))
    original=model.camera(np.zeros(4));target=original.copy()
    target[:3,:3]=Rotation.from_euler('z',offset).as_matrix()@target[:3,:3]
    target[:3,3]+=[offset,-offset,offset]
    semantic=target[:3,:3]@OPTICAL_FROM_SEMANTIC.rotation
    path=SimpleNamespace(poses_at=lambda times:(np.tile(target[:3,3],(len(times),1)),
                                               np.tile(semantic,(len(times),1,1))))
    observed=dict(centers=np.array([[1.3,.4,1.]]),radii=np.array([.1]),
                  velocities=np.zeros((1,3)),radius_rates=np.zeros(1),observation_stamp=0.)
    screen=CameraAttachmentScreen(model,observed,0.,0.)
    transformed=target@np.linalg.inv(original)@home
    expected=shape_clearance([dict(center_world=transformed[:3,:3]@shapes[0]['center_body']+transformed[:3,3],
        rotation_world=transformed[:3,:3],half_extents_body=shapes[0]['half_extents_body'])],
        observed['centers'],observed['radii'])
    assert screen.clearance(path)==pytest.approx(expected,abs=1e-12)
    assert len(screen.transforms)==1
    screen.obstacles=[(np.empty((0,3)),np.empty(0)) for _ in screen.times]
    assert screen.clearance(path) is None


def test_native_camera_and_obstacle_jacobians_match_central_differences():
    home=np.eye(4);home[0,3]=.8
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('xyz',[.08,-.1,.3]).as_matrix();base[2,3]=.5
    model=NativeKinematicModel(dict(base=base.tolist(),
        joints=[dict(omega=[0.,0.,1.],velocity=[0.,-.2,0.],q0=0.,lower=-2.,upper=2.,speed=.5),
                dict(omega=[0.,0.,0.],velocity=[0.,0.,1.],q0=0.,lower=-.4,upper=.4,speed=.1)],
        homes={'camera':home.tolist()},chains={'camera':[0,1]},
        shapes=[dict(body='camera',center_body=[.1,0.,0.],half_extents_body=[.2,.1,.15],part='camera_manipulator')],
        camera_parent='camera',camera_local=np.eye(4).tolist(),omnidirectional=False))
    x=np.array([.1,-.1,.2,.4,.1]);centers=np.array([[.8,.4,.5],[1.5,.1,.7]])
    radii=np.array([.1,.08])
    T,J,d,D=model.differential(x,centers,radii)
    for compiled,reference in zip((T,J,d,D),model.differential_reference(x,centers,radii)):
        np.testing.assert_allclose(compiled,reference,atol=1e-12)
    np.testing.assert_allclose(model.camera(x),model.camera_reference(x),atol=1e-12)
    merged_camera,merged_gap=model.camera_and_clearance(x,centers,radii)
    np.testing.assert_allclose(merged_camera,model.camera_reference(x),atol=1e-12)
    assert merged_gap==pytest.approx(shape_clearance(model.world_shapes(x),centers,radii),abs=1e-12)
    assert model.camera_and_clearance(x,[],[])[1] is None
    for name,value in model.poses(x).items():
        np.testing.assert_allclose(value,model.poses_reference(x)[name],atol=1e-12)
    np.testing.assert_allclose(d,model.obstacle_distances(x,centers,radii),atol=1e-12)
    for i in range(len(x)):
        a=x.copy();b=x.copy();a[i]+=1e-6;b[i]-=1e-6
        A=model.camera(a);B=model.camera(b)
        measured=np.r_[(A[:3,3]-B[:3,3])/2e-6,
                       Rotation.from_matrix(A[:3,:3]@B[:3,:3].T).as_rotvec()/2e-6]
        np.testing.assert_allclose(J[:,i],measured,atol=1e-8)
        numerical=(model.obstacle_distances(a,centers,radii)-model.obstacle_distances(b,centers,radii))/2e-6
        np.testing.assert_allclose(D[:,:,i],numerical,atol=1e-8)


def test_candidate_revisions_preserve_executed_prefix_and_original_deadline():
    from jev_agent_wbc.multi_robot_planning import candidate_paths
    original,_,_=bind_scenario('orbit',observation())
    for candidate in candidate_paths(original,'joint','explore',activation_s=2.3):
        for t in (0.,1.,2.299,8.):
            actual=candidate.sample(t);expected=original.sample(t)
            np.testing.assert_allclose(actual[0],expected[0],atol=1e-12)
            np.testing.assert_allclose(actual[1],expected[1],atol=1e-12)
        assert candidate.duration==original.duration


def test_environment_feedback_keeps_acquisition_and_verification_clocks_distinct():
    from jev_agent_wbc.wbc_physical import constraint_feedback
    evidence=dict(environment=dict(status='pass',stamp_s=1.15,scope='fresh_environment_map',
                                   clearance_m=.04,acquisition_stamp_s=1.,valid_until_s=1.25))
    current=constraint_feedback(evidence,1.15)
    assert current['environment']['status']=='pass'
    assert current['environment']['acquisition_stamp_s']==1.
    assert constraint_feedback(evidence,1.26)['environment']['status']=='unknown'
    evidence['environment']['valid_until_s']=2.
    with pytest.raises(ValueError,match='source lease'):
        constraint_feedback(evidence,1.15)


def test_native_base_allocation_removes_camera_lever_arm_velocity():
    from types import SimpleNamespace
    from jev_agent_wbc.multi_robot_runtime import AvoidanceRuntime
    runtime=AvoidanceRuntime.__new__(AvoidanceRuntime)
    runtime.base_name='base';runtime.scene_start=0.
    runtime.platform_limits=np.array([.5,.5,1.])
    runtime.should_brake=lambda stamp,bodies:False
    camera=np.eye(4);camera[0,3]=1.
    runtime.chosen=SimpleNamespace(duration=8.,
        sample=lambda t:(np.array([1.,0.,0.]),np.eye(3),np.array([0.,.2,0.,0.,0.,.2])))
    command=runtime.platform_velocity(1.,{'base':np.eye(4)},camera)
    np.testing.assert_allclose(command,[0.,0.,.2],atol=1e-12)
    np.testing.assert_allclose(runtime.platform_velocity(1.,{'base':np.eye(4)},camera,yaw_share=0.),
                               [0.,.2,0.],atol=1e-12)


@pytest.mark.parametrize('case',['orbit','movein'])
def test_batched_candidate_matches_unchanged_scalar_reference(case):
    from jev_agent_wbc.multi_robot_planning import candidate_paths
    from jev_agent_wbc.intent_paths import poses_at
    original,_,_=bind_scenario(case,observation())
    times=np.r_[-1.,np.linspace(0.,8.,51),8.4]
    for path in candidate_paths(original,'joint','explore',2.3):
        p,r=poses_at(path,times)
        expected=[path.pose(float(t)) for t in times]
        np.testing.assert_allclose(p,[v[0] for v in expected],atol=1e-12)
        np.testing.assert_allclose(r,[v[1] for v in expected],atol=1e-12)


def test_subject_visibility_labels_never_remove_physical_occupancy_or_movers():
    from jev_agent_wbc.multi_robot_audit import visibility_scene
    from jev_agent_wbc.task_space_planning import camera_scene
    snapshot=dict(observation_stamp=1.,surfel_centers=[[1.,0.,0.],[.5,0.,0.]],
        surfel_radii=[.1,.2],surfel_track_ids=[0,1],surfel_subject_mask=[True,True],
        subject_label_source='acquired_closest_hit_identity_visibility_only')
    physical=camera_scene(snapshot,1.1,2.)
    visual=visibility_scene(snapshot,1.1,2.)
    assert len(physical['centers'])==2
    np.testing.assert_allclose(visual['centers'],[[.5,0.,0.]])
    assert len(snapshot['surfel_centers'])==2


def test_mapping_side_labels_preserve_original_integration_contract():
    from jev_agent_wbc import multi_robot_runtime as runtime
    geometry=MeasuredColliderGeometry([
        dict(name='base',body='base',part='mobile_platform',center_body=[0.,0.,0.],
             half_extents_body=[.1,.1,.1]),
        dict(name='arm',body='base',part='camera_manipulator',center_body=[0.,0.,.2],
             half_extents_body=[.1,.1,.1])])
    geometry_filter=MeasuredGeometryFilter(geometry)
    runtime._mapping_initialize(geometry_filter)
    points=np.array([[1.,y,z] for y in np.arange(-.1,.11,.04)
                     for z in np.arange(.4,.61,.04)])
    packet=dict(points=points-[.3,0.,.5],rotation=np.eye(3),translation=np.array([.3,0.,.5]),
        state=geometry_filter.state({'base':np.eye(4)},np.eye(4)),base_z=0.,
        base_rotation=np.eye(3),stamp=1.,sensor='test',acquisition_duration_s=0.,
        exact_ray_geometry=True,subject_return_mask=np.ones(len(points),bool))
    for stamp in (1.,1.1,1.2,1.3):
        packet['stamp']=stamp
        snapshot=runtime._mapping_integrate([packet])
    assert len(snapshot['surfel_centers'])>0
    assert len(snapshot['surfel_subject_mask'])==len(snapshot['surfel_centers'])
    assert np.any(snapshot['surfel_subject_mask'])
    assert len(packet['points'])==len(points)


def test_tiago_audit_rejects_missing_post_step_physical_evidence():
    from copy import deepcopy
    from jev_agent_wbc.multi_robot_audit import tiago_guard_audit
    assert not tiago_guard_audit([])['pass_check']
    row=dict(t_s=1.,q=[0.]*14)
    full=dict(raw_physical_dofs_checked=14,max_position_excess=0.,max_velocity_excess=0.)
    motion=dict(raw_measured_velocity_limits_checked=True,max_speed_excess=0.,
                minimum_joint_stopping_reserve_mixed_units=.02)
    for key in ('measured_full_dof_guard','post_step_full_dof_guard'):row[key]=deepcopy(full)
    for key in ('measured_motion_guard','post_step_measured_motion_guard'):row[key]=deepcopy(motion)
    assert tiago_guard_audit([row])['pass_check']
    row.pop('post_step_full_dof_guard')
    assert not tiago_guard_audit([row])['pass_check']


def test_private_recalibration_and_public_prediction_use_the_same_acquisition():
    home=np.eye(4);home[0,3]=1.
    model=NativeKinematicModel(dict(base=np.eye(4).tolist(),base_name='base',
        joints=[dict(name='private_hinge',parent='base',axis_parent=[0.,0.,1.],
            point_parent=[.2,0.,0.],revolute=True,omega=[0.,0.,1.],velocity=[0.,-.2,0.],
            q0=.2,lower=-2.,upper=2.,speed=.5)],
        homes={'base':np.eye(4).tolist(),'camera':home.tolist()},
        chains={'base':[],'camera':[0]},
        shapes=[dict(name=body,body=body,part=part,center_body=[0.,0.,0.],half_extents_body=[.1,.1,.1])
                for body,part in [('base','mobile_platform'),('camera','camera_manipulator')]],
        camera_parent='camera',camera_local=np.eye(4).tolist(),omnidirectional=True))
    state=np.array([.2,-.1,.4,.3])
    fresh=model.recalibrated(model.poses(state),{'private_hinge':.5})
    np.testing.assert_allclose(fresh.camera(np.zeros(4)),model.camera(state),atol=1e-12)
    moved=state.copy();moved[3]+=.1
    np.testing.assert_allclose(fresh.camera([0.,0.,0.,.1]),model.camera(moved),atol=1e-12)
    obs=fresh.predicted_observation(np.zeros(4),5.)
    assert obs.source=='predicted_fk' and obs.stamp_s==5.
    assert obs.occupancy['mobile_platform'] and obs.occupancy['camera_manipulator']
    import json
    assert 'private_hinge' not in json.dumps(obs.to_dict())


def test_public_candidate_binds_exact_executable_camera_knots():
    from jev_agent_wbc.multi_robot_planning import candidate_paths,card
    original,_,_=bind_scenario('orbit',observation())
    candidate=next(candidate_paths(original,'framing','explore',1.2))
    report=dict(physical_valid=True,minimum_clearance_m=.1,geometry_samples=[],
                scope='test',states=[[0.]*9],times_s=[0.])
    public=card(candidate,report,observation(),9.)
    fresh=observation(1.2,.6)
    executed=timed_path(candidate,fresh,9.)
    assert public['visual']['geometry']['request']['camera_path_id']==executed.trajectory.id
    assert public['trajectory']['execution']['execution_camera_path_id']==executed.trajectory.id
    assert 'states' not in public['screen']
    for t,p in zip(executed.trajectory.times_s,executed.trajectory.poses):
        expected=candidate.sample(t)
        np.testing.assert_allclose(p.translation_m,expected[0],atol=1e-12)
        np.testing.assert_allclose(p.rotation,expected[1],atol=1e-12)
    assert timed_path(candidate,fresh,9.01).trajectory.id!=executed.trajectory.id


def test_compact_candidate_keeps_scored_failure_and_exact_deformation_meaning():
    from copy import deepcopy
    from jev_agent_wbc.multi_robot_planning import candidate_paths,card
    from jev_agent_wbc.candidates import path_hash
    original,_,_=bind_scenario('orbit',observation())
    candidate=list(candidate_paths(original,'joint','explore',1.2))[2]
    intent=dict(intent_complete=False,intent_violations=[dict(kind='incomplete_visible_intent')],
                segments=[dict(effective_coverage=.7,complete=False,coverage_bin_dwell_s=[0.,.1])],
                terminal_frame=dict(pass_check=False,target=dict(shape_model='calibrated_cylinder'),
                    predicted_or_actual=dict(valid=True,centroid=[.5,.6])))
    before=deepcopy(intent)
    report=dict(physical_valid=True,minimum_clearance_m=.1,geometry_samples=[],
                scope='test',states=[[0.]*9],times_s=[0.],cinematic_prediction=intent)
    identity=path_hash(candidate)
    public=card(candidate,report,observation(),9.)
    assert path_hash(candidate)==identity==public['id']
    assert intent==before
    reduced=public['visual']['intent']
    assert reduced['intent_complete'] is False and reduced['intent_violations']==intent['intent_violations']
    assert reduced['segments'][0]['effective_coverage']==.7
    assert reduced['terminal_frame']['pass_check'] is False
    assert 'cinematic_prediction' not in public['screen']
    program=public['trajectory']['program']
    assert program['kind']=='returning_camera_deformation'
    assert program['blend_s']==candidate.envelope.blend_s
    assert program['return_s']==candidate.return_s
    assert program['offsets'] and program['terminal']=='recover'


def test_public_search_feedback_keeps_bounded_failure_scope_and_untested_count():
    from jev_agent_wbc.multi_robot_evidence import native_search_feedback
    from jev_agent_wbc.representation_agent import search_outcome
    intent=dict(geometric_complete=True,motion_trend=dict(pass_check=True),camera_limits_pass=True)
    events=[dict(event='camera_proposal_screen',intent=intent) for _ in range(6)]
    events.append(dict(event='candidate',card=dict(physical_valid=False,
        trajectory=dict(program=dict(offsets=dict(height_m=-.2))),
        screen=dict(tracking_screen_pass=False,observed_collision_screen_pass=True,
                    nonholonomic_screen_pass=True,native_platform_screen_pass=True))))
    feedback=native_search_feedback(events,-1.,0.)
    assert feedback['untested_proposals']==5 and feedback['search_exhausted'] is True
    public=search_outcome(feedback)
    assert public['outcomes']==dict(camera_path_not_realized=1)
    assert public['interpretation']=='bounded_search_outcome_not_global_infeasibility'
    assert public['failures'][0]['route']=='height_m -0.200 m'


def test_fresh_reference_context_uses_the_same_full_mount_frame_as_current_geometry():
    from jev_agent_wbc.multi_robot_evidence import reference_context
    first=observation();original,_,_=bind_scenario('orbit',first)
    fresh=Observation(4.,Pose.from_rt(Rotation.from_euler('xyz',[.1,-.2,.8]).as_matrix(),[3.,2.,.7]),
                      first.mount_from_camera,first.occupancy)
    snapshot=dict(observation_stamp=3.9,surfel_centers=[[20.,20.,20.]],surfel_radii=[.1],
                  surfel_velocities=[[0.,0.,0.]],surfel_track_ids=[0],surfel_ages=[0.])
    context,_=reference_context(original,fresh,snapshot,5.)
    assert context['original_reference']['observation_id']==fresh.id
    assert context['nominal_camera_forecast']['observation_id']==fresh.id
    for sample in context['original_reference']['samples']:
        restored=fresh.world_from_mount.compose(Pose.from_dict(sample['camera']))
        position,rotation,_=original.sample(sample['shot_s'])
        np.testing.assert_allclose(restored.translation_m,position,atol=1e-12)
        np.testing.assert_allclose(restored.rotation,rotation,atol=1e-12)


def test_causal_lateral_yaw_fit_predicts_origin_motion_without_future_samples():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/ocs2_wheeled_safety.py'
    spec=importlib.util.spec_from_file_location('native_safety_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fit=module.MeasuredLateralYawModel()
    for i in range(100):
        result=fit.update(i*.01,-.27*.2,.2)
    assert result['gain_m']==pytest.approx(-.27,abs=1e-7)
    assert result['fit_rmse_mps']<1e-7
    previous=fit.gain_m
    fit.update(1.,0.,0.)
    assert abs(fit.gain_m-previous)<1e-7
    with pytest.raises(ValueError,match='increase'):
        fit.update(1.,0.,0.)


def test_nonholonomic_tangent_chooses_nearest_forward_or_reverse_branch():
    from types import SimpleNamespace
    from jev_agent_wbc.multi_robot_runtime import AvoidanceRuntime
    runtime=AvoidanceRuntime.__new__(AvoidanceRuntime)
    runtime.scene_start=10.
    runtime.chosen=SimpleNamespace(duration=8.,sample=lambda t:(np.array([t,t,0.]),None,None))
    assert runtime.tangent_heading(10.,0.)==pytest.approx(np.pi/4)
    assert runtime.tangent_heading(18.,np.pi)==pytest.approx(5*np.pi/4)


def test_native_body_rate_derivative_and_causal_slip_calibration():
    from jev_agent_wbc.multi_robot_planning import platform_step
    state=np.array([.2,.4,-.3,.1]);previous=np.array([.18,.39,-.32,.08])
    rates,jac=platform_step(state,previous,1.2,.1)
    for i in range(4):
        a=state.copy();b=state.copy();a[i]+=1e-6;b[i]-=1e-6
        np.testing.assert_allclose(jac[:,i],(
            platform_step(a,previous,1.2,.1)[0]-platform_step(b,previous,1.2,.1)[0])/2e-6,atol=1e-8)
    # A world-axis bound is not a body-axis velocity certificate.
    rates,_=platform_step(np.array([.022,.022,0.]),np.zeros(3),np.pi/4,.1)
    assert rates[0]>.22 and abs(rates[1])<1e-10


def test_audit_uses_selected_clock_and_rejects_changed_selection():
    from copy import deepcopy
    from jev_agent_wbc.multi_robot_audit import restore_execution_evidence
    from jev_agent_wbc.multi_robot_planning import candidate_paths,card
    from jev_agent_wbc.candidates import path_hash,path_spec
    from jev_agent_wbc.camera_intent import progress_at
    original,_,_=bind_scenario('orbit',observation())
    path=next(candidate_paths(original,'joint','explore',1.))
    report=dict(physical_valid=True,minimum_clearance_m=.1,geometry_samples=[],scope='test')
    public=card(path,report,observation(),9.)
    selected=dict(card=public,lift=dict(report,path_id=path_hash(path),
        camera_path_id=public['visual']['geometry']['request']['camera_path_id']))
    initial=dict(path_id=path_hash(original),trajectory=path_spec(original))
    event=dict(stamp_s=9.5,path_id=path_hash(path))
    _,restored,installed=restore_execution_evidence(original,initial,selected,[event],9.)
    assert installed==9.5
    assert progress_at(restored,3.)==pytest.approx(progress_at(path,3.))
    assert abs(progress_at(restored,3.)-3.)>.5
    changed=deepcopy(selected);changed['lift']['camera_path_id']='changed'
    with pytest.raises(ValueError,match='knots'):
        restore_execution_evidence(original,initial,changed,[event],9.)
    changed=deepcopy(selected);changed['card']['trajectory']['program']['phase_rate']=-.9
    with pytest.raises(ValueError,match='identity'):
        restore_execution_evidence(original,initial,changed,[event],9.)
    with pytest.raises(ValueError,match='recorded selection'):
        restore_execution_evidence(original,initial,None,[event],9.)


@pytest.mark.parametrize('index',[2,3,4,5])
def test_selected_batched_spatial_knots_survive_recording_and_scalar_audit(index):
    from copy import deepcopy
    from jev_agent_wbc.multi_robot_audit import restore_execution_evidence
    from jev_agent_wbc.multi_robot_planning import candidate_paths,card
    from jev_agent_wbc.candidates import path_hash,path_spec
    original,_,_=bind_scenario('orbit',observation())
    path=list(candidate_paths(original,'joint','explore',.9))[index]
    report=dict(physical_valid=True,minimum_clearance_m=.1,geometry_samples=[],scope='test')
    public=card(path,report,observation(),9.)
    selected=dict(card=public,lift=dict(report,path_id=path_hash(path),
        camera_path_id=public['visual']['geometry']['request']['camera_path_id']),
        camera_trajectory=timed_path(path,observation(),9.).trajectory.to_dict())
    initial=dict(path_id=path_hash(original),trajectory=path_spec(original))
    events=[dict(stamp_s=9.5,path_id=path_hash(path))]
    restore_execution_evidence(original,initial,selected,events,9.)
    changed=deepcopy(selected)
    changed['camera_trajectory']['samples'][90]['pose']['translation_m'][0]+=.001
    with pytest.raises(ValueError,match='knots'):
        restore_execution_evidence(original,initial,changed,events,9.)


def test_real_typed_decider_waits_for_fresh_numerical_seed_without_network(monkeypatch,tmp_path):
    """Exercise all three unchanged decision stages with a fake transport only."""
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    from copy import deepcopy
    import io,json,time
    from types import SimpleNamespace
    from pathlib import Path
    from jev_agent_wbc import jev,floating_camera,multi_robot_planning,native_visibility
    from jev_agent_wbc.multi_robot_runtime import AvoidanceRuntime
    calls=[];model_inputs=[]
    class LocalTransport:
        def __init__(self,**kwargs):
            assert kwargs['url']=='https://api.typesafe.ai/v1/systemone'
            assert kwargs['model']=='jev-1.14.0'
        def ask(self,evidence,questions):
            calls.append(deepcopy(evidence))
            key=next(iter(questions))
            value={'strategy':'combined','configuration':'wider_search','candidate':'c0'}[key]
            return {key:dict(choice=value,probabilities={value:1.})}
    def screen(path,model,observed,**kwargs):
        model_inputs.append((deepcopy(model),deepcopy(kwargs)))
        return dict(physical_valid=True,minimum_clearance_m=.1,geometry_samples=[],
                    scope='unit_fixture_not_physical_evidence')
    monkeypatch.setattr(jev,'SystemOneClient',LocalTransport)
    monkeypatch.setenv('TYPESAFE_API_KEY_FILE','unused_local_fixture')
    monkeypatch.setenv('TYPESAFE_API_KEY','unused_local_fixture')
    monkeypatch.setattr(multi_robot_planning,'lift',screen)
    class AttachmentFixture:
        def __init__(self,model,*args):
            assert model['private_hinge']=='fresh'
        def clearance(self,path):return .1
        def rank(self,clearance,score):return (False,-clearance)
    monkeypatch.setattr(multi_robot_planning,'CameraAttachmentScreen',AttachmentFixture)
    monkeypatch.setattr(floating_camera,'evaluate_path',lambda *a,**k:(
        SimpleNamespace(feasible=True,min_clearance_m=.1,to_dict=lambda:dict(feasible=True)),None))
    monkeypatch.setattr(native_visibility,'intent_prediction',lambda *a,**k:dict(
        geometric_complete=True,motion_trend=dict(pass_check=True),camera_limits_pass=True))
    runtime=AvoidanceRuntime.__new__(AvoidanceRuntime)
    runtime.original,scene,_=bind_scenario('orbit',observation())
    runtime.scene_start=9.;runtime.subject_model=scene['subject'];runtime.intrinsics=[.5,.5,.5,.5]
    runtime.package=Path(__file__).resolve().parents[1];runtime.output=tmp_path;runtime.robot='husky'
    runtime.events=io.StringIO();runtime.seed_request=None
    snapshot=dict(observation_stamp=3.,surfel_centers=[[3.,2.,1.]],surfel_radii=[.1],
                  surfel_track_ids=[1],surfel_subject_mask=[False],tracks=[])
    fresh=replace(observation(),stamp_s=9.05)
    with ThreadPoolExecutor(max_workers=1) as pool:
        runtime.planning_pool=pool
        future=pool.submit(runtime._plan,snapshot,observation(),dict(private_hinge='old'))
        deadline=time.monotonic()+2.
        while runtime.seed_request is None and not future.done() and time.monotonic()<deadline:
            time.sleep(.001)
        assert runtime.seed_request is not None and not future.done()
        seed,min_stamp=runtime.seed_request
        assert min_stamp==pytest.approx(6.9)  # Fresh construction during the fixed pre-roll.
        seed.set_result((dict(snapshot,observation_stamp=9.),fresh,dict(private_hinge='fresh')))
        path,report=future.result(timeout=2.)
    assert [call['stage'] for call in calls]==['strategy','configuration','selection']
    assert calls[0]['snapshot_time_s']==-6.
    assert calls[-1]['snapshot_time_s']==pytest.approx(.05)
    assert all(model['private_hinge']=='fresh' for model,_ in model_inputs)
    assert len(model_inputs)==1  # A valid result need not consume the maximum.
    assert all(options['progress_start_s']==pytest.approx(.05) for _,options in model_inputs)
    assert report['source_observation_stamp_s']==9.05 and runtime.seed_request is None
    assert 'private_hinge' not in json.dumps(calls)
    assert (tmp_path/'selected_plan.json').is_file()


def test_network_only_trials_remain_unverified_and_probes_never_become_agent_success():
    from jev_agent_wbc.multi_robot_audit import classify_agent_trial
    transport=[dict(status='transport_unverified_retry')]
    assert classify_agent_trial('jev',False,[],transport)=='unverified_network'
    assert classify_agent_trial('probe',False,[],[])=='diagnostic_not_agent'
    failed=[dict(event='fresh_installation_check',check=dict(reason='no_native_model_screened_candidate'))]
    assert classify_agent_trial('jev',False,failed,transport)=='no_feasible_native_candidate'
    assert classify_agent_trial('jev',False,failed+[dict(event='late_plan_rejected')],[])=='failed_installation'


def test_native_obstacle_relations_use_projected_surface_and_mount_frame():
    from jev_agent_wbc.multi_robot_geometry import obstacle_relations
    obs=observation(yaw=np.pi/2)
    # At acquisition the measured proxy is 0.9 m in front of the mount;
    # it moves another 0.02 m before this robot observation.
    center=obs.world_from_mount.apply([[.9,0.,0.]])[0]
    snapshot=dict(observation_stamp=2.8,surfel_centers=[center],surfel_radii=[.1],
        surfel_velocities=[[0.,.1,0.]],surfel_track_ids=[1],surfel_ages=[.1],
        motion_uncertainty_rate_m_s=.035,surfel_radius_rates=[.02])
    result=obstacle_relations(obs,snapshot)
    platform=next(r for r in result['relations'] if r['part']=='mobile_platform')
    np.testing.assert_allclose(platform['center_m'],[.92,0.,0.],atol=1e-12)
    np.testing.assert_allclose(platform['velocity_m_s'],[.1,0.,0.],atol=1e-12)
    assert platform['radius_m']==pytest.approx(.1+.035*.3+.02*.2)
    assert platform['proxy_clearance_m']==pytest.approx(.92-.4-platform['radius_m'])
    assert platform['obstacle_id']=='track:1' and platform['direction']=='front'
    assert result['frame']=='mount_snapshot'


@pytest.mark.parametrize('stamp',[2.7,3.01])
def test_native_obstacle_relations_reject_stale_and_future_acquisitions(stamp):
    from jev_agent_wbc.multi_robot_geometry import obstacle_relations
    result=obstacle_relations(observation(),dict(observation_stamp=stamp,surfel_centers=[[3.,4.,1.]]))
    assert result['status']=='unknown' and result['relations']==[]
