"""Cross-embodiment contracts, conservative geometry and real WBC priority."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.representation import (Pose,Box,Observation,Request,CompiledGuidance,
    compress_boxes,PARTS)
from jev_agent_wbc.representation_jvs import JVSGeometryBackend,request_for_profile
from jev_agent_wbc.whole_body_qp import whole_body_kinematics
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.intent_paths import IntentPath
from jev_agent_wbc.candidates import path_hash
from jev_agent_wbc.jev import compact_evidence,CandidateDecider
from jev_agent_wbc.search_configuration import offers

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture():
    runtime=json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    state=np.array(runtime['initial_state']); z=runtime['base_z']
    original=OriginalReference(ROOT/'scenarios/movein/task.json',runtime['subject']['position'])
    path=IntentPath(original,original,.5,dict(route='above',amplitude_m=.1,extension_s=0.))
    return state,z,path


def test_transform_chain_tilted_mobile_platform_and_roundtrip(fixture):
    state,z,_=fixture
    state[:3]=[1.,-.3,.5]
    rotation=Rotation.from_euler('xyz',[.2,-.3,.5]).as_matrix()
    obs=JVSGeometryBackend().observe(state,2.,base_z=z,base_rotation=rotation)
    p,r,_=whole_body_kinematics(state,z,rotation)
    np.testing.assert_allclose(obs.world_from_camera.translation_m,p,atol=1e-14)
    np.testing.assert_allclose(obs.world_from_camera.rotation,r,atol=1e-14)
    clone=Observation.from_dict(json.loads(json.dumps(obs.to_dict(),allow_nan=False)))
    assert clone.id==obs.id
    np.testing.assert_allclose(obs.world_from_mount.inverse().apply(obs.world_from_mount.apply([[1,2,3]])),[[1,2,3]],atol=1e-14)


@pytest.mark.parametrize('budget',[1,3,6])
def test_compressed_boxes_contain_every_backend_primitive(fixture,budget):
    state,z,_=fixture; backend=JVSGeometryBackend(budget)
    rng=np.random.default_rng(4)
    for _ in range(8):
        varied=state+rng.normal(0,.2,9)
        rotation=Rotation.from_euler('xyz',rng.normal(0,.3,3)).as_matrix()
        _,_,raw=backend.primitive_boxes(varied,z,rotation)
        obs=backend.observe(varied,0.,base_z=z,base_rotation=rotation)
        for part in PARTS:
            assert len(obs.occupancy[part])<=budget
            for primitive in raw[part]:
                # Every entire primitive box must fit within at least one merged box.
                assert any(np.all(np.abs(primitive.vertices-b.center_m)<=np.array(b.half_extents_m)+1e-12)
                           for b in obs.occupancy[part])


def test_sphere_cylinder_and_frustum_support_is_conservative(fixture):
    state,z,_=fixture;b=JVSGeometryBackend()
    mount,_,boxes=b.primitive_boxes(state,z)
    frames=b.geometry.frames(state,z)[0]; counts=dict.fromkeys(PARTS,0)
    rng=np.random.default_rng(7)
    for body,_,kind,size,offset,rotation in b.geometry.shapes:
        part=PARTS[body!=0]; box=boxes[part][counts[part]];counts[part]+=1
        if kind=='box':local=Box((0,0,0),tuple(size/2)).vertices
        elif kind=='sphere':
            d=rng.normal(size=(300,3));local=d/np.linalg.norm(d,axis=1)[:,None]*size[0]
        else:
            theta=np.linspace(0,2*np.pi,300)
            radius=size[0] if kind=='cylinder' else max(size[:2])
            height=size[1] if kind=='cylinder' else size[2]
            local=np.vstack([np.c_[radius*np.cos(theta),radius*np.sin(theta),np.full(300,h)] for h in (-height/2,height/2)])
        p,r=frames[body]
        points=mount.inverse().apply((local@rotation.T+offset)@r.T+p)
        assert np.max(box.distance(points))<1e-12


@pytest.mark.parametrize('mount,camera,empty',[
    (Pose((0,0,.3)),Pose((.2,0,.5)),False), # wheeled base + arm
    (Pose((0,0,.6),tuple(Rotation.from_euler('x',.2).as_quat())),Pose((.4,0,.3)),False), # quadruped fixture
    (Pose((0,0,1.1)),Pose((.3,0,.4)),False), # humanoid fixture
    (Pose((1,2,3),tuple(Rotation.from_euler('xyz',[.2,.3,.4]).as_quat())),Pose((0,0,-.1)),True), # rigid aerial camera
])
def test_same_contract_for_synthetic_embodiments(mount,camera,empty):
    obs=Observation(1.,mount,camera,{PARTS[0]:(Box((0,0,0),(.2,.2,.1)),),
        PARTS[1]:() if empty else (Box((.1,0,.2),(.1,.1,.2)),)},'synthetic_fk')
    assert Observation.from_dict(obs.to_dict()).to_dict()==obs.to_dict()
    request=Request(obs.id,'opaque-camera-path',1.,2.)
    result=CompiledGuidance(request,obs,None).feedback(obs)
    assert result['evidence_kind']=='predicted' and result['backend_status']=='unknown'
    assert result['preference_residual_norm']==0
    text=json.dumps(obs.to_dict())
    assert all(name not in text for name in ('joint','wheel','link','actuator'))


def test_single_box_can_fill_a_gap_that_union_preserves():
    boxes=[Box((-.5,0,0),(.1,.1,.1)),Box((.5,0,0),(.1,.1,.1))]
    assert min(b.distance([[0,0,0]])[0] for b in compress_boxes(boxes,3))==pytest.approx(.4)
    assert compress_boxes(boxes,1)[0].distance([[0,0,0]])[0]<0


@pytest.mark.parametrize('bad',[
    {'translation_m':[0,0,float('nan')],'quaternion_xyzw':[0,0,0,1]},
    {'translation_m':[0,0,0],'quaternion_xyzw':[0,0,0,2]},
    {'translation_m':[0,0,0],'quaternion_xyzw':[0,0,0,0]},
    {'translation_m':['0',0,0],'quaternion_xyzw':[0,0,0,1]},
    {'translation_m':[0,0,0],'quaternion_xyzw':[0,0,0,1],'frame':'implicit'},
])
def test_reject_invalid_pose(bad):
    with pytest.raises(ValueError):Pose.from_dict(bad)


def test_observation_rejects_unknown_versions_units_frames_and_internal_fields(fixture):
    state,z,_=fixture; document=JVSGeometryBackend().observe(state,0.,base_z=z).to_dict()
    for key,value in [('schema','v2'),('units','cm'),('occupancy_frame','world'),('joint_positions',[1,2])]:
        bad=deepcopy(document);bad[key]=value
        with pytest.raises(ValueError):Observation.from_dict(bad)
    bad=deepcopy(document);bad['occupancy']['other']=[]
    with pytest.raises(ValueError):Observation.from_dict(bad)


def test_quaternion_sign_has_one_serialization():
    assert Pose((0,0,0),(0,0,0,1)).to_dict()==Pose((0,0,0),(0,0,0,-1)).to_dict()


def test_binding_mutation_unknown_request_fields_and_intervals(fixture):
    state,z,path=fixture;b=JVSGeometryBackend();obs=b.observe(state,0.,base_z=z)
    request=request_for_profile(obs,path,'base')
    guide=b.compile(request,obs,path)
    assert Request.from_dict(request.to_dict()).to_dict()==request.to_dict()
    with pytest.raises(ValueError,match='observation binding'):b.compile(replace(request,observation_id='wrong'),obs,path)
    with pytest.raises(ValueError,match='camera path binding'):b.compile(replace(request,camera_path_id='wrong'),obs,path)
    with pytest.raises(ValueError,match='deadline'):b.compile(replace(request,end_s=99.),obs,path)
    with pytest.raises(ValueError,match='activation'):b.compile(replace(request,start_s=0.),obs,path)
    with pytest.raises(ValueError):replace(request,end_s=request.start_s)
    data=request.to_dict();data['joints']=[0]
    with pytest.raises(ValueError):Request.from_dict(data)
    before=guide.request.to_dict()
    request.pose_preferences[0]['weight']=9.
    assert guide.request.to_dict()==before  # no caller-owned nested references


def test_dynamic_observation_and_unknown_space(fixture):
    state,z,_=fixture;b=JVSGeometryBackend();obs=b.observe(state,.2,base_z=z)
    scene=dict(observation_stamp=0.,surfel_centers=[[1,0,0]],surfel_radii=[.1],
        surfel_velocities=[[1,0,0]],surfel_track_ids=[2],surfel_ages=[.1],motion_uncertainty_rate_m_s=.05)
    result=b.obstacle_relations(obs,scene)
    row=result['relations'][0]
    assert row['radius_m']==pytest.approx(.115)
    np.testing.assert_allclose(row['center_m'],obs.world_from_mount.inverse().apply([1.2,0,0]))
    assert b.obstacle_relations(obs,dict(scene,observation_stamp=-1))['status']=='unknown'
    assert b.obstacle_relations(obs,dict(scene,surfel_centers=[]))['status']=='unknown'


def test_pose_preference_tolerance_and_relative_frame_semantics(fixture):
    state,z,path=fixture;b=JVSGeometryBackend();obs=b.observe(state,0.,base_z=z)
    request=request_for_profile(obs,path,'base');guide=b.compile(request,obs,path)
    assert np.linalg.norm(guide.residual(obs))<1e-12
    moved=state.copy();moved[:2]+=[.5,-.2]
    after=b.observe(moved,1.,base_z=z)
    # Relative following/compactness is invariant to rigid global translation.
    assert np.linalg.norm(guide.residual(after))<1e-12
    world_pref=deepcopy(request.pose_preferences[0]);world_pref.update(frame='world',target=obs.world_from_mount.to_dict())
    fixed=b.compile(replace(request,pose_preferences=(world_pref,)),obs,path)
    assert np.linalg.norm(fixed.residual(after))>.1


@pytest.mark.parametrize('part',PARTS)
@pytest.mark.parametrize('relation',['stay_in_region','avoid_region'])
def test_occupancy_relations_affect_entire_part(fixture,part,relation):
    state,z,path=fixture;b=JVSGeometryBackend();obs=b.observe(state,0.,base_z=z)
    region=Box.enclosing(np.vstack([box.vertices for box in obs.occupancy[part]]))
    pref=dict(part=part,frame='mount',relation=relation,region=region.to_dict(),margin_m=.01,weight=1.)
    request=Request(obs.id,path_hash(path),path.activation_s,path.duration,occupancy_preferences=(pref,))
    guide=b.compile(request,obs,path)
    assert np.linalg.norm(guide.residual(obs))>0
    H,g=b.quadratic_guidance(guide,state,1.,z)
    assert H.shape==(8,8) and np.isfinite(g).all()
    assert np.linalg.eigvalsh(H).min()>-1e-10
    H0,g0=b.quadratic_guidance(guide,state,.1,z)
    assert not H0.any() and not g0.any()


def test_agent_boundary_excludes_robot_internals(fixture):
    state,z,_=fixture;b=JVSGeometryBackend();obs=b.observe(state,0.,base_z=z)
    evidence=dict(robot_geometry=obs.summary(),actual_robot_feedback=dict(state=state.tolist()),
        robot_capability=dict(joints=[3]),difficulty=dict(joint_margin_rad=.2,issues=['numerical_kinematic_gate_failed']),
        execution_history=[dict(time_s=0.,joint_margin_rad=.2)],
        validation_feedback=[dict(time_s=0.,error='joint1 failed',wbc_feasibility=dict(pairs=[dict(link='link1')]))],
        search_options=offers('view',{}))
    text=json.dumps(compact_evidence(evidence))
    for forbidden in ('joint','link1','actuator','allocation_order','actual_robot_feedback','arm_assisted'):
        assert forbidden not in text
    class Client:
        def ask(self,evidence,questions):
            assert 'platform_follow' in questions['configuration']['criteria']
            assert 'base' not in questions['configuration']['criteria']
            return dict(configuration=dict(choice='platform_follow'))
    decision=CandidateDecider(Client()).decide(dict(evidence,stage='configuration'))
    assert decision['search_config']=='base'


def test_geometry_preference_cannot_relax_real_wbc_primary_or_safety():
    from test_tpr_wbc import controller,STATE
    c=controller();b=JVSGeometryBackend();obs=b.observe(STATE,0.)
    p,r,J=whole_body_kinematics(STATE)
    pref=dict(part='mobile_platform',frame='world',target=Pose((10.,10.,10.)).to_dict(),
        position_tolerance_m=[0,0,0],orientation_tolerance_rad=0.,weight=10.)
    request=Request(obs.id,path_hash(c.trajectory),0.,1.,pose_preferences=(pref,))
    c.geometry_backend=b;c.geometry_guidance=b.compile(request,obs,c.trajectory)
    c.reconfiguration.update=lambda *args:np.zeros(8)
    u,info=c.solve(STATE,p,r,np.zeros(6),.02,commanded_q=STATE[3:])
    assert info['tpr_report']['geometry_preference_active']
    np.testing.assert_allclose(J@u,0.,atol=1e-8)
    assert max(abs(u)-c.limits.acceleration*.02)<1e-6
    assert info['command_sweep_lower_m']>=.025
    assert info['qp_violation']<1e-6


def test_pose_hash_stable_across_random_json_roundtrips():
    rng=np.random.default_rng(1)
    for _ in range(100):
        mount=Pose((0,0,0),tuple(Rotation.random(random_state=rng).as_quat()))
        obs=Observation.from_fk(0.,mount,mount.compose(Pose((0,0,1))),
            {PARTS[0]:(Box((0,0,0),(.1,.1,.1)),),PARTS[1]:()},source='synthetic_fk')
        assert Observation.from_dict(obs.to_dict()).id==obs.id


def test_geometry_guidance_moves_platform_in_camera_nullspace():
    from test_tpr_wbc import controller,STATE
    from jev_agent_wbc.actuation import advance_command
    c=controller();b=JVSGeometryBackend();obs=b.observe(STATE,0.)
    target=np.array(obs.world_from_mount.translation_m)+[.08,0,0]
    pref=dict(part='mobile_platform',frame='world',target=Pose(tuple(target)).to_dict(),
        position_tolerance_m=[0,0,0],orientation_tolerance_rad=.3,weight=10.)
    request=Request(obs.id,path_hash(c.trajectory),0.,1.,pose_preferences=(pref,))
    guide=b.compile(request,obs,c.trajectory);c.geometry_backend=b;c.geometry_guidance=guide
    c.reconfiguration.update=lambda *args:np.zeros(8)
    state=STATE.copy();initial=np.linalg.norm(guide.residual(obs))
    for i in range(15):
        t=i*.02;c.set_time(t);old=c.previous.copy()
        u,info=c.solve(state,*c.trajectory.sample(t),.02,trajectory_time=t,commanded_q=state[3:])
        state=advance_command(state,old,u,.02,True)
        assert info['command_sweep_lower_m']>=.025
    assert np.linalg.norm(guide.residual(b.observe(state,.3)))<initial-.005
    np.testing.assert_allclose(whole_body_kinematics(state)[0],c.trajectory.sample(.3)[0],atol=2e-4)


def test_replacing_camera_path_clears_old_geometry_guidance():
    from test_tpr_wbc import controller,Reference
    c=controller();c.geometry_guidance=object()
    c.set_trajectory(Reference())
    assert c.geometry_guidance is None


def test_semantic_preview_is_small_but_request_is_exact(fixture):
    from jev_agent_wbc.representation_agent import geometric_card
    state,z,path=fixture;b=JVSGeometryBackend();obs=b.observe(state,0.,base_z=z)
    request=request_for_profile(obs,path,'base');guide=b.compile(request,obs,path)
    geometry=dict(observation=obs.to_dict(),request=request.to_dict(),
        samples=[guide.feedback(b.observe(state,float(i),base_z=z,source='predicted_fk')) for i in range(5)],scope='sampled')
    card=geometric_card(dict(physical_valid=True,visual=dict(geometry=geometry)))
    assert len(card['predicted_geometry'])==3
    assert card['request']==request.to_dict()
    assert len(json.dumps(card))<len(json.dumps(geometry))/2
    assert 'regions' not in json.dumps(card['predicted_geometry'])
    payload=compact_evidence(dict(candidates=[dict(geometry=card)]))
    assert payload['candidates'][0]['geometry']['request']==request.to_dict()


def test_preview_budget_keeps_all_choices_and_exact_geometry_requests(fixture):
    from jev_agent_wbc.representation_agent import bound_geometry_preview
    state,z,path=fixture;b=JVSGeometryBackend();obs=b.observe(state,0.,base_z=z)
    request=request_for_profile(obs,path,'base').to_dict()
    cards=[dict(id=f'c{i}',geometry=dict(request=deepcopy(request),
        predicted_geometry=[dict(stamp_s=j,large_preview='x'*3000) for j in range(3)])) for i in range(3)]
    out=bound_geometry_preview(dict(robot_geometry=obs.summary(),candidates=cards))
    assert len(out['candidates'])==3
    for card in out['candidates']:
        assert card['geometry']['request']==request
        assert len(card['geometry']['predicted_geometry'])<=1
        if not card['geometry']['predicted_geometry']:
            assert card['geometry']['preview_scope']=='omitted_for_text_budget_full_timed_geometry_in_journal'
