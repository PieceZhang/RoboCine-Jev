"""Independent scalar oracles for batched geometry and sparse interval bounds."""
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.kc_svsdf import CollisionGeometry,primitive_sdf,advance,advance_sequence
from jev_agent_wbc.representation_jvs import JVSGeometryBackend,request_for_profile
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.intent_paths import IntentPath,poses_at,twists_at
from jev_agent_wbc.candidates import path_hash

ROOT=Path(__file__).resolve().parents[1]/'scenarios/movein'


def test_async_journal_drains_ordered_evidence_on_close(tmp_path):
    from jev_agent_wbc.async_journal import AsyncJournal
    path=tmp_path/'events.jsonl';journal=AsyncJournal(path)
    for i in range(100):
        journal.write(json.dumps(dict(index=i,payload='记录'*1000))+'\n');journal.flush()
    journal.close();journal.close()
    assert [json.loads(line)['index'] for line in path.read_text().splitlines()]==list(range(100))


def test_background_geometry_feedback_preserves_acquisition_binding():
    from copy import copy
    from jev_agent_wbc.engine import geometry_feedback_snapshot
    state,z,path=fixture();backend=JVSGeometryBackend()
    initial=backend.observe(state,0.,base_z=z)
    guide=backend.compile(request_for_profile(initial,path,'base'),initial,path)
    snapshot=copy(guide);snapshot.camera_path=None
    moved=state.copy();moved[4]+=.1
    expected=guide.feedback(backend.observe(moved,1.25,base_z=z),'measured')
    feedback,encoded=geometry_feedback_snapshot(snapshot,3,moved,1.25,z,None)
    assert feedback==expected and json.loads(encoded)==feedback
    assert feedback['stamp_s']==1.25 and feedback['request_id']==guide.request.id


@pytest.mark.parametrize('name',['orbit','movein','pedestal_up'])
def test_batch_terminal_axis_search_matches_scalar_bisection(name):
    from jev_agent_wbc.cinematic_geometry import terminal_band,terminal_error,inside_terminal_band
    from jev_agent_wbc.camera_intent import default_intrinsics
    import itertools
    root=ROOT.parent/name;model=json.loads((root/'runtime_scene.json').read_text())['subject']
    original=OriginalReference(root/'task.json',model['position']);intrinsics=default_intrinsics()
    expected=[np.zeros(3)];position=original.sample(original.duration)[0]
    for direction in itertools.product((-1.,0.,1.),repeat=3):
        if sum(v!=0 for v in direction)!=1:continue
        direction=np.asarray(direction);lo,hi=0.,.35
        for _ in range(14):
            mid=(lo+hi)/2
            if inside_terminal_band(terminal_error(original,position+mid*direction,model,intrinsics),.8):lo=mid
            else:hi=mid
        expected.append(lo*direction)
    actual=terminal_band(original,model,intrinsics,axis_only=True)
    np.testing.assert_array_equal(actual['offsets_m'],expected)


def test_batched_frustum_matches_scalar_edge_projection_and_finite_difference():
    rng=np.random.default_rng(912);points=rng.uniform(-.3,.3,(120,3));size=[.18,.13,.12]
    distance,normal=primitive_sdf(points,'frustum',size,True)
    edges=[(np.array(a),np.array(b)) for a,b in
        [([0.,-.06],[.18,-.06]),([.18,-.06],[.13,.06]),([.13,.06],[0.,.06])]]
    expected=[]
    for point in points:
        q=np.array([np.linalg.norm(point[:2]),point[2]])
        distances=[]
        for a,b in edges:
            v=b-a;u=min(1.,max(0.,float((q-a)@v/(v@v))))
            distances.append(np.linalg.norm(q-a-u*v))
        inside=abs(q[1])<=.06 and q[0]<=.18+(.13-.18)*(q[1]/.12+.5)
        expected.append(min(distances)*(-1. if inside else 1.))
    np.testing.assert_allclose(distance,expected,atol=1e-15)
    epsilon=1e-6
    jac=np.column_stack([(primitive_sdf(points+epsilon*np.eye(3)[j],'frustum',size)-
                          primitive_sdf(points-epsilon*np.eye(3)[j],'frustum',size))/(2*epsilon) for j in range(3)])
    np.testing.assert_allclose(normal,jac,atol=1e-7)


def test_batch_frame_moments_match_scalar_hull_oracle():
    from jev_agent_wbc.subject_state import frame_state,frame_states
    from jev_agent_wbc.reference import look_at
    rng=np.random.default_rng(892);model=dict(position=[1.,.3,.8],radius=.08,height=.3,
        orientation_xyzw=Rotation.from_euler('xyz',[.2,-.1,.3]).as_quat().tolist())
    intrinsics=[1.,1.2,.5,.5];positions=rng.uniform(-2,2,(80,3))
    rotations=np.array([look_at(p,model['position']) for p in positions])
    rotations[::10]=Rotation.from_rotvec(rng.normal(size=(8,3))).as_matrix()
    frames=frame_states(positions,rotations,model,intrinsics)
    for p,r,actual in zip(positions,rotations,frames):
        expected=frame_state(p,r,model,intrinsics)
        assert actual.keys()==expected.keys()
        for k,v in expected.items():
            if isinstance(v,(str,bool)) or v is None:assert actual[k]==v
            elif k=='silhouette_axis_rad':assert abs((actual[k]-v+np.pi/2)%np.pi-np.pi/2)<1e-10
            else:np.testing.assert_allclose(actual[k],v,rtol=2e-10,atol=2e-12)


def test_task_margin_certificate_is_a_bound_on_independent_lp():
    from scipy.optimize import linprog
    from jev_agent_wbc.tpr_wbc import task_margin_witness
    rng=np.random.default_rng(192)
    for _ in range(30):
        J=rng.normal(size=(6,8));point=rng.uniform(-.5,.5,8)
        desired=J@point;A=np.eye(8);lo=-np.ones(8);hi=np.ones(8)
        R=rng.normal(size=(25,8));scale=np.maximum(np.linalg.norm(R,axis=1)*.02,1e-5)
        low=R@point-rng.uniform(.1,1.,25)*scale
        bound=task_margin_witness(point+rng.normal(0,1e-8,8),J,desired,A,lo,hi,R,low,scale)
        assert bound is not None and bound>=.05
        result=linprog(np.r_[np.zeros(8),-1.],
            A_ub=np.vstack([np.c_[A,np.zeros(8)],np.c_[-A,np.zeros(8)],np.c_[-R,scale]]),
            b_ub=np.r_[hi,-lo,-low],A_eq=np.c_[J,np.zeros(6)],b_eq=desired,
            bounds=[(None,None)]*9,method='highs')
        assert result.success and bound<=result.x[-1]+1e-8
    J=np.c_[np.eye(6),np.zeros((6,2))];R=np.eye(8);point=np.zeros(8)
    # An equality correction which violates an actuator row is not evidence.
    assert task_margin_witness(point,J,np.ones(6)*2,np.eye(8),-np.ones(8),np.ones(8),R,-np.ones(8),np.ones(8)) is None
    # A weak witness does not decide that the optimum is also below threshold.
    assert task_margin_witness(point,J,np.zeros(6),np.eye(8),-np.ones(8),np.ones(8),R,-np.ones(8)*.01,np.ones(8)) is None


@pytest.mark.parametrize('regular',[False,True])
def test_batched_camera_pose_matches_scalar_formula_and_twist(regular):
    _,_,parent=fixture()
    path=IntentPath(parent.original,parent,1.,dict(route='above',amplitude_m=.06,extension_s=0.,regular_curve=regular))
    times=np.r_[-.1,0.,.5,1.,np.linspace(1.001,8.5,100)]
    pp,rr=poses_at(path,times)
    np.testing.assert_allclose(twists_at(path,times),[path.sample(float(t))[2] for t in times],atol=2e-11)
    np.testing.assert_allclose(twists_at(parent.original,times),[parent.original.sample(float(t))[2] for t in times],atol=2e-11)
    for t,p,r in zip(times,pp,rr):
        oldp,oldr=path.pose(t)
        np.testing.assert_allclose(p,oldp,atol=2e-14)
        np.testing.assert_allclose(r,oldr,atol=2e-14)
        actual=path.sample(t)
        np.testing.assert_allclose(actual[0],oldp,atol=2e-14)
        np.testing.assert_allclose(actual[1],oldr,atol=2e-14)
        if 1.<t<8.:
            a,b=max(1.,t-1e-4),min(8.,t+1e-4)
            expected=np.r_[path.translation(t,1),Rotation.from_matrix(path.pose(b)[1]@path.pose(a)[1].T).as_rotvec()/(b-a)]
            np.testing.assert_allclose(actual[2],expected,atol=2e-11)


def fixture():
    runtime=json.loads((ROOT/'runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'task.json',runtime['subject']['position'])
    path=IntentPath(original,original,.5,dict(route='above',amplitude_m=.1,extension_s=0.))
    return np.array(runtime['initial_state']),runtime['base_z'],path


def test_batched_sdf_gradients_and_swept_poses_match_scalar_oracles():
    rng=np.random.default_rng(412);geometry=CollisionGeometry()
    states=rng.uniform(-1,1,(8,9));centers=rng.uniform(-2,2,(8,23,3))
    rotations=Rotation.from_rotvec(rng.uniform(-1,1,(8,3))).as_matrix()
    positions,orientations=geometry.frame_arrays(states,.2,rotations)
    sequence=geometry.distances_sequence(states,centers,.2,rotations)
    for k,state in enumerate(states):
        frames=geometry.frames(state,.2,rotations[k])[0]
        np.testing.assert_allclose(positions[k],[p for p,_ in frames],atol=1e-14)
        np.testing.assert_allclose(orientations[k],[r for _,r in frames],atol=1e-14)
        distances,normals=geometry.primitive_arrays(frames,centers[k],True)
        expected=np.full((23,8),np.inf)
        for index,(body,_,kind,size,offset,local_rotation) in enumerate(geometry.shapes):
            p,r=frames[body];rotation=r@local_rotation
            distance,normal=primitive_sdf((centers[k]-(p+r@offset))@rotation,kind,size,True)
            np.testing.assert_allclose(distances[:,index],distance,atol=2e-14)
            np.testing.assert_allclose(normals[:,index],normal@rotation.T,atol=2e-14)
            expected[:,body]=np.minimum(expected[:,body],distance)
        np.testing.assert_allclose(sequence[k],expected,atol=2e-14)
    times=np.linspace(-.2,.2,21);control=rng.uniform(-1,1,8)
    np.testing.assert_allclose(advance_sequence(states[0],control,times),
        [advance(states[0],control,t) for t in times],atol=1e-14)


@pytest.mark.parametrize('frame',['world','mount_snapshot','camera_snapshot','mount','camera'])
def test_array_preferences_match_validated_contract_residuals(frame):
    state,z,path=fixture();backend=JVSGeometryBackend();observation=backend.observe(state,0.,base_z=z)
    request=request_for_profile(observation,path,'base')
    pose=dict(request.pose_preferences[0],frame=frame)
    occupancy=dict(request.occupancy_preferences[0],frame=frame)
    request=replace(request,pose_preferences=(pose,),occupancy_preferences=(occupancy,))
    guide=backend.compile(request,observation,path)
    rng=np.random.default_rng(109)
    for _ in range(5):
        moved=state+rng.uniform(-.15,.15,9)
        expected=guide.residual(backend.observe(moved,1.,base_z=z))
        actual=guide.residual_arrays(*backend.residual_geometry(moved,z))
        np.testing.assert_allclose(actual,expected,atol=2e-13)
        if frame=='mount':
            unmerged=guide.residual_arrays(*backend.residual_geometry(moved,z,compress=False))
            np.testing.assert_allclose(unmerged,expected,atol=2e-13)


def test_adaptive_self_sweep_bounds_dense_nonlinear_motion():
    from jev_agent_wbc.self_collision import SelfCollision
    from jev_agent_wbc.actuation import advance_command
    model=SelfCollision();state,_,_=fixture();rng=np.random.default_rng(72)
    states=state+rng.uniform(-.3,.3,(9,9))
    np.testing.assert_allclose(model.distances_sequence(states),[model.distances(s) for s in states],atol=1e-14)
    for _ in range(5):
        moved=state+rng.uniform(-.1,.1,9)
        old,new=rng.uniform(-.2,.2,(2,8));dt=.1
        bound=model.sweep(moved,old,new,dt)
        dense=min(model.distances(advance_command(moved,old,old+s*(new-old),s*dt,True)).min()
                  for s in np.linspace(0.,1.,129))
        assert bound<=dense+1e-12


def test_batched_relative_guidance_matches_full_contract_finite_differences():
    state,z,path=fixture();backend=JVSGeometryBackend();obs=backend.observe(state,0.,base_z=z)
    guide=backend.compile(request_for_profile(obs,path,'base'),obs,path)
    rng=np.random.default_rng(23);eps=1e-5
    for _ in range(4):
        moved=state+rng.uniform(-.3,.3,9)
        rotation=Rotation.from_euler('xyz',[.13,-.09,moved[2]]).as_matrix()
        residual=guide.residual(backend.observe(moved,1.,base_z=z,base_rotation=rotation))
        jacobian=np.zeros((len(residual),8))
        for j in range(8):
            control=np.eye(8)[j];values=[]
            for sign in (-1,1):
                r=Rotation.from_euler('z',sign*eps*control[1]).as_matrix()@rotation
                values.append(guide.residual(backend.observe(advance(moved,control,sign*eps),1.,base_z=z,base_rotation=r)))
            jacobian[:,j]=(values[1]-values[0])/(2*eps)
        h,g=backend.quadratic_guidance(guide,moved,1.,z,rotation)
        np.testing.assert_allclose(h,jacobian.T@jacobian,atol=2e-8)
        np.testing.assert_allclose(g,jacobian.T@(2*residual),atol=2e-8)


def test_worker_serialization_omits_caches_but_preserves_exact_path():
    _,_,path=fixture()
    times=np.linspace(0.,8.,101)
    samples=[path.sample(float(t)) for t in times]
    assert path._cache and path._pose_cache
    restored=pickle.loads(pickle.dumps(path))
    assert not restored._cache and not restored._pose_cache
    assert path_hash(restored)==path_hash(path)
    for t,expected in zip(times,samples):
        for actual,value in zip(restored.sample(float(t)),expected):
            np.testing.assert_allclose(actual,value,atol=1e-12)


def test_cached_frame_measurements_do_not_alias_observer_annotations():
    from jev_agent_wbc.subject_state import frame_state
    from jev_agent_wbc.camera_intent import default_intrinsics
    state,z,path=fixture();runtime=json.loads((ROOT/'runtime_scene.json').read_text())
    p,r,_=path.original.sample(0.)
    a=frame_state(p,r,runtime['subject'],default_intrinsics())
    original=a['centroid'].copy();a['centroid'][0]=100.;a['source']='observer'
    b=frame_state(p,r,runtime['subject'],default_intrinsics())
    assert b['centroid']==original and b['source']=='geometry_projection_not_observed_mask'


def test_async_sensor_poll_never_waits_for_unfinished_acquisition():
    from jev_agent_wbc.async_sensor import AsyncRuntimeLidar
    class Pending:
        def done(self):return False
        def result(self):pytest.fail('servo waited for sensor result')
    sensor=AsyncRuntimeLidar.__new__(AsyncRuntimeLidar)
    sensor.pending=Pending();sensor.observed=dict(observation_stamp=1.)
    sensor.requested=1.;sensor.completed=0
    assert sensor.poll(np.zeros(9),0.,1.5)['observation_stamp']==1.


@pytest.mark.parametrize('stamp',[.9,1.6])
def test_async_sensor_rejects_out_of_order_or_future_capture(stamp):
    from jev_agent_wbc.async_sensor import AsyncRuntimeLidar
    class Pending:
        def done(self):return True
        def result(self):return dict(observation_stamp=stamp)
    sensor=AsyncRuntimeLidar.__new__(AsyncRuntimeLidar)
    sensor.pending=Pending();sensor.observed=dict(observation_stamp=1.)
    sensor.requested=1.;sensor.completed=0
    with pytest.raises(ValueError,match='timestamp'):
        sensor.poll(np.zeros(9),0.,1.5)
