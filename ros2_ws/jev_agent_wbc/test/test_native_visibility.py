import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from jev_agent_wbc import camera_intent,native_visibility
from jev_agent_wbc.reference import look_at


def equivalent(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equivalent(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equivalent(x,y)
    elif isinstance(a,(float,np.floating)):
        assert a==pytest.approx(b,abs=1e-11)
    else:assert a==b


def test_compiled_silhouette_rays_match_reference_under_full_rigid_transforms():
    rng=np.random.default_rng(27)
    for _ in range(120):
        world=Rotation.random(random_state=rng);R=world.as_matrix();offset=rng.normal(size=3)
        position=np.array([-rng.uniform(.7,3.),rng.uniform(-.5,.5),rng.uniform(-.3,.3)])
        rotation=look_at(position,np.zeros(3))
        centers=rng.uniform([-.9,-.4,-.5],[.1,.4,.5],(12,3))
        model=dict(position=offset.tolist(),radius=.2,height=.5,orientation_xyzw=world.as_quat().tolist())
        scene=dict(valid=True,observation_stamp=0.,centers=centers@R.T+offset,
            velocities=rng.normal(size=(12,3))*.03,radii=rng.uniform(.03,.2,12),radius_rates=np.full(12,.01))
        p=R@position+offset;r=R@rotation;intrinsics=[.7,.9,.5,.5]
        equivalent(native_visibility.visibility(p,r,scene,.3,model,intrinsics),
                   camera_intent.visibility(p,r,scene,.3,model,intrinsics))


@pytest.mark.parametrize('case',['orbit','movein'])
@pytest.mark.parametrize('family',['framing','view','joint'])
def test_complete_intent_prediction_matches_existing_score(case,family):
    from jev_agent_wbc.representation import Observation,Pose,Box
    from jev_agent_wbc.multi_robot_scenarios import bind_scenario
    from jev_agent_wbc.multi_robot_planning import candidate_paths
    from jev_agent_wbc.floating_camera import intent_prediction
    obs=Observation(3.,Pose((0.,0.,0.)),Pose((.3,0.,1.)),dict(mobile_platform=(Box((0.,0.,0.),(.3,.2,.2)),),
        camera_manipulator=(Box((.3,0.,.7),(.1,.1,.3)),)))
    original,authored,_=bind_scenario(case,obs)
    model=authored['subject'];spec=camera_intent.IntentSpec(original,model)
    scene=dict(valid=True,observation_stamp=-.1,centers=np.array([[.9,.5,1.],[1.3,.2,.8]]),
        velocities=np.array([[0.,-.2,0.],[0.,0.,0.]]),radii=np.array([.1,.12]),radius_rates=np.array([.035,0.]))
    for path in candidate_paths(original,family,'timing',.9):
        equivalent(native_visibility.intent_prediction(spec,path,scene,0.,step=.1),
                   intent_prediction(spec,path,scene,0.,step=.1))
