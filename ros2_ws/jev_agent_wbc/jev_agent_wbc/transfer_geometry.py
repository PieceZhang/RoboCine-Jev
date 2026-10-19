"""Nearby, next-shot-feasible connector goals and short nonholonomic approaches."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .whole_body_qp import whole_body_kinematics,Limits
from .jvs_kinematics import JOINT_LOWER,JOINT_UPPER


def angle(value):return float(np.arctan2(np.sin(value),np.cos(value)))


def configuration_distance(start,end):
    return float(np.linalg.norm(end[:2]-start[:2])+.12*abs(angle(end[2]-start[2]))
        +.05*np.linalg.norm(end[3:]-start[3:]))


def fresh_reference(reference):
    """Keep the camera path, remove a posture seed belonging to another reset."""
    if not getattr(reference,'joint_seed',[]):return reference
    from .intent_paths import IntentPath
    geometry=dict(reference.geometry,joint_seed=[],joint_seed_preference_only=True)
    return IntentPath(reference.original,reference.previous,reference.activation_s,reference.program,
        reference.model,reference.intrinsics,geometry=geometry)


def capture_reference(reference,state,velocity,z=0.,keep_seed=False):
    """Join the same action smoothly from a nearby measured arrival pose."""
    from .intent_paths import IntentPath
    from .reference import smoothstep
    if not keep_seed:reference=fresh_reference(reference)
    p,r,J=whole_body_kinematics(state,z)
    geometry=dict(reference.geometry);times=np.asarray(geometry['times']);u=times/reference.duration
    positions=np.asarray(geometry['positions'])+(1.-smoothstep(u))[:,None]*(p-reference.sample(0.)[0])
    twist=J@velocity;old=np.asarray(geometry['start_velocity'])
    positions+=(u-6*u**3+8*u**4-3*u**5)[:,None]*reference.duration*(twist[:3]-old)
    geometry.update(positions=positions.tolist(),start_velocity=twist[:3].tolist(),
        initial_rotation_xyzw=Rotation.from_matrix(r).as_quat().tolist())
    return IntentPath(reference.original,reference.previous,0.,dict(reference.program,measured_initial_boundary=True),
        reference.model,reference.intrinsics,geometry=geometry)


def prepare_arrival(reference,state,z=0.):
    """Refit only redundant body motion for a selected nearby camera boundary.

    The caller runs this in a worker while the connector keeps controlling the
    robot. The camera curve is fixed, and execution still rechecks arrival.
    """
    from .kinematic_feasibility import KinematicSearch,summary
    from .joint_curve_optimizer import refine
    from .intent_paths import IntentPath
    seed=getattr(reference,'joint_seed',[])
    keep=bool(seed and np.linalg.norm(np.asarray(seed[0]['state'])-state)<.02)
    candidate=capture_reference(reference,state,np.zeros(8),z,keep_seed=keep)
    search=KinematicSearch(state,np.zeros(8),0.,z,Limits())
    proof=search.lift(candidate,candidate.duration);refinement=None
    if not proof['feasible'] or proof.get('min_joint_margin_rad',0.)<.10:
        candidate=IntentPath(candidate.original,candidate.previous,0.,
            dict(candidate.program,fixed_camera_path=True,fixed_spatial_shape=True,joint_reserve_rad=.12),
            candidate.model,candidate.intrinsics,geometry=candidate.geometry)
        fitted,refinement=refine(candidate,search,seconds=2.,max_iterations=30,knots=49)
        if fitted is not None:
            fitted_proof=search.lift(fitted,fitted.duration)
            if fitted_proof['feasible']:candidate,proof=fitted,fitted_proof
    return candidate,dict(kinematics=summary(proof),refinement=refinement,
        scope='asynchronous whole-body allocation for fixed camera path; arrival and current-observation lease must be rechecked')


def nearby_goal(state,authored,reference,scene,z=0.):
    """Keep the next camera pose, choose a close redundant body configuration.

    Full next-shot FK/actuator lifting and current occupancy are checked. This
    is not a prediction of future dynamic safety; execution stays certified.
    """
    from .kinematic_feasibility import KinematicSearch,summary
    from .dynamic_collision import DynamicCollision
    from .self_collision import SelfCollision
    reference=fresh_reference(reference)
    x=np.asarray(state,float);nominal=np.asarray(authored,float).copy()
    nominal[2]=x[2]+angle(nominal[2]-x[2])
    target_p,target_r,_=reference.sample(0.)
    lower=np.r_[x[:2]-1.,x[2]-np.pi,JOINT_LOWER+.035]
    upper=np.r_[x[:2]+1.,x[2]+np.pi,JOINT_UPPER-.035]
    metric=np.r_[1.4,1.4,.22,np.full(6,.09)]
    forward=np.array([np.cos(x[2]),np.sin(x[2])]);lateral=np.array([-forward[1],forward[0]])
    def residual(q):
        p,r,_=whole_body_kinematics(q,z)
        displacement=q-x;regularization=metric*displacement
        # A centimetre of sideways base motion can require two large turns.
        # Prefer arm compensation or longitudinal travel for such tiny shifts.
        regularization[:2]=[1.4*(displacement[:2]@forward),8.*(displacement[:2]@lateral)]
        joint_margin=np.minimum(q[3:]-JOINT_LOWER,JOINT_UPPER-q[3:])
        reserve=.8*np.maximum(0.,.16-joint_margin)
        return np.r_[300*(p-target_p),100*Rotation.from_matrix(target_r@r.T).as_rotvec(),regularization,reserve]
    seeds=[x,nominal,.5*(x+nominal)]
    for yaw in (-.6,.6):
        seed=x.copy();seed[2]+=yaw;seeds.append(seed)
    for offset in ([.3,0.],[-.3,0.],[0.,.3],[0.,-.3]):
        seed=nominal.copy();seed[:2]+=offset;seeds.append(seed)
    candidates=[nominal]
    for seed in seeds:
        fit=least_squares(residual,np.clip(seed,lower+1e-6,upper-1e-6),bounds=(lower,upper),max_nfev=100,
            ftol=1e-7,xtol=1e-7,gtol=1e-7)
        q=fit.x;p,r,_=whole_body_kinematics(q,z)
        if np.linalg.norm(p-target_p)<.003 and Rotation.from_matrix(target_r@r.T).magnitude()<.008:
            if all(np.linalg.norm(q-other)>.03 for other in candidates):candidates.append(q)
    collision=DynamicCollision(scene);self_collision=SelfCollision();attempts=[];thin=[]
    for goal in sorted(candidates,key=lambda q:configuration_distance(x,q)):
        distance=configuration_distance(x,goal)
        clearance=float(collision.clearance(goal,z).min());self_clear=float(self_collision.terms(goal)[0].min())
        if clearance<.04 or self_clear<.008:
            attempts.append(dict(distance=distance,clearance_m=clearance,reason='current_occupancy_or_self_clearance'));continue
        proof=KinematicSearch(goal,np.zeros(8),0.,z,Limits()).lift(reference,reference.duration)
        prepared=reference;body_report=None
        if hasattr(reference,'geometry') and (not proof['feasible'] or proof.get('min_joint_margin_rad',0.)<.08):
            prepared,body_report=prepare_arrival(reference,goal,z)
            proof=KinematicSearch(goal,np.zeros(8),0.,z,Limits()).lift(prepared,prepared.duration)
        lease=None
        if proof['feasible']:
            lease=KinematicSearch(goal,np.zeros(8),0.,z,Limits(),scene=dict(scene,observation_stamp=0.)).lift(prepared,min(2.,prepared.duration))
        attempts.append(dict(distance=distance,clearance_m=clearance,reason=proof['reason'],feasible=bool(proof['feasible']),
            min_joint_margin_rad=proof.get('min_joint_margin_rad'),observed_lease_feasible=bool(lease and lease['feasible'])))
        if proof['feasible'] and lease['feasible']:
            record=dict(selected='nearby_verified' if np.linalg.norm(goal-nominal)>.01 else 'authored',
                configuration_distance=distance,authored_configuration_distance=configuration_distance(x,nominal),
                goal_state=goal.tolist(),next_shot_kinematics=summary(proof),attempts=attempts,
                scope='next-shot sampled FK/actuators and current occupancy only; future collision unverified')
            if body_report is not None:record.update(prepared_geometry=prepared.geometry,prepared_program=prepared.program,body_allocation=body_report)
            if proof.get('min_joint_margin_rad',0.)>=.08:return goal,record
            thin.append((goal,record))
    if thin:return min(thin,key=lambda item:item[1]['configuration_distance'])
    return nominal,dict(selected='authored_fallback',goal_state=nominal.tolist(),attempts=attempts,
        configuration_distance=configuration_distance(x,nominal),authored_configuration_distance=configuration_distance(x,nominal),
        scope='no closer verified goal found; arrival still requires complete next-shot verification')


def approach_sign(state,goal,point):
    """Compare both complementary drive directions including final heading."""
    direction=float(np.arctan2(point[1]-state[1],point[0]-state[0]))
    def cost(sign):
        heading=direction+(np.pi if sign<0 else 0.)
        return abs(angle(heading-state[2]))+abs(angle(goal[2]-heading))
    return -1. if cost(-1.)+.08<cost(1.) else 1.
