"""Floating-camera counterfactuals seeded by real robot FK and observed surfels.

Only current FK is used, never a WBC rollout. The virtual camera converges from
its measured pose to the target over the activation lead. This is an explicit
ideal tracking assumption, not a prediction/certificate of whole-body execution.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .candidates import Candidate, SearchBudget, path_hash, support_violation
from .contracts import digest
from .floating_camera import evaluate_path, camera_candidates, intent_prediction
from .camera_intent import IntentSpec, progress_at
from .whole_body_qp import whole_body_kinematics

RADIUS = .06
CAPS = dict(radial_m=1.5,lateral_m=1.5,height_m=.65,yaw_rad=np.pi,pitch_rad=np.pi/2,roll_rad=np.pi/2)


class MeasuredCameraPrediction:
    def __init__(self,path,state,base_z,t,lead):
        self.path=path;self.original=getattr(path,'original',path)
        self.t=t;self.lead=lead
        self.duration=path.duration
        p,r,_=whole_body_kinematics(state,base_z)
        tp,tr,_=path.sample(t)
        self.offset=p-tp
        self.rotation_error=Rotation.from_matrix(r@tr.T).as_rotvec()

    def progress_time(self,t):
        return progress_at(self.path,t)

    def sample(self,t):
        p,r,v=self.path.sample(t)
        u=np.clip((t-self.t)/self.lead,0.,1.)
        residual=1.-(10*u**3-15*u**4+6*u**5)
        return p+residual*self.offset,Rotation.from_rotvec(residual*self.rotation_error).as_matrix()@r,v


def camera_scene(scene,t,end):
    """Same observed surfels as WBC, with the same time-growing uncertainty as WBC."""
    centers=np.asarray(scene.get('surfel_centers',[]),float).reshape(-1,3)
    radii=np.asarray(scene.get('surfel_radii',np.full(len(centers),scene.get('surfel_radius',.025))),float)
    tracks=np.asarray(scene.get('surfel_track_ids',np.zeros(len(centers))))
    ages=np.asarray(scene.get('surfel_ages',np.zeros(len(centers))))
    stamp=float(scene['observation_stamp'])
    rates=np.where(tracks>0,.035,0.)
    radii=radii+rates*ages
    return dict(valid=bool(len(centers)) and -1e-8<=t-stamp<=.6,observation_stamp=stamp,
        centers=centers,radius_rates=rates,velocities=np.asarray(scene.get('surfel_velocities',np.zeros_like(centers))),radii=radii,
        subject_model=scene.get('subject_model'),intrinsics=scene.get('intrinsics'))


def prediction(path,scene,state,base_z,t,end,lead,visual=False):
    virtual=MeasuredCameraPrediction(path,state,base_z,t,lead)
    value,risk=evaluate_path(virtual,camera_scene(scene,t,end),t,end,RADIUS,visual=visual)
    value.reason='measured_seed_ideal_camera_sweep_not_whole_body_prediction'
    return value,risk


def diagnose(original,scene,state,previous,t,base_z,limits,commanded_q=None,active=None,budget=None,history=()):
    budget=budget or SearchBudget()
    active=original if active is None else active
    observed=camera_scene(scene,t,active.duration)
    spec=IntentSpec(original,observed.get('subject_model'),observed.get('intrinsics'))
    value,risk=prediction(active,scene,state,base_z,t,active.duration,budget.activation_lead_s)
    virtual=MeasuredCameraPrediction(active,state,base_z,t,budget.activation_lead_s)
    metrics=intent_prediction(spec,virtual,observed,t,history,step=.1)
    issues=[]
    if risk is not None or not value.feasible:issues.append('predicted_camera_collision_or_low_clearance')
    if not metrics['intent_complete']:issues.append('incomplete_visible_intent')
    return dict(snapshot_time_s=t,physical_caps=CAPS,report=dict(issues=issues,first_risk_s=risk,
        min_clearance_m=value.min_clearance_m,joint_margin_rad=None,intent=metrics,
        local_repair_evidence='camera intent programs; measured pose seed; no WBC rollout'))


def build(original,active,strategy,scene,state,previous,t,base_z,limits,
          evidence=None,intrinsics=None,budget=None,commanded_q=None,history=()):
    budget=budget or SearchBudget();start=time.monotonic()
    observed=camera_scene(scene,t,active.duration)
    observed['intrinsics']=intrinsics or observed.get('intrinsics')
    # This legacy robot adapter retains its execution clock. The independent
    # FloatingAgent owns variable-duration shots; changing downstream robot
    # execution timing is deliberately outside this planner-only change.
    result=camera_candidates(original,active,strategy,observed,t,budget,RADIUS,history,
                             allow_time_extension=False)
    if result.get('error'):return result
    spec=IntentSpec(original,observed.get('subject_model'),observed.get('intrinsics'))
    candidates=[]
    for candidate in result['candidates']:
        value,_=prediction(candidate.path,scene,state,base_z,t,candidate.path.duration,budget.activation_lead_s,True)
        virtual=MeasuredCameraPrediction(candidate.path,state,base_z,t,budget.activation_lead_s)
        metrics=intent_prediction(spec,virtual,observed,t,history)
        if value.feasible and metrics['intent_complete'] and metrics['camera_limits_pass']:
            consequences=dict(intent=metrics,summary=metrics,source='measured_seed_camera_intent_no_wbc',
                prediction_interval_s=[t,candidate.path.duration])
            candidates.append(Candidate(candidate.path,value,consequences,t,strategy))
    result.update(candidates=candidates,wall_s=time.monotonic()-start,snapshot_time_s=t,
        error='' if candidates else 'measured_camera_seed_cannot_complete_intent')
    return result


def verify(candidate,approval,original,scene,state,previous,t,base_z,limits,
           evidence=None,intrinsics=None,budget=None,commanded_q=None,history=()):
    budget=budget or SearchBudget();start=time.monotonic()
    checks=dict(physical_valid='unknown',target_authorized='fail',predicted_visual_valid='unknown',
        state_fresh='fail',observed_visual_valid='unknown',whole_body_prediction='not_performed')
    if (candidate.id!=approval.candidate_id or path_hash(candidate.path)!=approval.candidate_id or
        digest(candidate.card())!=approval.card_sha256 or original.sha256!=approval.reference_sha256):
        return dict(error='approval_binding_mismatch',checks=checks)
    checks['target_authorized']='pass'
    error=support_violation(approval,evidence)
    if error:return dict(error=error,checks=checks)
    if not 0<=t-candidate.snapshot_time_s<=budget.max_prediction_age_s or t>=candidate.path.activation_s:
        return dict(error='candidate_expired_requires_new_program',checks=checks)
    if not -1e-8<=t-float(scene['observation_stamp'])<=.6:return dict(error='scene_stale',checks=checks)
    checks['state_fresh']='pass'
    full,_=prediction(candidate.path,scene,state,base_z,t,candidate.path.duration,budget.activation_lead_s,True)
    observed=camera_scene(scene,t,candidate.path.duration)
    spec=IntentSpec(original,observed.get('subject_model'),intrinsics or observed.get('intrinsics'))
    virtual=MeasuredCameraPrediction(candidate.path,state,base_z,t,budget.activation_lead_s)
    metrics=intent_prediction(spec,virtual,observed,t,history)
    checks['physical_valid']='pass' if full.feasible else 'fail'
    checks['predicted_visual_valid']='pass' if metrics['intent_complete'] and metrics['camera_limits_pass'] else 'fail'
    return dict(error='' if full.feasible and checks['predicted_visual_valid']=='pass' else 'latest_camera_intent_rejected',
        candidate=candidate,approval=approval,checks=checks,visual_checks=dict(intent=checks['predicted_visual_valid']),
        validation=full,snapshot_time_s=t,scene=scene,wall_s=time.monotonic()-start,
        prediction_model='floating_camera',base_z=base_z,intent=metrics)
