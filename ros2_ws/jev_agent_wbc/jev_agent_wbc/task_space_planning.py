"""Floating-camera counterfactuals seeded by real robot FK and observed surfels.

Only current FK is used, never a WBC rollout. The virtual camera converges from
its measured pose to the target over the activation lead. This is an explicit
ideal tracking assumption, not a prediction/certificate of whole-body execution.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .candidates import (Candidate, SearchBudget, propose_paths, path_hash,
                         contract_comparison, support_violation)
from .contracts import digest
from .floating_camera import evaluate_path
from .perception import visual_consequences
from .whole_body_qp import whole_body_kinematics

RADIUS = .06
CAPS = dict(radial_m=1.5,lateral_m=1.5,height_m=.65,yaw_rad=np.pi,pitch_rad=np.pi/2,roll_rad=np.pi/2)


class MeasuredCameraPrediction:
    def __init__(self,path,state,base_z,t,lead):
        self.path=path;self.original=getattr(path,'original',path)
        self.t=t;self.lead=lead
        p,r,_=whole_body_kinematics(state,base_z)
        tp,tr,_=path.sample(t)
        self.offset=p-tp
        self.rotation_error=Rotation.from_matrix(r@tr.T).as_rotvec()

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
        centers=centers,radius_rates=rates,velocities=np.asarray(scene.get('surfel_velocities',np.zeros_like(centers))),radii=radii)


def prediction(path,scene,state,base_z,t,end,lead,visual=False):
    virtual=MeasuredCameraPrediction(path,state,base_z,t,lead)
    value,risk=evaluate_path(virtual,camera_scene(scene,t,end),t,end,RADIUS,visual=visual)
    value.reason='measured_seed_ideal_camera_sweep_not_whole_body_prediction'
    return value,risk


def diagnose(original,scene,state,previous,t,base_z,limits,commanded_q=None,active=None,budget=None):
    budget=budget or SearchBudget()
    active=original if active is None else active
    value,risk=prediction(active,scene,state,base_z,t,min(t+5.,original.duration),budget.activation_lead_s)
    p,r,_=whole_body_kinematics(state,base_z)
    tracking=float(np.linalg.norm(p-active.sample(t)[0]))
    issues=[]
    if risk is not None or not value.feasible:issues.append('predicted_camera_collision_or_low_clearance')
    if tracking>.05:issues.append('actual_camera_tracking_residual')
    return dict(snapshot_time_s=t,physical_caps=CAPS,report=dict(issues=issues,first_risk_s=risk,
        min_clearance_m=value.min_clearance_m,joint_margin_rad=None,
        target_tracking_residual_m=tracking,actual_camera_position=p.tolist(),
        local_repair_evidence='floating task-space translations; measured FK seed; no WBC rollout'))


def build(original,active,strategy,scene,state,previous,t,base_z,limits,
          evidence=None,intrinsics=None,budget=None,commanded_q=None):
    budget=budget or SearchBudget();start=time.monotonic()
    end=min(original.duration,t+budget.max_horizon_s)
    if end-t-budget.activation_lead_s<.6:return dict(error='insufficient_time_for_exact_plan_activation')
    ranked=[];evaluations=[]
    for path in propose_paths(original,active,strategy,t,budget,CAPS)[:budget.max_proposals]:
        value,_=prediction(path,scene,state,base_z,t,end,budget.activation_lead_s)
        evaluations.append(dict(id=path_hash(path),coarse=value.to_dict()))
        if value.feasible:ranked.append((value.cost,path))
    candidates=[]
    for _,path in sorted(ranked,key=lambda pair:pair[0])[:budget.max_rollouts]:
        value,_=prediction(path,scene,state,base_z,t,end,budget.activation_lead_s,True)
        consequences=visual_consequences(original,value.camera_trace,evidence,intrinsics)
        consequences['source']='measured_seed_ideal_camera_no_wbc_prediction'
        candidates.append(Candidate(path,value,consequences,t,strategy))
        detail=value.to_dict();detail.pop('camera_trace',None)
        evaluations.append(dict(id=path_hash(path),full=detail))
    return dict(candidates=candidates,evaluations=evaluations,snapshot_time_s=t,wall_s=time.monotonic()-start,
        error='' if candidates else 'no_camera_candidate_passed_collision_screening')


def verify(candidate,approval,original,scene,state,previous,t,base_z,limits,
           evidence=None,intrinsics=None,budget=None,commanded_q=None):
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
        return dict(error='candidate_expired_no_retiming_allowed',checks=checks)
    if not -1e-8<=t-float(scene['observation_stamp'])<=.6:return dict(error='scene_stale',checks=checks)
    checks['state_fresh']='pass'
    full,_=prediction(candidate.path,scene,state,base_z,t,candidate.path.envelope.end_s,budget.activation_lead_s,True)
    consequences=visual_consequences(original,full.camera_trace,evidence,intrinsics)
    visual=contract_comparison(candidate.consequences,consequences,budget)
    checks['physical_valid']='pass' if full.feasible else 'fail' # camera sphere only
    checks['predicted_visual_valid']='fail' if 'fail' in visual.values() else 'unknown' if 'unknown' in visual.values() else 'pass'
    return dict(error='' if full.feasible and checks['predicted_visual_valid']!='fail' else 'latest_camera_prediction_rejected',
        candidate=candidate,approval=approval,checks=checks,visual_checks=visual,validation=full,
        snapshot_time_s=t,scene=scene,wall_s=time.monotonic()-start,prediction_model='floating_camera',base_z=base_z)
