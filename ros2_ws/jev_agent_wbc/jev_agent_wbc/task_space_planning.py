"""Camera intent planning with a state-seeded numerical kinematic gate.

The local FK/Jacobian polytope seeds search; continuous constrained differential
IK screens complete timed candidates. WBC execution and collision safety remain
independent. Jev never computes or relaxes these numerical constraints.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .candidates import SearchBudget, path_hash, support_violation
from .contracts import digest
from .floating_camera import evaluate_path, camera_candidates, intent_prediction
from .camera_intent import IntentSpec, progress_at, POLICY
from .whole_body_qp import whole_body_kinematics
from .kinematic_feasibility import KinematicSearch, summary

RADIUS = .06
CAPS = dict(radial_m=1.5,lateral_m=1.5,height_m=.65,yaw_rad=np.pi,pitch_rad=np.pi/2,roll_rad=np.pi/2)


class WitnessCameraPrediction:
    """Predicted FK camera motion, including numerical realization residuals."""
    def __init__(self,path,report):
        self.path=path;self.original=getattr(path,'original',path);self.duration=path.duration
        self.times=np.array([row['t'] for row in report['trace']])
        self.positions=np.array([row['position'] for row in report['trace']])
        self.rotations=Slerp(self.times,Rotation.from_quat([row['quaternion_xyzw'] for row in report['trace']]))

    def progress_time(self,t):
        return progress_at(self.path,t)

    def sample(self,t):
        at=np.clip(t,self.times[0],self.times[-1])
        return (np.array([np.interp(at,self.times,self.positions[:,j]) for j in range(3)]),
                self.rotations(at).as_matrix(),np.zeros(6))


class MeasuredCameraPrediction:
    def __init__(self,path,state,base_z,t,lead,base_rotation=None):
        self.path=path;self.original=getattr(path,'original',path)
        activation=getattr(path,'activation_s',None)
        if activation is not None and activation>t:
            lead=min(lead,activation-t)
        if not np.isfinite(lead) or lead<=0:
            raise ValueError('camera convergence lead must be positive')
        self.t=t;self.lead=lead
        self.duration=path.duration
        p,r,_=whole_body_kinematics(state,base_z,base_rotation)
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
    rates=np.where(tracks>0,float(scene.get('motion_uncertainty_rate_m_s',.035)),0.)
    radii=radii+rates*ages
    return dict(valid=bool(len(centers)) and -1e-8<=t-stamp<=.6,observation_stamp=stamp,
        centers=centers,radius_rates=rates,velocities=np.asarray(scene.get('surfel_velocities',np.zeros_like(centers))),radii=radii,
        subject_model=scene.get('subject_model'),intrinsics=scene.get('intrinsics'))


def prediction(path,scene,state,base_z,t,end,lead,visual=False,base_rotation=None):
    virtual=MeasuredCameraPrediction(path,state,base_z,t,lead,base_rotation)
    value,risk=evaluate_path(virtual,camera_scene(scene,t,end),t,end,RADIUS,visual=visual)
    value.reason='measured_seed_ideal_camera_sweep_not_whole_body_prediction'
    return value,risk


def diagnose(original,scene,state,previous,t,base_z,limits,commanded_q=None,active=None,budget=None,history=(),
             measured_control=None,base_rotation=None,accepted_intent=None):
    budget=budget or SearchBudget()
    active=original if active is None else active
    observed=camera_scene(scene,t,active.duration)
    spec=IntentSpec(original,observed.get('subject_model'),observed.get('intrinsics'))
    value,risk=prediction(active,scene,state,base_z,t,active.duration,budget.activation_lead_s,base_rotation=base_rotation)
    virtual=MeasuredCameraPrediction(active,state,base_z,t,budget.activation_lead_s,base_rotation)
    search=KinematicSearch(state,previous,t,base_z,limits,measured_control=measured_control,
        commanded_q=commanded_q,base_rotation=base_rotation)
    from .search_configuration import capability
    robot_capability=capability(search,original.subject)
    kinematics=search.lift(virtual,active.duration+POLICY.terminal_hold_s)
    metrics=intent_prediction(spec,virtual,observed,t,history,step=.1)
    issues=[]
    if risk is not None or not value.feasible:issues.append('predicted_camera_collision_or_low_clearance')
    accepted_degradation=False
    if not metrics['intent_complete']:
        accepted_degradation=bool(accepted_intent and
            metrics['effective_coverage']>=accepted_intent['effective_coverage']-.03 and
            metrics['longest_occlusion_s']<=accepted_intent['longest_occlusion_s']+.1 and
            metrics['motion_trend'].get('companion_travel_m',float('inf'))<=
                accepted_intent['motion_trend'].get('companion_travel_m',0.)+.03 and
            metrics['terminal_frame'].get('centroid_error',float('inf'))<=
                accepted_intent['terminal_frame'].get('centroid_error',0.)+.015 and
            metrics['terminal_frame'].get('log_area_error',float('inf'))<=
                accepted_intent['terminal_frame'].get('log_area_error',0.)+.04)
        if not accepted_degradation:issues.append('incomplete_visible_intent')
    if not kinematics['feasible']:issues.append('numerical_kinematic_gate_failed')
    return dict(snapshot_time_s=t,physical_caps=CAPS,kinematic_prior=search.prior(),robot_capability=robot_capability,
        kinematics=summary(kinematics),report=dict(issues=issues,first_risk_s=risk,
        min_clearance_m=value.min_clearance_m,joint_margin_rad=None,intent=metrics,
        accepted_intent_degradation_unchanged=accepted_degradation,
        local_repair_evidence='camera intent programs; numerical FK/actuator screening; no WBC rollout'))


def screen(path,search,spec,observed,budget,history):
    virtual=MeasuredCameraPrediction(path,search.state,search.base_z,search.t,budget.activation_lead_s,
        None if search.tilt is None else Rotation.from_euler('z',search.state[2]).as_matrix()@search.tilt)
    kinematics=search.lift(virtual,path.duration+POLICY.terminal_hold_s)
    result=dict(kinematics=kinematics,summary=summary(kinematics),error='')
    if not kinematics['feasible']:
        result['error']='kinematic_gate:'+kinematics['reason']
        return result
    realized=WitnessCameraPrediction(path,kinematics)
    value,_=evaluate_path(realized,observed,search.t,path.duration+POLICY.terminal_hold_s,RADIUS,visual=True)
    metrics=intent_prediction(spec,realized,observed,search.t,history)
    value.reason='camera_sweep_and_sampled_kinematic_witness'
    value.joint_margin_rad=kinematics['min_joint_margin_rad']
    value.max_velocity_ratio=kinematics['max_velocity_ratio']
    value.max_acceleration_ratio=kinematics['max_acceleration_ratio']
    value.max_tracking_error_m=kinematics['max_position_error_m']
    value.max_orientation_error_rad=kinematics['max_orientation_error_rad']
    if kinematics.get('whole_body_collision_checked'):
        value.min_clearance_m=min(value.min_clearance_m,kinematics['min_body_clearance_m'])
        value.reason='camera_and_whole_body_swept_kinematic_witness'
    if not value.feasible:
        result['error']='measured_camera_seed_collision'
    elif not metrics['camera_limits_pass']:
        result['error']='measured_camera_seed_exceeds_motion_limits'
    result.update(value=value,metrics=metrics)
    return result


def build(original,active,strategy,scene,state,previous,t,base_z,limits,
          evidence=None,intrinsics=None,budget=None,commanded_q=None,history=(),
          measured_control=None,base_rotation=None,search_config=None):
    budget=budget or SearchBudget();start=time.monotonic()
    observed=camera_scene(scene,t,active.duration)
    observed['intrinsics']=intrinsics or observed.get('intrinsics')
    search=KinematicSearch(state,previous,t,base_z,limits,measured_control=measured_control,
        commanded_q=commanded_q,base_rotation=base_rotation,scene=scene,search_config=search_config)
    spec=IntentSpec(original,observed.get('subject_model'),observed.get('intrinsics'))
    def project(path):
        from .fk_camera_path import FKCameraPath
        virtual=MeasuredCameraPrediction(path,search.state,search.base_z,search.t,budget.activation_lead_s,base_rotation)
        report=None
        for allocation in search.config['allocations'][:2]:
            report=search._lift(virtual,path.duration+POLICY.terminal_hold_s,allocation,project_camera=True)
            if report['feasible']:
                return dict(path=FKCameraPath.from_lift(path,report),summary=summary(report))
        return dict(error='fk_projection_unverified:'+report['reason'],summary=summary(report))
    # Filter every proposal before shortlist truncation, so an infeasible top
    # camera candidate cannot hide a feasible lower-ranked robot candidate.
    result=camera_candidates(original,active,strategy,observed,t,budget,RADIUS,history,
        allow_time_extension=False,kinematic_search=search,
        numerical_gate=lambda path:screen(path,search,spec,observed,budget,history),search_config=search_config,
        proposal_transform=project if search_config=='executable' else None)
    result.update(wall_s=time.monotonic()-start,snapshot_time_s=t,kinematic_prior=search.prior(),search_config=search_config)
    return result


def verify(candidate,approval,original,scene,state,previous,t,base_z,limits,
           evidence=None,intrinsics=None,budget=None,commanded_q=None,history=(),
           measured_control=None,base_rotation=None):
    budget=budget or SearchBudget();start=time.monotonic()
    checks=dict(physical_valid='unknown',target_authorized='fail',predicted_visual_valid='unknown',
        state_fresh='fail',observed_visual_valid='unknown',whole_body_prediction='not_performed',
        kinematic_feasibility='unknown')
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
    observed=camera_scene(scene,t,candidate.path.duration)
    spec=IntentSpec(original,observed.get('subject_model'),intrinsics or observed.get('intrinsics'))
    search=KinematicSearch(state,previous,t,base_z,limits,measured_control=measured_control,
        commanded_q=commanded_q,base_rotation=base_rotation,scene=scene,
        search_config=candidate.consequences.get('search_config'))
    screened=screen(candidate.path,search,spec,observed,budget,history)
    checks['kinematic_feasibility']='pass' if screened['kinematics']['feasible'] else 'fail'
    if screened.get('error'):
        return dict(error=screened['error'],checks=checks,kinematics=screened['kinematics'])
    full,metrics=screened['value'],screened['metrics']
    checks['physical_valid']='pass' if full.feasible else 'fail'
    checks['predicted_visual_valid']='pass' if metrics['intent_complete'] and metrics['camera_limits_pass'] else 'fail'
    checks['whole_body_prediction']='observed_geometry_swept_kinematic_lift'
    return dict(error='' if full.feasible else 'latest_camera_collision_rejected',
        candidate=candidate,approval=approval,checks=checks,visual_checks=dict(intent=checks['predicted_visual_valid']),
        validation=full,snapshot_time_s=t,scene=scene,wall_s=time.monotonic()-start,
        prediction_model='floating_camera',base_z=base_z,intent=metrics,
        kinematics=screened['kinematics'],kinematic_limits=limits)
