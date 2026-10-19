"""Camera intent planning with a state-seeded numerical kinematic gate.

The local FK/Jacobian polytope seeds search; continuous constrained differential
IK screens complete timed candidates. WBC execution and collision safety remain
independent. Jev never computes or relaxes these numerical constraints.
"""
import time
from dataclasses import asdict, replace
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


def configure_combo_samples(search, original):
    """Long combos retain the same check spacing instead of hitting the 8 s cap."""
    from .combo import is_combo
    if is_combo(original):
        required=int(np.ceil((original.duration+POLICY.terminal_hold_s-search.t)/search.policy.step_s))+2
        search.policy=replace(search.policy,max_samples=max(search.policy.max_samples,required))


def verification_snapshot(scene,state,previous,t,base_z,limits,measured_control,commanded_q,base_rotation):
    return digest(dict(scene=scene,state=np.asarray(state),previous=np.asarray(previous),time_s=t,
        base_z=base_z,limits=asdict(limits),measured_control=measured_control,
        commanded_q=commanded_q,base_rotation=base_rotation))


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

    def positions_at(self,times):
        return np.column_stack([np.interp(times,self.times,self.positions[:,j]) for j in range(3)])

    def poses_at(self,times):
        return self.positions_at(times),self.rotations(np.clip(times,self.times[0],self.times[-1])).as_matrix()


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

    def pose(self,t):
        from .intent_paths import pose_at
        p,r=pose_at(self.path,t)
        u=np.clip((t-self.t)/self.lead,0.,1.)
        residual=1.-(10*u**3-15*u**4+6*u**5)
        return p+residual*self.offset,Rotation.from_rotvec(residual*self.rotation_error).as_matrix()@r

    def sample(self,t):
        p,r=self.pose(t)
        return p,r,self.path.sample(t)[2]

    def positions_at(self,times):
        from .intent_paths import positions_at
        from .reference import smoothstep
        return positions_at(self.path,times)+(1.-smoothstep((times-self.t)/self.lead))[:,None]*self.offset

    def poses_at(self,times):
        from .intent_paths import poses_at
        from .reference import smoothstep
        p,r=poses_at(self.path,times)
        residual=1.-smoothstep((times-self.t)/self.lead)
        return p+residual[:,None]*self.offset,Rotation.from_rotvec(residual[:,None]*self.rotation_error).as_matrix()@r


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
             measured_control=None,base_rotation=None,accepted_intent=None,encode_journal=False):
    budget=budget or SearchBudget()
    active=original if active is None else active
    observed=camera_scene(scene,t,active.duration)
    spec=IntentSpec(original,observed.get('subject_model'),observed.get('intrinsics'))
    value,risk=prediction(active,scene,state,base_z,t,active.duration,budget.activation_lead_s,base_rotation=base_rotation)
    virtual=MeasuredCameraPrediction(active,state,base_z,t,budget.activation_lead_s,base_rotation)
    search=KinematicSearch(state,previous,t,base_z,limits,measured_control=measured_control,
        commanded_q=commanded_q,base_rotation=base_rotation)
    if budget.single_candidate:search.policy=replace(search.policy,step_s=.1)
    configure_combo_samples(search,original)
    from .search_configuration import capability
    robot_capability=capability(search,original.subject)
    # The installed curve already has a complete, approved FK witness. Online
    # diagnosis repairs the measured-state neighbourhood, while camera safety
    # and shot intent below still inspect the entire remaining curve. Re-solving
    # the full redundant trajectory here delayed confirmation of short-lived
    # servo disagreement beyond its deadline.
    from .combo import is_combo
    local=bool(is_combo(original) or (budget.single_candidate and active is not original and accepted_intent))
    end=min(active.duration+POLICY.terminal_hold_s,t+max(2.,budget.activation_lead_s+.8)) if local else active.duration+POLICY.terminal_hold_s
    kinematics=search.lift(virtual,end)
    metrics=intent_prediction(spec,virtual,observed,t,history,step=.1)
    issues=[]
    # The 105 mm warning starts proactive search on the original shot. A
    # verified repair is allowed to use that reserve; otherwise every accepted
    # 25+ mm clearance detour immediately requests yet another detour.
    if not value.feasible or (risk is not None and active is original):
        issues.append('predicted_camera_collision_or_low_clearance')
    accepted_degradation=False
    actual_tracking_error=float(np.linalg.norm(search.p-active.sample(t)[0]))
    if active is not original and actual_tracking_error>.015:
        issues.append('measured_camera_tracking_error')
    if not metrics['intent_complete']:
        # Missing historical coverage cannot be recovered by bending a second
        # curve onto the accepted one. Preserve that geometry while its future
        # safety, terminal frame and measured tracking remain intact.
        accepted_degradation=bool(active is not original and accepted_intent and
            metrics['terminal_frame']['pass_check'] and actual_tracking_error<=.015 and
            metrics['longest_occlusion_s']<=accepted_intent.get('longest_occlusion_s',0.)+.25)
        if not accepted_degradation:issues.append('incomplete_visible_intent')
        if active is not original and not metrics['terminal_frame']['pass_check']:
            issues.append('intent_recovery_required')
    if not kinematics['feasible']:
        imminent=kinematics.get('first_failure_s',t)<=t+1.
        if active is original or actual_tracking_error>.015 or imminent:
            issues.append('numerical_kinematic_gate_failed')
    result=dict(snapshot_time_s=t,physical_caps=CAPS,kinematic_prior=search.prior(),robot_capability=robot_capability,
        kinematics=summary(kinematics),kinematic_horizon_s=end-t,
        kinematic_scope='measured_state_rolling_confirmation' if local else 'complete_remaining_curve',
        report=dict(issues=issues,first_risk_s=risk,
        min_clearance_m=value.min_clearance_m,joint_margin_rad=None,intent=metrics,
        accepted_intent_degradation_unchanged=accepted_degradation,
        low_clearance_warning_s=risk,actual_tracking_error_m=actual_tracking_error,
        local_repair_evidence='camera intent programs; numerical FK/actuator screening; no WBC rollout'))
    if encode_journal:
        import json
        from .contracts import native
        result['_encoded_journal']=json.dumps(native(result),ensure_ascii=False,allow_nan=False)
    return result


def screen(path,search,spec,observed,budget,history):
    if getattr(path,'program',{}).get('curve_family'):
        from .regular_curve_search import bind_geometric_progress
        path=bind_geometric_progress(path)
    virtual=MeasuredCameraPrediction(path,search.state,search.base_z,search.t,budget.activation_lead_s,
        None if search.tilt is None else Rotation.from_euler('z',search.state[2]).as_matrix()@search.tilt)
    kinematics=search.lift(virtual,path.duration+POLICY.terminal_hold_s)
    result=dict(kinematics=kinematics,summary=summary(kinematics),error='',path=path)
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
    elif not metrics['terminal_frame']['pass_check']:
        result['error']='realized_terminal_frame_outside_band'
    result.update(value=value,metrics=metrics)
    return result


def build(original,active,strategy,scene,state,previous,t,base_z,limits,
          evidence=None,intrinsics=None,budget=None,commanded_q=None,history=(),
          measured_control=None,base_rotation=None,search_config=None,representation_boxes=3,geometry_preferences=True,
          construction_plan=None,initial_boundary=False):
    budget=budget or SearchBudget();start=time.monotonic()
    if budget.single_candidate and construction_plan is None:
        return dict(error='missing_jev_construction_contract',evaluations=[],generated_proposals=0,
                    wall_s=time.monotonic()-start)
    if initial_boundary and (t!=0. or active is not original or construction_plan is None):
        raise ValueError('measured boundary requires a Jev construction before shot startup')
    observed=camera_scene(scene,t,active.duration)
    observed['intrinsics']=intrinsics or observed.get('intrinsics')
    search=KinematicSearch(state,previous,t,base_z,limits,measured_control=measured_control,
        commanded_q=commanded_q,base_rotation=base_rotation,scene=scene,search_config=search_config)
    if construction_plan is not None:
        search.config=dict(search.config or {}, allocations=(construction_plan['allocation'],))
        search.policy=replace(search.policy,step_s=.15)
    configure_combo_samples(search,original)
    spec=IntentSpec(original,observed.get('subject_model'),observed.get('intrinsics'))
    def project(path):
        if path.program.get('curve_family')=='global_endpoint_polynomial':
            from .joint_curve_optimizer import refine
            fitted,diagnostics=refine(path,search,
                seconds=min(24.,budget.candidate_wall_budget_s*.8) if construction_plan is not None else min(8.,budget.candidate_wall_budget_s*.5),
                max_iterations=8 if construction_plan is not None else 30,
                knots=13 if construction_plan is not None else 33,
                step_tolerance=.01 if construction_plan is not None else 1e-4)
            if fitted is None:return dict(error='joint_curve_search:'+diagnostics['reason'],summary=diagnostics)
            return dict(path=fitted,summary=diagnostics)
        from .cinematic_geometry import regularize_lift,smooth_lift
        virtual=MeasuredCameraPrediction(path,search.state,search.base_z,search.t,budget.activation_lead_s,base_rotation)
        report=None
        allocations=list(dict.fromkeys(('path_heading',*search.config['allocations'])))[:2]
        for allocation in allocations:
            report=search._lift(virtual,path.duration+POLICY.terminal_hold_s,allocation,project_camera=True)
            if report['feasible']:
                fitted=smooth_lift(path,report)
                if fitted is None:fitted=regularize_lift(path,report,observed)
                if fitted is not None:
                    return dict(path=fitted,summary=summary(report))
                return dict(error='regular_curve_fit_unverified',summary=summary(report))
        return dict(error='fk_projection_unverified:'+report['reason'],summary=summary(report))
    # Filter every proposal before shortlist truncation, so an infeasible top
    # camera candidate cannot hide a feasible lower-ranked robot candidate.
    result=camera_candidates(original,active,strategy,observed,t,budget,RADIUS,history,
        allow_time_extension=False,kinematic_search=search,
        numerical_gate=lambda path:screen(path,search,spec,observed,budget,history),search_config=search_config,
        # Functional geometry preferences must not disable the backend's
        # ability to generate a realizable camera proposal. The final exact
        # path (including every deviation) still requires Agent selection.
        proposal_transform=project if search_config is not None else None,construction_plan=construction_plan,
        initial_boundary=(dict(position=search.p.tolist(),quaternion_xyzw=Rotation.from_matrix(search.r).as_quat().tolist(),
                               twist=(search.j@search.velocity).tolist()) if initial_boundary else None))
    from .representation_jvs import JVSGeometryBackend, candidate_geometry
    backend = JVSGeometryBackend(representation_boxes)
    observation = backend.observe(state,t,base_z=base_z,base_rotation=base_rotation)
    for candidate in result.get('candidates', []):
        # Bind the verified allocation into the approved trajectory. Re-solving
        # from an older coarse seed during installation could choose a different
        # branch and lose a curve that had already passed all strict checks.
        from .intent_paths import IntentPath
        witness=getattr(candidate,'kinematics',None)
        path=candidate.path
        if isinstance(path,IntentPath) and witness and witness.get('feasible') and path.program.get('route')!='continuation':
            exact_geometry=dict(path.geometry,joint_seed=witness['trace'],joint_seed_preference_only=False)
            candidate.path=IntentPath(path.original,path.previous,path.activation_s,
                dict(path.program,allocation_binding='verified_continuous_witness'),path.model,path.intrinsics,geometry=exact_geometry)
            candidate.id=path_hash(candidate.path)
        geometry = candidate_geometry(candidate,backend,observation,base_z,base_rotation,geometry_preferences)
        geometry['box_budget'] = representation_boxes
        candidate.consequences['geometry'] = geometry
        if initial_boundary:
            candidate.consequences['initial_verification_snapshot']=verification_snapshot(
                scene,state,previous,t,base_z,limits,measured_control,commanded_q,base_rotation)
    result.update(wall_s=time.monotonic()-start,snapshot_time_s=t,kinematic_prior=search.prior(),search_config=search_config,
                  activation_s=0. if initial_boundary else t+budget.activation_lead_s)
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
    guidance = None
    geometry = candidate.consequences.get('geometry')
    if geometry is not None:
        from .representation import Request, Observation
        from .representation_jvs import JVSGeometryBackend
        try:
            guidance = JVSGeometryBackend(geometry['box_budget']).compile(
                Request.from_dict(geometry['request']),Observation.from_dict(geometry['observation']),candidate.path)
        except (ValueError, KeyError, TypeError) as error:
            return dict(error='geometry_request_invalid:'+str(error),checks=checks)
    checks['target_authorized']='pass'
    error=support_violation(approval,evidence)
    if error:return dict(error=error,checks=checks)
    startup=bool(getattr(candidate.path,'program',{}).get('measured_initial_boundary') and
                 t==candidate.snapshot_time_s==candidate.path.activation_s==0.)
    if not 0<=t-candidate.snapshot_time_s<=budget.max_prediction_age_s or (t>=candidate.path.activation_s and not startup):
        return dict(error='candidate_expired_requires_new_program',checks=checks)
    if not -1e-8<=t-float(scene['observation_stamp'])<=.6:return dict(error='scene_stale',checks=checks)
    checks['state_fresh']='pass'
    observed=camera_scene(scene,t,candidate.path.duration)
    spec=IntentSpec(original,observed.get('subject_model'),intrinsics or observed.get('intrinsics'))
    search=KinematicSearch(state,previous,t,base_z,limits,measured_control=measured_control,
        commanded_q=commanded_q,base_rotation=base_rotation,scene=scene,
        search_config=candidate.consequences.get('search_config'))
    configure_combo_samples(search,original)
    identical_startup=(startup and candidate.kinematics is not None and candidate.kinematics.get('feasible') and
        candidate.consequences.get('initial_verification_snapshot')==verification_snapshot(
            scene,state,previous,t,base_z,limits,measured_control,commanded_q,base_rotation))
    if identical_startup:
        # The fully screened continuous witness is part of the exact approved
        # candidate. Re-solving the SAME initial state can switch redundancy
        # branches and reject a valid witness. No reuse across changed time,
        # observations, state, velocity, limits or base pose is permitted.
        screened=dict(kinematics=candidate.kinematics,value=candidate.validation,
                      metrics=candidate.consequences['intent'])
        checks['initial_witness_reused']='identical_snapshot_and_approved_binding'
    else:
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
        kinematics=screened['kinematics'],kinematic_limits=limits,geometry_guidance=guidance)
