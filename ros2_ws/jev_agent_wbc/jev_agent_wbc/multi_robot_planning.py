"""Native-model realization behind the unchanged typed Jev decision workflow."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .representation import Pose, Request
from .wbc_protocol import OPTICAL_FROM_SEMANTIC, optical_camera, semantic_camera
from .multi_robot_geometry import NativeKinematicModel, shape_clearance
from .contracts import Envelope, PARAMETERS, native
from .reference import DeformedReference
from .candidates import path_hash, path_spec
from .camera_intent import obstacle_state


def native_search_budget():
    """Use the existing PhysX entry-point defaults (scripts/run_isaac.py).

    The generic SearchBudget defaults are for a different entry point. Source
    observations/commands still expire independently while physics advances.
    """
    from .candidates import SearchBudget
    return SearchBudget(activation_lead_s=.9,candidate_wall_budget_s=18.)


class NativeDeformation(DeformedReference):
    """Batch evaluation of the existing C2 formula; no trajectory change."""
    def _batch_geometry(self,times):
        from .reference import smoothstep
        times=np.asarray(times,float)
        at=np.clip(times,0.,self.duration)
        p=self.original.translation(at)
        r=self.original.rotation(at).as_matrix()
        blend=smoothstep((times-self.activation_s)/self.envelope.blend_s)
        recover=smoothstep((times-self.return_s)/self.envelope.blend_s)
        values=(blend*(1.-recover))[:,None]*self.values
        radial=p-self.subject
        radial[:,2]=0.
        singular=np.linalg.norm(radial,axis=1)<1e-8
        radial[singular]=-r[singular,:,0];radial[singular,2]=0.
        radial/=np.maximum(np.linalg.norm(radial,axis=1),1e-8)[:,None]
        moved=p+values[:,0,None]*radial+values[:,1,None]*r[:,:,1]
        moved[:,2]+=values[:,2]
        return p,r,moved,values,blend

    def positions_at(self,times):
        return self._batch_geometry(times)[2]

    def poses_at(self,times):
        from .reference import look_at_many
        if not len(times):return np.empty((0,3)),np.empty((0,3,3))
        p,r,moved,values,blend=self._batch_geometry(times)
        if 'subject_aim' in self.envelope.protected:
            correction=look_at_many(moved,self.subject)@look_at_many(p,self.subject).transpose(0,2,1)
            r=Rotation.from_rotvec(blend[:,None]*Rotation.from_matrix(correction).as_rotvec()).as_matrix()@r
        r=r@Rotation.from_euler('ZYX',values[:,3:]).as_matrix()
        return moved,r


def platform_step(state, previous, initial_yaw, dt):
    """Finite-step native body rates and their analytic state derivative."""
    heading=initial_yaw+(state[2]+previous[2])/2
    dx,dy,dtheta=np.asarray(state[:3])-previous[:3]
    c,s=np.cos(heading),np.sin(heading)
    forward,lateral=dx*c+dy*s,dy*c-dx*s
    jacobian=np.zeros((3,len(state)))
    jacobian[:,:3]=np.array([[c,s,.5*lateral],[-s,c,-.5*forward],[0.,0.,1.]])/dt
    return np.array([forward,lateral,dtheta])/dt,jacobian


class CameraAttachmentScreen:
    """Cheap native geometry ranking before the bounded whole-body rollouts.

    Bodies with the camera's exact active-joint chain have a fixed measured
    transform relative to it. This uses their actual collider boxes instead
    of treating the optical center as the entire manipulator. It is a ranking
    heuristic; the complete native screen remains the acceptance gate.
    """
    def __init__(self,model,observed,start_s,progress_start_s,step=.1):
        from .camera_intent import obstacle_state
        if isinstance(model,dict):model=NativeKinematicModel(model)
        zero=np.zeros(3+len(model.joints))
        inverse=np.linalg.inv(model.camera(zero))
        selected=[shape for shape in model.shapes
                  if model.chains[shape['body']]==model.chains[model.camera_parent]]
        poses=model.poses(zero)
        self.transforms=[inverse@poses[shape['body']] for shape in selected]
        self.centers=np.array([shape['center_body'] for shape in selected]).reshape(-1,3)
        self.half=np.array([shape['half_extents_body'] for shape in selected]).reshape(-1,3)
        self.times=np.linspace(progress_start_s,8.,int(np.ceil((8.-progress_start_s)/step))+1)
        self.obstacles=[obstacle_state(observed,start_s+t) for t in self.times]

    def clearance(self,path):
        from .intent_paths import poses_at
        if not self.transforms:return None
        positions,rotations=poses_at(path,self.times)
        rotations=rotations@OPTICAL_FROM_SEMANTIC.rotation.T
        local=np.asarray(self.transforms)
        body_rotations=rotations[:,None]@local[None,:,:3,:3]
        origins=np.einsum('tij,sj->tsi',rotations,local[:,:3,3])+positions[:,None]
        origins+=np.einsum('tsij,sj->tsi',body_rotations,self.centers)
        gaps=[]
        for i,(centers,radii) in enumerate(self.obstacles):
            if not len(centers):continue
            points=np.einsum('nsi,sij->nsj',centers[:,None]-origins[i][None],body_rotations[i])
            delta=np.abs(points)-self.half
            distances=np.linalg.norm(np.maximum(delta,0.),axis=2)+np.minimum(delta.max(axis=2),0.)-radii[:,None]
            gaps.append(float(distances.min()))
        return min(gaps) if gaps else None

    @staticmethod
    def rank(clearance,camera_score):
        return (clearance is None or clearance<.025,
                -clearance if clearance is not None else float('inf'),
                not camera_score.feasible,-camera_score.min_clearance_m)


def lift(path, model_document, observed, *, step=.1, start_s=0., progress_start_s=0., relaxed=False, deadline=None):
    """Bounded differential/holonomic IK screen; execution still uses native WBC.

    Collision screening uses acquisition-time observations with the existing
    linear motion and growing uncertainty model, never authored obstacle paths.
    The returned platform curve is a subordinate native allocation preference.
    """
    model = NativeKinematicModel(model_document)
    fast=bool(model.omni and model.compiled_geometry)
    if model.compiled_geometry:
        from .native_lift_solver import solve as bounded_solve,residual as compiled_residual,geometry_arguments
        geometry_args=geometry_arguments(model)
    n = 3+len(model.joints)
    current = np.zeros(n)
    if not 0.<=progress_start_s<path.duration:raise ValueError('prediction must start in the remaining shot')
    times = np.linspace(progress_start_s,path.duration,int(np.ceil((path.duration-progress_start_s)/step))+1)
    from .intent_paths import poses_at
    target_positions,target_rotations=poses_at(path,times)
    optical_rotations=target_rotations@OPTICAL_FROM_SEMANTIC.rotation.T
    states, errors, angles, gaps, lateral, platform_ratios = [], [], [], [], [], []
    p0 = np.asarray(target_positions[0])
    yaw0 = np.arctan2(model.base[1, 0], model.base[0, 0])
    caps = np.r_[.95*model.platform_limits, model.speed]
    bounds_lo = np.r_[-3., -3., -2., model.lower-model.q0+.002]
    bounds_hi = np.r_[3., 3., 2., model.upper-model.q0-.002]
    for i, t in enumerate(times):
        if deadline is not None:
            import time
            if time.monotonic() >= deadline:
                raise TimeoutError('unchanged numerical candidate wall budget exhausted')
        p=target_positions[i]
        target_p,target_r=p,optical_rotations[i]
        previous = current.copy()
        dt = float(times[i]-times[i-1]) if i else .2
        lo, hi = np.maximum(bounds_lo, previous-caps*dt), np.minimum(bounds_hi, previous+caps*dt)
        # The platform follows most of the displacement, retaining arm reserve.
        base_preference = .85*(np.asarray(p)-p0)[:2]
        centers, radii = obstacle_state(observed, start_s+float(t))
        # Remove rows that are identically inactive near this bounded IK step
        # from the dense least-squares system. The full geometry is still
        # independently checked at every returned state below.
        initial_distances=model.obstacle_distances(previous,centers,radii)
        near_centers=np.any(initial_distances<.25,axis=1)
        cost_centers,cost_radii=centers[near_centers],radii[near_centers]
        cost_pairs=initial_distances[near_centers]<.25
        pair_i,pair_j=np.nonzero(cost_pairs)
        # Fixed row layout avoids repeated concatenate/identity allocation in
        # every trust-region evaluation. Residuals and weights are unchanged.
        posture=slice(8,8+n-3)
        regularization=slice(posture.stop,posture.stop+n)
        platform=slice(regularization.stop,regularization.stop+3)
        nonholonomic=platform.stop
        collision=nonholonomic+int(not model.omni)
        template=np.zeros((collision+int(cost_pairs.sum()),n))
        template[6:8,:2]=.15*np.eye(2)
        template[posture,3:]=.025*np.eye(n-3)
        template[regularization]=.01*np.eye(n)

        cache=[]
        def evaluate(x):
            if cache and np.array_equal(cache[0],x):
                return cache[1:3]
            if model.compiled_geometry:
                e,jac=compiled_residual(x,previous,geometry_args,target_p,target_r,base_preference,
                    cost_centers,cost_radii,pair_i,pair_j,model.platform_limits,
                    model.omni,model.lateral_yaw_gain,yaw0,dt)
                cache[:]=[x.copy(),e,jac]
                return e,jac
            camera,J,distances,D=model.differential(x,cost_centers,cost_radii,active_distance=.04)
            angle=Rotation.from_matrix(target_r@camera[:3,:3].T).as_rotvec()
            ax,ay,az=angle
            W=np.array([[0.,-az,ay],[az,0.,-ax],[-ay,ax,0.]])
            theta=np.linalg.norm(angle)
            coefficient=1./12 if theta<1e-5 else (1/theta**2-(1+np.cos(theta))/(2*theta*np.sin(theta)))
            log_jac=np.eye(3)+.5*W+coefficient*W@W
            e=np.empty(len(template));jac=template.copy()
            e[:3]=60*(camera[:3,3]-target_p);e[3:6]=16*angle
            e[6:8]=.15*(x[:2]-base_preference)
            e[posture]=.025*x[3:];e[regularization]=.01*(x-previous)
            jac[:3]=60*J[:3];jac[3:6]=-16*log_jac@J[3:]
            body_rates,body_jac=platform_step(x,previous,yaw0,dt)
            excess=np.maximum(np.abs(body_rates)-.95*model.platform_limits,0.)
            e[platform]=100.*excess
            jac[platform]=100.*np.where((excess>0.)[:,None],
                                        np.sign(body_rates)[:,None]*body_jac,0.)
            if not model.omni:
                heading = yaw0+(x[2]+previous[2])/2
                dx,dy=x[:2]-previous[:2]
                e[nonholonomic]=150*(dy*np.cos(heading)-dx*np.sin(heading)
                                    -model.lateral_yaw_gain*(x[2]-previous[2]))
                jac[nonholonomic,:3]=150*np.array([-np.sin(heading),np.cos(heading),
                    -.5*(dy*np.sin(heading)+dx*np.cos(heading))-model.lateral_yaw_gain])
            if len(cost_centers):
                # Prefer resolving predicted whole-body proximity before
                # spending the unchanged camera-error allowance. The final
                # 25 mm clearance and native bounds remain independent gates.
                e[collision:]=120*np.maximum(.04-distances[cost_pairs],0.)
                jac[collision:]=np.where((distances[cost_pairs]<.04)[:,None],-120*D[cost_pairs],0.)
            cache[:]=[x.copy(),e,jac]
            return e,jac
        if i:
            initial=np.clip(current,lo+1e-10,hi-1e-10)
            if np.any(lo>=hi):raise ValueError('empty native step bounds')
            if fast:
                current,_=bounded_solve(initial,previous,lo,hi,geometry_args,target_p,target_r,
                    base_preference,cost_centers,cost_radii,pair_i,pair_j,model.platform_limits,
                    model.omni,model.lateral_yaw_gain,yaw0,dt,maxeval=14)
                if not np.isfinite(current).all() or np.any(current<lo) or np.any(current>hi):
                    raise ValueError('invalid bounded native solver output')
            else:
                result = least_squares(lambda x:evaluate(x)[0], initial,
                                       jac=lambda x:evaluate(x)[1],
                                       bounds=(lo, hi), max_nfev=14, ftol=1e-5, xtol=1e-5, gtol=1e-5)
                current = result.x
        camera,gap = model.camera_and_clearance(current,centers,radii)
        errors.append(float(np.linalg.norm(camera[:3, 3]-target_p)))
        angles.append(float(Rotation.from_matrix(target_r@camera[:3, :3].T).magnitude()))
        gaps.append(gap)
        body_rates,_=platform_step(current,previous,yaw0,dt)
        platform_ratios.append(float(np.max(np.abs(body_rates)/model.platform_limits)))
        lateral.append(float(abs(body_rates[1]-model.lateral_yaw_gain*body_rates[2])))
        states.append(current.copy())
    finite_gaps = [g for g in gaps if g is not None]
    position_limit, angle_limit = (.08, .20) if relaxed else (.03, .08)
    tracking = max(errors) <= position_limit and max(angles) <= angle_limit
    clearance = min(finite_gaps) if finite_gaps else None
    geometry_samples=[dict(stamp_s=float(start_s+times[i]),evidence_kind='predicted',
        geometry=model.predicted_observation(states[i],start_s+times[i]).summary(),
        preference_residual_norm=0.,safety_certificate=False)
        for i in sorted(set((0,len(times)//2,len(times)-1)))]
    if deadline is not None:
        import time
        if time.monotonic()>deadline:
            raise TimeoutError('unchanged numerical candidate wall budget exhausted')
    return dict(solver_kind=("compiled_bounded_gauss_newton" if fast else
        "scipy_trust_region_compiled_residual" if model.compiled_geometry else "scipy_trust_region"),
        times_s=times.tolist(), states=np.asarray(states).tolist(),
        geometry_samples=geometry_samples,
        position_errors_m=errors, orientation_errors_rad=angles,
        maximum_position_error_m=max(errors), maximum_orientation_error_rad=max(angles),
        minimum_clearance_m=clearance, clearance_series_m=gaps,
        tracking_screen_pass=bool(tracking),
        nonholonomic_screen_pass=bool(model.omni or max(lateral) <= .005),
        native_platform_screen_pass=bool(max(platform_ratios)<=1.00001),
        maximum_native_platform_velocity_ratio=max(platform_ratios),
        observed_collision_screen_pass=bool(clearance is not None and clearance >= .025),
        physical_valid=bool(tracking and (model.omni or max(lateral) <= .005)
                            and max(platform_ratios)<=1.00001
                            and clearance is not None and clearance >= .025),
        scope='native_calibrated_kinematic_model_and_observed_proxy_forecast_no_dynamics',
        unobserved_space_is_free=False, full_body_dynamics_verified=False)


def candidate_paths(original, family, configuration, activation_s=0.):
    """Existing deformation family and bounds; no robot-specific semantic policy."""
    from .search_configuration import PROFILES
    from .intent_paths import IntentPath
    amplitudes = PROFILES[configuration]['amplitudes']
    # Existing fixed-deadline clock programs retain the complete geometric
    # shot. Two clock signs plus four spatial seeds respect max_proposals=6.
    for rate in (.7,-.7):
        yield IntentPath(original,original,activation_s,dict(
            route='timing',amplitude_m=0.,extension_s=0.,phase_rate=rate))
    operators={
        'framing':[((0,1.),),((0,-1.),),((2,1.),),((2,-1.),)],
        'view':[((1,1.),),((1,-1.),),((2,1.),),((2,-1.),)],
        'joint':[((0,1.),(2,1.)),((0,1.),(2,-1.)),
                 ((0,1.),),((2,-1.),)],
    }
    if family not in operators:
        raise ValueError('unsupported native camera search family')
    from .camera_intent import POLICY
    from .reference import smoothstep
    # The same six proposals and cinematic limits apply to every backend.
    # Normalize displacement by subject range, without consulting robot type.
    maximum = amplitudes[-1]
    blend=min(3.,(original.duration-activation_s)/2.)
    times=np.linspace(0.,original.duration,81)
    original_positions=original.translation(times)
    distance=np.linalg.norm(original_positions-original.subject,axis=1)
    weight=smoothstep((times-activation_s)/blend)*(1.-smoothstep((times-(original.duration-blend))/blend))
    relative=max(float(np.max(weight/np.maximum(distance,1e-6))),1e-6)
    for operator in operators[family]:
        values=np.zeros(6)
        for axis,sign in operator:
            cinematic=(np.expm1(POLICY.process_log_area_error_max/2.) if sign>0 else
                       -np.expm1(-POLICY.process_log_area_error_max/2.)) if axis==0 else np.tan(POLICY.process_view_error_max_rad)
            amplitude=min(maximum,POLICY.companion_displacement_max_m,.9*cinematic/relative)
            values[axis] = sign*amplitude
        magnitude=np.linalg.norm(values[:3])
        # Leave tracking room inside the unchanged measured cinematic limit.
        if magnitude>.95*POLICY.companion_displacement_max_m:
            values*=.95*POLICY.companion_displacement_max_m/magnitude
        bounds = {name: [-abs(value), abs(value)] for name,value in zip(PARAMETERS,values)}
        envelope = Envelope(original.sha256, 'numerical_proposal_only', 'obstacle',
            0., 0., original.duration, blend, bounds, ('subject_aim',), 'recover',
            'Bounded counterfactual; execution requires exact Jev selection.')
        path=NativeDeformation(original, envelope, values, activation_s=activation_s, return_s=original.duration-blend)
        deviation=path.positions_at(times)-original_positions
        travel=float(np.linalg.norm(np.diff(deviation,axis=0),axis=1).sum())
        if travel>POLICY.companion_travel_max_m:
            path.values*=.99*POLICY.companion_travel_max_m/travel
        yield path


def card(path, report, observation, start_s):
    from .multi_robot_scenarios import timed_path
    identity = path_hash(path)
    execution=timed_path(path,observation,start_s)
    # A selected candidate reuses these exact already-hashed knots. The native
    # executor independently regenerates/checks the path on task admission.
    path._native_execution_trajectory=execution.trajectory
    request = Request(observation.id,execution.trajectory.id,start_s,start_s+path.duration)
    trajectory=path_spec(path)
    from .multi_robot_evidence import intent_summary,deformation_program
    if not trajectory.get("program"):
        trajectory["program"]=deformation_program(path)
    trajectory.update(duration_s=path.duration,
        execution=dict(kind='exact_selected_camera_path',
                     original_deadline_unchanged=True, terminal='recover',
                     execution_camera_path_id=execution.trajectory.id,
                     execution_interpolation='immutable 0.04 s knots; linear position and SLERP orientation'))
    return dict(id=identity, trajectory=trajectory,
        physical_valid=report['physical_valid'],
        prediction=dict(min_clearance_m=report['minimum_clearance_m']),
        visual=dict(geometry=dict(request=request.to_dict(), observation=observation.to_dict(),
            samples=report['geometry_samples'], scope=report['scope']),
            intent=intent_summary(report.get('cinematic_prediction',dict(original_motion_preserved='not_verified',
                        exact_original_endpoint=True, original_reference_deviation_m=float(np.linalg.norm(path.values[:3])),
                        full_intent_acceptance='requires_measured_execution'))),
            smoothness='existing_C2_deformation'),
        screen=native({k: v for k, v in report.items() if k not in ('states', 'times_s',
            'position_errors_m', 'orientation_errors_rad', 'clearance_series_m','geometry_samples',
            'cinematic_prediction')}))
