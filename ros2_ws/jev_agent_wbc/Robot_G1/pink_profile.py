"""Declared native Pink soft-task schedule; no task pose or joint command edits."""
import math


def first_park_posture_refresh(*,enabled,entered,already_refreshed,owner_names,measured_joints,stamp_s,task_context):
    """Snapshot one subordinate preference at the original measured admission.

    The caller resets its latch only for a newly accepted Task. Neither later
    physical motion nor another PARK admission updates this Task's preference.
    """
    if not all(isinstance(v,bool) for v in (enabled,entered,already_refreshed)):
        raise ValueError('explicit posture refresh mode and latch required')
    if not enabled or not entered or already_refreshed:return None
    if (isinstance(stamp_s,bool) or not isinstance(stamp_s,(int,float)) or not math.isfinite(stamp_s)
            or stamp_s<0 or not isinstance(task_context,dict)
            or set(task_context)!=set(('schema','task_id','camera_path_id','task_sha256'))
            or task_context['schema']!='g1_solver_task_context/v1'
            or any(not isinstance(v,str) or not v for v in task_context.values())
            or len(task_context['task_sha256'])!=64 or len(task_context['camera_path_id'])!=64):
        raise ValueError('actual admission stamp and complete current Task context required')
    if (not isinstance(owner_names,(list,tuple)) or len(owner_names)!=15 or len(set(owner_names))!=15
            or not isinstance(measured_joints,dict) or any(n not in measured_joints for n in owner_names)):
        raise ValueError('all fifteen owned actual joint measurements required')
    target={}
    for name in owner_names:
        value=measured_joints[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('finite actual measured joint positions required')
        target[name]=float(value)
    return dict(kind='first_PARK_measured_soft_preference',stamp_s=float(stamp_s),targets=target,
        task_context=dict(task_context),source='measured_physx_joint_pos_at_original_PARK_admission',
        refresh_count=1,waist_yaw_home_override_preserved=True,
        camera_FrameTask_target_and_cost_unchanged=True,original_collision_and_joint_limits_unchanged=True,
        controlled_upper_dofs=15,hard_joint_lock=False)


def arm_pd_preflight(*,joint_names,stiffness_scale,source_stiffness,source_damping,source_armature,
                     source_effort_caps,source_velocity_caps,actual_stiffness,actual_damping,
                     actual_armature,actual_effort_caps,actual_velocity_caps,engine_stiffness,
                     engine_damping,upper_damping_scale,physics_dt_s=.005):
    """Readback gate for a declared explicit-arm PD comparison.

    The scalar armature inequality is a necessary screening assumption, not a
    coupled floating-base/contact stability or braking certificate.
    """
    import numpy as np
    if (len(joint_names)!=14 or len(set(joint_names))!=14
            or any(not isinstance(n,str) or not n.startswith(('left_','right_')) for n in joint_names)):
        raise ValueError('the fourteen actual arm joint names are required')
    if (isinstance(stiffness_scale,bool) or stiffness_scale not in (1.,2.,4.)
            or isinstance(upper_damping_scale,bool) or not 1<=upper_damping_scale<=5
            or isinstance(physics_dt_s,bool) or physics_dt_s!=.005):
        raise ValueError('declared native200Hz PD comparison required')
    values={}
    for name,value in dict(source_stiffness=source_stiffness,source_damping=source_damping,
            source_armature=source_armature,source_effort_caps=source_effort_caps,
            source_velocity_caps=source_velocity_caps,actual_stiffness=actual_stiffness,
            actual_damping=actual_damping,actual_armature=actual_armature,
            actual_effort_caps=actual_effort_caps,actual_velocity_caps=actual_velocity_caps,
            engine_stiffness=engine_stiffness,engine_damping=engine_damping).items():
        a=np.asarray(value)
        if a.shape!=(14,) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all() or np.any(a<0):
            raise ValueError('finite nonnegative actual arm parameter vectors required')
        values[name]=a.astype(float)
    failures=[]
    expected=dict(actual_stiffness=values['source_stiffness']*stiffness_scale,
        actual_damping=values['source_damping']*upper_damping_scale,
        actual_armature=values['source_armature'],actual_effort_caps=values['source_effort_caps'],
        actual_velocity_caps=values['source_velocity_caps'],engine_stiffness=np.zeros(14),engine_damping=np.zeros(14))
    for name,wanted in expected.items():
        if not np.allclose(values[name],wanted,rtol=1e-6,atol=1e-8):failures.append(name+'_readback_mismatch')
    threshold=(values['actual_stiffness']*physics_dt_s**2+2*values['actual_damping']*physics_dt_s)/4
    slack=values['actual_armature']-threshold
    if np.any(slack<=0):failures.append('scalar_armature_stability_screen_failed')
    if np.any(values['actual_effort_caps']<=0) or np.any(values['actual_velocity_caps']<=0):
        failures.append('nonpositive_native_cap')
    return dict(schema='g1_explicit_arm_PD_readback/v1',passed=not failures,failures=failures,
        joint_names=list(joint_names),stiffness_scale=float(stiffness_scale),upper_damping_scale=float(upper_damping_scale),
        physics_dt_s=physics_dt_s,parameters={k:v.tolist() for k,v in values.items()},
        scalar_necessary_inertia_threshold_kg_m2=threshold.tolist(),
        actual_armature_minus_scalar_threshold_kg_m2=slack.tolist(),
        effort_implementation='single explicit PD+gravity sum, original effort cap clamp; implicit engine PD zero',
        physical_execution_started=False,coupled_stability_verified=False,actual_braking_certified=False,
        scope='Source matching and scalar discrete screening only; original measured caps/FCL/stop acceptance remain required.')


def velocity_servo_overrides(profile):
    if profile not in ('baseline','low_gain','yaw_gain2','yaw_phase_2_3_v1','yaw_pi08_2_ff1_v1','yaw_phase_symmetric2_v1','yaw_phase_symmetric2_ki4_v1'):
        raise ValueError('unknown declared G1 input profile')
    if profile in ('yaw_phase_2_3_v1','yaw_pi08_2_ff1_v1','yaw_phase_symmetric2_v1','yaw_phase_symmetric2_ki4_v1'):
        return dict(feedforward_gain=(1.,1.,1.),negative_feedforward_gain=(1.,1.,1.))
    return dict(feedforward_gain=(1.,1.,2.)) if profile=='yaw_gain2' else {}


def phase_velocity_servo_overrides(profile,moving,requested_yaw_rad_s):
    """Calibrate authored moving yaw; stop-anchor feedback retains baseline FF."""
    velocity_servo_overrides(profile)
    if not isinstance(moving,bool) or not math.isfinite(requested_yaw_rad_s):
        raise ValueError('explicit moving phase and finite original yaw request required')
    if profile not in ('yaw_phase_2_3_v1','yaw_pi08_2_ff1_v1','yaw_phase_symmetric2_v1','yaw_phase_symmetric2_ki4_v1'):return None
    enabled=bool(moving and requested_yaw_rad_s!=0.)
    if profile=='yaw_phase_symmetric2_ki4_v1':
        gain=(1.,1.,2.) if enabled else (1.,1.,1.)
        return dict(feedforward_gain=gain,negative_feedforward_gain=gain,
            ki=(1.2,1.2,4. if enabled else 1.),
            phase='moving_yaw_symmetric_feedforward_faster_integral' if enabled else 'baseline',
            original_requested_yaw_rad_s=float(requested_yaw_rad_s))
    if profile=='yaw_phase_symmetric2_v1':
        gain=(1.,1.,2.) if enabled else (1.,1.,1.)
        return dict(feedforward_gain=gain,negative_feedforward_gain=gain,
            phase='moving_yaw_symmetric_feedforward' if enabled else 'baseline',
            original_requested_yaw_rad_s=float(requested_yaw_rad_s))
    if profile=='yaw_pi08_2_ff1_v1':
        return dict(feedforward_gain=(1.,1.,1.),negative_feedforward_gain=(1.,1.,1.),
            kp=(1.,1.,.8),ki=(1.2,1.2,2. if enabled else 1.),
            phase='moving_yaw_faster_integral' if enabled else 'baseline',
            original_requested_yaw_rad_s=float(requested_yaw_rad_s))
    return dict(feedforward_gain=(1.,1.,2.) if enabled else (1.,1.,1.),
        negative_feedforward_gain=(1.,1.,3.) if enabled else (1.,1.,1.),
        phase='moving_yaw' if enabled else 'baseline',original_requested_yaw_rad_s=float(requested_yaw_rad_s))


def task_cost_overrides(locomotion_task,moving,walking_posture_cost=None,terminal_damping_cost=None):
    if not isinstance(locomotion_task,bool) or not isinstance(moving,bool):
        raise ValueError('explicit task and phase booleans required')
    for value,lower,upper in [(walking_posture_cost,.03,.15),(terminal_damping_cost,.02,.5)]:
        if value is not None and (isinstance(value,bool) or not math.isfinite(value) or not lower<=value<=upper):
            raise ValueError('finite bounded native soft-task cost required')
    return dict(posture_cost_override=walking_posture_cost if locomotion_task and moving else None,
        damping_cost_override=terminal_damping_cost if locomotion_task and not moving else None)
