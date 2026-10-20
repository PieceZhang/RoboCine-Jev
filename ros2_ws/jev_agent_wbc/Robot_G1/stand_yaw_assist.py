"""Optional measured PARK-only native yaw input, without learned-action edits."""
import math
import numpy as np


def command(parking_command,previous_applied,*,mode,anchor_yaw_error_rad,measured_yaw_rate_rad_s,dt_s):
    """Preserve brake/settle; assist yaw only after native PARK admission.

    Original parking entry and drift exit are owned by the unchanged supervisor.
    This proportional input preference has no integrator and no new safety or
    training-distribution certificate. Downstream vetoes remain authoritative.
    """
    base=np.asarray(parking_command);previous=np.asarray(previous_applied)
    if (base.shape!=(3,) or previous.shape!=(3,) or base.dtype.kind not in 'fiu'
            or previous.dtype.kind not in 'fiu' or not np.isfinite(np.r_[base,previous]).all()
            or mode not in ('BRAKE','SETTLE','PARK')):
        raise ValueError('actual finite native input and declared parking mode required')
    for value in (anchor_yaw_error_rad,measured_yaw_rate_rad_s,dt_s):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.floating)) or not math.isfinite(value):
            raise ValueError('finite measured yaw/error and timestep required')
    if not 0<dt_s<=.1 or np.any(np.abs(base)>[.5,.5,1.]) or np.any(np.abs(previous)>[.5,.5,1.]):
        raise ValueError('native limits and bounded positive timestep required')
    result=base.astype(float).copy();raw=1.5*anchor_yaw_error_rad-.8*measured_yaw_rate_rad_s
    if mode=='PARK':
        if np.any(np.abs(previous[:2])>.5*dt_s+1e-12):
            raise ValueError('PARK literal-zero XY would exceed the original slew')
        low=max(-.08,float(previous[2]-.8*dt_s));high=min(.08,float(previous[2]+.8*dt_s))
        if low>high+1e-12:raise ValueError('PARK yaw cap and original slew have no feasible input')
        result=np.array([0.,0.,np.clip(raw,low,high)])
    return result,dict(active=mode=='PARK',mode=mode,anchor_yaw_error_rad=float(anchor_yaw_error_rad),
        measured_yaw_rate_rad_s=float(measured_yaw_rate_rad_s),unbounded_yaw_input_rad_s=float(raw),
        requested_native_input=result.tolist(),yaw_input_cap_rad_s=.08,original_yaw_slew_rad_s2=.8,
        yaw_position_gain_per_s=1.5,yaw_rate_gain=.8,integrator_used=False,
        original_parking_entry_exit_unchanged=True,learned_joint_actions_modified=False,
        world_camera_task_modified=False,physical_stopping_verified=False)
