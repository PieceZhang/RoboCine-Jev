"""Public task status from actual bounded solver/physical evidence.

Missing fresh evidence is unknown, not an infeasibility certificate. This helper
does not validate sequence, task context, leases, or physical safety itself.
"""
import math


def execution_lease_status(packet,*,stamp_s,wall_s,horizon_s=0.):
    """Check the original lease through the next declared execution interval."""
    if not isinstance(packet,dict) or packet.get('ready') is not True:
        return dict(valid=False,reason='no_fresh_command')
    try:
        start_sim=float(packet['stamp_s']);start_wall=float(packet['command_sent_wall_s'])
        end_sim=float(packet['lease_deadline_sim_s']);end_wall=float(packet['lease_deadline_wall_s'])
        if (not all(map(math.isfinite,(stamp_s,wall_s,horizon_s,start_sim,start_wall,end_sim,end_wall)))
                or isinstance(horizon_s,bool) or horizon_s<0):
            raise ValueError('finite execution interval required')
        valid=bool(start_sim-1e-9<=stamp_s and start_wall-1e-9<=wall_s and
            stamp_s+horizon_s<=end_sim+1e-9 and wall_s+horizon_s<=end_wall+1e-9)
        return dict(valid=valid,reason='within_original_lease' if valid else 'command_stale',
            sim_age_s=float(stamp_s-start_sim),wall_age_s=float(wall_s-start_wall),execution_horizon_s=float(horizon_s))
    except (KeyError,TypeError,ValueError):
        return dict(valid=False,reason='missing_or_invalid_command_lease')


def public_support(packet,collision_evidence=None):
    if collision_evidence and collision_evidence.get('status')=='fail':
        return dict(reason='self_collision',feasible=False)
    if not isinstance(packet,dict):
        return dict(reason='command_stale',feasible=None)
    status=packet.get('status','')
    if status=='reference_step_clipped_hold_upper_targets':
        return dict(reason='configuration_limit',feasible=False)
    if 'deadline_exceeded' in status:
        return dict(reason='deadline_exceeded',feasible=False)
    if (packet.get('ready') is True and packet.get('ok') is True
            and status=='fresh_command'):
        return dict(reason='accepted',feasible=None)
    if (packet.get('error') or status in ('reply_context_mismatch','invalid_context')
            or packet.get('failure_kind') in ('solver_failure','transport_failure')):
        return dict(reason='solver_failure',feasible=False)
    return dict(reason='command_stale',feasible=None)
