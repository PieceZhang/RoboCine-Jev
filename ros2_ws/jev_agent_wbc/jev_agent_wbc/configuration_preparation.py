"""Task-preserving preparation certificates and measured execution monitoring.

The earliest acceptable sampled state is sought on a validated continuous lift.
This is a bounded-search result, not a globally time-optimal control solution.
The camera clock is never stopped or reset. Maintenance remains active afterwards.
"""
import numpy as np
from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER

JOINT_NEIGHBORHOOD_RAD = .45
BASE_NEIGHBORHOOD_M = .20
YAW_NEIGHBORHOOD_RAD = .40
TAIL_MARGIN_RAD = .08
TAIL_SPEED_RATIO = .98


def preparation(report, activation, limits):
    result = dict(status='not_found', duration_s=None, start_s=float(activation),
        shortest_scope='earliest sampled feasible time on the chosen validated allocation; not global optimum',
        joint_neighborhood_rad=JOINT_NEIGHBORHOOD_RAD, base_neighborhood_m=BASE_NEIGHBORHOOD_M,
        yaw_neighborhood_rad=YAW_NEIGHBORHOOD_RAD, required_tail_joint_margin_rad=TAIL_MARGIN_RAD,
        required_tail_velocity_ratio=TAIL_SPEED_RATIO, camera_clock='unchanged', maintenance='rolling_task_priority_QP')
    trace = report.get('trace', [])
    if not report.get('feasible') or not trace:
        return result
    times=np.array([r['t'] for r in trace]);states=np.array([r['state'] for r in trace])
    controls=np.array([r['control'] for r in trace])
    start=np.array([np.interp(activation,times,states[:,j]) for j in range(9)])
    if times[0]<=activation<=times[-1] and not np.any(np.isclose(times,activation,atol=1e-9,rtol=0.)):
        index=np.searchsorted(times,activation)
        initial_control=np.array([np.interp(activation,times,controls[:,j]) for j in range(8)])
        times=np.insert(times,index,activation)
        states=np.insert(states,index,start,axis=0)
        controls=np.insert(controls,index,initial_control,axis=0)
    margin=np.min(np.minimum(states[:,3:]-JOINT_LOWER,JOINT_UPPER-states[:,3:]),axis=1)
    # A state includes velocity: stopping distance is reserved in the direction
    # of travel. Tail comfort therefore depends on the actual future motion.
    stopping=controls[:,2:]**2/(2*limits.acceleration[2:])
    directed=np.where(controls[:,2:]>=0,JOINT_UPPER-states[:,3:],states[:,3:]-JOINT_LOWER)-stopping
    reserve=np.minimum(margin,directed.min(axis=1))
    speed=np.maximum(np.max(abs(controls)/limits.velocity,axis=1),
                     np.max(abs(controls@limits.wheels.T),axis=1)/limits.wheel_speed)
    tail_margin=np.minimum.accumulate(reserve[::-1])[::-1]
    tail_speed=np.maximum.accumulate(speed[::-1])[::-1]
    stayed_nearby=True
    for i in np.flatnonzero(times>=activation-1e-8):
        delta=states[i]-start;delta[2]=(delta[2]+np.pi)%(2*np.pi)-np.pi
        nearby=(np.max(abs(delta[3:]))<=JOINT_NEIGHBORHOOD_RAD and
                np.linalg.norm(delta[:2])<=BASE_NEIGHBORHOOD_M and abs(delta[2])<=YAW_NEIGHBORHOOD_RAD)
        stayed_nearby=stayed_nearby and nearby
        if stayed_nearby and tail_margin[i]>=TAIL_MARGIN_RAD and tail_speed[i]<=TAIL_SPEED_RATIO:
            result.update(status='verified',duration_s=max(0.,float(times[i]-activation)),
                end_s=float(times[i]),target_state=states[i].tolist(),target_control=controls[i].tolist(),
                predicted_tail_margin_rad=float(tail_margin[i]),predicted_tail_velocity_ratio=float(tail_speed[i]),
                max_joint_change_rad=float(np.max(abs(delta[3:]))),
                base_change_m=float(np.linalg.norm(delta[:2])),resolution_s=report['step_s'])
            break
    return result


class ConfigurationMonitor:
    def __init__(self):
        self.reset(None,None)

    def reset(self, trajectory_id, report):
        self.trajectory_id=trajectory_id
        self.plan=None if report is None else report.get('preparation')
        self.anchor=None
        self.reached_s=None

    def update(self, state, velocity, t, info, limits=None):
        plan=self.plan
        if not plan or t<plan['start_s']:
            return dict(status='inactive', trajectory_id=self.trajectory_id)
        if self.anchor is None:
            self.anchor=np.asarray(state).copy()
        status='not_found' if plan['status']!='verified' else 'preparing'
        reserve_ok=True
        actual_reserve=None
        if limits is not None:
            q=np.asarray(state)[3:];u=np.asarray(velocity)[2:]
            actual_reserve=float(min(np.min(np.minimum(JOINT_UPPER-q,q-JOINT_LOWER)),
                np.min(np.where(u>=0,JOINT_UPPER-q,q-JOINT_LOWER)-u*u/(2*limits.acceleration[2:]))))
            reserve_ok=(actual_reserve>=TAIL_MARGIN_RAD and
                np.max(abs(np.asarray(velocity))/limits.velocity)<=TAIL_SPEED_RATIO and
                np.max(abs(limits.wheels@np.asarray(velocity)))/limits.wheel_speed<=TAIL_SPEED_RATIO)
        delta=np.asarray(state)-self.anchor
        nearby=(np.max(abs(delta[3:]))<=JOINT_NEIGHBORHOOD_RAD and
                np.linalg.norm(delta[:2])<=BASE_NEIGHBORHOOD_M and abs(delta[2])<=YAW_NEIGHBORHOOD_RAD)
        if plan['status']=='verified':
            expected=np.asarray(plan['target_state'])
            # Only record measured arrival, never predicted arrival as success.
            arrived=(t>=plan['end_s']-1e-8 and t<=plan['end_s']+.12 and nearby and reserve_ok and
                np.max(abs(np.asarray(state)[2:]-expected[2:]))<=.06 and
                np.linalg.norm(np.asarray(state)[:2]-expected[:2])<=.035 and
                np.max(abs(np.asarray(velocity)-plan['target_control']))<=.15 and
                info.get('tpr_status')!='degraded' and info.get('position_error',1.)<=.015 and
                info.get('orientation_error',1.)<=.035)
            if self.reached_s is None and arrived:
                self.reached_s=float(t)
            if self.reached_s is not None:
                status='maintaining' if info.get('tpr_status')!='degraded' and reserve_ok else 'maintenance_degraded'
            elif t>plan['end_s']+.12 or not nearby:
                status='preparation_not_confirmed'
        return dict(status=status,trajectory_id=self.trajectory_id,
            planned_duration_s=plan['duration_s'], measured_reached_s=self.reached_s,
            measured_duration_s=None if self.reached_s is None else self.reached_s-plan['start_s'],
            nearby_during_preparation=nearby if self.reached_s is None else None,
            measured_stopping_margin_rad=actual_reserve, measured_reserve_sufficient=reserve_ok,
            camera_clock='unchanged', target_tracking_m=info.get('position_error'))
