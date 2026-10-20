"""Causal short-horizon passage-clock preference, never a safety certificate."""
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from .whole_body_qp import whole_body_kinematics
from .kc_svsdf import advance
from .body_anticipation import path_frame


class RemainingReference:
    def __init__(self,reference,start):
        self.reference=reference;self.start=start;self.duration=reference.duration-start
    def sample(self,t):return self.reference.sample(t+self.start)


def build_guide(reference,state,start,z):
    from .nominal import nominal_rollout
    times,states,failure=nominal_rollout(RemainingReference(reference,start),state,z)
    if failure:return None
    delta=np.diff(states,axis=0);yaw=.5*(states[1:,2]+states[:-1,2])
    controls=np.column_stack([np.sum(delta[:,:2]*np.c_[np.cos(yaw),np.sin(yaw)],axis=1)/
        np.sinc(delta[:,2]/(2*np.pi)),delta[:,2:]])/np.diff(times)[:,None]
    return dict(times=times+start,states=states,controls=controls)


def guide_state(guide,stamp):
    return np.array([np.interp(stamp,guide['times'],guide['states'][:,j]) for j in range(9)])


def reachable_guide_step(guide,state,previous,stamp,rate,dt,limits,expected=None,feedforward=None):
    """Propagate from measured state; the nominal allocation cannot teleport it."""
    expected=guide_state(guide,stamp) if expected is None else expected
    delta=expected-state
    delta[2]=np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
    feedforward=(np.array([np.interp(stamp,guide['times'][:-1],guide['controls'][:,j]) for j in range(8)])
                 if feedforward is None else feedforward)*rate
    correction=np.r_[delta[:2]@np.array([np.cos(state[2]),np.sin(state[2])]),delta[2:]]
    desired=np.clip(feedforward+2.*correction,-limits.velocity,limits.velocity)
    desired[:2]/=max(1.,float(np.max(abs(limits.wheels@desired)))/limits.wheel_speed)
    change=np.clip(desired-previous,-limits.acceleration*dt,limits.acceleration*dt)
    change[:2]/=max(1.,float(np.max(abs(limits.wheels@change)))/(limits.wheel_acceleration*dt))
    command=previous+change
    return advance(state,command,dt),command


def poll_guide(c,reference,state,at,z):
    if getattr(c,'clock_guide_reference',None) is not reference:
        c.clock_guide_reference=reference;c.clock_guide=None;c.clock_guide_future=None
    pending=c.clock_guide_future
    if pending is not None and pending.done():
        try:c.clock_guide=pending.result()
        except Exception:c.clock_guide=None
    elif pending is None and getattr(c.reconfiguration,'pool',None) is not None:
        c.clock_guide_future=c.reconfiguration.pool.submit(build_guide,reference,state.copy(),at,z)
    return c.clock_guide


def update(c,reference,state,velocity,at,z,rotation=None):
    guide=poll_guide(c,reference,state,at,z)
    last=getattr(c,'clock_observation',None)
    if last is not None and at-last['at']<.1-1e-8:return last
    collision=c.collision
    movers=np.flatnonzero(collision.track_ids>0)
    result=dict(at=float(at),active=False,rate=1.,source='observed_full_body_passage_clock')
    if not len(movers) or reference.duration-at<.7:
        c.clock_observation=result;return result
    p,r,_=whole_body_kinematics(state,z,rotation)
    projector=SimpleNamespace()
    frame=path_frame(projector,reference,p,at)
    if frame is None:
        c.clock_observation=result;return result
    progress=frame[-1];phase=getattr(reference,'original',reference).phase(at)
    projected_progress=float(progress)
    if last is not None and last.get('active') and last.get('phase_start')==phase['start_s']:
        # Keep a bounded smooth command-clock lead, but anchor every update to measured
        # progress. Forecasts below still start at the measured configuration;
        # the command target may lead to recover physical tracking lag.
        progress=max(progress,min(progress+1.0,last['progress']+last['rate']*(at-last['at'])))
    progress=min(phase['end_s'],progress)
    result.update(phase_start=phase['start_s'],projected_progress=projected_progress)
    # Remaining authored time guides speed recovery without changing deadlines.
    desired=np.clip((phase['end_s']-progress)/max(.3,phase['end_s']-at-.3),.3,2.5)
    oldrate=1. if last is None else last['rate']
    elapsed=.1 if last is None else at-last['at']
    from .clock_profiles import profiles
    horizon=min(4.,phase['end_s']-at);nodes=12;step=horizon/nodes
    if step<=0.:c.clock_observation=result;return result
    schedules=profiles(oldrate,desired,horizon,nodes,elapsed)
    forecasts=[];end_progress=[];executed_progress=[]
    for schedule in schedules:
        x=state.copy();old=velocity.copy();trace=[]
        clocks=np.minimum(phase['end_s'],progress+np.cumsum(schedule['increments']))
        if guide is not None:
            expected=np.column_stack([np.interp(clocks,guide['times'],guide['states'][:,j]) for j in range(9)])
            feedforwards=np.column_stack([np.interp(clocks,guide['times'][:-1],guide['controls'][:,j]) for j in range(8)])
        for k,(stamp,rate) in enumerate(zip(clocks,schedule['rates'])):
            if guide is not None:
                x,old=reachable_guide_step(guide,x,old,stamp,rate,step,c.limits,expected[k],feedforwards[k])
                trace.append(x.copy());continue
            target,orientation,twist=reference.sample(stamp)
            cp,cr,j=whole_body_kinematics(x,z,rotation)
            error=np.r_[target-cp,Rotation.from_matrix(orientation@cr.T).as_rotvec()]
            command=np.linalg.lstsq(j,twist*rate+2.*error,rcond=1e-4)[0]
            command=np.clip(command,np.maximum(-c.limits.velocity,old-step*c.limits.acceleration),
                np.minimum(c.limits.velocity,old+step*c.limits.acceleration))
            x=advance(x,command,step);old=command;trace.append(x.copy())
        forecasts.extend(trace);end_progress.append(clocks[-1])
        future_rotation=None if rotation is None else Rotation.from_euler('z',x[2]-state[2]).as_matrix()@rotation
        future_position=whole_body_kinematics(x,z,future_rotation)[0]
        from .body_anticipation import tracking_progress
        projection=SimpleNamespace(progress_cache_key=projector.progress_cache_key,
            progress_cache=projector.progress_cache,achieved_progress=phase['start_s'])
        tracking_progress(projection,reference,future_position,at)
        executed_progress.append(float(projection.achieved_progress))
    # Batch the identical geometry operation across candidate clocks. Future
    # base yaw rotates the measured tilt; a constant physical rotation would
    # silently freeze chassis orientation during an avoidance forecast.
    states=np.asarray(forecasts)
    futures=np.tile(np.arange(1,nodes+1)*step,len(schedules))
    centers=collision.centers[movers][None]+futures[:,None,None]*collision.velocities[movers][None]
    radii=collision.radii[movers][None]+futures[:,None]*collision.radius_rates[movers][None]
    rotations=None if rotation is None else Rotation.from_euler('z',states[:,2]-state[2]).as_matrix()@rotation
    clear=(collision.geometry.distances_sequence(states,centers,z,rotations)-radii[:,:,None]).min(axis=(1,2)).reshape(len(schedules),nodes)
    minimum=clear.min(axis=1)
    nominal=next(i for i,p in enumerate(schedules) if p['name']=='1.0')
    if (minimum[nominal]>.16 and not (last and last.get('active'))
            and not getattr(c,'body_adaptation_seen',False)):
        result.update(predicted_clearance_m=float(minimum[nominal]),reason='original schedule clear')
        c.clock_observation=result;return result
    # Time-varying schedules can yield and then recover the original deadline.
    # The terminal progress cost prevents a locally safe indefinite wait.
    target_progress=min(phase['end_s'],progress+desired*horizon)
    debt=target_progress-np.asarray(executed_progress)
    scores=4000.*np.max(np.maximum(0.,.10-clear)**2,axis=1)+2.*debt**2
    scores+=.05*np.array([sum(np.diff(np.r_[oldrate,p['rates']])**2) for p in schedules])
    selected=int(np.argmin(scores));rate=float(schedules[selected]['next_rate'])
    result.update(clock_policy='measured_state_receding_time_profile',
        selected_profile=schedules[selected]['name'],predicted_progress_end_s=float(end_progress[selected]),
        predicted_executed_progress_end_s=float(executed_progress[selected]))
    result.update(active=True,progress=float(progress),rate=rate,
        predicted_clearance_m=float(minimum[selected]),nominal_rate=float(desired),
        forecast_start='measured configuration and velocity',measured_clock_lead_s=float(progress-projected_progress))
    if guide is not None:
        expected=guide_state(guide,progress);delta=expected-state
        delta[2]=np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
        command=np.array([np.interp(progress,guide['times'][:-1],guide['controls'][:,j]) for j in range(8)])*rate
        correction=np.r_[delta[:2]@np.array([np.cos(state[2]),np.sin(state[2])]),delta[2:]]
        result['body_control_preference']=(command+np.clip(2.*correction,-.25*c.limits.velocity,.25*c.limits.velocity)).tolist()
        result['guide_scope']='obstacle-free unverified configuration preference; current execution gates mandatory'
    c.clock_observation=result
    return result
