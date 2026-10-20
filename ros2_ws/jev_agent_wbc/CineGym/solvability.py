"""Constructive, offline timing witnesses for scene authoring only.

The clocks preserve every authored phase boundary. They are never supplied to
the evaluated Agent. Failure of this restricted family is not infeasibility.
"""
import itertools
import numpy as np
from jev_agent_wbc.whole_body_qp import Limits
from jev_agent_wbc.kc_svsdf import advance,CollisionGeometry
from jev_agent_wbc.self_collision import SelfCollision
from jev_agent_wbc.jvs_kinematics import JOINT_LOWER,JOINT_UPPER
from .authoring import audit_nominal


def timing_pool(reference,scene,nominal_times,nominal_states,selected_clocks=None):
    dt=.02;times=np.arange(int(np.ceil(nominal_times[-1]/dt))+1)*dt
    geometry=CollisionGeometry();own=SelfCollision();limits=Limits()
    clocks=[ab for ab in itertools.product((-.8,-.4,0.,.4,.8),repeat=2)
           if ab[0]**2+ab[1]**2<=.85**2]
    clocks.sort(key=lambda ab:abs(ab[0])+abs(ab[1]))
    count=len(reference.task['shot_phase'])
    modes=[tuple([ab]*count) for ab in clocks]
    if count==2:
        modes += [(a,b) for a in clocks for b in clocks if a!=b]
    if selected_clocks is not None:
        modes=[tuple(tuple(ab) for ab in selected_clocks)]
        if len(modes[0])!=count or any(ab not in clocks for ab in modes[0]):
            raise ValueError('invalid authoring witness clocks')
    pool=[]
    for mode in modes:
        progress=times.copy()
        for phase,(a,b) in zip(reference.task['shot_phase'],mode):
            mask=(times>=phase['start_s'])&(times<=phase['end_s'])
            span=phase['end_s']-phase['start_s'];u=(times[mask]-phase['start_s'])/span
            progress[mask]=phase['start_s']+span*(u+a*np.sin(2*np.pi*u)/(2*np.pi)+b*(1-np.cos(2*np.pi*u))/(2*np.pi))
        requested=np.column_stack([np.interp(progress,nominal_times,nominal_states[:,j]) for j in range(9)])
        delta=np.diff(requested,axis=0);yaw=.5*(requested[1:,2]+requested[:-1,2])
        controls=np.column_stack([np.sum(delta[:,:2]*np.c_[np.cos(yaw),np.sin(yaw)],axis=1)/np.sinc(delta[:,2]/(2*np.pi)),delta[:,2:]])/dt
        # Match runner.py: zero-order-held velocity commands with bounded
        # discrete command slew. This is a CPU-model witness, not a claim
        # about an unmodelled physical servo's acceleration response.
        changes=np.diff(np.vstack([np.zeros(8),controls,np.zeros(8)]),axis=0)
        if (np.any(abs(controls)>limits.velocity+1e-7) or
            np.any(abs(changes)>dt*limits.acceleration+1e-7) or
            np.any(abs(controls@limits.wheels.T)>limits.wheel_speed+1e-7) or
            np.any(abs(changes@limits.wheels.T)>dt*limits.wheel_acceleration+1e-7)):continue
        states=integrate_controls(scene['initial_state'],controls,dt)
        if np.any(states[:,3:]<JOINT_LOWER+limits.position_margin) or np.any(states[:,3:]>JOINT_UPPER-limits.position_margin):continue
        self_speed=float(np.max(2*abs(controls[:,2:])@geometry.reach[1:]))
        self_lower=float(own.distances_sequence(states).min())-self_speed*dt/2
        if self_lower<own.margin:continue
        speed=float(np.max(abs(controls[:,0])+abs(controls[:,1:])@geometry.reach))
        pool.append(dict(clock=[list(ab) for ab in mode],times=times,states=states,controls=controls,
            progress=progress,robot_speed_bound=speed,self_lower=self_lower,
            frames=geometry.frame_arrays(states,scene['base_z'])))
    return pool


def integrate_controls(initial,controls,dt):
    """Vectorized exact differential-drive zero-order-hold integration."""
    initial=np.asarray(initial,float);controls=np.asarray(controls,float)
    angles=controls[:,1]*dt
    heading=initial[2]+np.r_[0.,np.cumsum(angles)[:-1]]+.5*angles
    travel=controls[:,0]*dt*np.sinc(angles/(2*np.pi))
    increments=np.column_stack([travel*np.cos(heading),travel*np.sin(heading),dt*controls[:,1:]])
    return np.vstack([initial,initial+np.cumsum(increments,axis=0)])


def safe_clock_indices(pool,indices,curve,radius,obstacle_speed,obstacle=None):
    geometry=CollisionGeometry();safe=[]
    for index in indices:
        item=pool[index];times=item['times'];p,r=item['frames']
        points=curve(np.clip(times,curve.x[0],curve.x[-1]))
        from jev_agent_wbc.runtime_obstacles import local_cover
        offsets,radii=local_cover(obstacle) if obstacle is not None else (np.zeros((1,3)),np.array([radius]))
        clearance=float((geometry._primitive_arrays(p,r,points[:,None]+offsets).min(axis=-1)-radii).min())
        # Each time lies within dt/2 of a sampled endpoint. Rigid primitive
        # motion is Lipschitz bounded by executable controls and link reaches.
        lower=clearance-(item['robot_speed_bound']+obstacle_speed)*np.max(np.diff(times))/2
        if lower>=.025:safe.append(index)
    return safe


def finish_witness(reference,scene,pool,indices,curves):
    from .obstacles import motion_bounds
    geometry=CollisionGeometry()
    for index in indices:
        item=pool[index]
        acceptance=audit_nominal(reference,scene,item['times'],item['states'],curves)
        if not acceptance['accepted']:continue
        lower=np.inf
        for obstacle,curve in zip(scene['obstacles'],curves):
            p,r=item['frames'];points=curve(np.clip(item['times'],curve.x[0],curve.x[-1]))
            from jev_agent_wbc.runtime_obstacles import local_cover
            offsets,radii=local_cover(obstacle)
            gap=float((geometry._primitive_arrays(p,r,points[:,None]+offsets).min(axis=-1)-radii).min())
            lower=min(lower,gap-(item['robot_speed_bound']+motion_bounds(curve)['speed_m_s'])*np.max(np.diff(item['times']))/2)
        if lower<.025 or item['self_lower']<.004:continue
        return dict(status='certified_cpu_timing_witness',clock=item['clock'],
            continuous_external_lower_m=lower,continuous_self_lower_m=item['self_lower'],
            shot_accepted=True,visibility_includes_obstacles=True,
            phase_boundaries_preserved=True,control_step_s=float(np.max(np.diff(item['times']))),
            actuator_model='CPU_constant_velocity_command_with_bounded_discrete_slew',
            scope='Offline CPU control witness matching runner.py; truth is used only for authoring and never sent to the Agent; not PhysX or hardware certification')
    return None
