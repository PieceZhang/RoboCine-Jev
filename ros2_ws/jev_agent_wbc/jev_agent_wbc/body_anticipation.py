"""Observed-motion preparation for nonholonomic body clearance.

These preferences guide body preparation and geometric path tracking. They
are not collision certificates. No scene names, obstacle labels, future
keyframes or evaluation acceptance thresholds enter the policy.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER


def barrier_gains(collision, indices):
    """Consistent local barrier policy for prediction and execution.

    These are sufficient velocity inequalities, not changes to separation or
    actuator limits. Every accepted command still needs a nonlinear sweep.
    """
    incoming=(collision.track_ids[indices]>0)&(np.linalg.norm(collision.velocities[indices],axis=1)>.04)
    platform=collision.geometry.shape_bodies==0
    return np.where(incoming[:,None],np.where(platform[None,:],12.,6.),2.)


def phase_axis(original, phase, position):
    motion=phase['motion'].lower()
    if motion in ('movein','moveout'):
        axis=position-original.subject
        return axis/max(np.linalg.norm(axis),1e-9)*(-1 if motion=='movein' else 1)
    if 'truck' in motion:
        axis=original.sample(phase['start_s'])[1][:,1]
        delta=original.sample(phase['end_s'])[0]-original.sample(phase['start_s'])[0]
        return axis*np.sign(delta@axis)
    if 'pedestal' in motion:return np.array([0.,0.,1. if motion=='pedestal_up' else -1.])
    if motion=='orbit':
        a=original.sample(phase['start_s'])[0]-original.subject
        b=original.sample(phase['end_s'])[0]-original.subject
        sign=np.sign(np.cross(a[:2],b[:2]))
        p=position-original.subject;axis=np.array([-p[1],p[0],0.])*sign
        return axis/max(np.linalg.norm(axis),1e-9)
    return None


def tracking_progress(controller, reference, position, at):
    """Follow achieved forward progress instead of reversing to a stale clock.

    This only adjusts feedback inside the current main phase. The phase order,
    authored deadline and final pose remain unchanged; transitions are separate.
    """
    original=getattr(reference,'original',reference)
    phase=original.phase(at)
    if phase['motion'].lower()=='transition':return at
    key=(id(reference),phase['name'])
    if getattr(controller,'progress_cache_key',None)!=key:
        times=np.linspace(phase['start_s'],phase['end_s'],201)
        from .intent_paths import positions_at
        controller.progress_cache=(times,positions_at(reference,times))
        controller.progress_cache_key=key
        controller.achieved_progress=phase['start_s']
    times,positions=controller.progress_cache
    motion=phase['motion'].lower()
    feature=None
    if motion in ('movein','moveout'):
        sign=-1. if motion=='movein' else 1.
        features=sign*np.linalg.norm(positions-original.subject,axis=1)
        feature=sign*np.linalg.norm(position-original.subject)
    elif motion=='orbit':
        offsets=positions-original.subject
        angles=np.unwrap(np.arctan2(offsets[:,1],offsets[:,0]))
        sign=np.sign(angles[-1]-angles[0])
        angle=np.arctan2(position[1]-original.subject[1],position[0]-original.subject[0])
        angle=angles[0]+np.arctan2(np.sin(angle-angles[0]),np.cos(angle-angles[0]))
        features=sign*angles;feature=sign*angle
    elif 'truck' in motion or 'pedestal' in motion:
        axis=phase_axis(original,phase,position)
        features=positions@axis;feature=position@axis
    if feature is not None and np.ptp(features)>1e-8:
        # Progress is the directed visual feature. A sideways avoidance bow
        # must not falsely advance a pedestal's height or a truck's parallax.
        monotone=np.maximum.accumulate(features)
        unique,indices=np.unique(monotone,return_index=True)
        at_segment=float(np.interp(feature,unique,times[indices]))
        # Flat portions carry no semantic progress information. Millimetric
        # height/tracking error must not skip a horizontal approach leg.
        segment_lengths=np.linalg.norm(np.diff(positions,axis=0),axis=1)
        flat=(abs(np.diff(monotone))<np.maximum(1e-8,.02*segment_lengths))&(abs(monotone[:-1]-feature)<.004)
        if np.any(flat):
            ids=np.flatnonzero(flat);chords=positions[ids+1]-positions[ids]
            fractions=np.clip(np.sum((position-positions[ids])*chords,axis=1)/np.maximum(np.sum(chords**2,axis=1),1e-16),0.,1.)
            chosen=int(np.argmin(np.linalg.norm(positions[ids]+fractions[:,None]*chords-position,axis=1)))
            at_segment=float(times[ids[chosen]]+fractions[chosen]*(times[ids[chosen]+1]-times[ids[chosen]]))
    else:
        chords=np.diff(positions,axis=0)
        fractions=np.clip(np.sum((position-positions[:-1])*chords,axis=1)/np.maximum(np.sum(chords**2,axis=1),1e-16),0.,1.)
        projected=positions[:-1]+fractions[:,None]*chords
        index=int(np.argmin(np.linalg.norm(projected-position,axis=1)))
        at_segment=times[index]+fractions[index]*(times[index+1]-times[index])
    controller.achieved_progress=max(controller.achieved_progress,float(at_segment))
    return min(phase['end_s'],max(at,controller.achieved_progress))


def local_forward_target(controller,reference,at):
    """Do not aim across a curve's turn when execution falls behind its clock."""
    times,points=controller.progress_cache
    achieved=controller.achieved_progress
    if at<=achieved+.1:return at
    chords=np.diff(points,axis=0);lengths=np.linalg.norm(chords,axis=1)
    valid=np.flatnonzero(lengths>1e-7)
    if not len(valid):return at
    i=int(valid[np.argmin(abs((times[valid]+times[valid+1])/2-achieved))])
    here=reference.sample(achieved)[0];target=reference.sample(at)[0]
    if (target-here)@chords[i]>=0.:return at
    arc=np.r_[0.,lengths.cumsum()]
    lookahead=np.interp(achieved,times,arc)+.08
    return min(at,float(np.interp(lookahead,arc,times)))


def path_frame(projector, reference, position, at):
    """Five local path constraints; forward progress is the free coordinate.

    Camera angular change is coupled to arc length, so an orbit can advance
    without being constrained to its old clock's viewing direction.
    """
    original=getattr(reference,'original',reference);phase=original.phase(at)
    if phase['motion'].lower()=='transition':return None
    tracking_progress(projector,reference,position,at)
    progress=projector.achieved_progress
    times,points=projector.progress_cache
    chord=np.diff(points,axis=0);length=np.linalg.norm(chord,axis=1)
    valid=np.flatnonzero(length>1e-7)
    if not len(valid):return None
    index=int(valid[np.argmin(abs((times[valid]+times[valid+1])/2-progress))])
    tangent=chord[index]/length[index]
    _,_,basis=np.linalg.svd(tangent[None,:]);normals=basis[1:]
    ra=reference.sample(times[index])[1];rb=reference.sample(times[index+1])[1]
    angular_per_m=Rotation.from_matrix(rb@ra.T).as_rotvec()/length[index]
    transform=np.zeros((5,6));transform[:2,:3]=normals
    transform[2:,:3]=-np.outer(angular_per_m,tangent);transform[2:,3:]=np.eye(3)
    p,r,_=reference.sample(progress)
    return p,r,transform,tangent,progress


def preview_seed(controller, reference, state, velocity, at, step, nodes, z=0., rotation=None):
    """A turning seed exposes nonholonomic motion to the local linearization.

    It is never executed or certified directly. The predictive QP and its
    nonlinear actuator/collision checks must accept a complete refinement.
    """
    from types import SimpleNamespace
    from .whole_body_qp import whole_body_kinematics
    from .kc_svsdf import advance
    projector=SimpleNamespace();x=state.copy();old=velocity.copy();controls=[]
    memory={key:(value[0].copy(),value[1].copy()) for key,value in getattr(controller,'body_escape_memory',{}).items()}
    original=getattr(reference,'original',reference)
    for k in range(nodes):
        stamp=min(reference.duration,at+(k+1)*step)
        p,r,J=whole_body_kinematics(x,z,rotation)
        frame=path_frame(projector,reference,p,stamp)
        if frame is None:return np.tile(velocity,(nodes,1))
        target,orientation,T,axis,_=frame
        future=controller.collision.at_time((k+1)*step)
        preference,_=base_preference(future,x,old,z,rotation,axis,memory)
        u=np.zeros(8)
        if preference is not None:u[:2]=preference
        else:
            feedforward=reference.sample(stamp)[2][:2]
            u[0]=feedforward@np.array([np.cos(x[2]),np.sin(x[2])])
        error=np.r_[target-p,Rotation.from_matrix(orientation@r.T).as_rotvec()]
        M=T@J[:,2:];right=2.*T@error-T@J[:,:2]@u[:2]
        u[2:]=np.linalg.lstsq(M,right,rcond=1e-5)[0]
        lim=controller.limits
        u=np.clip(u,np.maximum(-lim.velocity,old-step*lim.acceleration),np.minimum(lim.velocity,old+step*lim.acceleration))
        u[2:]=np.clip(u[2:],(JOINT_LOWER+.02-x[3:])/step,(JOINT_UPPER-.02-x[3:])/step)
        controls.append(u);x=advance(x,.5*(old+u),step);old=u
    return np.array(controls)


def base_preference(collision, state, velocity, z=0., rotation=None, task_axis=None, escape_memory=None,camera_locked=True):
    moving=np.flatnonzero((collision.track_ids>0)&(np.linalg.norm(collision.velocities[:,:2],axis=1)>.04))
    info=dict(active=False,source='observed_motion_and_robot_geometry',threats=[])
    if not len(moving):return None,info
    direction=np.array([np.cos(state[2]),np.sin(state[2])])
    # Predict the mover relative to the current base position, rather than to
    # a rapidly changing escape command; otherwise the preference chatters.
    centers=collision.centers[moving];speeds=collision.velocities[moving]
    relative=centers[:,:2]-state[:2]
    closest=-np.sum(relative*speeds[:,:2],axis=1)/np.maximum(np.sum(speeds[:,:2]**2,axis=1),1e-8)
    lookahead=np.linspace(0.,2.5,11)
    predicted=centers[:,None,:]+speeds[:,None,:]*lookahead[None,:,None]
    distances,gradients=collision.geometry.primitive_distances(state,predicted.reshape(-1,3),z,rotation)
    distances=distances.reshape(len(moving),len(lookahead),-1)
    # Predict the full corridor, but request platform escape only for the
    # chassis or proximal witnesses with little independent arm authority.
    # Treating every upper-arm threat as a chassis threat needlessly fights
    # a camera's already clear lateral bow.
    gradients=gradients.reshape(len(moving),len(lookahead),-1,9)
    bodies=collision.geometry.shape_bodies[None,None,:]
    base=(bodies==0)|((bodies<=2)&(np.linalg.norm(gradients[...,3:],axis=-1)<.05))
    clear=np.where(base,distances-collision.radii[moving,None,None],np.inf)
    minima=clear.reshape(len(moving),-1).argmin(axis=1)
    when,links=np.unravel_index(minima,clear.shape[1:])
    clearances=clear[np.arange(len(moving)),when,links]
    ttc=lookahead[when]
    threatened=(ttc>.05)&(clearances<.08)
    if not threatened.any():return None,info
    indices=np.flatnonzero(threatened)
    i=int(indices[np.argmin(clearances[indices]+.03*ttc[indices])])
    # Escape to the side of the projected crossing corridor, not directly
    # away from its current surface. The latter can select the wrong branch
    # when a fast object is still approaching nearly perpendicular to drive.
    corridor_offset=state[:2]-(centers[i,:2]+speeds[i,:2]*max(0.,closest[i]))
    normal=corridor_offset.copy()
    if np.linalg.norm(normal)<.03:
        normal=np.array([-speeds[i,1],speeds[i,0]])
        if normal@direction<0:normal=-normal
    normal/=max(np.linalg.norm(normal),1e-9)
    track_id=int(collision.track_ids[moving[i]])
    moving_direction=speeds[i,:2]/max(np.linalg.norm(speeds[i,:2]),1e-9)
    if escape_memory is not None:
        old=escape_memory.get(track_id)
        if old is not None and old[1]@moving_direction>.85:
            if normal@old[0]<0.:normal=-normal
        else:
            # With useful warning time, crossing to the other side of the
            # future corridor can preserve arm reach for a directed shot.
            # Commit to that side until the observed mover changes direction;
            # recomputing the nearest side each tick causes late reversals.
            # Cross the mover's corridor only when the extra traverse is small.
            # Direction compatibility cannot justify a half-metre crossing
            # when stopping/turning authority is needed in the next seconds.
            extra_traverse=2.*abs(float(corridor_offset@normal))
            if task_axis is not None and ttc[i]>1.5 and extra_traverse<.20 and normal@np.asarray(task_axis)[:2]<-.25:
                normal=-normal
            escape_memory[track_id]=(normal.copy(),moving_direction.copy())
    crossing_distance=max(0.,-2.*float(corridor_offset@normal))
    corridor_side=normal.copy()
    if task_axis is not None and np.linalg.norm(np.asarray(task_axis)[:2])>.5:
        task_direction=np.asarray(task_axis)[:2].copy()
        task_direction/=np.linalg.norm(task_direction)
        # Advancing the camera's own planar action preserves arm reach while
        # escaping. Keep a transverse component away from the observed mover.
        if task_direction@normal>-.25:
            following=task_direction@speeds[i,:2]>.5*np.linalg.norm(speeds[i,:2])
            normal=normal+(.5 if following else 3.)*task_direction
            normal/=np.linalg.norm(normal)
    angle=np.arctan2(normal[1],normal[0])
    turns=np.arctan2(np.sin(np.array([angle,angle+np.pi])-state[2]),np.cos(np.array([angle,angle+np.pi])-state[2]))
    shoulder=state[3]+state[8]-turns
    penalty=np.maximum(JOINT_LOWER[0]+JOINT_LOWER[5]+.15-shoulder,0.)+np.maximum(shoulder-JOINT_UPPER[0]-JOINT_UPPER[5]+.15,0.)
    if not camera_locked:penalty[:]=0.
    branch=int(np.argmin(abs(turns)+15*penalty))
    turn=float(turns[branch]);sign=1. if branch==0 else -1.
    required_speed=max(.03,(.18-clearances[i]+crossing_distance)/max(ttc[i],.4))
    required_speed/=max(.3,float(normal@corridor_side))
    along=float(speeds[i,:2]@normal)
    if along>.5*np.linalg.norm(speeds[i,:2]):
        # When advancing in the same direction as an approaching mover,
        # displacement/TTC alone lets it catch the chassis from behind.
        required_speed=max(required_speed,along+.07)
    speed=sign*min(.40,required_speed)
    omega=float(np.clip(1.5*turn,-.65,.65))
    info.update(active=True,track_id=int(collision.track_ids[moving[i]]),
        escape_side_world=normal.tolist(),
        corridor_crossing_extra_m=crossing_distance,
        threatened_link=collision.geometry.shapes[int(links[i])][1],
        predicted_base_clearance_m=float(clearances[i]),closest_approach_s=float(ttc[i]),
        drive_target_m_s=float(speed),yaw_target_rad_s=omega,
        threats=[dict(track_id=int(collision.track_ids[moving[j]]),clearance_m=float(clearances[j]),
                      closest_approach_s=float(ttc[j])) for j in indices])
    return np.array([speed,omega]),info
