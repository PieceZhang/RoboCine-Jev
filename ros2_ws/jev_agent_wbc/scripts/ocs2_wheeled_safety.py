"""OCS2 worker transport and hard measured-state execution guards; no optimizer."""
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import itertools
import ctypes
from collections import deque

import numpy as np


class MeasuredLateralYawModel:
    """Causal skid-steer base-origin model, fitted from raw measured velocities.

    The fit changes prediction only. It never creates lateral wheel commands or
    changes physical actuator, collision, tilt, or freshness limits.
    """
    def __init__(self):
        self.samples=deque()
        self.gain_m=0.
        self.fit_rmse_mps=None

    def update(self,stamp_s,lateral_mps,yaw_radps):
        if not np.isfinite([stamp_s,lateral_mps,yaw_radps]).all():
            raise ValueError('finite fresh body velocity required')
        if self.samples and stamp_s<=self.samples[-1][0]:
            raise ValueError('body velocity acquisitions must increase')
        self.samples.append((stamp_s,lateral_mps,yaw_radps))
        while self.samples and self.samples[0][0]<stamp_s-.5:
            self.samples.popleft()
        values=np.asarray(self.samples)
        active=np.abs(values[:,2])>=.05
        if np.count_nonzero(active)>=10:
            velocity=values[active,1];yaw=values[active,2]
            estimate=float(np.clip(velocity@yaw/(yaw@yaw),-.35,.35))
            self.gain_m=.8*self.gain_m+.2*estimate
            self.fit_rmse_mps=float(np.sqrt(np.mean((velocity-self.gain_m*yaw)**2)))
        return dict(gain_m=float(self.gain_m),fit_rmse_mps=self.fit_rmse_mps,
                    acquisition_stamp_s=float(stamp_s),window_s=.5,
                    source='causal_raw_body_velocity_regression')


class NativeWorker:
    """One persistent native process, one in-flight solve, bounded supervision."""
    def __init__(self, binary, backend, model, journal):
        self.stderr = open(Path(journal).with_suffix('.stderr.log'), 'w')
        self.journal = open(journal, 'w', buffering=1)
        self.process = subprocess.Popen([str(binary), str(backend), str(model)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1)
        self.jobs, self.answers = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.busy = False
        self._closing = False
        self.submitted_wall = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            request = self.jobs.get()
            if request is None:
                return
            start = time.monotonic()
            try:
                self.process.stdin.write(json.dumps(request, allow_nan=False)+'\n')
                self.process.stdin.flush()
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError('native OCS2 worker exited without a response')
                result = json.loads(line)
                # Boost property_tree error records are deliberately not accepted as truthy strings.
                if result.get('ok') is not True:
                    raise RuntimeError('native SLQ rejected request: '+str(result))
                if result.get('request_id') != request['request_id'] or result.get('state_stamp_s') != request['state_stamp_s']:
                    raise RuntimeError('native result request/state binding mismatch')
                result.update(transport_wall_s=time.monotonic()-self.submitted_wall,request_wall_s=self.submitted_wall,
                              camera_path_id=request.get('camera_path_id'))
            except Exception as exc:
                result = dict(ok=False, request_id=request['request_id'], state_stamp_s=request['state_stamp_s'],
                              reason=str(exc), transport_wall_s=time.monotonic()-start)
            self.journal.write(json.dumps(dict(request=request, result=result), allow_nan=False)+'\n')
            if self._closing:
                return
            self.answers.put(result)

    def submit(self, request):
        if self.busy:
            return False
        self.busy = True
        self.submitted_wall = time.monotonic()
        self.jobs.put_nowait(request)
        return True

    def poll(self):
        try:
            result = self.answers.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def close(self):
        self._closing = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.)
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.)
        self.journal.close()
        self.stderr.close()


def policy_velocity(policy, stamp, nu, *, ttl_s=.2):
    """Use absolute acquisition time. A stale result never resumes motion."""
    if policy.get('ok') is not True:
        raise ValueError('native policy is unavailable')
    times = np.asarray(policy['times_s'], float)
    states = np.asarray(policy['states'], float)
    if (times.ndim != 1 or len(times)<2 or states.shape != (len(times),9+nu)
            or not np.isfinite(times).all() or not np.isfinite(states).all()
            or np.any(np.diff(times)<=0)):
        raise ValueError('invalid native policy trajectory')
    if stamp < policy['state_stamp_s'] or stamp-policy['state_stamp_s']>ttl_s or stamp<times[0] or stamp>times[-1]:
        raise ValueError('native policy expired or belongs to a future measurement')
    if 'request_wall_s' in policy and time.monotonic()-policy['request_wall_s']>ttl_s:
        raise ValueError('native policy wall-clock lease expired')
    return np.array([np.interp(stamp,times,states[:,9+i]) for i in range(nu)])


class HardCommandGuard:
    """Command slew caps plus a separate measured arm-speed/braking audit.

    The acceleration bound applies to the applied velocity setpoint. It is not
    a claim about contact dynamics or the measured acceleration of a PD drive.
    """
    def __init__(self, omni, lower, upper, dt=.01):
        self.nu = 9 if omni else 8
        self.dt = dt
        self.speed = np.r_[([.22,.22,.35] if omni else [.22,.35]), np.full(6,.65)]
        self.acceleration = np.r_[([.45,.45,.65] if omni else [.45,.65]), np.ones(6)]
        self.lower, self.upper = np.array(lower),np.array(upper)
        self.previous = np.zeros(self.nu)
        self.last_audit = None

    def command(self, desired, q, qd, *, brake=False):
        desired, q, qd = np.asarray(desired,float),np.asarray(q,float),np.asarray(qd,float)
        if desired.shape!=(self.nu,) or q.shape!=(6,) or qd.shape!=(6,) or not np.isfinite(desired).all() or not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError('invalid physical guard state/command')
        a=self.acceleration[-6:]
        # These use real measured qd, independently of the command-rate state.
        outward_upper=np.maximum(qd,0.);outward_lower=np.maximum(-qd,0.)
        upper_margin=self.upper-q-outward_upper*self.dt-outward_upper**2/(2*a)
        lower_margin=q-self.lower-outward_lower*self.dt-outward_lower**2/(2*a)
        configuration_ok=bool(np.all(q>=self.lower) and np.all(q<=self.upper))
        velocity_ratio=float(np.max(np.abs(qd)/self.speed[-6:]))
        braking_ok=bool(np.all(upper_margin>=-1e-10) and np.all(lower_margin>=-1e-10))
        self.last_audit=dict(configuration_ok=configuration_ok,actual_velocity_ok=velocity_ratio<=1.00001,
                             actual_braking_ok=braking_ok,maximum_actual_velocity_ratio=velocity_ratio,
                             minimum_actual_braking_margin_rad=float(min(upper_margin.min(),lower_margin.min())))
        measured_ok=all(self.last_audit[k] for k in ['configuration_ok','actual_velocity_ok','actual_braking_ok'])
        if np.any(q<self.lower-.015) or np.any(q>self.upper+.015):
            raise RuntimeError('measured physical joint outside its original bound')
        lo=np.maximum(-self.speed,self.previous-self.acceleration*self.dt)
        hi=np.minimum(self.speed,self.previous+self.acceleration*self.dt)
        # Reserve actual outward travel during the next measurement interval.
        # A lagging real drive cannot acquire permission from a small command.
        safe=lambda distance: np.maximum(0.,np.sqrt((a*self.dt)**2+2*a*np.maximum(distance-.005,0.))-a*self.dt)
        lo[-6:]=np.maximum(lo[-6:],-safe(q-self.lower-outward_lower*self.dt))
        hi[-6:]=np.minimum(hi[-6:],safe(self.upper-q-outward_upper*self.dt))
        if np.any(lo>hi+1e-10):
            raise RuntimeError('joint braking envelope conflicts with bounded acceleration')
        result=np.clip(np.zeros(self.nu) if brake or not measured_ok else desired,lo,hi)
        self.previous=result.copy()
        return result


class SkidSteerVelocityServo:
    """Measured body-yaw PI mapped to rate-limited physical rolling wheels.

    The OCS2 reference remains a body velocity. This actuator mapping compensates
    wheel/ground slip; its virtual yaw is not a new body reference or measurement.
    """
    def __init__(self, radius, track, dt=.01):
        self.radius,self.track,self.dt=float(radius),float(track),float(dt)
        self.wheel_speed=8.
        self.wheel_slew=(.45+.65*self.track/2)/self.radius
        self.integral=0.
        self.previous=np.zeros(2)
        self.turn_braking=False
        self.last_audit=None

    def command(self, velocity, measured_yaw_rate, *, brake=False, turn_in_place=False):
        v,w=map(float,velocity);measured=float(measured_yaw_rate)
        if not np.isfinite([v,w,measured]).all():raise ValueError('nonfinite skid servo state')
        previous_integral=self.integral
        if turn_in_place:
            if abs(measured)>=.22:self.turn_braking=True
            elif abs(measured)<=.12:self.turn_braking=False
        else:self.turn_braking=False
        effective_brake=bool(brake or self.turn_braking)
        operating_wheel_cap=1. if turn_in_place else self.wheel_speed
        if effective_brake:
            self.integral=0.;virtual=0.;desired=np.zeros(2)
        else:
            error=w-measured
            # The integral is limited before wheel-speed saturation; conditional
            # integration prevents continued wind-up at the virtual actuator cap.
            candidate=float(np.clip(self.integral+error*self.dt,-.65,.65))
            raw=w+3.*error+6.*candidate
            actuator_cap=min(4.,max(0.,(operating_wheel_cap*self.radius-abs(v))*2/self.track))
            virtual=float(np.clip(raw,-actuator_cap,actuator_cap))
            if abs(raw)<=actuator_cap or error*raw<0:self.integral=candidate
            desired=np.array([v-virtual*self.track/2,v+virtual*self.track/2])/self.radius
        desired=np.clip(desired,-operating_wheel_cap,operating_wheel_cap)
        result=np.clip(desired,self.previous-self.wheel_slew*self.dt,self.previous+self.wheel_slew*self.dt)
        if not effective_brake and (w-measured)*((desired[1]-desired[0])-(result[1]-result[0]))>1e-12:
            self.integral=previous_integral
        self.last_audit=dict(target_body_velocity=[v,w],measured_yaw_rate=measured,
                             virtual_actuator_yaw_rate=virtual,integral=self.integral,
                             wheel_velocity_targets=result.tolist(),wheel_velocity_cap=self.wheel_speed,
                             wheel_velocity_slew_cap=self.wheel_slew,brake=effective_brake,
                             turn_in_place=bool(turn_in_place),turn_braking=self.turn_braking,
                             operating_wheel_cap=operating_wheel_cap)
        self.previous=result.copy()
        return result


def reference_window(absolute_times, serialized_poses, stamp, horizon):
    """An exact contiguous subset of original knots, including interpolation ends."""
    times=np.asarray(absolute_times,float)
    if len(times)!=len(serialized_poses) or len(times)<2 or np.any(np.diff(times)<=0):
        raise ValueError('invalid immutable reference knots')
    first=max(0,int(np.searchsorted(times,stamp,side='right'))-1)
    last=min(len(times),int(np.searchsorted(times,stamp+horizon,side='left'))+1)
    if times[first]>stamp or times[last-1]<stamp+horizon-1e-9:
        raise ValueError('original reference does not cover requested horizon')
    return times[first:last].tolist(),serialized_poses[first:last],(first,last)


def terminal_reference_window(absolute_times, serialized_poses, stamp, horizon):
    """Original knots plus an explicit constant continuation of the same endpoint.

    This never changes the public trajectory, its timestamps or tolerances.
    It only specifies the prediction reference after the public task has ended.
    """
    times=np.asarray(absolute_times,float)
    if (len(times)!=len(serialized_poses) or len(times)<2 or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0) or not np.isfinite([stamp,horizon]).all()
            or horizon<=0 or stamp<times[0]):
        raise ValueError('invalid original reference or terminal prediction window')
    first=max(0,min(len(times)-1,int(np.searchsorted(times,stamp,side='right'))-1))
    last=min(len(times),int(np.searchsorted(times,stamp+horizon,side='left'))+1)
    selected_times=times[first:last].tolist()
    selected_poses=list(serialized_poses[first:last])
    if stamp+horizon>times[-1]+1e-9:
        selected_times.append(float(stamp+horizon))
        selected_poses.append(serialized_poses[-1])
    metadata=dict(mode='constant_original_endpoint_after_end',original_end_s=float(times[-1]),
                  T_world_original_endpoint=serialized_poses[-1])
    return selected_times,selected_poses,(first,last),metadata


def camera_task_metrics(position, orientation, final_position, final_orientation,
                        position_tolerance, orientation_tolerance):
    """Keep RMS, pointwise errors and the final task gate explicitly distinct."""
    p,o=np.asarray(position,float),np.asarray(orientation,float)
    values=np.r_[p,o,[final_position,final_orientation,position_tolerance,orientation_tolerance]]
    if (p.ndim!=1 or o.shape!=p.shape or not p.size or not np.isfinite(values).all()
            or np.any(p<0) or np.any(o<0) or final_position<0 or final_orientation<0
            or position_tolerance<=0 or orientation_tolerance<=0):
        raise ValueError('invalid measured camera-task errors or tolerance')
    return dict(position_rmse_m=float(np.sqrt(np.mean(p**2))),position_max_m=float(p.max()),
                orientation_rmse_rad=float(np.sqrt(np.mean(o**2))),orientation_max_rad=float(o.max()),
                position_exceeding_samples=int(np.count_nonzero(p>position_tolerance)),
                orientation_exceeding_samples=int(np.count_nonzero(o>orientation_tolerance)),
                position_sample_count=int(p.size),final_position_error_m=float(final_position),
                final_orientation_error_rad=float(final_orientation),
                final_task_tolerance_pass=bool(final_position<position_tolerance and final_orientation<orientation_tolerance))


def obb_separation(center_a, axes_a, extents_a, center_b, axes_b, extents_b):
    """Conservative separating-axis lower bound; bbox overlap is not mesh contact."""
    axes=[*axes_a.T,*axes_b.T]
    axes += [np.cross(a,b) for a in axes_a.T for b in axes_b.T]
    directions=np.asarray([a/np.linalg.norm(a) for a in axes if np.linalg.norm(a)>1e-9])
    delta=np.asarray(center_b)-center_a
    values=np.abs(directions@delta)-np.abs(directions@axes_a)@extents_a-np.abs(directions@axes_b)@extents_b
    return float(np.max(values))


def convex_separation(vertices_a, vertices_b, iterations=32):
    """GJK supporting-plane lower bound between convex outer hulls.

    Returning zero means overlap/unknown, never exact mesh penetration. The
    separating support plane provides a conservative bound even before convergence.
    """
    A,B=np.asarray(vertices_a,float),np.asarray(vertices_b,float)
    def support(direction):
        return A[np.argmax(A@direction)]-B[np.argmin(B@direction)]
    direction=A.mean(axis=0)-B.mean(axis=0)
    if np.linalg.norm(direction)<1e-9:direction=np.array([1.,0.,0.])
    simplex=[support(-direction)];closest=simplex[0];bound=0.
    for _ in range(iterations):
        norm=np.linalg.norm(closest)
        if norm<1e-10:return 0.
        point=support(-closest);bound=max(0.,float(closest@point/norm))
        if norm-bound<1e-8:return bound
        candidates=[*simplex,point];best=None
        for count in range(1,min(4,len(candidates))+1):
            for subset in itertools.combinations(range(len(candidates)),count):
                points=np.array([candidates[i] for i in subset])
                matrix=np.block([[points@points.T,np.ones((count,1))],[np.ones((1,count)),np.zeros((1,1))]])
                weights=np.linalg.lstsq(matrix,np.r_[np.zeros(count),1.],rcond=None)[0][:count]
                if np.min(weights)<-1e-9:continue
                weights=np.maximum(weights,0.);weights/=weights.sum();p=weights@points;squared=float(p@p)
                if best is None or squared<best[0]:best=(squared,p,[candidates[i] for i,w in zip(subset,weights) if w>1e-10])
        if best is None:raise RuntimeError('GJK simplex failed')
        _,closest,simplex=best
    return bound


def add_convex_vertices(stage, geometry):
    """Convex outer hull of the actual physical collision mesh, in body metres."""
    from pxr import UsdGeom
    from scipy.spatial import ConvexHull, QhullError
    cache=UsdGeom.XformCache();scale=float(UsdGeom.GetStageMetersPerUnit(stage))
    for shape in geometry.shapes:
        prim=stage.GetPrimAtPath(shape['name']);body=prim
        while body.GetName()!=shape['body']:body=body.GetParent()
        if prim.IsA(UsdGeom.Mesh):
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
            local_to_world=np.asarray(cache.GetLocalToWorldTransform(prim)).T
            world=(points@local_to_world[:3,:3].T+local_to_world[:3,3])*scale
            T=np.asarray(cache.GetLocalToWorldTransform(body)).T;u,_,vt=np.linalg.svd(T[:3,:3]);R=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
            local=(world-T[:3,3]*scale)@R
            try:local=local[ConvexHull(local).vertices]
            except QhullError:pass
        else:
            # The existing body-local primitive bounds conservatively contain
            # smooth shapes too; never replace a cylinder/sphere by an inner polygon.
            signs=np.array(list(itertools.product((-1.,1.),repeat=3)))
            local=np.asarray(shape['center_body'])+signs*np.asarray(shape['half_extents_body'])
        if len(local)<4 or not np.isfinite(local).all():raise ValueError('invalid collision convex outer hull')
        shape['vertices_body']=local.tolist()


class CollisionGuard:
    """All actual collider boxes, with rigid-fixed and adjacent joints excluded.

    Positive clearance certifies separation of conservative boxes. A negative
    value is reported as uncertain/interfering bounds, never exact mesh contact.
    """
    def __init__(self, shapes, joints):
        self.shapes=shapes
        names={s['body'] for s in shapes}|{name for a,b,_ in joints for name in (a,b) if name}
        parent={name:name for name in names}
        def find(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]];x=parent[x]
            return x
        for a,b,fixed in joints:
            if a in parent and b in parent and fixed:
                parent[find(a)]=find(b)
        self.groups={name:find(name) for name in parent}
        adjacent={frozenset((find(a),find(b))) for a,b,_ in joints if a in parent and b in parent}
        self.pairs=[]
        for i,a in enumerate(shapes):
            for j in range(i):
                b=shapes[j]
                ga,gb=self.groups[a['body']],self.groups[b['body']]
                if ga!=gb and frozenset((ga,gb)) not in adjacent:
                    self.pairs.append((i,j))

    def check(self, body_poses):
        centers=[];rotations=[];extents=[]
        for shape in self.shapes:
            pose=np.asarray(body_poses[shape['body']])
            centers.append(pose[:3,:3]@shape['center_body']+pose[:3,3]);rotations.append(pose[:3,:3]);extents.append(shape['half_extents_body'])
        if not self.pairs:return float('inf'),None
        centers=np.asarray(centers);rotations=np.asarray(rotations);extents=np.asarray(extents)
        ids=np.asarray(self.pairs);i,j=ids[:,0],ids[:,1];delta=centers[j]-centers[i]
        world_extent=np.einsum('nij,nj->ni',np.abs(rotations),extents)
        distance=np.max(np.abs(delta)-world_extent[i]-world_extent[j],axis=1)
        close=np.flatnonzero(distance<.05)
        if len(close):
            ia,ib=i[close],j[close];A,B=rotations[ia],rotations[ib]
            cross=np.cross(A.transpose(0,2,1)[:,:,None,:],B.transpose(0,2,1)[:,None,:,:]).reshape(-1,9,3)
            axes=np.concatenate([A.transpose(0,2,1),B.transpose(0,2,1),cross],axis=1)
            norm=np.linalg.norm(axes,axis=2);valid=norm>1e-9;axes/=np.maximum(norm[:,:,None],1e-30)
            center=np.abs(np.einsum('nki,ni->nk',axes,delta[close]))
            ra=np.einsum('nkj,nj->nk',np.abs(np.einsum('nki,nij->nkj',axes,A)),extents[ia])
            rb=np.einsum('nkj,nj->nk',np.abs(np.einsum('nki,nij->nkj',axes,B)),extents[ib])
            separation=center-ra-rb;separation[~valid]=-np.inf
            distance[close]=np.max(separation,axis=1)
        narrow=np.flatnonzero(distance<.025)
        for index in narrow:
            ia,ib=self.pairs[index];a,b=self.shapes[ia],self.shapes[ib]
            if 'vertices_body' in a and 'vertices_body' in b:
                pa,pb=np.asarray(body_poses[a['body']]),np.asarray(body_poses[b['body']])
                va=np.asarray(a['vertices_body'])@pa[:3,:3].T+pa[:3,3]
                vb=np.asarray(b['vertices_body'])@pb[:3,:3].T+pb[:3,3]
                distance[index]=max(distance[index],convex_separation(va,vb))
        best=int(np.argmin(distance));ia,ib=self.pairs[best]
        return float(distance[best]),[self.shapes[ia]['name'],self.shapes[ib]['name']]


class NativeCollisionGuard:
    """Same GJK hull lower bound over measured bodies, without Python hot loops."""
    def __init__(self, library, model_path):
        self.model=json.loads(Path(model_path).read_text())
        self.names=[b['name'] for b in self.model['body_kinematics']]
        self.shapes=self.model['collision_shapes']
        self.library=ctypes.CDLL(str(library))
        self.library.jev_guard_create.argtypes=[ctypes.c_char_p]
        self.library.jev_guard_create.restype=ctypes.c_void_p
        self.library.jev_guard_check.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
        self.library.jev_guard_check.restype=ctypes.c_int
        self.library.jev_guard_error.restype=ctypes.c_char_p
        self.library.jev_guard_destroy.argtypes=[ctypes.c_void_p]
        self.handle=self.library.jev_guard_create(str(model_path).encode())
        if not self.handle:raise RuntimeError(self.library.jev_guard_error().decode())

    def check(self, body_poses):
        matrices=np.ascontiguousarray([body_poses[name] for name in self.names],dtype=np.float64)
        if matrices.shape!=(len(self.names),4,4) or not np.isfinite(matrices).all():
            raise ValueError('complete finite measured body transforms required')
        minimum=ctypes.c_double();a=ctypes.c_int();b=ctypes.c_int()
        code=self.library.jev_guard_check(self.handle,matrices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(self.names),ctypes.byref(minimum),ctypes.byref(a),ctypes.byref(b))
        if code:raise RuntimeError(self.library.jev_guard_error().decode())
        return minimum.value,[self.shapes[a.value]['name'],self.shapes[b.value]['name']]

    def close(self):
        if self.handle:self.library.jev_guard_destroy(self.handle);self.handle=None
