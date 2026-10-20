"""OCS2 worker transport and hard measured-state execution guards; no optimizer."""
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import itertools
import ctypes

import numpy as np


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
                result.update(transport_wall_s=time.monotonic()-self.submitted_wall,request_wall_s=self.submitted_wall)
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
    """Physical velocity/acceleration and anticipatory arm-joint braking caps."""
    def __init__(self, omni, lower, upper, dt=.01):
        self.nu = 9 if omni else 8
        self.dt = dt
        self.speed = np.r_[([.22,.22,.35] if omni else [.22,.35]), np.full(6,.65)]
        self.acceleration = np.r_[([.45,.45,.65] if omni else [.45,.65]), np.ones(6)]
        self.lower, self.upper = np.array(lower),np.array(upper)
        self.previous = np.zeros(self.nu)

    def command(self, desired, q, *, brake=False):
        desired, q = np.asarray(desired,float),np.asarray(q,float)
        if desired.shape!=(self.nu,) or q.shape!=(6,) or not np.isfinite(desired).all() or not np.isfinite(q).all():
            raise ValueError('invalid physical guard state/command')
        if np.any(q<self.lower-.015) or np.any(q>self.upper+.015):
            raise RuntimeError('measured physical joint outside its original bound')
        lo=np.maximum(-self.speed,self.previous-self.acceleration*self.dt)
        hi=np.minimum(self.speed,self.previous+self.acceleration*self.dt)
        a=self.acceleration[-6:]
        # One physics/control interval is reserved before full deceleration.
        safe=lambda distance: np.maximum(0.,np.sqrt((a*self.dt)**2+2*a*np.maximum(distance-.005,0.))-a*self.dt)
        lo[-6:]=np.maximum(lo[-6:],-safe(q-self.lower))
        hi[-6:]=np.minimum(hi[-6:],safe(self.upper-q))
        if np.any(lo>hi+1e-10):
            raise RuntimeError('joint braking envelope conflicts with bounded acceleration')
        result=np.clip(np.zeros(self.nu) if brake else desired,lo,hi)
        self.previous=result.copy()
        return result


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
