"""Measured-link convex collision distances for the legged backend journal.

Geometry comes from the actual USD collision GPrims. Broad-phase AABBs only
skip distant pairs; distances/nearest points are FCL geometry queries. Official
filtered pairs and collision-disabled adjacent joints are explicitly excluded.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def extract_collision_geometry(stage,root):
    """Call after Kit import; return body-local physical collision geometry."""
    from pxr import Usd,UsdGeom,UsdPhysics
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage),1.):
        raise ValueError('legged FCL extractor requires a metre-denominated physical stage')
    cache = UsdGeom.XformCache()
    shapes,allowed,joints = [],set(),[]
    root_prim = stage.GetPrimAtPath(root)
    for prim in Usd.PrimRange(root_prim,Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            for path in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                allowed.add(tuple(sorted((prim.GetName(),path.name))))
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            a,b = joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets()
            if a and b:
                pair = tuple(sorted((a[0].name,b[0].name)))
                if not joint.GetCollisionEnabledAttr().Get():
                    allowed.add(pair)
                joints.append(dict(name=prim.GetName(),bodies=list(pair),collision_enabled=bool(joint.GetCollisionEnabledAttr().Get())))
    visited = set()
    owners = {}
    for candidate in Usd.PrimRange(root_prim,Usd.TraverseInstanceProxies()):
        if not candidate.IsA(UsdGeom.Gprim):
            continue
        prim=candidate
        while prim and prim.GetPath().HasPrefix(root_prim.GetPath()) and not prim.HasAPI(UsdPhysics.CollisionAPI):
            prim=prim.GetParent()
        if not prim or not prim.GetPath().HasPrefix(root_prim.GetPath()) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        descendants = [candidate]
        for shape in descendants:
            if str(shape.GetPath()) in visited:
                continue
            visited.add(str(shape.GetPath()))
            owner = shape
            while owner and owner.GetPath().HasPrefix(root_prim.GetPath()) and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if not owner or not owner.GetPath().HasPrefix(root_prim.GetPath()) or not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f'Collision has no rigid owner: {shape.GetPath()}')
            if owner.GetName() in owners and owners[owner.GetName()]!=str(owner.GetPath()):
                raise ValueError('Duplicate body names cannot identify measured collision geometry')
            owners[owner.GetName()]=str(owner.GetPath())
            T_body = np.asarray(cache.GetLocalToWorldTransform(owner),float).T
            u,_,v=np.linalg.svd(T_body[:3,:3])
            T_body[:3,:3]=u@np.diag([1.,1.,np.linalg.det(u@v)])@v
            T_shape = np.asarray(cache.GetLocalToWorldTransform(shape),float).T
            local = np.linalg.inv(T_body)@T_shape
            record = dict(name=str(shape.GetPath()),body=owner.GetName(),T_body_shape=local.tolist())
            if shape.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(shape).GetPointsAttr().Get(),float)
                # Apply possible baked Xform scale to points, then FCL geometry
                # uses the body's measured rigid transform without scale.
                vertices = points@local[:3,:3].T+local[:3,3]
                hull = ConvexHull(vertices)
                vertices = vertices[hull.vertices]
                hull = ConvexHull(vertices)
                faces = hull.simplices.copy()
                for i,face in enumerate(faces):
                    if np.cross(vertices[face[1]]-vertices[face[0]],vertices[face[2]]-vertices[face[0]])@hull.equations[i,:3]<0:
                        faces[i] = face[::-1]
                mesh_api=shape
                while mesh_api and mesh_api.GetPath().HasPrefix(root_prim.GetPath()) and not mesh_api.HasAPI(UsdPhysics.MeshCollisionAPI):
                    mesh_api=mesh_api.GetParent()
                approximation=UsdPhysics.MeshCollisionAPI(mesh_api).GetApproximationAttr().Get() if mesh_api and mesh_api.GetPath().HasPrefix(root_prim.GetPath()) else None
                if approximation not in ('convexHull',):
                    raise ValueError(f'Unsupported collision approximation {approximation}: {prim.GetPath()}')
                record.update(kind='convexHull',vertices_body=vertices.tolist(),triangles=faces.tolist(),approximation=approximation)
            elif shape.IsA(UsdGeom.Sphere):
                scale = np.linalg.norm(local[:3,:3],axis=0)
                if not np.allclose(scale,scale[0]):
                    raise ValueError('Nonuniformly scaled sphere unsupported')
                record.update(kind='sphere',radius=float(UsdGeom.Sphere(shape).GetRadiusAttr().Get()*scale[0]),center_body=local[:3,3].tolist())
            elif shape.IsA(UsdGeom.Cube):
                scale = np.linalg.norm(local[:3,:3],axis=0)
                size = float(UsdGeom.Cube(shape).GetSizeAttr().Get())
                pose = local.copy();pose[:3,:3] /= scale
                record.update(kind='box',size=(size*scale).tolist(),T_body_shape=pose.tolist())
            elif shape.IsA(UsdGeom.Cylinder) or shape.IsA(UsdGeom.Capsule):
                geom = UsdGeom.Cylinder(shape) if shape.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(shape)
                axis = str(geom.GetAxisAttr().Get())
                scale = np.linalg.norm(local[:3,:3],axis=0)
                axis_i = {'X':0,'Y':1,'Z':2}[axis]
                radial = [i for i in range(3) if i!=axis_i]
                if not np.isclose(scale[radial[0]],scale[radial[1]]):
                    raise ValueError('Nonuniform cylinder/capsule radial scale unsupported')
                pose = local.copy();pose[:3,:3] /= scale
                axis_rotation = np.eye(3) if axis=='Z' else Rotation.from_euler('y' if axis=='X' else 'x',np.pi/2).as_matrix()
                pose[:3,:3] = pose[:3,:3]@axis_rotation
                record.update(kind='cylinder' if shape.IsA(UsdGeom.Cylinder) else 'capsule',
                    radius=float(geom.GetRadiusAttr().Get()*scale[radial[0]]),
                    height=float(geom.GetHeightAttr().Get()*scale[axis_i]),T_body_shape=pose.tolist())
            else:
                raise ValueError(f'Unsupported physical collision GPrim: {shape.GetPath()} {shape.GetTypeName()}')
            shapes.append(record)
    if not shapes:
        raise ValueError('No physical collision shapes extracted')
    return dict(schema='legged_actual_usd_collision/v1',shapes=shapes,
        allowed_pairs=[list(x) for x in sorted(allowed)],joint_collision_flags=joints,
        source='actual enabled USD collision GPrims; convexHull point set, primitive dimensions, owner transforms')


class CollisionDistances:
    """Exact FCL geometry checks; no Jacobians or clearance certificates invented.

    The broad phase gives a conservative .08m lower bound for skipped pairs.
    Unsupported PhysX mesh approximations are rejected by the extractor.
    """
    def __init__(self, metadata, activation_m=.08):
        import fcl
        self.fcl,self.activation=fcl,float(activation_m)
        self.shapes=metadata['shapes'];self.objects=[];self.locals=[];self.corners=[]
        allowed={tuple(sorted(pair)) for pair in metadata['allowed_pairs']}
        for shape in self.shapes:
            kind=shape['kind'];local=np.eye(4)
            if kind=='convexHull':
                vertices=np.asarray(shape['vertices_body'],float);triangles=np.asarray(shape['triangles'],np.int64)
                faces=np.c_[np.full(len(triangles),3),triangles].flatten()
                geom=fcl.Convex(vertices,len(triangles),faces)
                low,high=vertices.min(axis=0),vertices.max(axis=0)
            elif kind=='sphere':
                geom=fcl.Sphere(shape['radius']);local[:3,3]=shape['center_body']
                high=np.ones(3)*shape['radius'];low=-high
            elif kind=='box':
                geom=fcl.Box(*shape['size']);local=np.asarray(shape['T_body_shape'])
                high=np.asarray(shape['size'])/2;low=-high
            elif kind in ('capsule','cylinder'):
                geom=(fcl.Capsule if kind=='capsule' else fcl.Cylinder)(shape['radius'],shape['height'])
                local=np.asarray(shape['T_body_shape']);half=shape['height']/2+(shape['radius'] if kind=='capsule' else 0.)
                high=np.array([shape['radius'],shape['radius'],half]);low=-high
            else:raise ValueError('unsupported original physical collision geometry')
            self.objects.append(fcl.CollisionObject(geom));self.locals.append(local)
            self.corners.append(np.array(list(itertools.product(*zip(low,high)))))
        self.pairs=np.array([(i,j) for i in range(len(self.shapes)) for j in range(i+1,len(self.shapes))
            if self.shapes[i]['body']!=self.shapes[j]['body']
            and tuple(sorted((self.shapes[i]['body'],self.shapes[j]['body']))) not in allowed],int).reshape(-1,2)

    def query(self,poses):
        bounds=[]
        for shape,obj,local,corners in zip(self.shapes,self.objects,self.locals,self.corners):
            T=np.asarray(poses[shape['body']],float)@local
            if T.shape!=(4,4) or not np.isfinite(T).all():raise ValueError('finite measured body transform required')
            obj.setTransform(self.fcl.Transform(T[:3,:3],T[:3,3]))
            points=corners@T[:3,:3].T+T[:3,3];bounds.append([points.min(axis=0),points.max(axis=0)])
        bounds=np.asarray(bounds);minimum=self.activation;records=[]
        if len(self.pairs):
            first,second=self.pairs.T
            broad=np.linalg.norm(np.maximum(np.maximum(bounds[first,0]-bounds[second,1],bounds[second,0]-bounds[first,1]),0.),axis=1)
            for i,j in self.pairs[broad<=self.activation]:
                result=self.fcl.DistanceResult()
                distance=self.fcl.distance(self.objects[i],self.objects[j],
                    self.fcl.DistanceRequest(enable_nearest_points=True,enable_signed_distance=True),result)
                if not np.isfinite(distance):raise RuntimeError('nonfinite original-geometry FCL distance')
                minimum=min(minimum,float(distance))
                records.append(dict(first=self.shapes[i]['body'],second=self.shapes[j]['body'],distance_m=float(distance)))
        return dict(minimum_clearance_lower_bound_m=minimum,clear=minimum>=.005,safety_margin_m=.005,
            physical_shape_count=len(self.shapes),nonallowed_pair_count=len(self.pairs),fine_queries=len(records),
            close_pairs=records,source='original collision primitives and convexHull FCL queries at measured body transforms',
            future_swept_collision_certified=False)


class DistanceProcess:
    """One asynchronous clearance query in a separate existing FCL interpreter."""
    def __init__(self,python,source,metadata,stderr_path,max_age_s=.06):
        import subprocess,time
        self.stderr=open(stderr_path,'w');self.age=float(max_age_s);self.pending=None;self.latest=None;self.seq=0
        self.process=subprocess.Popen([str(python),str(source),'--distance-worker',str(metadata)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1,
            env=dict(__import__('os').environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=''))
        ready=self.receive(15.)
        if not ready or not ready.get('ok'):self.close();raise RuntimeError(f'FCL init failed: {ready}')

    def receive(self,timeout=0.):
        import select
        if not select.select([self.process.stdout],[],[],timeout)[0]:return None
        line=self.process.stdout.readline()
        return json.loads(line) if line else dict(ok=False,error='FCL worker exited')

    def submit(self,stamp_s,poses):
        import time
        if self.pending is not None or self.process.poll() is not None:return False
        request=dict(seq=self.seq,stamp_s=float(stamp_s),poses={n:np.asarray(T).tolist() for n,T in poses.items()})
        self.seq+=1;self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        self.pending=dict(seq=request['seq'],stamp_s=stamp_s,sent=time.monotonic(),poses=request['poses']);return True

    def poll(self,stamp_s):
        import time
        result=self.receive()
        if result is not None:
            pending=self.pending;self.pending=None
            self.latest=None
            if pending and result.get('ok') and result.get('seq')==pending['seq'] and result.get('stamp_s')==pending['stamp_s']:
                self.latest=dict(result,received_wall=time.monotonic(),T_world_collision_bodies=pending['poses'])
        if self.pending and time.monotonic()-self.pending['sent']>3*self.age:
            if self.process.poll() is None:self.process.terminate()
            return dict(status='unknown',reason='collision_worker_deadline',stamp_s=stamp_s)
        if (self.latest and 0<=stamp_s-self.latest['stamp_s']<=self.age
                and time.monotonic()-self.latest['received_wall']<=self.age):
            return dict(self.latest,status='pass' if self.latest['clear'] else 'fail')
        return dict(status='unknown',reason='collision_evidence_unavailable_or_stale',stamp_s=stamp_s)

    def close(self):
        if getattr(self,'process',None) is not None and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=2.)
            except Exception:self.process.kill();self.process.wait(timeout=2.)
        if getattr(self,'stderr',None):self.stderr.close()


def worker(metadata):
    import importlib.metadata,sys,traceback
    try:
        model=CollisionDistances(json.loads(Path(metadata).read_text()))
        print(json.dumps(dict(ok=True,fcl_version=importlib.metadata.version('python-fcl'))),flush=True)
    except Exception as exc:
        print(json.dumps(dict(ok=False,error=repr(exc))),flush=True);return
    for line in sys.stdin:
        request=json.loads(line)
        try:response=dict(model.query(request['poses']),ok=True,seq=request['seq'],stamp_s=request['stamp_s'])
        except Exception as exc:
            traceback.print_exc(file=sys.stderr);response=dict(ok=False,error=repr(exc),seq=request['seq'],stamp_s=request['stamp_s'])
        print(json.dumps(response),flush=True)


if __name__=='__main__':
    import sys
    worker(sys.argv[2])

