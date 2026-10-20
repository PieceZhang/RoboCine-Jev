"""Measured-link convex collision distances and hard velocity barriers.

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
    cache = UsdGeom.XformCache()
    shapes,allowed,joints = [],set(),[]
    root_prim = stage.GetPrimAtPath(root)
    for prim in Usd.PrimRange(root_prim):
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
    for prim in Usd.PrimRange(root_prim,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        descendants = [prim] if prim.IsA(UsdGeom.Gprim) else [p for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Gprim)]
        for shape in descendants:
            if str(shape.GetPath()) in visited:
                continue
            visited.add(str(shape.GetPath()))
            owner = prim
            while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if not owner:
                raise ValueError(f'Collision has no rigid owner: {shape.GetPath()}')
            T_body = np.asarray(cache.GetLocalToWorldTransform(owner),float).T
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
                approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
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
    return dict(schema='tiago_actual_usd_collision/v1',shapes=shapes,
        allowed_pairs=[list(x) for x in sorted(allowed)],joint_collision_flags=joints,
        source='actual enabled USD collision GPrims; convexHull point set, primitive dimensions, owner transforms')


def point_jacobian(body_jacobian,point_world,body_origin_world):
    J = np.asarray(body_jacobian,float).copy()
    J[:3] += np.cross(J[3:].T,np.asarray(point_world)-body_origin_world).T
    return J[:3]


def same_collision_geometry(prepared,live,tolerance_m=1e-8):
    """Check CPU-prepared body-local geometry against the live Kit stage."""
    from scipy.spatial.distance import cdist
    if prepared['allowed_pairs']!=live['allowed_pairs'] or len(prepared['shapes'])!=len(live['shapes']):
        raise ValueError('Prepared/live collision inventory differs')
    maximum=0.
    for old,new in zip(prepared['shapes'],live['shapes']):
        if old['body']!=new['body'] or old['kind']!=new['kind']:
            raise ValueError('Prepared/live shape owner/type differs')
        for key in ('radius','height','size','center_body','T_body_shape'):
            if key in old and key in new:
                error=float(np.max(np.abs(np.asarray(old[key])-np.asarray(new[key]))))
                maximum=max(maximum,error)
        if old['kind']=='convexHull':
            distances=cdist(old['vertices_body'],new['vertices_body'])
            maximum=max(maximum,float(max(distances.min(axis=0).max(),distances.min(axis=1).max())))
    if maximum>tolerance_m:raise ValueError(f'Prepared/live geometry mismatch {maximum}')
    return dict(passed=True,shape_count=len(live['shapes']),maximum_local_entry_or_vertex_error=maximum,
        tolerance_m=tolerance_m,physical_state_substituted=False)


class CollisionBarriers:
    def __init__(self,metadata,activation_m=.08,safe_m=.005):
        import fcl
        self.fcl,self.activation,self.safe = fcl,activation_m,safe_m
        self.shapes = metadata['shapes']
        allowed = {tuple(sorted(x)) for x in metadata['allowed_pairs']}
        self.objects = []
        for shape in self.shapes:
            if shape['kind']=='convexHull':
                vertices = np.asarray(shape['vertices_body'],float)
                triangles = np.asarray(shape['triangles'],np.int64)
                faces = np.c_[np.full(len(triangles),3),triangles].flatten()
                geometry = fcl.Convex(vertices,len(triangles),faces)
                low,high = vertices.min(axis=0),vertices.max(axis=0)
                local = np.eye(4)
            elif shape['kind']=='sphere':
                geometry = fcl.Sphere(shape['radius'])
                local = np.eye(4);local[:3,3]=shape['center_body']
                low,high = -np.ones(3)*shape['radius'],np.ones(3)*shape['radius']
            elif shape['kind']=='box':
                geometry = fcl.Box(*shape['size'])
                local = np.asarray(shape['T_body_shape'])
                low,high = -np.asarray(shape['size'])/2,np.asarray(shape['size'])/2
            elif shape['kind'] in ('cylinder','capsule'):
                constructor = fcl.Cylinder if shape['kind']=='cylinder' else fcl.Capsule
                geometry = constructor(shape['radius'],shape['height'])
                local = np.asarray(shape['T_body_shape'])
                half = shape['height']/2+(shape['radius'] if shape['kind']=='capsule' else 0.)
                high = np.array([shape['radius'],shape['radius'],half]);low=-high
            else:
                raise ValueError('Unsupported collision geometry')
            shape['_local'] = local
            shape['_corners'] = np.array(list(itertools.product(*zip(low,high))))
            self.objects.append(fcl.CollisionObject(geometry))
        self.pairs = [(i,j) for i in range(len(self.shapes)) for j in range(i+1,len(self.shapes))
            if self.shapes[i]['body']!=self.shapes[j]['body'] and tuple(sorted((self.shapes[i]['body'],self.shapes[j]['body']))) not in allowed]

    def query(self,poses,jacobians):
        bounds = []
        fcl = self.fcl
        for shape,obj in zip(self.shapes,self.objects):
            T = np.asarray(poses[shape['body']])@shape['_local']
            obj.setTransform(fcl.Transform(T[:3,:3],T[:3,3]))
            points = shape['_corners']@T[:3,:3].T+T[:3,3]
            bounds.append((points.min(axis=0),points.max(axis=0)))
        A,b,records = [],[],[]
        minimum = np.inf
        lower_bound = np.inf
        query_count,uncontrollable_checked=0,0
        for i,j in self.pairs:
            low0,high0 = bounds[i];low1,high1 = bounds[j]
            aabb_distance = np.linalg.norm(np.maximum(np.maximum(low0-high1,low1-high0),0.))
            if aabb_distance>self.activation:
                lower_bound=min(lower_bound,float(aabb_distance))
                continue
            first,second = self.shapes[i]['body'],self.shapes[j]['body']
            result = fcl.DistanceResult()
            distance = fcl.distance(self.objects[i],self.objects[j],
                fcl.DistanceRequest(enable_nearest_points=True,enable_signed_distance=True),result)
            query_count+=1
            if not np.isfinite(distance):
                raise RuntimeError('Nonfinite FCL self distance')
            lower_bound=min(lower_bound,float(distance))
            if distance < -.0005:
                raise RuntimeError(f'Nonallowed physical collision: {first}/{second}, {distance}')
            p0,p1 = np.asarray(result.nearest_points[0]),np.asarray(result.nearest_points[1])
            J0 = point_jacobian(jacobians[first],p0,np.asarray(poses[first])[:3,3])
            J1 = point_jacobian(jacobians[second],p1,np.asarray(poses[second])[:3,3])
            delta = p1-p0
            length = np.linalg.norm(delta)
            # Rigidly connected geometry has no controllable relative motion;
            # it cannot introduce a WBC separation task. Preserve evidence.
            if np.max(np.abs(J1-J0))<1e-9:
                uncontrollable_checked+=1
                continue
            if length<1e-10:
                raise RuntimeError(f'Undefined separation normal: {first}/{second}')
            normal = delta/length
            # Positive normal points first -> second; relative separation rate.
            row = normal@(J1-J0)
            if np.max(np.abs(row))<1e-9:
                uncontrollable_checked+=1
                continue
            minimum = min(minimum,float(distance))
            lower_bound=min(lower_bound,float(distance))
            if distance>=self.activation:
                continue
            A.append(row)
            b.append(-3.*(float(distance)-self.safe))
            records.append(dict(first=first,second=second,distance_m=float(distance),
                row=row.tolist(),lower_bound=b[-1]))
        return np.asarray(A).reshape((-1,10)),np.asarray(b),dict(
            minimum_controlled_distance_m=None if not np.isfinite(minimum) else float(minimum),
            minimum_distance_lower_bound_m=None if not np.isfinite(lower_bound) else float(lower_bound),
            active_pair_count=len(A),distance_queries=query_count,uncontrollable_pairs_distance_checked=uncontrollable_checked,barriers=records,
            excluded_pair_policy='official USD filtered pairs and collision-disabled adjacent joints; uncontrollable nonallowed pairs still distance/penetration checked')
