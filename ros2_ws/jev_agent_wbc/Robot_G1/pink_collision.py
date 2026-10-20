"""Actual-USD collision geometry attached to the genuine fixed-base Pink model.

Import this only in the isolated Pin/HPP-FCL worker. Source filtered pairs are
preserved. Pink constrains upper-controlled pairs; the independent FCL journal
still checks every nonfiltered physical pair, including the policy-owned legs.
"""
import numpy as np


def full_configuration(model,joints):
    import pinocchio as pin
    q=pin.neutral(model)
    for name in list(model.names)[1:]:q[model.joints[model.getJointId(name)].idx_q]=joints[name]
    return q


def frame_offsets(model,metadata,joints,T_world_pelvis,body_poses):
    """One measured initialization calibration, never fitted again at runtime."""
    import pinocchio as pin
    data=model.createData();q=full_configuration(model,joints)
    pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
    pelvis=np.asarray(T_world_pelvis,float);offsets={}
    for body in sorted({s['body'] for s in metadata['shapes']}):
        fid=model.getFrameId(body)
        if fid>=model.nframes:raise ValueError('Actual collision body absent from pinned Pin model: '+body)
        offsets[body]=(np.linalg.inv(pelvis@data.oMf[fid].np)@np.asarray(body_poses[body],float)).tolist()
    return offsets


def geometry_model(model,metadata,offsets):
    import pinocchio as pin
    import hppfcl
    result=pin.GeometryModel()
    allowed={tuple(sorted(x)) for x in metadata['allowed_pairs']}
    for shape in metadata['shapes']:
        kind=shape['kind'];local=np.eye(4)
        if kind=='convexHull':
            vertices=hppfcl.StdVec_Vec3f();triangles=hppfcl.StdVec_Triangle()
            for v in shape['vertices_body']:vertices.append(np.asarray(v,float))
            for face in shape['triangles']:triangles.append(hppfcl.Triangle(*map(int,face)))
            geom=hppfcl.Convex(vertices,triangles)
        elif kind=='sphere':
            geom=hppfcl.Sphere(shape['radius']);local[:3,3]=shape['center_body']
        elif kind=='box':geom=hppfcl.Box(*shape['size']);local=np.asarray(shape['T_body_shape'],float)
        elif kind in ('capsule','cylinder'):
            geom=(hppfcl.Capsule if kind=='capsule' else hppfcl.Cylinder)(shape['radius'],shape['height'])
            local=np.asarray(shape['T_body_shape'],float)
        else:raise ValueError('Unsupported actual USD collision kind: '+kind)
        fid=model.getFrameId(shape['body']);frame=model.frames[fid]
        placement=frame.placement*pin.SE3(np.asarray(offsets[shape['body']])@local)
        obj=pin.GeometryObject(shape['name'],frame.parent,geom,placement);obj.parentFrame=fid
        result.addGeometryObject(obj)
    for i,first in enumerate(metadata['shapes']):
        for j in range(i+1,len(metadata['shapes'])):
            second=metadata['shapes'][j]
            if first['body']!=second['body'] and tuple(sorted((first['body'],second['body']))) not in allowed:
                result.addCollisionPair(pin.CollisionPair(i,j))
    return result


def upper_collision_model(model,geometry,locked,q):
    """Pairs immobile under all 15 owned joints remain in the external audit."""
    import pinocchio as pin
    reduced,collision=pin.buildReducedModel(model,geometry,locked,q)
    for pair in list(collision.collisionPairs):
        first,second=collision.geometryObjects[pair.first],collision.geometryObjects[pair.second]
        if first.parentJoint==second.parentJoint or not (first.parentJoint or second.parentJoint):
            collision.removeCollisionPair(pair)
    return reduced,collision


def predicted_body_poses(model,joints,T_world_pelvis,offsets):
    import pinocchio as pin
    data=model.createData();pin.forwardKinematics(model,data,full_configuration(model,joints));pin.updateFramePlacements(model,data)
    pelvis=np.asarray(T_world_pelvis,float)
    return {name:(pelvis@data.oMf[model.getFrameId(name)].np@np.asarray(offset)).tolist() for name,offset in offsets.items()}
