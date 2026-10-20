"""Source-URDF state fixture for CPU bridge tests; never a measured Observation."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from wbc_backend import ROOT,ARM_JOINTS


def source_state(spec,q=None,urdf_path=None):
    document=ET.parse(urdf_path or ROOT/'urdf/tiago_physics.urdf').getroot()
    q=np.array([j['home'] for j in spec['joints']]) if q is None else np.asarray(q)
    values=dict(spec.get('passive_joint_positions',{}));values.update(zip(ARM_JOINTS,q))
    children={}
    for joint in document.findall('joint'):
        children.setdefault(joint.find('parent').get('link'),[]).append(joint)
    poses,ancestors={spec['base_link']:np.eye(4)},{spec['base_link']:[]}
    stack=[spec['base_link']]
    while stack:
        parent=stack.pop()
        for joint in children.get(parent,[]):
            origin=joint.find('origin')
            T=np.eye(4)
            if origin is not None:
                T[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            T=poses[parent]@T
            active=list(ancestors[parent])
            axis_node=joint.find('axis')
            axis=np.fromstring(axis_node.get('xyz','1 0 0'),sep=' ') if axis_node is not None else np.array([1.,0.,0.])
            if joint.get('name') in ARM_JOINTS:
                active.append((ARM_JOINTS.index(joint.get('name'))+2,joint.get('type'),T[:3,:3]@axis,T[:3,3].copy()))
            value=values.get(joint.get('name'),0.)
            M=np.eye(4)
            if joint.get('type')=='prismatic': M[:3,3]=axis*value
            elif joint.get('type') in ('revolute','continuous'): M[:3,:3]=Rotation.from_rotvec(axis*value).as_matrix()
            child=joint.find('child').get('link')
            poses[child]=T@M;ancestors[child]=active;stack.append(child)
    jacobians={}
    for name,T in poses.items():
        J=np.zeros((6,10));J[:3,0]=[1.,0.,0.]
        J[:3,1]=np.cross([0.,0.,1.],T[:3,3]);J[3:,1]=[0.,0.,1.]
        for i,kind,axis,origin in ancestors[name]:
            J[:3,i]=axis if kind=='prismatic' else np.cross(axis,T[:3,3]-origin)
            if kind!='prismatic':J[3:,i]=axis
        jacobians[name]=J
    return poses,jacobians
