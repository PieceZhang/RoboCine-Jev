"""Measured-state/model and actuator mapping for an OCS2 integration.

No optimizer runs in this module. Existing NumPy/SciPy controllers are never
silently substituted for the requested OCS2 MPC backend.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def rigid_transform(value):
    value=np.asarray(value,dtype=float)
    if (value.shape!=(4,4) or not np.isfinite(value).all()
            or not np.allclose(value[3],[0,0,0,1],atol=1e-8)
            or not np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=1e-7)
            or not np.isclose(np.linalg.det(value[:3,:3]),1.,atol=1e-7)):
        raise ValueError('expected finite SE(3) transform')
    return value


class WheeledBackendAdapter:
    def __init__(self, folder=None):
        self.folder=Path(folder or Path(__file__).resolve().parent)
        self.spec=json.loads((self.folder/'backend.json').read_text())
        self.model=json.loads((self.folder/'model.json').read_text())
        self.omni=self.spec['base_model']=='omnidirectional'
        self.joint_names=self.spec['arm_joints']
        self.T_ee_camera=rigid_transform(self.model['T_ee_camera_optical'])
        self.M=rigid_transform(self.model['T_base_ee_at_authored_pose'])
        self.q_rest=np.array([j['q_at_authored_pose'] for j in self.model['joints']])
        self.nominal_height=float(np.asarray(self.model['T_world_base_at_authored_pose'])[2,3])

    def state_from_measured(self, T_world_base, joints):
        base=rigid_transform(T_world_base)
        q=np.array([joints[name] for name in self.joint_names],dtype=float)
        if not np.isfinite(q).all():
            raise ValueError('nonfinite measured joint state')
        return np.r_[base[:2,3],np.arctan2(base[1,0],base[0,0]),q]

    def flow_map(self, state, control):
        """Python reference of the compiled OCS2 dynamics, not an OCS2 solver."""
        x,u=np.asarray(state,dtype=float),np.asarray(control,dtype=float)
        if x.shape!=(9,) or u.shape!=(self.spec['input_dim'],) or not np.isfinite(x).all() or not np.isfinite(u).all():
            raise ValueError('wrong state/input dimensions or nonfinite values')
        vx,vy=u[0],u[1] if self.omni else 0.
        c,s=np.cos(x[2]),np.sin(x[2])
        return np.r_[c*vx-s*vy,s*vx+c*vy,u[2 if self.omni else 1],u[-6:]]

    def camera_in_base(self, q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(6,) or not np.isfinite(q).all():
            raise ValueError('expected six finite arm joints')
        result=np.eye(4)
        for value,joint,rest in zip(q,self.model['joints'],self.q_rest):
            angle=value-rest
            omega=np.array(joint['screw_omega']);v=np.array(joint['screw_v']);w=skew(omega)
            exp=np.eye(4)
            exp[:3,:3]=np.eye(3)+np.sin(angle)*w+(1-np.cos(angle))*(w@w)
            exp[:3,3]=(np.eye(3)*angle+(1-np.cos(angle))*w+(angle-np.sin(angle))*(w@w))@v
            result=result@exp
        return result@self.M@self.T_ee_camera

    def measured_camera_pose(self, T_world_base, q):
        """FK diagnostic with full measured base rotation and height."""
        return rigid_transform(T_world_base)@self.camera_in_base(q)

    def camera_pose(self, state, base_height=None):
        x=np.asarray(state,dtype=float)
        if x.shape!=(9,) or not np.isfinite(x).all():
            raise ValueError('expected finite 9D state')
        base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',x[2]).as_matrix()
        base[:3,3]=[x[0],x[1],self.nominal_height if base_height is None else base_height]
        return base@self.camera_in_base(x[3:])

    def tool_reference(self, T_frame_camera, frame='world', T_world_frame=None):
        """Preserve the optical trajectory exactly while changing only tool frame."""
        reference=rigid_transform(T_frame_camera)
        if frame=='map':
            if T_world_frame is None:
                raise ValueError('map requires an explicit calibrated T_world_map')
            reference=rigid_transform(T_world_frame)@reference
        elif frame!='world':
            raise ValueError('only fixed world/map reference frames are supported')
        return reference@np.linalg.inv(self.T_ee_camera)

    def actuators(self, state, control):
        """Map a solver policy to the actual plant, without issuing commands."""
        derivative=self.flow_map(state,control)
        control=np.asarray(control)
        if self.omni:
            return dict(kind='native_planar_joint_velocities',
                joints=['dummy_base_prismatic_x_joint','dummy_base_prismatic_y_joint','dummy_base_revolute_z_joint']+self.joint_names,
                velocities=np.r_[derivative[:3],control[-6:]].tolist(),
                wheel_ground_dynamics_verified=False)
        profile=json.loads((self.folder.parent/'robot.json').read_text())
        v,w=control[:2];r=profile['wheel_radius_m'];track=profile['wheel_separation_m']
        return dict(kind='rolling_wheel_and_arm_velocities',
            wheel_joints=profile['left_wheels']+profile['right_wheels'],
            wheel_velocities=[(v-w*track/2)/r]*2+[(v+w*track/2)/r]*2,
            arm_joints=self.joint_names,arm_velocities=control[-6:].tolist(),
            skid_slip_compensated=False)

    def negotiate(self, geometry_requests=()):
        if geometry_requests:
            return dict(supported=False,feasible=None,reason='geometry_costs_and_environment_constraints_not_integrated')
        return dict(supported=False,feasible=None,reason='model_adapter_does_not_dispatch_motion_use_run_ocs2',
                    prepared_model=self.spec['base_model'],reference_modification_allowed=False)

    def native_request(self, T_world_base, joints, timestamps_s, camera_poses, frame='world', T_world_frame=None, position_tolerance_m=.02, orientation_tolerance_rad=.05):
        """Internal native-solver input; never an Agent joint/FK interface.

        Optical waypoints stay unchanged. The native model uses planar base
        prediction; full measured base transform remains explicit diagnostic data.
        This prepares input only and does not issue plant commands.
        """
        if not all(np.isfinite(v) and v>0 for v in [position_tolerance_m,orientation_tolerance_rad]):
            raise ValueError('positive finite optical pose tolerances required')
        times=np.asarray(timestamps_s,dtype=float)
        if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all()
                or times[0]!=0 or np.any(np.diff(times)<=0) or times[-1]>10):
            raise ValueError('expected strictly increasing relative timestamps in [0,10]')
        if len(camera_poses)!=len(times):
            raise ValueError('one optical pose per timestamp is required')
        optical=[]
        for pose in camera_poses:
            tool=self.tool_reference(pose,frame,T_world_frame)
            optical.append((tool@self.T_ee_camera).tolist())
        return dict(schema='ocs2_wheeled_internal_request/v1',frame='world',
                    initial_state=self.state_from_measured(T_world_base,joints).tolist(),
                    timestamps_s=times.tolist(),camera_poses=optical,
                    position_tolerance_m=position_tolerance_m,orientation_tolerance_rad=orientation_tolerance_rad,
                    T_world_base_measured=rigid_transform(T_world_base).tolist(),
                    planar_prediction_only=True,physical_commands_issued=False)
