"""TIAGo integration seams for the pinned, genuine Giskard core.

No local optimizer is a fallback for Giskard.  The older shared SLSQP controller
and its physical evidence remain a separate backend.  The factories below use
upstream classes and fail when their real dependencies are unavailable.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, RotationSpline

ROOT = Path(__file__).resolve().parent
CAMERA_FRAME = 'tiago_camera_optical_frame'
ARM_JOINTS = ('torso_lift_joint', *(f'arm_{i}_joint' for i in range(1, 8)))


class BackendUnavailable(RuntimeError):
    pass


def add_pinned_core_paths():
    """Use unmodified upstream sources; never alias a local QP as giskardpy."""
    source = ROOT/'wbc/vendor/cognitive_robot_abstract_machine'
    lock = json.loads((ROOT/'wbc/FRAMEWORK_SOURCE_LOCK.json').read_text())
    if not source.is_dir():
        raise BackendUnavailable('Pinned Giskard sources missing; run wbc/fetch_sources.py')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    modified = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
    if commit!=lock['repositories'][0]['commit'] or modified:
        raise BackendUnavailable('Giskard tracked sources differ from the pinned unmodified checkout')
    for package in ('giskardpy', 'krrood', 'semantic_digital_twin'):
        path = str(source/package/'src')
        if path not in sys.path:
            sys.path.insert(0, path)
    return source,commit


def camera_mount():
    config = json.loads((ROOT/'sensors.json').read_text())
    spec = config['robots']['tiago']['camera']
    forward, up = np.asarray(spec['forward_parent']), np.asarray(spec['up_parent'])
    if not (np.isclose(forward@up, 0.) and np.isclose(forward@forward, 1.) and np.isclose(up@up, 1.)):
        raise ValueError('camera forward/up are not orthonormal')
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([np.cross(forward, up), -up, forward])
    matrix[:3, 3] = spec['translation_m']
    return spec['parent'], matrix


def model_urdf():
    """Add the exact optical frame and resolve local meshes without source edits."""
    path = ROOT/'urdf/tiago_physics.urdf'
    document = ET.parse(path).getroot()
    profile = json.loads((ROOT/'robot.json').read_text())
    if tuple(j['name'] for j in profile['joints']) != ARM_JOINTS:
        raise ValueError('Expected single-arm TIAGo: active torso plus seven arm joints')
    for item in profile['joints']:
        joint = document.find(f"joint[@name='{item['name']}']")
        for key in ('lower', 'upper', 'velocity', 'effort'):
            if not np.isclose(float(joint.find('limit').get(key)), item[key], atol=1e-12,rtol=0.):
                raise ValueError(f'profile/source {key} mismatch: {item["name"]}')
    for mesh in document.findall('.//mesh'):
        asset = (path.parent/mesh.get('filename')).resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        mesh.set('filename', str(asset))
    parent, mount = camera_mount()
    if document.find(f"link[@name='{CAMERA_FRAME}']") is not None:
        raise ValueError('optical-frame name already exists')
    ET.SubElement(document, 'link', name=CAMERA_FRAME)
    joint = ET.SubElement(document, 'joint', name='tiago_camera_optical_fixed', type='fixed')
    ET.SubElement(joint, 'parent', link=parent)
    ET.SubElement(joint, 'child', link=CAMERA_FRAME)
    ET.SubElement(joint, 'origin', xyz=' '.join(map(str, mount[:3,3])),
                  rpy=' '.join(map(str, Rotation.from_matrix(mount[:3,:3]).as_euler('xyz'))))
    return ET.tostring(document, encoding='unicode')


@dataclass(frozen=True)
class CameraReferenceSample:
    t_s: float
    T_world_optical: np.ndarray
    twist_world: np.ndarray


class TimedCameraReference:
    """Explicit continuous-time curve, with unchanged authored knot times.

    Position uses Hermite knots with REQUIRED authored linear velocities.
    RotationSpline supplies continuous angular rate/acceleration; quaternion
    coefficients are never interpolated as four unrelated scalar positions.
    This defines an interpolation contract, not validation of robot tracking.
    """
    def __init__(self, samples):
        t = np.asarray([s['t_s'] for s in samples], dtype=float)
        p = np.asarray([s['position_m'] for s in samples], dtype=float)
        v = np.asarray([s['linear_velocity_world_mps'] for s in samples], dtype=float)
        q = np.asarray([s['quaternion_xyzw'] for s in samples], dtype=float)
        if (len(t)<2 or t[0]!=0 or np.any(np.diff(t)<=0) or p.shape!=(len(t),3)
                or v.shape!=p.shape or q.shape!=(len(t),4)
                or not all(np.isfinite(x).all() for x in (t,p,v,q))
                or not np.allclose(np.linalg.norm(q,axis=1),1.,atol=1e-8,rtol=0)):
            raise ValueError('finite increasing times, Nx3 position/velocity and unit XYZW quaternions required')
        self.times = t.copy()
        self.times.flags.writeable = False
        self.position = CubicHermiteSpline(t,p,v,extrapolate=False)
        self.rotation = RotationSpline(t,Rotation.from_quat(q))

    def evaluate(self, t_s):
        if not np.isfinite(t_s) or t_s<self.times[0] or t_s>self.times[-1]:
            raise ValueError('Reference time outside authored interval; no silent retiming/extrapolation')
        matrix = np.eye(4)
        orientation = self.rotation(t_s)
        matrix[:3,:3] = orientation.as_matrix()
        matrix[:3,3] = self.position(t_s)
        # Spatial angular rate from R(t+h) R(t-h)^T, hence world axes. This
        # avoids treating the spline's body angular rate as a spatial twist.
        left, right = max(0.,t_s-1e-5), min(float(self.times[-1]),t_s+1e-5)
        omega = (self.rotation(right)*self.rotation(left).inv()).as_rotvec()/(right-left)
        return CameraReferenceSample(float(t_s),matrix,np.r_[self.position(t_s,1),omega])


def differential_wheel_rates(forward_mps, yaw_radps):
    profile = json.loads((ROOT/'robot.json').read_text())
    if not np.isfinite([forward_mps,yaw_radps]).all():
        raise ValueError('nonfinite base rate')
    radius, separation = profile['wheel_radius_m'],profile['wheel_separation_m']
    return np.asarray([forward_mps-yaw_radps*separation/2,forward_mps+yaw_radps*separation/2])/radius


def build_giskard_world_for_preflight():
    """Construct the real upstream diff-drive world; no robot commands issued.

    The upstream stock world also contains six localization DOFs. A production
    feedback/command bridge must lock these non-actuated coordinates and verify
    the exact 10 actuated variables before any physical execution. That bridge
    is deliberately not represented as complete by this factory.
    """
    add_pinned_core_paths()
    from giskardpy.model.world_config import WorldWithDiffDriveRobot
    config = WorldWithDiffDriveRobot(urdf=model_urdf())
    config.setup_world()
    from semantic_digital_twin.world_description.connections import DifferentialDrive
    drives = [c for c in config.world.connections if isinstance(c,DifferentialDrive)]
    if len(drives)!=1:
        raise BackendUnavailable('Expected exactly one real DifferentialDrive connection')
    return config


class GiskardCameraGoalBinding:
    """Actual Giskard CartesianPose with managed, continuously updated pose values.

    No executor, inverse kinematics or local QP is hidden here. Compiling and
    physically closing the real Executor loop is a separate readiness gate.
    """
    def __init__(self, context, initial_sample):
        add_pinned_core_paths()
        import krrood.symbolic_math.symbolic_math as sm
        from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
        from semantic_digital_twin.spatial_types import Point3,Quaternion,Pose
        self.context = context
        self.variables = [sm.FloatVariable(f'tiago_camera_reference_{i}') for i in range(7)]
        for variable in self.variables:
            context.float_variable_data.register_expression(variable)
        root = context.world.root
        self.pose = Pose(position=Point3(*self.variables[:3],reference_frame=root),
                         orientation=Quaternion(*self.variables[3:],reference_frame=root),reference_frame=root)
        self.goal = CartesianPose(name='timed_tiago_camera',root_link=root,
            tip_link=context.world.get_body_by_name(CAMERA_FRAME),goal_pose=self.pose)
        self.last_t = -np.inf
        self.update(initial_sample)

    def update(self, sample):
        if sample.t_s<=self.last_t:
            raise ValueError('stale/repeated reference time')
        T = np.asarray(sample.T_world_optical)
        if (T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1])
                or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8)
                or not np.isclose(np.linalg.det(T[:3,:3]),1.,atol=1e-8)):
            raise ValueError('invalid optical SE(3) reference')
        values = np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_quat()]
        for variable,value in zip(self.variables,values):
            self.context.float_variable_data.set_value(variable,float(value))
        self.last_t = sample.t_s

    def ingest_environment(self, snapshot):
        raise BackendUnavailable('No measured environment/collision-world ingress exists yet; cannot silently ignore obstacles')
