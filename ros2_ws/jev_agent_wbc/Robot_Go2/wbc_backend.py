"""Native Go2-Piper WBC optical-camera adapter, without claiming extra DOFs.

Native command = [x_link0, y_link0, z_WORLD, qw_link0, qx, qy, qz].
World optical camera axes are x right, y down, z forward at this boundary.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM_MDP = ROOT/'vendor/LeggedManip_Lab/source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab'


def rigid(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1])
            or not np.allclose(value[:3, :3].T@value[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-6)):
        raise ValueError('expected a finite rigid transform')
    return value


def camera_to_mixed(T_world_camera_optical, T_world_link0, T_ee_camera_optical):
    """Remove full EE-camera extrinsics before compiling the upstream command."""
    link0 = rigid(T_world_link0)
    if np.linalg.cond(link0[:3, :3].T[:2, :2]) > 1e6:
        raise ValueError('mixed-frame xy/world-z chart is singular at this link0 attitude')
    ee = rigid(T_world_camera_optical) @ np.linalg.inv(rigid(T_ee_camera_optical))
    relative = np.linalg.inv(link0) @ ee
    quat = Rotation.from_matrix(relative[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return np.r_[relative[:2, 3], ee[2, 3], quat]


def mixed_to_world_ee(command, T_world_link0):
    """Exact inverse of the metrics' mixed frame, including roll and pitch."""
    command = np.asarray(command, dtype=float)
    link0 = rigid(T_world_link0)
    if command.shape != (7,) or not np.isfinite(command).all() or not np.isclose(np.linalg.norm(command[3:]), 1., atol=1e-6):
        raise ValueError('expected finite mixed xyz plus unit wxyz quaternion')
    inverse = link0[:3, :3].T
    plane = inverse[:2, :2]
    if np.linalg.cond(plane) > 1e6:
        raise ValueError('mixed-frame xy/world-z chart is singular at this link0 attitude')
    position = np.r_[np.linalg.solve(plane, command[:2]+inverse[:2]@link0[:3, 3]
                                     -inverse[:2, 2]*command[2]), command[2]]
    result = np.eye(4)
    result[:3, :3] = link0[:3, :3] @ Rotation.from_quat(command[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = position
    return result


def mixed_to_world_camera(command, T_world_link0, T_ee_camera_optical):
    return mixed_to_world_ee(command, T_world_link0) @ rigid(T_ee_camera_optical)


def set_camera_target(ee_command_term, T_world_camera_optical, T_world_link0, T_ee_camera_optical):
    """Write ONLY the native EE command; the learned policy retains all 18 joints."""
    import torch
    command = camera_to_mixed(T_world_camera_optical, T_world_link0, T_ee_camera_optical)
    ee_command_term.pose_command[0] = torch.as_tensor(command, dtype=ee_command_term.pose_command.dtype,
                                                    device=ee_command_term.pose_command.device)
    return command


def check_upstream_domain(command, config_path=None, sampler_path=None):
    """Exclude commands outside the pinned upstream configuration's outer domain.

    Inside this envelope means only "not excluded": the checkpoint's actual
    training distribution and a dynamically feasible whole-body solution remain
    unknown. This function never clamps/edits the requested camera or policy.
    """
    import ast
    import hashlib
    import operator
    command = np.asarray(command, dtype=float)
    if (command.shape != (7,) or not np.isfinite(command).all()
            or not np.isclose(np.linalg.norm(command[3:]), 1., atol=1e-6)):
        raise ValueError('expected finite mixed xyz plus unit wxyz quaternion')
    config_path = Path(config_path or UPSTREAM_MDP/'config/go2_piper/wbc_env_cfg.py')
    sampler_path = Path(sampler_path or UPSTREAM_MDP/'mdp/pose_command_wbc.py')
    def numeric(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return numeric(node.operand)*(1 if isinstance(node.op, ast.UAdd) else -1)
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            return {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                    ast.Div: operator.truediv}[type(node.op)](numeric(node.left), numeric(node.right))
        raise ValueError('upstream domain contains an unsupported numeric expression')
    try:
        config = ast.parse(config_path.read_text()); sampler = ast.parse(sampler_path.read_text())
        owner = next(n for n in config.body if isinstance(n, ast.ClassDef) and n.name == 'WBCCommandsCfg')
        call = next(n.value for n in owner.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'ee_pose' for t in n.targets))
        ranges = {}
        for keyword in call.keywords:
            if keyword.arg in ('ranges', 'limit_ranges'):
                ranges[keyword.arg] = {k.arg: [numeric(x) for x in k.value.elts]
                                      for k in keyword.value.keywords if k.arg in ('pos_x', 'pos_y', 'pos_z')}
        xyz = [[min(ranges[k][axis][0] for k in ('ranges', 'limit_ranges')),
                max(ranges[k][axis][1] for k in ('ranges', 'limit_ranges'))]
               for axis in ('pos_x', 'pos_y', 'pos_z')]
        clamp = next(n for n in ast.walk(sampler) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'clamp'
                     and len(n.args) == 3 and isinstance(n.args[0], ast.Name) and n.args[0].id == 'euler_angles')
        euler_range = [numeric(clamp.args[1]), numeric(clamp.args[2])]
        bounds = np.asarray(xyz+[euler_range]*3)
        if not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError('invalid upstream bounds')
    except (OSError, ValueError, TypeError, AttributeError, KeyError, StopIteration, ZeroDivisionError) as exc:
        return dict(supported=False, feasible=None, status='capability_metadata_unknown',
                    reason=str(exc), checkpoint_training_coverage='unknown')
    return _check_domain_envelope(command,xyz,euler_range,
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (config_path,sampler_path)})


def _check_domain_envelope(command, xyz, euler_range, sources):
    command=np.asarray(command,float)
    if (command.shape!=(7,) or not np.isfinite(command).all()
            or not np.isclose(np.linalg.norm(command[3:]),1.,atol=1e-6)):
        raise ValueError('expected finite mixed xyz plus unit wxyz quaternion')
    bounds=np.asarray(list(xyz)+[list(euler_range)]*3)
    angles = Rotation.from_quat(command[[4, 5, 6, 3]]).as_euler('xyz')
    values = np.r_[command[:3], angles]
    violations = [name for name, value, interval in zip(
        ('x_link0', 'y_link0', 'z_world', 'roll_link0', 'pitch_link0', 'yaw_link0'), values, bounds)
        if value < interval[0]-1e-8 or value > interval[1]+1e-8]
    return dict(supported=False if violations else None, feasible=None,
                status='out_of_policy_domain' if violations else 'inside_outer_domain_unverified',
                violations=violations, xyz_outer_bounds=xyz, euler_outer_bounds_rad=euler_range,
                checkpoint_training_coverage='unknown',
                sources_sha256=dict(sources),
                interpretation='outer exclusion only; inside does not certify reachability, contact, collision or learned tracking')


def camera_command_support(T_world_camera_optical, T_world_link0, T_ee_camera_optical):
    """Preflight a physical optical target before the future Agent sends it."""
    return check_upstream_domain(camera_to_mixed(T_world_camera_optical, T_world_link0, T_ee_camera_optical))


class PolicyDomain:
    """Compile immutable pinned configuration metadata once before physical time.

    A native runner freezes those source files and hashes before constructing
    this object. Per-tick checks contain no file IO, parsing or training claims.
    """
    def __init__(self,config_path=None,sampler_path=None):
        metadata=check_upstream_domain([.5,0.,.5,1.,0.,0.,0.],config_path,sampler_path)
        if metadata['status']!='inside_outer_domain_unverified':
            raise ValueError('upstream domain metadata unavailable')
        self.xyz=tuple(tuple(x) for x in metadata['xyz_outer_bounds'])
        self.euler=tuple(metadata['euler_outer_bounds_rad'])
        self.sources=dict(metadata['sources_sha256'])

    def check(self,command):
        return _check_domain_envelope(command,self.xyz,self.euler,self.sources)
