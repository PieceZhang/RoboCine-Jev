"""Read-only 200 Hz root/link velocity consistency evidence.

The native instantaneous solver velocities and finite pose increments are kept
separately. This diagnostic never replaces a measured velocity or changes a
policy observation, command, or stop threshold.
"""
import json


def analyze(samples, base_id):
    """Compare native values without declaring them interchangeable."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    stamps = np.asarray([s['stamp_s'] for s in samples])
    if len(stamps)<2 or not np.isfinite(stamps).all() or np.any(np.diff(stamps)<=0):
        raise ValueError('ordered finite substep measurements required')
    def rotation(raw):
        return Rotation.from_quat(np.asarray(raw)[..., 3:7]).as_matrix()
    root = np.asarray([s['root_transform_raw_after_fk'] for s in samples])
    base = np.asarray([s['base_transform_raw'] for s in samples])
    before = np.asarray([s['root_transform_raw_before_fk'] for s in samples])
    root_velocity = np.asarray([s['root_velocity_raw_after_fk'] for s in samples])
    base_velocity = np.asarray([s['base_com_velocity_raw'] for s in samples])
    root_data = np.asarray([s['root_com_velocity_w'] for s in samples])
    root_R,base_R,before_R = rotation(root),rotation(base),rotation(before)
    w_fd = Rotation.from_matrix(base_R[1:]@base_R[:-1].transpose(0, 2, 1)).as_rotvec()/np.diff(stamps)[:, None]
    v_fd = np.diff(base[:, :3], axis=0)/np.diff(stamps)[:, None]
    def summary(start):
        mask = stamps[1:]>=start
        return dict(start_s=float(start),samples=int(mask.sum()),
            pose_angular_mean_world=w_fd[mask].mean(0).tolist(),
            native_angular_mean_world=base_velocity[1:, 3:][mask].mean(0).tolist(),
            angular_difference_rmse_rad_s=np.sqrt(np.mean((w_fd[mask]-base_velocity[1:, 3:][mask])**2,axis=0)).tolist(),
            pose_linear_mean_world=v_fd[mask].mean(0).tolist(),
            native_com_linear_mean_world=base_velocity[1:, :3][mask].mean(0).tolist())
    return dict(samples=len(samples),first_stamp_s=float(stamps[0]),last_stamp_s=float(stamps[-1]),
        dt_min_s=float(np.min(np.diff(stamps))),dt_max_s=float(np.max(np.diff(stamps))),
        requested_base_is_articulation_root=bool(base_id==0),
        max_root_base_position_difference_m=float(np.max(np.linalg.norm(root[:,:3]-base[:,:3],axis=1))),
        max_root_base_orientation_difference_rad=float(np.max(Rotation.from_matrix(root_R@base_R.transpose(0,2,1)).magnitude())),
        max_root_base_native_velocity_difference=float(np.max(np.abs(root_velocity-base_velocity))),
        max_cached_root_native_velocity_difference=float(np.max(np.abs(root_data-root_velocity))),
        max_kinematic_refresh_root_position_change_m=float(np.max(np.linalg.norm(root[:,:3]-before[:,:3],axis=1))),
        max_kinematic_refresh_root_orientation_change_rad=float(np.max(Rotation.from_matrix(root_R@before_R.transpose(0,2,1)).magnitude())),
        max_kinematic_refresh_root_velocity_change=float(np.max(np.abs(root_velocity-np.asarray([s['root_velocity_raw_before_fk'] for s in samples])))),
        full=summary(stamps[1]),last_two_seconds=summary(max(stamps[1],stamps[-1]-2.)),
        last_half_second=summary(max(stamps[1],stamps[-1]-.5)),
        finite_pose_velocity_replaces_native_velocity=False,
        note='Contact/joint split-impulse integration can produce distinct pose increments and reported solver velocities; both remain evidence.')


class VelocityAudit:
    def __init__(self, robot, output, base_name):
        import hashlib
        import inspect
        from pathlib import Path
        self.robot = robot
        self.base_id = robot.body_names.index(base_name)
        self.stream = (output/'velocity_substeps.jsonl').open('w', buffering=1)
        self.metadata = dict(base_name=base_name, base_id=self.base_id,
            articulation_body_names=list(robot.body_names),
            articulation_root_body_name=robot.body_names[0],
            articulation_prim_paths=robot.root_physx_view.prim_paths,
            articulation_link_paths=robot.root_physx_view.link_paths,
            isaaclab_data_source_file=inspect.getfile(type(robot.data)),
            isaaclab_data_source_sha256=hashlib.sha256(Path(inspect.getfile(type(robot.data))).read_bytes()).hexdigest(),
            source='PhysX articulation tensor view and IsaacLab ArticulationData',
            quaternion_order_raw='xyzw', quaternion_order_data='wxyz',
            linear_velocity_origin='center_of_mass unless link explicitly named',
            angular_velocity_axes='world', acquisition='after every 0.005s physics step',
            diagnostic_only=True, changes_policy_or_thresholds=False)
        (output/'velocity_audit.json').write_text(json.dumps(self.metadata, indent=2)+'\n')

    def record(self, stamp_s):
        robot = self.robot
        view = robot.root_physx_view
        # Preserve the state before body_link_pose_w requests kinematic refresh.
        before_pose = view.get_root_transforms()[0].clone()
        before_velocity = view.get_root_velocities()[0].clone()
        state = robot.data.body_link_state_w[0, self.base_id].clone()
        root_pose = robot.data.root_link_pose_w[0].clone()
        root_velocity = robot.data.root_com_vel_w[0].clone()
        row = dict(stamp_s=float(stamp_s),
            root_transform_raw_before_fk=before_pose.tolist(),
            root_velocity_raw_before_fk=before_velocity.tolist(),
            base_link_state_w=state.tolist(),
            root_link_pose_w=root_pose.tolist(), root_com_velocity_w=root_velocity.tolist(),
            root_link_velocity_w=robot.data.root_link_vel_w[0].tolist(),
            root_com_pose_w=robot.data.root_com_pose_w[0].tolist(),
            base_com_pose_body=robot.data.body_com_pose_b[0,self.base_id].tolist(),
            root_transform_raw_after_fk=view.get_root_transforms()[0].tolist(),
            root_velocity_raw_after_fk=view.get_root_velocities()[0].tolist(),
            base_transform_raw=view.get_link_transforms()[0, self.base_id].tolist(),
            base_com_velocity_raw=view.get_link_velocities()[0, self.base_id].tolist(),
            data_timestamp_s=float(robot.data._sim_timestamp))
        self.stream.write(json.dumps(row)+'\n')

    def close(self):
        self.stream.close()
        path=self.stream.name
        with open(path) as stream:
            result=analyze([json.loads(line) for line in stream],self.base_id)
        from pathlib import Path
        Path(path).with_name('velocity_consistency.json').write_text(json.dumps(result,indent=2)+'\n')
