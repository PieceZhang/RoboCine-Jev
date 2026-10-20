"""Offline camera feasibility checks; no rollout is given to the Agent."""
import numpy as np
from jev_agent_wbc.camera_intent import POLICY, default_intrinsics, visibility
from jev_agent_wbc.shot_acceptance import evaluate_shot
from jev_agent_wbc.self_collision import SelfCollision
from jev_agent_wbc.whole_body_qp import whole_body_kinematics


class WithTerminalHold:
    def __init__(self, reference):
        self.reference=reference
        self.duration=reference.duration+POLICY.terminal_hold_s

    def __getattr__(self, name):
        return getattr(self.reference,name)


def audit_nominal(reference, scene, times, states, obstacle_curves=None):
    indices=np.unique(np.r_[np.arange(0,len(times),5),len(times)-1])
    poses=[whole_body_kinematics(x,scene['base_z'])[:2] for x in states[indices]]
    times=times[indices]
    clear=dict(valid=True,observation_stamp=0.,centers=[],radii=[],velocities=[])
    measurements=[]
    for (p,r),t in zip(poses,times):
        observed=clear
        if obstacle_curves is not None:
            from jev_agent_wbc.runtime_obstacles import local_cover
            centers=[];radii=[]
            for obstacle,curve in zip(scene['obstacles'],obstacle_curves):
                offsets,sizes=local_cover(obstacle)
                centers.extend(curve(np.clip(t,curve.x[0],curve.x[-1]))+offsets);radii.extend(sizes)
            observed=dict(valid=True,observation_stamp=float(t),
                centers=np.asarray(centers).tolist(),radii=radii,
                velocities=np.zeros((len(centers),3)).tolist())
        measurements.append(visibility(p,r,observed,float(t),scene['subject'],default_intrinsics()))
    acceptance=evaluate_shot(reference,scene['subject'],times,
        [p for p,_ in poses],[r for _,r in poses],
        measurements,np.minimum(times,reference.duration),
        safety_pass=True,execution_ok=True,intrinsics=default_intrinsics(),
        motion_end_s=reference.duration,safety_source='offline obstacle-free nominal self-clearance')
    clearance=float(SelfCollision().distances_sequence(states).min())
    return dict(accepted=bool(acceptance['accepted'] and clearance>=.004),
        failed_checks=acceptance['failed_checks']+(['self_clearance'] if clearance<.004 else []),min_self_clearance_m=clearance,
        checks=acceptance['checks'],
        scope=('Kinematic shot acceptance including authored obstacle visibility; separate swept safety proof required'
            if obstacle_curves is not None else 'Obstacle-free kinematic witness with original shot acceptance and terminal hold; NOT an avoidance solvability proof'))
