"""Randomized subject-centered shots derived from the twelve demo designs."""
import numpy as np
from scipy.spatial.transform import Rotation

from jev_agent_wbc.reference import look_at, smoothstep
from jev_agent_wbc.whole_body_qp import whole_body_kinematics
from .config import TEMPLATES


def camera_task(rng, family, config):
    actions = TEMPLATES.get(family, ('moveout',))
    if family == 'random_combo':
        pool = ('orbit', 'movein', 'moveout', 'truck_left', 'truck_right', 'pedestal_up', 'pedestal_down')
        count = int(rng.integers(2, 5)) if config.combo_actions is None else config.combo_actions
        actions = tuple(rng.choice(pool, size=count, replace=False))
    first = actions[0]
    orbit_sign = float(rng.choice([-1., 1.]))
    yaw = -.9*orbit_sign if first == 'orbit' else -1.1 if first == 'truck_right' else 1.1 if first == 'truck_left' else 0.
    initial = np.array([0., 0., yaw, -yaw, -.9, 2.1, 0., -1.2, 0.])
    if first == 'pedestal_down':
        initial[4:6] = [-.8, 1.5]
    initial[4:6] += rng.uniform(-.035, .035, 2)
    initial[7] = -initial[4]-initial[5]
    world_yaw = float(rng.uniform(-np.pi, np.pi))
    initial[:2] = rng.uniform(-.5, .5, 2)
    initial[2] += world_yaw
    p, r, _ = whole_body_kinematics(initial, .002)
    subject = p + rng.uniform(1.5, 2.)*r[:, 0]
    rows, phases, parameters = [], [], []
    clock = 0.
    for i, motion in enumerate(actions):
        duration = round(float(rng.uniform(*config.action_duration_s)), 2)
        start = p.copy()
        forward = subject-start
        forward[2] = 0.
        forward /= np.linalg.norm(forward)
        left = np.cross([0., 0., 1.], forward)
        length = float(rng.uniform(.25, .50) if len(actions) > 1 else rng.uniform(.50, .80))
        angle = float((orbit_sign if i == 0 else rng.choice([-1., 1.]))*rng.uniform(.22, .45))
        height = float(rng.uniform(.06, .12) if len(actions) > 1 else rng.uniform(.10, .16))
        # Keep successive dolly actions outside the physical subject.
        length = min(length, max(.12, np.linalg.norm((subject-start)[:2])-.65)) if motion == 'movein' else length
        timing = float(rng.uniform(-.65, .65))
        for local in np.linspace(0., duration, int(np.ceil(duration/config.sample_step_s))+1):
            if i and local == 0.:
                continue
            f = smoothstep(local/duration)
            f += timing*f*(1-f)*(2*f-1)
            if motion == 'orbit':
                p = subject+Rotation.from_euler('z', angle*f).apply(start-subject)
            elif motion in ('movein', 'moveout'):
                p = start+f*length*forward*(1 if motion == 'movein' else -1)
            elif motion.startswith('truck'):
                p = start+f*length*left*(1 if motion == 'truck_left' else -1)
            else:
                dz = min(height, 1.03-start[2]) if motion == 'pedestal_up' else -min(height, start[2]-.58)
                p = start+f*(.65*length*forward+np.array([0., 0., dz]))
            rows.append(dict(t=round(clock+float(local), 9), position=p.tolist(),
                quaternion_xyzw=Rotation.from_matrix(look_at(p, subject)).as_quat().tolist()))
        phases.append(dict(name=f'action_{i+1:02d}_{motion}', motion=motion, start_s=clock, end_s=round(clock+duration, 9)))
        parameters.append(dict(motion=motion, duration_s=duration, length_m=length,
                               orbit_angle_rad=angle if motion == 'orbit' else None, timing_skew=timing))
        clock = phases[-1]['end_s']
    task = dict(reference_camera_trajectory=dict(frame='world_z_up_camera_link', samples=rows),
        shot_phase=phases, subject=dict(id='blue_subject', description='A small matte blue cylinder with a yellow top'),
        intent='Perform '+', then '.join(actions)+'. Preserve each action direction and endpoint, recognizable subject-centered framing and a level horizon. Use the original elapsed-time budget. Adjacent actions share a continuous endpoint; no transition gap needs completion.')
    scene = dict(name='', initial_state=initial.tolist(), base_z=.002,
        pre_roll_motion='constant_initial_velocity',
        subject=dict(id='blue_subject', position=subject.tolist(), orientation_xyzw=[0., 0., 0., 1.],
                     radius=.13, height=.36, source='runtime_subject_initial_calibration_and_rendering'),
        obstacles=[], description='CineGym randomized whole-body cinematic avoidance benchmark',
        evaluation=dict(progress_correspondence='original_elapsed_time', capture_fps=8, width=384, height=288, vfov_deg=55.))
    return task, scene, dict(actions=parameters, world_yaw_rad=world_yaw)
