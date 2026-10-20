"""Short independent shots and a runtime-owned scene/robot fixture catalog."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from .whole_body_qp import whole_body_kinematics
from .reference import OriginalReference, look_at, smoothstep
from .jvs_kinematics import CAMERA_OFFSET


NAMES = ('orbit', 'movein', 'truck_left', 'truck_right', 'pedestal_up', 'pedestal_down')


def generate(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    duration = 8.
    intents = {
        'orbit': 'Orbit left around the small blue subject to reveal its side while maintaining a recognizable, steady subject-centered shot.',
        'movein': 'Move closer to the blue subject for increasing emphasis; preserve a clear overall sense of approach and keep the subject visible.',
        'truck_left': 'Truck left past the blue subject to reveal right-side background parallax; keep the subject recognizable and preserve leftward travel.',
        'truck_right': 'Truck right past the blue subject to reveal left-side background parallax; keep the subject recognizable and preserve rightward travel.',
        'pedestal_up': 'Raise the camera while dollying forward to reveal the top of the blue subject; preserve upward revelation and a calm subject-centered composition.',
        'pedestal_down': 'Lower the camera while dollying forward to reveal the lower part of the blue subject; preserve downward revelation and a calm subject-centered composition.',
    }
    for name in NAMES:
        # Start with real vertical/rotational reserve. Reset state is runtime robot
        # configuration, separate from the four-field upstream cinematic task.
        initial = np.array([0., 0., 0., 0., -.9, 2.1, 0., -1.2, 0.])
        if name == 'pedestal_down':
            initial[4:6] = [-.8, 1.5]
            initial[7] = -.7
        if name.startswith('truck') or name == 'orbit':
            initial[2] = .9 if name == 'orbit' else (-1.1 if name.endswith('right') else 1.1)
            initial[3] = -initial[2]
        p0, r0, _ = whole_body_kinematics(initial, .002)
        subject = p0+np.array([1.55, 0., 0.])
        rows = []
        for t in np.linspace(0., duration, 41):
            f = smoothstep(t/duration)
            p = p0.copy()
            if name == 'orbit':
                p = subject+Rotation.from_euler('z', -.65*f).apply(p0-subject)
                p[2] += .06*f
            elif name == 'movein':
                p[0] += .90*f
                p[2] += .08*f
            elif name.startswith('truck'):
                p[1] += (.90 if name.endswith('left') else -.90)*f
                p[2] += .06*f
            else:
                # A planar base cannot contribute to a purely vertical shot.
                # Couple the pedestal move to an explicit forward dolly.
                p[0] += .65*f
                p[2] += (.15 if name.endswith('up') else -.20)*f
            rotation = look_at(p, subject)
            rows.append(dict(t=float(t), position=p.tolist(), quaternion_xyzw=Rotation.from_matrix(rotation).as_quat().tolist()))
        task = dict(reference_camera_trajectory=dict(frame='world_z_up_camera_link', samples=rows),
            shot_phase=[dict(name='establish', motion=name, start_s=0., end_s=1.2),
                        dict(name='develop', motion=name, start_s=1.2, end_s=6.4),
                        dict(name='resolve', motion=name, start_s=6.4, end_s=duration)],
            subject=dict(id='blue_subject', description='A small matte blue sculptural cylinder with a yellow top'), intent=intents[name])
        # Author an intersection with link6 at the SAME physical time as the
        # moving camera. FK gives the wrist from the reference pose, avoiding
        # a crossing aimed at the robot's obsolete initial footprint.
        crossing_s = 4.8
        reference = OriginalReference(task, subject)
        camera, rotation, _ = reference.sample(crossing_s)
        wrist = camera-rotation@CAMERA_OFFSET
        crossing = wrist+rotation@np.array([.019, 0., .05825])
        side = 1. if name == 'truck_right' else -1.
        velocity = np.array([.035, side*.24, 0.])
        obstacle = dict(id='crossing_ball', radius=.10, color=[.85, .20, .10],
            keyframes=[dict(t=t, position=(crossing+(t-crossing_s)*velocity).tolist(),
                            velocity=velocity.tolist()) for t in (0., duration)])
        fixture = dict(name=name, initial_state=initial.tolist(), base_z=.002,
            pre_roll_motion='constant_initial_velocity',
            subject=dict(id='blue_subject', position=subject.tolist(), orientation_xyzw=[0., 0., 0., 1.],
                         radius=.13, height=.36, source='runtime_subject_initial_calibration_and_rendering'),
            obstacles=[obstacle], description=f'{name}: 8 s whole-body shot; elevated ball crosses link6 at t=4.8 s on the nominal reference',
            evaluation=dict(progress_correspondence='original_elapsed_time', capture_fps=8,
                            width=384, height=288, vfov_deg=55.))
        folder = directory/name
        folder.mkdir(exist_ok=True)
        (folder/'task.json').write_text(json.dumps(task, indent=2)+'\n')
        (folder/'runtime_scene.json').write_text(json.dumps(fixture, indent=2)+'\n')
    (directory/'catalog.json').write_text(json.dumps(dict(cases=list(NAMES), duration_s=duration,
        multi_shot_transitions_tested=False), indent=2)+'\n')


class RuntimeScene:
    def __init__(self, path):
        self.document = deepcopy(path) if isinstance(path, dict) else json.loads(Path(path).read_text())
        self.curves = {}
        self.collision_covers = {}
        for obj in self.document['obstacles']:
            from .runtime_obstacles import local_cover
            self.collision_covers[obj['id']]=local_cover(obj)
            frames = obj['keyframes']
            self.curves[obj['id']] = CubicHermiteSpline([k['t'] for k in frames],
                np.array([k['position'] for k in frames]), np.array([k['velocity'] for k in frames]))

    def obstacles(self, t):
        result = []
        for obj in self.document['obstacles']:
            curve = self.curves[obj['id']]
            clipped = float(np.clip(t, curve.x[0], curve.x[-1]))
            center = curve(clipped)
            velocity = curve(clipped, 1) if curve.x[0] <= t <= curve.x[-1] else np.zeros(3)
            if t < curve.x[0] and self.document.get('pre_roll_motion') == 'constant_initial_velocity':
                # A moving obstacle is already in motion during sensor startup.
                # This is observed past motion, not future truth given to WBC.
                velocity = curve(curve.x[0], 1)
                center = curve(curve.x[0])+velocity*(t-curve.x[0])
            result.append(dict(id=obj['id'], center=center, radius=obj['radius'], velocity=velocity,
                shape=obj.get('shape','sphere'),**({key:obj[key] for key in ('size','yaw_rad') if key in obj})))
        return result

    def truth_collision(self, t):
        """Independent auditing only. Never passed to Jev, its planner or the WBC."""
        obstacles = self.obstacles(t)
        centers=[];radii=[];velocities=[];ids=[]
        for index,obstacle in enumerate(obstacles):
            offsets,sizes=self.collision_covers[obstacle['id']]
            centers.extend(obstacle['center']+offsets);radii.extend(sizes)
            velocities.extend(np.tile(obstacle['velocity'],(len(offsets),1)));ids.extend([index+1]*len(offsets))
        return dict(format='jvs_lidar_surfels/v1',
            surfel_centers=np.array(centers).reshape(-1,3),surfel_radius=max(radii,default=.025),
            surfel_radii=np.array(radii),minimum_clearance_m=.025,
            surfel_velocities=np.array(velocities).reshape(-1,3),
            surfel_track_ids=np.array(ids),observation_stamp=t,
            truth_geometry='exact spheres; conservative full-volume cuboid ball covers')
