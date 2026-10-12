#!/usr/bin/env python3
"""Geometric camera-space orbit, pan and moveout examples; no joint-space path."""
import json
from pathlib import Path

import numpy as np


CASES = [
    dict(name='orbit_ccw10', shot='orbit', duration=24., angle_deg=10., radius_m=.6),
    dict(name='orbit_cw15', shot='orbit', duration=28., angle_deg=-15., radius_m=.8),
    dict(name='pan_left20', shot='pan', duration=16., angle_deg=20.),
    dict(name='pan_right30', shot='pan', duration=20., angle_deg=-30.),
    dict(name='moveout_040', shot='moveout', duration=16., distance_m=.4),
    dict(name='moveout_080', shot='moveout', duration=24., distance_m=.8),
]


def generate(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        duration = case['duration']
        hold = 2.
        samples = []
        for t in np.linspace(0., duration, round(duration*20)+1):
            tau = np.clip((t-hold)/(duration-2*hold), 0., 1.)
            s = 10*tau**3-15*tau**4+6*tau**5
            pose = np.zeros(6)
            if case['shot'] in ('orbit', 'pan'):
                theta = np.deg2rad(case['angle_deg'])*s
                pose[5] = theta
                if case['shot'] == 'orbit':
                    radius = case['radius_m']
                    # Subject c=[R,0,0]. c-p=R[cos(theta),sin(theta),0],
                    # exactly the +X viewing axis of Rz(theta).
                    pose[:2] = [radius*(1-np.cos(theta)), -radius*np.sin(theta)]
            else:
                pose[0] = -case['distance_m']*s
            samples.append(dict(time_from_start=float(t), pose=pose.tolist()))
        metadata = dict(**case, motion_window_s=[hold, duration-hold],
                        shot_axes='initial camera_link: +X view, +Y left, +Z up',
                        timing='quintic smoothstep; 2 s hold at each endpoint')
        if case['shot'] == 'orbit':
            metadata['subject_initial_camera_xyz_m'] = [case['radius_m'], 0., 0.]
        document = dict(format='jvs_relative_task_space_trajectory/v1',
            relative_to='current_camera_link', frame_id='initial_camera_link',
            dof_order=['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
            units=dict(translation='m', rotation='rad'), sample_rate_hz=20., duration_s=duration,
            shot_metadata=metadata, samples=samples)
        (folder/f'wbc_{case["name"]}.json').write_text(json.dumps(document, indent=2)+'\n')
    (folder/'camera_shots_manifest.json').write_text(json.dumps(dict(
        description='Normal unobstructed camera tracking examples; no stress cases',
        cases=[dict(**c, trajectory=f'wbc_{c["name"]}.json') for c in CASES]), indent=2)+'\n')


if __name__ == '__main__':
    generate(Path(__file__).resolve().parents[1]/'trajectories')
