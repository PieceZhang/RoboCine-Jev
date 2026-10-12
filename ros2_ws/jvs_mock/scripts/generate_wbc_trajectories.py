#!/usr/bin/env python3
"""Generate reproducible, reachable 6D camera references from smooth WB motions."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from jvs_mock.whole_body_qp import whole_body_kinematics


def generate(folder):
    folder.mkdir(parents=True, exist_ok=True)
    p0, r0, _ = whole_body_kinematics(np.zeros(9))
    for name, duration, travel, turn, scale in [
        ('wbc_coordinated', 24., .65, .35, 1.),
        ('wbc_reverse', 28., -.5, -.45, 1.),
        ('wbc_fast', 8., .65, .35, 1.),
        ('wbc_limit_stress', 4., 1.4, .7, 1.5),
    ]:
        samples = []
        for t in np.linspace(0, duration, int(duration * 20) + 1):
            tau = np.clip((t-1.)/(duration-2.), 0., 1.)
            s = 10*tau**3 - 15*tau**4 + 6*tau**5
            yaw = turn*s
            # Constant-curvature differential-drive path, including reverse.
            x = travel*np.sin(yaw)/turn
            y = travel*(1-np.cos(yaw))/turn
            q = scale*np.array([.10*s, -.3*s, .55*s, .12*s, -.4*s, -.08*s])
            p, r, _ = whole_body_kinematics(np.r_[x, y, yaw, q])
            relative = np.r_[p-p0, Rotation.from_matrix(r @ r0.T).as_euler('xyz')]
            samples.append(dict(time_from_start=float(t), pose=relative.tolist()))
        document = dict(format='jvs_relative_task_space_trajectory/v1',
                        relative_to='current_camera_link', frame_id='base_link',
                        dof_order=['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
                        units=dict(translation='m', rotation='rad'), sample_rate_hz=20.,
                        duration_s=duration, samples=samples)
        (folder / f'{name}.json').write_text(json.dumps(document, indent=2)+'\n')


if __name__ == '__main__':
    generate(Path(__file__).resolve().parents[1] / 'trajectories')
