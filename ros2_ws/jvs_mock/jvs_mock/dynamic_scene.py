"""Validated obstacle models and physical-time trajectories (no ROS/USD imports)."""
import json
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline


def model_axis_rotation(model):
    """Canonical cylinder Z axis to its axis in the initial-base frame."""
    axis = model.get('axis', 'z')
    if axis not in ('x', 'y', 'z') or ('axis' in model and model.get('type') != 'cylinder'):
        raise ValueError('model axis must be x, y or z on a cylinder')
    return {'x': np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]),
            'y': np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]),
            'z': np.eye(3)}[axis]


class DynamicScene:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.document = json.loads(self.path.read_text())
        if self.document.get('format') != 'jvs_dynamic_scene/v1':
            raise ValueError('expected jvs_dynamic_scene/v1')
        self.frame = self.document.get('frame', 'initial_base')
        if self.frame not in ('initial_base', 'scene_y_up'):
            raise ValueError('scene frame must be initial_base or scene_y_up')
        self.obstacles = self.document.get('obstacles', [])
        if not self.obstacles:
            raise ValueError('dynamic scene must contain obstacles')
        self.curves = {}
        self.times = {}
        for obstacle in self.obstacles:
            name = obstacle.get('name', '')
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name) or name in self.curves:
                raise ValueError('obstacle names must be unique USD identifiers')
            model = obstacle['model']
            kind = model['type']
            dimensions = (model.get('size', []) if kind == 'box' else
                          [model.get('radius', 0), model.get('height', 0)] if kind == 'cylinder' else
                          [model.get('radius', 0)] if kind == 'sphere' else [])
            if (len(dimensions) != {'box': 3, 'cylinder': 2, 'sphere': 1}.get(kind, -1)
                    or not np.isfinite(dimensions).all() or min(dimensions) <= 0):
                raise ValueError('invalid obstacle model')
            model_axis_rotation(model)
            color = np.asarray(model.get('color', [.85, .3, .12]), float)
            if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
                raise ValueError('model color must be RGB in [0,1]')
            samples = obstacle['trajectory']
            times = np.array([sample['time_s'] for sample in samples], float)
            positions = np.array([sample['position'] for sample in samples], float)
            if (len(times) < 2 or times[0] != 0 or not np.all(np.diff(times) > 0)
                    or positions.shape != (len(times), 3) or not np.isfinite(times).all()
                    or not np.isfinite(positions).all()):
                raise ValueError('trajectory needs finite, ordered positions starting at t=0')
            vertical = 2 if self.frame == 'initial_base' else 1
            if np.ptp(positions[:, vertical]) > 1e-8:
                raise ValueError('dynamic tracking requires horizontal motion at a fixed height')
            # Zero endpoint derivatives at every waypoint give C1 motion and
            # explicit pauses; obstacle speed never jumps at a segment boundary.
            self.curves[name] = CubicHermiteSpline(times, positions, np.zeros_like(positions))
            self.times[name] = times
        self.duration = max(times[-1] for times in self.times.values())

    def sample(self, name, time_s):
        times = self.times[name]
        t = float(np.clip(time_s, 0, times[-1]))
        curve = self.curves[name]
        return curve(t), curve(t, 1) if 0 < time_s < times[-1] else np.zeros(3)

    def to_initial_base(self, position, spawn):
        if self.frame == 'initial_base':
            return np.asarray(position, float)
        from scipy.spatial.transform import Rotation
        native = np.asarray(position)[[0, 2, 1]] * [1, -1, 1]
        origin = np.array([spawn['x'], -spawn['z'], spawn['height']])
        return Rotation.from_euler('z', spawn['yaw'] - np.pi / 2).inv().apply(native - origin)


def combo_trajectories(directory):
    """Every existing combo JSON, numbered order then its action variant."""
    paths = [path for path in Path(directory).glob('shot_combo*.json')
             if re.fullmatch(r'shot_combo[0-9]+(?:_actions)?\.json', path.name)]
    paths.sort(key=lambda path: (int(re.search(r'[0-9]+', path.stem).group()), '_actions' in path.stem))
    if not paths:
        raise ValueError('no combo trajectories found')
    return paths
