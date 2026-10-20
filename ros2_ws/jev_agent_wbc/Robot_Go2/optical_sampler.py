"""Exact pose-only optical sampling of a validated, frozen public TimedTask.

This local legged helper avoids computing and discarding reference twists. It
preserves the public semantic trajectory's interpolation and absolute clock;
it does not admit tasks, modify references, extrapolate, or relax freshness.
Consumers needing feedforward twists must keep LiveTaskSession.reference().
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from jev_agent_wbc.representation import fingerprint, number
from jev_agent_wbc.wbc_protocol import OPTICAL_FROM_SEMANTIC, TimedTask


class OpticalSampler:
    """Cache validated immutable interpolation inputs once at task admission."""

    __slots__ = ('_start', '_end', '_task_id', '_path_id', '_task_sha256',
                 '_times', '_positions', '_slerp', '_semantic_from_optical',
                 '_first', '_last')

    def __init__(self, accepted_task: TimedTask):
        if not isinstance(accepted_task, TimedTask):
            raise ValueError('validated accepted TimedTask required')
        # Reparse exactly once so nested caller mutation cannot alter the
        # accepted clock, knots, coordinates or metadata after this point.
        task = TimedTask.from_dict(accepted_task.to_dict())
        trajectory = task.trajectory
        self._start, self._end = trajectory.start_s, trajectory.end_s
        self._task_id, self._path_id = task.task_id, trajectory.id
        self._task_sha256 = fingerprint(task.to_dict())
        self._times = np.array(trajectory.times_s, dtype=float)
        self._positions = np.array([p.translation_m for p in trajectory.poses])
        self._semantic_from_optical = OPTICAL_FROM_SEMANTIC.rotation.T
        # Interpolate the original semantic quaternions before applying the
        # constant optical rotation. This also preserves pi-arc tie choices.
        self._slerp = Slerp(self._times, Rotation.from_quat(
            [p.quaternion_xyzw for p in trajectory.poses]))
        self._first = self._matrix(0., 0)
        # The public API samples the absolute end using end-start, whose
        # floating-point subtraction can be just below the relative last knot.
        # Cache that identical endpoint rather than silently snapping time.
        end_relative = min(max(self._end-self._start, 0.), self._times[-1])
        self._last = self._matrix(end_relative, len(self._times)-2)
        for array in (self._times, self._positions, self._semantic_from_optical,
                      self._first, self._last):
            array.setflags(write=False)

    @property
    def start_s(self):
        return self._start

    @property
    def end_s(self):
        return self._end

    @property
    def task_id(self):
        return self._task_id

    @property
    def camera_path_id(self):
        return self._path_id

    @property
    def task_sha256(self):
        return self._task_sha256

    def _matrix(self, relative_s, index):
        fraction = ((relative_s-self._times[index]) /
                    (self._times[index+1]-self._times[index]))
        result = np.eye(4)
        result[:3, 3] = ((1.-fraction)*self._positions[index] +
                         fraction*self._positions[index+1])
        result[:3, :3] = self._slerp(relative_s).as_matrix() @ self._semantic_from_optical
        return result

    def sample(self, stamp_s, *, hold_terminal=False):
        """Return fresh T_world_camera_optical, with explicit optional end hold.

        Before-start, nonnumeric and nonfinite times always fail. After-end
        times fail unless hold_terminal=True explicitly requests the unchanged
        original endpoint. Holding never extends or changes the Task itself.
        """
        stamp = number(stamp_s, 'stamp_s', self._start)
        if not isinstance(hold_terminal, bool):
            raise ValueError('hold_terminal must be boolean')
        if stamp > self._end:
            if not hold_terminal:
                raise ValueError('trajectory expired; extrapolation is forbidden')
            return self._last.copy()
        # Keep the same absolute-subtraction clipping as CameraTrajectory.
        relative = min(max(stamp-self._start, 0.), self._times[-1])
        if relative <= 0.:
            return self._first.copy()
        if stamp == self._end or relative >= self._times[-1]:
            return self._last.copy()
        index = min(max(int(np.searchsorted(self._times, relative, side='right'))-1, 0),
                    len(self._times)-2)
        return self._matrix(relative, index)

    __call__ = sample

    def manifest(self):
        return dict(task_id=self._task_id, camera_path_id=self._path_id,
                    task_sha256=self._task_sha256, start_s=self._start, end_s=self._end,
                    output_frame='world_optical_x_right_y_down_z_forward',
                    interpolation='original_linear_translation_shortest_semantic_so3_then_same_optical_rotation',
                    terminal_hold='only_explicit_original_endpoint',
                    supplies_twist=False, camera_reference_modified=False)
