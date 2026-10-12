"""Moving-surface distances and relative-motion swept clearance bounds."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .kc_svsdf import SweptCollision, advance, primitive_sdf


class DynamicCollision(SweptCollision):
    def __init__(self, scene, physical_time=None, uncertainty_rate=.035):
        super().__init__(scene)
        self.observed_centers = self.centers.copy()
        self.observed_radii = self.radii.copy()
        self.velocities = np.asarray(scene.get('surfel_velocities', np.zeros_like(self.centers)), float)
        self.track_ids = np.asarray(scene.get('surfel_track_ids', np.zeros(len(self.centers))), int)
        self.ages = np.asarray(scene.get('surfel_ages', np.zeros(len(self.centers))), float)
        if (self.velocities.shape != self.centers.shape or self.track_ids.shape != self.radii.shape
                or self.ages.shape != self.radii.shape or not np.isfinite(self.velocities).all()
                or not np.isfinite(self.ages).all() or np.any(self.ages < 0)):
            raise ValueError('invalid dynamic surface attributes')
        self.uncertainty_rate = float(uncertainty_rate)
        if not np.isfinite(self.uncertainty_rate) or self.uncertainty_rate < 0:
            raise ValueError('invalid motion uncertainty rate')
        self.radius_rates = np.where(self.track_ids > 0, self.uncertainty_rate, 0.)
        self.stamp = float(scene.get('observation_stamp', 0.))
        self.physical_time = self.stamp
        self.set_time(self.stamp if physical_time is None else physical_time)

    def set_time(self, now):
        self.physical_time = float(now)
        age = max(0., self.physical_time - self.stamp)
        self.centers = self.observed_centers + self.velocities * age
        self.radii = (self.observed_radii
                      + self.radius_rates * (self.ages + age))
        self.tree = cKDTree(self.centers)

    def at_time(self, offset):
        return DynamicCollision(self.scene, self.physical_time + offset, self.uncertainty_rate)

    def static_collision(self):
        mask = self.track_ids == 0
        return SweptCollision({**self.scene, 'surfel_centers': self.centers[mask], 'surfel_radii': self.radii[mask]})

    def primitive_motion(self, state, centers, z=0., base_rotation=None):
        """Explicit obstacle contribution to d_dot for every robot primitive."""
        if not len(centers):
            return np.empty((0, len(self.geometry.shapes)))
        indices = self.tree.query(centers, k=1)[1]
        moving = self.track_ids[indices] > 0
        result = np.zeros((len(centers), len(self.geometry.shapes)))
        if not np.any(moving):
            return result
        centers, indices = centers[moving], indices[moving]
        frames = self.geometry.frames(state, z, base_rotation)[0]
        derivatives = []
        for body, _, kind, size, p, r in self.geometry.shapes:
            bp, br = frames[body]
            center, rotation = bp + br @ p, br @ r
            _, normal = primitive_sdf((centers - center) @ rotation, kind, size, with_gradient=True)
            derivatives.append(np.einsum('ni,ni->n', normal @ rotation.T, self.velocities[indices])
                               - self.radius_rates[indices])
        result[moving] = np.array(derivatives).T
        return result

    def sweep(self, x, u, duration, z=0., tolerance=.003, max_samples=257, base_rotation=None):
        # The distance Lipschitz bound contains BOTH robot surface speed and
        # obstacle speed/uncertainty growth. Static endpoint checks cannot
        # certify an obstacle that crosses a stationary robot mid-interval.
        obstacle_speed = float(np.max(np.linalg.norm(self.velocities, axis=1) + self.radius_rates,
                                      initial=0.))
        speed = abs(u[0]) + np.abs(u[1:]) @ self.geometry.reach + obstacle_speed
        count = min(max_samples, max(3, int(np.ceil(speed * duration / (2 * tolerance))) + 1))
        times = np.linspace(0, duration, count)
        states = [advance(x, u, t) for t in times]
        rotations = ([None] * count if base_rotation is None else
                     [Rotation.from_euler('z', u[1] * t).as_matrix() @ base_rotation for t in times])
        # Include every moving surface even if it starts outside the static
        # nearest-witness neighbourhood. Nearby static surfaces use the same
        # established selection as the static swept oracle.
        omitted_lower = np.inf
        if duration <= .1:
            # Short execution intervals do not need repeated nearest-body SDF
            # searches at every sample. Include ALL witnesses near the initial
            # body origins plus ALL movers, and analytically bound the omitted
            # static points using body extent and maximum surface displacement.
            origins = np.array([p for p, _ in self.geometry.frames(x, z, base_rotation)[0]])
            nearby = self.tree.query_ball_point(origins, 1.5)
            selected = {i for group in nearby for i in group}
            selected.update(np.flatnonzero(self.track_ids > 0).tolist())
            if not selected:
                selected.update(np.atleast_1d(self.tree.query(origins, k=1)[1]).tolist())
            indices = np.array(sorted(selected), dtype=int)
            radii = []
            for _, _, kind, size, offset, _ in self.geometry.shapes:
                bound = (size[0] if kind == 'sphere' else np.linalg.norm(size)/2 if kind == 'box'
                         else np.hypot(size[0], size[1]/2) if kind == 'cylinder'
                         else np.hypot(max(size[:2]), size[2]/2))
                radii.append(np.linalg.norm(offset)+bound)
            static_radius = float(np.max(self.radii[self.track_ids == 0], initial=0.))
            robot_speed = abs(u[0])+np.abs(u[1:])@self.geometry.reach
            omitted_lower = 1.5-max(radii)-static_radius-robot_speed*duration
        else:
            centers, _ = self.active_sequence(states, z, rotations)
            indices = np.unique(np.r_[self.tree.query(centers, k=1)[1], np.flatnonzero(self.track_ids > 0)])
        moving_centers = self.centers[indices][None]+times[:, None, None]*self.velocities[indices][None]
        moving_radii = self.radii[indices][None]+times[:, None]*self.radius_rates[indices][None]
        values = self.geometry.distances_sequence(states, moving_centers, z, rotations)-moving_radii[:, :, None]
        upper = values.min(axis=0)
        error = speed * duration / (2 * (count - 1))
        return dict(lower=np.minimum(upper-error, omitted_lower), upper=upper, error=error, index=values.argmin(axis=0),
                    states=states, times=times, values=values)

    def terms(self, x, u, duration, z=0., coupled=True, influence=.065):
        static = self.static_collision()
        output, filtered = static.terms(x, u, duration, z, coupled, influence) if len(static.centers) else ([], 0)
        indices = np.flatnonzero(self.track_ids > 0)
        if not len(indices):
            return output, filtered
        times = np.linspace(0, duration, 3)
        states = [advance(x, u, t) for t in times]
        centers = [self.centers[indices] + self.velocities[indices] * t for t in times]
        results = [self.clearance_to(state, points, self.radii[indices] + self.radius_rates[indices] * t,
                    z, gradient=True) for state, points, t in zip(states, centers, times)]
        values = np.array([result[0] for result in results])
        gradients = np.array([result[1] for result in results])
        robot_speed = abs(u[0]) + np.abs(u[1:]) @ self.geometry.reach
        speed = robot_speed + np.max(np.linalg.norm(self.velocities[indices], axis=1) + self.radius_rates[indices])
        error = speed * duration / 4
        worst, minima = values.min(axis=0) - error, values.argmin(axis=0)
        for body in range(8):
            nearest = np.argsort(worst[:, body])[:2]
            for obstacle in nearest:
                if worst[obstacle, body] < influence:
                    i = minima[obstacle, body]
                    # A measured moving witness acts on the complete common
                    # nine-state Jacobian; no independent link controls exist.
                    output.append((float(worst[obstacle, body]), gradients[i, obstacle, body],
                                   float(times[i] / duration), int(indices[obstacle]), body))
        return output, filtered
