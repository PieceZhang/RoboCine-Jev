"""Moving-surface distances and relative-motion swept clearance bounds."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .kc_svsdf import SweptCollision, advance, advance_sequence, primitive_sdf


class DynamicCollision(SweptCollision):
    adaptive_sweep = True
    def __init__(self, scene, physical_time=None, uncertainty_rate=None, geometry=None):
        super().__init__(scene,geometry)
        self.observed_centers = self.centers.copy()
        self.observed_radii = self.radii.copy()
        self.velocities = np.asarray(scene.get('surfel_velocities', np.zeros_like(self.centers)), float)
        self.track_ids = np.asarray(scene.get('surfel_track_ids', np.zeros(len(self.centers))), int)
        self.ages = np.asarray(scene.get('surfel_ages', np.zeros(len(self.centers))), float)
        if (self.velocities.shape != self.centers.shape or self.track_ids.shape != self.radii.shape
                or self.ages.shape != self.radii.shape or not np.isfinite(self.velocities).all()
                or not np.isfinite(self.ages).all() or np.any(self.ages < 0)):
            raise ValueError('invalid dynamic surface attributes')
        self.uncertainty_rate = float(scene.get('motion_uncertainty_rate_m_s', .035)
                                      if uncertainty_rate is None else uncertainty_rate)
        if not np.isfinite(self.uncertainty_rate) or self.uncertainty_rate < 0:
            raise ValueError('invalid motion uncertainty rate')
        self.age_radius_rates = np.where(self.track_ids > 0, self.uncertainty_rate, 0.)
        extra_rates=np.asarray(scene.get('surfel_radius_rates',np.zeros(len(self.centers))),float)
        if extra_rates.shape!=self.radii.shape or not np.isfinite(extra_rates).all() or np.any(extra_rates<0):
            raise ValueError('invalid observed occupancy radius rates')
        self.radius_rates = self.age_radius_rates+extra_rates
        self.stamp = float(scene.get('observation_stamp', 0.))
        self.physical_time = self.stamp
        self.set_time(self.stamp if physical_time is None else physical_time)

    def set_time(self, now):
        self.physical_time = float(now)
        age = max(0., self.physical_time - self.stamp)
        self.centers = self.observed_centers + self.velocities * age
        self.radii = (self.observed_radii
                      + self.age_radius_rates*self.ages + self.radius_rates*age)
        self.tree = cKDTree(self.centers)

    def at_time(self, offset):
        # Geometry and the original timestamped observation are immutable.
        # Reuse them across prediction samples; only projected arrays/tree/time
        # belong to the new snapshot. Rebuilding all URDF primitives per probe
        # dominated the configuration predictor's runtime.
        from copy import copy
        projected=copy(self)
        projected.set_time(self.physical_time+offset)
        return projected

    def static_collision(self):
        mask = self.track_ids == 0
        return SweptCollision({**self.scene, 'surfel_centers': self.centers[mask], 'surfel_radii': self.radii[mask]})

    def active_obstacles(self,x,z=0.,base_rotation=None,per_body=8):
        centers,radii=super().active_obstacles(x,z,base_rotation,per_body)
        if not self.is_mesh:return centers,radii
        # A distant center can have a nearby occupied surface when a lost
        # mover's uncertainty grows. Center-only spatial pruning must never
        # hide that volume from the controller before the final swept gate.
        moving=np.flatnonzero(self.track_ids>0)
        if not len(moving):return centers,radii
        indices=np.unique(np.r_[np.atleast_1d(self.tree.query(centers,k=1)[1]),moving])
        return self.centers[indices],self.radii[indices]

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
        _,normal=self.geometry.primitive_arrays(frames,centers,True)
        result[moving]=np.einsum('npi,ni->np',normal,self.velocities[indices])-self.radius_rates[indices,None]
        return result

    def sweep(self, x, u, duration, z=0., tolerance=.003, max_samples=257, base_rotation=None):
        # The distance Lipschitz bound contains BOTH robot surface speed and
        # obstacle speed/uncertainty growth. Static endpoint checks cannot
        # certify an obstacle that crosses a stationary robot mid-interval.
        obstacle_speed = float(np.max(np.linalg.norm(self.velocities, axis=1) + self.radius_rates,
                                      initial=0.))
        speed = abs(u[0]) + np.abs(u[1:]) @ self.geometry.reach + obstacle_speed
        count = min(max_samples, max(3, int(np.ceil(speed * duration / (2 * tolerance))) + 1))
        # Include every moving surface even if it starts outside the static
        # nearest-witness neighbourhood. Nearby static surfaces use the same
        # established selection as the static swept oracle.
        omitted_lower = np.inf
        if duration <= .2+1e-9:
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
            static_radius = float(np.max(self.radii[self.track_ids == 0], initial=0.))
            robot_speed = abs(u[0])+np.abs(u[1:])@self.geometry.reach
            omitted_lower = 1.5-self.geometry.max_body_extent-static_radius-robot_speed*duration
        else:
            times = np.linspace(0, duration, count)
            states = advance_sequence(x,u,times)
            rotations = None if base_rotation is None else Rotation.from_euler('z',u[1]*times).as_matrix()@base_rotation
            centers, _ = self.active_sequence(states, z, rotations)
            indices = np.unique(np.r_[self.tree.query(centers, k=1)[1], np.flatnonzero(self.track_ids > 0)])
        def evaluate(samples):
            times = np.linspace(0, duration, samples)
            states = advance_sequence(x,u,times)
            rotations = None if base_rotation is None else Rotation.from_euler('z',u[1]*times).as_matrix()@base_rotation
            moving_centers = self.centers[indices][None]+times[:, None, None]*self.velocities[indices][None]
            moving_radii = self.radii[indices][None]+times[:, None]*self.radius_rates[indices][None]
            values = self.geometry.distances_sequence(states, moving_centers, z, rotations)-moving_radii[:, :, None]
            upper = values.min(axis=0)
            error = speed * duration / (2 * (samples - 1))
            return dict(lower=np.minimum(upper-error, omitted_lower), upper=upper, error=error, index=values.argmin(axis=0),
                        states=states, times=times, values=values, obstacle_indices=indices)
        if self.adaptive_sweep and duration<=.2+1e-9 and count>3:
            coarse=evaluate(3)
            # The same global Lipschitz bound proves the ENTIRE interval.
            # Far obstacles need no tighter bound; ambiguous intervals retain
            # the original sample budget/tolerance and all selected witnesses.
            if np.min(coarse['lower'])>=self.margin:
                return coarse
        return evaluate(count)

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
