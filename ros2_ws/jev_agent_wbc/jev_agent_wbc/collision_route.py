"""Static planar search supplies a soft, collision-free MPC position guide.

Camera translations need not be feasible base positions. Project them onto the
free-space component reachable from the initial base, never through a wall by
successively pushing against overlapping mesh circles. Execution still uses
the whole-body constraints and swept collision certificate.
"""
import heapq

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import label
from scipy.spatial import cKDTree


class CollisionRouteError(RuntimeError):
    """The conservative planar model has no guide from the measured start."""


class CollisionRoute:
    resolution = .025

    def __init__(self, collision, trajectory, initial):
        times = np.linspace(0, trajectory.duration,
                            max(3, int(trajectory.duration / 1.) + 1))
        p0 = trajectory.sample(0)[0]
        self.initial = np.array(initial[:2], dtype=float)
        points = np.array([self.initial + trajectory.sample(t)[0][:2] - p0[:2]
                           for t in times])
        self.circles = collision.planar_circles(points)
        self.centers = np.array([c for c, _ in self.circles]).reshape(-1, 2)
        self.radii = np.array([r for _, r in self.circles])
        self.duration = trajectory.duration
        self.projected_points = 0
        self.max_projection_m = 0.
        if self.circles:
            self._build_grid(points)
            # The grid component is fixed for the whole shot. A point inside a
            # dense obstacle union is projected to reachable space in one step.
            for k in range(1, len(points)):
                node = self._nearest_node(points[k])
                projected = self._coord(node)
                if not self.safe_segment(points[k], projected):
                    distance = float(np.linalg.norm(projected - points[k]))
                    self.projected_points += 1
                    self.max_projection_m = max(self.max_projection_m, distance)
                    points[k] = projected
        expanded_p = [points[0]]
        expanded_t = [times[0]]
        for i in range(1, len(points)):
            path = self.search(points[i - 1], points[i])
            lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            if lengths[-1] < 1e-10:
                expanded_p.append(points[i])
                expanded_t.append(times[i])
                continue
            for j in range(1, len(path)):
                if lengths[j] - lengths[j - 1] <= 1e-10:
                    continue
                expanded_p.append(path[j])
                expanded_t.append(times[i - 1] + lengths[j] / lengths[-1] * (times[i] - times[i - 1]))
        # Cubic interpolation can cut into obstacles between certified edges.
        self.curve = interp1d(expanded_t, np.array(expanded_p), axis=0)

    def safe_segment(self, a, b):
        if not len(self.radii):
            return True
        v = b - a
        alpha = np.clip((self.centers - a) @ v / max(v @ v, 1e-12), 0., 1.)
        delta = a + alpha[:, None] * v - self.centers
        return bool(np.all(np.sum(delta * delta, axis=1) >= self.radii ** 2))

    def _coord(self, node):
        return self.origin + self.resolution * np.asarray(node)

    def reachable(self, point):
        """Can this point connect to the initial base's free component?"""
        if not len(self.radii):
            return True
        return (self.safe_segment(point, point)
                and self.safe_segment(point, self._coord(self._nearest_node(point))))

    def _nearest_node(self, point):
        _, index = self.free_tree.query(point)
        return tuple(self.free_nodes[index])

    def _build_grid(self, points):
        if not self.safe_segment(self.initial, self.initial):
            raise CollisionRouteError('initial base lies inside the conservative planar guide envelope')
        # One domain for the entire trajectory, including room to go around
        # obstacle ends. The old per-edge 0.7 m box could exclude a valid detour.
        self.origin = np.minimum(points.min(axis=0), (self.centers - self.radii[:, None]).min(axis=0)) - .7
        high = np.maximum(points.max(axis=0), (self.centers + self.radii[:, None]).max(axis=0)) + .7
        shape = np.ceil((high - self.origin) / self.resolution).astype(int) + 1
        grid = np.indices(shape).reshape(2, -1).T
        coords = self.origin + self.resolution * grid
        free = np.ones(len(grid), dtype=bool)
        # Every point on a grid edge is within half a diagonal of an endpoint;
        # this reserve certifies all adjacent edges without Python circle loops.
        reserve = self.resolution / np.sqrt(2)
        for center, radius in self.circles:
            free &= np.sum((coords - center) ** 2, axis=1) >= (radius + reserve) ** 2
        components, _ = label(free.reshape(tuple(shape)))
        candidates = np.flatnonzero(free & (np.linalg.norm(coords - self.initial, axis=1) <= 3 * self.resolution))
        candidates = sorted(candidates, key=lambda i: np.linalg.norm(coords[i] - self.initial))
        seed = next((i for i in candidates if self.safe_segment(self.initial, coords[i])), None)
        if seed is None:
            raise CollisionRouteError('no free planar guide grid adjacent to initial base')
        component = components[tuple(grid[seed])]
        self.free = components == component
        self.free_nodes = grid[self.free.ravel()]
        self.free_tree = cKDTree(self.origin + self.resolution * self.free_nodes)

    def search(self, start, goal):
        if self.safe_segment(start, goal):
            return np.array([start, goal])
        source, target = self._nearest_node(start), self._nearest_node(goal)
        if not (self.safe_segment(start, self._coord(source))
                and self.safe_segment(self._coord(target), goal)):
            raise CollisionRouteError('planar guide endpoint cannot connect to reachable grid')
        queue = [(0., 0., source)]
        costs, parents = {source: 0.}, {}
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost > costs[node]:
                continue
            if node == target:
                chain = [goal, self._coord(node)]
                while node != source:
                    node = parents[node]
                    chain.append(self._coord(node))
                chain.append(start)
                chain.reverse()
                out, i = [chain[0]], 0
                while i < len(chain) - 1:
                    j = len(chain) - 1
                    while j > i + 1 and not self.safe_segment(chain[i], chain[j]):
                        j -= 1
                    out.append(chain[j])
                    i = j
                return np.array(out)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxt = node[0] + dx, node[1] + dy
                if (min(nxt) < 0 or nxt[0] >= self.free.shape[0]
                        or nxt[1] >= self.free.shape[1] or not self.free[nxt]):
                    continue
                score = cost + self.resolution * np.hypot(dx, dy)
                if score < costs.get(nxt, np.inf):
                    costs[nxt], parents[nxt] = score, node
                    heapq.heappush(queue, (score + np.linalg.norm(self._coord(nxt) - self._coord(target)), score, nxt))
        raise CollisionRouteError('no planar collision guide found in reachable grid')

    def sample(self, t):
        return self.curve(np.clip(t, 0., self.duration))
