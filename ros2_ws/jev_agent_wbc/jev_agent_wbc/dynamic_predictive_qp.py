"""Independent dynamic KC-WBC planner; the static trajfollow is untouched."""
import numpy as np
import osqp
from scipy import sparse

from .collision_route import CollisionRoute
from .dynamic_collision import DynamicCollision
from .obstacle_predictive_qp import ObstaclePredictiveQP
from .shot_predictive_qp import integration_matrix
from .shot_preview_worker import AsyncObstaclePredictiveQP
from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER


def compute_dynamic_preview(snapshot, state, t, base_z, commanded_q):
    controller = DynamicPredictiveQP(snapshot['limits'], scene=snapshot['scene'],
        horizon_steps=snapshot['n'], horizon_dt=snapshot['horizon_dt'],
        replan_s=snapshot['replan_s'], blend_s=snapshot['blend_s'], tracking_weight=snapshot['tracking_weight'])
    controller.set_trajectory(snapshot['trajectory'])
    for name in ('previous', 'home', 'plan', 'directions', 'context_override', 'direction_costs',
                 'path_rate', 'route', 'route_initial', 'plan_failures', 'transition_entries', 'guide_costs'):
        setattr(controller, name, snapshot[name])
    controller.set_time(snapshot['physical_time'])
    controller.make_plan(state, t, base_z, commanded_q)
    result = {name: getattr(controller, name) for name in ('plan', 'reference', 'directions', 'direction_costs',
        'plan_info', 'collision_terms', 'coupled_terms', 'plan_failures', 'route', 'planner_braking',
        'transition_entries', 'guide_costs')}
    request = snapshot.get('recovery_request')
    if request is not None and controller.route is not None:
        position, goal = request
        route = controller.route
        start = position if route.safe_segment(position, position) else route._coord(route._nearest_node(position))
        try:
            result.update(recovery_path=route.search(start, goal), recovery_goal=goal)
        except RuntimeError:
            result.update(recovery_path=None, recovery_goal=None)
    return result


class DynamicPredictiveQP(ObstaclePredictiveQP):
    def __init__(self, *args, scene, **kwargs):
        super().__init__(*args, scene=scene, collision_mode='kc', **kwargs)
        self.tracking_gain = 8.
        # Runtime execution preferences reserve joint authority and avoid a
        # large camera residual immediately demanding full recovery speed.
        # These are robot-control objectives, never Jev permission or scoring
        # thresholds; all authored feedforward and hard limits are preserved.
        self.runtime_tracking_profile = dict(position_recovery_speed_m_s=.5*self.limits.velocity[0],
            joint_reserve_rad=.15, joint_reserve_weight=10000.)
        # Dense, almost parallel moving-surface rows can produce a false
        # infeasibility certificate at OSQP's default 1e-4 tolerance. A tighter
        # certificate lets the existing same-constraint refinement run; all
        # command and collision residual gates remain unchanged.
        self.fast_infeasibility_eps = 1e-8
        self.physical_time = float(scene.get('observation_stamp', 0.))
        self.plan_path_time = 0.
        self.last_plan_physical_time = -np.inf
        self.collision = DynamicCollision(scene)

    def set_trajectory(self, trajectory):
        profile = getattr(trajectory, 'document', {}).get('dynamic_camera_tracking', {})
        allowed = {'orientation_weight_scale', 'position_gain', 'position_recovery_speed_m_s',
                   'joint_reserve_rad', 'joint_reserve_weight'}
        if not isinstance(profile, dict) or set(profile) - allowed:
            raise ValueError('invalid dynamic_camera_tracking profile')
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not np.isfinite(v) or v <= 0 for v in profile.values()):
            raise ValueError('dynamic_camera_tracking values must be positive and finite')
        if profile.get('joint_reserve_rad', 0.) >= .5*np.min(JOINT_UPPER-JOINT_LOWER):
            raise ValueError('dynamic_camera_tracking joint reserve exceeds joint range')
        super().set_trajectory(trajectory)
        self.camera_tracking_profile = {**self.runtime_tracking_profile, **profile}

    def camera_task(self, error, feedforward, hold):
        desired, task = super().camera_task(error, feedforward, hold)
        profile = getattr(self, 'camera_tracking_profile', {})
        task[3:] *= profile.get('orientation_weight_scale', 1.)
        # Cap only error recovery, preserving the authored shot feedforward.
        # All eight actuator commands are still solved together; no delayed
        # command filter or clipping follows the constrained QP.
        recovery = profile.get('position_gain', self.tracking_gain)*error[:3]
        cap = profile.get('position_recovery_speed_m_s', np.inf)
        recovery *= min(1., cap/max(np.linalg.norm(recovery), 1e-12))
        desired[:3] = np.asarray(feedforward)[:3]+recovery
        return desired, task

    def set_time(self, now):
        self.physical_time = float(now)
        self.collision.set_time(now)
        self.base_reference_cache = None

    def update_collision_map(self, scene):
        self.collision = DynamicCollision(scene, self.physical_time,geometry=self.collision.geometry)
        self.base_reference_cache = None

    def make_plan(self, state, t, base_z, commanded_q):
        self.last_plan_physical_time = self.physical_time
        self.plan_path_time = float(t)
        if self.route is None:
            # Moving obstacles affect the time-dependent whole-body plan and
            # execution constraints; they cannot permanently disconnect the
            # static route or project the original camera reference elsewhere.
            try:
                self.route = CollisionRoute(self.collision.static_collision(), self.trajectory,
                    state if self.route_initial is None else self.route_initial)
            except RuntimeError as error:
                self.plan_failures += 1
                self.plan = None
                self.reference = np.zeros(8)
                self.last_plan = t
                self.plan_info['preview_failure'] = str(error)
                self.planner_braking = self.collision.clearance(state, base_z).min() < self.collision.margin + .02
                return
        return super().make_plan(state, t, base_z, commanded_q)

    def preview_collision(self, x, u, h, z, a0, s0, a1, s1, cost, constrain, future_t):
        current = self.collision
        # State sensitivity contains one eight-control block per preceding
        # physical step, independent of the connector's camera-path rate.
        active = np.flatnonzero(np.any(np.abs(s0) > 1e-12, axis=0))
        step = 0 if not len(active) else int(active[-1] // 8 + 1)
        offset = step * h
        self.collision = current.at_time(offset)
        try:
            return super().preview_collision(x, u, h, z, a0, s0, a1, s1, cost, constrain, future_t)
        finally:
            self.collision = current

    def execution_collision(self, state, z, br, dt, H, g, A, lo, hi, brake=False):
        profile = getattr(self, 'camera_tracking_profile', {})
        if not brake and not self.preparing and profile.get('joint_reserve_rad', 0.) > 0.:
            # Keep compensation authority before a wrist reaches its stop.
            # This is a soft preference inside the original feasible set;
            # collision, acceleration and physical joint bounds still win.
            reserve = profile['joint_reserve_rad']
            q = np.asarray(state)[3:]
            low = np.maximum(JOINT_LOWER+reserve-q, 0.)
            high = np.maximum(q-JOINT_UPPER+reserve, 0.)
            weight = profile.get('joint_reserve_weight', 10000.)*np.minimum(1., (low+high)/reserve)**2
            H[2:, 2:] += np.diag(weight)
            g[2:] -= weight*2.*(low-high)
        if not self.limits.obstacle_avoidance:
            # CPU ablation: retain the identical camera, posture, joint-reserve
            # and actuator objectives above. Skip all static/dynamic collision
            # rows, obstacle-dependent tracking weights, retreat and braking.
            self.safety_info = dict(safety_constraint_count=0, dynamic_anticipation_rows=0,
                dynamic_escape_omega_target=0., dynamic_braking_rows=0)
            return H, g, A, lo, hi
        tracking_H, tracking_g = H.copy(), g.copy()
        original_count = len(A)
        H, g, A, lo, hi = super().execution_collision(state, z, br, dt, H, g, A, lo, hi, brake)
        self.safety_info['dynamic_anticipation_rows'] = 0
        self.safety_info['dynamic_escape_omega_target'] = 0.
        self.safety_info['dynamic_braking_rows'] = 0
        centers, radii, distances, gradients = self.execution_witnesses
        rows, lower, mask = self.static_barrier_terms(
            state, distances, gradients, self.static_barrier_braking_control)
        motion = self.collision.primitive_motion(state, centers, z, br)
        # d_dot = J_robot B u + n_obstacle v_obstacle - radius_growth.
        # The static implementation contributed the same mask/rows above.
        ids = self.collision.tree.query(centers, k=1)[1]
        moving = self.collision.track_ids[ids] > 0
        if (not brake and not self.preparing and not self.planner_braking
                and (self.route is None or not self.terminal_approach(self.execution_time))):
            # The inherited static controller relaxes camera tracking close to
            # walls. A moving witness must not also amplify our soft retreat by
            # reducing camera authority to 8%. Preserve the relaxation warranted
            # by static furniture and keep every hard collision/actuator row.
            threshold, width = (.10, .05) if self.collision.is_mesh else (.16, .10)
            static_min = float(distances[~moving].min()) if np.any(~moving) else np.inf
            all_activation = np.clip((threshold-float(distances.min()))/width, 0., 1.)
            static_activation = np.clip((threshold-static_min)/width, 0., 1.)
            restore = .92*(all_activation-static_activation)
            H += restore*tracking_H
            g += restore*tracking_g
        # A crossing surface rotates its normal as it passes the base. The
        # static wall gain overconstrains that harmless lateral passage; the
        # relative-motion barrier uses its own positive gain. Hard actuator
        # bounds and the independent continuous sweep still gate execution.
        dynamic_lower = -12. * (distances - self.collision.margin - .006)
        required = np.where(moving[:, None], dynamic_lower, lower) - motion
        lo[original_count:] = required[mask]
        extra = moving[:, None] & ~mask & (rows @ self.previous < required + .12)
        if np.any(extra):
            A = np.vstack([A, rows[extra]])
            lo = np.r_[lo, required[extra]]
            hi = np.r_[hi, np.full(np.sum(extra), np.inf)]
            self.safety_info['safety_constraint_count'] += int(np.sum(extra))
        closest, predicted = np.inf, np.inf
        if np.any(moving):
            closest = float(distances[moving].min())
        translating = moving & (np.linalg.norm(self.collision.velocities[ids], axis=1) > .04)
        if np.any(moving):
            relative_rate = rows @ self.previous + motion
            # A moving witness must not suddenly release the early braking
            # already required when that object was classified as stationary.
            # Tighten the retained barrier toward an anticipatory separation
            # rate, capped by ONE acceleration-feasible escape command. Every
            # original collision and actuator row remains present and checked.
            base_shapes = np.array([i for i, shape in enumerate(self.collision.geometry.shapes) if shape[0] == 0])
            threatened_shapes = range(distances.shape[1])
            early = -1.2*(distances-.08)-motion
            braking_mask = np.zeros(distances.shape, dtype=bool)
            for body in threatened_shapes:
                candidates = np.flatnonzero(moving & (distances[:, body] < .40)
                    & (relative_rate[:, body] < -1.2*(distances[:, body]-.08)+.02))
                if len(candidates):
                    braking_mask[candidates[np.argsort(distances[candidates, body])[:2]], body] = True
            if (np.any(braking_mask) and len(A) >= 8
                    and np.array_equal(A[:8], np.eye(8))):
                urgency = np.where(braking_mask, early-rows@self.previous, -np.inf)
                witness = np.unravel_index(np.argmax(urgency), urgency.shape)
                if urgency[witness] > 0.:
                    normal = rows[witness]
                    escape_control = None
                    for coupled in (False, True):
                        candidate = np.clip(self.previous, lo[:8], hi[:8])
                        axes = 2 if coupled else 1
                        candidate[:axes] = np.where(normal[:axes] >= 0., hi[:axes], lo[:axes])
                        value = A@candidate
                        if np.all(value >= lo-1e-9) and np.all(value <= hi+1e-9):
                            escape_control = candidate
                            break
                    if escape_control is None:
                        # Furniture or another body witness can prevent a
                        # simple maximum-deceleration command. Find one common
                        # control in the ORIGINAL feasible polytope before
                        # tightening any early-braking row.
                        reserve = rows[braking_mask]
                        target = early[braking_mask]
                        hessian = np.eye(8)+1000.*reserve.T@reserve
                        gradient = -self.previous-1000.*reserve.T@target
                        solver = osqp.OSQP()
                        solver.setup(P=sparse.csc_matrix(np.triu(hessian)), q=gradient,
                            A=sparse.csc_matrix(A), l=lo, u=hi, verbose=False,
                            eps_abs=1e-8, eps_rel=1e-8, max_iter=5000, polish=True)
                        solver.warm_start(x=self.previous)
                        result = solver.solve()
                        if (result.info.status_val in (1, 2) and result.x is not None
                                and np.all(np.isfinite(result.x))):
                            value = A@result.x
                            if np.all(value >= lo-1e-7) and np.all(value <= hi+1e-7):
                                escape_control = result.x
                    if escape_control is not None:
                        # Leave numerical room around the common feasible
                        # command; the original hard lower bounds are retained.
                        tighter = np.minimum(early, rows@escape_control-1e-5)
                        selected = braking_mask & (tighter > required+1e-9)
                        if np.any(selected):
                            A = np.vstack([A, rows[selected]])
                            lo = np.r_[lo, tighter[selected]]
                            hi = np.r_[hi, np.full(np.sum(selected), np.inf)]
                            self.safety_info['dynamic_braking_rows'] = int(np.sum(selected))
                            self.safety_info['safety_constraint_count'] += int(np.sum(selected))
        if np.any(translating):
            # Record closing risk over 2.5 physical seconds. This prediction
            # never changes the camera clock; retreat and heading objectives
            # below act on the shared whole-body controls instead.
            predicted = float(np.min(distances[translating] + 2.5 * np.minimum(relative_rate[translating], 0.)))
            # A differential-drive base cannot immediately translate sideways.
            # Prepare an escape heading while the measured mover is still far
            # enough away. Choose forward/reverse alignment with the shoulder
            # yaw reserve required to compensate the base turn at the camera.
            candidates = [(i, body) for i in np.flatnonzero(translating) for body in threatened_shapes
                          if motion[i, body] < -.02 and distances[i, body] < .60]
            if candidates and not brake and not self.preparing and not self.planner_braking:
                i, body = min(candidates, key=lambda pair: distances[pair])
                normal = gradients[i, body, :2]
                track_id = self.collision.track_ids[ids[i]]
                track = next((track for track in self.collision.scene.get('tracks', []) if track['id'] == track_id), None)
                if track is not None:
                    center = np.asarray(track['center'])[:2] + np.asarray(track['velocity'])[:2]*max(0., self.physical_time-self.collision.stamp)
                    normal = state[:2]-center
                length = np.linalg.norm(normal)
                alignment = abs(normal @ np.array([np.cos(state[2]), np.sin(state[2])]))/max(length, 1e-9)
                # A base already able to retreat along its drive axis needs no
                # extra turn. Model centres avoid wheel-witness corner switches.
                if distances[i, body] > .18 and length > .1 and alignment < .60:
                    escape = np.arctan2(normal[1], normal[0])
                    turns = np.arctan2(np.sin(np.array([escape, escape+np.pi])-state[2]),
                                       np.cos(np.array([escape, escape+np.pi])-state[2]))
                    shoulder = state[3]-turns
                    reserve_violation = np.maximum(JOINT_LOWER[0]+.15-shoulder, 0.) + np.maximum(shoulder-JOINT_UPPER[0]+.15, 0.)
                    immediate_limit = ((state[3] > JOINT_UPPER[0]-.35) & (turns < 0.)) | ((state[3] < JOINT_LOWER[0]+.35) & (turns > 0.))
                    turn = float(turns[np.argmin(np.abs(turns)+40.*reserve_violation+10000.*immediate_limit)])
                    omega = .4*np.tanh(2.*turn/.4)
                    weight = 4000.*float(np.clip((.60-distances[i, body])/.35, 0., 1.))
                    H[1, 1] += weight
                    g[1] -= weight*omega
                    self.safety_info['dynamic_escape_omega_target'] = float(omega)
            # Begin making space before acceleration bounds make the final
            # barrier infeasible. All body rows use the same eight controls;
            # arm compensation can preserve the camera while the base yields.
            reserve_rows, reserve_targets = [], []
            for body in range(distances.shape[1]):
                closing = relative_rate[:, body] < -.02
                candidates = np.flatnonzero(translating & (closing | (motion[:, body] < -.02))
                    & (distances[:, body] < .40))
                if not len(candidates):
                    continue
                nearest = candidates[np.argsort(distances[candidates, body])[:2]]
                reserve_rows.extend(rows[nearest, body])
                reserve_targets.extend(np.maximum(0., -motion[nearest, body]
                    - 1.2*(distances[nearest, body]-.08)))
            if reserve_rows and not brake and not self.preparing:
                reserve = np.asarray(reserve_rows)
                target = np.asarray(reserve_targets)
                # A single physical surface can contribute many primitive/voxel
                # witnesses. Normalize its soft cost so voxel density cannot
                # multiply the retreat priority and displace the camera.
                # A threatened nonholonomic chassis needs stronger early
                # retreat than a redundant arm link. Use measured geometry,
                # never shot names or semantic tolerances, to choose this soft
                # allocation priority. All hard barrier rows remain identical.
                base_threat = bool(len(base_shapes) and np.any(distances[translating][:, base_shapes] < .40))
                weight = (6000. if base_threat else 1500.)/len(reserve_rows)
                H += weight*reserve.T@reserve
                g -= weight*reserve.T@target
            self.safety_info['dynamic_anticipation_rows'] = len(reserve_rows)
        self.safety_info.update(dynamic_tracks=len(self.collision.scene.get('tracks', [])),
            dynamic_constraint_count=int(np.sum(moving[:, None] & mask) + np.sum(extra)
                                         + self.safety_info['dynamic_braking_rows']),
            dynamic_clearance_m=closest, dynamic_predicted_clearance_m=predicted,
            dynamic_observation_age_s=max(0., self.physical_time - self.collision.stamp))
        return H, g, A, lo, hi

    def replan_due(self, trajectory_time):
        # A slow stationary connector must not slow dynamic-obstacle updates.
        # Async workers accept one request at a time, without a stale backlog.
        return (getattr(self, 'pending', None) is None
                and (not np.isfinite(self.last_plan)
                     or self.physical_time-self.last_plan_physical_time >= self.replan_s-1e-8))


class AsyncDynamicPredictiveQP(AsyncObstaclePredictiveQP, DynamicPredictiveQP):
    def make_plan(self, state, t, base_z, commanded_q):
        if self.pending is not None:
            return
        if self.route_initial is None:
            self.route_initial = np.array(state)
        rebuild = self.execution_clock - self.route_map_time >= 2.
        snapshot = {name: getattr(self, name) for name in ('limits', 'n', 'horizon_dt', 'replan_s', 'blend_s',
            'trajectory', 'previous', 'home', 'plan', 'context_override', 'direction_costs', 'path_rate',
            'tracking_weight', 'route_initial', 'plan_failures', 'physical_time')}
        snapshot.update(scene=self.collision.scene, route=None if rebuild else self.route,
            directions=dict(self.directions), transition_entries=dict(self.transition_entries),
            guide_costs={} if rebuild else dict(self.guide_costs),
            recovery_request=getattr(self, 'recovery_request', None))
        if self.plan is not None and self.applied_time is not None:
            age = max(0., self.execution_clock - self.applied_execution_time)
            grid = np.arange(self.n) * self.horizon_dt
            snapshot['plan'] = np.column_stack([np.interp(grid + age, grid, self.plan[:, j]) for j in range(8)])
            if np.max(np.abs(snapshot['plan'][0] - self.previous) / self.limits.velocity) > .25:
                snapshot['plan'] = None
        if rebuild:
            self.route_map_time = self.execution_clock
        self.pending = self.pool.submit(compute_dynamic_preview, snapshot, np.array(state), t, base_z, np.array(commanded_q))
        self.pending_generation = self.input_generation
        self.pending_time = self.last_plan = t
        self.last_plan_physical_time = self.physical_time
        self.pending_execution_time = self.execution_clock
