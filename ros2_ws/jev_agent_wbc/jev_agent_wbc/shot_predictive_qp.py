"""Shot-aware receding-horizon allocation and constrained 50 Hz camera WBC.

A condensed LTV kinematic MPC predicts the nine-state nonholonomic system.
Its first control supplies a soft allocation reference to the fast constrained
QP. Both optimizers operate on the *same* eight whole-body controls. No inverse
or post-solve actuator clipping is used. This is not collision avoidance.
"""
import time
import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER
from .whole_body_qp import WholeBodyQP, whole_body_kinematics
from .arm_comfort import COMFORT_POSE, COMFORT_REACH, posture_velocity


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def integration_matrix(yaw):
    b = np.zeros((9, 8))
    b[:2, 0] = [np.cos(yaw), np.sin(yaw)]
    b[2:, 1:] = np.eye(7)
    return b


def remove_box_redundancy(matrix, lower, upper):
    """Drop only rows implied by the first eight actuator box constraints.

    Every retained feasible command also satisfies a removed row. No nearest-k
    approximation or obstacle deletion is involved; callers audit the full QP.
    """
    if len(matrix) < 8 or not np.array_equal(matrix[:8], np.eye(8)):
        return matrix, lower, upper
    lo, hi = lower[:8], upper[:8]
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
        return matrix, lower, upper
    positive, negative = np.maximum(matrix, 0.), np.minimum(matrix, 0.)
    minimum = positive@lo+negative@hi
    maximum = positive@hi+negative@lo
    redundant = (minimum >= lower+1e-10) & (maximum <= upper-1e-10)
    redundant[:8] = False
    keep = ~redundant
    return matrix[keep], lower[keep], upper[keep]


def pose_jacobian(state, base_z):
    p, r, j = whole_body_kinematics(state, base_z)
    c = np.zeros((6, 9))
    c[:2, :2] = np.eye(2)
    c[:, 2:] = j[:, 1:]
    return p, r, c


def refine_execution_qp(hessian, gradient, matrix, lower, upper, initial):
    """Solve the same small convex QP when dense active rows stall ADMM.

    SLSQP receives analytic derivatives and normalized constraints. Its result
    must still pass the caller's original, unscaled hard residual gate.
    """
    scale = max(1., np.max(np.abs(hessian)), np.max(np.abs(gradient)))
    H, g = hessian/scale, gradient/scale
    norms = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
    A, lo, hi = matrix/norms[:, None], lower/norms, upper/norms
    equal = lo == hi
    low, high = np.isfinite(lo) & ~equal, np.isfinite(hi) & ~equal
    rows = np.vstack([A[low], -A[high]])
    offsets = np.r_[lo[low], -hi[high]]
    constraints = [dict(type='ineq', fun=lambda u: rows@u-offsets, jac=lambda u: rows)]
    if np.any(equal):
        constraints.append(dict(type='eq', fun=lambda u: A[equal]@u-lo[equal],
                                jac=lambda u: A[equal]))
    return minimize(lambda u: .5*u@H@u+g@u, initial,
                    jac=lambda u: H@u+g, method='SLSQP', constraints=constraints,
                    options=dict(ftol=1e-12, maxiter=100))


class ShotPredictiveQP(WholeBodyQP):
    def preview_control(self, *args):
        pass

    def preview_collision(self, *args):
        """Stage 3 extension hook; Stage 2 has no obstacle terms."""

    def tracking_scale(self, state, base_z):
        return 1.

    def execution_collision(self, state, base_z, base_rotation, dt, hessian, gradient, A, lo, hi, brake=False):
        return hessian, gradient, A, lo, hi

    def __init__(self, limits=None, *, preview=True, horizon_steps=8, horizon_dt=.4,
                 replan_s=.2, blend_s=2., tracking_weight=10.):
        super().__init__(limits)
        if not 2 <= horizon_steps <= 20 or not .1 <= horizon_dt <= .6:
            raise ValueError('invalid MPC horizon')
        if replan_s <= 0 or blend_s <= 0:
            raise ValueError('replan/blend durations must be positive')
        if not np.isfinite(tracking_weight) or tracking_weight <= 0:
            raise ValueError('tracking_weight must be positive and finite')
        self.tracking_weight = float(tracking_weight)
        self.tracking_gain = 4.
        self.preview = preview
        self.n = horizon_steps
        self.horizon_dt = horizon_dt
        self.replan_s, self.blend_s = replan_s, blend_s
        self.trajectory = None
        self.reference = np.zeros(8)
        self.filtered_reference = np.zeros(8)
        self.last_plan = -np.inf
        self.plan = None
        self.home = None
        self.translating_shots = set()
        self.preparing = False
        self.path_rate = 1.
        self.reference_base_z = 0.
        self.comfort_cache = {}
        self.directions = {}
        self.plan_info = dict(mpc_violation=0., mpc_solve_s=0., mpc_iterations=0,
                              predicted_terminal_yaw=0., preview_active=0., mpc_update_s=0.,
                              mpc_stationarity=0., mpc_retried=0)
        self.context_override = None
        self.direction_costs = (0., 0.)
        self.fast_max_iter = 4000
        self.fast_eps = 1e-8
        self.fast_infeasibility_eps = 1e-4

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory
        self.translating_shots = {
            i for i, shot in enumerate(trajectory.shots)
            if np.linalg.norm(np.ptp([trajectory.sample(t)[0] for t in
                np.linspace(shot['start_s'], shot['end_s'], 9)], axis=0)) > .02
        }

    def motion_weights(self, t):
        weights = self.weights(t)
        # A target-locked Pan arc translates the camera. Its semantic label
        # must not apply the stationary-Pan base suppression or remove posture
        # and heading preparation at the Orbit/Pan junction.
        if self.context_override is None and any(
                i in self.translating_shots and shot['type'] == 'pan'
                and shot['start_s']-self.blend_s/2 <= t <= shot['end_s']+self.blend_s/2
                for i, shot in enumerate(self.trajectory.shots)):
            weights['moveout'] += weights['pan']
            weights['pan'] = 0.
        return weights

    def set_shot_input(self, context):
        """Optional live shot input. Times refer to trajectory elapsed seconds."""
        required = ('shot_type', 'shot_phase', 'next_shot', 'remaining_s')
        if not all(k in context for k in required):
            raise ValueError('shot input requires shot_type, shot_phase, next_shot, remaining_s')
        if context['shot_type'] not in ('neutral', 'orbit', 'pan', 'moveout') or context['next_shot'] not in ('neutral', 'orbit', 'pan', 'moveout'):
            raise ValueError('unknown shot type')
        if not 0 <= float(context['shot_phase']) <= 1 or not np.isfinite(float(context['remaining_s'])) or float(context['remaining_s']) < 0:
            raise ValueError('invalid shot phase/remaining_s')
        if not np.isfinite(float(context.get('received_s', 0.))):
            raise ValueError('invalid shot timestamp')
        self.context_override = dict(context)
        self.context_override.setdefault('received_s', 0.)
        self.last_plan = -np.inf

    def context(self, t):
        if self.context_override is None:
            return self.trajectory.shot_context(t)
        value = self.context_override
        end = float(value['received_s']) + float(value['remaining_s'])
        duration = max(float(value['remaining_s'])/max(1-float(value['shot_phase']), 1e-6), .02)
        return dict(shot_type=value['shot_type'] if t < end else value['next_shot'],
                    shot_phase=float(np.clip(1-(end-t)/duration, 0, 1)), next_shot=value['next_shot'],
                    segment_index=-1, start_s=end-duration, end_s=end)

    def weights(self, t):
        if self.context_override is None:
            return self.trajectory.shot_weights(t, self.blend_s)
        v = self.context_override
        end = v['received_s']+v['remaining_s']
        s = np.clip((t-end)/self.blend_s+.5, 0., 1.)
        a = s*s*(3-2*s)
        w = dict(neutral=0., orbit=0., pan=0., moveout=0.)
        w[v['shot_type']] += 1-a
        w[v['next_shot']] += a
        return w

    def stationary_transition(self, t):
        """Return fixed-pose windows; moving automatic connectors use tracking."""
        window = getattr(self.trajectory, 'transition', lambda _: None)(t)
        if window is None:
            return None
        key = (id(self.trajectory), window['start_s'], window['end_s'])
        cache = getattr(self, '_stationary_windows', {})
        if key not in cache:
            first_p, first_r, _ = self.trajectory.sample(window['start_s'])
            stationary = True
            for probe in np.linspace(window['start_s'], window['end_s'], 5)[1:]:
                p, r, _ = self.trajectory.sample(probe)
                stationary &= (np.linalg.norm(p-first_p) < 1e-5
                               and np.linalg.norm(r-first_r) < 1e-5)
            cache[key] = bool(stationary)
            self._stationary_windows = cache
        return window if cache[key] else None

    def hold_weight(self, t):
        window = self.stationary_transition(t)
        if window is None:
            return 0.
        edge = min(.5, (window['end_s']-window['start_s'])/2)
        fraction = float(np.clip(min(t-window['start_s'], window['end_s']-t)/edge, 0., 1.))
        return fraction*fraction*(3.-2.*fraction)

    def heading(self, t, yaw, state=None):
        ctx = self.context(t)
        tangent = self.trajectory.tangent(t)
        if tangent is None:
            return yaw, 1
        if ctx['shot_type'] == 'moveout' or ctx['segment_index'] in self.translating_shots:
            key = (ctx['segment_index'], ctx['end_s'])
            if key not in self.directions:
                # Compare forward/reverse orientation effort once per shot. Lock
                # the branch to avoid sign chatter as the horizon moves forward.
                candidates = [(abs(wrap(tangent-yaw)), 1), (abs(wrap(tangent+np.pi-yaw)), -1)]
                self.directions[key] = min(candidates)[1]
            direction = self.directions[key]
            return tangent+(np.pi if direction < 0 else 0.), direction
        return tangent, 1

    def allocation_weights(self, weights, q):
        # A soft Pan penalty, relaxed as yaw joints approach their hard limits.
        margin = np.minimum(q-JOINT_LOWER, JOINT_UPPER-q)
        reserve = min(margin[[0, 3, 5]])
        relief = np.clip(reserve/.3, .02, 1.)**2
        pan, travel = weights['pan'], weights['orbit']+weights['moveout']
        return (np.array([.02, .02, .12, .08, .08, .08, .08, .04])*travel
                + np.array([10., 8., .03, .02, .02, .02, .02, .004])*pan*np.r_[relief, relief, np.ones(6)]
                + .002)

    def comfort_pose(self):
        # A slight elbow bend avoids treating the fully straight reset arm as
        # the preferred posture. Camera height/orientation remain primary tasks.
        return COMFORT_POSE.copy()

    def heading_error(self, heading, state):
        error = wrap(heading-state[2])
        # Near a half-turn both rotation senses are comparable. Unwind the
        # arm's yaw joints instead of taking the slightly shorter rotation
        # that drives them into their stops and loses the next camera pose.
        if abs(error) > 2.5 and error*(state[3]+state[8]) < 0:
            error -= np.sign(error)*2*np.pi
        return error

    def comfort_reach_limit(self, state, target_p, t):
        return COMFORT_REACH

    def comfort_reference(self, state, t, *, yaw=None, initial_q=None):
        # A stationary frame is requested at several path times in one tick.
        # Cache the actual IK inputs so heading/reach/parking share that solve.
        # Measured branch seeds and every pose/height change invalidate it.
        p, r, _ = self.trajectory.sample(t)
        yaw = float(state[2] if yaw is None else yaw)
        height = float(p[2]-self.reference_base_z)
        key = (yaw, height, tuple(r.flat),
               None if initial_q is None else tuple(initial_q))
        if key not in self.comfort_cache:
            if len(self.comfort_cache) > 32:
                self.comfort_cache.clear()
            from .arm_comfort import preferred_posture
            self.comfort_cache[key] = preferred_posture(yaw, r, height, initial_q=initial_q)
        return self.comfort_cache[key]

    def make_plan(self, state, t, base_z, commanded_q):
        update_start = time.perf_counter()
        # Enumerate the two direction branches when a new MoveOut first enters
        # the horizon. Both candidates solve the same constrained whole-body
        # problem; objective values include their constant terms. Lock the
        # selected feasible branch for the shot, rather than switching per tick.
        pending = None
        for tk in ([t+(k+1)*self.horizon_dt*self.path_rate for k in range(self.n)] if self.preview else [t]):
            ctx = self.context(tk)
            key = (ctx['segment_index'], ctx['end_s'])
            if (ctx['shot_type'] == 'moveout' or ctx['segment_index'] in self.translating_shots) and key not in self.directions:
                pending = key
                break
        if pending is None:
            self._make_plan(state, t, base_z, commanded_q)
            self.plan_info['mpc_update_s'] = time.perf_counter()-update_start
            return
        old_plan = None if self.plan is None else self.plan.copy()
        candidates = []
        for direction in (1, -1):
            self.directions[pending] = direction
            self.plan = old_plan
            try:
                objective = self._make_plan(state, t, base_z, commanded_q)
                candidates.append((objective, direction, self.plan.copy(), self.reference.copy(), dict(self.plan_info)))
            except RuntimeError:
                pass
        if not candidates:
            del self.directions[pending]
            raise RuntimeError('both MoveOut MPC direction candidates infeasible')
        # At a stopped junction both costs can be numerically identical before
        # the next motion enters the horizon. Resolve that tie by orientation
        # effort, rather than accidentally committing to forward by list order.
        tangent = self.trajectory.tangent(tk)
        def score(candidate):
            angle = 0. if tangent is None else wrap(tangent+(np.pi if candidate[1] < 0 else 0.)-state[2])
            return candidate[0]+1e-5*angle*angle
        objective, direction, self.plan, self.reference, self.plan_info = min(candidates, key=score)
        self.directions[pending] = direction
        self.direction_costs = tuple(next((c[0] for c in candidates if c[1] == d), -1.) for d in (1, -1))
        self.plan_info['mpc_update_s'] = time.perf_counter()-update_start

    def _make_plan(self, state, t, base_z, commanded_q):
        start = time.perf_counter()
        self.reference_base_z = base_z
        n, h = self.n, self.horizon_dt
        nv = 8*n
        nominal = np.tile(self.previous, (n, 1)) if self.plan is None else self.plan.copy()
        H, g = np.eye(nv)*1e-5, np.zeros(nv)
        constant = 0.
        matrices, lows, highs = [], [], []
        xbar = state.copy()
        affine, sensitivity = state.copy(), np.zeros((9, nv))
        lim = self.limits

        def cost(matrix, target, weight):
            nonlocal H, g, constant
            w = np.broadcast_to(np.asarray(weight), (matrix.shape[0],))
            H += matrix.T @ (w[:, None]*matrix)
            g -= matrix.T @ (w*target)
            constant += .5*float(np.sum(w*np.asarray(target)**2))

        def constrain(matrix, lo, hi):
            matrices.append(matrix)
            lows.append(np.broadcast_to(lo, (matrix.shape[0],)))
            highs.append(np.broadcast_to(hi, (matrix.shape[0],)))

        for k in range(n):
            xstart, astart, sstart = xbar.copy(), affine.copy(), sensitivity.copy()
            select = np.zeros((8, nv)); select[:, 8*k:8*k+8] = np.eye(8)
            self.preview_control(select, nominal[k], cost, constrain)
            tk = t+k*h*self.path_rate
            w = self.motion_weights(tk if self.preview else t)
            aw = self.allocation_weights(w, xbar[3:])
            cost(select, np.zeros(8), aw*h)
            difference = select.copy()
            prev = self.previous if k == 0 else np.zeros(8)
            if k:
                difference[:, 8*(k-1):8*k] -= np.eye(8)
            cost(difference, prev, .15/h)
            constrain(select, -lim.velocity, lim.velocity)
            constrain(lim.wheels@select, -lim.wheel_speed, lim.wheel_speed)
            constrain(difference, prev-lim.acceleration*h, prev+lim.acceleration*h)
            wp = lim.wheels@prev
            constrain(lim.wheels@difference, wp-lim.wheel_acceleration*h, wp+lim.wheel_acceleration*h)
            # Linearize x+=h*S(yaw)*u about the warm-start nonlinear rollout.
            B = integration_matrix(xbar[2])
            A = np.eye(9)
            A[:2, 2] += h*nominal[k, 0]*np.array([-np.sin(xbar[2]), np.cos(xbar[2])])
            f = xbar+h*B@nominal[k]
            affine = A@affine + f-A@xbar-h*B@nominal[k]
            sensitivity = A@sensitivity + h*B@select
            xbar = f
            self.preview_collision(xstart, nominal[k], h, base_z, astart, sstart,
                                   affine, sensitivity, cost, constrain, tk+h*self.path_rate)
            p, r, C = pose_jacobian(xbar, base_z)
            tp, tr, current_twist = self.trajectory.sample(tk+h*self.path_rate if self.preview else t)
            if not self.preview:
                # Matched ablation: same horizon, dynamics, weights, smoothing
                # and update rate; only future trajectory/shot information is
                # removed. Extrapolate the currently observed camera twist.
                tp = tp+(k+1)*h*current_twist[:3]
                tr = Rotation.from_rotvec((k+1)*h*current_twist[3:]).as_matrix()@tr
            error = np.r_[tp-p, Rotation.from_matrix(tr@r.T).as_rotvec()]
            # Preview trades camera error against a reachable base guide and
            # collision penalties. Scaling its camera term like the fast QP
            # suppresses necessary detours and posture preparation. Preserve
            # that horizon allocation; increase execution accuracy separately.
            cost(C@sensitivity, error-C@(affine-xbar), np.array([1800.]*3+[600.]*3)
                 *h*self.tracking_scale(xbar, base_z))
            heading, _ = self.heading(tk+h*self.path_rate if self.preview else t, xbar[2], state=xbar)
            heading_weight = 8.*(w['orbit']+w['moveout'])
            cost(sensitivity[2:3], [self.heading_error(heading, xbar)-(affine[2]-xbar[2])], heading_weight*h)
            # Keep the arm's reference configuration for translation shots.
            # Pan retains redundancy but does not have to centre every joint.
            # Prefer the initial pitch/elbow branch for travelling shots.
            # Pure minimum-speed allocation can straighten the arm and lose
            # vertical camera authority while trying to align the base. This
            # soft preference is not a singularity-avoidance guarantee.
            posture = np.array([3., 2., 2., 2., 2., 3.])
            preferred_q = self.comfort_reference(xbar, tk+h*self.path_rate if self.preview else t)[0]
            cost(sensitivity[3:], preferred_q-affine[3:], posture*h)
            for offset, margin in ((np.zeros(6), 0.), (commanded_q-state[3:], lim.position_margin)):
                constrain(sensitivity[3:], JOINT_LOWER+margin-affine[3:]-offset,
                          JOINT_UPPER-margin-affine[3:]-offset)
            # Conservative piecewise-linear stopping-distance envelope over the
            # entire speed range. Secants upper-bound dq²/(2a) in each interval.
            # Includes the predicted state after this control interval.
            for j in range(6):
                vmax, acc = lim.velocity[j+2], lim.acceleration[j+2]
                edges = np.linspace(0., vmax, 5)
                for sign in (-1., 1.):
                    for a, b in zip(edges[:-1], edges[1:]):
                        slope, intercept = (a+b)/(2*acc), -a*b/(2*acc)
                        row = sign*sensitivity[3+j:4+j]+slope*sign*select[j+2:j+3]
                        bound = (JOINT_UPPER[j]-affine[3+j] if sign > 0
                                 else affine[3+j]-JOINT_LOWER[j])
                        # Covers both measured and integrated servo target paths.
                        bound -= max(sign*(commanded_q[j]-state[3+j])+lim.position_margin, 0.)
                        constrain(row, -np.inf, bound-intercept)
        A = np.vstack(matrices); lo, hi = np.concatenate(lows), np.concatenate(highs)
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(H)), q=g, A=sparse.csc_matrix(A), l=lo, u=hi,
                     verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=15000, polish=True)
        solver.warm_start(x=nominal.ravel())
        result = solver.solve()
        if result.info.status_val not in (1, 2) or result.x is None or not np.all(np.isfinite(result.x)):
            raise RuntimeError('MPC failed: '+result.info.status)
        violation = float(max(0., np.max(lo-A@result.x), np.max(A@result.x-hi)))
        # Independently validate primal feasibility and normalized stationarity,
        # including OSQP's "solved inaccurate" status. A status string alone is
        # neither sufficient to accept a plan nor a reason to reject a feasible
        # converged soft allocation reference near an active joint bound.
        hx, aty = H@result.x, A.T@result.y
        stationarity = float(np.max(np.abs(hx+g+aty))/max(1., np.max(np.abs(hx)),
                            np.max(np.abs(g)), np.max(np.abs(aty))))
        retried = 0
        if violation > 1e-6 or stationarity > 1e-5:
            # Tighten a marginal numerical solve instead of weakening the hard
            # residual gate or clipping a candidate into feasibility.
            solver.update_settings(eps_abs=1e-8, eps_rel=1e-8, max_iter=30000)
            result = solver.solve()
            retried = 1
            if (result.info.status_val not in (1, 2) or result.x is None
                    or not np.all(np.isfinite(result.x))):
                raise RuntimeError('MPC retry failed: '+result.info.status)
            violation = float(max(0., np.max(lo-A@result.x), np.max(A@result.x-hi)))
            hx, aty = H@result.x, A.T@result.y
            stationarity = float(np.max(np.abs(hx+g+aty))/max(1., np.max(np.abs(hx)),
                                np.max(np.abs(g)), np.max(np.abs(aty))))
        if not np.isfinite(stationarity) or violation > 1e-6 or stationarity > 1e-5:
            raise RuntimeError(f'MPC constraint residual {violation}')
        self.plan = result.x.reshape(n, 8)
        self.reference = self.plan[0].copy()
        terminal = affine+sensitivity@result.x
        self.plan_info = dict(mpc_violation=violation, mpc_solve_s=time.perf_counter()-start,
                             mpc_stationarity=stationarity,
                             mpc_retried=retried,
                             mpc_iterations=int(result.info.iter), predicted_terminal_yaw=float(terminal[2]),
                             preview_active=float(self.preview and self.context(t)['end_s']-t < n*h))
        self.last_plan = t
        return float(.5*result.x@H@result.x+g@result.x+constant)

    def replan_due(self, trajectory_time):
        return trajectory_time-self.last_plan >= self.replan_s*max(.15, self.path_rate)-1e-8

    def camera_task(self, error, feedforward, hold):
        desired = np.asarray(feedforward)+self.tracking_gain*error
        task = self.tracking_weight*(1.+4.*hold)*np.array([1000., 1000., 1000., 180., 180., 180.])
        return desired, task

    def solve(self, state, target_p, target_r, feedforward, dt, *, base_z=0., base_rotation=None,
              commanded_q=None, brake=False, trajectory_time=0., preparing=False):
        control_start = time.perf_counter()
        if self.trajectory is None:
            raise ValueError('set_trajectory must be called before shot-aware solve')
        if not np.isfinite(dt) or not 0 < dt <= .1:
            raise ValueError('invalid controller dt')
        state = np.asarray(state)
        qcmd = state[3:] if commanded_q is None else np.asarray(commanded_q)
        if self.home is None:
            self.home = state[3:].copy()
        t = trajectory_time
        self.time_s = t
        self.reference_base_z = base_z
        self.preparing = preparing
        ctx = self.context(t)
        replanned = not brake and not preparing and self.replan_due(t)
        if replanned:
            self.make_plan(state, t, base_z, qcmd)
        # Only the soft allocation reference is filtered; executed commands
        # still come directly from the hard-constrained QP. This avoids a
        # sample-and-hold allocation kick at each slower preview update.
        self.filtered_reference += (1.-np.exp(-dt/.3))*(self.reference-self.filtered_reference)
        p, r, j = whole_body_kinematics(state, base_z, base_rotation)
        error = np.r_[target_p-p, Rotation.from_matrix(target_r@r.T).as_rotvec()]
        hold = self.hold_weight(t)
        desired, task = self.camera_task(error, feedforward, hold)
        weights = self.motion_weights(t)
        allocation = self.allocation_weights(weights, state[3:])
        travel = weights['orbit']+weights['moveout']
        # The fast tracker must retain the planned base allocation. With only
        # a tiny base regularizer it can repeatedly undo the preview's heading
        # preparation while compensating small model/servo errors with base yaw.
        allocation[:2] += np.array([1., 1.])*travel
        # Strong camera task plus a soft reference from the entire future path.
        # Suppression weights remain soft: hard joint feasibility wins in Pan.
        # Camera tracking must dominate soft base-guide and posture preferences.
        # At the old scale the 1200-weight base objective could hold the camera
        # almost 10 cm away even in free space. This execution-task scale is
        # independent of preview allocation; collision/actuator bounds stay hard.
        # Preparing the base is secondary to the stationary camera task. Ramp
        # its authority smoothly inside the hold, so parking/arm recovery does
        # not spend centimetres of camera motion on a nominally fixed frame.
        hessian = j.T@(task[:, None]*j)+np.diag(allocation)+.15*np.eye(8)
        gradient = -j.T@(task*desired)-allocation*self.filtered_reference-.15*self.previous
        # Recenter in camera-task nullspace: a compensating base motion lets
        # the arm recover reserve without asking the camera to leave its path.
        nullspace = np.eye(8)-np.linalg.pinv(j, rcond=1e-4)@j
        preferred_q = self.comfort_reference(state, t)[0]
        arm_velocity, comfort_info = posture_velocity(state[3:], self.limits.velocity[2:], preferred_q)
        preferred_velocity = np.r_[0., 0., arm_velocity]
        secondary = nullspace.T@np.diag([1., 1., 4., 2., 2., 3., 2., 4.])@nullspace
        hessian += secondary
        gradient -= secondary@preferred_velocity
        # A small soft preference also acts when hard bounds have removed the
        # task nullspace. Otherwise a joint can stay pinned at its stop forever.
        posture_weight = np.array([8., 6., 6., 4., 6., 8.])
        hessian[2:, 2:] += np.diag(posture_weight)
        gradient[2:] -= posture_weight*arm_velocity
        reach_limit = self.comfort_reach_limit(state, target_p, t)
        comfort_info['arm_reach_limit_m'] = reach_limit
        shoulder = state[:2]+.098*np.array([np.cos(state[2]), np.sin(state[2])])
        offset = p[:2]-shoulder
        reach = np.linalg.norm(offset)
        if reach > reach_limit:
            row = np.r_[0., 0., offset@j[:2,2:]/max(reach, 1e-8)]
            hessian += 800.*np.outer(row, row)
            gradient += 800.*2.*(reach-reach_limit)*row
        heading, direction = self.heading(t, state[2], state=state)
        # The executed QP must honor the same forward/reverse heading branch
        # as the MPC; a weak velocity reference lets arm compensation rotate
        # the base onto the opposite branch and invalidate every warm start.
        heading_weight = 12.*travel
        hessian[1, 1] += heading_weight
        turn_scale = .5*self.limits.velocity[1]
        turn_scale += hold*(min(.2, turn_scale)-turn_scale)
        turn_reference = turn_scale*np.tanh(2.*self.heading_error(heading, state)/turn_scale)
        gradient[1] -= heading_weight*turn_reference
        if preparing:
            # Move off the straight reset singularity before advancing the
            # camera path. The folded seed has the same initial camera pose.
            hessian = j.T@(task[:,None]*j)+np.diag([50.,50.,80.,80.,80.,80.,80.,80.])
            gradient = -j.T@(task*desired)
            gradient[2:] -= 80.*2.*(self.comfort_pose()-state[3:])
        if brake:
            hessian, gradient = np.eye(8), np.zeros(8)
        A, lo, hi = self.bounds(state[3:], dt, commanded_q)
        if np.any(lo > hi):
            bad = np.flatnonzero(lo > hi)
            raise RuntimeError('infeasible hard bounds before collision QP: outside braking envelope; '
                               f'rows={bad.tolist()}, gaps={(lo[bad]-hi[bad]).tolist()}')
        hessian, gradient, A, lo, hi = self.execution_collision(
            state, base_z, base_rotation, dt, hessian, gradient, A, lo, hi, brake)
        if np.any(lo > hi):
            raise RuntimeError('infeasible hard bounds: state outside braking envelope')
        full_A, full_lo, full_hi = A, lo, hi
        A, lo, hi = remove_box_redundancy(A, lo, hi)
        solver = osqp.OSQP()
        begin = time.perf_counter()
        solver.setup(P=sparse.csc_matrix(np.triu(hessian)), q=gradient, A=sparse.csc_matrix(A), l=lo, u=hi,
                     verbose=False, eps_abs=self.fast_eps, eps_rel=self.fast_eps,
                     eps_prim_inf=self.fast_infeasibility_eps, eps_dual_inf=self.fast_infeasibility_eps,
                     max_iter=self.fast_max_iter, polish=True)
        solver.warm_start(x=self.previous)
        result = solver.solve()
        retried = 0
        retry_statuses = (1, 2, osqp.constant('OSQP_MAX_ITER_REACHED'))
        candidate = None if result.x is None else np.asarray(result.x, dtype=float)
        if (self.fast_eps > 1e-8 and result.info.status_val in retry_statuses
                and candidate is not None and np.all(np.isfinite(candidate))):
            residual = max(0., np.max(full_lo-full_A@candidate), np.max(full_A@candidate-full_hi))
            if result.info.status_val != 1 or residual > 1e-6:
                # Dense mesh halfspaces do not always polish successfully at
                # the coarse tolerance. Refine before applying the unchanged
                # hard residual gate; never publish or clip that first result.
                solver.update_settings(eps_abs=1e-8, eps_rel=1e-8)
                result = solver.solve()
                retried = 1
        solution = None if result.x is None else np.asarray(result.x, dtype=float)
        status = result.info.status
        solved = result.info.status_val == 1
        fast_solver = 'osqp'
        finite = solution is not None and np.all(np.isfinite(solution))
        violation = (float(max(0., np.max(full_lo-full_A@solution), np.max(full_A@solution-full_hi)))
                     if finite else np.inf)
        if (self.fast_eps > 1e-8 and result.info.status_val in retry_statuses
                and (not solved or violation > 1e-6)):
            refined = refine_execution_qp(hessian, gradient, A, lo, hi, self.previous)
            if refined.success and np.all(np.isfinite(refined.x)):
                solution, solved, fast_solver = refined.x, True, 'slsqp'
                violation = float(max(0., np.max(full_lo-full_A@solution), np.max(full_A@solution-full_hi)))
            else:
                status += '; refinement: ' + refined.message
        if not solved or solution is None or not np.all(np.isfinite(solution)):
            self.failure_problem = dict(state=state.tolist(), previous=self.previous.tolist(), dt=dt,
                commanded_q=qcmd.tolist(), matrix=full_A.tolist(), lower=full_lo.tolist(), upper=full_hi.tolist(),
                status=status, hessian=hessian.tolist(), gradient=gradient.tolist())
            raise RuntimeError('shot WBC failed: '+status)
        if violation > 1e-6:
            raise RuntimeError(f'WBC constraint residual {violation}')
        self.previous = solution.copy()
        nextq = qcmd+dt*solution[2:]
        info = dict(position_error=float(np.linalg.norm(error[:3])), orientation_error=float(np.linalg.norm(error[3:])),
                    qp_violation=violation, solve_s=time.perf_counter()-begin,
                    fast_retried=retried, fast_solver=fast_solver,
                    qp_total_rows=len(full_A), qp_solver_rows=len(A),
                    command_position_violation=float(max(0, np.max(JOINT_LOWER-nextq), np.max(nextq-JOINT_UPPER))),
                    command_diff_drive_violation=0., solver_iterations=int(result.info.iter),
                    shot_type=ctx['shot_type'], shot_phase=ctx['shot_phase'], next_shot=ctx['next_shot'],
                    heading_target=float(heading), heading_error=float(wrap(heading-state[2])),
                    moveout_direction=direction, **self.plan_info)
        info.update(direction_cost_forward=self.direction_costs[0], direction_cost_reverse=self.direction_costs[1])
        info.update(comfort_info)
        info['arm_preparing'] = int(preparing)
        info.update(mpc_replanned=int(replanned), controller_s=time.perf_counter()-control_start)
        info.update({f'weight_{name}': float(value) for name, value in weights.items()})
        info.update({f'preview_{name}': float(v) for name, v in zip(['v', 'omega']+[f'dq{i}' for i in range(1, 7)], self.reference)})
        return solution.copy(), info
