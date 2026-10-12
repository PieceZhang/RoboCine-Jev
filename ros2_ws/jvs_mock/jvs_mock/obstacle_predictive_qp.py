"""Shot MPC with SVSDF / KC-SVSDF and a common constrained safety execution QP."""
import time
import numpy as np
from .shot_predictive_qp import ShotPredictiveQP, integration_matrix, wrap
from .kc_svsdf import SweptCollision


class ObstaclePredictiveQP(ShotPredictiveQP):
    def __init__(self, *args, scene, collision_mode='kc', **kwargs):
        super().__init__(*args, **kwargs)
        if collision_mode not in ('none','per_link','kc'):
            raise ValueError('collision_mode must be none, per_link, kc')
        self.collision_mode = collision_mode
        self.collision = SweptCollision(scene)
        self.collision_terms = 0
        self.coupled_terms = 0
        self.safety_info = {}
        self.plan_failures = 0
        self.planner_braking = False
        self.route = None
        self.route_initial = None
        self.execution_time = 0.
        self.planning = False
        self.transition_entries = {}
        self.guide_costs = {}
        self.relax_trust = False
        self.base_reference_cache = None
        self.fast_max_iter = 12000
        # Dense surfel CBF rows are poorly scaled at 1e-8 even when the final
        # polished solution satisfies every hard row to machine precision.
        self.fast_eps = 1e-5

    def update_collision_map(self, scene):
        """Replace the immutable observation used by execution and future previews."""
        self.collision = SweptCollision(scene)
        self.base_reference_cache = None

    def preview_control(self, select, nominal, cost, constrain):
        if self.collision_mode == 'none': return
        trust=np.array([.08,.25,.25,.25,.25,.25,.25,.25])
        if not self.relax_trust:
            constrain(select,nominal-trust,nominal+trust)
        cost(select,nominal,.3)

    def base_reference(self, state, t):
        # Heading and velocity objectives request this same reference in one
        # control tick. Reuse only exact inputs; do not run the grid search
        # twice or reuse a result after state/scene/shot input has changed.
        def key():
            return (id(self.route), id(self.trajectory), id(self.context_override), float(t),
                    self.reference_base_z, tuple(state), frozenset(self.directions.items()))
        cache_key = key()
        if self.base_reference_cache is not None and self.base_reference_cache[0] == cache_key:
            return self.base_reference_cache[1]
        result = self.follow_transition(t, self._base_reference(state, t))
        self.base_reference_cache = (key(), result)
        return result

    def follow_transition(self, t, reference):
        """Join prepared heading to the moving guide as the next shot starts."""
        shots = getattr(self.trajectory, 'shots', [])
        for previous, following in zip(shots, shots[1:]):
            start = previous.get('motion_end_s', previous['end_s'])
            end = following.get('motion_start_s', following['start_s'])
            key = ('transition_heading', start, end)
            edge = self.blend_s/2
            if end <= t < end+edge and key in self.directions:
                prepared = self.directions[key]
                difference = wrap(reference[1]-prepared)
                turn_key = ('transition_follow_turn', start, end)
                anchor = self.directions.get(turn_key, difference)
                if not self.planning:
                    self.directions.setdefault(turn_key, anchor)
                fraction = float(np.clip((t-end)/edge, 0., 1.))
                blend = fraction*fraction*(3.-2.*fraction)
                return reference[0], float(prepared+blend*(anchor+wrap(difference-anchor))), reference[2]
        return reference

    def _base_reference(self, state, t):
        heading, direction = super().heading(t, state[2])
        if self.route is None:
            return 0., state[2], direction
        transition = self.stationary_transition(t)
        if transition is not None:
            return self.transition_reference(state, t, transition)
        position = self.comfort_base_position(state, t)
        velocity = (self.route.sample(t+.1)-self.route.sample(t-.1))/.2
        tangent = velocity.copy()
        context = self.context(t)
        if np.linalg.norm(tangent) < .002:
            probes = np.linspace(context['start_s'], context['end_s'], 17)
            for probe in sorted(probes, key=lambda value: abs(value-t)):
                tangent = (self.route.sample(probe+.1)-self.route.sample(probe-.1))/.2
                if np.linalg.norm(tangent) >= .002:
                    break
        if np.linalg.norm(tangent) < .002:
            return 0., state[2], direction
        tangent /= np.linalg.norm(tangent)
        angle = np.arctan2(tangent[1], tangent[0])
        key = ('base_guide', context['segment_index'], context['end_s'])
        if key not in self.directions:
            posture_costs = self.guide_posture_costs(context)
            self.directions[key] = min((cost+.025*wrap(angle+(np.pi if d < 0 else 0.)-state[2])**2, d)
                                       for cost, d in zip(posture_costs, (1, -1)))[1]
        direction = self.directions[key]
        if self.terminal_approach(t):
            position = self.route.sample(self.trajectory.duration)
            if not self.route.safe_segment(state[:2], position):
                position = self.recovery_waypoint(state[:2], position)
            displacement = position-state[:2]
            if np.linalg.norm(displacement) <= .035:
                camera_offset = self.trajectory.sample(t)[0][:2]-state[:2]
                return 0., float(np.arctan2(camera_offset[1], camera_offset[0])), direction
            angle = np.arctan2(displacement[1], displacement[0])
            yaw_sum = state[3]+state[8]
            candidates = []
            for candidate in (1, -1):
                heading = angle+(np.pi if candidate < 0 else 0.)
                effort = wrap(heading-state[2])
                candidates.append((abs(effort)+.4*(yaw_sum-effort)**2, candidate, heading))
            _, direction, heading = min(candidates)
            speed_limit = .18 if self.projected_exit(t) else self.limits.velocity[0]
            speed = direction*min(speed_limit, 2.*np.linalg.norm(displacement))
            speed *= max(0., np.cos(wrap(heading-state[2])))**4
            return float(speed), float(heading), direction
        if hasattr(self.route, 'search') and not self.route.safe_segment(state[:2], position):
            # Time-indexed pursuit can cut across a furniture corner when the
            # base lags. Recover through visible free-space waypoints instead
            # of pushing diagonally into the same CBF indefinitely.
            waypoint = self.recovery_waypoint(state[:2], position)
            displacement = waypoint-state[:2]
            heading = np.arctan2(displacement[1], displacement[0])+(np.pi if direction < 0 else 0.)
            speed = direction*min(self.limits.velocity[0], 2.*np.linalg.norm(displacement))
            speed *= max(0., np.cos(wrap(heading-state[2])))
            return float(speed), float(heading), direction
        error = position-state[:2]
        along = float(error@tangent)
        lateral = error-along*tangent
        steering = tangent+3.*lateral
        heading = np.arctan2(steering[1], steering[0])+(np.pi if direction < 0 else 0.)
        speed = direction*(float(velocity@tangent)*self.path_rate+2.*along)
        # Recover a stopped target's lateral error continuously. The former
        # 2.5 cm switch also appeared inside the transition's outgoing guide,
        # repeatedly switching its heading as parking crossed that radius.
        fraction = float(np.clip((np.linalg.norm(error)-.015)/.065, 0., 1.))
        recovery = fraction*fraction*(3.-2.*fraction)
        fraction = float(np.clip(np.linalg.norm(velocity)/.005, 0., 1.))
        recovery *= 1.-fraction*fraction*(3.-2.*fraction)
        if recovery > 0.:
            parking_yaw = np.arctan2(error[1], error[0])+(np.pi if direction < 0 else 0.)
            difference = wrap(parking_yaw-heading)
            key = ('stopped_turn', context['segment_index'], context['end_s'])
            anchor = self.directions.get(key, difference)
            if not self.planning:
                self.directions.setdefault(key, anchor)
            heading += recovery*(anchor+wrap(difference-anchor))
            parking_speed = direction*min(self.limits.velocity[0], 2.*np.linalg.norm(error))
            speed += recovery*(parking_speed-speed)
        speed = np.clip(speed, -self.limits.velocity[0], self.limits.velocity[0])
        speed *= max(0., np.cos(wrap(heading-state[2])))
        return float(speed), float(heading), direction

    def guide_posture_costs(self, context):
        """Cache pose-fit costs; choose direction later from measured heading."""
        key = ('base_guide', context['segment_index'], context['end_s'])
        if key not in self.guide_costs:
            from scipy.spatial.transform import Rotation
            from .arm_comfort import preferred_posture, COMFORT_POSE
            from .jvs_kinematics import forward_kinematics
            costs = []
            for candidate in (1, -1):
                values = []
                for probe in np.linspace(context['start_s'], context['end_s'], 17):
                    velocity = self.route.sample(probe+.1)-self.route.sample(probe-.1)
                    if np.linalg.norm(velocity) < .0004:
                        continue
                    yaw = np.arctan2(velocity[1], velocity[0])+(np.pi if candidate < 0 else 0.)
                    p, r, _ = self.trajectory.sample(probe)
                    height = p[2]-self.reference_base_z
                    preferred, _ = preferred_posture(yaw, r, height)
                    local_p, local_r = forward_kinematics(preferred)
                    body_r = Rotation.from_euler('z', yaw).as_matrix()
                    rotation_error = Rotation.from_matrix(r@(body_r@local_r).T).as_rotvec()
                    # A comfortable posture can still miss camera height or
                    # orientation at the real joint limits. Include its task
                    # residual using the same scales as the posture IK.
                    values.append(np.sum((preferred-COMFORT_POSE)**2)
                                  +2500.*(height-local_p[2])**2+400.*(rotation_error@rotation_error))
                costs.append(float(np.mean(values)) if values else 0.)
            self.guide_costs[key] = tuple(costs)
        return self.guide_costs[key]

    def recovery_waypoint(self, position, goal):
        if self.route.safe_segment(position, position):
            return self.route.search(position, goal)[1]
        # The live body may be outside the conservative planar envelope while
        # still clear in 3D. The nearest free grid cell can lie sideways in the
        # narrow gap between two pieces of furniture. Stopping pursuit there
        # traps a differential-drive base. Continue to the first corner of its
        # free-space route, so it backs out before turning. The execution QP
        # and swept whole-body certificate still validate every actual step.
        start = self.route._coord(self.route._nearest_node(position))
        path = self.route.search(start, goal)
        return path[min(1, len(path)-1)]

    def transition_reference(self, state, t, window):
        """Use the stationary camera interval to park for the upcoming motion."""
        future = window['reference_s']
        key = ('transition', window.get('start_s'), window['end_s'])
        tangent = self.route.sample(future+.1)-self.route.sample(future-.1)
        # Once execution commits the direction, a nonzero route tangent fixes
        # its preparation heading. Re-evaluating the future soft base goal on
        # every hold tick repeated an IK solve whose heading was discarded.
        direction = self.directions.get(key)
        if direction is None or np.linalg.norm(tangent) <= 1e-6:
            _, future_heading, future_direction = self._base_reference(state, future)
            direction = self.directions.setdefault(key, future_direction)
        desired_yaw = (np.arctan2(tangent[1], tangent[0])
                       +(np.pi if direction < 0 else 0.)
                       if np.linalg.norm(tangent) > 1e-6 else future_heading)
        self.directions.setdefault(('transition_heading', window.get('start_s'), window['end_s']),
                                   float(desired_yaw))
        if (self.collision.scene.get('format') == 'jvs_lidar_surfels/v1'
                and self.projected_exit(future)):
            # A live map may project the stopped junction onto the far side
            # of furniture. Pursuing that point during the camera hold spends
            # the following MoveOut before its shot even begins. Prepare the
            # heading here; leave the escape translation to the MoveOut.
            return 0., float(desired_yaw), direction
        p, r, _ = self.trajectory.sample(t)
        _, offset = self.comfort_reference(state, t, yaw=desired_yaw)
        goal = p[:2]-offset[:2]
        if not self.route.reachable(goal):
            goal = self.route.sample(window['end_s'])
        if not self.route.safe_segment(state[:2], goal):
            goal = self.recovery_waypoint(state[:2], goal)
        delta = goal-state[:2]
        # Small offsets only need along-heading preparation for the next shot.
        # The old 7 cm switch and per-tick forward/reverse choice made the base
        # alternate between incompatible headings while the camera was stopped.
        along = float(delta@np.array([np.cos(desired_yaw), np.sin(desired_yaw)]))
        fraction = float(np.clip((abs(along)-.015)/.055, 0., 1.))
        approach = fraction*fraction*(3.-2.*fraction)
        heading = desired_yaw
        speed = .08*np.tanh(1.5*along/.08)*approach*max(0., np.cos(wrap(heading-state[2])))**4
        # Large lateral offsets need a real parking move before heading
        # preparation, otherwise the wrist can reach its stop at the next shot.
        # Commit these phases from measured execution only, then blend out of
        # parking once. Position noise must not restart the arrival manoeuvre.
        distance = np.linalg.norm(delta)
        parking_key = ('transition_parking', window.get('start_s'), window['end_s'])
        parking = self.directions.get(parking_key)
        angle = np.arctan2(delta[1], delta[0])
        if parking is None:
            choices = [(abs(wrap(angle+(np.pi if d < 0 else 0.)-state[2]))
                        +.4*(state[3]+state[8]-wrap(angle+(np.pi if d < 0 else 0.)-state[2]))**2,
                        d) for d in (1, -1)]
            parking = min(choices)[1] if distance > .08 else 0
            if not self.planning:
                self.directions[parking_key] = parking
        if parking:
            parking_yaw = angle+(np.pi if parking < 0 else 0.)
            parking_speed = parking*.08*np.tanh(1.5*distance/.08)*max(0., np.cos(wrap(parking_yaw-state[2])))**4
            arrival_key = ('transition_arrival', window.get('start_s'), window['end_s'])
            arrival = self.directions.get(arrival_key)
            # A camera-preserving QP may prevent all requested parking travel.
            # Reserve the second half of the existing window for turning even
            # in that case, instead of postponing it until the outgoing edge.
            turn_deadline = window['start_s']+(window['end_s']-window['start_s'])/2
            if arrival is None and (distance < .04 or t >= turn_deadline):
                turn = self.heading_error(desired_yaw, state)-wrap(parking_yaw-state[2])
                arrival = (t, parking_yaw, parking_speed, turn)
                if not self.planning:
                    self.directions[arrival_key] = arrival
            if arrival is None:
                speed, heading = parking_speed, parking_yaw
            else:
                phase = float(np.clip((t-arrival[0])/.5, 0., 1.))
                heading = arrival[1]+phase*phase*(3.-2.*phase)*arrival[3]
                phase = float(np.clip((t-arrival[0])/.1, 0., 1.))
                speed = arrival[2]*(1.-phase*phase*(3.-2.*phase))
        # Also match the adjacent guides at both ends of the stationary window.
        # This is a reference blend inside the constrained QP, not a filter or
        # modification of its executed commands or collision certificates.
        start = window.get('start_s', t)
        edge = min(self.blend_s/2, (window['end_s']-start)/2)
        adjacent = None
        if edge > 0. and t < start+edge:
            entry_key = (start, window['end_s'])
            adjacent = self.transition_entries.get(entry_key)
            if adjacent is None:
                adjacent = self._base_reference(state, start-1e-6)
                # The incoming guide belongs to the actual entry state. Later
                # parking motion can cross its stopped-position threshold;
                # recomputing it would inject a new turn into the same blend.
                # Future MPC probes must not latch a hypothetical entry state.
                if not self.planning:
                    self.transition_entries[entry_key] = adjacent
            fraction = float(np.clip((t-start)/edge, 0., 1.))
            turn_key = ('transition_entry_turn', start, window['end_s'])
        elif edge > 0. and t > window['end_s']-edge:
            adjacent = self._base_reference(state, window['end_s']+1e-6)
            # The outgoing stopped-position guide can require a different
            # orientation to finish lateral recovery. Complete heading
            # preparation first; follow_transition joins that guide after the
            # boundary, while the next camera motion gains speed.
            adjacent = (adjacent[0], desired_yaw, adjacent[2])
            fraction = float(np.clip((window['end_s']-t)/edge, 0., 1.))
            turn_key = ('transition_exit_turn', start, window['end_s'])
        if adjacent is not None:
            blend = fraction*fraction*(3.-2.*fraction)
            speed = adjacent[0]+blend*(speed-adjacent[0])
            difference = wrap(heading-adjacent[1])
            anchor = self.directions.setdefault(turn_key, difference)
            heading = adjacent[1]+blend*(anchor+wrap(difference-anchor))
        return float(speed), float(heading), direction

    def transition_preparing(self, t):
        window = self.stationary_transition(t)
        if window is None:
            return False
        suffix = (window['start_s'], window['end_s'])
        return bool(self.directions.get(('transition_parking', *suffix), 0)
                    and ('transition_arrival', *suffix) not in self.directions)

    def comfort_base_position(self, state, t):
        if self.terminal_approach(t):
            return self.route.sample(self.trajectory.duration)
        position = self.route.sample(t)
        p, r, twist = self.trajectory.sample(t)
        offset = self.comfort_reference(state, t)[1]
        fraction = float(np.clip(np.linalg.norm(twist[:3])/.005, 0., 1.))
        if fraction < 1.:
            # At stopped endpoints the neutral IK seed can change branch on a
            # tiny chassis turn, moving the base goal by centimetres. Recover
            # that soft goal on the measured branch, then smoothly restore the
            # travelling posture guide as the next camera motion gains speed.
            measured = self.comfort_reference(state, t, initial_q=state[3:])[1]
            blend = fraction*fraction*(3.-2.*fraction)
            offset = measured+blend*(offset-measured)
        target = p[:2]-offset[:2]
        if hasattr(self.route, 'safe_segment') and self.route.safe_segment(position, target):
            return target
        # A blocked straight segment is not proof of base unreachability.
        # Prefer the compact-arm target when a detour in this component exists.
        if hasattr(self.route, 'reachable') and self.route.reachable(target):
            return target
        return position

    def terminal_approach(self, t):
        if self.projected_exit(t):
            return True
        context = self.context(t)
        if (context.get('next_shot') != 'neutral'
                or t < (context['start_s']+context['end_s'])/2):
            return False
        p, r, twist = self.trajectory.sample(t)
        end_p, end_r, _ = self.trajectory.sample(self.trajectory.duration)
        from scipy.spatial.transform import Rotation
        return (np.linalg.norm(p-end_p) < .03
                and np.linalg.norm(Rotation.from_matrix(r@end_r.T).as_rotvec()) < .05
                and np.linalg.norm(twist[:3]) < .05)

    def projected_exit(self, t):
        """A final MoveOut must leave a furniture-projected junction monotonically.

        A succession of nearest projections along a bed edge changes the base
        goal as the camera leaves the bed. Use the reachable final goal for this
        particular escape, rather than repeatedly approaching and retreating.
        Ordinary feasible MoveOut paths retain time-indexed tracking.
        """
        if self.route is None or not hasattr(self.route, 'initial'):
            return False
        context = self.context(t)
        if context['shot_type'] != 'moveout' or context.get('next_shot') != 'neutral':
            return False
        start = context['start_s']
        nominal = self.route.initial+self.trajectory.sample(start)[0][:2]-self.trajectory.sample(0.)[0][:2]
        return bool(np.linalg.norm(self.route.sample(start)-nominal) > .15)

    def heading(self, t, yaw, state=None):
        if self.route is not None and state is not None and self.collision_mode != 'none':
            _, heading, direction = self.base_reference(state, t)
            return heading, direction
        if self.route is None and self.collision_mode != 'none':
            return yaw, 1
        return super().heading(t, yaw, state=state)

    def comfort_reach_limit(self, state, target_p, t):
        limit = super().comfort_reach_limit(state, target_p, t)
        if self.route is None:
            return limit
        if self.projected_exit(t):
            # The camera position is soft in this geometrically infeasible
            # section. Keep the arm compact while the base performs MoveOut.
            return limit
        # Permit additional reach required by the obstacle-projected base
        # reference, not reach caused by the current base lagging behind it.
        needed = np.linalg.norm(target_p[:2]-self.comfort_base_position(state, t))-.098
        return max(limit, needed)

    def tracking_scale(self, state, base_z):
        if self.collision_mode == 'none': return 1.
        d=float(np.min(self.collision.clearance(state,base_z)))
        return float(np.clip((d-.025)/.12,.08,1.))

    def preview_collision(self, x, u, h, z, a0, s0, a1, s1, cost, constrain, future_t):
        if self.collision_mode == 'none': return
        cost(s1[:2],self.comfort_base_position(x, future_t)-a1[:2],1200.*h)
        terms, coupled = self.collision.terms(x,u,h,z,coupled=self.collision_mode=='kc')
        self.collision_terms += len(terms); self.coupled_terms += coupled
        from .kc_svsdf import advance
        for distance,gradient,tau,_,_ in terms:
            nominal=advance(x,u,tau*h)
            affine=(1-tau)*a0+tau*a1
            sensitivity=(1-tau)*s0+tau*s1
            violation=max(.065-distance,0.)
            # Gauss–Newton quadratic model of a cubic hinge; no artificial
            # independent-link velocities. Every term acts on the SAME 8N controls.
            row=gradient@sensitivity
            target=.065-distance-gradient@(affine-nominal)
            cost(row[None,:], [target], 12000.*max(violation,.005)*h)

    def make_plan(self,state,t,base_z,commanded_q):
        if self.collision_mode == 'none':
            return super().make_plan(state,t,base_z,commanded_q)
        started=time.perf_counter()
        self.collision_terms=self.coupled_terms=0
        self.planning = True
        self.reference_base_z = base_z
        # Successive relinearization uses the identical dynamic hard constraints.
        # The first solve includes forward/reverse branch enumeration.
        try:
            if self.route is None:
                from .collision_route import CollisionRoute
                self.route=CollisionRoute(self.collision,self.trajectory,
                    state if self.route_initial is None else self.route_initial)
            # A direction choice evaluates dozens of arm IK postures. Prepare
            # their costs in the preview worker; execution then makes its cheap
            # direction choice using the actual heading at that junction.
            for shot in self.trajectory.shots:
                self.guide_posture_costs(self.context((shot['start_s']+shot['end_s'])/2))
            for relaxed in (False, True):
                self.relax_trust = relaxed
                try:
                    super().make_plan(state,t,base_z,commanded_q)
                    self._make_plan(state,t,base_z,commanded_q)
                    break
                except RuntimeError:
                    if relaxed:
                        raise
                    # A stale nominal can make an algorithmic trust box
                    # incompatible with braking at a joint boundary. Retry
                    # without that box; actuator/position bounds stay hard.
                    self.plan = None
            self.planner_braking = False
            self.plan_info['preview_failure'] = ''
        except RuntimeError as error:
            self.plan_failures += 1
            self.reference=np.zeros(8)
            self.plan=None
            self.last_plan=t
            # A preview failure far from the clearance boundary must not freeze
            # the independent fast tracking QP. Its CBF and swept certificate
            # remain hard constraints; reserve braking for a genuinely close
            # state where continuing without a predictive plan is unsafe.
            minimum=float(np.min(self.collision.clearance(state,base_z)))
            self.planner_braking = minimum < self.collision.margin + .02
            self.plan_info['preview_failure'] = str(error)
        finally:
            self.planning = False
        self.plan_info['guide_projected_points'] = 0 if self.route is None else self.route.projected_points
        self.plan_info['guide_max_projection_m'] = 0. if self.route is None else self.route.max_projection_m
        self.plan_info['mpc_trust_relaxed'] = int(self.relax_trust)
        self.plan_info['mpc_update_s']=time.perf_counter()-started

    def static_barrier_terms(self, state, distances, gradients, braking_control=None):
        """Shared static rows, with acceleration-feasible early LiDAR braking.

        A newly observed voxel can reduce the nearest clearance by one cell.
        Begin braking one surfel radius before the original velocity barrier.
        The extra band only tightens that barrier and is capped by one common
        hardware-feasible braking control. Its activation therefore cannot
        remove that feasible braking command, even when a witness changes.
        Dynamic subclasses reuse this exact mask before adding relative motion.
        """
        # For a wall, following d_dot >= -gain*d requires braking at
        # gain*closing_speed. The old 1.2/s gain demands 0.6 m/s^2 at the
        # 0.5 m/s base limit, exceeding its 0.5 m/s^2 hard acceleration bound.
        # Start slowing earlier in dense meshes and leave discrete-time reserve.
        # ROS feedback can arrive in larger increments than a control tick.
        # Retain braking authority for that lag instead of using almost the
        # entire acceleration budget just to stay on the velocity barrier.
        gain = (min(1.2, .6*float(np.min(self.limits.acceleration/self.limits.velocity)))
                if self.collision.is_mesh else 1.2)
        rows = gradients @ integration_matrix(state[2])
        lower = -gain*(distances-self.collision.margin-.006)
        if self.collision.is_mesh:
            # Nearest-k witnesses can switch to an already violated CBF while
            # acceleration bounds prevent braking in one tick. Inspect all
            # local surfels and activate approaching halfspaces beforehand,
            # with a 0.1 m/s derivative reserve.
            mask=(distances < .08) | (rows @ self.previous < lower + .1)
            count=min(4,len(distances))
            for body in range(distances.shape[1]):
                nearest=np.argpartition(distances[:,body],count-1)[:count]
                mask[nearest,body]=True
            if braking_control is not None:
                braking_rate = rows @ braking_control
                if np.all(braking_rate[mask] >= lower[mask]-1e-9):
                    radius = float(self.collision.scene.get('surfel_radius',
                        np.min(self.collision.radii)))
                    lower = np.maximum(lower, np.minimum(lower+gain*radius, braking_rate))
                    mask |= rows @ self.previous < lower+.1
        else:
            # The analytic scenes contain only a handful of primitives, so keep
            # every CBF row present before it can become acceleration-infeasible.
            mask=np.ones(distances.shape,dtype=bool)
        return rows, lower, mask

    @staticmethod
    def hardware_braking_control(A, lo, hi):
        """One common braking command, checked against every actuator row."""
        if len(A) < 8 or not np.array_equal(A[:8], np.eye(8)):
            return None
        command = np.clip(np.zeros(8), lo[:8], hi[:8])
        value = A @ command
        return command if np.all(value >= lo-1e-9) and np.all(value <= hi+1e-9) else None

    def execution_collision(self,state,z,br,dt,H,g,A,lo,hi,brake=False):
        centers,radii=self.collision.active_obstacles(state,z,br,per_body=None)
        distances,gradients=self.collision.geometry.primitive_distances(state,centers,z,br)
        distances-=radii[:,None]
        minimum=float(np.min(distances))
        self.safety_info=dict(clearance_m=minimum,collision_active_pairs=int(np.sum(distances<.08)),
                             safety_constraint_count=0)
        if self.collision_mode == 'none': return H,g,A,lo,hi
        self.static_barrier_braking_control = self.hardware_braking_control(A, lo, hi)
        all_rows, all_lower, mask = self.static_barrier_terms(
            state, distances, gradients, self.static_barrier_braking_control)
        rows = all_rows[mask]
        # Common CBF for BOTH collision modes, evaluated at actual measured pose.
        # 6mm reserve covers servo/model/discrete-time errors in ordinary shots.
        lower=all_lower[mask]
        A=np.vstack([A,rows]);lo=np.r_[lo,lower];hi=np.r_[hi,np.full(len(rows),np.inf)]
        self.safety_info['safety_constraint_count']=len(rows)
        if brake or self.planner_braking:return np.eye(8),np.zeros(8),A,lo,hi
        if self.preparing:
            return H,g,A,lo,hi
        # Near obstacles retain the safe predictive allocation and relax camera
        # tracking. This is still ONE constrained QP, never post-solve clipping.
        # A dense scene mesh usually has nearby static surfaces even when the
        # commanded motion is safely separated from them.  Preserve camera
        # authority until the robot is genuinely close to the clearance
        # boundary; the hard CBF rows and swept certificate remain active at
        # every distance.  Keep the established primitive-scene tuning for
        # the synthetic regression scenarios.
        if self.collision.is_mesh:
            activation=float(np.clip((.10-minimum)/.05,0,1))
        else:
            activation=float(np.clip((.16-minimum)/.10,0,1))
        scale=1-.92*activation
        if self.route is not None and self.terminal_approach(self.execution_time):
            # Restore accurate camera holding once the base can settle. All
            # CBF rows and the continuous sweep gate remain unchanged.
            scale = 1.
        H=scale*H+np.eye(8)*(4.*activation)
        g=scale*g-4.*activation*self.filtered_reference
        if self.route is not None:
            speed, heading, _ = self.base_reference(state, self.execution_time)
            turn_scale = (.75 if self.terminal_approach(self.execution_time) else .5)*self.limits.velocity[1]
            hold = self.hold_weight(self.execution_time)
            turn_scale += hold*(min(.2, turn_scale)-turn_scale)
            turn = turn_scale*np.tanh(2.*self.heading_error(heading, state)/turn_scale)
            # Keep the base under the camera along reachable space. These are
            # objectives in the SAME hard-constrained QP, not separate commands.
            weights = self.motion_weights(self.execution_time)
            follow = np.array([1200., 500.])*(weights['orbit']+weights['moveout'])
            H[:2,:2] += np.diag(follow)
            g[:2] -= follow*np.array([speed, turn])
        return H,g,A,lo,hi

    def solve(self,state,target_p,target_r,feedforward,dt,**kwargs):
        self.execution_time = kwargs.get('trajectory_time', 0.)
        previous=self.previous.copy()
        command,info=super().solve(state,target_p,target_r,feedforward,dt,**kwargs)
        certificate=self.collision.sweep(state,command,dt,kwargs.get('base_z',0.),tolerance=.001,max_samples=17,
                                         base_rotation=kwargs.get('base_rotation'))
        info['command_sweep_lower_m']=float(np.min(certificate['lower']))
        if self.collision_mode != 'none' and info['command_sweep_lower_m'] < self.collision.margin:
            self.previous=previous
            raise RuntimeError('candidate fails continuous command clearance certificate')
        info.update(self.safety_info,svsdf_terms=self.collision_terms,kc_filtered_terms=self.coupled_terms,
                    avoidance_plan_failures=self.plan_failures,planner_braking=int(self.planner_braking),
                    collision_mode=self.collision_mode,
                    preview_failure=self.plan_info.get('preview_failure', ''),
                    mpc_trust_relaxed=int(self.plan_info.get('mpc_trust_relaxed', False)),
                    guide_projected_points=0 if self.route is None else self.route.projected_points,
                    guide_max_projection_m=0. if self.route is None else self.route.max_projection_m)
        return command,info
