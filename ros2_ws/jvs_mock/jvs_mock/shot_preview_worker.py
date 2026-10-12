"""Nonblocking ROS preview worker; the fast constrained QP stays in the node.

A spawned process avoids Python/GIL or solver work delaying the 50 Hz feedback
callback. Immutable snapshots cross the process boundary; only completed plans
from the current shot-input generation are installed. Lockstep Isaac tests use
the same planner synchronously for deterministic controller comparisons.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import signal
import numpy as np
from .shot_predictive_qp import ShotPredictiveQP
from .obstacle_predictive_qp import ObstaclePredictiveQP


def initialize_worker():
    # The ROS node owns bounded braking and closes its pool afterwards. A
    # launch-group SIGINT must not interrupt a child while the parent brakes.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def compute_preview(snapshot, state, t, base_z, commanded_q):
    cls = ObstaclePredictiveQP if 'scene' in snapshot else ShotPredictiveQP
    extras = dict(scene=snapshot['scene'], collision_mode=snapshot['collision_mode']) if 'scene' in snapshot else {}
    controller = cls(snapshot['limits'], **extras, horizon_steps=snapshot['n'],
        horizon_dt=snapshot['horizon_dt'], replan_s=snapshot['replan_s'], blend_s=snapshot['blend_s'],
        tracking_weight=snapshot['tracking_weight'])
    controller.set_trajectory(snapshot['trajectory'])
    for name in ('trajectory','previous','home','plan','directions','context_override','direction_costs','path_rate'):
        setattr(controller,name,snapshot[name])
    if isinstance(controller, ObstaclePredictiveQP):
        controller.route=snapshot.get('route')
        controller.route_initial=snapshot.get('route_initial')
        controller.plan_failures=snapshot.get('plan_failures',0)
        controller.transition_entries=dict(snapshot.get('transition_entries', {}))
        controller.guide_costs=dict(snapshot.get('guide_costs', {}))
    controller.make_plan(state,t,base_z,commanded_q)
    result = {name:getattr(controller,name) for name in ('plan','reference','directions','direction_costs','plan_info')}
    if isinstance(controller, ObstaclePredictiveQP):
        result.update({name:getattr(controller,name) for name in ('collision_terms','coupled_terms','plan_failures','route','planner_braking','transition_entries','guide_costs')})
        request = snapshot.get('recovery_request')
        if request is not None and controller.route is not None:
            position, goal = request
            route = controller.route
            start = (position if route.safe_segment(position, position)
                     else route._coord(route._nearest_node(position)))
            try:
                result['recovery_path'] = route.search(start, goal)
                result['recovery_goal'] = goal
            except RuntimeError:
                # A changed live map can disconnect the old request. The
                # execution guide holds until a current route is available.
                result['recovery_path'] = None
                result['recovery_goal'] = None
    return result


class AsyncShotPredictiveQP(ShotPredictiveQP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                        initializer=initialize_worker)
        self.pending = None
        self.input_generation = 0
        self.pending_generation = 0
        self.pending_time = 0.
        self.applied_time = None
        self.execution_clock = 0.
        self.pending_execution_time = 0.
        self.applied_execution_time = None
        self.stale_plans = 0
        self.route_map_time = -np.inf

    def set_shot_input(self, context):
        super().set_shot_input(context)
        self.input_generation += 1

    def make_plan(self, state, t, base_z, commanded_q):
        if self.pending is not None:
            return
        snapshot = {name:getattr(self,name) for name in ('limits','n','horizon_dt','replan_s','blend_s',
            'trajectory','previous','home','plan','directions','context_override','direction_costs','path_rate',
            'tracking_weight')}
        # Serialize current immutable values; parent arrays are subsequently
        # replaced, never edited in place, by the execution QP.
        snapshot['directions'] = dict(self.directions)
        if isinstance(self, ObstaclePredictiveQP):
            if self.route_initial is None:
                self.route_initial = np.array(state)
            live_map = self.collision.scene.get('format') == 'jvs_lidar_surfels/v1'
            rebuild_route = live_map and self.execution_clock - self.route_map_time >= 2.
            snapshot.update(scene=self.collision.scene, collision_mode=self.collision_mode,
                            route=None if rebuild_route else self.route,
                            route_initial=self.route_initial, plan_failures=self.plan_failures,
                            transition_entries=dict(self.transition_entries),
                            guide_costs={} if rebuild_route else dict(self.guide_costs),
                            recovery_request=getattr(self, 'recovery_request', None))
            if rebuild_route:
                self.route_map_time = self.execution_clock
            if self.plan is not None and self.applied_time is not None:
                age=max(0., self.execution_clock-self.applied_execution_time)
                grid=np.arange(self.n)*self.horizon_dt
                snapshot['plan']=np.column_stack([np.interp(grid+age,grid,self.plan[:,j]) for j in range(8)])
                if np.max(np.abs(snapshot['plan'][0]-self.previous)/self.limits.velocity)>.25:
                    snapshot['plan']=None
        self.pending = self.pool.submit(compute_preview,snapshot,np.array(state),t,base_z,np.array(commanded_q))
        self.pending_generation = self.input_generation
        self.pending_time = t
        self.pending_execution_time = self.execution_clock
        self.last_plan = t

    def solve(self, *args, trajectory_time=0., **kwargs):
        # MPC controls are indexed by physical simulation time, independently
        # of the slowed shot clock. A 0.1 s-old path sample can otherwise hide
        # a several-second-old physical plan during a cinematic hold.
        self.execution_clock += float(kwargs.get('dt', args[4] if len(args) > 4 else 0.))
        if self.pending is not None and self.pending.done():
            finished = self.pending
            self.pending = None
            result = finished.result()
            # The static guide is tied to this trajectory/scene, not to the
            # age of a control snapshot. Keep it even when startup/planning
            # took too long to execute the returned MPC controls.
            if (self.pending_generation == self.input_generation
                    and isinstance(self, ObstaclePredictiveQP)
                    and result.get('route') is not None):
                self.route = result['route']
                self.recovery_path = result.get('recovery_path')
                self.recovery_goal = result.get('recovery_goal')
                # These trajectory/route costs are useful even when the
                # physical control snapshot is too old. Preserve execution's
                # branches, and never copy stale velocity controls here.
                self.guide_costs.update(result.get('guide_costs', {}))
            if (self.pending_generation == self.input_generation
                    and self.execution_clock-self.pending_execution_time <= .6):
                # A preview may have started before execution entered a hold.
                # Preserve parking/turn branches already chosen by execution;
                # installing its older dictionary must not reverse the base.
                if 'directions' in result:
                    execution_keys = ('base_guide', 'stopped_turn', 'transition', 'transition_parking',
                                      'transition_arrival', 'transition_heading', 'transition_follow_turn',
                                      'transition_entry_turn', 'transition_exit_turn')
                    # A preview's hypothetical future state must not commit an
                    # execution branch before the real junction is reached.
                    result['directions'] = {key: value for key, value in result['directions'].items()
                                            if key[0] not in execution_keys}
                    result['directions'].update({key: value for key, value in self.directions.items()
                                                if key[0] in execution_keys})
                if 'transition_entries' in result:
                    result['transition_entries'].update(self.transition_entries)
                for name,value in result.items():
                    setattr(self,name,value)
                self.applied_time = self.pending_time
                self.applied_execution_time = self.pending_execution_time
            else:
                self.stale_plans += 1
                self.last_plan = -np.inf
        if isinstance(self, ObstaclePredictiveQP) and self.plan is not None and self.applied_time is not None:
            # Consume the part of the plan corresponding to NOW, not the
            # first input of a snapshot taken several hundred milliseconds ago.
            age=max(0., self.execution_clock-self.applied_execution_time)
            if age >= self.n*self.horizon_dt:
                # np.interp otherwise repeats the final velocity indefinitely
                # when every subsequent preview misses its acceptance deadline.
                self.plan = None
                self.reference = np.zeros(8)
                self.applied_time = None
                self.applied_execution_time = None
                self.last_plan = -np.inf
            else:
                grid=np.arange(self.n)*self.horizon_dt
                self.reference=np.array([np.interp(age,grid,self.plan[:,j]) for j in range(8)])
        command,info = super().solve(*args,trajectory_time=trajectory_time,**kwargs)
        info.update(preview_age_s=-1. if self.applied_execution_time is None else self.execution_clock-self.applied_execution_time,
                    preview_inflight=int(self.pending is not None), preview_stale_count=self.stale_plans)
        return command,info

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)


class AsyncObstaclePredictiveQP(AsyncShotPredictiveQP, ObstaclePredictiveQP):
    """The same collision MPC in a spawned worker, with local safety constraints."""

    def recovery_waypoint(self, position, goal):
        # A* in a dense live surface map can take >100 ms. Submit the soft
        # pursuit request with the next preview; execution never runs A*.
        self.recovery_request = (np.array(position), np.array(goal))
        path = getattr(self, 'recovery_path', None)
        previous_goal = getattr(self, 'recovery_goal', None)
        if path is not None and previous_goal is not None and np.linalg.norm(goal-previous_goal) <= .15:
            start = (position if self.route.safe_segment(position, position)
                     else self.route._coord(self.route._nearest_node(position)))
            for waypoint in reversed(path[1:]):
                if self.route.safe_segment(start, waypoint):
                    return waypoint
        # Holding the soft position guide is safe while awaiting the worker.
        # The execution QP still checks the latest whole-body map and sweep.
        return np.array(position)
