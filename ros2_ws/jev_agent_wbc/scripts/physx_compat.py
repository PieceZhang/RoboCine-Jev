"""PhysX-only numerical and sensor adapters. Shared CPU/Agent code is untouched.

The coarse optimizer already delegates camera FK to KinematicSearch, but rejects
any supplied base orientation before doing that. Its proxy keeps the measured
orientation in FK and external body distances while satisfying the legacy guard.
All proposals still pass the unmodified fine verification and approval pipeline.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import pickle


def _report_future_failure(future):
    if not future.cancelled():
        error=future.exception()
        if error:
            import traceback
            traceback.print_exception(type(error),error,error.__traceback__)


class PhysxProcessPool(ProcessPoolExecutor):
    def submit(self, function, *args, **kwargs):
        # The base executor serializes later on its queue thread. Freeze the
        # measured snapshot now, before the live sensor/Agent can mutate it.
        args,kwargs=freeze_task_inputs(args,kwargs)
        future=super().submit(function,*args,**kwargs)
        future.add_done_callback(_report_future_failure)
        return future


def freeze_task_inputs(args, kwargs):
    return pickle.loads(pickle.dumps((args,kwargs),protocol=pickle.HIGHEST_PROTOCOL))


def normalized_solver_result(result, api):
    """Expose the CPU OSQP 1.x result names/codes for Isaac's bundled 0.6.x."""
    codes = ('SOLVED', 'SOLVED_INACCURATE', 'PRIMAL_INFEASIBLE',
             'PRIMAL_INFEASIBLE_INACCURATE', 'DUAL_INFEASIBLE',
             'DUAL_INFEASIBLE_INACCURATE', 'MAX_ITER_REACHED',
             'TIME_LIMIT_REACHED', 'NON_CVX', 'SIGINT', 'UNSOLVED')
    values = {}
    for i,name in enumerate(codes):
        try:
            values[int(api.constant('OSQP_'+name))] = i+1
        except ValueError:
            pass  # OSQP 0.6 does not expose the interrupt constant.
    info = SimpleNamespace(status=result.info.status,
        status_val=values.get(int(result.info.status_val), int(result.info.status_val)),
        prim_res=float(result.info.prim_res if hasattr(result.info,'prim_res') else result.info.pri_res))
    x = None if result.x is None else np.asarray(result.x, dtype=float)
    if x is not None and not np.isfinite(x).all():
        x = None  # bundled 0.6 returns object arrays of None on infeasibility
    return SimpleNamespace(x=x, y=result.y, info=info)


class MotorTargets:
    """Continuous physical drive targets for the CPU's endpoint commands.

    This is a hardware interface, not a change to the Agent/WBC Limits. Slew
    uses those same physical joint and wheel acceleration caps at physics rate.
    Measured states remain unmodified and are independently audited.
    """
    def __init__(self, limits):
        self.limits = limits
        self.velocity = np.zeros(8)
        self.control = np.zeros(8)
        self.acceleration = np.r_[np.full(2,limits.wheel_acceleration),limits.acceleration[2:]]

    def step(self, command, dt, measured=None, positions=None, measured_wheels=None):
        anchor=(self.control if measured is None else np.asarray(measured,float)).copy()
        if measured_wheels is not None:
            # Motor continuity is relative to motor encoders. Contact-induced
            # chassis slip is a physical measurement, not a wheel-speed jump.
            wl,wr=np.asarray(measured_wheels,float)
            anchor[:2]=[(wl+wr)*self.limits.wheel_radius/2,
                         (wr-wl)*self.limits.wheel_radius/self.limits.wheel_separation]
        desired=np.asarray(command,float).copy()
        if positions is not None:
            from jev_agent_wbc.jvs_kinematics import JOINT_LOWER, JOINT_UPPER
            # Keep float32 PhysX roundoff inside the unchanged CPU position
            # band. This guards the motor target, never the measured state.
            padding=2e-5
            low=np.maximum(np.asarray(positions)-JOINT_LOWER-self.limits.position_margin-padding,0.)
            high=np.maximum(JOINT_UPPER-self.limits.position_margin-padding-np.asarray(positions),0.)
            a=self.limits.acceleration[2:]
            speed_low=np.sqrt((a*dt)**2+2*a*low)-a*dt
            speed_high=np.sqrt((a*dt)**2+2*a*high)-a*dt
            desired[2:]=np.clip(desired[2:],-speed_low,speed_high)
        delta = np.clip(desired-anchor,
                        -self.limits.acceleration*dt,self.limits.acceleration*dt)
        wheel_delta=self.limits.wheels@delta
        ratio=max(1.,float(np.max(abs(wheel_delta)))/(self.limits.wheel_acceleration*dt))
        delta[:2]/=ratio
        self.control=anchor+delta
        self.velocity=np.r_[self.limits.wheels@self.control,self.control[2:]]
        return self.velocity.copy()


class SolverAPI:
    def __init__(self, api):
        self._api = api

    def OSQP(self):
        solver, api = self._api.OSQP(), self._api
        class CompatibleSolver:
            def __getattr__(self, name):
                return getattr(solver, name)
            def solve(self, *args, **kwargs):
                return normalized_solver_result(solver.solve(*args, **kwargs), api)
        return CompatibleSolver()


class OrientedCollision:
    def __init__(self, collision, tilt):
        self._collision, self._tilt = collision, tilt

    def __getattr__(self, name):
        return getattr(self._collision, name)

    def at_time(self, offset):
        return OrientedCollision(self._collision.at_time(offset), self._tilt)

    def clearance(self, state, base_z=0., base_rotation=None, gradient=False):
        rotation = (Rotation.from_euler('z', state[2]).as_matrix()@self._tilt
                    if base_rotation is None else base_rotation)
        return self._collision.clearance(state, base_z, rotation, gradient=gradient)


class OrientedSearch:
    tilt = None  # only the legacy coarse-solver guard reads this attribute

    def __init__(self, search):
        self._search = search
        self.collision = (None if search.collision is None else
                          OrientedCollision(search.collision, search.tilt))

    def __getattr__(self, name):
        # In particular, fk stays bound to the ORIGINAL measured-tilt search.
        return getattr(self._search, name)


_original_refine = None
_original_tasks = {}


def oriented_refine(path, search, **kwargs):
    try:
        return _original_refine(path, OrientedSearch(search) if search.tilt is not None else search, **kwargs)
    except Exception:
        import traceback
        traceback.print_exc()
        raise


def _run_task(name, args, kwargs):
    try:
        return _original_tasks[name](*args, **kwargs)
    except Exception:
        import traceback
        traceback.print_exc()
        raise


def physx_build(*args, **kwargs):
    return _run_task('build', args, kwargs)


def physx_diagnose(*args, **kwargs):
    return _run_task('diagnose', args, kwargs)


def physx_verify(*args, **kwargs):
    return _run_task('verify', args, kwargs)


def initialize_physx_worker():
    from jev_agent_wbc.shot_preview_worker import initialize_worker
    initialize_worker()
    from jev_agent_wbc import joint_curve_optimizer
    global _original_refine
    if _original_refine is None:
        _original_refine = joint_curve_optimizer.refine
        joint_curve_optimizer.refine = oriented_refine
        joint_curve_optimizer.osqp = SolverAPI(joint_curve_optimizer.osqp)
    from jev_agent_wbc import task_space_planning
    if not _original_tasks:
        for name, function in (('build', physx_build), ('diagnose', physx_diagnose), ('verify', physx_verify)):
            _original_tasks[name] = getattr(task_space_planning, name)
            setattr(task_space_planning, name, function)


def install_worker_adapter():
    # Process-local binding consumed by the existing ProcessPool constructor.
    # CPU executables never import this module or change their initializer.
    import jev_agent_wbc.engine as engine
    engine.initialize_worker = initialize_physx_worker
    engine.ProcessPoolExecutor = PhysxProcessPool


def subject_from_returns(points, calibration, stamp):
    from jev_agent_wbc.subject_state import fit_subject_points
    estimate = fit_subject_points(points, calibration, stamp)
    estimate['stamp_s'] = float(stamp)
    if estimate['valid']:
        estimate.update(rotation=np.eye(3).tolist(), orientation_valid=True,
            heading_source='stationary_subject_calibration', stationary=True)
    else:
        estimate['orientation_valid'] = False
    return estimate



def generalized_gravity(raw, dof_count, indices):
    raw=np.asarray(raw,float)
    root_dofs=len(raw)-dof_count
    if root_dofs not in (0,6):
        raise RuntimeError('unexpected PhysX generalized gravity layout')
    return raw[root_dofs+np.asarray(indices,int)]


def configure_rolling_contacts(stage, robot_path):
    """Use a centre-line analytic tread for CPU's ideal rolling assumption.

    Radius, joint centres, visual width, inertia and torque caps are unchanged.
    The thin collision tread is contained in the full cylinder still used by
    the shared conservative body collision model. Retain analytic cylinders:
    faceted wheel meshes can lose normal load to the coplanar passive casters.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    scope=stage.GetPrimAtPath(robot_path).GetParent().GetPath()
    records=[]
    for link in ('left_wheel_link','right_wheel_link'):
        container=stage.GetPrimAtPath(scope.AppendChild(link).AppendChild('collisions'))
        ancestors=[];cursor=container
        while cursor and cursor.GetPath()!=scope:
            ancestors.append(cursor.GetPath());cursor=cursor.GetParent()
        for path in reversed(ancestors):
            prim=stage.GetPrimAtPath(path)
            if prim.IsInstance():prim.SetInstanceable(False)
        container=stage.GetPrimAtPath(scope.AppendChild(link).AppendChild('collisions'))
        cylinders=[p.GetPath() for p in Usd.PrimRange(container)
                   if p.IsA(UsdGeom.Cylinder) and p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(cylinders)!=1:raise RuntimeError('expected one wheel collision cylinder')
        cylinder=UsdGeom.Cylinder(stage.GetPrimAtPath(cylinders[0]))
        original_height=float(cylinder.GetHeightAttr().Get())
        cylinder.CreateHeightAttr(.0001)
        records.append(dict(link=link,collision=str(cylinders[0]),original_height_m=original_height,
            contact_height_m=.0001,radius_m=float(cylinder.GetRadiusAttr().Get()),
            inside_unchanged_cylindrical_safety_envelope=True))
    return records
