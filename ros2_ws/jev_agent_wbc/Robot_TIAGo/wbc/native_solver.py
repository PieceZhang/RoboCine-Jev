"""Real Giskard explicit-QP adapter used by the isolated physical worker.

This composes tasks into upstream QPDataExplicit/QPSolverPIQP; it is not the
complete upstream World/MotionStatechart/Executor. No local optimizer fallback.
"""
import numpy as np
from scipy import sparse

from wbc_backend import add_pinned_core_paths
from collision_model import CollisionBarriers


class NativeGiskardSolver:
    def __init__(self,spec,collision_metadata):
        self.source,self.commit = add_pinned_core_paths()
        from giskardpy.qp.qp_data import QPDataExplicit
        from giskardpy.qp.solvers.qp_solver_piqp import QPSolverPIQP
        self.Data,self.solver = QPDataExplicit,QPSolverPIQP()
        # This 10-variable actuator projection must not turn a stationary
        # native proposal into micrometre/second creep merely because the
        # upstream solver defaults permit a 1e-5 duality gap. Use the genuine
        # solver's public numerical settings; all hard physical bounds and
        # failure checks remain unchanged, and big_ball_mode stays false.
        self.projection_settings=dict(eps_abs=1e-10,eps_rel=1e-10,
            eps_duality_gap_abs=1e-12,eps_duality_gap_rel=1e-12)
        for name,value in self.projection_settings.items():
            setattr(self.solver.solver.settings,name,value)
        self.spec = spec
        self.lower = np.array([j['lower'] for j in spec['joints']])
        self.upper = np.array([j['upper'] for j in spec['joints']])
        self.home = np.array([j['home'] for j in spec['joints']])
        self.speed = np.r_[.22,.35,[j['velocity'] for j in spec['joints']]]
        self.accel = np.r_[.45,.65,[j.get('acceleration',1.) for j in spec['joints']]]
        self.wheels = np.zeros((2,10))
        self.wheels[:,:2] = np.array([[1.,-spec['wheel_separation_m']/2],[1.,spec['wheel_separation_m']/2]])/spec['wheel_radius_m']
        self.collisions = CollisionBarriers(collision_metadata)

    def bounds(self,q,previous,dt=.01):
        lo = np.maximum(-self.speed,previous-self.accel*dt)
        hi = np.minimum(self.speed,previous+self.accel*dt)
        lo[2:] = np.maximum(lo[2:],(self.lower+.002-q)/dt)
        hi[2:] = np.minimum(hi[2:],(self.upper-.002-q)/dt)
        a = self.accel[2:]
        def stoppable(distance):
            return np.sqrt((a*dt)**2+2*a*np.maximum(distance,0.))-a*dt
        lo[2:] = np.maximum(lo[2:],-stoppable(q-self.lower-.002))
        hi[2:] = np.minimum(hi[2:],stoppable(self.upper-.002-q))
        if np.any(lo>hi+1e-10):
            raise RuntimeError('Hard position/acceleration/braking constraints infeasible')
        return lo,hi

    def solve(self,request):
        q,previous = np.asarray(request['q']),np.asarray(request['previous'])
        if q.shape!=(8,) or previous.shape!=(10,) or not np.isfinite([*q,*previous]).all():
            raise ValueError('Invalid measured actuator state')
        lo,hi = self.bounds(q,previous)
        collision_A,collision_b,collision_report = self.collisions.query(request['poses'],request['body_jacobians'])
        if request['mode']=='camera':
            J,error,ff = [np.asarray(request[k],float) for k in ('jacobian','error','feedforward')]
            if J.shape!=(6,10) or error.shape!=(6,) or ff.shape!=(6,) or not all(np.isfinite(a).all() for a in (J,error,ff)):
                raise ValueError('Invalid measured camera task')
            target = ff+np.r_[4.*error[:3],3.*error[3:]]
            distance = np.minimum(q-self.lower,self.upper-q)
            zone = np.minimum(.25,.2*(self.upper-self.lower))
            pressure = np.minimum(np.maximum(0.,1.-distance/zone)**2,1.)
            away = np.where(q-self.lower<=self.upper-q,1.,-1.)
            weights = np.r_[.06,.03,.002+.5*pressure]
            preferred = np.r_[0.,0.,-.12*(q-self.home)+.3*pressure*away]
            hessian = np.r_[weights,np.ones(6)]
            linear = np.r_[-weights*preferred,np.zeros(6)]
            equality = np.c_[J,-np.eye(6)]
            lower,upper = np.r_[lo,np.full(6,-np.inf)],np.r_[hi,np.full(6,np.inf)]
            slack = 6
        elif request['mode']=='actuator':
            target = np.asarray(request['desired_rates'],float)
            if target.shape!=(10,) or not np.isfinite(target).all():
                raise ValueError('Invalid actuator tracking request')
            hessian,linear = np.ones(10),-target
            equality = np.zeros((0,10));target=np.zeros(0)
            lower,upper,slack = lo,hi,0
        else:
            raise ValueError('Unsupported task mode')
        inequalities = np.vstack([self.wheels,collision_A])
        inequalities = np.c_[inequalities,np.zeros((len(inequalities),slack))]
        qp = self.Data(num_equality_slack_variables=slack,num_inequality_slack_variables=0,
            quadratic_weights=hessian,linear_weights=linear,box_lower_constraints=lower,box_upper_constraints=upper,
            equality_matrix=sparse.csc_matrix(equality),equality_bounds=target,
            inequality_matrix=sparse.csc_matrix(inequalities),
            inequality_lower_bounds=np.r_[[-8.,-8.],collision_b],
            inequality_upper_bounds=np.r_[[8.,8.],np.full(len(collision_b),np.inf)])
        solution = self.solver.solver_call(qp)
        u = solution[:10].copy()
        violation = max(0.,float(np.max(lo-u)),float(np.max(u-hi)),float(np.max(np.abs(self.wheels@u)-8.)))
        if len(collision_b):
            violation = max(violation,float(np.max(collision_b-collision_A@u)))
        if not np.isfinite(u).all() or violation>1e-6:
            raise RuntimeError(f'Native QP output violates a hard constraint: {violation}')
        return dict(command=u,hard_constraint_violation=violation,collision=collision_report,
            solver='giskardpy.qp.solvers.qp_solver_piqp.QPSolverPIQP',upstream_commit=self.commit,
            full_world_executor=False,big_ball_mode=self.solver.big_ball_mode)
