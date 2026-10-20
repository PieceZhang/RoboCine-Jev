"""Current-measurement recovery proposals for the independent drive guard.

A proposal has no permission to execute: the caller must retain actuator, ramp,
external and self swept checks. It never counts as a successful camera replan.
"""
import numpy as np
from .shot_predictive_qp import integration_matrix
from .tpr_wbc import qp
from .whole_body_qp import WholeBodyQP,whole_body_kinematics


def recovery_proposal(collision,state,measured,limits,dt,base_z=0.,base_rotation=None):
    bounds=WholeBodyQP(limits);bounds.previous=np.asarray(measured).copy()
    A,lo,hi=bounds.bounds(np.asarray(state)[3:],dt,commanded_q=np.asarray(state)[3:])
    centers,radii=collision.active_obstacles(state,base_z,base_rotation,per_body=None)
    distance,gradient=collision.geometry.primitive_distances(state,centers,base_z,base_rotation)
    distance-=radii[:,None]
    rows=gradient@integration_matrix(state[2])
    motion=collision.primitive_motion(state,centers,base_z,base_rotation)
    selected=(distance<.40)&((rows@measured+motion<.05)|(distance<.10))
    if not np.any(selected):raise RuntimeError('no observed threat for guard recovery')
    reserve=rows[selected];target=-1.5*(distance[selected]-.10)-motion[selected]
    _,_,jacobian=whole_body_kinematics(state,base_z,base_rotation)
    weight=1e4/len(reserve)
    H=np.eye(8)+.1*jacobian.T@jacobian+weight*reserve.T@reserve
    g=-.2*np.asarray(measured)-weight*reserve.T@target
    command,residual=qp(H,g,A,lo,hi,measured)
    return command,dict(source='fresh_observation_physical_guard',qp_violation=float(residual),
        scope='proposal only; current interval full-body, ramp and self certificates required',
        camera_task_preserved=False,observed_threat_rows=int(len(reserve)))
