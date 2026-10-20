"""One phase-preserving clock optimization from current observed occupancy.

The body lift and time/progress grid are unverified numerical seeds. No authored
obstacle keyframes or offline CineGym witnesses enter this module.
"""
from copy import copy
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize, differential_evolution
from numpy.polynomial import Polynomial
from .intent_paths import IntentPath,positions_at,progress_at


def clock_basis(times,phases,activation):
    columns=[];derivatives=[]
    base=Polynomial([0.,0.,0.,64.,-192.,192.,-64.])
    for phase in phases:
        start=max(activation,phase['start_s']);end=phase['end_s'];span=end-start
        if span<=1e-6:continue
        u=np.clip((times-start)/span,0.,1.);mask=(times>=start)&(times<=end)
        for polynomial in (base,base*Polynomial([-1.,2.])):
            columns.append(np.where(mask,span*polynomial(u),0.))
            derivatives.append(np.where(mask,polynomial.deriv()(u),0.))
    return np.array(columns).T,np.array(derivatives).T


def construct(original,active,scene,activation,model,intrinsics,search,contract):
    if search is None:return []
    def failed(reason,**values):
        search.body_timing_diagnostics=dict(reason=reason,**values)
        return []
    if active.duration-activation<1.5:return failed('insufficient_remaining_time')
    from .task_space_planning import MeasuredCameraPrediction
    virtual=MeasuredCameraPrediction(active,search.state,search.base_z,search.t,
                                     max(.2,activation-search.t))
    nominal=copy(search);nominal.collision=None
    nominal.policy=replace(search.policy,step_s=.1,
        max_samples=max(search.policy.max_samples,int(np.ceil((active.duration-search.t)/.1))+2))
    lift=nominal._lift(virtual,active.duration,contract['allocation'])
    if not lift['feasible']:return failed('nominal_body_lift:'+str(lift.get('reason')),first_failure_s=lift.get('first_failure_s'))
    trace=lift['trace'];guide_times=np.array([r['t'] for r in trace])
    guide_states=np.array([r['state'] for r in trace]);guide_controls=np.array([r['control'] for r in trace])
    checks=np.unique(np.r_[np.linspace(activation,active.duration,
        max(25,int(np.ceil((active.duration-activation)/.12))+1)),
        [p['end_s'] for p in original.task['shot_phase'] if p['end_s']>activation]])
    basis,derivative=clock_basis(checks,original.task['shot_phase'],activation)
    if not basis.shape[1]:return []
    collision=search.collision
    if collision is None:return []
    # Keep all observed movers and nearby static witnesses for the seed.
    # Independent fine verification subsequently checks the complete map.
    selected=set(np.flatnonzero(collision.track_ids>0).tolist())
    for state in guide_states[::max(1,len(guide_states)//8)]:
        centers,_=collision.active_obstacles(state,search.base_z)
        selected.update(np.atleast_1d(collision.tree.query(centers,k=1)[1]).tolist())
    ids=np.array(sorted(selected),int)
    geometry=collision.geometry;p,r=geometry.frame_arrays(guide_states,search.base_z)
    gaps=[]
    for at in checks:
        forecast=collision.at_time(float(at-search.t))
        ds=geometry._primitive_arrays(p,r,forecast.centers[ids][None])
        gaps.append((ds-forecast.radii[ids][None,:,None]).min(axis=(1,2)))
    gaps=np.asarray(gaps)
    def clear(coefficients):
        progress=checks+basis@coefficients
        return np.array([np.interp(at,guide_times,row) for at,row in zip(progress,gaps)])
    # C2 clock perturbations leave pose, velocity, acceleration and every
    # phase boundary unchanged. The numerical domain requires monotonicity.
    dense=np.linspace(activation,active.duration,401)
    _,dense_derivative=clock_basis(dense,original.task['shot_phase'],activation)
    initial=np.zeros(basis.shape[1])
    if min(clear(initial))<collision.margin+.025:
        # Crossing-time feasibility is nonconvex. Find one numerical seed in
        # this selected clock family before the constrained refinement; no
        # alternative family, approval or physical gate is bypassed.
        def violation(x):
            gaps=np.maximum(0.,collision.margin+.025-clear(x))
            monotonic=np.maximum(0.,.1-(1.+dense_derivative@x))
            return 1e4*float(gaps@gaps)+1e3*float(monotonic@monotonic)+.5*float(x@x)
        initial=differential_evolution(violation,[(-.22,.22)]*basis.shape[1],
            maxiter=32,popsize=8,seed=0,polish=False,tol=1e-5).x
    result=minimize(lambda x:.5*float(x@x),initial,
        jac=lambda x:x,bounds=[(-.22,.22)]*basis.shape[1],
        constraints=[dict(type='ineq',fun=lambda x:clear(x)-collision.margin-.025),
                     dict(type='ineq',fun=lambda x:1.+dense_derivative@x-.1)],
        method='SLSQP',options=dict(maxiter=100,ftol=1e-8))
    if not result.success or min(clear(result.x))<collision.margin+.0249:return failed('clock_optimization_unverified',solver=result.message,min_seed_clearance_m=float(min(clear(result.x))))
    times=np.unique(np.r_[np.linspace(activation,active.duration,
        max(101,int(np.ceil((active.duration-activation)/.05))+1)),
        [p['start_s'] for p in original.task['shot_phase'] if p['start_s']>activation]])
    b,d=clock_basis(times,original.task['shot_phase'],activation)
    progress=times+b@result.x;rates=1.+d@result.x
    if np.any(np.diff(progress)<=0.):return []
    states=np.column_stack([np.interp(progress,guide_times,guide_states[:,j]) for j in range(9)])
    controls=np.column_stack([np.interp(progress,guide_times,guide_controls[:,j]) for j in range(8)])*rates[:,None]
    seed=[row for row in trace if row['t']<activation]
    seed += [dict(t=float(t),state=x.tolist(),control=u.tolist()) for t,x,u in zip(times,states,controls)]
    eps=.001
    program=dict(route='timing',amplitude_m=0.,extension_s=0.,regular_curve=True,
        curve_family='whole_body_phase_clock',fixed_spatial_shape=True,
        endpoint_offset_m=[0.,0.,0.],construction_contract=dict(contract),
        clock_coefficients=result.x.tolist(),clock='C2_phase_preserving_whole_body',
        seed_grid_min_clearance_m=float(min(clear(result.x))),
        search_role='observed_whole_body_timing')
    path=IntentPath(original,active,activation,program,model,intrinsics,
        geometry=dict(times=times.tolist(),positions=positions_at(active,progress).tolist(),
            progress_times=[progress_at(active,t) for t in progress],
            start_velocity=active.sample(activation)[2][:3].tolist(),
            start_acceleration=((active.sample(activation+eps)[2][:3]-active.sample(activation-eps)[2][:3])/(2*eps)).tolist(),
            joint_seed=seed,joint_seed_preference_only=False))
    search.body_timing_diagnostics=dict(reason='seed_constructed_requires_full_validation')
    return [path]
