"""Whole-suffix polynomial curves with jointly chosen framing endpoints.

Only timestamped observed geometry is used. Numerical fits are proposals and
must pass the independent continuous robot lift before being offered to Jev.
"""
import time
import numpy as np
from scipy.optimize import minimize
from .intent_paths import IntentPath
from .cinematic_geometry import terminal_band, terminal_error, inside_terminal_band, curve_quality
from .camera_intent import obstacle_state, default_intrinsics, POLICY
from .reference import look_at
from .jvs_kinematics import CAMERA_OFFSET


def global_curves(original, active, scene, activation, model, intrinsics, count=6, seconds=3., diagnostics=None, tempo=2.):
    started=time.monotonic()
    intrinsics=default_intrinsics() if intrinsics is None else intrinsics
    end=active.duration; span=end-activation
    if span<.5:return []
    p0,_,v0=active.sample(activation);h=.001
    a0=(active.sample(activation+h)[2][:3]-active.sample(activation-h)[2][:3])/(2*h)
    pe=original.sample(original.duration)[0]
    times=np.linspace(activation,end,65);u=(times-activation)/span
    from scipy.special import comb
    chord=pe-p0;length=max(np.linalg.norm(chord),1e-8);axis=chord/length
    vstart=float(v0[:3]@axis);astart=float(a0@axis)
    c1=np.clip(vstart*span/(7*length),0.,.25)
    c2=np.clip(2*c1+astart*span**2/(42*length),c1,.5)
    controls=np.array([0.,c1,c2,max(c2,.25/tempo),max(c2,.6/tempo),1.,1.,1.])
    bernstein=np.column_stack([comb(7,k)*u**k*(1-u)**(7-k) for k in range(8)])
    s=bernstein@controls
    base=p0+s[:,None]*chord
    # Keep the measured incoming tangent/acceleration; the scalar clock already
    # supplies their chord components, so only transverse residuals remain.
    base+=(u-6*u**3+8*u**4-3*u**5)[:,None]*span*(v0[:3]-vstart*axis)
    base+=.5*(u**2-3*u**3+3*u**4-u**5)[:,None]*span**2*(a0-astart*axis)
    q=s/(.2+.8*s)
    bow=(256./27.)*q**3*(1-q)
    basis=np.column_stack([bow,bow*(2*s-1),10*u**3-15*u**4+6*u**5])
    centers=[];radii=[]
    for t in times:
        c,r=obstacle_state(scene,float(t));centers.append(c);radii.append(r)
    centers=np.asarray(centers);radii=np.asarray(radii)
    if centers.size:
        nearby=np.min(np.linalg.norm(base[:,None]-centers,axis=2)-radii,axis=0)<1.
        centers=centers[:,nearby];radii=radii[:,nearby]
    band=terminal_band(original,model,intrinsics)
    offsets=np.asarray(band['offsets_m'])
    extent=np.maximum(np.max(np.abs(offsets),axis=0),1e-5)
    chord=pe-p0;axis=chord/max(np.linalg.norm(chord),1e-8)
    side=np.cross(axis,[0.,0.,1.]);side/=max(np.linalg.norm(side),1e-8)
    vertical=np.cross(side,axis);vertical/=max(np.linalg.norm(vertical),1e-8)
    motion=original.phase(activation)['motion'].lower()
    def positions(x):return base+basis@x.reshape(3,3)
    def direction(points):
        if motion=='orbit':
            a=points[:,:2]-original.subject[:2]
            sign=np.sign(np.cross(p0[:2]-original.subject[:2],pe[:2]-original.subject[:2]))
            return sign*(a[:-1,0]*a[1:,1]-a[:-1,1]*a[1:,0])
        if motion in ('movein','moveout'):
            return np.diff(np.linalg.norm(points-original.subject,axis=1))*(-1 if motion=='movein' else 1)
        ax=original.sample(0)[1][:,1] if 'truck' in motion else np.array([0.,0.,1.])
        return np.diff(points@ax)*np.sign(chord@ax)
    def constraints(x):
        pts=positions(x)
        e=terminal_error(original,pts[-1],model,intrinsics)
        fields=(('centroid_error',POLICY.terminal_centroid_tolerance),
                ('log_area_error',POLICY.terminal_log_area_tolerance),
                ('bbox_log_error',POLICY.terminal_bbox_log_tolerance),
                ('view_angle_error_rad',POLICY.terminal_view_tolerance_rad))
        vals=[direction(pts)+1e-7,np.array([.8-e[k]/lim for k,lim in fields])]
        if e.get('silhouette_axis_required'):
            vals.append(np.array([.8-e['silhouette_axis_error_rad']/POLICY.terminal_silhouette_axis_tolerance_rad]))
        # Camera and calibrated wrist are checked during construction. The
        # complete URDF body and continuous sweeps remain independent gates.
        if centers.size:
            wrist=np.array([p-look_at(p,original.subject)@CAMERA_OFFSET for p in pts])
            vals.extend([(np.linalg.norm(pts[:,None]-centers,axis=2)-radii-.11).ravel(),
                         (np.linalg.norm(wrist[:,None]-centers,axis=2)-radii-.09).ravel()])
        return np.concatenate(vals)
    def objective(x):
        pts=positions(x);d2=np.diff(pts,n=2,axis=0)
        return float(8*np.sum(d2*d2)+.025*np.sum(x[:3]**2)+.12*np.sum(x[3:6]**2)+.03*np.sum(x[6:]**2))
    directions=[np.zeros(3),vertical,-vertical,side,-side,(side-vertical)/np.sqrt(2)]
    results=[]
    diagnostics=[] if diagnostics is None else diagnostics
    for d in directions:
        if diagnostics and time.monotonic()-started>=seconds:break
        x=np.r_[.3*d,np.zeros(3),np.zeros(3)]
        fit=minimize(objective,x,method='SLSQP',bounds=[(-.65,.65)]*6+list(zip(-extent,extent)),
            constraints=[{'type':'ineq','fun':constraints},{'type':'eq','fun':lambda x:x[:6].reshape(2,3)@axis}],options={'maxiter':35,'ftol':1e-7})
        gap=float(np.min(constraints(fit.x)))
        diagnostics.append(dict(seed=d.tolist(),success=bool(fit.success),min_constraint=gap,min_index=int(np.argmin(constraints(fit.x))),parameters=fit.x.tolist(),objective=float(fit.fun)))
        if gap < -1e-4:continue
        endpoint=pe+fit.x[6:]
        if not inside_terminal_band(terminal_error(original,endpoint,model,intrinsics),.8):
            # Ray-contract the optimized endpoint against actual image errors,
            # then recheck the changed path instead of treating a box as a tube.
            lo,hi=0.,1.
            for _ in range(12):
                mid=(lo+hi)/2
                if inside_terminal_band(terminal_error(original,pe+mid*fit.x[6:],model,intrinsics),.8):lo=mid
                else:hi=mid
            fit.x[6:]*=lo
            if np.min(constraints(fit.x)) < -1e-4:continue
        pts=positions(fit.x)
        if any(np.max(np.linalg.norm(pts-np.asarray(p.geometry['positions']),axis=1))<.025 for p in results):continue
        # Route is an informational geometric description, not a fixed policy.
        deviation=fit.x[:3]
        route='above' if deviation[2]>.06 else 'below' if deviation[2]<-.06 else 'left' if deviation@side>0 else 'right'
        program=dict(route=route,amplitude_m=float(np.linalg.norm(deviation)),extension_s=0.,regular_curve=True,
            curve_family='global_endpoint_polynomial',endpoint_offset_m=fit.x[6:].tolist(),
            shape_coefficients=fit.x[:6].tolist(),curve_tempo=tempo,endpoint_region='continuous_image_checked',search_role='coordinated_clearance')
        geometry=dict(times=times.tolist(),positions=pts.tolist(),shape_basis=basis.tolist(),start_velocity=v0[:3].tolist(),start_acceleration=a0.tolist())
        path=IntentPath(original,active,activation,program,model,intrinsics,geometry=geometry)
        q=curve_quality(path)
        diagnostics[-1]['quality']=q
        if q['direction_pass'] and q['regular_curve_pass']:results.append(path)
        if len(results)>=count:break
    if not results:
        # The joint optimizer can repair collision in this geometric seed.
        # This is never a candidate certificate and cannot bypass later gates.
        program=dict(route='timing',amplitude_m=0.,extension_s=0.,regular_curve=True,
            curve_family='global_endpoint_polynomial',curve_tempo=tempo,
            endpoint_offset_m=[0.,0.,0.],search_role='coordinated_clearance')
        geometry=dict(times=times.tolist(),positions=base.tolist(),shape_basis=basis.tolist(),
            start_velocity=v0[:3].tolist(),start_acceleration=a0.tolist())
        results.append(IntentPath(original,active,activation,program,model,intrinsics,geometry=geometry))
    return results


def bounded_paths(original,active,scene,activation,model,intrinsics,budget):
    from .curve_timing import timed_curves
    timed=timed_curves(original,active,scene,activation,model,intrinsics,count=2)
    remaining=max(0,budget.max_proposals-3-len(timed))
    global_paths=global_curves(original,active,scene,activation,model,intrinsics,
        count=remaining,seconds=min(2.,budget.candidate_wall_budget_s*.18)) if remaining else []
    if len(global_paths)==1 and remaining>=2:
        seed=global_paths[0]
        global_paths.append(IntentPath(original,active,activation,dict(seed.program,joint_branch='opposite'),
            model,intrinsics,geometry=dict(seed.geometry)))
    paths=timed+global_paths
    # A tiny complementary family retains early clearance when the simultaneous
    # curve fit has found a different local homotopy. No combinatorial grid.
    offsets=np.asarray(terminal_band(original,model,intrinsics)['offsets_m'])
    p=original.sample(original.duration)[0];radial=p-original.subject;radial[2]=0.
    radial/=max(np.linalg.norm(radial),1e-8);left=np.array([radial[1],-radial[0],0.])
    for route,d in [('left',left),('right',-left)]:
        program=dict(route=route,amplitude_m=.2,extension_s=0.,profile='arch',phase_rate=-.7,
            regular_curve=True,shape_skew=.2,endpoint_offset_m=offsets[np.argmax(offsets@d)].tolist(),
            endpoint_region='26_ray_bisection_rechecked_endpoints',search_role='coordinated_clearance')
        paths.append(IntentPath(original,active,activation,program,model,intrinsics))
    continuation=IntentPath(original,active,activation,dict(route='continuation',amplitude_m=0.,extension_s=0.),model,intrinsics)
    return ([continuation]+paths)[:budget.max_proposals]


def bind_geometric_progress(path):
    """Record the authored feature clock for an explicitly retimed geometry.

    This does not award visibility or completion; those still use actual poses.
    It prevents a deliberate wait from being scored as a spatial detour.
    """
    original=path.original;ts=np.linspace(0.,original.duration,401)
    reference=np.array([original.sample(t)[0] for t in ts])
    points=np.asarray(path.geometry['positions']);motion=original.phase(path.activation_s)['motion'].lower()
    if motion=='orbit':
        def feature(x):return np.unwrap(np.arctan2(x[:,1]-original.subject[1],x[:,0]-original.subject[0]))
    elif motion in ('movein','moveout'):
        def feature(x):return np.linalg.norm(x-original.subject,axis=1)
    else:
        axis=original.sample(0.)[1][:,1] if 'truck' in motion else np.array([0.,0.,1.])
        def feature(x):return x@axis
    ref=feature(reference);value=feature(points)
    if motion=='orbit':value+=round((ref[0]-value[0])/(2*np.pi))*2*np.pi
    sign=1. if ref[-1]>=ref[0] else -1.
    progress=np.interp(sign*value,sign*ref,ts)
    geometry=dict(path.geometry,progress_times=progress.tolist())
    return IntentPath(original,path.previous,path.activation_s,path.program,path.model,path.intrinsics,geometry=geometry)
