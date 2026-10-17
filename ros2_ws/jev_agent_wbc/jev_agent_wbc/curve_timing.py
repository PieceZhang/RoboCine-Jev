"""Small time/progress lattice followed by a smooth constrained clock fit.

Geometry stays one line or one arch. Waiting for a moving obstacle changes the
clock rather than accumulating spatial detours. All outputs are unverified seeds.
"""
import numpy as np
import osqp
from scipy import sparse
from .reference import look_at
from .intent_paths import IntentPath
from .camera_intent import obstacle_state
from .jvs_kinematics import CAMERA_OFFSET
from .cinematic_geometry import curve_quality


def timed_curves(original,active,scene,activation,model,intrinsics,count=3,diagnostics=None,displacement=None,skew=.2,initial_boundary=None):
    end=active.duration;span=end-activation
    if span<1.5:return []
    p0,_,v0=active.sample(activation);pe=original.sample(original.duration)[0]
    if initial_boundary is not None:
        p0=np.asarray(initial_boundary['position']);v0=np.asarray(initial_boundary['twist'])
    chord=pe-p0;length=np.linalg.norm(chord);axis=chord/max(length,1e-8)
    side=np.cross(axis,[0.,0.,1.]);side/=max(np.linalg.norm(side),1e-8)
    n=max(12,int(np.ceil(span/.16))+1);ts=np.linspace(activation,end,n);dt=ts[1]-ts[0]
    grid=np.linspace(0,1,101);step=.01;vmax=min(12,max(2,int(.6*dt/(length*step))))
    ds=np.arange(vmax+1);speed0=int(np.clip(round(v0[:3]@axis*dt/(length*step)),0,vmax))
    nominal=np.clip((np.array([original.sample(t)[0] for t in ts])-p0)@axis/max(length,1e-8),0.,1.)
    candidates=[]
    diagnostics=[] if diagnostics is None else diagnostics
    motion=original.phase(activation)['motion'].lower()
    vertical=np.array([0.,0.,1.])
    directions=[(np.zeros(3),1.),(.2*side,.2),(-.2*side,.2),
        ((-.3*vertical if 'pedestal' not in motion else .55*side),1.),(.4*side,1.),(-.4*side,1.)]
    if displacement is not None:
        directions=[(np.asarray(displacement,float),float(skew))]
    def arch(q,skew):
        w=q/(skew+(1-skew)*q)
        return (256./27.)*w**3*(1-w)
    for displacement,skew in directions:
        bow=arch(grid,skew)
        points=p0+grid[:,None]*chord+bow[:,None]*displacement
        wrists=np.array([p-look_at(p,original.subject)@CAMERA_OFFSET for p in points])
        safe=np.ones((n,101),bool)
        for k,t in enumerate(ts):
            centers,radii=obstacle_state(scene,float(t))
            if len(centers):
                gap=np.min(np.linalg.norm(points[:,None]-centers,axis=2)-radii,axis=1)-.115
                wrist_gap=np.min(np.linalg.norm(wrists[:,None]-centers,axis=2)-radii,axis=1)-.10
                safe[k]=(gap>=0)&(wrist_gap>=0)
        costs=np.full((101,vmax+1),np.inf);costs[0,speed0]=0.
        parents=np.full((n,101,vmax+1),-1,dtype=np.int16)
        for k in range(1,n):
            new=np.full_like(costs,np.inf)
            for d in ds:
                previous_q=np.arange(101)-d;valid=(previous_q>=0)&safe[k]
                qs=np.flatnonzero(valid)
                choices=np.arange(max(0,d-1),min(vmax,d+1)+1)
                vals=costs[previous_q[qs,None],choices[None]]+.002*(choices-d)**2
                best=np.argmin(vals,axis=1)
                new[qs,d]=vals[np.arange(len(qs)),best]+.02*(grid[qs]-nominal[k])**2
                parents[k,qs,d]=choices[best]
            costs=new
        diagnostics.append(dict(displacement=displacement.tolist(),lattice_feasible=bool(np.isfinite(costs[-1,0]))))
        if not np.isfinite(costs[-1,0]):continue
        qidx=np.zeros(n,dtype=int);qidx[-1]=100;d=0
        for k in range(n-1,0,-1):
            old=parents[k,qidx[k],d];qidx[k-1]=qidx[k]-d;d=old
        q=grid[qidx];lo=np.empty(n);hi=np.empty(n)
        for k,j in enumerate(qidx):
            a=b=int(j)
            while a>0 and safe[k,a-1]:a-=1
            while b<100 and safe[k,b+1]:b+=1
            lo[k]=grid[a];hi[k]=grid[b]
        lo[0]=hi[0]=0.;lo[-1]=hi[-1]=1.
        D=np.diff(np.eye(n),axis=0)/dt;D2=np.diff(np.eye(n),n=2,axis=0)/dt**2
        D3=np.diff(np.eye(n),n=3,axis=0)/dt**3
        H=np.eye(n)*.02+2*D2.T@D2+.1*D3.T@D3
        A=np.vstack([np.eye(n),D,D2,D[:1],D[-1:]])
        lower=np.r_[lo,np.zeros(n-1),np.full(n-2,-.5/length),v0[:3]@axis/length,0.]
        upper=np.r_[hi,np.full(n-1,.65/length),np.full(n-2,.5/length),v0[:3]@axis/length,0.]
        qp=osqp.OSQP();qp.setup(P=sparse.csc_matrix(H),q=-.02*q,A=sparse.csc_matrix(A),l=lower,u=upper,verbose=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=6000,polish=False)
        result=qp.solve()
        diagnostics[-1]['clock_qp']=result.info.status
        if result.x is None or result.info.status_val not in (1,2):continue
        q=result.x;bow=arch(q,skew)
        pts=p0+q[:,None]*chord+bow[:,None]*displacement
        h=.001;a0=(active.sample(activation+h)[2][:3]-active.sample(activation-h)[2][:3])/(2*h)
        if initial_boundary is not None:a0=np.zeros(3)
        basis=np.column_stack([64*q**3*(1-q)**3,64*q**3*(1-q)**3*(2*q-1),10*q**3-15*q**4+6*q**5])
        geometry=dict(times=ts.tolist(),positions=pts.tolist(),start_velocity=v0[:3].tolist(),start_acceleration=a0.tolist(),shape_basis=basis.tolist())
        program=dict(route='timing' if not np.any(displacement) else 'right' if displacement@side>0 else 'left',amplitude_m=float(np.linalg.norm(displacement)),
            extension_s=0.,regular_curve=True,curve_family='progress_lattice_curve',endpoint_offset_m=[0.,0.,0.],
            clock='monotone_collision_tunnel',shape_skew=skew,search_role='coordinated_clearance')
        if initial_boundary is not None:
            program['measured_initial_boundary']=True
            geometry['initial_rotation_xyzw']=initial_boundary['quaternion_xyzw']
        path=IntentPath(original,active,activation,program,model,intrinsics,geometry=geometry)
        quality=curve_quality(path)
        diagnostics[-1]['quality']=quality
        if quality['direction_pass'] and quality['regular_curve_pass']:candidates.append(path)
        if len(candidates)>=count:break
    return candidates
