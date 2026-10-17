"""Bounded sparse SQP of a simple camera curve and whole-body configurations.

Coarse knots are a search seed only. Every output requires the existing fine
continuous FK, swept collision, actuator and exact-installation checks.
"""
import time
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from .intent_paths import IntentPath,positions_at,poses_at
from .reference import look_at
from .jvs_kinematics import JOINT_LOWER,JOINT_UPPER
from .cinematic_geometry import curve_quality,terminal_error,inside_terminal_band,terminal_band
from .camera_intent import POLICY


def refine(path,search,seconds=5.,max_iterations=12,knots=33,step_tolerance=1e-4):
    start=time.monotonic()
    if search.tilt is not None:return None,dict(reason='tilted_base_not_supported')
    end=path.duration;activation=path.activation_s
    if end-search.t<1.:return None,dict(reason='insufficient_horizon')
    n=knots;ts=np.linspace(search.t,end,n);dt=ts[1]-ts[0]
    original=path.original;span=end-activation
    u=np.clip((ts-activation)/span,0,1);s=10*u**3-15*u**4+6*u**5
    b=64*u**3*(1-u)**3;basis=np.column_stack([b,b*(2*u-1),s])
    if 'shape_basis' in path.geometry:
        basis=np.column_stack([np.interp(ts,path.geometry['times'],np.asarray(path.geometry['shape_basis'])[:,j],left=0.) for j in range(3)])
    target=positions_at(path,ts);initial_offset=search.p-target[0]
    join=(1-(10*np.clip((ts-search.t)/(activation-search.t),0,1)**3-15*np.clip((ts-search.t)/(activation-search.t),0,1)**4+6*np.clip((ts-search.t)/(activation-search.t),0,1)**5)
          if activation>search.t else np.zeros_like(ts))
    target+=join[:,None]*initial_offset
    states=np.tile(search.state,(n,1));xcurve=np.zeros(9)
    if path.program.get('joint_branch')=='opposite':
        chord=target[-1]-target[0];side=np.cross(chord,[0.,0.,1.]);side/=max(np.linalg.norm(side),1e-8)
        xcurve[:3]=.3*side

    # Start from the measured posture at every knot: this seed satisfies the
    # nonholonomic equalities and keeps the first convex subproblem feasible.
    planar=target[-1,:2]-target[0,:2]
    if np.linalg.norm(planar)>.1:
        goal=np.arctan2(planar[1],planar[0]);delta=(goal-search.state[2]+np.pi)%(2*np.pi)-np.pi
        if abs(delta)>np.pi/2:delta-=np.sign(delta)*np.pi
        w=np.clip((ts-search.t)/max(1.5,activation-search.t),0.,1.);w=10*w**3-15*w**4+6*w**5
        yaw=search.state[2]+w*delta
        if path.program.get('construction_contract'):
            # Seed the platform from the actual curve tangent, which can
            # differ greatly from the endpoint chord on an avoidance arch.
            # Choose one forward/reverse branch from the measured heading;
            # this is a continuous allocation seed, not an alternate path.
            planar_steps=np.diff(target[:,:2],axis=0)
            heading=np.unwrap(np.arctan2(planar_steps[:,1],planar_steps[:,0]))
            first=int(np.argmax(np.linalg.norm(planar_steps,axis=1)>1e-5))
            offset_angle=(heading[first]-search.state[2]+np.pi)%(2*np.pi)-np.pi
            branch=heading[first]-offset_angle-search.state[2]
            if abs(offset_angle)>np.pi/2:branch+=np.sign(offset_angle)*np.pi
            desired=np.r_[heading,heading[-1]]-branch
            desired=search.state[2]+w*(desired-search.state[2])
            yaw[0]=search.state[2]
            for k in range(1,n):
                yaw[k]=yaw[k-1]+np.clip(desired[k]-yaw[k-1],-.7*dt,.7*dt)
        offset=search.p[:2]-search.state[:2]
        for k in range(1,n):
            a=yaw[k]-search.state[2];rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            states[k,:2]=target[k,:2]-rot@offset;states[k,2]=yaw[k]
            states[k,3]=np.clip(search.state[3]-a,JOINT_LOWER[0]+.025,JOINT_UPPER[0]-.025)
    size=9*n+9
    collisions=[]
    camera_observed=None
    directed=bool(path.program.get('construction_contract'))
    camera_probes=None
    if search.collision is not None:
        for t in ts:collisions.append(search.collision.at_time(float(t-search.t)))
        from .task_space_planning import camera_scene
        from .camera_intent import obstacle_state
        camera_observed=camera_scene(search.collision.scene,search.t,end)
        if directed:
            probe_times=np.linspace(activation,end,65)[1:]
            probe_basis=np.column_stack([np.interp(probe_times,path.geometry['times'],np.asarray(path.geometry['shape_basis'])[:,j]) for j in range(3)])
            probe_target=positions_at(path,probe_times)
            observations=[obstacle_state(camera_observed,float(t)) for t in probe_times]
            camera_probes=(probe_basis,probe_target,np.array([v[0] for v in observations]),np.array([v[1] for v in observations]))
    best=None;last={}
    import itertools
    offsets=np.asarray(terminal_band(original,path.model,path.intrinsics,
        axis_only=bool(path.program.get('construction_contract')))['offsets_m'])
    axis_radius=[]
    for a in range(3):
        selected=offsets[np.sum(abs(offsets)>1e-8,axis=1)==1]
        vals=abs(selected[:,a]);axis_radius.append(max(1e-5,min(vals[vals>1e-8])))
    endpoint_rows=np.array(list(itertools.product((-1.,1.),repeat=3)))/np.array(axis_radius)
    reference_end=original.sample(original.duration)[0]
    for iteration in range(max_iterations):
        if iteration and time.monotonic()-start>seconds:break
        rows=[];values=[];cons=[];los=[];his=[]
        def residual(j,r,weight):rows.append(j*weight);values.append(np.atleast_1d(r)*weight)
        def constraint(j,lo,hi):cons.append(j);los.append(np.broadcast_to(lo,len(j)));his.append(np.broadcast_to(hi,len(j)))
        poses=target+basis@xcurve.reshape(3,3)
        max_position_error=max_orientation_error=0.
        if camera_probes is not None:
            pb,pt,pc,pr=camera_probes
            relative=(pt+pb@xcurve.reshape(3,3))[:,None]-pc
            distance=np.linalg.norm(relative,axis=2);clear=distance-pr-.06
            mask=clear<.08
            if np.any(mask):
                knots_index,_=np.nonzero(mask)
                normal=relative[mask]/np.maximum(distance[mask,None],1e-9)
                C=np.zeros((len(normal),size))
                C[:,-9:]=(pb[knots_index,:,None]*normal[:,None,:]).reshape(-1,9)
                residual(C,clear[mask]-.065,3000.)
        for k in range(n):
            if camera_observed is not None and not directed and ts[k]>activation:
                centers,radii=obstacle_state(camera_observed,float(ts[k]))
                relative=poses[k]-centers;distance=np.linalg.norm(relative,axis=1)
                clear=distance-radii-.06
                mask=clear<(.08 if directed else .045)
                if np.any(mask):
                    C=np.zeros((sum(mask),size))
                    C[:,-9:]=(relative[mask]/np.maximum(distance[mask,None],1e-9))@np.kron(basis[k:k+1],np.eye(3))
                    residual(C,clear[mask]-(.065 if directed else .045),3000. if directed else 220.)
            x=states[k];p,r,j=search.fk(x)
            j9=np.column_stack([np.r_[1.,0,0,0,0,0],np.r_[0.,1,0,0,0,0],j[:,1:]])
            tr=path.sample(ts[k])[1]
            if ts[k]>activation:tr=look_at(poses[k],original.subject)@look_at(target[k],original.subject).T@tr
            err=np.r_[p-poses[k],Rotation.from_matrix(r@tr.T).as_rotvec()]
            max_position_error=max(max_position_error,float(np.linalg.norm(err[:3])))
            max_orientation_error=max(max_orientation_error,float(np.linalg.norm(err[3:])))
            J=np.zeros((6,size));J[:,9*k:9*k+9]=j9
            if ts[k]>activation:
                jr=np.empty((3,3));eps=1e-5
                for a in range(3):
                    pp=poses[k].copy();pp[a]+=eps
                    rr=look_at(pp,original.subject)@look_at(poses[k],original.subject).T@tr
                    jr[:,a]=Rotation.from_matrix(rr@tr.T).as_rotvec()/eps
                J[:,-9:]=-np.vstack([np.eye(3),jr])@np.kron(basis[k:k+1],np.eye(3))
            residual(J,err,500. if path.program.get('construction_contract') else 120.)
            sigma=search.sigma(j)
            if k and sigma<.045:
                gradient=np.zeros(9);eps=2e-4
                for a in range(3,9):
                    xp=x.copy();xm=x.copy();xp[a]+=eps;xm[a]-=eps
                    gradient[a]=(search.sigma(search.fk(xp)[2])-search.sigma(search.fk(xm)[2]))/(2*eps)
                S=np.zeros((1,size));S[0,9*k:9*k+9]=gradient
                constraint(S,.027-sigma,np.inf)

            if k and collisions:
                col=collisions[k];d=col.clearance(x,search.base_z)
                # One closest surface per body: fixed eight rows per knot.
                idx=np.argmin(d,axis=0);dist=d[idx,np.arange(d.shape[1])]
                mask=dist<col.margin+.04
                if np.any(mask):
                    d,g=col.clearance(x,search.base_z,gradient=True)
                    idx=np.argmin(d,axis=0);dist=d[idx,np.arange(d.shape[1])];grad=g[idx,np.arange(d.shape[1])]
                    J=np.zeros((sum(mask),size));J[:,9*k:9*k+9]=grad[mask]
                    residual(J,dist[mask]-col.margin-.04,180.)
                dist=search.self_collision.distances(x);mask=dist<search.self_collision.margin+.012
                if np.any(mask):
                    dist,grad=search.self_collision.terms(x)
                    J=np.zeros((sum(mask),size));J[:,9*k:9*k+9]=grad[mask]
                    residual(J,dist[mask]-search.self_collision.margin-.012,120.)
        controls=[];control_j=[]
        for k in range(n-1):
            x,y=states[k:k+2];ang=(x[2]+y[2])/2;f=np.array([np.cos(ang),np.sin(ang)]);normal=np.array([-f[1],f[0]])
            diff=y-x
            J=np.zeros((1,size));J[0,9*k:9*k+2]=-normal;J[0,9*(k+1):9*(k+1)+2]=normal
            J[0,9*k+2]=J[0,9*(k+1)+2]=-.5*diff[:2]@f
            constraint(J,-diff[:2]@normal,-diff[:2]@normal)
            c=np.r_[diff[:2]@f,diff[2:]]/dt
            C=np.zeros((8,size));C[0,9*k:9*k+2]=-f/dt;C[0,9*(k+1):9*(k+1)+2]=f/dt
            C[0,9*k+2]=C[0,9*(k+1)+2]=.5*diff[:2]@normal/dt
            C[1:,9*k+2:9*k+9]=-np.eye(7)/dt;C[1:,9*(k+1)+2:9*(k+1)+9]=np.eye(7)/dt
            controls.append(c);control_j.append(C)
            constraint(C,-.88*search.limits.velocity-c,.88*search.limits.velocity-c)
            w=search.limits.wheels
            constraint(w@C,-.9*search.limits.wheel_speed-w@c,.9*search.limits.wheel_speed-w@c)
            residual(C,c,.12)
        for k,(c,C) in enumerate(zip(controls,control_j)):
            prev=search.velocity if k==0 else controls[k-1]
            D=C if k==0 else C-control_j[k-1]
            acc=(c-prev)/dt
            constraint(D/dt,-.75*search.limits.acceleration-acc,.75*search.limits.acceleration-acc)
            residual(D/dt,acc,.12)
        residual(control_j[-1],controls[-1],30.)
        chord=path.sample(end)[0]-path.sample(activation)[0];axis=chord/max(np.linalg.norm(chord),1e-8)
        motion=original.phase(activation)['motion'].lower()
        axes=[axis]
        if 'pedestal' in motion:
            axes=[np.array([0.,0.,np.sign(chord[2])])]
            planar=chord.copy();planar[2]=0.
            if np.linalg.norm(planar)>.05:axes.append(planar/np.linalg.norm(planar))
        elif 'truck' in motion:
            axis=original.sample(0.)[1][:,1];axis*=np.sign(chord@axis);axes=[axis]
        for axis in axes:
            for k in range(n-1):
                # The fixed incoming prefix cannot be changed by any curve
                # coefficient. Measured initial pose error can temporarily
                # oppose the authored direction while it converges. Do not
                # turn that immutable observation into an impossible zero-row
                # constraint in the suffix optimizer. Full execution direction
                # and curve checks still report this deviation independently.
                if ts[k+1]<=activation:continue
                C=np.zeros((1,size));C[0,-9:]=np.kron(basis[k+1]-basis[k],axis)
                constraint(C,-(poses[k+1]-poses[k])@axis,np.inf)
        # Low-dimensional curve regularity and continuous endpoint freedom.
        if path.program.get('construction_axis_world') is not None:
            selected_axis=np.asarray(path.program['construction_axis_world'])
            normals=np.asarray(path.program['construction_plane_normal_world'])[None]
            initial=np.asarray(path.program.get('shape_coefficients',path.program.get('unverified_seed_direction',[0.,0.,0.])+[0.,0.,0.]))
            for j in range(2):
                C=np.zeros((1,size));C[:,-9+3*j:-6+3*j]=normals
                fixed=-normals@(initial[3*j:3*j+3]+xcurve[3*j:3*j+3])
                width=path.program['construction_plane_tolerance_m']
                constraint(C,fixed-width,fixed+width)
            C=np.zeros((1,size));C[0,-9:-6]=selected_axis
            constraint(C,-selected_axis@(initial[:3]+xcurve[:3]),np.inf)
        J=np.zeros((9,size));J[:,-9:]=np.eye(9)
        residual(J,xcurve,.5)
        C=np.zeros((3,size));C[:,-6:-3]=np.eye(3)
        initial=np.asarray(path.program.get('shape_coefficients',[0.]*6))[3:6]
        constraint(C,-initial-xcurve[3:6],-initial-xcurve[3:6])
        C=np.zeros((8,size));C[:,-3:]=endpoint_rows
        constraint(C,np.full(8,-np.inf),.8-endpoint_rows@(poses[-1]-reference_end))
        residual(np.eye(size),np.zeros(size),.03)
        A=np.vstack(rows);e=np.concatenate(values)
        H=sparse.csc_matrix(A.T@A);g=A.T@e
        trust=.5 if iteration<2 else .15
        lower=np.full(size,-trust);upper=np.full(size,trust)
        lower[:9]=upper[:9]=0.
        for k in range(1,n):
            reserve=.12 if path.program.get('construction_contract') else .02
            lower[9*k+3:9*k+9]=np.maximum(lower[9*k+3:9*k+9],JOINT_LOWER+reserve-states[k,3:])
            upper[9*k+3:9*k+9]=np.minimum(upper[9*k+3:9*k+9],JOINT_UPPER-reserve-states[k,3:])
        curve_trust=.6 if path.program.get('construction_contract') and iteration<2 else .06
        lower[-9:]=-curve_trust;upper[-9:]=curve_trust
        constraint(np.eye(size),lower,upper)
        qp=osqp.OSQP();qp.setup(P=H,q=g,A=sparse.csc_matrix(np.vstack(cons)),l=np.concatenate(los),u=np.concatenate(his),verbose=False,eps_abs=1e-4 if directed else 2e-5,eps_rel=1e-4 if directed else 2e-5,max_iter=1500 if directed else 4000,polish=False)
        result=qp.solve()
        if (result.x is None or not np.isfinite(result.x).all() or result.info.status_val in (3,4,5,6) or
                (result.info.status_val not in (1,2) and result.info.prim_res>.005)):
            matrix=np.vstack(cons);lower_all=np.concatenate(los);upper_all=np.concatenate(his)
            zero_rows=np.max(np.abs(matrix),axis=1)<1e-12
            last=dict(reason='coarse_qp_'+result.info.status,iteration=iteration,
                contradictory_constant_rows=int(np.sum(zero_rows&((lower_all>1e-8)|(upper_all< -1e-8)))));break
        step=result.x;states+=step[:9*n].reshape(n,9);xcurve+=step[-9:]
        last=dict(reason='coarse_sqp_seed',iteration=iteration+1,max_step=float(abs(step).max()),objective=float(e@e),qp_status=result.info.status,qp_residual=float(result.info.prim_res),linearization_position_m=max_position_error,linearization_orientation_rad=max_orientation_error)
        best=(states.copy(),xcurve.copy())
        if np.max(np.abs(step))<step_tolerance:break
    if best is None:return None,dict(last,wall_s=time.monotonic()-start)
    states,xcurve=best
    times=np.asarray(path.geometry['times']);u=(times-activation)/span;s=10*u**3-15*u**4+6*u**5;b=64*u**3*(1-u)**3
    fine_basis=np.asarray(path.geometry.get('shape_basis',np.column_stack([b,b*(2*u-1),s])))
    points=np.asarray(path.geometry['positions'])+fine_basis@xcurve.reshape(3,3)
    terminal=terminal_error(original,points[-1],path.model,path.intrinsics)
    if not inside_terminal_band(terminal):return None,dict(last,reason='joint_endpoint_outside_band',terminal=terminal,wall_s=time.monotonic()-start)
    controls=[]
    for k in range(n-1):
        x,y=states[k:k+2];a=(x[2]+y[2])/2
        controls.append(np.r_[(y[:2]-x[:2])@np.array([np.cos(a),np.sin(a)]),y[2:]-x[2:]]/dt)
    trace=[]
    for k,t in enumerate(ts):
        p,r,_=search.fk(states[k]);c=search.velocity if k==0 else np.zeros(8) if k==n-1 else .5*(controls[k-1]+controls[k])
        trace.append(dict(t=float(t),state=states[k].tolist(),control=np.asarray(c).tolist(),position=p.tolist(),quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist()))
    geometry=dict(path.geometry,positions=points.tolist(),joint_seed=trace,joint_seed_preference_only=False)
    program=dict(path.program,whole_body_search='sparse_joint_curve_sqp',endpoint_offset_m=(points[-1]-original.sample(original.duration)[0]).tolist())
    if path.program.get('construction_axis_world') is not None:
        initial=np.asarray(path.program.get('shape_coefficients',path.program.get('unverified_seed_direction',[0.,0.,0.])+[0.,0.,0.]))
        coefficients=(initial+xcurve[:6]).reshape(2,3)
        selected_axis=np.asarray(path.program['construction_axis_world'])
        perpendicular=coefficients@np.asarray(path.program['construction_plane_normal_world'])
        if np.max(np.abs(perpendicular))>path.program['construction_plane_tolerance_m']+1e-4 or coefficients[0]@selected_axis< -1e-4:
            return None,dict(last,reason='construction_direction_contract',wall_s=time.monotonic()-start)
        program['shape_coefficients']=coefficients.ravel().tolist()
    fitted=IntentPath(original,path.previous,activation,program,path.model,path.intrinsics,geometry=geometry)
    residuals=[]
    targets,rotations=poses_at(fitted,ts)
    for k,t in enumerate(ts):
        p,r,_=search.fk(states[k]);tp,tr=targets[k],rotations[k]
        residuals.append([float(np.linalg.norm(p-tp)),float(Rotation.from_matrix(r@tr.T).magnitude())])
    last['coarse_max_position_error_m']=max(row[0] for row in residuals)
    last['coarse_max_orientation_error_rad']=max(row[1] for row in residuals)
    quality=curve_quality(fitted)
    if not quality['direction_pass'] or not quality['regular_curve_pass']:return None,dict(last,reason='joint_curve_shape',quality=quality,wall_s=time.monotonic()-start)
    return fitted,dict(last,wall_s=time.monotonic()-start,quality=quality)
