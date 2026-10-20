"""Private bounded native-input operating forecast; not learned-policy dynamics.

All three planned inputs retain their actual first value,50Hz slew and150ms
held-input history. Other-axis pending-rate residuals use the original150ms
inertia window, without an additional fitted time constant. Future joint/contact
or stability is unknown and actual measured safety remains authoritative.
"""
import numpy as np
from scipy.spatial.transform import Rotation

DT=.02
DELAY=.15
CAPS=np.array([.2,.2,.2])
SLEW=np.array([.5,.5,.8])


class BoundedBodyPrediction:
    def __init__(self,backend,*,stamp_s,body,link,measured_velocity,previous_applied,input_history,
                 prediction_intervals=(0,.2,.4,.6,.8,1.,1.2),pending_mode='linear_residual'):
        if not np.isfinite(stamp_s):raise ValueError('finite actual prediction time required')
        if pending_mode not in ('linear_residual','measured_hold'):raise ValueError('explicit pending model required')
        self.stamp=float(stamp_s);self.pending_mode=pending_mode
        self.body=backend.rigid(body).copy();self.link=backend.rigid(link).copy()
        self.measured=np.asarray(measured_velocity,float).copy();self.previous=np.asarray(previous_applied,float).copy()
        if self.measured.shape!=(3,)or self.previous.shape!=(3,)or not np.isfinite(np.r_[self.measured,self.previous]).all()or np.any(abs(self.previous)>CAPS+1e-9):
            raise ValueError('finite actual body velocity and bounded previous input required')
        if len(input_history)<9:raise ValueError('actual50Hz input history must cover150ms delay')
        self.history_times=np.asarray([float(t)-self.stamp for t,_ in input_history]);self.history_inputs=np.asarray([u for _,u in input_history],float)
        if (self.history_inputs.shape!=(len(input_history),3)or not np.isfinite(np.r_[self.history_times,self.history_inputs.reshape(-1)]).all()
                or np.any(abs(self.history_inputs)>CAPS+1e-9)
                or not np.allclose(np.diff(self.history_times),DT,atol=1e-9,rtol=0)
                or not np.isclose(self.history_times[-1],-DT,atol=1e-9,rtol=0)
                or self.history_times[0]>-DELAY+1e-9
                or not np.allclose(self.history_inputs[-1],self.previous,atol=1e-9,rtol=0)):
            raise ValueError('fresh complete actual history must end one50Hz tick ago at previous input')
        self.queries=np.asarray(prediction_intervals,float)
        if self.queries.ndim!=1 or not len(self.queries)or not np.isfinite(self.queries).all()or np.any(self.queries<0)or np.any(self.queries>1.2+1e-9):
            raise ValueError('finite original0..1.2s query intervals required')
        self.horizon=float(max(self.queries));self.command_times=np.arange(max(1,int(np.ceil(self.horizon/DT))+1))*DT
        events=np.r_[0.,DELAY,self.history_times+DELAY,self.command_times+DELAY,self.queries,self.horizon]
        self.events=np.unique(events[(events>=0)&(events<=self.horizon+1e-12)])
        self.events[-1]=self.horizon;self.delta=np.diff(self.events)
        # Preserve the last isclose match even when explicit nodes are nearly
        # coincident. Repeated exact configured queries need no event scan.
        self.query_event_indices={float(query):int(np.flatnonzero(np.isclose(
            self.events,query,atol=1e-10,rtol=0))[-1])for query in self.queries}
        held_times=np.r_[self.history_times,self.command_times]
        self.delayed_indices=np.clip(np.searchsorted(held_times,self.events[:-1]-DELAY+1e-12,side='right')-1,0,len(held_times)-1)
        old_index=max(0,np.searchsorted(self.history_times,-DELAY+1e-12,side='right')-1)
        self.oldest_delayed_input=self.history_inputs[old_index].copy()
        self.root_to_link=np.linalg.inv(self.body)@self.link
        self.cache={}

    def admissible(self,first):
        value=np.asarray(first,float)
        if (value.shape!=(3,)or not np.isfinite(value).all()or np.any(abs(value)>CAPS+1e-9)
                or np.any(abs(value-self.previous)>SLEW*DT+1e-9)):
            raise ValueError('forecast first input must equal an actual cap/slew-admissible command')
        return value

    def trajectory(self,first,future):
        first=self.admissible(first);future=np.asarray(future,float)
        if future.shape!=(3,)or not np.isfinite(future).all()or np.any(abs(future)>CAPS+1e-9):raise ValueError('bounded finite future native target required')
        key=(tuple(first),tuple(future))
        if key in self.cache:return self.cache[key]
        commands=first+np.clip(future-first,-self.command_times[:,None]*SLEW,self.command_times[:,None]*SLEW)
        held=np.vstack([self.history_inputs,commands]);delayed=held[self.delayed_indices]
        dt=self.delta
        steady=delayed[:,0]*np.where(delayed[:,0]<0,.55,1.)
        attenuation=np.exp(-np.cumsum(dt)/.35);decay=np.exp(-dt/.35)
        after=attenuation*(self.measured[0]+np.cumsum(steady*(-np.expm1(-dt/.35))/attenuation))
        before=np.r_[self.measured[0],after[:-1]]
        dx=before*.35*(-np.expm1(-dt/.35))+steady*(dt-.35*(-np.expm1(-dt/.35)))
        start=self.events[:-1];end=self.events[1:]
        if self.pending_mode=='linear_residual':
            # Initial rate is exactly measured. Pending real inputs continue
            # through the old delay. Its measured-command residual decays
            # linearly over that existing150ms inertia window, not a new tau.
            b=np.minimum(end,DELAY);a=np.minimum(start,DELAY)
            residual_integral=(b-a)-(b*b-a*a)/(2*DELAY)
            dy=delayed[:,1]*dt+(self.measured[1]-self.oldest_delayed_input[1])*residual_integral
            da=delayed[:,2]*dt+(self.measured[2]-self.oldest_delayed_input[2])*residual_integral
        else:
            prefix=np.minimum(end,DELAY)-np.minimum(start,DELAY)
            dy=self.measured[1]*prefix+delayed[:,1]*(dt-prefix)
            da=self.measured[2]*prefix+delayed[:,2]*(dt-prefix)
        yaw=np.r_[0.,np.cumsum(da)];middle=yaw[:-1]+da/2
        displacement=np.c_[dx,dy,np.zeros_like(dx)]@self.body[:3,:3].T
        c,s=np.cos(middle),np.sin(middle)
        planar=np.c_[c*displacement[:,0]-s*displacement[:,1],s*displacement[:,0]+c*displacement[:,1]]
        roots=np.tile(np.eye(4),(len(self.events),1,1));roots[:,:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@self.body[:3,:3]
        roots[:,:3,3]=self.body[:3,3];roots[1:,:2,3]+=np.cumsum(planar,axis=0)
        links=roots@self.root_to_link;links[0]=self.link
        result=dict(links=links,roots=roots,issued=commands,command_times=self.command_times.copy())
        self.cache[key]=result;return result

    def predict_link(self,first,future,elapsed):
        try:
            index=self.query_event_indices.get(elapsed)if isinstance(elapsed,(int,float,np.integer,np.floating))else None
        except TypeError:
            index=None
        if index is None:
            indices=np.flatnonzero(np.isclose(self.events,elapsed,atol=1e-10,rtol=0))
            if not len(indices):raise ValueError('prediction query must be an explicitly configured original node')
            index=int(indices[-1])
        return self.trajectory(first,future)['links'][index].copy()

    def manifest(self):
        return dict(profile='bounded_ramp_history_v2',dt_s=DT,delay_s=DELAY,
            first_input='actual issued cap/slew-admissible input',future_input='original50Hz slew ramp to bounded target',
            x_model='continuous steady(u)=u*(.55 if negative else1),tau.35; forward response uncalibrated',
            other_axes_model=self.pending_mode,
            other_axes_note='Original150ms inertia window only; measured initial rate plus actual pending input history; uncalibrated operating approximation',
            scope='workspace support model only; learned18/joints/contact/stability unknown')
