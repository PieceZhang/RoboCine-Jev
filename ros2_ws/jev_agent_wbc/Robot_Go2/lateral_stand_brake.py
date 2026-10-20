"""Fresh measured velocity braking in the original late3s stop window.

This is an input feedback operating hypothesis, not a learned/contact dynamics
certificate. Original WORLD camera Task and18 policy actions remain unchanged.
"""
import importlib.util
from pathlib import Path
import numpy as np

_spec=importlib.util.spec_from_file_location('go2_lateral_bounded_model',Path(__file__).with_name('bounded_body_prediction.py'))
_model=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_model)
PROFILE='vy_p05_v1'
XY_PROFILE='vxy_p05_v1'
GAIN=.5
ASSIST_CAP=.03
DT=.02
SLEW=np.array([.5,.5,.8])
CAPS=np.array([.2,.2,.2])


def safe_brake(previous):
    previous=np.asarray(previous,float)
    if previous.shape!=(3,)or not np.isfinite(previous).all()or np.any(abs(previous)>CAPS+1e-9):
        raise ValueError('finite bounded actual previous input required')
    return np.clip(np.zeros(3),previous-SLEW*DT,previous+SLEW*DT)


class LateralStandBrake:
    def __init__(self,backend,*,stamp_s,measurement_stamp_s,task_end_s,stop_seconds,
            body,link,measured_velocity,previous_applied,input_history,prediction_intervals,profile=PROFILE):
        if profile not in (PROFILE,XY_PROFILE):raise ValueError('explicit supported velocity-brake profile required')
        self.profile=profile
        scalars=[stamp_s,measurement_stamp_s,task_end_s,stop_seconds]
        if any(isinstance(v,(bool,np.bool_))or not np.isfinite(v)for v in scalars):
            raise ValueError('finite same-stamp late-stop evidence required')
        if (abs(stamp_s-measurement_stamp_s)>1e-9 or stop_seconds!=5.
                or not task_end_s+2-1e-9<=stamp_s<task_end_s+stop_seconds-1e-9):
            raise ValueError('active brake requires original same-stamp end+2..end+5 window')
        self.stamp=float(stamp_s);self.measurement_stamp=float(measurement_stamp_s);self.end=float(task_end_s)
        self.measured=np.asarray(measured_velocity,float).copy();self.previous=np.asarray(previous_applied,float).copy()
        safe_brake(self.previous)
        if self.measured.shape!=(3,)or not np.isfinite(self.measured).all():raise ValueError('fresh actual body velocity required')
        if profile==PROFILE:
            if np.any(abs(self.previous[[0,2]])>1e-12)or abs(self.previous[1])>ASSIST_CAP+1e-9:
                raise ValueError('original allocation must finish braking before lateral-only assist')
            self.raw=np.array([0.,-GAIN*self.measured[1],0.])
            assist_caps=np.array([0.,ASSIST_CAP,0.])
        else:
            if abs(self.previous[2])>1e-12 or np.any(abs(self.previous[:2])>ASSIST_CAP+1e-9):
                raise ValueError('original allocation must finish braking before planar assist')
            self.raw=np.array([-GAIN*self.measured[0],-GAIN*self.measured[1],0.])
            assist_caps=np.array([ASSIST_CAP,ASSIST_CAP,0.])
        self.target=np.clip(self.raw,-assist_caps,assist_caps)
        self.first=np.clip(self.target,np.maximum(-CAPS,self.previous-SLEW*DT),np.minimum(CAPS,self.previous+SLEW*DT))
        self.model=_model.BoundedBodyPrediction(backend,stamp_s=stamp_s,body=body,link=link,
            measured_velocity=self.measured,previous_applied=self.previous,input_history=input_history,
            prediction_intervals=prediction_intervals,pending_mode='linear_residual')
        self.model.admissible(self.first)
        self.report=dict(profile=profile,phase='active_lateral_brake'if profile==PROFILE else'active_planar_brake',stamp_s=self.stamp,
            measurement_stamp_s=self.measurement_stamp,reference_stamp_s=self.end,
            reference_mode='explicit_original_endpoint_hold',requested_stop_velocity=[0.,0.,0.],
            measured_body_velocity=self.measured.tolist(),feedback_gain=GAIN,assist_vy_cap_mps=ASSIST_CAP,
            raw_feedback_input=self.raw.tolist(),bounded_feedback_target=self.target.tolist(),
            cap_slew_admissible_first=self.first.tolist(),previous_actual_input=self.previous.tolist(),
            future_input_assumption='Frozen current measured feedback target with original50Hz slew; next real tick replans from fresh measurement.',
            prediction_model=self.model.manifest(),original_world_task_modified=False,all18_learned_actions_unchanged=True)
        if profile==XY_PROFILE:self.report.update(assist_vx_cap_mps=ASSIST_CAP,assist_caps_mps=assist_caps.tolist(),feedback_axes=['body_vx','body_vy'])

    def predict_link(self,first,elapsed):
        first=self.model.admissible(first)
        if not np.allclose(first,self.first,atol=1e-12,rtol=0):raise ValueError('active brake forecast must use its actual locked first input')
        return self.model.predict_link(self.first,self.target,elapsed)

    def finalize(self,candidate,*,veto_reasons=()):
        candidate=np.asarray(candidate,float)
        if candidate.shape!=(3,)or not np.isfinite(candidate).all():raise ValueError('finite final body input required')
        reasons=list(veto_reasons)
        if not np.allclose(candidate,self.first,atol=1e-12,rtol=0):reasons.append('later_body_input_override')
        issued=safe_brake(self.previous)if reasons else self.first.copy()
        self.report.update(final_body_input=issued.tolist(),veto_reasons=reasons,
            actual_standing_input=bool(np.all(issued==0)),final_feedback_only=not bool(reasons),
            guard_projection_delta=(issued-self.first).tolist())
        return issued

    def manifest(self):return self.report
