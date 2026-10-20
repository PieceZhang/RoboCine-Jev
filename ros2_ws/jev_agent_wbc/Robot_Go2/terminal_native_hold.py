"""Explicit end+2..end+5 native-input hold; immutable WORLD Task is still audited.

Only an already consumed float32 native EE7 is selected. Learned18 actions and
all downstream real domain, FCL, body-slew and Task feedback guards remain live.
"""
import hashlib
import json
import numpy as np

PROFILE='end2_actual_v1'


def task_binding(task):
    data=task.to_dict()if hasattr(task,'to_dict')else task
    encoded=json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)
    from jev_agent_wbc.wbc_protocol import TimedTask
    value=TimedTask.from_dict(json.loads(encoded))
    return value,hashlib.sha256(encoded.encode()).hexdigest()


def same_native_bits(a,b):
    a=np.asarray(a,dtype=np.float32);b=np.asarray(b,dtype=np.float32)
    return a.shape==b.shape==(7,)and np.array_equal(a.view(np.uint32),b.view(np.uint32))


class TerminalNativeHold:
    def __init__(self,task,stop_seconds=5.):
        self.task,self.sha=task_binding(task)
        if stop_seconds!=5.:raise ValueError('native hold requires the original fixed5s stop')
        self.end=self.task.trajectory.end_s;self.stop_end=self.end+stop_seconds
        self.command=None;self.capture=None;self.first_consumption=None;self.cancelled=False
        self.last_report=None

    def in_window(self,stamp_s):return self.end+2-1e-9<=stamp_s<self.stop_end-1e-9

    def select(self,candidate,*,stamp_s,measurement_stamp_s,previous_consumed_native,previous_issue_stamp_s,
            previous_consumption_seq,previous_body_input,issued_body_input,domain,collision_evidence,
            original_support_ok,world_position_error_m,world_orientation_error_rad):
        native=np.asarray(candidate,float).copy()
        report=dict(profile=PROFILE,active=False,selected=False,status='before_fixed_hold_window',
            task_sha256=self.sha,task_end_s=self.end,original_world_reference_modified=False)
        if self.last_report is not None and self.last_report['status']=='prepared_waiting_actual_policy_consumption':
            self.cancelled=True;self.command=None
        self.last_report=report
        if not self.in_window(stamp_s):return native,report
        try:
            scalars=[stamp_s,measurement_stamp_s,previous_issue_stamp_s,world_position_error_m,world_orientation_error_rad]
            if any(isinstance(v,(bool,np.bool_))or not np.isfinite(v)for v in scalars):raise ValueError('finite_fresh_hold_evidence_required')
            if self.cancelled:raise ValueError('hold_cancelled_for_this_task')
            if abs(stamp_s-measurement_stamp_s)>1e-9:raise ValueError('hold_requires_same_tick_measurement')
            if (not np.all(np.asarray(previous_body_input)==0)or not np.all(np.asarray(issued_body_input)==0)):
                raise ValueError('hold_requires_actual_previous_and_current_literal_zero_body_input')
            if not original_support_ok:raise ValueError('original_task_support_or_domain_rejected')
            if (collision_evidence.get('status')!='pass'or
                    not -1e-9<=stamp_s-collision_evidence.get('stamp_s',-np.inf)<=.06+1e-9):
                raise ValueError('fresh_original_FCL_pass_required')
            if (world_position_error_m>=self.task.position_tolerance_m or
                    world_orientation_error_rad>=self.task.orientation_tolerance_rad):
                raise ValueError('original_task_measured_error_outside_tolerance')
            if self.command is None:
                if stamp_s>self.end+2+.02+1e-9:raise ValueError('fixed_first_hold_tick_was_missed')
                if not -1e-9<=stamp_s-previous_issue_stamp_s<=.06+1e-9:raise ValueError('previous_consumed_input_not_fresh')
                command=np.asarray(previous_consumed_native,dtype=np.float32)
                if command.shape!=(7,)or not np.isfinite(command).all():raise ValueError('finite_consumed_float32_native7_required')
                if domain.check(command.astype(float))['status']!='inside_outer_domain_unverified':raise ValueError('consumed_input_outside_original_domain')
                self.command=command.copy()
                self.capture=dict(stamp_s=float(stamp_s),source_issue_stamp_s=float(previous_issue_stamp_s),
                    source_consumption_seq=int(previous_consumption_seq),native7_float32=command.astype(float).tolist())
            if domain.check(self.command.astype(float))['status']!='inside_outer_domain_unverified':raise ValueError('held_input_outside_original_domain')
            report.update(selected=True,status='prepared_waiting_actual_policy_consumption',
                selected_issue_stamp_s=float(stamp_s),selected_consumption_seq=int(previous_consumption_seq)+1,
                capture_stamp_s=self.capture['stamp_s'],source_issue_stamp_s=self.capture['source_issue_stamp_s'])
            return self.command.astype(float).copy(),report
        except (ValueError,TypeError)as error:
            self.cancelled=True;self.command=None;report.update(status=str(error))
            return native,report

    def confirm(self,report,*,actual_native,actual_body_input,stamp_s,consumption_seq,override_reasons):
        if (report is not self.last_report or not report['selected']or
                report['status']!='prepared_waiting_actual_policy_consumption'
                or report['task_sha256']!=self.sha or not self.in_window(stamp_s)):return False
        reasons=[reason for reason in override_reasons if reason!='terminal_native_hold']
        if (self.command is None or reasons or not np.all(np.asarray(actual_body_input)==0)
                or not same_native_bits(actual_native,self.command)
                or stamp_s!=report['selected_issue_stamp_s']or consumption_seq!=report['selected_consumption_seq']):
            self.cancelled=True;self.command=None
            report.update(active=False,status='later_override_cancelled_hold',override_reasons=reasons)
            return False
        if self.first_consumption is None:
            self.first_consumption=dict(seq=int(consumption_seq),stamp_s=float(stamp_s))
        report.update(active=True,status='actual_policy_consumed_hold',
            first_consumption_seq=self.first_consumption['seq'],first_consumption_stamp_s=self.first_consumption['stamp_s'])
        return True

    def manifest(self):
        return dict(profile=PROFILE,task=self.task.to_dict(),task_sha256=self.sha,task_end_s=self.end,
            fixed_capture_offset_s=2.,fixed_stop_end_s=self.stop_end,capture=self.capture,
            first_consumption=self.first_consumption,cancelled=self.cancelled,original_world_reference_modified=False)
