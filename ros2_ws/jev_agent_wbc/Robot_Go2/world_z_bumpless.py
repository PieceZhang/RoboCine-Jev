"""Fresh Task nativeZ initialization around unchanged pinned WorldZFeedback.

The full immutable Task, absolute clock and native learned18 actions are not
modified. Profilev1 requires an observed steady bias window. Profilev2 only
preserves a fresh actual control output; it makes no steady bias claim. Both
retain original fresh safety/domain evidence and first-tick rollback fate.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

_BASE_PATH=Path(__file__).with_name('world_z_feedback.py')
_BASE_SHA='3d495f4cfbd81ed45ba744ead7d831c53253d4e597163e0319a44871b9d37a6a'
if hashlib.sha256(_BASE_PATH.read_bytes()).hexdigest()!=_BASE_SHA:
    raise RuntimeError('bumpless wrapper requires its reviewed immutable WorldZFeedback transaction version')
_spec=importlib.util.spec_from_file_location('go2_seed_original_world_z',_BASE_PATH)
_base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_base)

DT=.02
MAX_AGE=.06
WINDOW=.5
SPAN=.001
PROFILES=('fresh_steady_native_z_v1','fresh_actual_output_continuity_v2')


def _finite(value):
    return isinstance(value,(int,float,np.integer,np.floating))and not isinstance(value,(bool,np.bool_))and np.isfinite(value)


def canonical_task(task):
    value=task.to_dict()if hasattr(task,'to_dict')else task
    encoded=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    frozen=json.loads(encoded)
    from jev_agent_wbc.wbc_protocol import TimedTask
    TimedTask.from_dict(frozen)
    return frozen,hashlib.sha256(encoded.encode()).hexdigest()


def _actual_native(row):
    # The unchanged runner passes this exact float32 EE7 into the policy's
    # unclipped(-100,100)feature history. Reconstruct its actual precision.
    value=np.asarray(row['actual_native_command'],dtype=np.float32).astype(float)
    if value.shape!=(7,)or not np.isfinite(value).all():raise ValueError('invalid actual native7 history')
    return value


def _verified_terminal_hold(rows,manifest):
    """Narrow exception for intentional fixed-clock hold, never safety vetoes."""
    if manifest is None:return False
    frozen,task_sha=canonical_task(manifest['task'])
    from jev_agent_wbc.wbc_protocol import TimedTask
    task=TimedTask.from_dict(frozen);end=task.trajectory.end_s
    capture=manifest['capture'];first=manifest['first_consumption']
    if (manifest.get('profile')!='end2_actual_v1'or manifest.get('cancelled')is not False
            or manifest.get('task_sha256')!=task_sha or manifest.get('task_end_s')!=end
            or manifest.get('fixed_capture_offset_s')!=2. or manifest.get('fixed_stop_end_s')!=end+5
            or capture is None or first is None):raise ValueError('unverified dedicated terminal hold manifest')
    if (not end+2-1e-9<=capture['stamp_s']<=end+2+.02+1e-9 or first['stamp_s']!=capture['stamp_s']
            or first['seq']!=capture['source_consumption_seq']+1
            or not 0<=capture['stamp_s']-capture['source_issue_stamp_s']<=MAX_AGE+1e-9):
        raise ValueError('terminal hold original clock/actual first consumption mismatch')
    command=np.asarray(capture['native7_float32'],dtype=np.float32)
    if command.shape!=(7,)or not np.isfinite(command).all():raise ValueError('terminal hold actual float32 native7 missing')
    for row in rows:
        report=row.get('terminal_native_hold',{});zi=row['actual_native_domain']['world_z_feedback']
        actual=np.asarray(row['actual_native_command'],dtype=np.float32)
        if (report.get('profile')!='end2_actual_v1'or report.get('active')is not True
                or report.get('selected')is not True
                or report.get('status')!='actual_policy_consumed_hold'or report.get('task_sha256')!=task_sha
                or report.get('task_end_s')!=end or report.get('capture_stamp_s')!=capture['stamp_s']
                or report.get('source_issue_stamp_s')!=capture['source_issue_stamp_s']
                or not _finite(report.get('selected_issue_stamp_s'))
                or abs(report['selected_issue_stamp_s']-(row['stamp_s']-DT))>1e-9
                or report.get('selected_consumption_seq')!=zi.get('sample_seq')
                or report.get('first_consumption_seq')!=first['seq']or report.get('first_consumption_stamp_s')!=first['stamp_s']
                or row['task_feedback']['task_id']!=task.task_id
                or not end+2-1e-9<=row['stamp_s']-DT<end+5
                or not np.all(np.asarray(row['applied_policy_velocity'])==0)
                or actual.shape!=(7,)or not np.array_equal(actual.view(np.uint32),command.view(np.uint32))):
            raise ValueError('all26 rows must bind one original Task hold, actual bit-identical7 and zero body input')
        if (zi.get('integration_allowed')is not False or zi.get('externally_rolled_back')is not True
                or zi.get('external_rollback_reason')!='terminal_native_hold'
                or zi.get('native_correction_accepted')is not False
                or abs(zi['integral_after_m']-zi['integral_before_m'])>1e-12):
            raise ValueError('intentional terminal hold cannot hide integration or another veto')
    return True


def prepare_calibration(backend,task,*,now_s,observation_stamp_s,reference_optical,measured_optical,
        link0,extrinsic,previous_applied_native,previous_issue_stamp_s,history_rows,domain,previous_terminal_hold=None,
        initialization_profile='fresh_steady_native_z_v1'):
    """One admission scan, with a separately declared steady or continuity scope."""
    frozen,task_sha=canonical_task(task)
    if initialization_profile not in PROFILES:raise ValueError('explicit supported admission initialization profile required')
    continuity=initialization_profile=='fresh_actual_output_continuity_v2'
    record=dict(calibration_id=task_sha[:20],task_sha256=task_sha,task_id=frozen['task_id'],
        initialization_profile=initialization_profile,steady_bias_calibration_claim=not continuity,
        admission_stamp_s=float(now_s)if _finite(now_s)else None,
        eligible=False,reason='not_evaluated',seed_raw_m=None,seed_clipped_m=0.,
        expected_remaining_native_z_jump_m=None,transaction_status='not_prepared',
        task_modified=False,task_clock_modified=False,base_feedback_sha256=_BASE_SHA)
    try:
        if (not _finite(now_s)or not _finite(observation_stamp_s)or now_s<0
                or abs(now_s-observation_stamp_s)>1e-9 or abs(now_s-frozen['issued_at_s'])>1e-9
                or abs(now_s-frozen['camera_trajectory']['start_s'])>1e-9):
            raise ValueError('fresh same-stamp Task admission required')
        ref=backend.rigid(reference_optical);camera=backend.rigid(measured_optical)
        link=backend.rigid(link0);mount=backend.rigid(extrinsic)
        from jev_agent_wbc.wbc_protocol import TimedTask,optical_camera
        pose=optical_camera(TimedTask.from_dict(frozen).trajectory.sample(now_s))
        actual_task_reference=np.eye(4);actual_task_reference[:3,:3]=pose.rotation;actual_task_reference[:3,3]=pose.translation_m
        if not np.allclose(ref,actual_task_reference,atol=1e-9,rtol=0):raise ValueError('reference must bind the complete accepted Task optical pose')
        if not np.allclose(ref,camera,atol=1e-9,rtol=0):raise ValueError('new original reference must equal current measured camera')
        record.update(measured_camera_sha256=hashlib.sha256(camera.tobytes()).hexdigest(),
            measured_link0_sha256=hashlib.sha256(link.tobytes()).hexdigest(),
            camera_extrinsic_sha256=hashlib.sha256(mount.tobytes()).hexdigest())
        raw_native=backend.camera_to_mixed(ref,link,mount)
        record['raw_reference_ee_world_z_m']=float(raw_native[2])
        if previous_applied_native is None or previous_issue_stamp_s is None:
            raise ValueError('first Task or missing prior actually issued native7')
        previous=np.asarray(previous_applied_native,dtype=np.float32).astype(float)
        if previous.shape!=(7,)or not np.isfinite(previous).all()or not _finite(previous_issue_stamp_s):
            raise ValueError('finite prior actually issued native7 and timestamp required')
        age=now_s-previous_issue_stamp_s
        if age<-1e-9 or age>MAX_AGE+1e-9:raise ValueError('prior actually issued native7 is future or older than60ms')
        if domain.check(raw_native)['status']!='inside_outer_domain_unverified' or domain.check(previous)['status']!='inside_outer_domain_unverified':
            raise ValueError('fresh native reference and prior applied input must be in original domain')
        selected=[row for row in history_rows if row['stamp_s']>=now_s-WINDOW-1e-9]
        stamps=np.asarray([row['stamp_s']for row in selected],float)
        if (len(stamps)<26 or not np.isfinite(stamps).all()or abs(stamps[-1]-now_s)>1e-9
                or stamps[0]>now_s-WINDOW+1e-9 or stamps[0]<now_s-WINDOW-1e-9
                or not np.allclose(np.diff(stamps),DT,atol=1e-9,rtol=0)):
            raise ValueError('complete fresh original50Hz half-second steady window required')
        native=np.asarray([_actual_native(row)for row in selected])
        if not np.array_equal(native[-1],previous)or abs((stamps[-1]-DT)-previous_issue_stamp_s)>1e-9:
            raise ValueError('last completed record must bind the actually issued native7/time')
        fixed_hold=_verified_terminal_hold(selected,previous_terminal_hold)
        record['prior_calibration_source']=('verified_fixed_terminal_actual_native_hold'if fixed_hold else
            'verified_actual_native_feedback_history'if continuity else'steady_unoverridden_native_feedback')
        old_task_ids=[]
        for row in selected:
            feedback=row['task_feedback'];old_task_ids.append(feedback['task_id'])
            if feedback.get('feasible')is False:raise ValueError('prior unsupported task interval cannot calibrate')
            for family in ('configuration','actuation','self_collision','policy_domain'):
                evidence=feedback['constraints'][family]
                if evidence['status']!='pass':raise ValueError('prior native safety family not pass')
                stamp=evidence['stamp_s']
                if not _finite(stamp)or not -1e-9<=row['stamp_s']-stamp<=MAX_AGE+1e-9:
                    raise ValueError('prior safety evidence is stale or future')
            if row['fine_collision_evidence']['status']!='pass':raise ValueError('prior fresh FCL pass required')
            if row.get('native_domain_recovery',{}).get('active')or row.get('native_camera_parking',{}).get('active'):
                raise ValueError('prior recovery or native hold cannot calibrate')
            if row['actual_native_domain']['status']!='inside_outer_domain_unverified' or row['requested_native_domain']['status']!='inside_outer_domain_unverified':
                raise ValueError('prior native command/reference domain failure')
            zi=row['actual_native_domain']['world_z_feedback']
            if not fixed_hold and (zi.get('externally_rolled_back')or not zi.get('native_correction_accepted')):
                raise ValueError('prior overridden native correction cannot calibrate')
        if len(set(old_task_ids))!=1:raise ValueError('steady window must belong to one prior accepted Task')
        inverse=np.linalg.inv(mount)
        actual_ee_z=np.asarray([(backend.rigid(row['T_world_camera_optical'])@inverse)[2,3]for row in selected])
        native_z=native[:,2];bias=native_z-actual_ee_z
        spans=dict(native_z=float(np.ptp(native_z)),actual_ee_z=float(np.ptp(actual_ee_z)),bias=float(np.ptp(bias)))
        record.update(window_first_stamp_s=float(stamps[0]),window_last_stamp_s=float(stamps[-1]),
            window_samples=len(stamps),prior_task_id=old_task_ids[0],steady_spans_m=spans,
            native_z_sequence_m=native_z.tolist(),actual_ee_z_sequence_m=actual_ee_z.tolist(),bias_sequence_m=bias.tolist(),
            previous_native_issue_stamp_s=float(previous_issue_stamp_s),previous_native_age_s=float(age),
            previous_actual_native_z_m=float(previous[2]),applied_precision='actual float32 native policy feature')
        record['steady_window_within_legacy_1mm']=not any(value>SPAN+1e-12 for value in spans.values())
        if not continuity and not record['steady_window_within_legacy_1mm']:
            raise ValueError('prior nativeZ/actualEEZ/bias span exceeds1mm')
        seed=float(previous[2]-raw_native[2]);clipped=float(np.clip(seed,-.025,.025))
        corrected=ref.copy();corrected[2,3]+=clipped
        candidate=backend.camera_to_mixed(corrected,link,mount)
        if domain.check(candidate)['status']!='inside_outer_domain_unverified':raise ValueError('bounded seeded native input outside original domain')
        record.update(eligible=True,reason='fresh_actual_output_continuity'if continuity else'fresh_steady_prior_native_bias',seed_raw_m=seed,seed_clipped_m=clipped,
            expected_remaining_native_z_jump_m=float(candidate[2]-previous[2]),transaction_status='eligible_not_committed')
    except (ValueError,KeyError,TypeError,IndexError,np.linalg.LinAlgError)as error:
        record.update(eligible=False,reason=str(error),seed_clipped_m=0.,transaction_status='rejected_reset_zero')
    return record


class BumplessWorldZFeedback(_base.WorldZFeedback):
    """Seed prepares only at full-SHA admission, commits with actual first issue."""
    def __init__(self,initialization_profile='fresh_steady_native_z_v1',**kwargs):
        super().__init__(**kwargs)
        if initialization_profile not in PROFILES:raise ValueError('explicit supported admission initialization profile required')
        self.initialization_profile=initialization_profile
        if not self.enabled:raise ValueError('disabled profile must use the original unwrapped WorldZFeedback')
        self.calibrations=[];self.admission=None;self._seed_pending=None;self._seed_update_id=None
        self._admission_matrices=None
        self._seed_expected_native=None

    def admit(self,backend,task,**context):
        record=prepare_calibration(backend,task,initialization_profile=self.initialization_profile,**context)
        if self.admission is not None and record['task_sha256']==self.admission['task_sha256']:
            keys=('admission_stamp_s','measured_camera_sha256','measured_link0_sha256','camera_extrinsic_sha256',
                  'eligible','seed_raw_m','seed_clipped_m','previous_native_issue_stamp_s')
            if all(record.get(key)==self.admission.get(key)for key in keys):return self.admission
        self.reset(record['task_id'])  # Task-local I only; global time/seq retained.
        record['calibration_id']=f"{record['task_sha256'][:16]}-{len(self.calibrations):02d}"
        self.admission=record;self.calibrations.append(record);self._seed_pending=record
        self._seed_update_id=None
        self._seed_expected_native=None
        self._admission_matrices=([backend.rigid(context[key]).copy()for key in
            ['reference_optical','measured_optical','link0','extrinsic']]if record['eligible']else None)
        return record

    def feedback_native_command(self,backend,reference_optical,measured_optical,link0,extrinsic,**kwargs):
        if self._seed_update_id is not None:
            # A first input not confirmed in its own synchronous policy tick
            # cannot propagate its provisional seed into another measurement.
            self.rollback_last_update('seed_unconfirmed_before_next_feedback',update_id=self._seed_update_id)
        pending=self._seed_pending
        seeded=False
        if pending is not None:
            # Admission is single-use: late, blocked or invalid first tick must
            # not turn into a later unreviewed seed/retry or integrate a gap.
            self._seed_pending=None
            clock,_,_=self._clock(kwargs.get('stamp_s'),kwargs.get('now_s'),kwargs.get('dt_s',self.nominal_dt_s),
                kwargs.get('sample_seq'),kwargs.get('fresh',True),False)
            same=(kwargs.get('task_id')==pending['task_id']and clock and
                abs(kwargs['stamp_s']-pending['admission_stamp_s'])<=1e-9 and
                abs(kwargs['now_s']-pending['admission_stamp_s'])<=1e-9)
            if self._admission_matrices is not None:
                same=bool(same and all(np.allclose(backend.rigid(value),bound,atol=1e-9,rtol=0)
                    for value,bound in zip([reference_optical,measured_optical,link0,extrinsic],self._admission_matrices)))
            if pending['eligible']and same and kwargs.get('integration_allowed',True):
                self.integral_m=pending['seed_clipped_m'];seeded=True
            else:
                self.integral_m=0.
                if pending['eligible']:pending.update(transaction_status='first_tick_rejected_reset_zero')
        native,check=super().feedback_native_command(backend,reference_optical,measured_optical,link0,extrinsic,**kwargs)
        report=check['world_z_feedback']
        if self.admission is not None:
            report['calibration_id']=self.admission['calibration_id']
            report['calibration_status']=self.admission['transaction_status']
        if seeded:
            valid=report['measurement_fresh_and_clock_valid']and report['native_correction_accepted']and not report['domain_correction_rejected']
            if self._rollback is not None:self._rollback['old_integral']=0.
            if valid:
                # Rollback covers this seed as well as this tick's I increment.
                self._seed_update_id=report['update_id']
                self._seed_expected_native=np.asarray(native,dtype=np.float32).copy()
                report.update(seed_pending_commit=True,seed_task_sha256=pending['task_sha256'],integral_before_m=0.,
                    seed_initialization_m=pending['seed_clipped_m'],integral_updated=bool(self.integral_m!=0.))
                pending.update(transaction_status='prepared_waiting_actual_issue',first_update_id=report['update_id'])
                report['calibration_status']=pending['transaction_status']
            else:
                self.integral_m=0.;report.update(integral_after_m=0.,integral_updated=False)
                pending.update(transaction_status='first_tick_domain_or_clock_rejected_reset_zero')
                report['calibration_status']=pending['transaction_status']
        return native,check

    def rollback_last_update(self,reason,*,update_id=None):
        transaction=self._rollback
        seeded=(transaction is not None and self.admission is not None and
            transaction['update_id']==self.admission.get('first_update_id')and (update_id is None or update_id==transaction['update_id']))
        result=super().rollback_last_update(reason,update_id=update_id)
        if result and seeded:
            self.admission.update(transaction_status='outer_override_rolled_back_reset_zero',rollback_reason=reason)
            transaction['report']['calibration_status']=self.admission['transaction_status']
            transaction['report']['seed_pending_commit']=False
            self._seed_update_id=None
        return result

    def finish_seed_update(self,update_id,*,applied_native=None,overridden=False):
        if self._seed_update_id!=update_id:return False
        if (self._rollback is None or self._rollback['update_id']!=update_id
                or self.last_report is not self._rollback['report']
                or self.last_report.get('update_id')!=update_id
                or not self.last_report.get('seed_pending_commit')
                or self.admission.get('transaction_status')!='prepared_waiting_actual_issue'):
            self.rollback_last_update('seed_confirmation_not_current_first_tick',update_id=update_id)
            return False
        if overridden or applied_native is None:
            self.rollback_last_update('seed_not_confirmed_actual_issue',update_id=update_id);return False
        value=np.asarray(applied_native,dtype=np.float32).astype(float)
        if value.shape!=(7,)or not np.isfinite(value).all():
            self.rollback_last_update('seed_invalid_actual_issue',update_id=update_id);return False
        if not np.array_equal(value,np.asarray(self._seed_expected_native,dtype=float)):
            self.rollback_last_update('seed_actual_native_mismatch',update_id=update_id);return False
        self.admission.update(transaction_status='committed_to_native_input',issued_native_z_m=float(value[2]),
            remaining_native_z_jump_m=float(value[2]-self.admission['previous_actual_native_z_m']))
        if self.last_report is not None:
            self.last_report['seed_pending_commit']=False
            self.last_report['calibration_status']=self.admission['transaction_status']
        self._seed_update_id=None
        return True
