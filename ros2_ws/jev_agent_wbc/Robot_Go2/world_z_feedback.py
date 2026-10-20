"""Private bounded world-Z integral beside the unchanged native camera P term.

Only the seven-dimensional native command is adapted. The original WORLD Task,
measured camera error, eighteen learned actions and final domain/safety checks
remain the caller's responsibility. This helper does not certify convergence.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation


class WorldZFeedback:
    """Synchronous prepare/check/commit wrapper with an optional outer rollback.

    Call ``reset(task_id)`` when a Task is accepted. A fresh call with a new ID
    also resets the integral. Measurement stamps and optional sample sequences
    stay globally monotone across Tasks; the absolute clock is never retimed.
    ``integration_allowed=False`` freezes I without declaring fresh data stale.
    Missing/bad-time data freezes I and returns the uncorrected reference; the
    first fresh recovery sample reanchors time without integrating the gap.

    ``enabled=False`` is the original P-only calculation, including rejection
    fallback, and does not require the additional integration clock arguments.
    """
    POSITION_GAIN = .35
    ORIENTATION_GAIN = .25
    POSITION_CAP_M = .025
    ORIENTATION_CAP_RAD = .05
    INTEGRAL_CAP_M = .025

    def __init__(self, *, enabled=False, ki_per_s=.2, nominal_dt_s=.02,
                 max_dt_s=.06, max_measurement_age_s=.06):
        if type(enabled) is not bool:
            raise ValueError('enabled must be boolean')
        if (not all(math.isfinite(x) and x > 0 for x in
                    (ki_per_s, nominal_dt_s, max_dt_s, max_measurement_age_s))
                or nominal_dt_s > max_dt_s):
            raise ValueError('finite positive gain and ordered integration times required')
        self.enabled = enabled
        self.ki_per_s = float(ki_per_s)
        self.nominal_dt_s = float(nominal_dt_s)
        self.max_dt_s = float(max_dt_s)
        self.max_measurement_age_s = float(max_measurement_age_s)
        self.task_id = None
        self.integral_m = 0.
        self._last_stamp_s = None
        self._last_sample_seq = None
        self._first_in_task = True
        self._reanchor = False
        self._update_id = 0
        self._rollback = None
        self.last_report = None

    def reset(self, task_id):
        """Reset Task-local memory without resetting the absolute-clock guard."""
        if not isinstance(task_id, str) or not task_id:
            raise ValueError('nonempty accepted Task ID required')
        self.task_id = task_id
        self.integral_m = 0.
        self._first_in_task = True
        self._reanchor = False
        self._rollback = None

    def _clock(self, stamp_s, now_s, dt_s, sample_seq, fresh, new_task):
        if type(fresh) is not bool:
            raise ValueError('fresh must be boolean')
        if not fresh:
            return False, 0., 'missing_fresh_measurement'
        if (stamp_s is None or now_s is None or dt_s is None
                or not all(math.isfinite(x) for x in (stamp_s, now_s, dt_s))
                or min(stamp_s, now_s) < 0 or dt_s <= 0 or dt_s > self.max_dt_s):
            return False, 0., 'invalid_measurement_time_or_dt'
        age = now_s-stamp_s
        if age < -1e-9 or age > self.max_measurement_age_s:
            return False, 0., 'future_or_stale_measurement'
        if sample_seq is not None:
            if type(sample_seq) is not int or sample_seq < 0:
                return False, 0., 'invalid_sample_sequence'
            if self._last_sample_seq is not None and sample_seq <= self._last_sample_seq:
                return False, 0., 'nonmonotone_sample_sequence'
        if self._last_stamp_s is not None and stamp_s <= self._last_stamp_s+1e-12:
            return False, 0., 'nonmonotone_measurement_stamp'
        if self._reanchor and not new_task:
            return True, 0., 'fresh_recovery_reanchored_without_gap_integration'
        if not new_task and not self._first_in_task and self._last_stamp_s is not None:
            elapsed = stamp_s-self._last_stamp_s
            if elapsed > self.max_dt_s or not math.isclose(elapsed, dt_s, rel_tol=1e-6, abs_tol=1e-9):
                return False, 0., 'measurement_gap_or_dt_mismatch'
        return True, float(dt_s), 'fresh_measurement'

    @staticmethod
    def _check(backend, domain, native):
        # Keep the real pinned checker last; never clamp a rejected command.
        return dict(domain.check(native) if domain is not None else backend.check_upstream_domain(native))

    @staticmethod
    def _report_time(value):
        # Invalid/stale input must remain loggable with allow_nan=False; a
        # rejected NaN timestamp cannot itself break the caller's recovery log.
        return float(value) if value is not None and math.isfinite(value) else None

    def feedback_native_command(self, backend, reference_optical, measured_optical,
                                link0, extrinsic, *, task_id=None, stamp_s=None,
                                now_s=None, dt_s=None, fresh=True,
                                integration_allowed=True, sample_seq=None, domain=None):
        """Return ``(native7, actual_domain_check)`` like the original P wrapper.

        The report under ``check['world_z_feedback']`` contains ``update_id``.
        If a later outer guard/recovery/parking mode overrides this command,
        call ``rollback_last_update(reason, update_id=...)`` in the same tick.
        A stale token cannot roll back a newer update.
        """
        if type(integration_allowed) is not bool:
            raise ValueError('integration_allowed must be boolean')
        reference = backend.rigid(reference_optical)
        # A missing sample need not contain a camera matrix. Disabled operation
        # deliberately follows the exact legacy requirements/calculation.
        measured = backend.rigid(measured_optical) if (not self.enabled or fresh) else None
        self._update_id += 1
        self._rollback = None
        new_task = self.enabled and task_id != self.task_id
        if self.enabled and (not isinstance(task_id, str) or not task_id):
            raise ValueError('nonempty accepted Task ID required for integral feedback')
        dt_s = self.nominal_dt_s if dt_s is None else dt_s
        valid, integration_dt, reason = (True, 0., 'disabled_legacy_p_only')
        if self.enabled:
            valid, integration_dt, reason = self._clock(
                stamp_s, now_s, dt_s, sample_seq, fresh, new_task)
            if valid and new_task:
                self.reset(task_id)
        old_integral = self.integral_m
        candidate = old_integral
        cap_frozen = bound_saturated = False
        corrected = reference.copy()
        correction = np.zeros(3)
        rotation = np.zeros(3)
        if valid:
            error = reference[:3, 3]-measured[:3, 3]
            if self.enabled and integration_allowed:
                raw_integral = old_integral+self.ki_per_s*float(error[2])*integration_dt
                candidate = float(np.clip(raw_integral, -self.INTEGRAL_CAP_M, self.INTEGRAL_CAP_M))
                bound_saturated = candidate != raw_integral
                raw = self.POSITION_GAIN*error+np.array([0., 0., candidate])
                projected = raw*min(1., self.POSITION_CAP_M/max(np.linalg.norm(raw), 1e-12))
                cap_frozen = bool((candidate-old_integral)*(raw[2]-projected[2]) > 1e-15)
                if cap_frozen:
                    candidate = old_integral
            # Preserve the legacy arithmetic when disabled. Enabled I is a
            # world-Z addition BEFORE the same full3D25mm norm projection.
            correction = self.POSITION_GAIN*error
            if self.enabled:
                correction = correction+np.array([0., 0., candidate])
            correction = correction*min(1., self.POSITION_CAP_M/max(np.linalg.norm(correction), 1e-12))
            corrected[:3, 3] += correction
            rotation = self.ORIENTATION_GAIN*Rotation.from_matrix(
                reference[:3, :3]@measured[:3, :3].T).as_rotvec()
            rotation = rotation*min(1., self.ORIENTATION_CAP_RAD/max(np.linalg.norm(rotation), 1e-12))
            corrected[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()@reference[:3, :3]
        native = backend.camera_to_mixed(corrected, link0, extrinsic)
        check = self._check(backend, domain, native)
        rejected = check.get('status') != 'inside_outer_domain_unverified'
        if rejected:
            proposed = native
            native = backend.camera_to_mixed(reference, link0, extrinsic)
            actual = self._check(backend, domain, native)
            actual['rejected_feedback_correction'] = dict(native_command=proposed.tolist(), domain_check=check)
            check = actual
            candidate = old_integral
        if self.enabled:
            if valid:
                self._last_stamp_s = float(stamp_s)
                if sample_seq is not None:
                    self._last_sample_seq = sample_seq
                self._first_in_task = False
                self._reanchor = False
                self.integral_m = candidate
            else:
                self._reanchor = True
        report = dict(enabled=self.enabled, task_id=self.task_id, update_id=self._update_id,
            measurement_stamp_s=self._report_time(stamp_s), now_s=self._report_time(now_s),
            sample_seq=sample_seq if type(sample_seq) is int else None,
            measurement_fresh_and_clock_valid=valid, clock_status=reason,
            integration_allowed=integration_allowed, integration_dt_s=integration_dt,
            ki_per_s=self.ki_per_s, integral_before_m=old_integral, integral_after_m=self.integral_m,
            integral_updated=bool(self.enabled and candidate != old_integral),
            cap_antiwindup_frozen=cap_frozen, integral_bound_saturated=bound_saturated,
            proposed_position_correction_m=correction.tolist(),
            proposed_position_correction_norm_m=float(np.linalg.norm(correction)),
            proposed_orientation_correction_rad=rotation.tolist(),
            total_position_cap_m=self.POSITION_CAP_M, orientation_cap_rad=self.ORIENTATION_CAP_RAD,
            domain_correction_rejected=rejected,
            native_correction_accepted=bool(valid and not rejected),
            externally_rolled_back=False,
            scope='private native7 input feedback only; original WORLD Task and measured errors unchanged')
        self.last_report = report
        check['world_z_feedback'] = report
        if self.enabled and valid:
            self._rollback = dict(update_id=self._update_id, task_id=self.task_id,
                                  old_integral=old_integral, report=report)
        return native, check

    def rollback_last_update(self, reason, *, update_id=None):
        """Undo this tick's I if a later real guard overrides its native command.

        Observation time/sequence remain consumed: a rejected interval can never
        be integrated again. Repeated rollback and stale tokens are harmless.
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError('explicit outer override reason required')
        transaction = self._rollback
        if (transaction is None or (update_id is not None and update_id != transaction['update_id'])
                or transaction['task_id'] != self.task_id):
            return False
        self.integral_m = transaction['old_integral']
        transaction['report'].update(integral_after_m=self.integral_m, integral_updated=False,
            externally_rolled_back=True, external_rollback_reason=reason,
            native_correction_accepted=False, externally_applied_correction_unknown=True)
        self._rollback = None
        return True
