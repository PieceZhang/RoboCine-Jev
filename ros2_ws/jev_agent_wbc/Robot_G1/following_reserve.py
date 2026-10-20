"""Conditional following reserve, appended as a genuine Pink native Limit.

This module neither runs physics nor replaces the native FrameTask, barriers,
limits or solver output. Actual braking/servo dynamics remain unverified. The
assumed deceleration below is NOT implied by the issued acceleration upper cap.

Evidence describes physical writes separately from a same-tick planned write.
Call ``validate_application`` before using the reply; a bound established in the
request is never silently recomputed using a newer reference.

Required evidence schema ``g1_following_evidence/v1``:
``task_id``, full admitted ``task_sha256`` from solver_context, acquisition and
velocity stamps, reduced native15 ``joint_names``, ``measured_q_rad``, measured
``measured_qd_rad_s``, ``measurement_source='measured_physx'``, and a fresh
``committed_reference_snapshot``. The optional ``planned_advance_bound`` is a
separate declaration of a known not-yet-written target. Optional independently
measured pair closing speeds are ``measured_pair_closing={stamp_s, speed_m_s}``,
where speed_m_s maps canonical string native pair indices to nonnegative m/s.

Application gating is BEFORE FIRST USE. After that gate the packet's own native
Delta_q creates a new execution target; the caller must separately authorize
only that exact unprojected target (including declared write quantization),
hold it for substeps and retain Task/TTL checks. Reapplying the old request's
advance box to that packet's own new target would reject legitimate execution.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

LATENCY_S = .06
GUARD_LOOKAHEAD_S = .06
ASSUMED_BRAKING_FRACTION = .5
ISSUED_ACCELERATION_CAP_RAD_S2 = 1.
ISSUED_SPEED_CAP_RAD_S = .5
NATIVE_DT_S = .02
MINIMUM_DISTANCE_M = .0065
MAX_AGE_S = .06
_CLOCK_TOL = 1e-9
_POSITION_TOL = 1e-10


class FollowingReserveRejected(ValueError):
    """Missing/inconsistent evidence; optional profile must hold and retry."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise FollowingReserveRejected(name + '_must_be_finite_number')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FollowingReserveRejected(name + '_must_be_finite_number') from error
    if not np.isfinite(result):
        raise FollowingReserveRejected(name + '_must_be_finite_number')
    return result


def _names(value):
    if not isinstance(value, (tuple, list)) or len(value) != 15:
        raise FollowingReserveRejected('exactly_15_owned_joint_names_required')
    if any(not isinstance(name, str) or not name for name in value) or len(set(value)) != 15:
        raise FollowingReserveRejected('unique_owned_joint_names_required')
    return tuple(value)


def _vector(value, name, size=15):
    try:
        raw = np.asarray(value, dtype=object)
        if raw.shape != (size,) or any(isinstance(x, (bool, np.bool_)) or
                not isinstance(x, (int, float, np.integer, np.floating)) for x in raw):
            raise FollowingReserveRejected(name + '_must_be_numeric_vector_without_booleans')
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError, OverflowError) as error:
        raise FollowingReserveRejected(name + '_must_be_finite_vector') from error
    if array.shape != (size,) or not np.isfinite(array).all():
        raise FollowingReserveRejected(name + '_must_be_finite_vector')
    array = array.copy()
    array.setflags(write=False)
    return array


def _task(task_id, allow_startup=False):
    if task_id is None and allow_startup:
        return None
    if not isinstance(task_id, str) or not task_id:
        raise FollowingReserveRejected('current_nonempty_task_id_required')
    return task_id


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise FollowingReserveRejected(name + '_sha256_required')
    return value


def _token(task_id, task_sha256, names, positions):
    # Repeated physical writes of exactly the same target in the same Task have
    # equal value tokens. commit_seq separately identifies each physical write.
    return _fingerprint(dict(task_id=task_id, task_sha256=task_sha256, joint_names=list(names),
                             positions_rad=np.asarray(positions).tolist()))


def committed_reference_snapshot(*, task_id, joint_names, positions_rad,
                                 commit_seq, commit_stamp_s, snapshot_stamp_s,
                                 task_sha256=None):
    """Snapshot an ALREADY physical-written target ledger, never a planned one.

    The caller must populate its ledger only after the actual physical write.
    This helper validates declarations; it cannot independently observe PhysX.
    Startup Task=None is allowed here but is rejected by an active Task Limit.
    """
    task_id = _task(task_id, allow_startup=True)
    if task_id is None:
        if task_sha256 is not None:
            raise FollowingReserveRejected('startup_reference_cannot_claim_Task_sha')
    else:
        task_sha256 = _sha(task_sha256, 'full_admitted_task')
    names = _names(joint_names)
    positions = _vector(positions_rad, 'committed_positions')
    if isinstance(commit_seq, (bool, np.bool_)) or not isinstance(commit_seq, (int, np.integer)) or commit_seq < 0:
        raise FollowingReserveRejected('physical_commit_seq_required')
    commit_stamp = _number(commit_stamp_s, 'commit_stamp')
    snapshot_stamp = _number(snapshot_stamp_s, 'reference_snapshot_stamp')
    if min(commit_stamp, snapshot_stamp) < 0 or commit_stamp > snapshot_stamp + _CLOCK_TOL:
        raise FollowingReserveRejected('reference_cannot_be_committed_in_future')
    return dict(schema='g1_committed_reference/v1', source='physically_committed',
                task_id=task_id, task_sha256=task_sha256, joint_names=list(names), positions_rad=positions.tolist(),
                commit_seq=int(commit_seq), commit_stamp_s=commit_stamp,
                snapshot_stamp_s=snapshot_stamp, token=_token(task_id, task_sha256, names, positions))


def _reference(value):
    if not isinstance(value, Mapping) or value.get('schema') != 'g1_committed_reference/v1' or value.get('source') != 'physically_committed':
        raise FollowingReserveRejected('physically_committed_reference_required')
    try:
        checked = committed_reference_snapshot(**{key: value[key] for key in
            ('task_id', 'task_sha256', 'joint_names', 'positions_rad', 'commit_seq', 'commit_stamp_s', 'snapshot_stamp_s')})
    except KeyError as error:
        raise FollowingReserveRejected('incomplete_committed_reference') from error
    if value.get('token') != checked['token']:
        raise FollowingReserveRejected('committed_reference_token_mismatch')
    return checked


def planned_advance_bound(reference, planned_positions_rad, *, task_id, request_stamp_s,
                          write_dtype='exact'):
    """Declare one known planned target advance without calling it committed.

    The interval from zero to the known delta covers old/new/partial writes.
    Its per-axis worst closing projection is charged with no opening credit.
    """
    reference = _reference(reference)
    task_id = _task(task_id)
    stamp = _number(request_stamp_s, 'request_stamp')
    if reference['task_id'] != task_id:
        raise FollowingReserveRejected('planned_advance_cross_task')
    if abs(reference['snapshot_stamp_s'] - stamp) > _CLOCK_TOL:
        raise FollowingReserveRejected('planned_advance_requires_same_stamp_reference')
    planned = _vector(planned_positions_rad, 'planned_positions')
    delta = planned - np.asarray(reference['positions_rad'])
    old = np.asarray(reference['positions_rad'])
    roundoff = np.zeros(15)
    if write_dtype == 'float32':
        with np.errstate(over='ignore'):
            old32, new32 = old.astype(np.float32), planned.astype(np.float32)
        if not np.isfinite(old32).all() or not np.isfinite(new32).all() or not np.array_equal(old32.astype(float), old) or not np.array_equal(new32.astype(float), planned):
            raise FollowingReserveRejected('actual_float32_write_values_required')
        roundoff = .5*(np.abs(np.spacing(old32).astype(float))+np.abs(np.spacing(new32).astype(float)))
    elif write_dtype != 'exact':
        raise FollowingReserveRejected('known_planned_write_dtype_required')
    # The native velocity still has its original .5cap. This allowance covers
    # only explicitly declared old/new float32 rounding, not extra velocity.
    if np.any(np.abs(delta) > ISSUED_SPEED_CAP_RAD_S * NATIVE_DT_S + roundoff + 4*np.finfo(float).eps):
        raise FollowingReserveRejected('planned_advance_exceeds_original_issued_speed_cap')
    return dict(schema='g1_planned_reference_advance/v1', source='planned_not_committed',
                task_id=task_id, task_sha256=reference['task_sha256'], request_stamp_s=stamp,
                from_reference_token=reference['token'], delta_rad=delta.tolist(),
                write_dtype=write_dtype, native_float32_roundoff_rad=roundoff.tolist())


def _reserve_rows(distances, jacobian, measured_qd, reference_error, closing_speed_floor=None,
                  planned_delta=None, native_roundoff_rad=None):
    """Pure conditional algebra; J m/rad, Delta_q rad, G*Delta_q <= h metres."""
    J = np.asarray(jacobian, float)
    d = np.asarray(distances, float)
    if J.ndim != 2 or J.shape[1] != 15 or d.shape != (len(J),) or not np.isfinite(J).all() or not np.isfinite(d).all():
        raise FollowingReserveRejected('finite_native_distance_jacobian_rows_required')
    qd = _vector(measured_qd, 'measured_qd')
    lead = _vector(reference_error, 'reference_error')
    c = np.maximum(0., -J @ qd)
    if closing_speed_floor is not None:
        floor = _vector(closing_speed_floor, 'pair_closing_speed', len(J))
        if np.any(floor < 0):
            raise FollowingReserveRejected('nonnegative_pair_closing_speed_required')
        c = np.maximum(c, floor)
    assumed = ASSUMED_BRAKING_FRACTION * np.sum(np.abs(J), axis=1) * ISSUED_ACCELERATION_CAP_RAD_S2
    if np.any((assumed == 0.) & (c > 0.)):
        raise FollowingReserveRejected('zero_projected_braking_with_actual_closing')
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        braking = np.divide(c*c, 2.*assumed, out=np.zeros_like(c), where=assumed > 0.)
    closing_lead = np.maximum(0., -J @ lead)
    advance = np.zeros_like(c)
    if planned_delta is not None:
        delta = _vector(planned_delta, 'planned_delta')
        # Max closing projection over the explicitly declared per-axis box
        # between old and planned references. No opening coordinate credit.
        advance = np.maximum(0., -J * delta).sum(axis=1)
    rounding = np.zeros_like(c)
    if native_roundoff_rad is not None:
        roundoff = _vector(native_roundoff_rad, 'native_roundoff')
        if np.any(roundoff < 0):
            raise FollowingReserveRejected('nonnegative_native_roundoff_required')
        rounding = np.abs(J) @ roundoff
    reserve = (LATENCY_S + GUARD_LOOKAHEAD_S)*c + braking + closing_lead + advance + rounding
    h = d - MINIMUM_DISTANCE_M - reserve
    physical_margin = d - MINIMUM_DISTANCE_M - LATENCY_S*c - braking
    if not np.isfinite(np.r_[assumed, reserve, h, physical_margin]).all():
        raise FollowingReserveRejected('nonfinite_following_reserve')
    return -J.copy(), h, dict(actual_closing_m_s=c, assumed_braking_m_s2=assumed,
        braking_distance_m=braking, closing_reference_lead_m=closing_lead,
        planned_closing_advance_m=advance, native_float32_roundoff_m=rounding, reserve_m=reserve,
        assumed_physical_stopping_margin_m=physical_margin)


class FollowingReserveLimit:
    """Frozen native Limit rows bound to one measured acquisition and reference."""

    @classmethod
    def from_native_barrier(cls, configuration, native_barrier, evidence, *,
                            task_id, task_sha256, reference_token, now_s):
        from pink.barriers import SelfCollisionBarrier
        if not isinstance(native_barrier, SelfCollisionBarrier):
            raise FollowingReserveRejected('genuine_native_SelfCollisionBarrier_required')
        if not isinstance(evidence, Mapping) or evidence.get('schema') != 'g1_following_evidence/v1':
            raise FollowingReserveRejected('following_evidence_schema_required')
        self = cls()
        try:
            self.task_id = _task(task_id)
            self.task_sha256 = _sha(task_sha256, 'full_admitted_task')
            if evidence['task_id'] != self.task_id:
                raise FollowingReserveRejected('measurement_cross_task')
            if evidence['task_sha256'] != self.task_sha256:
                raise FollowingReserveRejected('measurement_cross_full_task')
            self.stamp_s = _number(evidence['acquisition_stamp_s'], 'acquisition_stamp')
            self.names = _names(evidence['joint_names'])
            self.q = _vector(evidence['measured_q_rad'], 'measured_q')
            qd = _vector(evidence['measured_qd_rad_s'], 'measured_qd')
            velocity_stamp = _number(evidence['velocity_stamp_s'], 'velocity_stamp')
            if evidence['measurement_source'] != 'measured_physx' or abs(velocity_stamp - self.stamp_s) > _CLOCK_TOL:
                raise FollowingReserveRejected('same_stamp_actual_velocity_required')
            self.reference = _reference(evidence['committed_reference'])
        except KeyError as error:
            raise FollowingReserveRejected('incomplete_following_evidence') from error
        self._age(now_s, 0.)
        if list(configuration.model.names)[1:] != list(self.names) or configuration.model.nq != 15 or configuration.model.nv != 15:
            raise FollowingReserveRejected('native_owned_joint_order_mismatch')
        if not np.array_equal(np.asarray(configuration.q), self.q):
            raise FollowingReserveRejected('native_configuration_differs_from_measured_acquisition')
        if self.reference['task_id'] != self.task_id or self.reference['task_sha256'] != self.task_sha256 or self.reference['joint_names'] != list(self.names):
            raise FollowingReserveRejected('committed_reference_cross_task_or_joint_order')
        if abs(self.reference['snapshot_stamp_s'] - self.stamp_s) > _CLOCK_TOL:
            raise FollowingReserveRejected('same_stamp_committed_reference_snapshot_required')
        if self.reference['token'] != reference_token:
            raise FollowingReserveRejected('current_committed_reference_token_mismatch')
        self.planned_delta = None
        self.native_roundoff = np.zeros(15)
        self.planned_write_dtype = 'exact'
        plan = evidence.get('planned_advance_bound')
        if plan is not None:
            if not isinstance(plan, Mapping):
                raise FollowingReserveRejected('planned_advance_schema_required')
            try:
                checked = planned_advance_bound(self.reference,
                    np.asarray(self.reference['positions_rad']) + _vector(plan['delta_rad'], 'planned_delta'),
                    task_id=self.task_id, request_stamp_s=self.stamp_s,
                    write_dtype=plan.get('write_dtype'))
            except KeyError as error:
                raise FollowingReserveRejected('incomplete_planned_advance') from error
            # Exact identities exclude a plan borrowed from a different
            # acquisition/Task/reference. The planned target is never committed.
            for key in ('schema', 'source', 'task_id', 'task_sha256', 'from_reference_token'):
                if plan.get(key) != checked[key]:
                    raise FollowingReserveRejected('planned_advance_context_mismatch')
            if abs(_number(plan.get('request_stamp_s'), 'planned_request_stamp') - self.stamp_s) > _CLOCK_TOL:
                raise FollowingReserveRejected('planned_advance_stamp_mismatch')
            if not np.array_equal(_vector(plan.get('native_float32_roundoff_rad'), 'declared_roundoff'), checked['native_float32_roundoff_rad']):
                raise FollowingReserveRejected('planned_advance_roundoff_mismatch')
            self.planned_delta = _vector(checked['delta_rad'], 'planned_delta')
            self.native_roundoff = _vector(checked['native_float32_roundoff_rad'], 'native_roundoff')
            self.planned_write_dtype = checked['write_dtype']
        if abs(float(native_barrier.d_min) - MINIMUM_DISTANCE_M) > 1e-15:
            raise FollowingReserveRejected('original_native_minimum_distance_required')
        distances = np.asarray([r.min_distance for r in configuration.collision_data.distanceResults], float)
        count = len(configuration.collision_model.collisionPairs)
        if distances.shape != (count,) or not np.isfinite(distances).all() or native_barrier.dim != min(20, count) or count <= 0:
            raise FollowingReserveRejected('current_native_closest20_rows_required')
        self.pair_indices = np.argpartition(-distances, -native_barrier.dim)[-native_barrier.dim:]
        h_indices = np.argpartition(-(distances - native_barrier.d_min), -native_barrier.dim)[-native_barrier.dim:]
        if not np.array_equal(self.pair_indices, h_indices):
            raise FollowingReserveRejected('native_barrier_row_pairing_mismatch')
        J = np.asarray(native_barrier.compute_jacobian(configuration), float)
        native_h = np.asarray(native_barrier.compute_barrier(configuration), float)
        if J.shape != (native_barrier.dim, 15) or not np.allclose(native_h, distances[self.pair_indices] - MINIMUM_DISTANCE_M, rtol=0, atol=1e-14):
            raise FollowingReserveRejected('native_barrier_row_distance_mismatch')
        pair_floor = evidence.get('measured_pair_closing')
        floor = None
        if pair_floor is not None:
            if not isinstance(pair_floor, Mapping) or abs(_number(pair_floor.get('stamp_s'), 'pair_closing_stamp') - self.stamp_s) > _CLOCK_TOL:
                raise FollowingReserveRejected('same_stamp_pair_closing_required')
            speeds = pair_floor.get('speed_m_s')
            if not isinstance(speeds, Mapping):
                raise FollowingReserveRejected('pair_closing_speed_mapping_required')
            if any(not isinstance(k, str) or not k.isdigit() or str(int(k)) != k or not 0 <= int(k) < count for k in speeds):
                raise FollowingReserveRejected('pair_closing_index_mismatch')
            floor = np.asarray([_number(speeds.get(str(int(i)), 0.), 'pair_closing_speed') for i in self.pair_indices])
        self.G, self.h, self.metrics = _reserve_rows(distances[self.pair_indices], J, qd,
            np.asarray(self.reference['positions_rad']) - self.q, floor, self.planned_delta, self.native_roundoff)
        self.distances = distances[self.pair_indices].copy()
        for value in (self.G, self.h, self.pair_indices, self.distances):
            value.setflags(write=False)
        self.evidence_sha256 = _fingerprint(json.loads(json.dumps(evidence, allow_nan=False)))
        return self

    def _age(self, now_s, execution_horizon_s):
        now = _number(now_s, 'now_s')
        horizon = _number(execution_horizon_s, 'execution_horizon_s')
        if self.stamp_s < 0 or horizon < 0 or now < self.stamp_s - _CLOCK_TOL or now + horizon > self.stamp_s + MAX_AGE_S + _CLOCK_TOL:
            raise FollowingReserveRejected('acquisition_future_or_expired_for_execution')

    def compute_qp_inequalities(self, q, dt):
        """Pink displacement interface: return additional G*Delta_q <= h."""
        if abs(_number(dt, 'native_dt') - NATIVE_DT_S) > 1e-12:
            raise FollowingReserveRejected('original_native_20ms_interval_required')
        if not np.array_equal(_vector(q, 'native_q'), self.q):
            raise FollowingReserveRejected('native_q_changed_since_acquisition')
        return self.G.copy(), self.h.copy()

    def validate_application(self, *, task_id, task_sha256, current_reference, now_s, execution_horizon_s=NATIVE_DT_S):
        """Check fresh reply against ACTUAL current commit, never rebase rows.

        Only reference advance explicitly bounded in the frozen request may
        change its content token. Any unplanned advance rejects the reply.
        """
        return validate_application_manifest(self.manifest(), task_id=task_id, task_sha256=task_sha256,
            current_reference=current_reference, now_s=now_s, execution_horizon_s=execution_horizon_s)

    def manifest(self):
        rows = []
        for k, pair in enumerate(self.pair_indices):
            row = dict(pair_index=int(pair), distance_m=float(self.distances[k]))
            row.update({name: float(values[k]) for name, values in self.metrics.items()})
            rows.append(row)
        binding = dict(schema='g1_following_application/v1', task_id=self.task_id,
            task_sha256=self.task_sha256, acquisition_stamp_s=self.stamp_s,
            committed_reference=json.loads(json.dumps(self.reference)),
            planned_delta_rad=None if self.planned_delta is None else self.planned_delta.tolist(),
            planned_write_dtype=self.planned_write_dtype,
            native_float32_roundoff_rad=self.native_roundoff.tolist(),
            evidence_sha256=self.evidence_sha256)
        return dict(schema='g1_conditional_following_reserve/v1', task_id=self.task_id,
            task_sha256=self.task_sha256, application_binding=binding,
            application_binding_sha256=_fingerprint(binding),
            acquisition_stamp_s=self.stamp_s, reference_token=self.reference['token'],
            evidence_sha256=self.evidence_sha256, selected_native_rows=len(rows), pairs=rows,
            latency_s=LATENCY_S, guard_lookahead_s=GUARD_LOOKAHEAD_S,
            assumed_braking_fraction=ASSUMED_BRAKING_FRACTION,
            original_issued_acceleration_cap_rad_s2=ISSUED_ACCELERATION_CAP_RAD_S2,
            actual_braking_certified=False,
            any_negative_assumed_physical_margin=any(row['assumed_physical_stopping_margin_m'] < 0 for row in rows),
            no_opening_reference_credit=True, planned_advance_included=self.planned_delta is not None,
            units='J:m/rad; Delta_q/r-q:rad; G*Delta_q,h,reserve:m; measured closing:m/s; assumed deceleration:m/s^2',
            required_integration='Append this Limit and keep all original FrameTasks/native limits/20pairCBF. Validate actual application context before execution. Do not overwrite solver outputs.',
            limitations='Only measured owned upper15 velocities contribute. Locked/lower-body motion, actual following/braking and J-curvature bounds are unverified; no full physical collision certificate. Stopping formula assumes closing does not increase during latency and actual braking at least the hypothesized value afterwards.')


def validate_application_manifest(report, *, task_id, task_sha256, current_reference,
                                  now_s, execution_horizon_s=NATIVE_DT_S):
    """Pure NumPy application gate: safe to call inside Kit without Pink/Pin.

    ``report`` is the frozen worker Limit.manifest(). This gate is before first
    use; it does not authorize arbitrary subsequent target integration. Existing
    worker Task-context/freshness and exact executed-target checks remain.
    """
    if not isinstance(report, Mapping) or report.get('schema') != 'g1_conditional_following_reserve/v1':
        raise FollowingReserveRejected('following_application_report_required')
    binding = report.get('application_binding')
    if not isinstance(binding, Mapping) or binding.get('schema') != 'g1_following_application/v1':
        raise FollowingReserveRejected('following_application_binding_required')
    if report.get('application_binding_sha256') != _fingerprint(binding):
        raise FollowingReserveRejected('following_application_binding_corrupted')
    try:
        admitted_task = _task(task_id)
        task_sha = _sha(task_sha256, 'full_admitted_task')
        if report.get('task_id') != binding['task_id'] or report.get('task_sha256') != binding['task_sha256']:
            raise FollowingReserveRejected('following_report_binding_task_mismatch')
        if admitted_task != binding['task_id'] or task_sha != binding['task_sha256']:
            raise FollowingReserveRejected('application_cross_task')
        stamp = _number(binding['acquisition_stamp_s'], 'acquisition_stamp')
        now = _number(now_s, 'now_s')
        horizon = _number(execution_horizon_s, 'execution_horizon_s')
        if stamp < 0 or horizon < 0 or now < stamp - _CLOCK_TOL or now+horizon > stamp+MAX_AGE_S+_CLOCK_TOL:
            raise FollowingReserveRejected('acquisition_future_or_expired_for_execution')
        reference = _reference(binding['committed_reference'])
        current = _reference(current_reference)
        if reference['task_id'] != admitted_task or reference['task_sha256'] != task_sha or current['task_id'] != admitted_task or current['task_sha256'] != task_sha:
            raise FollowingReserveRejected('application_cross_task')
        if current['joint_names'] != reference['joint_names']:
            raise FollowingReserveRejected('application_joint_order_mismatch')
        if abs(reference['snapshot_stamp_s']-stamp) > _CLOCK_TOL or abs(current['snapshot_stamp_s']-now) > _CLOCK_TOL:
            raise FollowingReserveRejected('fresh_application_reference_snapshot_required')
        if current['commit_seq'] < reference['commit_seq'] or current['commit_stamp_s'] < reference['commit_stamp_s']-_CLOCK_TOL:
            raise FollowingReserveRejected('application_commit_ledger_reversed')
        if current['commit_seq'] == reference['commit_seq'] and (current['token'] != reference['token'] or current['commit_stamp_s'] != reference['commit_stamp_s']):
            raise FollowingReserveRejected('application_commit_event_did_not_advance')
        if current['token'] != reference['token'] and current['commit_stamp_s'] < stamp-_CLOCK_TOL:
            raise FollowingReserveRejected('planned_reference_changed_before_acquisition')
        difference = np.asarray(current['positions_rad'])-np.asarray(reference['positions_rad'])
        planned = binding['planned_delta_rad']
        if planned is None:
            if current['token'] != reference['token']:
                raise FollowingReserveRejected('application_unplanned_reference_token')
        else:
            planned = _vector(planned, 'planned_delta')
            checked_plan = planned_advance_bound(reference, np.asarray(reference['positions_rad'])+planned,
                task_id=admitted_task, request_stamp_s=stamp, write_dtype=binding['planned_write_dtype'])
            if not np.array_equal(_vector(binding['native_float32_roundoff_rad'], 'declared_roundoff'), checked_plan['native_float32_roundoff_rad']):
                raise FollowingReserveRejected('planned_advance_roundoff_mismatch')
            lo, hi = np.minimum(0., planned), np.maximum(0., planned)
            if np.any(difference < lo-_POSITION_TOL) or np.any(difference > hi+_POSITION_TOL):
                raise FollowingReserveRejected('application_reference_outside_declared_advance_bound')
        _sha(binding['evidence_sha256'], 'evidence')
    except KeyError as error:
        raise FollowingReserveRejected('incomplete_following_application_binding') from error
    return dict(status='same_reference_or_within_frozen_planned_bound',
                evidence_sha256=binding['evidence_sha256'], original_reference_token=reference['token'],
                application_reference_token=current['token'], reference_rows_rebased=False,
                actual_braking_certified=False)
