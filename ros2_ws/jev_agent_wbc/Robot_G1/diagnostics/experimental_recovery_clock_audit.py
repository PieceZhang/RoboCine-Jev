"""UNTESTED EXPERIMENTAL DRAFT; excluded from all acceptance gates.

Development stopped at the user's scope reduction before execution or tests.
This file is retained as an unfinished prototype, not a verified audit tool.
Offline direct recovery clocks never grant a motion lease or safety pass.

Time starts at the backend's original observed waiting/failure start, even when
the first reported failed result arrives later. The endpoint is completion of
the resumed physical tick, not reply acceptance. A long pause before observation
is exposed as phase jitter rather than silently charged to a different budget.
Frozen sources are hashed as text only; their control semantics need separate
source review. No frozen source is imported or evaluated.
"""
import hashlib
import json
import math


DT = .02
BUDGET = .18
LEASE = .06
TOL = 1e-8
PHASES = ('task_servo', 'native_controller', 'physics',
          'measurement_geometry', 'feedback_log')


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite nonboolean clock required')
    return float(value)


def _equal(a, b):
    return abs(_number(a)-_number(b)) <= TOL


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _context(value):
    if (not isinstance(value, dict) or value.get('schema') != 'g1_solver_task_context/v1'
            or not isinstance(value.get('task_id'), str) or not value['task_id']
            or not _sha(value.get('camera_path_id')) or not _sha(value.get('task_sha256'))):
        raise ValueError('complete original task context required')
    return value


def audit_episode_clock(first_failure_row, resumed_row, all_rows, wall_timings,
                        expected_context):
    """Check direct observation -> resumed whole-execution within 180 ms.

    ``wall_timings`` is an offline evidence envelope::

      {schema: 'g1_recovery_clock_evidence/v1', records: [original timing rows],
       measurement_rows_sha256: canonical_sha256(all_rows),
       timing_rows_sha256: canonical_sha256(records),
       frozen_sources: {backend: {sha256, utf8}, runner: {sha256, utf8}},
       provenance_frozen_sha256: {g1_backend.py: SHA, adaptive_runner.py: SHA}}

    All measurement/timing rows are required, in original order. The caller
    builds the envelope from immutable run files after timing. Missing direct
    clocks permit the caller to try its conservative old bound; any partial or
    invalid direct evidence forbids such fallback. Other recovery, 60 ms motion
    lease, actual hold, physical safety and realtime gates remain independent.
    """
    errors = []
    result = dict(schema='independent_g1_direct_recovery_clock_audit/v1', passed=False,
        errors=errors, duration_wall_s=None, duration_sim_s=None,
        basis='backend_observation_to_resumed_whole_execution_end',
        direct_records_present=False, conservative_fallback_permitted=False,
        original_recovery_budget_s=BUDGET, original_motion_lease_s=LEASE,
        motion_lease_renewed=False, actual_physical_safety_verified=False,
        scope='Direct clock consistency only; phase jitter and physical realtime, '
              'motion leases, holds, tracking, collision and stopping remain separate.')
    def require(condition, reason):
        if not condition:
            raise ValueError(reason)
    try:
        require(isinstance(all_rows, list) and bool(all_rows), 'complete measurement rows required')
        present = any(isinstance(r, dict) and isinstance(r.get('pink_runtime_recovery'), dict)
                      and r['pink_runtime_recovery'].get('recovery_clock') is not None for r in all_rows)
        result['direct_records_present'] = present
        if not present:
            result['basis'] = 'direct_clock_unavailable'
            result['conservative_fallback_permitted'] = True
            errors.append('direct_clock_unavailable')
            return result
        context = _context(expected_context)
        require(isinstance(wall_timings, dict)
                and wall_timings.get('schema') == 'g1_recovery_clock_evidence/v1',
                'complete direct clock evidence envelope required')
        timings = wall_timings['records']
        require(isinstance(timings, list) and len(timings) == len(all_rows),
                'all original measurement and timing rows required')
        require(wall_timings.get('measurement_rows_sha256') == canonical_sha256(all_rows),
                'measurement row hash binding mismatch')
        require(wall_timings.get('timing_rows_sha256') == canonical_sha256(timings),
                'timing row hash binding mismatch')
        provenance = wall_timings['provenance_frozen_sha256']
        for role, filename in (('backend', 'g1_backend.py'), ('runner', 'adaptive_runner.py')):
            source = wall_timings['frozen_sources'][role]
            require(isinstance(source, dict) and isinstance(source.get('utf8'), str)
                    and _sha(source.get('sha256')), 'frozen source bytes and SHA required')
            require(hashlib.sha256(source['utf8'].encode()).hexdigest() == source['sha256']
                    == provenance[filename], 'frozen source/provenance hash mismatch')
        wall_bounds = []
        for i, (row, timing) in enumerate(zip(all_rows, timings)):
            require(isinstance(row, dict) and isinstance(timing, dict), 'malformed original row')
            require(_equal(row['stamp_s'], (i+1)*DT), 'original measurement clock/order changed')
            require(timing.get('schema') == 'g1_control_wall_timing/v1'
                    and type(timing.get('step')) is int and timing['step'] == i
                    and _equal(timing['stamp_s'], row['stamp_s']), 'timing row identity mismatch')
            start = _number(timing['tick_start_wall_s'])
            end = _number(timing['measurement_log_end_wall_s'])
            require(0 <= start <= end and _equal(timing['tick_work_wall_s'], end-start),
                    'invalid physical tick wall interval')
            require(not wall_bounds or start >= wall_bounds[-1][1]-TOL,
                    'physical tick wall time regressed or overlapped')
            phase = timing['phase_wall_ms']
            require(isinstance(phase, dict) and set(phase) == set(PHASES)
                    and all(_number(phase[k]) >= 0 for k in PHASES)
                    and _equal(sum(phase.values())/1000., end-start), 'phase timing does not cover original tick')
            require(_number(timing['start_lateness_wall_s']) >= 0
                    and _number(timing['end_lateness_wall_s']) >= 0
                    and timing.get('own_timing_record_write_excluded') is True,
                    'complete explicit wall timing diagnostics required')
            wall_bounds.append((start, end))
        def locate(row):
            matches = [i for i, r in enumerate(all_rows) if _equal(r['stamp_s'], row['stamp_s'])]
            require(len(matches) == 1 and canonical_sha256(all_rows[matches[0]]) == canonical_sha256(row),
                    'episode endpoint is not an original measured row')
            return matches[0]
        failure_i, resumed_i = locate(first_failure_row), locate(resumed_row)
        require(failure_i < resumed_i, 'resumption must follow observed failure')
        failure_packet = first_failure_row['pink_runtime_recovery']
        require(first_failure_row.get('controller_status') == 'stale_or_failed_result_dropped'
                and failure_packet.get('failure_kind') in
                    ('solver_failure', 'transport_failure', 'stale_or_mismatched_transport'),
                'first failure is not an observed module-result drop')
        first = failure_packet['recovery_clock']
        require(isinstance(first, dict), 'first failure direct clock missing')
        observed = _number(first['recovery_started_wall_s'])
        observed_stamp = _number(first['recovery_started_stamp_s'])
        starts = [i for i, row in enumerate(all_rows)
                  if _equal(row['stamp_s']-DT, observed_stamp)]
        require(len(starts) == 1, 'original observed-start measurement missing')
        start_i = starts[0]
        require(start_i <= failure_i and start_i > 0, 'start must precede failure and have causal prior execution')
        def clock_at(i, episode):
            row = all_rows[i]
            require(row.get('current_solver_task_context') == context
                    and isinstance(row.get('task_feedback'), dict)
                    and row['task_feedback'].get('task_id') == context['task_id'],
                    'direct clock row is not bound to original task')
            clock = row['pink_runtime_recovery']['recovery_clock']
            require(isinstance(clock, dict) and clock.get('schema') == 'g1_module_recovery_clock/v1',
                    'complete backend direct clock schema required')
            poll = _number(clock['poll_wall_s']);sample = _number(clock['sample_wall_s'])
            low, high = wall_bounds[i]
            require(low-TOL <= poll <= sample+TOL <= high+2*TOL
                    and _equal(clock['poll_stamp_s'], row['stamp_s']-DT),
                    'poll/sample is outside original physical tick')
            require(_equal(clock['budget_s'], BUDGET) and _equal(clock['original_motion_lease_s'], LEASE),
                    'original recovery or motion-lease budget changed')
            previous_clock = all_rows[i-1]['pink_runtime_recovery']['recovery_clock'] if i else None
            if previous_clock is not None:
                require(_equal(clock['poll_interval_wall_s'], poll-_number(previous_clock['poll_wall_s']))
                        and _number(clock['poll_interval_wall_s']) >= 0, 'poll interval lost elapsed CPU wall time')
            if episode:
                require(clock.get('recovery_started_context') == context
                        and _equal(clock['fault_detected_wall_s'], observed)
                        and _equal(clock['recovery_started_wall_s'], observed)
                        and _equal(clock['recovery_started_stamp_s'], observed_stamp),
                        'episode start or task context reset')
                require(_equal(clock['elapsed_wall_s'], sample-observed)
                        and sample >= observed-TOL, 'reported recovery elapsed time is inconsistent')
            return clock, poll, sample
        for i in range(start_i, resumed_i+1):
            clock, poll, sample = clock_at(i, True)
            if i == start_i:
                require(wall_bounds[i][0]-TOL <= observed <= sample+TOL
                        and poll-TOL <= observed, 'start is not an actual observation in its logged tick')
            if i < resumed_i:
                require(all_rows[i].get('controller_status') != 'fresh_command'
                        and clock.get('event') in ('waiting', 'failure_detected'),
                        'episode crossed a reset, new task or already resumed execution')
            else:
                require(clock.get('event') == 'resumed'
                        and all_rows[i].get('controller_status') == 'fresh_command',
                        'endpoint is not actual fresh resumption')
        require(first.get('event') == 'failure_detected', 'failure clock event is inconsistent')
        previous_fresh = [i for i in range(start_i)
                          if all_rows[i].get('controller_status') == 'fresh_command']
        require(bool(previous_fresh), 'same-task causal pre-observation execution missing')
        prior_i = previous_fresh[-1]
        prior_clock, _, _ = clock_at(prior_i, False)
        prior = all_rows[prior_i]['pink_runtime_recovery']
        require(prior.get('task_context') == context and prior.get('task_context_status') == 'same_task',
                'prior fresh execution was for another task')
        prior_end = _number(prior['command_sent_wall_s'])+_number(
            all_rows[prior_i]['upper_execution_lease']['execution_end_wall_age_s'])
        require(wall_bounds[prior_i][0]-TOL <= prior_end <= wall_bounds[prior_i][1]+TOL
                and prior_end <= observed+TOL, 'observed start precedes actual prior fresh execution')
        packet = resumed_row['pink_runtime_recovery'];lease = resumed_row['upper_execution_lease']
        require(packet.get('task_context') == context and packet.get('task_context_status') == 'same_task',
                'resumed motion context mismatch')
        sent = _number(packet['command_sent_wall_s']);accepted = _number(packet['accepted_wall_age_s'])
        end_age = _number(lease['execution_end_wall_age_s'])
        end_wall = sent+end_age
        resumed_clock = packet['recovery_clock']
        require(observed-TOL <= sent <= _number(resumed_clock['sample_wall_s'])+TOL
                and 0 <= accepted <= end_age and _equal(sent+accepted, resumed_clock['sample_wall_s']),
                'resumed request send/acceptance clock is inconsistent')
        require(wall_bounds[resumed_i][0]-TOL <= end_wall <= wall_bounds[resumed_i][1]+TOL
                and end_wall >= _number(resumed_clock['sample_wall_s'])-TOL,
                'resumed whole-execution end is outside original physical tick')
        acquisition = _number(packet['acquisition_stamp_s'])
        require(acquisition >= _number(first['poll_stamp_s'])-TOL
                and _equal(acquisition+_number(lease['execution_end_sim_age_s']), resumed_row['stamp_s']),
                'resumed whole-execution simulation clock is inconsistent')
        duration_wall = end_wall-observed
        duration_sim = _number(resumed_row['stamp_s'])-observed_stamp
        result.update(duration_wall_s=duration_wall, duration_sim_s=duration_sim,
            observation_start_wall_s=observed, observation_start_stamp_s=observed_stamp,
            resumed_whole_execution_end_wall_s=end_wall,
            phase_jitter=dict(prior_fresh_end_to_observation_wall_s=observed-prior_end,
                maximum_tick_work_wall_s=max(t['tick_work_wall_s'] for t in timings),
                maximum_start_lateness_wall_s=max(t['start_lateness_wall_s'] for t in timings),
                maximum_end_lateness_wall_s=max(t['end_lateness_wall_s'] for t in timings),
                maximum_intertick_wall_gap_s=max([0.]+[wall_bounds[i][0]-wall_bounds[i-1][1]
                                                        for i in range(1,len(wall_bounds))])),
            source_semantics_executed_or_authenticated=False)
        require(0 <= duration_wall <= BUDGET+TOL, 'observed recovery exceeds original 180ms wall budget')
        require(0 <= duration_sim <= BUDGET+TOL, 'observed recovery exceeds original 180ms simulation budget')
        result['passed'] = True
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as exc:
        errors.append(str(exc))
    return result
