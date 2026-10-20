"""Aggregate backend evidence without treating missing runs as successful ones."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def unanswered_transport_requests(root):
    """Transport failures are unknown trials; a successful retry resolves one."""
    failed, answered = set(), set()
    for path in Path(root).rglob('*jev_api.jsonl'):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            request = json.dumps(row.get('request'), sort_keys=True)
            if row.get('status') == 'transport_unverified_retry':
                failed.add(request)
            elif row.get('response') is not None and not row.get('error'):
                answered.add(request)
    return len(failed-answered)


def summarize_runs(root):
    records = []
    for manifest in sorted(Path(root).rglob('cinegym_run.json')):
        meta = json.loads(manifest.read_text())
        paths = list(manifest.parent.glob('*/summary.json'))
        summary = json.loads(paths[0].read_text()) if len(paths) == 1 else {}
        evidence_source='backend_summary'
        if not summary:
            # Older runners could crash while evaluating a one-sample camera
            # history after an already recorded execution failure. Preserve
            # that known failure; absence of safety measurements stays unknown.
            failures=[]
            for journal in manifest.parent.glob('*/events.jsonl'):
                for line in journal.read_text().splitlines():
                    try: row=json.loads(line)
                    except json.JSONDecodeError: continue
                    if row.get('event')=='run_failed' and row.get('failure'):
                        failures.append(row)
            if failures:
                row=failures[-1]
                summary=dict(completed=False,failure=row['failure'],elapsed_shot_s=row['time_s'],
                    shot_acceptance=dict(status='fail',accepted=False,failed_checks=['execution_error']))
                evidence_source='execution_failure_journal; safety_metrics_unavailable'
        acceptance = summary.get('shot_acceptance') or {}
        network_unanswered = unanswered_transport_requests(manifest.parent)
        interrupted=(manifest.parent/'cinegym_interrupted.json').exists()
        timed_out=(manifest.parent/'cinegym_timeout.json').is_file()
        complete = summary.get('completed') is True
        # SIGINT at the process wall limit can run the backend's finally block
        # and save a partial summary with an empty failure string. Such a prefix
        # is not a conclusive algorithm failure. Keep independently recorded
        # execution/safety failures and complete outcomes conclusive even when
        # shutdown itself subsequently times out.
        known_failure=bool(summary.get('failure') or summary.get('safety_pass') is False or
                           (summary.get('collision_samples') or 0)>0)
        partial_timeout=timed_out and not complete and not known_failure
        unverified = (interrupted or partial_timeout or not summary or summary.get('elapsed_shot_s') is None or
                      summary.get('diagnostic_duration_limit_s') is not None or
                      acceptance.get('status') == 'unverified' or
                      summary.get('physx_reporting_policy', {}).get('network_invalid_trial', False) or network_unanswered > 0)
        accepted = complete and acceptance.get('accepted') is True
        records.append(dict(backend=meta['backend'], family=meta['family'], difficulty=meta['difficulty'],
            seed=meta['seed'], episode_sha256=meta['episode_sha256'], output=str(manifest.parent),
            status='unverified' if unverified else 'pass' if accepted else 'fail', completed=complete,
            safety_pass=summary.get('safety_pass'), min_clearance_m=summary.get('min_truth_clearance_m'),
            collision_samples=summary.get('collision_samples'), physical_contacts=summary.get('contacts'),
            position_rms_m=summary.get('original_position_rms_m'),
            orientation_rms_rad=summary.get('original_orientation_rms_rad'),
            control_loop_wall_ms=summary.get('control_loop_wall_ms'), realtime=summary.get('realtime'),
            failure=summary.get('failure'), elapsed_shot_s=summary.get('elapsed_shot_s'), wall_s=summary.get('wall_s'),
            timed_out=timed_out,interrupted=interrupted,partial_summary_timeout=partial_timeout))
        records[-1].update(mode=meta.get('mode', summary.get('mode')),
                           evidence_source=evidence_source,
                           network_unanswered_requests=network_unanswered,
                           installed_plans=summary.get('installed_plans'),
                           shot_failed_checks=acceptance.get('failed_checks'))
    groups = defaultdict(list)
    for record in records:
        for key in (record['backend']+'/all', record['backend']+'/difficulty/'+record['difficulty'],
                    record['backend']+'/family/'+record['family']):
            groups[key].append(record)
    def aggregate(rows):
        verified = [r for r in rows if r['status'] != 'unverified']
        known_collisions = [r for r in rows if r['completed'] and r['collision_samples'] is not None]
        rms = [r['position_rms_m'] for r in rows if r['completed'] and r['position_rms_m'] is not None]
        safe_completed=sum(r['completed'] and r['safety_pass'] is True for r in verified)
        audited=[r for r in verified if r['collision_samples'] is not None]
        return dict(total=len(rows), verified=len(verified), unverified=len(rows)-len(verified),
            passed=sum(r['status'] == 'pass' for r in rows), failed=sum(r['status'] == 'fail' for r in rows),
            success_rate=None if not verified else sum(r['status'] == 'pass' for r in verified)/len(verified),
            completed=sum(r['completed'] for r in rows),
            verified_safe_completed=safe_completed,
            safe_completion_rate=None if not verified else safe_completed/len(verified),
            collision_rate_all_audited=None if not audited else sum(r['collision_samples']>0 for r in audited)/len(audited),
            collision_rate_completed=None if not known_collisions else sum(r['collision_samples'] > 0 for r in known_collisions)/len(known_collisions),
            mean_position_rms_m=None if not rms else float(np.mean(rms)))
    return dict(version='cinegym_evaluation/v1', runs=records,
                groups={key: aggregate(rows) for key, rows in sorted(groups.items())},
                policy='CPU and PhysX grouped separately; success requires completed execution and original shot acceptance; unverified trials excluded from success/failure denominator; missing metrics stay null')
