"""Evaluate every actual PhysX sample, including controller waiting intervals."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .continuous_combo import action_task
from .reference import OriginalReference
from .shot_acceptance import evaluate_shot


def assess_session(summary,rows,task,subject,intrinsics,output):
    summary=dict(summary)
    evidence=next((r['info'].get('session_result') for r in reversed(rows) if r['info'].get('session_done')),None)
    if evidence is None and (Path(output)/'session/summary.json').exists():
        evidence=json.loads((Path(output)/'session/summary.json').read_text())
    completed=bool(evidence and evidence['completed'])
    error=(evidence or {}).get('failure') or ('' if completed else 'physical_session_incomplete')
    if summary.get('actuator_fault'):error=summary['failure'] or 'physical_actuator_fault'
    # The existing PhysX audit has already evaluated contacts and hard limits;
    # controller-side CPU certificates cannot override physical measurements.
    safe=bool(summary['safety_pass'])
    phases=[]
    # Only the free-connector driver emits separately started phase sessions.
    # A CineGym contiguous multi-motion shot uses one immutable global clock
    # and must retain evaluate_shot's complete ordered-phase/continuity checks.
    from .combo import gaps
    is_combo=bool(gaps(task))
    definitions=task['shot_phase'] if is_combo else [None]
    for index,phase in enumerate(definitions):
        ref=OriginalReference(action_task(task,phase) if phase is not None else task,subject['position'])
        markers=[r for r in rows if r['info'].get('session_stage')=='main_action' and r['info'].get('phase_index')==index]
        if not markers:
            phases.append(dict(index=index,accepted=False,reason='phase_not_executed'));continue
        start=markers[0]['info']['phase_start_s']
        segment=[r for r in rows if start-1e-8<=r['t']<=start+ref.duration+.4+1e-8]
        times=[r['t']-start for r in segment]
        controller_phase=next((p for p in (evidence or {}).get('phases',[]) if p.get('index')==index),None) if is_combo else None
        phase_completed=(bool(controller_phase and controller_phase['completed']) if is_combo else completed)
        acceptance=evaluate_shot(ref,subject,times,[r['camera_position'] for r in segment],
            [Rotation.from_quat(r['camera_quaternion_xyzw']).as_matrix() for r in segment],
            [r['visual'] for r in segment],times,safety_pass=safe,
            execution_ok=phase_completed,intrinsics=intrinsics,motion_end_s=ref.duration,
            safety_source='every measured PhysX sample, contacts and physical actuator audit')
        phases.append(dict(index=index,start_s=start,samples=len(segment),acceptance=acceptance,accepted=acceptance['accepted']))
    accepted=bool(completed and not error and safe and all(p['accepted'] for p in phases))
    summary.update(session_controller=evidence,physical_phase_acceptance=phases,
        source_task_duration_s=task['reference_camera_trajectory']['samples'][-1]['t'],
        original_duration_s=task['reference_camera_trajectory']['samples'][-1]['t'],
        completed=bool(completed and not error),failure=error,
        within_budget_success=accepted,intent_success=accepted,
        shot_acceptance=dict(accepted=accepted,status='pass' if accepted else 'fail',
            failed_checks=[f'phase_{p["index"]+1}' for p in phases if not p['accepted']]+([error] if error else [])+([] if safe else ['safety']),
            scope='measured physical main shots; free connectors have safety gates, no cinematic deadline'),
        camera_output_source='execution')
    return summary
