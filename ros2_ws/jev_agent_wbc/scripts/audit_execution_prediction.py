#!/usr/bin/env python3
"""Compare conditional execution forecasts with raw measured PhysX camera poses.

This audit never modifies acceptance, retimes a reference, or counts an unknown
forecast as a successful prediction. Coverage below is an empirical sample
fraction, not a calibrated whole-trajectory probability.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_agent_wbc.execution_prediction import failure_evidence


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    return None if not len(values) else dict(zip(('p50','p95','max'),
        map(float,np.quantile(values,[.5,.95,1.]))))


def audit(case):
    summary=read(case/'summary.json')
    records=[json.loads(line) for line in (case/'execution.jsonl').read_text().splitlines()]
    initial_scene=read(case/'initial_observed_scene.json') if (case/'initial_observed_scene.json').exists() else {}
    initial_motion={r['id']:np.array(r['velocity']) for r in initial_scene.get('tracks',[])}
    phases=[]
    physical=summary.get('physical_phase_acceptance',[])
    if not physical:
        physical=[dict(index=0,start_s=0.,acceptance=summary.get('shot_acceptance',{}))]
    for phase in physical:
        index=phase['index'];start=phase.get('start_s',0.)
        folder=case/'session'/f'phase_{index+1:02d}'
        if not folder.exists():folder=case/'session'
        prediction_file=folder/'startup_execution_prediction.json'
        acceptance=phase.get('acceptance',{})
        value=dict(index=index,start_s=start,accepted=bool(acceptance.get('accepted')),
            failed_checks=acceptance.get('failed_checks',[]),prediction_available=prediction_file.exists())
        if not prediction_file.exists():
            phases.append(value);continue
        forecast=read(prediction_file);trace=forecast.get('trace',[])
        if len(trace)<2:
            value['comparison_status']='incomplete_forecast';phases.append(value);continue
        knots=np.array([r['t'] for r in trace])
        rows=[r for r in records if knots[0]-1e-8<=r['t']-start<=knots[-1]+1e-8]
        if not rows:
            value['comparison_status']='no_matching_physical_samples';phases.append(value);continue
        times=np.clip([r['t']-start for r in rows],knots[0],knots[-1])
        positions=np.array([r['position'] for r in trace])
        predicted=np.column_stack([np.interp(times,knots,positions[:,j]) for j in range(3)])
        errors=np.linalg.norm(np.array([r['camera_position'] for r in rows])-predicted,axis=1)
        rotations=Slerp(knots,Rotation.from_quat([r['quaternion_xyzw'] for r in trace]))(times)
        actual_rotations=Rotation.from_quat([r['camera_quaternion_xyzw'] for r in rows])
        angles=(actual_rotations*rotations.inv()).magnitude()
        envelope=np.interp(times,knots,[r['tube_radius_m'] for r in trace])
        supported=times<=forecast.get('safety_supported_until_s',knots[0])+1e-8
        coverage=errors<=envelope
        failures=[failure_evidence('executed_'+check,source='physical_shot_acceptance',measured=True)
                  for check in value['failed_checks']]
        waits=[r['t']-start for r in rows if r['info'].get('waiting_for_controller')]
        if waits:failures.append(failure_evidence('command_age_expired',waits[0],source='physical_watchdog',measured=True))
        breaches=np.flatnonzero(~coverage)
        if len(breaches):failures.append(failure_evidence('prediction_position_envelope_exceeded',
            float(times[breaches[0]]),source='measured_camera_comparison',measured=True))
        changed=[r['t']-start for r in rows if any(track['id'] not in initial_motion or
            np.linalg.norm(np.array(track['velocity'])-initial_motion[track['id']])>.08
            for track in r.get('runtime_tracks',[]))]
        before_motion_change=supported & (times < min(changed) if changed else np.ones(len(times),dtype=bool))
        value.update(comparison_status='same_physical_clock_no_retiming',
            predicted_curve_quality=forecast.get('curve_quality'),
            actual_curve_quality=acceptance.get('checks',{}).get('natural_curve',{}).get('measurements'),
            position_error_m=distribution(errors),orientation_error_rad=distribution(angles),
            empirical_position_envelope_sample_coverage=float(np.mean(coverage)),
            supported_prefix_position_envelope_sample_coverage=(float(np.mean(coverage[supported])) if np.any(supported) else None),
            supported_prefix_samples=int(supported.sum()),samples=len(rows),
            pre_motion_change_samples=int(before_motion_change.sum()),
            pre_motion_change_position_envelope_sample_coverage=(float(np.mean(coverage[before_motion_change]))
                if np.any(before_motion_change) else None),
            safety_supported_until_s=forecast.get('safety_supported_until_s'),
            initial_position_error_m=float(errors[0]),
            initial_position_inside_envelope=bool(coverage[0]),
            first_observed_motion_domain_change_s=min(changed) if changed else None,
            initial_condition_scope='Later combo phases were forecast at planned stopped arrival; compare this mismatch before interpreting coverage.',
            command_wait_samples=len(waits),failures=failures,
            confidence_scope='empirical conditional position coverage only; no collision or whole-trajectory probability')
        phases.append(value)
    timing=read(case/'realtime_timing.json') if (case/'realtime_timing.json').exists() else {}
    return dict(case=summary.get('case',case.name),completed=summary.get('completed'),
        safety_pass=summary.get('safety_pass'),accepted=summary.get('shot_acceptance',{}).get('accepted'),
        failure=summary.get('failure'),official_jev_calls=summary.get('jev_calls'),
        physics_hz=summary.get('physics_hz'),pace=summary.get('pace'),
        timing={key:timing.get(key) for key in ('timing_pass','sim_wall_ratio','max_tick_wall_s','ticks_over_20ms')},
        command_packets_rejected=summary.get('rejected_command_packets'),
        min_truth_clearance_m=summary.get('min_truth_clearance_m'),phases=phases)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cases=[audit(path.parent) for path in sorted(args.run.glob('*/summary.json'))
           if (path.parent/'execution.jsonl').exists()]
    result=dict(run=str(args.run.resolve()),cases=cases,
        totals=dict(cases=len(cases),completed=sum(bool(c['completed']) for c in cases),
            safe=sum(bool(c['safety_pass']) for c in cases),accepted=sum(bool(c['accepted']) for c in cases)),
        protocol='Existing acceptance unchanged; every raw main-shot sample retained; no reference retiming.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result['totals']))


if __name__=='__main__':main()
