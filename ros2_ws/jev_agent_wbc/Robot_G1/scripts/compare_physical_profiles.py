#!/usr/bin/env python3
"""Read-only measured profile comparison and fixed-measurement PI replay."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='g1_measured_profile_comparison/v1',physical_execution=False,
        scope='Original full5s moving intervals and last complete0.5s26-row stop windows; no resimulation or new feasibility certificate.',
        runs={},evidence_sha256={})
    for directory in args.runs:
        rows=[json.loads(s) for s in (directory/'measurements.jsonl').open()]
        summary=json.loads((directory/'summary.json').read_text())
        names=rows[0]['joint_names'];upper=[n for n in names if any(x in n for x in ('shoulder','elbow','wrist','waist'))]
        config=json.loads((directory/'velocity_servo_config.json').read_text())
        spec=importlib.util.spec_from_file_location('frozen_measured_servo',directory/'sources/velocity_servo.py')
        servo=importlib.util.module_from_spec(spec);spec.loader.exec_module(servo)
        output=[]
        for case in summary['cases']:
            name=case['case'];block=[r for r in rows if r['case']==name]
            tail=[r for r in block if r['stamp_s']>=block[-1]['stamp_s']-.5-1e-9]
            moving=[i for i,r in enumerate(rows) if r['case']==name and r['phase']=='move']
            upper_stats=[]
            for n in upper:
                k=names.index(n);native=np.array([r['pink_upper_velocity'].get(n,0.) for r in tail])
                actual=np.array([r['measured_joint_velocity'][k] for r in tail])
                q=np.array([r['joint_positions'][k] for r in tail]);ref=np.array([r['applied_joint_position_target'][k] for r in tail])
                upper_stats.append(dict(joint=n,actual_qspan_rad=float(np.ptp(q)),
                    actual_mean_qdot_rad_s=float(actual.mean()),native_mean_qdot_rad_s=float(native.mean()),
                    native_abs_mean_qdot_rad_s=float(np.abs(native).mean()),
                    mean_reference_lead_rad=float(np.mean(ref-q)),maximum_absolute_reference_lead_rad=float(np.max(np.abs(ref-q)))))
            replay=None
            if name in ('yaw_left','yaw_right'):
                controllers={str(ki):servo.VelocityServo(limits=config['limits'],**config['parameters']) for ki in (1.,2.)}
                traces={key:[] for key in controllers}
                for i in moving:
                    row=rows[i];measured=rows[i-1]['measured_body_velocity']
                    for key,c in controllers.items():
                        c.kp=np.array([1.,1.,.8]);c.ki=np.array([1.2,1.2,float(key)])
                        c.feedforward_gain=np.ones(3);c.negative_feedforward_gain=np.ones(3)
                        proposed=c.update(row['requested_body_velocity'],measured,.02)
                        if row.get('policy_velocity_guard') is not None:
                            proposed=np.clip(0.,c.command-c.slew*.02,c.command+c.slew*.02)
                            c.applied(proposed,transient_hold=True)
                        traces[key].append(float(proposed[2]))
                replay=dict(scope='Same recorded preceding measurements/guard events; plant response would change. No predicted speed or camera result.',
                    native_yaw_inputs={key:dict(mean=float(np.mean(v)),last_second_mean=float(np.mean(v[-50:])),
                        peak=float(np.max(np.abs(v))),terminal_integral=float(controllers[key].integral[2])) for key,v in traces.items()})
            output.append(dict(case=name,original_result=case,
                tail_window_s=[tail[0]['stamp_s'],tail[-1]['stamp_s']],tail_samples=len(tail),
                tail_PARK_count=int(sum(bool(r['native_standing_mode']) for r in tail)),
                tail_native_input_min=np.min([r['applied_policy_velocity'] for r in tail],axis=0).tolist(),
                tail_native_input_max=np.max([r['applied_policy_velocity'] for r in tail],axis=0).tolist(),
                tail_angular_world_mean=np.mean([r['base_angular_velocity_world'] for r in tail],axis=0).tolist(),
                tail_position_error_mean_m=float(np.mean([r['position_error_m'] for r in tail])),
                tail_orientation_error_mean_rad=float(np.mean([r['orientation_error_rad'] for r in tail])),
                upper_measured_and_native=upper_stats,moving_fixed_measurement_Ki_replay=replay,
                parking_transitions=[dict(stamp_s=r['stamp_s'],**r['parking_transition']) for r in block
                    if r.get('parking_transition') and (r['parking_transition']['entered'] or r['parking_transition']['exited'])]))
        report['runs'][directory.name]=dict(provenance_arguments=json.loads((directory/'provenance.json').read_text())['arguments'],cases=output)
        for p in (directory/'measurements.jsonl',directory/'summary.json',directory/'provenance.json',
                  directory/'velocity_servo_config.json',directory/'sources/velocity_servo.py'):
            report['evidence_sha256'][str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    report['evidence_sha256'][str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    shutil.copy2(Path(__file__),args.output/'executed_source.py')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,run in report['runs'].items():
        for case in run['cases']:
            print(name,case['case'],'PARK',case['tail_PARK_count'],'camera_tail',case['tail_position_error_mean_m'],case['tail_orientation_error_mean_rad'])
            print('upper_max',max(case['upper_measured_and_native'],key=lambda r:r['actual_qspan_rad']))
            if case['moving_fixed_measurement_Ki_replay']:print(json.dumps(case['moving_fixed_measurement_Ki_replay']['native_yaw_inputs']))


if __name__=='__main__':main()
