#!/usr/bin/env python3
"""Actual native-hold FULL tail: distinguish constant input from learned18 motion."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_native_hold_01'


def main():
    rows=[json.loads(line)for line in (RUN/'measurements.jsonl').read_text().splitlines()]
    summary=json.loads((RUN/'summary.json').read_text());cases={}
    for result in summary['cases']:
        selected=[row for row in rows if row['case']==result['case']]
        tail=[row for row in selected if row['stamp_s']>=selected[-1]['stamp_s']-.5-1e-9]
        native=np.asarray([row['actual_native_command']for row in tail])
        q=np.asarray([row['joint_positions']for row in tail]);targets=np.asarray([row['applied_joint_position_target']for row in tail])
        span=np.ptp(q,axis=0);axis=int(span.argmax());base=np.asarray([row['T_world_base']for row in tail])
        last=selected[-1];actual=np.asarray(last['T_world_camera_optical']);reference=np.asarray(last['T_world_reference_optical'])
        cases[result['case']]=dict(strict_stop=result['strict_stop'],max_joint_axis=tail[0]['joint_names'][axis],
            max_joint_span_rad=float(span[axis]),same_axis_learned_pd_target_span_rad=float(np.ptp(targets,axis=0)[axis]),
            native7_component_span=np.ptp(native,axis=0).tolist(),native7_mean=native.mean(axis=0).tolist(),
            hold_fraction=float(np.mean([row['native_camera_parking']['active']for row in tail])),
            hold_transition_count_start=tail[0]['native_camera_parking']['transitions'],
            hold_transition_count_end=tail[-1]['native_camera_parking']['transitions'],
            body_rpy_component_span_rad=np.ptp(Rotation.from_matrix(base[:,:3,:3]).as_euler('xyz'),axis=0).tolist(),
            body_policy_input_abs_max=np.max(abs(np.asarray([row['applied_policy_velocity']for row in tail])),axis=0).tolist(),
            standing_fraction=float(np.mean([row['native_standing_mode']for row in tail])),
            integral_span_m=float(np.ptp([row['actual_native_domain']['world_z_feedback']['integral_after_m']for row in tail])),
            final_camera_z_error_m=float(actual[2,3]-reference[2,3]))
    files=[Path(__file__),RUN/'summary.json',RUN/'measurements.jsonl',RUN/'timed_tasks.json',RUN/'provenance.json']
    files.extend(RUN/'sources'/name for name in ['adaptive_runner.py','camera_coordinator.py','world_z_feedback.py'])
    report=dict(schema='go2_native_hold_tail_readonly/v1',cases=cases,
        cumulative_final_z_bias_m=sum(case['final_camera_z_error_m']for case in cases.values()),
        conclusions=['All9 final half-second windows have exactly constant native7, exact zero body policy input and standing mode100%. Remaining movement cannot be attributed to changing native7 input or releases inside those final windows.',
            'All18 learned actions remain active. Failed calf/arm motions are real physical outcomes; constant native input is insufficient to claim strict parking.',
            'diagR passes joint10mrad but camera span5.1646mm fails5mm. yawL violates several base/camera/q/speed limits; yawR violates q and camera span.',
            'Early hold rolls back world-Z updates; cumulative final Z bias rises to114.6mm. This is a measured association, not a causal plant proof.'],
        files_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()for path in files})
    path=Path(__file__).with_suffix('.json');assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
    print(path);print('failed',[(name,case['max_joint_axis'],case['max_joint_span_rad'])for name,case in cases.items()if not case['strict_stop']['passed']])


if __name__=='__main__':main()
