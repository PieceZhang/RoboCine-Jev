#!/usr/bin/env python3
"""Independent read-only Go2 terminal operating-model boundary review."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.package.resolve()
    model_path=root/'validation/go2_terminal_allocation_model_husky_20261020.py'
    model=load('review_go2_terminal_model',model_path)
    run=root/'Robot_Go2/results/adaptive_suite_world_z_01';result=root/'Robot_Go2/results/terminal_allocation_cpu_01'
    backend=load('review_go2_backend',run/'sources/go2_backend.py');backend.UPSTREAM_MDP=run/'sources/go2_upstream'
    domain=backend.PolicyDomain();bounds=domain.check([.5,0,.5,1,0,0,0]);xyz=np.array(bounds['xyz_outer_bounds']);euler=np.array(bounds['euler_outer_bounds_rad'])
    rows=[json.loads(line) for line in (run/'measurements.jsonl').open()]
    import sys
    sys.path.insert(0,str(root))
    from jev_agent_wbc.wbc_protocol import TimedTask
    tasks=[TimedTask.from_dict(t) for t in json.loads((run/'timed_tasks.json').read_text())]
    task_by_case=dict(zip([r['case'] for r in json.loads((run/'summary.json').read_text())['cases']],tasks))
    sampler_module=load('review_terminal_sampler',run/'sources/optical_sampler.py')
    extrinsic=np.array(json.loads((run/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    report=json.loads((result/'model_results.json').read_text());checks=[];branch=[]
    for case,variants in report['results'].items():
        index=next(i for i,r in enumerate(rows) if r['case']==case and r['phase']=='stop')
        initial=rows[index-1];body=np.asarray(initial['T_world_base']);previous=np.asarray(initial['applied_policy_velocity']);velocity=np.asarray(initial['measured_body_velocity'])
        task=task_by_case[case];end=sampler_module.OpticalSampler(task).sample(task.trajectory.end_s);end_ee=end@np.linalg.inv(extrinsic)
        history=[(row['stamp_s']-.02-task.trajectory.end_s,row['applied_policy_velocity']) for row in rows[max(0,index-32):index]]
        for variant in variants:
            saved=np.load(result/f'{case}_{variant}.npz')
            times=saved['times'];native=saved['native_xyz'];angles=saved['native_angles'];issued=saved['issued'];issued_times=saved['issued_times']
            reserve=np.minimum(1.,times/.8)
            margins=np.c_[native-xyz[:,0]-reserve[:,None]*[.035,.035,.015],xyz[:,1]-native-reserve[:,None]*[.035,.035,.015],angles-euler[0]-.015*reserve[:,None],euler[1]-angles-.015*reserve[:,None]]
            slew=np.diff(np.vstack([previous,issued]),axis=0)/.02
            yaw=np.r_[0.,np.cumsum(saved['body_velocity'][:-1,2]*.005)]
            position_error=[];angle_error=[]
            for k in (0,1,319,320,399,400,500,999,1000):
                link=np.eye(4);link[:3,:3]=Rotation.from_euler('z',yaw[k]).as_matrix()@body[:3,:3]
                link[:3,3]=saved['base_positions'][k]+link[:3,:3]@np.array([0,0,.07])
                command=np.r_[native[k],Rotation.from_euler('xyz',angles[k]).as_quat()[[3,0,1,2]]]
                recovered=backend.mixed_to_world_camera(command,link,extrinsic)
                position_error.append(float(np.linalg.norm(recovered[:3,3]-end[:3,3])))
                angle_error.append(float(Rotation.from_matrix(recovered[:3,:3]@end[:3,:3].T).magnitude()))
            checks.append(dict(case=case,variant=variant,endpoint_roundtrip_max_position_error_m=max(position_error),
                endpoint_roundtrip_max_angle_rad=max(angle_error),dense5ms_min_reserved_margin=float(margins.min()),
                declared_sparse_min_margin=variants[variant]['nonlinear_min_reserved_domain_margin'],
                original_cap_pass=bool(np.all(abs(issued)<=.2+1e-9)),original_slew_pass=bool(np.all(abs(slew)<=[.5+1e-9,.5+1e-9,.8+1e-9])),
                last3s_exact_command_zero=bool(np.all(issued[issued_times>=2.-1e-9]==0)),
                all_body_axis_rate_at_5s=saved['body_velocity'][-1].tolist()))
        # This exposes the whole-rollout backward/forward branch switch. It
        # is a model limitation, not evidence of a learned-policy transition.
        lower=model.model_rollout([-1e-9,0.],previous,velocity,history,body,end_ee,xyz,euler)
        upper=model.model_rollout([+1e-9,0.],previous,velocity,history,body,end_ee,xyz,euler)
        branch.append(dict(case=case,initial_vx=float(velocity[0]),previous_vx=float(previous[0]),
            whole_rollout_branch_changes_at_desired_zero=bool(not (velocity[0]<-.01 or previous[0]<0)),
            native_xy_endpoint_difference_across2nanomps_input_m=(upper['native_xyz'][-1,:2]-lower['native_xyz'][-1,:2]).tolist(),
            base_xy_endpoint_difference_m=(upper['base_positions'][-1,:2]-lower['base_positions'][-1,:2]).tolist()))
    source_paths=[model_path,result/'model_results.json',run/'timed_tasks.json',run/'sources/go2_backend.py',run/'sources/optical_sampler.py',Path(__file__).resolve()]
    output=dict(schema='independent_go2_terminal_model_review/v1',physical_execution=False,native_NN_performance_inferred=False,
        checks=checks,branch_boundary_probe=branch,
        findings=[
          dict(kind='verified_model_invariant',detail='Saved model commands obey original .2 axes caps/.5,.5,.8 slew, and all three issued body inputs are literal zero from2s to5s. This is command-space, not proof of zero measured body/joint rates.'),
          dict(kind='verified_reference',detail='Real mixed_to_world_camera round trips at phase boundaries reconstruct the unchanged original absolute Task.end optical pose. No authored trajectory knot/time/end pose is changed.'),
          dict(kind='model_boundary',detail='The entire5s x-response model switches between backward gain.55/tau.35 and instantaneous delayed forward response based in part on desired_vx sign. A near-zero candidate sign change may discontinuously change predictions for the same previous/history/initial measured velocity. This precludes smooth-model/global-optimum claims and should not be copied unnoticed into a runtime derivative or Jacobian.'),
          dict(kind='unmodeled_feedback',detail='The model omits the measured camera P correction and persistent bounded world-Z integral. Saved reserved margins describe the uncorrected endpoint. Runtime must retain the real final native domain check and integral rollback/freeze on override or rejection; the existing total25mm position and50mrad orientation cap cannot be treated as zero.'),
          dict(kind='limited_domain',detail='Only the source outer operating envelope is modeled. All18 joint/effort/contact/FCL, source grasp geometry, real inertia/latency, actual fresh feedback, safety overrides and strict measured parking remain unverified by this model.')],
        evidence_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths})
    with a.output.open('x') as stream:json.dump(output,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(models=len(checks),all_cap_slew_zero=all(c['original_cap_pass'] and c['original_slew_pass'] and c['last3s_exact_command_zero'] for c in checks),dense_min_margin=min(c['dense5ms_min_reserved_margin'] for c in checks),max_roundtrip_position_error_m=max(c['endpoint_roundtrip_max_position_error_m'] for c in checks),largest_branch_xy_jump_m=max(np.linalg.norm(c['native_xy_endpoint_difference_across2nanomps_input_m']) for c in branch))))


if __name__=='__main__':main()
