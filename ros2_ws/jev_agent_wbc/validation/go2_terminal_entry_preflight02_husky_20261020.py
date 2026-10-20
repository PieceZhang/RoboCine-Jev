#!/usr/bin/env python3
"""Separate real control cost from independent duplicate proof on2250 old stops."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_world_z_01'
OUTPUT=ROOT/'Robot_Go2/results/terminal_entry_cpu_02'
sys.path.insert(0,str(ROOT))
from jev_agent_wbc.wbc_protocol import TimedTask


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=False);frozen=OUTPUT/'sources';frozen.mkdir()
    files=[ROOT/'Robot_Go2/terminal_allocator.py',ROOT/'Robot_Go2/camera_coordinator.py',ROOT/'Robot_Go2/adaptive_runner.py',ROOT/'test/test_wbc_go2_terminal_allocator.py',ROOT/'test/test_wbc_go2_terminal_integration.py',Path(__file__)]
    for path in files:shutil.copy2(path,frozen/path.name)
    helper=load('fresh_terminal_helper',frozen/'terminal_allocator.py')
    backend=load('fresh_terminal_backend',RUN/'sources/go2_backend.py');backend.UPSTREAM_MDP=RUN/'sources/go2_upstream'
    coordinator=load('fresh_terminal_coordinator',frozen/'camera_coordinator.py')
    sampler=load('fresh_terminal_sampler',RUN/'sources/optical_sampler.py')
    domain=backend.PolicyDomain();bounds=domain.check([.5,0,.5,1,0,0,0]);xyz=np.asarray(bounds['xyz_outer_bounds']);euler=np.asarray(bounds['euler_outer_bounds_rad'])
    rows=[json.loads(line)for line in (RUN/'measurements.jsonl').read_text().splitlines()]
    tasks=[TimedTask.from_dict(task)for task in json.loads((RUN/'timed_tasks.json').read_text())]
    cases=[case['case']for case in json.loads((RUN/'summary.json').read_text())['cases']]
    endpoints=[sampler.OpticalSampler(task).sample(task.trajectory.end_s)for task in tasks]
    extrinsic=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    fixed=coordinator.CameraCoordinator(extrinsic,backend=backend,prediction_profile='persistent_bias_12_v1')
    history=[];checked=[];wall=[];proof_wall=[];components=[];started=time.monotonic()
    for index,row in enumerate(rows):
        stamp=row['stamp_s']-.02
        if row['phase']=='stop'and row['case']!='settle':
            task_index=cases.index(row['case']);task=tasks[task_index];previous=rows[index-1]
            body=np.asarray(previous['T_world_base']);link=body.copy();link[:3,3]+=body[:3,:3]@np.array([0,0,.07])
            remaining=max(0.,task.trajectory.end_s+5.-stamp);intervals=fixed.prediction_intervals(min(1.2,remaining))
            tick=time.perf_counter()
            context=helper.TerminalContext(backend,stamp_s=stamp,task_end_s=task.trajectory.end_s,stop_seconds=5,
                endpoint_optical=endpoints[task_index],extrinsic=extrinsic,body=body,link=link,
                measured_velocity=previous['measured_body_velocity'],previous_applied=previous['applied_policy_velocity'],
                input_history=history,target_xy=(.58,.15),prediction_intervals=intervals)
            setup_done=time.perf_counter()
            proposal=context.solve();solve_done=time.perf_counter()
            support=fixed.command(stamp,lambda t:endpoints[task_index].copy(),link,body,proposal,
                np.asarray(previous['measured_body_velocity']),max_lookahead_s=min(1.2,remaining),
                previous_applied=np.asarray(previous['applied_policy_velocity']),command_dt=.02,
                prediction_provider=context.predict_link,locked_input=context.phase!='allocate')
            guard_done=time.perf_counter()
            actual=context.guard_applied(support['base_command']);manifest=context.manifest()
            control_done=time.perf_counter();wall.append(control_done-tick)
            components.append([setup_done-tick,solve_done-setup_done,guard_done-solve_done,control_done-guard_done])
            values=[]
            for elapsed in intervals:
                predicted=context.predict_link(actual,float(elapsed))
                ee=context.endpoint@np.linalg.inv(extrinsic)
                local=predicted[:3,:3].T@(ee[:3,3]-predicted[:3,3]);position=np.r_[local[:2],ee[2,3]]
                angles=Rotation.from_matrix(predicted[:3,:3].T@ee[:3,:3]).as_euler('xyz')
                reserve=min(1.,elapsed/max(min(.8,remaining),1e-9))
                values.extend(position-xyz[:,0]-np.array([.035,.035,.015])*reserve)
                values.extend(xyz[:,1]-position-np.array([.035,.035,.015])*reserve)
                values.extend(angles-euler[0]-.015*reserve);values.extend(euler[1]-angles-.015*reserve)
            proof_wall.append(time.perf_counter()-control_done)
            checked.append(dict(case=row['case'],stamp_s=stamp,stop_elapsed_s=stamp-task.trajectory.end_s,
                phase=context.phase,proposal=proposal.tolist(),actual=actual.tolist(),camera_support_feasible=support['feasible'],live_guard_margin=support['predicted_min_domain_margin'],
                model_reserved_margin=float(min(values)),model_final_posture_error_xy_m=context.model_target_error.tolist(),
                original_endpoint_matrix_unchanged=bool(np.array_equal(context.endpoint,endpoints[task_index])),
                actual_standing_input=bool(np.all(actual==0)),
                command_caps_pass=bool(np.all(abs(actual)<=.2+1e-9)),
                command_slew_pass=bool(np.all(abs(actual-np.asarray(previous['applied_policy_velocity']))<=helper.SLEW*.02+1e-9)),
                source_original_applied_history=True))
        history.append((stamp,np.asarray(row['applied_policy_velocity'])));history=history[-32:]
    violations=[record for record in checked if not record['command_caps_pass']or not record['command_slew_pass']or not record['original_endpoint_matrix_unchanged']]
    negative=[record for record in checked if record['model_reserved_margin']< -1e-7 or record['camera_support_feasible'] is False or record['live_guard_margin']< -1e-7]
    late_nonzero=[record for record in checked if record['stop_elapsed_s']>=2.-1e-9 and not record['actual_standing_input']]
    result=dict(schema='go2_terminal_fresh_context_cpu/v1',samples=len(checked),
        scope='2250 separate fresh proposals on unchanged original measured stop states/history. Newly proposed commands never replace later historical measurements/history; not a simulated closed loop or learned18 parking proof.',
        cap_slew_or_endpoint_violation_count=len(violations),negative_reserved_margin_count=len(negative),
        post2s_nonzero_count=len(late_nonzero),model_min_reserved_margin=float(min(record['model_reserved_margin']for record in checked)),
        control_only_wall_ms=dict(mean=float(np.mean(wall)*1000),p95=float(np.quantile(wall,.95)*1000),max=float(max(wall)*1000)),
        independent_proof_wall_ms=dict(mean=float(np.mean(proof_wall)*1000),p95=float(np.quantile(proof_wall,.95)*1000),max=float(max(proof_wall)*1000)),
        mean_control_component_ms=dict(zip(['context_init','operating_solve','original_reserved_domain_guard','final_input_guard_and_manifest'],(np.mean(components,axis=0)*1000).tolist())),
        control_only_by_phase_ms={phase:dict(mean=float(np.mean([value for record,value in zip(checked,wall)if record['phase']==phase])*1000),p95=float(np.quantile([value for record,value in zip(checked,wall)if record['phase']==phase],.95)*1000))for phase in ['allocate','brake','zero']},
        original_7node_forecast_retained=True,
        cpu_wall_s=time.monotonic()-started,physical_verified=False,
        cpu_gate_pass=not violations and not negative and not late_nonzero,
        physical_correction_domain_status='Final original P+worldZI must be rechecked by live original domain checker; this model does not replace it.',
        files_sha256={str(path.relative_to(ROOT)):sha(path)for path in files+[RUN/'measurements.jsonl',RUN/'timed_tasks.json']})
    (OUTPUT/'cpu_replay.json').write_text(json.dumps(result,indent=2)+'\n');(OUTPUT/'cpu_rows.json').write_text(json.dumps(checked,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
