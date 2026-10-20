#!/usr/bin/env python3
"""Fixed-trace bounded world-Z integral input proposal, never a physics prediction."""
from pathlib import Path
import hashlib,json,sys,importlib.util,time
import numpy as np
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parents[1];R=P/'Robot_Go2/results/adaptive_suite_softpose_01';F=R/'sources';sys.path.insert(0,str(P))
from jev_agent_wbc.wbc_protocol import TimedTask

def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 start=time.monotonic();rows=[json.loads(s) for s in (R/'measurements.jsonl').open() if 'task_feedback' in json.loads(s)];tasks=[TimedTask.from_dict(t) for t in json.loads((R/'timed_tasks.json').read_text())];sampler_cls=load('z_probe_optical',F/'optical_sampler.py').OpticalSampler;samplers={t.task_id:sampler_cls(t) for t in tasks};backend=load('z_probe_backend',F/'go2_backend.py');backend.UPSTREAM_MDP=F/'go2_upstream';domain=backend.PolicyDomain();E=np.array(json.loads((R/'sensor_mounts.json').read_text())['camera']['T_parent_optical']);source=load('z_probe_source',F/'camera_coordinator.py')
 ref_error=0.;prepared=[]
 for r in rows:
  taskid=r['task_feedback']['task_id'];s=samplers[taskid];T=np.array(r['T_world_reference_optical']);A=np.array(r['T_world_camera_optical']);B=np.array(r['T_world_base']);L=B.copy();L[:3,3]+=B[:3,:3]@np.array([0,0,.07]);ref_error=max(ref_error,float(np.max(abs(T-s.sample(min(r['stamp_s'],s.end_s))))));err=T[:3,3]-A[:3,3];rot=.25*Rotation.from_matrix(T[:3,:3]@A[:3,:3].T).as_rotvec();rot*=min(1.,.05/max(np.linalg.norm(rot),1e-12));corrected=T.copy();corrected[:3,:3]=Rotation.from_rotvec(rot).as_matrix()@T[:3,:3];prepared.append((r,taskid,T,A,L,err,corrected))
 report=dict(schema='go2_fixed_trace_world_z_integral/v1',scope='CPU counterfactual native-input evaluation on immutable actual camera/body traces. Same samples and measured errors are reused: this cannot predict physical tracking improvement, error convergence, changed next-case initialization, learned18actions or stop pass.',total_position_correction_cap_m=.025,orientation_correction_cap_rad=.05,P_gain=.35,integral_state_bound_m=.025,integral_reset='each newly accepted task',maximum_original_Task_reference_replay_error=ref_error,results={},source_sha256={})
 for ki in [.1,.2,.4]:
  I=0.;last_task=None;out=[];domain_reject=cap_freeze=0;maxnorm=0.;maxI=0.;fallback_out=0
  for r,taskid,T,A,L,err,corrected in prepared:
   if last_task!=taskid:I=0.;last_task=taskid
   oldI=I;step=ki*float(err[2])*.02;candidate=float(np.clip(I+step,-.025,.025));raw=.35*err+np.array([0.,0.,candidate]);proposal=raw*min(1.,.025/max(np.linalg.norm(raw),1e-12));sat=bool((candidate-I)*(raw[2]-proposal[2])>1e-15)
   if sat:candidate=I;cap_freeze+=1
   raw=.35*err+np.array([0.,0.,candidate]);correction=raw*min(1.,.025/max(np.linalg.norm(raw),1e-12));target=corrected.copy();target[:3,3]+=correction;native=backend.camera_to_mixed(target,L,E);check=domain.check(native)
   if check['status']!='inside_outer_domain_unverified':
    I=oldI;domain_reject+=1;native=backend.camera_to_mixed(T,L,E);check=domain.check(native);fallback_out+=check['status']!='inside_outer_domain_unverified';correction=np.zeros(3)
   else:I=candidate
   maxnorm=max(maxnorm,float(np.linalg.norm(correction)));maxI=max(maxI,abs(I));out.append(dict(case=r['case'],stamp_s=r['stamp_s'],position_error_z_m=float(err[2]),I_m=I,position_correction_m=correction.tolist(),native_xyz=native[:3].tolist(),domain_status=check['status'],cap_integrator_frozen=sat))
  cases={}
  for name in dict.fromkeys(r['case'] for r in out):
   selected=[r for r in out if r['case']==name];end=selected[-1]['stamp_s'];tail=[r for r in selected if r['stamp_s']>=end-.5-1e-9];cases[name]=dict(final_I_m=selected[-1]['I_m'],mean_last_half_second_I_m=float(np.mean([r['I_m'] for r in tail])),last_half_second_I_span_m=float(np.ptp([r['I_m'] for r in tail])),mean_last_half_second_total_Z_correction_m=float(np.mean([r['position_correction_m'][2] for r in tail])),mean_last_half_second_actual_error_Z_m=float(np.mean([r['position_error_z_m'] for r in tail])),final_native_xyz=selected[-1]['native_xyz'])
  report['results'][str(ki)]=dict(ki_per_s=ki,rows=len(out),maximum_total_correction_norm_m=maxnorm,maximum_abs_integral_m=maxI,domain_rejected_updates=domain_reject,cap_antiwindup_frozen_updates=cap_freeze,fallback_out_of_domain=fallback_out,all_total_corrections_within_original_cap=maxnorm<=.025+1e-14,all_native_commands_inside_original_outer_domain=domain_reject==0,cases=cases)
 report['interpretation']=['Preserved steady measured Z bias supports investigating a bounded native-input integral, rather than editing the already accepted WORLD target or freezing learned actions.','In the frozen trace the error never changes when I changes, so I accumulates until its cap or antiwindup stops it. Bounded I and legal domain here are not physical error convergence.','Original full3D correction norm cap25mm and original orientation correction50mrad are retained; no per-axis extra25mm is stacked beyond the total cap. Domain failures must reject the update and freeze/unwind I; this trace only evaluates a freeze-on-rejection candidate.','If integrated, preserve exact Task/optical conversion, fresh measured state, original input/drive/limits/collision guards and full18action policy; new physical camera and strictstop tests remain necessary.'];report['cpu_wall_s']=time.monotonic()-start
 files=[Path(__file__),R/'measurements.jsonl',R/'timed_tasks.json',R/'provenance.json',R/'sensor_mounts.json',F/'camera_coordinator.py',F/'go2_backend.py',F/'optical_sampler.py'];report['source_sha256']={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};out=P/'validation/go2_z_integral_trace_husky_20261020.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(out)
 for k,v in report['results'].items():print('Ki',k,'cap',v['maximum_total_correction_norm_m'],'domainReject',v['domain_rejected_updates'],'maxI',v['maximum_abs_integral_m'],'AW',v['cap_antiwindup_frozen_updates'],'finalIs',[round(c['final_I_m']*1000,3) for c in v['cases'].values()])
if __name__=='__main__':main()
