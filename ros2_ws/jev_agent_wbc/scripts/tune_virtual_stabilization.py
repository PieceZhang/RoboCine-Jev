#!/usr/bin/env python3
"""Causal parameter sweep on frozen real physics samples; never smooth raw videos."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.signal import lfilter,lfilter_zi,butter,sosfiltfilt
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_agent_wbc.virtual_stabilizer import se3_log


def errors(folder):
 rows=json.loads((folder/'virtual_camera_physics.json').read_text());t=np.array([x['t'] for x in rows]);rp=np.array([x['rp'] for x in rows]);rr=np.array([x['rr'] for x in rows]);ap=np.array([x['ap'] for x in rows]);ar=np.array([x['ar'] for x in rows]);raw=np.array([se3_log(r.T@(a-p),r.T@b) for p,r,a,b in zip(rp,rr,ap,ar)])
 return dict(folder=str(folder),t=t,fs=1/np.median(np.diff(t)),rp=rp,rr=rr,ap=ap,ar=ar,raw=raw)


def replay(d,pc,rc,stages):
 out=d['raw'].copy()
 for component,cutoff in ((slice(0,3),pc),(slice(3,6),rc)):
  alpha=-np.expm1(-2*np.pi*cutoff/d['fs'])
  b,a=np.array([alpha]),np.array([1.,-(1-alpha)])
  for _ in range(stages):
   out[:,component],_=lfilter(b,a,out[:,component],axis=0,zi=lfilter_zi(b,a)[:,None]*out[0,component][None,:])
 # Batch SE(3) exponential, including the rotational/translation coupling.
 v,phi=out[:,:3],out[:,3:];theta=np.linalg.norm(phi,axis=1);cross=np.cross(phi,v);small=theta<1e-5
 A=np.empty(len(theta));B=A.copy();A[small]=.5-theta[small]**2/24;B[small]=1/6-theta[small]**2/120
 A[~small]=(1-np.cos(theta[~small]))/theta[~small]**2;B[~small]=(theta[~small]-np.sin(theta[~small]))/theta[~small]**3
 delta=v+A[:,None]*cross+B[:,None]*np.cross(phi,cross)
 p=d['rp']+np.einsum('nij,nj->ni',d['rr'],delta);r=d['rr']@Rotation.from_rotvec(phi).as_matrix()
 correction=p-d['ap'];norm=np.linalg.norm(correction,axis=1);p=d['ap']+correction*np.minimum(1.,.12/np.maximum(norm,1e-15))[:,None]
 relative=Rotation.from_matrix(d['ar'].transpose(0,2,1)@r).as_rotvec();rn=np.linalg.norm(relative,axis=1);r=d['ar']@Rotation.from_rotvec(relative*np.minimum(1.,np.deg2rad(8)/np.maximum(rn,1e-15))[:,None]).as_matrix()
 transl=np.einsum('nji,nj->ni',d['rr'],p-d['rp']);phi=Rotation.from_matrix(d['rr'].transpose(0,2,1)@r).as_rotvec()
 # V(phi)^-1 * translation in closed form.
 normphi=np.linalg.norm(phi,axis=1);c=np.full(len(phi),1/12.);m=normphi>1e-5;c[m]=(1-.5*normphi[m]/np.tan(.5*normphi[m]))/normphi[m]**2
 residual=np.c_[transl-.5*np.cross(phi,transl)+c[:,None]*np.cross(phi,np.cross(phi,transl)),phi]
 mask=(d['t']>=1.)&(d['t']<=d['t'][-1]-.5);sos=butter(4,1.,fs=d['fs'],btype='highpass',output='sos')
 a,b=(sosfiltfilt(sos,x,axis=0)[mask] for x in (d['raw'],residual))
 rms=lambda x:float(np.sqrt(np.mean(np.sum(x*x,axis=1))))
 pr=rms(b[:,:3])/max(rms(a[:,:3]),1e-15);rot=rms(b[:,3:])/max(rms(a[:,3:]),1e-15)
 return dict(position_ratio=pr,rotation_ratio=rot,translation_limit_fraction=float(np.mean(norm>.12)),rotation_limit_fraction=float(np.mean(rn>np.deg2rad(8))),max_uncapped_translation_mm=float(1000*norm.max()),mean_pose_difference_mm=float(1000*np.linalg.norm(p-d['ap'],axis=1).mean()))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('folders',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();data=[errors(f) for f in a.folders];results=[]
 for stages in (1,2,3):
  for pc in (.35,.5,.7,1.,1.5):
   for rc in (.35,.5,.7,1.,1.5):
    scores=[replay(d,pc,rc,stages) for d in data];worst=max(max(s['position_ratio'],s['rotation_ratio']) for s in scores);limited=max(max(s['translation_limit_fraction'],s['rotation_limit_fraction']) for s in scores)
    # Fixed score: worst-case (>1 Hz) residual + bounded-correction saturation penalty.
    score=worst+2*limited+.001*sum(s['mean_pose_difference_mm'] for s in scores)/len(scores)
    results.append(dict(score=score,translation_cutoff_hz=pc,rotation_cutoff_hz=rc,stages=stages,cases=scores))
 results.sort(key=lambda x:x['score']);a.output.mkdir(exist_ok=False,parents=True)
 (a.output/'parameter_sweep.json').write_text(json.dumps(dict(source_folders=a.folders,objective='worst-case 1Hz highpass residual ratio + 2*saturation fraction + .001*mean correction mm',ranking=results),default=str,indent=2)+'\n')
 winner={k:results[0][k] for k in ('translation_cutoff_hz','rotation_cutoff_hz','stages')};winner.update(max_translation_correction_m=.12,max_rotation_correction_deg=8.,reset_gap_s=.15)
 (a.output/'selected_config.json').write_text(json.dumps(winner,indent=2)+'\n');print(json.dumps(results[:5],indent=2))
