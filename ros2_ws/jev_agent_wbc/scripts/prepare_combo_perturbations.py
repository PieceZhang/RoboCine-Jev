#!/usr/bin/env python3
"""Reproducible independent randomized hard-combo fixtures; never reject failures."""
import argparse
from copy import deepcopy
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jev_agent_wbc.jvs_kinematics import JOINT_LOWER,JOINT_UPPER


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenarios',type=Path,default=ROOT/'scenarios');p.add_argument('--seeds',nargs='+',type=int,default=[20261019,20261020,20261021,20261022])
    p.add_argument('--cases',nargs='+',default=['combo_05_hard','combo_06_hard']);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'video_combo').mkdir();manifest=[]
    for name in a.cases:
        shutil.copy2(a.scenarios/'video_combo'/(name+'_nominal.json'),a.output/'video_combo'/(name+'_nominal.json'))
        task=json.loads((a.scenarios/name/'task.json').read_text());source=json.loads((a.scenarios/name/'runtime_scene.json').read_text())
        for seed in a.seeds:
            rng=np.random.default_rng(np.random.SeedSequence([seed,int(name.split('_')[1])]))
            scene=deepcopy(source);case=name+f'__seed_{seed}';scene['name']=case;scene['source_combo']=name
            perturbation=np.r_[rng.uniform(-.05,.05,2),rng.uniform(-.12,.12),rng.uniform(-.04,.04,6)]
            initial=np.asarray(source['initial_state'])+perturbation
            initial[3:]=np.clip(initial[3:],JOINT_LOWER+.04,JOINT_UPPER-.04);scene['initial_state']=initial.tolist()
            changes=[]
            for obstacle in scene['obstacles']:
                shift=rng.uniform([-.12,-.12,-.035],[.12,.12,.035]);speed=float(rng.uniform(.8,1.2))
                for key in obstacle['keyframes']:
                    key['position']=(np.asarray(key['position'])+shift).tolist()
                    key['velocity']=(np.asarray(key['velocity'])*speed).tolist();key['t']/=speed
                changes.append(dict(id=obstacle['id'],position_offset_m=shift.tolist(),speed_factor=speed))
            scope=dict(seed=seed,source_combo=name,obstacles=changes,initial_state_delta=(initial-np.asarray(source['initial_state'])).tolist(),
                sampling='one independently sampled case per seed; no safety/pass rejection sampling',task_sha256=hashlib.sha256((a.scenarios/name/'task.json').read_bytes()).hexdigest())
            folder=a.output/case;folder.mkdir()
            for filename,value in [('task.json',task),('runtime_scene.json',scene),('perturbation.json',scope)]:
                (folder/filename).write_text(json.dumps(value,indent=2)+'\n')
            manifest.append(dict(case=case,**scope))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(' '.join(row['case'] for row in manifest))


if __name__=='__main__':main()
